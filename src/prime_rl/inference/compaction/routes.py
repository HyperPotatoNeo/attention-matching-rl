"""FastAPI routes for KV cache compaction generation and RSA.

Provides /compact_generate endpoint that generates text with mid-sequence
KV compaction via the CompactionWorker's collective_rpc method.

Individual /compact_generate requests are transparently batched into
compact_generate_batch calls for higher GPU utilization.

Also provides /rsa_generate for Recursive Self-Aggregation with persistent
compacted memory.
"""

import asyncio
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_dp_engines(engine):
    """Resolve the list of DP engine identities from the top-level engine."""
    client = getattr(engine, "engine_core", engine)
    dp_engines = getattr(client, "core_engines", None)
    if dp_engines and len(dp_engines) > 1:
        return client, dp_engines
    return None, None


_dp_counter = 0
_session_dp_map: dict[str, int] = {}

# Backpressure signaling: waiters block when blocks are exhausted;
# compact_session_delete notifies them that blocks may be available.
_blocks_freed: dict[int, asyncio.Condition] = {}  # per DP rank
_blocks_freed_global: asyncio.Condition | None = None  # DP=1

_BLOCK_WAIT_MAX_RETRIES = 30
_BLOCK_WAIT_TIMEOUT = 10.0  # seconds per wait


def _get_blocks_freed_condition(dp_rank: int | None) -> asyncio.Condition:
    global _blocks_freed_global
    if dp_rank is None:
        if _blocks_freed_global is None:
            _blocks_freed_global = asyncio.Condition()
        return _blocks_freed_global
    if dp_rank not in _blocks_freed:
        _blocks_freed[dp_rank] = asyncio.Condition()
    return _blocks_freed[dp_rank]


async def _notify_blocks_freed(dp_rank: int | None):
    cond = _get_blocks_freed_condition(dp_rank)
    async with cond:
        cond.notify_all()


async def _session_rpc(engine, method: str, session_id: str, args: tuple = (), kwargs: dict | None = None):
    """Route a session RPC to a specific DP engine based on session_id.

    With DP > 1, collective_rpc broadcasts to ALL engines, causing every
    session to allocate KV blocks on every GPU.  This pins each session to
    a single DP rank so the block budget is split across engines.
    Round-robin assignment guarantees even distribution.
    """
    global _dp_counter
    client, dp_engines = _get_dp_engines(engine)
    if dp_engines is not None:
        if method == "compact_session_create":
            if session_id in _session_dp_map:
                dp_rank = _session_dp_map[session_id]
            else:
                dp_rank = _dp_counter % len(dp_engines)
                _dp_counter += 1
                _session_dp_map[session_id] = dp_rank
        elif session_id in _session_dp_map:
            dp_rank = _session_dp_map[session_id]
            if method == "compact_session_delete":
                del _session_dp_map[session_id]
        else:
            dp_rank = hash(session_id) % len(dp_engines)
        return await client._call_utility_async(
            "collective_rpc", method, None, args, kwargs or {},
            engine=dp_engines[dp_rank],
        )
    return await engine.collective_rpc(method, args=args, kwargs=kwargs)

MAX_BATCH_SIZE = 32
MAX_WAIT_SECONDS = 1.0


class CompactGenerateRequest(BaseModel):
    prompt_ids: list[int]
    max_seq_len: int = 8192
    max_tokens_per_segment: int | None = None
    compact_target_ratio: float = 0.3
    n_compacts: int = 3
    compact_window: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_kv_len: int | None = None
    max_total_tokens: int | None = None
    compute_beta: bool = False
    use_suffix_queries: bool = True
    compaction_mode: str = "attention_matching"
    carryover_ratio: float = 0.5


class CompactGenerateBatchRequest(BaseModel):
    prompt_ids_list: list[list[int]]
    max_seq_len: int = 8192
    max_tokens_per_segment: int | None = None
    compact_target_ratio: float = 0.3
    n_compacts: int = 3
    compact_window: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_kv_len: int | None = None
    max_total_tokens: int | None = None
    compute_beta: bool = False
    use_suffix_queries: bool = True
    compaction_mode: str = "attention_matching"
    carryover_ratio: float = 0.5


class _RequestBatcher:
    """Accumulates individual compact_generate requests and processes them in batch.

    Waits up to max_wait_seconds for requests to accumulate (up to max_batch_size),
    then sends them all through compact_generate_batch in a single collective_rpc call.
    """

    def __init__(self, max_batch_size: int = MAX_BATCH_SIZE, max_wait: float = MAX_WAIT_SECONDS):
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait
        self._queue: asyncio.Queue[tuple[CompactGenerateRequest, asyncio.Future]] = asyncio.Queue()
        self._started = False

    def _ensure_started(self, app):
        if not self._started:
            self._started = True
            asyncio.create_task(self._worker(app))

    async def submit(self, body: CompactGenerateRequest, app) -> dict:
        self._ensure_started(app)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((body, future))
        return await future

    async def _worker(self, app):
        while True:
            item = await self._queue.get()
            batch = [item]

            deadline = asyncio.get_event_loop().time() + self.max_wait
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            try:
                results = await self._process_batch(app, batch)
                for (_, future), result in zip(batch, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)

    async def _process_batch(self, app, batch: list[tuple[CompactGenerateRequest, asyncio.Future]]) -> list[dict]:
        engine = app.state.engine_client
        tokenizer = engine.get_tokenizer()
        eos_token_id = tokenizer.eos_token_id

        first_body = batch[0][0]
        prompt_ids_list = [b.prompt_ids for b, _ in batch]
        B = len(prompt_ids_list)

        if first_body.max_kv_len is not None:
            max_tokens_per_segment = 0
        elif first_body.max_tokens_per_segment is not None:
            max_tokens_per_segment = first_body.max_tokens_per_segment
        else:
            max_prompt = max(len(p) for p in prompt_ids_list)
            available = first_body.max_seq_len - max_prompt
            max_tokens_per_segment = available // max(first_body.n_compacts + 1, 1)

        logger.info("Auto-batch: B=%d, max_kv_len=%s, max_total_tokens=%s",
                     B, first_body.max_kv_len, first_body.max_total_tokens)

        results = await engine.collective_rpc(
            "compact_generate_batch",
            args=(
                prompt_ids_list,
                max_tokens_per_segment,
                first_body.n_compacts,
                first_body.compact_target_ratio,
                first_body.compact_window,
                first_body.temperature,
                first_body.top_p,
                eos_token_id,
            ),
            kwargs={
                "max_kv_len": first_body.max_kv_len,
                "max_total_tokens": first_body.max_total_tokens,
                "compute_beta": first_body.compute_beta,
                "use_suffix_queries": first_body.use_suffix_queries,
                "compaction_mode": first_body.compaction_mode,
                "carryover_ratio": first_body.carryover_ratio,
            },
        )

        batch_results = results[0]
        responses = []
        for result in batch_results:
            final_text = tokenizer.decode(result["all_token_ids"], skip_special_tokens=True)
            responses.append({
                "all_token_ids": result["all_token_ids"],
                "all_logprobs": result["all_logprobs"],
                "final_text": final_text,
                "diagnostics": result.get("diagnostics", {}),
            })
        return responses


_batcher = _RequestBatcher()

SESSION_STEP_BATCH_SIZE = 16
SESSION_STEP_WAIT_SECONDS = 0.3


class _SessionStepBatcher:
    """Accumulates /compact_session/step requests and processes them in batch.

    Groups requests by DP rank (sessions on the same GPU batch together).
    Uses compact_session_step_batch for parallel model forward passes.
    """

    def __init__(self, max_batch_size: int = SESSION_STEP_BATCH_SIZE, max_wait: float = SESSION_STEP_WAIT_SECONDS):
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait
        # Per DP-rank queue; None key = DP=1
        self._queues: dict[int | None, asyncio.Queue] = {}
        self._started: set[int | None] = set()

    def _ensure_started(self, app, dp_rank: int | None):
        if dp_rank not in self._started:
            self._started.add(dp_rank)
            if dp_rank not in self._queues:
                self._queues[dp_rank] = asyncio.Queue()
            asyncio.create_task(self._worker(app, dp_rank))

    async def submit(self, body, app, dp_rank: int | None) -> dict:
        self._ensure_started(app, dp_rank)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queues[dp_rank].put((body, future))
        return await future

    async def _worker(self, app, dp_rank: int | None):
        queue = self._queues[dp_rank]
        while True:
            item = await queue.get()
            batch = [item]
            deadline = asyncio.get_event_loop().time() + self.max_wait
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            try:
                results = await self._process_batch(app, batch, dp_rank)
                for (_, future), result in zip(batch, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)

    async def _process_batch(self, app, batch, dp_rank: int | None) -> list[dict]:
        engine = app.state.engine_client
        tokenizer = engine.get_tokenizer()

        session_ids = [b.session_id for b, _ in batch]
        new_token_ids_list = [b.new_token_ids for b, _ in batch]
        max_response_tokens_list = [b.max_response_tokens for b, _ in batch]
        B = len(batch)

        logger.info("SessionStepBatch: B=%d, dp_rank=%s", B, dp_rank)

        client, dp_engines = _get_dp_engines(engine)
        if dp_engines is not None and dp_rank is not None:
            results = await client._call_utility_async(
                "collective_rpc", "compact_session_step_batch", None,
                (session_ids, new_token_ids_list, max_response_tokens_list), {},
                engine=dp_engines[dp_rank],
            )
        else:
            results = await engine.collective_rpc(
                "compact_session_step_batch",
                args=(session_ids, new_token_ids_list, max_response_tokens_list),
            )

        batch_data = results[0]
        responses = []
        for data in batch_data:
            if isinstance(data, dict) and data.get("error"):
                responses.append(data)
            else:
                final_text = tokenizer.decode(data["all_token_ids"], skip_special_tokens=True)
                responses.append({
                    "all_token_ids": data["all_token_ids"],
                    "all_logprobs": data.get("all_logprobs", []),
                    "final_text": final_text,
                    "current_seq_len": data["current_seq_len"],
                    "diagnostics": data.get("diagnostics", {}),
                })
        return responses


_session_step_batcher = _SessionStepBatcher()


@router.post("/compact_generate")
async def compact_generate(body: CompactGenerateRequest, request: Request):
    return await _batcher.submit(body, request.app)


@router.post("/compact_generate_batch")
async def compact_generate_batch(body: CompactGenerateBatchRequest, request: Request):
    engine = request.app.state.engine_client

    tokenizer = engine.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    if body.max_kv_len is not None:
        max_tokens_per_segment = 0
    elif body.max_tokens_per_segment is not None:
        max_tokens_per_segment = body.max_tokens_per_segment
    else:
        max_prompt = max(len(p) for p in body.prompt_ids_list)
        available_tokens = body.max_seq_len - max_prompt
        max_tokens_per_segment = available_tokens // max(body.n_compacts + 1, 1)

    B = len(body.prompt_ids_list)
    logger.info(
        "/compact_generate_batch: B=%d, max_kv_len=%s, max_total_tokens=%s",
        B, body.max_kv_len, body.max_total_tokens,
    )

    results = await engine.collective_rpc(
        "compact_generate_batch",
        args=(
            body.prompt_ids_list,
            max_tokens_per_segment,
            body.n_compacts,
            body.compact_target_ratio,
            body.compact_window,
            body.temperature,
            body.top_p,
            eos_token_id,
        ),
        kwargs={
            "max_kv_len": body.max_kv_len,
            "max_total_tokens": body.max_total_tokens,
            "use_suffix_queries": body.use_suffix_queries,
            "compaction_mode": body.compaction_mode,
            "carryover_ratio": body.carryover_ratio,
        },
    )

    # collective_rpc returns one result per worker; all identical for TP
    batch_results = results[0]

    responses = []
    for result in batch_results:
        final_text = tokenizer.decode(result["all_token_ids"], skip_special_tokens=True)
        responses.append({
            "all_token_ids": result["all_token_ids"],
            "all_logprobs": result["all_logprobs"],
            "final_text": final_text,
            "diagnostics": result.get("diagnostics", {}),
        })

    return {"results": responses}


class SessionCreateRequest(BaseModel):
    session_id: str
    prompt_ids: list[int]
    max_kv_len: int
    max_response_tokens: int = 512
    compact_target_ratio: float = 0.25
    compact_window: int | None = None
    temperature: float = 0.6
    top_p: float = 0.95
    compaction_mode: str = "attention_matching"
    use_suffix_queries: bool = True
    n_max_turns: int = -1
    n_preserved_turns: int = 0


class SessionStepRequest(BaseModel):
    session_id: str
    new_token_ids: list[int]
    max_response_tokens: int = 512


@router.post("/compact_session/create")
async def compact_session_create(body: SessionCreateRequest, request: Request):
    from fastapi.responses import JSONResponse

    engine = request.app.state.engine_client
    tokenizer = engine.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    logger.info(
        "/compact_session/create: session=%s, prompt_len=%d, max_kv_len=%d",
        body.session_id, len(body.prompt_ids), body.max_kv_len,
    )

    dp_rank_for_wait = _session_dp_map.get(body.session_id)

    for attempt in range(_BLOCK_WAIT_MAX_RETRIES + 1):
        result = await _session_rpc(
            engine, "compact_session_create", body.session_id,
            args=(
                body.session_id,
                body.prompt_ids,
                body.max_kv_len,
                body.max_response_tokens,
                eos_token_id,
            ),
            kwargs={
                "compact_target_ratio": body.compact_target_ratio,
                "compact_window": body.compact_window,
                "temperature": body.temperature,
                "top_p": body.top_p,
                "compaction_mode": body.compaction_mode,
                "use_suffix_queries": body.use_suffix_queries,
                "n_max_turns": body.n_max_turns,
                "n_preserved_turns": body.n_preserved_turns,
            },
        )

        data = result[0]
        if isinstance(data, dict) and data.get("error") == "blocks_exhausted":
            if attempt >= _BLOCK_WAIT_MAX_RETRIES:
                _session_dp_map.pop(body.session_id, None)
                return JSONResponse(status_code=503, content=data)
            if dp_rank_for_wait is None:
                dp_rank_for_wait = _session_dp_map.get(body.session_id)
            logger.debug(
                "Session %s waiting for blocks (attempt %d/%d, need=%d, free=%d)",
                body.session_id, attempt + 1, _BLOCK_WAIT_MAX_RETRIES,
                data["blocks_needed"], data["blocks_free"],
            )
            cond = _get_blocks_freed_condition(dp_rank_for_wait)
            async with cond:
                try:
                    await asyncio.wait_for(cond.wait(), timeout=_BLOCK_WAIT_TIMEOUT)
                except asyncio.TimeoutError:
                    pass
            continue

        final_text = tokenizer.decode(data["all_token_ids"], skip_special_tokens=True)
        return {
            "session_id": data["session_id"],
            "all_token_ids": data["all_token_ids"],
            "all_logprobs": data.get("all_logprobs", []),
            "final_text": final_text,
            "current_seq_len": data["current_seq_len"],
            "diagnostics": data.get("diagnostics", {}),
        }


@router.post("/compact_session/step")
async def compact_session_step(body: SessionStepRequest, request: Request):
    """Route session step through auto-batcher for parallel GPU inference."""
    dp_rank = _session_dp_map.get(body.session_id)
    return await _session_step_batcher.submit(body, request.app, dp_rank)


@router.delete("/compact_session/{session_id}")
async def compact_session_delete(session_id: str, request: Request):
    engine = request.app.state.engine_client

    dp_rank = _session_dp_map.get(session_id)

    await _session_rpc(
        engine, "compact_session_delete", session_id,
        args=(session_id,),
    )

    await _notify_blocks_freed(dp_rank)

    return {"deleted": session_id}


DEFAULT_AGG_TEMPLATE = (
    "\n\nHere are some previous attempts at solving this problem. "
    "Review them carefully and provide an improved solution:\n\n"
    "{peer_cots}\n\n"
    "Now provide your improved solution:"
)


class RsaGenerateRequest(BaseModel):
    prompt_ids: list[int]
    K: int = 4
    N: int | None = None
    T: int = 2
    k_peers: int = 2
    max_tokens_per_candidate: int = 2048
    compact_target_ratio: float = 0.25
    probe_tokens: int = 512
    agg_template: str = DEFAULT_AGG_TEMPLATE
    temperature: float = 0.7
    top_p: float = 0.95
    selection_strategy: str = "random"


@router.post("/rsa_generate")
async def rsa_generate(body: RsaGenerateRequest, request: Request):
    engine = request.app.state.engine_client
    tokenizer = engine.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    result = await engine.collective_rpc(
        "rsa_generate",
        args=(
            body.prompt_ids,
            body.K,
            body.T,
            body.k_peers,
            body.max_tokens_per_candidate,
            body.compact_target_ratio,
            body.probe_tokens,
            body.agg_template,
            body.temperature,
            body.top_p,
            eos_token_id,
        ),
        kwargs={"selection_strategy": body.selection_strategy, "N": body.N},
    )

    rsa_result = result[0]

    # Decode populations to text (they're already text from _batch_generate)
    populations = rsa_result["populations"]

    return {
        "populations": populations,
        "best": rsa_result["best"],
        "diagnostics": rsa_result["diagnostics"],
    }


class ParallelGenerateRequest(BaseModel):
    coordinator_prompt_ids: list[int]
    document_ids_list: list[list[int]]
    compact_target_ratio: float = 0.25
    probe_tokens: int = 256
    max_gen_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    compute_beta: bool = True
    summary_prompt: str | None = None


@router.post("/parallel_generate")
async def parallel_generate(body: ParallelGenerateRequest, request: Request):
    engine = request.app.state.engine_client
    tokenizer = engine.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    result = await engine.collective_rpc(
        "parallel_generate",
        args=(
            body.coordinator_prompt_ids,
            body.document_ids_list,
            body.compact_target_ratio,
            body.probe_tokens,
            body.max_gen_tokens,
            body.temperature,
            body.top_p,
            eos_token_id,
        ),
        kwargs={
            "compute_beta": body.compute_beta,
            "summary_prompt": body.summary_prompt,
        },
    )

    data = result[0]
    return {
        "all_token_ids": data["all_token_ids"],
        "all_logprobs": data["all_logprobs"],
        "final_text": data["final_text"],
        "diagnostics": data.get("diagnostics", {}),
    }


class ParallelGenerateFusedRequest(BaseModel):
    prompt_ids: list[int]
    K: int = 4
    max_candidate_tokens: int = 4096
    compact_target_ratio: float = 0.25
    max_gen_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    compute_beta: bool = True
    probe_tokens: int = 256
    synthesis_prompt: str | None = None
    coordinator_prompt_ids: list[int] | None = None


@router.post("/parallel_generate_fused")
async def parallel_generate_fused(
    body: ParallelGenerateFusedRequest, request: Request,
):
    engine = request.app.state.engine_client
    tokenizer = engine.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    result = await engine.collective_rpc(
        "parallel_generate_fused",
        args=(
            body.prompt_ids,
            body.K,
            body.max_candidate_tokens,
            body.compact_target_ratio,
            body.max_gen_tokens,
            body.temperature,
            body.top_p,
            eos_token_id,
        ),
        kwargs={
            "compute_beta": body.compute_beta,
            "probe_tokens": body.probe_tokens,
            "synthesis_prompt": body.synthesis_prompt,
            "coordinator_prompt_ids": body.coordinator_prompt_ids,
        },
    )

    data = result[0]
    return {
        "all_token_ids": data["all_token_ids"],
        "all_logprobs": data["all_logprobs"],
        "final_text": data["final_text"],
        "candidates": data.get("candidates", []),
        "diagnostics": data.get("diagnostics", {}),
    }
