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
import math

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_chat_eos_token_id(tokenizer) -> int:
    """Get the EOS token ID that chat models actually use to end responses.

    Many chat models (e.g. OLMo-3) set eos_token to <|endoftext|> but
    actually end assistant turns with <|im_end|>. Using the wrong one
    causes the worker to generate past the natural stop, producing garbage.
    """
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end_ids) == 1:
        logger.info("Using <|im_end|> (id=%d) as chat EOS instead of default eos_token_id=%d",
                     im_end_ids[0], tokenizer.eos_token_id)
        return im_end_ids[0]
    logger.info("No single-token <|im_end|>, using default eos_token_id=%d", tokenizer.eos_token_id)
    return tokenizer.eos_token_id


def _decode_response(tokenizer, token_ids: list[int]) -> str:
    """Decode generated tokens, preserving tool-call markers but stripping BOS/EOS.

    Using skip_special_tokens=True strips model-specific tool markers like
    Ministral's [TOOL_CALLS] and [ARGS].  Instead we decode with all tokens
    then manually remove only BOS/EOS boundaries.
    """
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    for attr in ("bos_token", "eos_token"):
        token_str = getattr(tokenizer, attr, None)
        if token_str:
            text = text.replace(token_str, "")
    # MistralTokenizer lacks bos_token/eos_token attrs — strip common markers
    for marker in ("</s>", "<s>"):
        text = text.replace(marker, "")
    return text.strip()


def _get_dp_engines(engine):
    """Resolve the list of DP engine identities from the top-level engine."""
    client = getattr(engine, "engine_core", engine)
    dp_engines = getattr(client, "core_engines", None)
    if dp_engines and len(dp_engines) > 1:
        return client, dp_engines
    return None, None


_dp_counter = 0
_session_dp_map: dict[str, int] = {}


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
MAX_WAIT_SECONDS = 0.1


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
        eos_token_id = _get_chat_eos_token_id(tokenizer)

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

        # Adaptive batch sizing: cap B at what the block budget can support.
        try:
            block_info = (await engine.collective_rpc("get_block_budget"))[0]
            free_blocks = block_info["free_blocks"]
            block_size = block_info["block_size"]
            max_prompt = max(len(p) for p in prompt_ids_list)
            if first_body.max_kv_len is not None:
                max_possible_len = max(first_body.max_kv_len, max_prompt + 2)
            else:
                seg = first_body.max_tokens_per_segment or (
                    (first_body.max_seq_len - max_prompt) // max(first_body.n_compacts + 1, 1))
                max_possible_len = max_prompt + seg * (first_body.n_compacts + 1)
            blocks_per_seq = math.ceil(max_possible_len / block_size)
            max_B = max(1, free_blocks // blocks_per_seq) if blocks_per_seq > 0 else B
            if max_B < B:
                logger.warning(
                    "Block budget allows %d seqs (%d blocks/seq, %d free), capping from %d",
                    max_B, blocks_per_seq, free_blocks, B)
                for req_future in batch[max_B:]:
                    await self._queue.put(req_future)
                batch = batch[:max_B]
                prompt_ids_list = prompt_ids_list[:max_B]
                B = max_B
        except Exception as e:
            logger.debug("Block budget query failed, using full batch: %s", e)

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
            final_text = _decode_response(tokenizer, result["all_token_ids"])
            responses.append({
                "all_token_ids": result["all_token_ids"],
                "all_logprobs": result["all_logprobs"],
                "final_text": final_text,
                "diagnostics": result.get("diagnostics", {}),
            })
        return responses


_batcher = _RequestBatcher()


@router.post("/compact_generate")
async def compact_generate(body: CompactGenerateRequest, request: Request):
    return await _batcher.submit(body, request.app)


@router.post("/compact_generate_batch")
async def compact_generate_batch(body: CompactGenerateBatchRequest, request: Request):
    engine = request.app.state.engine_client

    tokenizer = engine.get_tokenizer()
    eos_token_id = _get_chat_eos_token_id(tokenizer)

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
        final_text = _decode_response(tokenizer, result["all_token_ids"])
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
    system_prompt_len: int | None = None


class SessionStepRequest(BaseModel):
    session_id: str
    new_token_ids: list[int]
    max_response_tokens: int = 512
    is_summary: bool = False
    trigger_compact: bool = False


SESSION_CREATE_MAX_BATCH = 64
SESSION_CREATE_MAX_WAIT = 0.1


class _SessionCreateBatcher:
    """Accumulates individual session create requests and dispatches them in batch.

    Groups by DP rank so all sessions in a batch go to the same engine.
    """

    def __init__(self):
        self._queue: asyncio.Queue[tuple[SessionCreateRequest, int, asyncio.Future]] = asyncio.Queue()
        self._started = False

    def _ensure_started(self, app):
        if not self._started:
            self._started = True
            asyncio.create_task(self._worker(app))

    async def submit(self, body: SessionCreateRequest, eos_token_id: int, app) -> dict:
        self._ensure_started(app)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((body, eos_token_id, future))
        return await future

    async def _worker(self, app):
        while True:
            item = await self._queue.get()
            batch = [item]

            deadline = asyncio.get_event_loop().time() + SESSION_CREATE_MAX_WAIT
            while len(batch) < SESSION_CREATE_MAX_BATCH:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            try:
                await self._dispatch_batch(app, batch)
            except Exception as e:
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(e)

    async def _dispatch_batch(self, app, batch):
        engine = app.state.engine_client
        tokenizer = engine.get_tokenizer()

        _, dp_engines = _get_dp_engines(engine)
        by_rank: dict[int, list[tuple[SessionCreateRequest, int, asyncio.Future]]] = {}

        global _dp_counter
        for body, eos_token_id, future in batch:
            if dp_engines is not None:
                rank = _dp_counter % len(dp_engines)
                _dp_counter += 1
                _session_dp_map[body.session_id] = rank
            else:
                rank = 0
            by_rank.setdefault(rank, []).append((body, eos_token_id, future))

        # Dispatch all DP ranks in parallel
        await asyncio.gather(*[
            self._dispatch_create_rank(engine, dp_engines, tokenizer, rank, group)
            for rank, group in by_rank.items()
        ])

    async def _dispatch_create_rank(self, engine, dp_engines, tokenizer, rank, group):
        first = group[0][0]
        session_ids = [b.session_id for b, _, _ in group]
        prompt_ids_list = [b.prompt_ids for b, _, _ in group]
        eos_token_id = group[0][1]

        try:
            if dp_engines is not None:
                client, _ = _get_dp_engines(engine)
                results = await client._call_utility_async(
                    "collective_rpc", "compact_session_create_batch", None,
                    (session_ids, prompt_ids_list, first.max_kv_len,
                     first.max_response_tokens, eos_token_id), {
                        "compact_target_ratio": first.compact_target_ratio,
                        "compact_window": first.compact_window,
                        "temperature": first.temperature,
                        "top_p": first.top_p,
                        "compaction_mode": first.compaction_mode,
                        "use_suffix_queries": first.use_suffix_queries,
                        "n_max_turns": first.n_max_turns,
                        "n_preserved_turns": first.n_preserved_turns,
                        "system_prompt_len": first.system_prompt_len,
                    },
                    engine=dp_engines[rank],
                )
            else:
                results = await engine.collective_rpc(
                    "compact_session_create_batch",
                    args=(session_ids, prompt_ids_list, first.max_kv_len,
                          first.max_response_tokens, eos_token_id),
                    kwargs={
                        "compact_target_ratio": first.compact_target_ratio,
                        "compact_window": first.compact_window,
                        "temperature": first.temperature,
                        "top_p": first.top_p,
                        "compaction_mode": first.compaction_mode,
                        "use_suffix_queries": first.use_suffix_queries,
                        "n_max_turns": first.n_max_turns,
                        "n_preserved_turns": first.n_preserved_turns,
                        "system_prompt_len": first.system_prompt_len,
                    },
                )

            batch_results = results[0]
            for j, (_, _, future) in enumerate(group):
                data = batch_results[j]
                final_text = _decode_response(tokenizer, data["all_token_ids"])
                if not future.done():
                    future.set_result({
                        "session_id": data["session_id"],
                        "all_token_ids": data["all_token_ids"],
                        "all_logprobs": data.get("all_logprobs", []),
                        "final_text": final_text,
                        "current_seq_len": data["current_seq_len"],
                        "diagnostics": data.get("diagnostics", {}),
                    })
        except Exception as e:
            for _, _, future in group:
                if not future.done():
                    future.set_exception(e)


_session_create_batcher = _SessionCreateBatcher()


@router.post("/compact_session/create")
async def compact_session_create(body: SessionCreateRequest, request: Request):
    engine = request.app.state.engine_client
    tokenizer = engine.get_tokenizer()
    eos_token_id = _get_chat_eos_token_id(tokenizer)

    logger.info(
        "/compact_session/create: session=%s, prompt_len=%d, max_kv_len=%d",
        body.session_id, len(body.prompt_ids), body.max_kv_len,
    )

    return await _session_create_batcher.submit(body, eos_token_id, request.app)


SESSION_STEP_MAX_BATCH = 128
SESSION_STEP_MAX_WAIT = 0.2


class _SessionStepBatcher:
    """Accumulates individual session step requests and dispatches them in batch.

    Groups by DP rank so all sessions in a batch go to the same engine.
    Falls back to sequential dispatch for requests with is_summary or trigger_compact.
    """

    def __init__(self):
        self._queue: asyncio.Queue[tuple[SessionStepRequest, asyncio.Future]] = asyncio.Queue()
        self._started = False

    def _ensure_started(self, app):
        if not self._started:
            self._started = True
            asyncio.create_task(self._worker(app))

    async def submit(self, body: SessionStepRequest, app) -> dict:
        self._ensure_started(app)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((body, future))
        return await future

    async def _worker(self, app):
        while True:
            item = await self._queue.get()
            batch = [item]

            deadline = asyncio.get_event_loop().time() + SESSION_STEP_MAX_WAIT
            while len(batch) < SESSION_STEP_MAX_BATCH:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            try:
                await self._dispatch_batch(app, batch)
            except Exception as e:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)

    async def _dispatch_batch(self, app, batch):
        engine = app.state.engine_client
        tokenizer = engine.get_tokenizer()

        # Split: special requests (is_summary/trigger_compact) go sequential,
        # normal requests get batched by DP rank.
        special = [(b, f) for b, f in batch if b.is_summary or b.trigger_compact]
        normal = [(b, f) for b, f in batch if not b.is_summary and not b.trigger_compact]

        # Dispatch special requests sequentially
        for body, future in special:
            try:
                result = await _session_rpc(
                    engine, "compact_session_step", body.session_id,
                    args=(body.session_id, body.new_token_ids, body.max_response_tokens),
                    kwargs={"is_summary": body.is_summary, "trigger_compact": body.trigger_compact},
                )
                data = result[0]
                final_text = _decode_response(tokenizer, data["all_token_ids"])
                if not future.done():
                    future.set_result({
                        "all_token_ids": data["all_token_ids"],
                        "all_logprobs": data.get("all_logprobs", []),
                        "final_text": final_text,
                        "current_seq_len": data["current_seq_len"],
                        "diagnostics": data.get("diagnostics", {}),
                    })
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

        if not normal:
            return

        # Group normal requests by DP rank
        _, dp_engines = _get_dp_engines(engine)
        by_rank: dict[int, list[tuple[SessionStepRequest, asyncio.Future]]] = {}
        for body, future in normal:
            if dp_engines is not None and body.session_id in _session_dp_map:
                rank = _session_dp_map[body.session_id]
            else:
                rank = 0
            by_rank.setdefault(rank, []).append((body, future))

        # Dispatch all DP ranks in parallel
        await asyncio.gather(*[
            self._dispatch_step_rank(engine, dp_engines, tokenizer, rank, group)
            for rank, group in by_rank.items()
        ])

    async def _dispatch_step_rank(self, engine, dp_engines, tokenizer, rank, group):
        session_ids = [b.session_id for b, _ in group]
        new_token_ids_list = [b.new_token_ids for b, _ in group]
        max_resp_list = [b.max_response_tokens for b, _ in group]

        try:
            if dp_engines is not None:
                client, _ = _get_dp_engines(engine)
                results = await client._call_utility_async(
                    "collective_rpc", "compact_session_step_batch", None,
                    (session_ids, new_token_ids_list, max_resp_list), {},
                    engine=dp_engines[rank],
                )
            else:
                results = await engine.collective_rpc(
                    "compact_session_step_batch",
                    args=(session_ids, new_token_ids_list, max_resp_list),
                )

            batch_results = results[0]
            for j, (_, future) in enumerate(group):
                data = batch_results[j]
                final_text = _decode_response(tokenizer, data["all_token_ids"])
                if not future.done():
                    future.set_result({
                        "all_token_ids": data["all_token_ids"],
                        "all_logprobs": data.get("all_logprobs", []),
                        "final_text": final_text,
                        "current_seq_len": data["current_seq_len"],
                        "diagnostics": data.get("diagnostics", {}),
                    })
        except Exception as e:
            for _, future in group:
                if not future.done():
                    future.set_exception(e)


_session_step_batcher = _SessionStepBatcher()


@router.post("/compact_session/step")
async def compact_session_step(body: SessionStepRequest, request: Request):
    return await _session_step_batcher.submit(body, request.app)


@router.delete("/compact_session/{session_id}")
async def compact_session_delete(session_id: str, request: Request):
    engine = request.app.state.engine_client

    await _session_rpc(
        engine, "compact_session_delete", session_id,
        args=(session_id,),
    )

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
    eos_token_id = _get_chat_eos_token_id(tokenizer)

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
    eos_token_id = _get_chat_eos_token_id(tokenizer)

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
    eos_token_id = _get_chat_eos_token_id(tokenizer)

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
