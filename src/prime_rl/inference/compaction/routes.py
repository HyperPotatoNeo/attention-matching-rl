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
    n_protect_turns: int = -1


class SessionStepRequest(BaseModel):
    session_id: str
    new_token_ids: list[int]
    max_response_tokens: int = 512


@router.post("/compact_session/create")
async def compact_session_create(body: SessionCreateRequest, request: Request):
    engine = request.app.state.engine_client
    tokenizer = engine.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    logger.info(
        "/compact_session/create: session=%s, prompt_len=%d, max_kv_len=%d",
        body.session_id, len(body.prompt_ids), body.max_kv_len,
    )

    result = await engine.collective_rpc(
        "compact_session_create",
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
            "n_protect_turns": body.n_protect_turns,
        },
    )

    data = result[0]
    final_text = tokenizer.decode(data["all_token_ids"], skip_special_tokens=True)
    return {
        "session_id": data["session_id"],
        "all_token_ids": data["all_token_ids"],
        "final_text": final_text,
        "current_seq_len": data["current_seq_len"],
        "diagnostics": data.get("diagnostics", {}),
    }


@router.post("/compact_session/step")
async def compact_session_step(body: SessionStepRequest, request: Request):
    engine = request.app.state.engine_client
    tokenizer = engine.get_tokenizer()

    logger.info(
        "/compact_session/step: session=%s, new_tokens=%d",
        body.session_id, len(body.new_token_ids),
    )

    result = await engine.collective_rpc(
        "compact_session_step",
        args=(
            body.session_id,
            body.new_token_ids,
            body.max_response_tokens,
        ),
    )

    data = result[0]
    final_text = tokenizer.decode(data["all_token_ids"], skip_special_tokens=True)
    return {
        "all_token_ids": data["all_token_ids"],
        "final_text": final_text,
        "current_seq_len": data["current_seq_len"],
        "diagnostics": data.get("diagnostics", {}),
    }


@router.delete("/compact_session/{session_id}")
async def compact_session_delete(session_id: str, request: Request):
    engine = request.app.state.engine_client

    await engine.collective_rpc(
        "compact_session_delete",
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
