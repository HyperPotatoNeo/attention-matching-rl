"""FastAPI routes for KV cache compaction generation.

Provides /compact_generate endpoint that generates text with mid-sequence
KV compaction via the CompactionWorker's collective_rpc method.
"""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class CompactGenerateRequest(BaseModel):
    prompt_ids: list[int]
    max_seq_len: int = 8192
    max_tokens_per_segment: int | None = None
    compact_target_ratio: float = 0.3
    n_compacts: int = 3
    compact_window: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95


@router.post("/compact_generate")
async def compact_generate(body: CompactGenerateRequest, request: Request):
    engine = request.app.state.engine_client

    tokenizer = engine.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    prompt_len = len(body.prompt_ids)

    if body.max_tokens_per_segment is not None:
        max_tokens_per_segment = body.max_tokens_per_segment
    else:
        available_tokens = body.max_seq_len - prompt_len
        max_tokens_per_segment = available_tokens // max(body.n_compacts + 1, 1)

    logger.info(
        "/compact_generate: prompt_len=%d, max_tokens_per_seg=%d, "
        "n_compacts=%d, ratio=%.2f",
        prompt_len, max_tokens_per_segment, body.n_compacts,
        body.compact_target_ratio,
    )

    results = await engine.collective_rpc(
        "compact_generate",
        args=(
            body.prompt_ids,
            max_tokens_per_segment,
            body.n_compacts,
            body.compact_target_ratio,
            body.compact_window,
            body.temperature,
            body.top_p,
            eos_token_id,
        ),
    )

    # collective_rpc returns one result per worker; all are identical for TP
    result = results[0]

    final_text = tokenizer.decode(result["all_token_ids"], skip_special_tokens=True)

    return {
        "all_token_ids": result["all_token_ids"],
        "all_logprobs": result["all_logprobs"],
        "final_text": final_text,
        "diagnostics": result.get("diagnostics", {}),
    }
