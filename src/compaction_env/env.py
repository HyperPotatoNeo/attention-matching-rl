"""
CompactionEnv: Verifiers environment that generates with KV cache compaction.

The entire generate->compact->continue loop happens server-side via /compact_generate.
From the trainer's perspective, this produces a standard rollout with a single
TrajectoryStep containing all tokens across all segments.

The inner env (e.g. countdown, math) provides dataset and scoring rubric.
"""
import logging
import time

import httpx
import verifiers as vf
from verifiers.types import (
    Messages,
    Response,
    ResponseMessage,
    ResponseTokens,
    SamplingArgs,
    State,
    Tool,
    Usage,
)

logger = logging.getLogger(__name__)


class CompactionEnv(vf.SingleTurnEnv):
    """Environment that routes generation through /compact_generate."""

    def __init__(
        self,
        inner_env: vf.Environment,
        max_seq_len: int = 2048,
        max_tokens_per_segment: int | None = None,
        compact_target_ratio: float = 0.3,
        compact_window: int | None = None,
        n_compacts: int = 2,
        max_kv_len: int | None = None,
        max_total_tokens: int | None = None,
        compute_beta: bool = False,
        use_suffix_queries: bool = True,
        **kwargs,
    ):
        self.inner_env = inner_env
        self.compact_max_seq_len = max_seq_len
        self.compact_max_tokens_per_segment = max_tokens_per_segment
        self.compact_target_ratio = compact_target_ratio
        self.compact_window = compact_window
        self.n_compacts = n_compacts
        self.compact_max_kv_len = max_kv_len
        self.compact_max_total_tokens = max_total_tokens
        self.compute_beta = compute_beta
        self.use_suffix_queries = use_suffix_queries
        self._last_segment_boundaries: list[int] | None = None
        self._last_compaction_indices: list | None = None

        super().__init__(
            dataset=inner_env.dataset,
            eval_dataset=getattr(inner_env, "eval_dataset", None),
            system_prompt=inner_env.system_prompt,
            parser=inner_env.parser,
            rubric=inner_env.rubric,
            **kwargs,
        )

    async def get_model_response(
        self,
        state: State,
        prompt: Messages | str,
        client: vf.Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> Response:
        """Call /compact_generate and return a verifiers Response with token data."""
        client = client if client is not None else state["client"]
        model = model or state["model"]
        sampling_args = sampling_args or state.get("sampling_args") or {}

        # Get the base URL from the underlying OpenAI client
        oai_client = client.client
        base_url = str(oai_client.base_url).rstrip("/")
        server_url = base_url.rsplit("/v1", 1)[0] if "/v1" in base_url else base_url

        # Tokenize prompt via vLLM's /tokenize endpoint
        async with httpx.AsyncClient(timeout=7200.0) as http_client:
            # Convert messages to dicts for JSON serialization
            prompt_dicts = []
            if isinstance(prompt, str):
                prompt_dicts = [{"role": "user", "content": prompt}]
            else:
                for msg in prompt:
                    if hasattr(msg, "model_dump"):
                        prompt_dicts.append(msg.model_dump())
                    elif isinstance(msg, dict):
                        prompt_dicts.append(msg)
                    else:
                        prompt_dicts.append({"role": getattr(msg, "role", "user"), "content": str(getattr(msg, "content", ""))})

            tokenize_resp = await http_client.post(
                f"{server_url}/tokenize",
                json={
                    "model": model,
                    "messages": prompt_dicts,
                    "add_generation_prompt": True,
                },
            )
            tokenize_resp.raise_for_status()
            prompt_ids = tokenize_resp.json()["tokens"]

            temperature = sampling_args.get("temperature", 0.7)
            top_p = sampling_args.get("top_p", 0.95)

            request_body = {
                "prompt_ids": prompt_ids,
                "max_seq_len": self.compact_max_seq_len,
                "compact_target_ratio": self.compact_target_ratio,
                "n_compacts": self.n_compacts,
                "temperature": temperature,
                "top_p": top_p,
            }
            if self.compact_max_tokens_per_segment is not None:
                request_body["max_tokens_per_segment"] = self.compact_max_tokens_per_segment
            if self.compact_window is not None:
                request_body["compact_window"] = self.compact_window
            if self.compact_max_kv_len is not None:
                request_body["max_kv_len"] = self.compact_max_kv_len
            if self.compact_max_total_tokens is not None:
                request_body["max_total_tokens"] = self.compact_max_total_tokens
            if self.compute_beta:
                request_body["compute_beta"] = True
            if self.use_suffix_queries:
                request_body["use_suffix_queries"] = True

            resp = await http_client.post(
                f"{server_url}/compact_generate",
                json=request_body,
            )
            resp.raise_for_status()
            result = resp.json()

        all_token_ids = result["all_token_ids"]
        all_logprobs = result["all_logprobs"]
        final_text = result["final_text"]

        diagnostics = result.get("diagnostics", {})
        self._last_segment_boundaries = diagnostics.get("segment_boundaries")
        events = diagnostics.get("compaction_events", [])
        self._last_compaction_indices = [
            e.get("compaction_indices") for e in events
        ] if events else None

        tokens = ResponseTokens(
            prompt_ids=prompt_ids,
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=all_token_ids,
            completion_mask=[1] * len(all_token_ids),
            completion_logprobs=all_logprobs,
            routed_experts=None,
        )

        return Response(
            id=f"compact-{int(time.time())}",
            created=int(time.time()),
            model=model,
            usage=Usage(
                prompt_tokens=len(prompt_ids),
                reasoning_tokens=0,
                completion_tokens=len(all_token_ids),
                total_tokens=len(prompt_ids) + len(all_token_ids),
            ),
            message=ResponseMessage(
                content=final_text,
                finish_reason="stop",
                is_truncated=False,
                tokens=tokens,
            ),
        )

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ):
        await super().add_model_response(state, prompt_messages, response)
        if self._last_segment_boundaries is not None:
            last_step = state["trajectory"][-1]
            last_step["extras"]["segment_boundaries"] = self._last_segment_boundaries
            if self._last_compaction_indices is not None:
                last_step["extras"]["compaction_indices"] = self._last_compaction_indices
            self._last_segment_boundaries = None
            self._last_compaction_indices = None


def load_environment(
    gym: str = "countdown",
    max_seq_len: int = 2048,
    max_tokens_per_segment: int | None = None,
    compact_target_ratio: float = 0.3,
    compact_window: int | None = None,
    n_compacts: int = 2,
    max_kv_len: int | None = None,
    max_total_tokens: int | None = None,
    compute_beta: bool = False,
    use_suffix_queries: bool = True,
    **inner_env_kwargs,
) -> CompactionEnv:
    """Load a CompactionEnv wrapping the specified gym environment.

    Called by verifiers' load_environment() when env_id="compaction_env".
    """
    inner_env = vf.load_environment(gym, **inner_env_kwargs)
    return CompactionEnv(
        inner_env=inner_env,
        max_seq_len=max_seq_len,
        max_tokens_per_segment=max_tokens_per_segment,
        compact_target_ratio=compact_target_ratio,
        compact_window=compact_window,
        n_compacts=n_compacts,
        max_kv_len=max_kv_len,
        max_total_tokens=max_total_tokens,
        compute_beta=compute_beta,
        use_suffix_queries=use_suffix_queries,
    )
