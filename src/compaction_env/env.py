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
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from verifiers.types import Messages, ModelResponse, SamplingArgs, State

logger = logging.getLogger(__name__)


class CompactionEnv(vf.SingleTurnEnv):
    """Environment that routes generation through /compact_generate."""

    def __init__(
        self,
        inner_env: vf.Environment,
        max_seq_len: int = 2048,
        compact_target_ratio: float = 0.3,
        n_compacts: int = 2,
        **kwargs,
    ):
        self.inner_env = inner_env
        self.compact_max_seq_len = max_seq_len
        self.compact_target_ratio = compact_target_ratio
        self.n_compacts = n_compacts

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
        prompt: Messages,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Call /compact_generate and return a ChatCompletion with token data.

        The verifiers framework's parse_response_tokens() extracts:
        - response.prompt_token_ids  (set via setattr)
        - response.choices[0].token_ids  (set via setattr)
        - response.choices[0].logprobs.content  (standard OpenAI field)
        """
        client = client or state["client"]
        model = model or state["model"]
        sampling_args = sampling_args or state.get("sampling_args") or {}

        # Tokenize prompt via vLLM's /tokenize endpoint (supports chat messages)
        tokenize_resp = await client.post(
            "/tokenize",
            body={
                "model": model,
                "messages": prompt,
                "add_generation_prompt": True,
            },
            cast_to=object,
        )
        prompt_ids = tokenize_resp["tokens"]

        temperature = sampling_args.get("temperature", 0.7)
        top_p = sampling_args.get("top_p", 0.95)

        # Call the compaction endpoint (not under /v1/)
        base_url = str(client.base_url).rstrip("/")
        server_url = base_url.rsplit("/v1", 1)[0] if "/v1" in base_url else base_url

        async with httpx.AsyncClient(timeout=600.0) as http_client:
            resp = await http_client.post(
                f"{server_url}/compact_generate",
                json={
                    "prompt_ids": prompt_ids,
                    "max_seq_len": self.compact_max_seq_len,
                    "compact_target_ratio": self.compact_target_ratio,
                    "n_compacts": self.n_compacts,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
            resp.raise_for_status()
            result = resp.json()

        all_token_ids = result["all_token_ids"]
        all_logprobs = result["all_logprobs"]
        final_text = result["final_text"]

        # Build logprobs in OpenAI format for parse_response_tokens
        from openai.types.chat.chat_completion import ChoiceLogprobs
        logprobs_content = [
            {"token": "", "logprob": lp, "bytes": None, "top_logprobs": []}
            for lp in all_logprobs
        ]

        completion = ChatCompletion(
            id=f"compact-{int(time.time())}",
            model=model,
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=final_text,
                    ),
                    finish_reason="stop",
                    logprobs=ChoiceLogprobs(content=None),
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(all_token_ids),
                total_tokens=len(prompt_ids) + len(all_token_ids),
            ),
        )

        # Monkey-patch the attributes that parse_response_tokens reads via getattr
        completion.prompt_token_ids = prompt_ids
        completion.choices[0].token_ids = all_token_ids
        # Set logprobs as dict (parse_response_tokens handles both obj and dict)
        completion.choices[0].logprobs = {"content": logprobs_content}

        return completion


def load_environment(
    gym: str = "countdown",
    max_seq_len: int = 2048,
    compact_target_ratio: float = 0.3,
    n_compacts: int = 2,
    **inner_env_kwargs,
) -> CompactionEnv:
    """Load a CompactionEnv wrapping the specified gym environment.

    Called by verifiers' load_environment() when env_id="compaction_env".
    """
    inner_env = vf.load_environment(gym, **inner_env_kwargs)
    return CompactionEnv(
        inner_env=inner_env,
        max_seq_len=max_seq_len,
        compact_target_ratio=compact_target_ratio,
        n_compacts=n_compacts,
    )
