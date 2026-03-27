"""
CompactionEnv: Verifiers environment that generates with KV cache compaction.

The entire generate->compact->continue loop happens server-side via /compact_generate.
From the trainer's perspective, this produces a standard rollout with a single
TrajectoryStep containing all tokens across all segments.

The inner env (e.g. countdown, math, balrog-bench) provides dataset, scoring rubric,
and multi-turn game logic. CompactionEnv wraps any Environment (single- or multi-turn)
and replaces model calls with /compact_generate requests.

TurnCompactionEnv uses /compact_session/create + /compact_session/step to maintain
KV cache state across turns. Compaction fires between turns (server-side), and
segment_boundaries are stored on the first trajectory step so interleave_rollout
can reconstruct the full compacted sequence for training.
"""
import json
import logging
import re
import time
import uuid

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
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


async def _compute_suffix_ids(
    http_client: httpx.AsyncClient,
    server_url: str,
    model: str,
    oai_tools: list | None,
) -> list[int]:
    """Compute chat template suffix tokens appended after an assistant message.

    Same approach as the verifiers TITO client: tokenize a dummy conversation
    and extract the tokens that appear after the assistant content but before
    the next user message.  These are template-specific separators
    (e.g. ``<|im_end|>\\n`` for Qwen3).
    """
    dummy_content = "World!"
    # Tokenize just the content to find its tokens
    content_body = {
        "model": model,
        "prompt": dummy_content,
    }
    content_resp = await http_client.post(f"{server_url}/tokenize", json=content_body)
    content_resp.raise_for_status()
    dummy_content_ids = content_resp.json()["tokens"]

    # Tokenize a full conversation to find what comes after the content
    msgs_body: dict = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": dummy_content},
        ],
        "add_generation_prompt": False,
    }
    if oai_tools:
        msgs_body["tools"] = oai_tools
    msgs_resp = await http_client.post(f"{server_url}/tokenize", json=msgs_body)
    msgs_resp.raise_for_status()
    dummy_msgs_ids = msgs_resp.json()["tokens"]

    # Find last occurrence of the content's final token, then take everything after
    last_token = dummy_content_ids[-1]
    for i in range(len(dummy_msgs_ids) - 1, -1, -1):
        if dummy_msgs_ids[i] == last_token:
            return dummy_msgs_ids[i + 1:]
    return []


def _find_suffix_overlap(prefix: list[int], suffix: list[int]) -> int:
    """Find the largest overlap between the end of prefix and start of suffix."""
    max_possible = min(len(prefix), len(suffix))
    for overlap_len in range(max_possible, 0, -1):
        if prefix[-overlap_len:] == suffix[:overlap_len]:
            return overlap_len
    return 0


def _parse_hermes_tool_calls(text: str) -> tuple[str | None, list[ToolCall]]:
    """Parse Hermes-format tool calls from model output text.

    Returns (content_before_tools, list_of_tool_calls).
    If no tool calls found, returns (text, []).
    """
    if "<tool_call>" not in text:
        return text, []

    tool_calls = []
    for i, match in enumerate(_TOOL_CALL_RE.finditer(text)):
        raw = match.group(1).strip()
        parsed = json.loads(raw)
        arguments = parsed.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=parsed["name"],
            arguments=arguments,
        ))

    content = text[:text.find("<tool_call>")].strip() or None
    return content, tool_calls


def _tool_defs_to_openai(tool_defs: list[Tool] | None) -> list[dict] | None:
    """Convert vf.Tool objects to OpenAI tool format for /tokenize."""
    if not tool_defs:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tool_defs
    ]


def _tool_call_to_openai(tc: dict) -> dict:
    """Convert a verifiers-style tool_call dict to OpenAI format."""
    if "function" in tc:
        return tc
    return {
        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
        "type": "function",
        "function": {"name": tc["name"], "arguments": tc["arguments"]},
    }


def _messages_to_openai(prompt) -> list[dict]:
    """Convert a list of messages to OpenAI-compatible dicts for /tokenize."""
    dicts = []
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    for msg in prompt:
        if hasattr(msg, "model_dump"):
            d = msg.model_dump()
        elif isinstance(msg, dict):
            d = dict(msg)
        else:
            d = {"role": getattr(msg, "role", "user"), "content": str(getattr(msg, "content", ""))}
        if "tool_calls" in d and d["tool_calls"]:
            d["tool_calls"] = [_tool_call_to_openai(tc) for tc in d["tool_calls"]]
        dicts.append(d)
    return dicts


def _format_observation(obs) -> str | None:
    """Format an observation from the inner env into a text string."""
    if obs is None:
        return None
    if isinstance(obs, dict) and "text" in obs:
        text = obs["text"]
        if isinstance(text, dict) and "long_term_context" in text:
            return text["long_term_context"]
        return str(text)
    if isinstance(obs, str):
        return obs
    return str(obs)


def _fix_prompt_after_setup(state: State) -> None:
    """Post-process prompt after inner env setup: fix mission and inject initial obs.

    BalrogEnv creates dataset rows (with system prompt) before reset(), so the
    mission is None. After setup_state calls reset(), the real mission is available.
    Also, verifiers MultiTurnEnv doesn't call env_response on turn 0, so the model
    never sees the initial observation unless we inject it here.
    """
    game_env = state.get("env")
    if game_env is not None and hasattr(game_env, "get_instruction_prompt"):
        # gym.Wrapper.__getattr__ skips attrs starting with "_", so walk the
        # wrapper chain to find _mission on the underlying env.
        mission = None
        e = game_env
        while e is not None:
            mission = e.__dict__.get("_mission")
            if mission is not None:
                break
            e = getattr(e, "env", None)
        if mission is not None:
            updated_prompt = game_env.get_instruction_prompt(instructions=mission)
            for msg in state.get("prompt", []):
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                if role == "system":
                    if isinstance(msg, dict):
                        msg["content"] = updated_prompt
                    else:
                        msg.content = updated_prompt
                    break

    obs = state.get("observation")
    if obs is not None:
        formatted = _format_observation(obs)
        if formatted:
            state["prompt"].append({"role": "user", "content": formatted})


class CompactionEnv(vf.MultiTurnEnv):
    """Environment that routes generation through /compact_generate.

    Wraps any verifiers Environment (single- or multi-turn) and overrides
    get_model_response to call /compact_generate instead of the standard
    OpenAI chat completions API. Multi-turn behavior (env_response,
    setup_state, stop conditions) is delegated to the inner env.
    """

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
        compaction_mode: str = "attention_matching",
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
        self.compaction_mode = compaction_mode
        self._last_segment_boundaries: list[int] | None = None
        self._last_compaction_indices: list | None = None

        inner_max_turns = getattr(inner_env, "max_turns", 1)
        inner_eval_dataset = getattr(inner_env, "eval_dataset", None)
        inner_dataset = inner_env.dataset if inner_env.dataset is not None else inner_eval_dataset

        super().__init__(
            max_turns=inner_max_turns,
            dataset=inner_dataset,
            eval_dataset=inner_eval_dataset,
            system_prompt=inner_env.system_prompt,
            parser=inner_env.parser,
            rubric=inner_env.rubric,
            tool_defs=getattr(inner_env, "tool_defs", None),
            **kwargs,
        )

    async def setup_state(self, state: State) -> State:
        outer_task = state.get("task")
        state = await self.inner_env.setup_state(state)
        if outer_task is not None:
            state["task"] = outer_task
        _fix_prompt_after_setup(state)
        return state

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages | str:
        return await self.inner_env.env_response(messages, state, **kwargs)

    @vf.stop
    async def inner_env_done(self, state: State, **kwargs) -> bool:
        """Proxy stop conditions from the inner env."""
        for condition in self.inner_env._stop_conditions:
            if await condition(state):
                return True
        return False

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
        if tool_defs is None:
            tool_defs = state.get("tool_defs") or None

        # Get the base URL from the underlying OpenAI client
        oai_client = client.client
        base_url = str(oai_client.base_url).rstrip("/")
        server_url = base_url.rsplit("/v1", 1)[0] if "/v1" in base_url else base_url

        # Tokenize prompt via vLLM's /tokenize endpoint
        async with httpx.AsyncClient(timeout=7200.0) as http_client:
            prompt_dicts = _messages_to_openai(prompt)

            tokenize_body = {
                "model": model,
                "messages": prompt_dicts,
                "add_generation_prompt": True,
            }
            oai_tools = _tool_defs_to_openai(tool_defs)
            if oai_tools:
                tokenize_body["tools"] = oai_tools

            tokenize_resp = await http_client.post(
                f"{server_url}/tokenize",
                json=tokenize_body,
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
            if self.compaction_mode != "attention_matching":
                request_body["compaction_mode"] = self.compaction_mode

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

        content = final_text
        parsed_tool_calls = None
        finish_reason = "stop"
        if tool_defs:
            content, parsed_tool_calls = _parse_hermes_tool_calls(final_text)
            if parsed_tool_calls:
                finish_reason = "tool_calls"

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
                content=content,
                finish_reason=finish_reason,
                is_truncated=False,
                tokens=tokens,
                tool_calls=parsed_tool_calls or None,
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
    compaction_mode: str = "attention_matching",
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
        compaction_mode=compaction_mode,
    )


class TurnCompactionEnv(vf.MultiTurnEnv):
    """Turn-based compaction: maintains KV across turns via the session API.

    Uses /compact_session/create + /compact_session/step to persist and compact
    the KV cache between turns rather than re-prefilling each time. Compaction
    fires between turns (server-side, controlled by n_max_turns/n_preserved_turns).

    Segment boundaries are accumulated across turns and stored on the first
    trajectory step's extras. interleave_rollout reads them from there to build
    a single TrainingSample with the correct compaction points for segmented_forward.
    """

    def __init__(
        self,
        inner_env: vf.Environment,
        max_kv_len: int = 4096,
        max_response_tokens: int = 512,
        compact_target_ratio: float = 0.25,
        compact_window: int | None = None,
        n_max_turns: int = -1,
        n_preserved_turns: int = 0,
        max_turns: int = -1,
        temperature: float = 0.6,
        top_p: float = 0.95,
        compaction_mode: str = "attention_matching",
        use_suffix_queries: bool = True,
        **kwargs,
    ):
        self.inner_env = inner_env
        self.max_kv_len = max_kv_len
        self.max_response_tokens = max_response_tokens
        self.compact_target_ratio = compact_target_ratio
        self.compact_window = compact_window
        self.n_max_turns = n_max_turns
        self.n_preserved_turns = n_preserved_turns
        self.temperature = temperature
        self.top_p = top_p
        self.compaction_mode = compaction_mode
        self.use_suffix_queries = use_suffix_queries

        inner_eval_dataset = getattr(inner_env, "eval_dataset", None)
        inner_dataset = inner_env.dataset if inner_env.dataset is not None else inner_eval_dataset
        if max_turns > 0:
            effective_max_turns = max_turns
        elif n_max_turns > 0:
            effective_max_turns = n_max_turns
        else:
            effective_max_turns = getattr(inner_env, "max_turns", -1)

        super().__init__(
            max_turns=effective_max_turns,
            dataset=inner_dataset,
            eval_dataset=inner_eval_dataset,
            system_prompt=inner_env.system_prompt,
            parser=inner_env.parser,
            rubric=inner_env.rubric,
            tool_defs=getattr(inner_env, "tool_defs", None),
            **kwargs,
        )

    async def setup_state(self, state: State) -> State:
        outer_task = state.get("task")
        state = await self.inner_env.setup_state(state)
        if outer_task is not None:
            state["task"] = outer_task
        _fix_prompt_after_setup(state)
        return state

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages | str:
        return await self.inner_env.env_response(messages, state, **kwargs)

    @vf.stop
    async def inner_env_done(self, state: State, **kwargs) -> bool:
        for condition in self.inner_env._stop_conditions:
            if await condition(state):
                await self._delete_session(state)
                return True
        return False

    async def _delete_session(self, state: State) -> None:
        if state.get("_tc_cleaned") or "_tc_session_id" not in state:
            return
        state["_tc_cleaned"] = True
        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                await http_client.delete(
                    f"{state['_tc_server_url']}/compact_session/{state['_tc_session_id']}"
                )
        except Exception:
            pass  # best-effort cleanup

    async def get_model_response(
        self,
        state: State,
        prompt: Messages | str,
        client: vf.Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> Response:
        try:
            return await self._get_model_response_impl(state, prompt, client, model, tool_defs, sampling_args)
        except Exception:
            await self._delete_session(state)
            raise

    async def _get_model_response_impl(
        self,
        state: State,
        prompt: Messages | str,
        client: vf.Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> Response:
        client = client if client is not None else state["client"]
        model = model or state["model"]
        sampling_args = sampling_args or state.get("sampling_args") or {}
        if tool_defs is None:
            tool_defs = state.get("tool_defs") or None

        oai_client = client.client
        base_url = str(oai_client.base_url).rstrip("/")
        server_url = base_url.rsplit("/v1", 1)[0] if "/v1" in base_url else base_url

        temperature = sampling_args.get("temperature", self.temperature)
        top_p = sampling_args.get("top_p", self.top_p)

        prompt_dicts = _messages_to_openai(prompt)

        async with httpx.AsyncClient(timeout=7200.0) as http_client:
            tokenize_body = {
                "model": model,
                "messages": prompt_dicts,
                "add_generation_prompt": True,
            }
            oai_tools = _tool_defs_to_openai(tool_defs)
            if oai_tools:
                tokenize_body["tools"] = oai_tools

            tokenize_resp = await http_client.post(
                f"{server_url}/tokenize",
                json=tokenize_body,
            )
            tokenize_resp.raise_for_status()
            full_ids = tokenize_resp.json()["tokens"]

            if "_tc_session_id" not in state:
                # Turn 0: use full tokenization directly
                prompt_ids = full_ids
                session_id = str(uuid.uuid4())
                session_max_kv_len = max(
                    self.max_kv_len,
                    len(prompt_ids) + self.n_max_turns * (self.max_response_tokens + 300),
                ) if self.n_max_turns > 0 else self.max_kv_len
                resp = await http_client.post(
                    f"{server_url}/compact_session/create",
                    json={
                        "session_id": session_id,
                        "prompt_ids": prompt_ids,
                        "max_kv_len": session_max_kv_len,
                        "max_response_tokens": self.max_response_tokens,
                        "compact_target_ratio": self.compact_target_ratio,
                        "compact_window": self.compact_window,
                        "temperature": temperature,
                        "top_p": top_p,
                        "compaction_mode": self.compaction_mode,
                        "use_suffix_queries": self.use_suffix_queries,
                        "n_max_turns": self.n_max_turns,
                        "n_preserved_turns": self.n_preserved_turns,
                    },
                )
                state["_tc_session_id"] = session_id
                state["_tc_server_url"] = server_url
                state["_tc_user_delta_len"] = 0

                # Compute suffix_ids once: chat template tokens appended after
                # an assistant message (e.g. <|im_end|>\n). Needed for TITO-style
                # prompt building on subsequent turns.
                state["_tc_suffix_ids"] = await _compute_suffix_ids(
                    http_client, server_url, model, oai_tools,
                )
            else:
                # Turn t > 0: build prompt_ids using exact previous tokens
                # (TITO-style) to preserve the extension property for
                # interleave_rollout. Retokenizing the full prompt can produce
                # different tokens at BPE merge boundaries, breaking extension.
                prev_prompt_ids = state["_tc_prev_prompt_ids"]
                prev_completion_ids = state["_tc_prev_completion_ids"]
                prev_prefix = prev_prompt_ids + prev_completion_ids

                suffix_ids = state.get("_tc_suffix_ids", [])
                overlap_len = _find_suffix_overlap(prev_prefix, suffix_ids)
                extended_prefix = prev_prefix + suffix_ids[overlap_len:]

                env_response_ids = full_ids[len(extended_prefix):]
                prompt_ids = extended_prefix + env_response_ids

                user_delta = prompt_ids[len(prev_prompt_ids) + len(prev_completion_ids):]
                new_token_ids = [state["_tc_last_token"]] + user_delta
                resp = await http_client.post(
                    f"{server_url}/compact_session/step",
                    json={
                        "session_id": state["_tc_session_id"],
                        "new_token_ids": new_token_ids,
                        "max_response_tokens": self.max_response_tokens,
                    },
                )
                state["_tc_user_delta_len"] = len(user_delta)

            resp.raise_for_status()
            result = resp.json()

        state["_tc_pending_events"] = result.get("diagnostics", {}).get("compaction_events", [])

        all_token_ids = result["all_token_ids"]
        all_logprobs = result.get("all_logprobs", [])
        final_text = result["final_text"]

        content = final_text
        parsed_tool_calls = None
        finish_reason = "stop"
        if tool_defs:
            content, parsed_tool_calls = _parse_hermes_tool_calls(final_text)
            if parsed_tool_calls:
                finish_reason = "tool_calls"

        tokens = ResponseTokens(
            prompt_ids=prompt_ids,
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=all_token_ids,
            completion_mask=[1] * len(all_token_ids),
            completion_logprobs=all_logprobs,
            routed_experts=None,
        )

        return Response(
            id=f"tc-{int(time.time())}",
            created=int(time.time()),
            model=model,
            usage=Usage(
                prompt_tokens=len(prompt_ids),
                reasoning_tokens=0,
                completion_tokens=len(all_token_ids),
                total_tokens=len(prompt_ids) + len(all_token_ids),
            ),
            message=ResponseMessage(
                content=content,
                finish_reason=finish_reason,
                is_truncated=False,
                tokens=tokens,
                tool_calls=parsed_tool_calls or None,
            ),
        )

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ):
        await super().add_model_response(state, prompt_messages, response)

        completion_ids = list(response.message.tokens.completion_ids)
        prompt_ids = list(response.message.tokens.prompt_ids)
        turn = state.get("_tc_turn", 0)

        pending_events = state.pop("_tc_pending_events", [])
        user_delta_len = state.pop("_tc_user_delta_len", 0)

        cumulative = state.get("_tc_cumulative", 0) + user_delta_len + len(completion_ids)
        seg_boundaries = list(state.get("_tc_seg_boundaries", []))

        all_compaction_indices = list(state.get("_tc_compaction_indices", []))
        all_compact_windows = list(state.get("_tc_compact_windows", []))

        if pending_events:
            seg_boundaries.append(cumulative)
            for ev in pending_events:
                all_compaction_indices.append(ev.get("compaction_indices"))
                all_compact_windows.append(ev.get("compact_window"))

        # Always ensure the last element equals the total completion length
        if not seg_boundaries or seg_boundaries[-1] != cumulative:
            seg_boundaries_full = seg_boundaries + [cumulative]
        else:
            seg_boundaries_full = seg_boundaries

        # Store on first trajectory step so interleave_rollout picks it up
        if state["trajectory"]:
            extras = state["trajectory"][0].setdefault("extras", {})
            extras["segment_boundaries"] = seg_boundaries_full
            if all_compaction_indices:
                extras["compaction_indices"] = all_compaction_indices
            if any(w is not None for w in all_compact_windows):
                extras["compact_windows"] = all_compact_windows

        state["_tc_cumulative"] = cumulative
        state["_tc_seg_boundaries"] = seg_boundaries
        state["_tc_compaction_indices"] = all_compaction_indices
        state["_tc_compact_windows"] = all_compact_windows
        state["_tc_prev_prompt_ids"] = prompt_ids
        state["_tc_prev_completion_ids"] = completion_ids
        state["_tc_last_token"] = completion_ids[-1] if completion_ids else 0
        state["_tc_turn"] = turn + 1

        if self.max_turns > 0 and turn + 1 >= self.max_turns:
            await self._delete_session(state)


def load_turn_compaction_environment(
    gym: str = "balrog-bench",
    max_kv_len: int = 4096,
    max_response_tokens: int = 512,
    compact_target_ratio: float = 0.25,
    compact_window: int | None = None,
    n_max_turns: int = -1,
    n_preserved_turns: int = 0,
    max_turns: int = -1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    compaction_mode: str = "attention_matching",
    use_suffix_queries: bool = True,
    **inner_env_kwargs,
) -> TurnCompactionEnv:
    """Load a TurnCompactionEnv wrapping the specified gym environment.

    Called by verifiers' load_environment() when env_id="turn_compaction_env".
    """
    inner_env = vf.load_environment(gym, **inner_env_kwargs)
    return TurnCompactionEnv(
        inner_env=inner_env,
        max_kv_len=max_kv_len,
        max_response_tokens=max_response_tokens,
        compact_target_ratio=compact_target_ratio,
        compact_window=compact_window,
        n_max_turns=n_max_turns,
        n_preserved_turns=n_preserved_turns,
        max_turns=max_turns,
        temperature=temperature,
        top_p=top_p,
        compaction_mode=compaction_mode,
        use_suffix_queries=use_suffix_queries,
    )
