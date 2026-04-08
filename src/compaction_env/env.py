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
import asyncio
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
_FUNCTION_CALL_RE = re.compile(r"<function_calls>(.*?)</function_calls>", re.DOTALL)
_FUNC_SIGNATURE_RE = re.compile(r"(\w+)\((.*)\)", re.DOTALL)
_MISTRAL_TOOL_RE = re.compile(r"\[TOOL_CALLS\]\s*(\w+)\[ARGS\]\s*(\{.*?\})", re.DOTALL)
_RAW_TOOL_RE = re.compile(r"(\w+)\s*(\{[^}]*\})", re.DOTALL)


async def _tokenize_with_retry(
    http_client: httpx.AsyncClient,
    url: str,
    body: dict,
    max_attempts: int = 5,
    delay: float = 3.0,
) -> dict:
    """POST to /tokenize with retries on transient 400 and connection errors."""
    for attempt in range(max_attempts):
        try:
            resp = await http_client.post(url, json=body)
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            if attempt < max_attempts - 1:
                logger.warning(f"/tokenize connection error (attempt {attempt + 1}/{max_attempts}), retrying in {delay}s: {exc}")
                await asyncio.sleep(delay)
                continue
            raise
        if resp.status_code == 400 and attempt < max_attempts - 1:
            logger.warning(f"/tokenize 400 (attempt {attempt + 1}/{max_attempts}), retrying in {delay}s: {resp.text[:300]}")
            if attempt == 0:
                for i, m in enumerate(body.get("messages", [])):
                    if m.get("role") == "assistant" and "tool_calls" in m:
                        logger.warning(f"  msg[{i}] tool_calls: {m['tool_calls']}")
                    if m.get("role") == "assistant":
                        content = m.get("content") or ""
                        if "<tool_call>" in content:
                            logger.warning(f"  msg[{i}] has raw <tool_call> in content: {content[:200]}")
            await asyncio.sleep(delay)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"/tokenize failed after {max_attempts} attempts")


_LOCAL_TOKENIZER_CACHE: dict[str, "PreTrainedTokenizer"] = {}


def _get_local_tokenizer(model: str):
    """Get or create a cached local tokenizer for fallback tokenization."""
    if model not in _LOCAL_TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _LOCAL_TOKENIZER_CACHE[model] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return _LOCAL_TOKENIZER_CACHE[model]


def _build_completion_mask(token_ids: list[int], model: str) -> list[int]:
    """Build completion_mask that zeros out pad token garbage and trailing EOS.

    Chat models like Qwen3 have a split: eos_token=<|im_end|> (turn delimiter)
    and pad_token=<|endoftext|> (document boundary). The model can hallucinate
    <|endoftext|> mid-completion; everything from the first pad token onward is
    garbage and should not receive gradient. The trailing <|im_end|> (forced stop
    signal) is also masked since it's not a model decision.
    """
    tokenizer = _get_local_tokenizer(model)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None or pad_id == eos_id:
        return [1] * len(token_ids)

    mask = []
    active = True
    for tid in token_ids:
        if active and tid == pad_id:
            active = False
        mask.append(1 if active else 0)

    # Mask the trailing EOS (<|im_end|>) — it's a forced stop, not a model action
    if mask and token_ids and token_ids[-1] == eos_id:
        mask[-1] = 0

    return mask


def _strip_template_validation(template: str) -> str:
    """Remove all validation checks from a Jinja chat template.

    Mistral templates enforce strict role alternation and content requirements
    that don't account for multi-turn tool-call patterns.  Stripping all
    raise_exception blocks lets us tokenize these conversations locally
    without changing the actual rendered output.
    """
    import re
    # Remove alternation counter loop
    template = re.sub(
        r"\{%-?\s*set\s+ns\s*=\s*namespace\(index=0\).*?\{%-?\s*endfor\s*%\}",
        "",
        template,
        flags=re.DOTALL,
    )
    # Remove all if-blocks that only raise exceptions
    template = re.sub(
        r"\{%-?\s*if[^}]*%\}\s*\{\{-?\s*raise_exception\([^)]*\)\s*-?\}\}\s*\{%-?\s*endif\s*-?%\}",
        "",
        template,
    )
    return template


_LENIENT_TEMPLATE_CACHE: dict[str, str] = {}


def _tokenize_messages_local(model: str, body: dict) -> dict:
    """Tokenize a chat messages body locally when vLLM /tokenize rejects it."""
    tokenizer = _get_local_tokenizer(model)
    messages = body["messages"]
    kwargs = {}
    if "tools" in body:
        kwargs["tools"] = body["tools"]
    add_gen = body.get("add_generation_prompt", False)

    # Use lenient template (no strict alternation check) if available
    if tokenizer.chat_template and model not in _LENIENT_TEMPLATE_CACHE:
        _LENIENT_TEMPLATE_CACHE[model] = _strip_template_validation(tokenizer.chat_template)
    if model in _LENIENT_TEMPLATE_CACHE:
        kwargs["chat_template"] = _LENIENT_TEMPLATE_CACHE[model]

    # Sanitize None content — Mistral template crashes on content=None
    for msg in messages:
        if msg.get("content") is None:
            msg["content"] = ""

    result = tokenizer.apply_chat_template(messages, add_generation_prompt=add_gen, **kwargs)
    # Some tokenizers return BatchEncoding instead of list[int] when using custom templates
    if hasattr(result, "input_ids"):
        tokens = result.input_ids
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
    else:
        tokens = result
    return {"tokens": list(tokens)}


async def _tokenize_with_retry_messages(
    http_client: httpx.AsyncClient,
    url: str,
    body: dict,
    model: str,
    max_attempts: int = 3,
) -> dict:
    """Tokenize chat messages, using local tokenizer directly when the body contains messages.

    vLLM's /tokenize endpoint rejects chat messages for some tokenizer modes
    (e.g. Mistral). Rather than wasting time on retries, we always tokenize
    messages locally and only use the server for plain prompt strings.
    """
    if "messages" in body:
        return _tokenize_messages_local(model, body)
    return await _tokenize_with_retry(http_client, url, body, max_attempts=max_attempts)


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

    Falls back to local tokenizer if the server /tokenize endpoint rejects
    chat messages (e.g. Mistral tokenizer_mode).
    """
    dummy_content = "World!"
    # Tokenize just the content to find its tokens
    content_body = {
        "model": model,
        "prompt": dummy_content,
    }
    content_data = await _tokenize_with_retry(http_client, f"{server_url}/tokenize", content_body)
    dummy_content_ids = content_data["tokens"]

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
    msgs_data = await _tokenize_with_retry_messages(http_client, f"{server_url}/tokenize", msgs_body, model)
    dummy_msgs_ids = msgs_data["tokens"]

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
    """Parse tool calls from model output text.

    Supports formats:
    - Hermes/Qwen:  <tool_call>{"name": "fn", "arguments": {"k": "v"}}</tool_call>
    - OLMo-3:       <function_calls>fn(k="v")</function_calls>
    - Ministral:    [TOOL_CALLS]fn_name[ARGS]{"k": "v"}
    - Raw:          fn_name{"k": "v"}  (fallback for models that omit markers)

    Returns (content_before_tools, list_of_tool_calls).
    If no tool calls found, returns (text, []).
    """
    has_hermes = "<tool_call>" in text
    has_olmo = "<function_calls>" in text
    has_mistral = "[TOOL_CALLS]" in text

    tool_calls = []

    # Hermes format: <tool_call>{"name": "fn", "arguments": {...}}</tool_call>
    if has_hermes:
        for match in _TOOL_CALL_RE.finditer(text):
            raw = match.group(1).strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Skipping unparseable tool_call: {raw[:100]}")
                continue
            if "name" not in parsed:
                logger.warning(f"Skipping tool_call missing 'name': {raw[:100]}")
                continue
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

    # OLMo-3 format: <function_calls>fn(k="v", k2="v2")</function_calls>
    if has_olmo:
        for match in _FUNCTION_CALL_RE.finditer(text):
            raw = match.group(1).strip()
            for line in raw.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                sig_match = _FUNC_SIGNATURE_RE.match(line)
                if not sig_match:
                    logger.warning(f"Skipping unparseable function_call: {line[:100]}")
                    continue
                func_name = sig_match.group(1)
                args_str = sig_match.group(2).strip()
                arguments = {}
                if args_str:
                    for part in re.split(r",\s*(?=\w+=)", args_str):
                        if "=" in part:
                            k, v = part.split("=", 1)
                            v = v.strip().strip('"').strip("'")
                            arguments[k.strip()] = v
                tool_calls.append(ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=func_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ))
        content = text[:text.find("<function_calls>")].strip() or None
        return content, tool_calls

    # Ministral format: [TOOL_CALLS]fn_name[ARGS]{"k": "v"}
    if has_mistral:
        for match in _MISTRAL_TOOL_RE.finditer(text):
            func_name = match.group(1)
            args_raw = match.group(2).strip()
            try:
                arguments = json.loads(args_raw)
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
            except json.JSONDecodeError:
                arguments = args_raw
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=func_name,
                arguments=arguments,
            ))
        content = text[:text.find("[TOOL_CALLS]")].strip() or None
        return content, tool_calls

    # Raw fallback: fn_name{"k": "v"} (no markers)
    match = _RAW_TOOL_RE.search(text)
    if match:
        func_name = match.group(1)
        args_raw = match.group(2).strip()
        try:
            arguments = json.loads(args_raw)
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
        except json.JSONDecodeError:
            arguments = args_raw
        tool_calls.append(ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=func_name,
            arguments=arguments,
        ))
        content = text[:match.start()].strip() or None
        return content, tool_calls

    return text, []


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


def _sanitize_tool_arguments(args: str) -> str:
    """Ensure tool call arguments are valid JSON for vLLM's _postprocess_messages."""
    if not args or not args.strip():
        return "{}"
    try:
        json.loads(args)
        return args
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"raw": args})


def _tool_call_to_openai(tc: dict) -> dict:
    """Convert a verifiers-style tool_call dict to OpenAI format."""
    if "function" in tc:
        fn = tc["function"]
        if "arguments" in fn:
            fn["arguments"] = _sanitize_tool_arguments(fn["arguments"])
        return tc
    return {
        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
        "type": "function",
        "function": {
            "name": tc["name"],
            "arguments": _sanitize_tool_arguments(tc.get("arguments", "{}")),
        },
    }


def _last_n_turns(turn_msgs: list[dict], n: int) -> list[dict]:
    """Return the messages belonging to the last *n* turns.

    A "turn" starts at each ``user`` message and includes all subsequent
    non-``user`` messages (assistant replies, tool responses, etc.).
    This correctly handles tool-calling envs where each turn has 3+
    messages instead of the simple user/assistant pair.
    """
    if n <= 0:
        return []
    # Walk backwards, counting user messages as turn boundaries.
    count = 0
    cut = len(turn_msgs)
    for i in range(len(turn_msgs) - 1, -1, -1):
        if turn_msgs[i].get("role") == "user":
            count += 1
            if count >= n:
                cut = i
                break
    return turn_msgs[cut:]


def _messages_to_openai(prompt) -> list[dict]:
    """Convert a list of messages to OpenAI-compatible dicts for /tokenize.

    Strips verifiers-specific fields (reasoning_content, thinking_blocks) that
    vLLM doesn't understand, and removes None/empty tool_calls to avoid tripping
    vLLM's chat template rendering.
    """
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
        # Strip verifiers-only fields that vLLM doesn't understand
        for key in ("reasoning_content", "thinking_blocks"):
            d.pop(key, None)
        if "tool_calls" in d:
            if d["tool_calls"]:
                d["tool_calls"] = [_tool_call_to_openai(tc) for tc in d["tool_calls"]]
            else:
                del d["tool_calls"]
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
            updated_prompt += "\n\nYou must call the take_action tool exactly once per response. Do not include any other text or explanation — just the tool call."
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

            tokenize_data = await _tokenize_with_retry_messages(
                http_client, f"{server_url}/tokenize", tokenize_body, model,
            )
            prompt_ids = tokenize_data["tokens"]

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
            completion_mask=_build_completion_mask(all_token_ids, model),
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
        summary_max_tokens: int = 512,
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
        self.summary_max_tokens = summary_max_tokens

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

    def _rebuild_prompt_with_summary(self, prompt_dicts: list[dict], state: State) -> list[dict]:
        """Replace old turns with summary in prompt_dicts for TITO compatibility.

        After a summary reset, verifiers still passes the full message history.
        This rebuilds the prompt to match what the session KV actually contains:
        [system | preserved_turns | sum_user | sum_asst | new_turns_since_reset | current_obs].
        """
        current_user = prompt_dicts[-1]
        turn_msgs = prompt_dicts[1:-1]

        # Keep preserved turns + turns accumulated since last summary reset
        summary_turn_count = state.get("_tc_summary_turn_count", 0)
        n_recent = self.n_preserved_turns + summary_turn_count
        recent = _last_n_turns(turn_msgs, n_recent) if n_recent > 0 else turn_msgs

        summary_msgs = state.get("_tc_summary_msgs", [])
        return [prompt_dicts[0]] + recent + summary_msgs + [current_user]

    def _rebuild_prompt_after_pure_reset(self, prompt_dicts: list[dict], state: State) -> list[dict]:
        """Trim prompt_dicts to match session KV after a markovian_pure reset.

        After reset, the session only contains [system | preserved_turns]. Verifiers
        still passes full history, so trim to [system | recent_turns | current_obs].
        """
        current_user = prompt_dicts[-1]
        turn_msgs = prompt_dicts[1:-1]

        n_recent = self.n_preserved_turns + state.get("_tc_summary_turn_count", 0)
        recent = _last_n_turns(turn_msgs, n_recent) if n_recent > 0 else turn_msgs

        return [prompt_dicts[0]] + recent + [current_user]

    async def _markovian_pure_reset(self, state: State, prompt_dicts: list[dict]) -> list[dict]:
        """Drop old turns, delete session, return prompt with only preserved turns."""
        current_user = prompt_dicts[-1]
        turn_msgs = prompt_dicts[1:-1]

        n_keep = self.n_preserved_turns
        preserved = _last_n_turns(turn_msgs, n_keep) if n_keep > 0 else turn_msgs

        await self._delete_session(state)
        for key in ["_tc_session_id", "_tc_prev_prompt_ids", "_tc_prev_completion_ids",
                     "_tc_last_token", "_tc_suffix_ids"]:
            state.pop(key, None)
        state["_tc_cleaned"] = False
        state["_tc_summary_turn_count"] = 0
        state["_tc_pure_reset_done"] = True

        # Reset cumulative tracking — no server-side compaction occurred,
        # so stale boundaries would be meaningless for training.
        state["_tc_cumulative"] = 0
        state["_tc_seg_boundaries"] = []
        state["_tc_compaction_indices"] = []
        state["_tc_compact_windows"] = []
        if state.get("trajectory"):
            extras = state["trajectory"][0].get("extras", {})
            extras.pop("segment_boundaries", None)
            extras.pop("compaction_indices", None)
            extras.pop("compact_windows", None)

        return [prompt_dicts[0]] + preserved + [current_user]

    def _build_summary_prompt_text(self, turn_msgs: list[dict], prev_summary: str) -> str:
        """Build the summary prompt text from all window turns + prior summary."""
        parts = []
        for i in range(0, len(turn_msgs), 2):
            if i + 1 < len(turn_msgs):
                obs = (turn_msgs[i].get("content") or "")[:2000]
                resp = (turn_msgs[i + 1].get("content") or "")[:2000]
                parts.append(f"Observation:\n{obs}\nResponse:\n{resp}")

        context = f"Previous context:\n{prev_summary}\n\n" if prev_summary else ""
        interaction = "\n---\n".join(parts)

        return (
            "Briefly summarize this interaction. "
            "Focus on mission, actions, discoveries, and progress. "
            "2-3 sentences.\n\n"
            f"{context}Interaction:\n{interaction}"
        )

    async def _summary_reset(
        self,
        state: State,
        prompt_dicts: list[dict],
        http_client: httpx.AsyncClient,
        server_url: str,
        model: str,
    ) -> list[dict]:
        """Generate summary of old turns, delete session, return modified prompt.

        Summary becomes a user+assistant turn at the END of the preserved window:
        [sys][preserved_turns][sum_user][sum_asst][current_user].
        """
        system_msg = prompt_dicts[0]
        current_user = prompt_dicts[-1]
        turn_msgs = prompt_dicts[1:-1]

        n_keep = self.n_preserved_turns
        preserved = _last_n_turns(turn_msgs, n_keep) if n_keep > 0 else []

        prev_summary = state.get("_tc_summary", "")
        summary_prompt_text = self._build_summary_prompt_text(turn_msgs, prev_summary)

        resp = await http_client.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": summary_prompt_text}],
                "max_tokens": self.summary_max_tokens,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        summary_text = resp.json()["choices"][0]["message"]["content"]
        if "</think>" in summary_text:
            summary_text = summary_text.split("</think>", 1)[-1].strip()

        state["_tc_summary"] = summary_text
        state["_tc_summary_msgs"] = [
            {"role": "user", "content": summary_prompt_text},
            {"role": "assistant", "content": summary_text},
        ]
        logger.info("Summary reset: generated %d-char summary", len(summary_text))

        # Delete old session and clean up state to force recreation
        await self._delete_session(state)
        for key in ["_tc_session_id", "_tc_prev_prompt_ids", "_tc_prev_completion_ids",
                     "_tc_last_token", "_tc_suffix_ids"]:
            state.pop(key, None)
        state["_tc_cleaned"] = False
        state["_tc_summary_turn_count"] = 0

        # Reset cumulative tracking — no server-side compaction occurred,
        # so stale boundaries would be meaningless for training.
        state["_tc_cumulative"] = 0
        state["_tc_seg_boundaries"] = []
        state["_tc_compaction_indices"] = []
        state["_tc_compact_windows"] = []
        if state.get("trajectory"):
            extras = state["trajectory"][0].get("extras", {})
            extras.pop("segment_boundaries", None)
            extras.pop("compaction_indices", None)
            extras.pop("compact_windows", None)

        return [system_msg] + preserved + state["_tc_summary_msgs"] + [current_user]

    async def _kv_summary_compact(
        self,
        state: State,
        prompt_dicts: list[dict],
        http_client: httpx.AsyncClient,
        server_url: str,
        model: str,
    ) -> None:
        """Generate summary with full KV context, then trigger markovian compaction.

        Unlike _summary_reset, the session stays alive. The summary is generated
        as a compact_session/step with is_summary=True (keeping KVs), then
        trigger_compact=True fires markovian deletion of the oldest turn.
        """
        turn_msgs = prompt_dicts[1:-1]
        prev_summary = state.get("_tc_summary", "")
        summary_prompt_text = self._build_summary_prompt_text(turn_msgs, prev_summary)

        # Build TITO-compatible token delta: tokenize the full conversation with
        # the summary user message appended (through the chat template), then
        # extract the delta after the known prefix.
        prev_prompt_ids = state["_tc_prev_prompt_ids"]
        prev_completion_ids = state["_tc_prev_completion_ids"]
        suffix_ids = state.get("_tc_suffix_ids", [])
        overlap_len = _find_suffix_overlap(prev_prompt_ids + prev_completion_ids, suffix_ids)
        extended_prefix = prev_prompt_ids + prev_completion_ids + suffix_ids[overlap_len:]

        # Replace current user obs with summary prompt for tokenization
        summary_prompt_dicts = prompt_dicts[:-1] + [{"role": "user", "content": summary_prompt_text}]
        tokenize_resp = await _tokenize_with_retry_messages(
            http_client, f"{server_url}/tokenize",
            {"model": model, "messages": summary_prompt_dicts, "add_generation_prompt": True},
            model,
        )
        full_ids = tokenize_resp["tokens"]

        summary_user_ids = full_ids[len(extended_prefix):]
        new_token_ids = [state["_tc_last_token"]] + summary_user_ids

        resp = await http_client.post(
            f"{server_url}/compact_session/step",
            json={
                "session_id": state["_tc_session_id"],
                "new_token_ids": new_token_ids,
                "max_response_tokens": self.summary_max_tokens,
                "is_summary": True,
                "trigger_compact": True,
            },
        )
        if resp.status_code >= 500:
            for retry in range(4):
                logger.warning(
                    "%s returned %d (attempt %d/5), retrying in 5s: %s",
                    resp.request.url, resp.status_code, retry + 2, resp.text[:300],
                )
                await asyncio.sleep(5)
                resp = await http_client.post(
                    str(resp.request.url),
                    json=json.loads(resp.request.content),
                )
                if resp.status_code < 500:
                    break
        if resp.status_code >= 400:
            logger.error("%s returned %d: %s", resp.request.url, resp.status_code, resp.text[:500])
            resp.raise_for_status()

        result = resp.json()
        summary_text = result["final_text"]
        if "</think>" in summary_text:
            summary_text = summary_text.split("</think>", 1)[-1].strip()

        state["_tc_summary"] = summary_text
        state["_tc_summary_msgs"] = [
            {"role": "user", "content": summary_prompt_text},
            {"role": "assistant", "content": summary_text},
        ]
        state["_tc_summary_turn_count"] = 0

        # Update TITO state: the summary step's tokens become the new prev
        all_token_ids = result["all_token_ids"]
        state["_tc_prev_prompt_ids"] = extended_prefix + summary_user_ids
        state["_tc_prev_completion_ids"] = all_token_ids
        state["_tc_last_token"] = all_token_ids[-1] if all_token_ids else state["_tc_last_token"]
        state["_tc_user_delta_len"] = len(summary_user_ids)

        # Store compaction events for segment boundary tracking
        pending = result.get("diagnostics", {}).get("compaction_events", [])
        state["_tc_pending_events"] = state.get("_tc_pending_events", []) + pending

        logger.info("KV-summary compact: generated %d-char summary, %d compaction events",
                     len(summary_text), len(pending))

    def _rebuild_prompt_with_kv_summary(self, prompt_dicts: list[dict], state: State) -> list[dict]:
        """Trim prompt to match KV state after kv_summary compaction.

        After kv_summary, the KV cache contains:
        [sys | preserved_turns | sum_user | sum_asst | new_turns_since_compact].
        """
        current_user = prompt_dicts[-1]
        turn_msgs = prompt_dicts[1:-1]

        summary_turn_count = state.get("_tc_summary_turn_count", 0)
        n_recent = self.n_preserved_turns + summary_turn_count
        recent = _last_n_turns(turn_msgs, n_recent) if n_recent > 0 else turn_msgs

        summary_msgs = state.get("_tc_summary_msgs", [])
        return [prompt_dicts[0]] + recent + summary_msgs + [current_user]

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

        # After a client-side reset, rebuild prompt to match session KV contents
        if state.get("_tc_summary_msgs") and self.compaction_mode in ("kv_summary", "kv_summary_grad"):
            prompt_dicts = self._rebuild_prompt_with_kv_summary(prompt_dicts, state)
        elif state.get("_tc_summary_msgs"):
            prompt_dicts = self._rebuild_prompt_with_summary(prompt_dicts, state)
        elif state.get("_tc_pure_reset_done"):
            prompt_dicts = self._rebuild_prompt_after_pure_reset(prompt_dicts, state)

        async with httpx.AsyncClient(timeout=7200.0) as http_client:
            # Client-managed compaction when window is full
            _CLIENT_MANAGED_MODES = ("summary", "markovian_pure", "kv_summary", "kv_summary_grad")
            if (self.compaction_mode in _CLIENT_MANAGED_MODES
                    and self.n_max_turns >= 0
                    and state.get("_tc_summary_turn_count", 0) >= self.n_max_turns
                    and "_tc_session_id" in state):
                if self.compaction_mode == "summary":
                    prompt_dicts = await self._summary_reset(
                        state, prompt_dicts, http_client, server_url, model,
                    )
                elif self.compaction_mode in ("kv_summary", "kv_summary_grad"):
                    await self._kv_summary_compact(
                        state, prompt_dicts, http_client, server_url, model,
                    )
                else:
                    prompt_dicts = await self._markovian_pure_reset(state, prompt_dicts)

            tokenize_body = {
                "model": model,
                "messages": prompt_dicts,
                "add_generation_prompt": True,
            }
            oai_tools = _tool_defs_to_openai(tool_defs)
            if oai_tools:
                tokenize_body["tools"] = oai_tools

            tokenize_data = await _tokenize_with_retry_messages(
                http_client, f"{server_url}/tokenize", tokenize_body, model,
            )
            full_ids = tokenize_data["tokens"]

            if "_tc_session_id" not in state:
                # Turn 0 (or first turn after summary reset): use full tokenization
                prompt_ids = full_ids
                session_id = str(uuid.uuid4())

                # Tokenize just the system message to get its token length.
                # This lets the server separate system prompt from U0 so that
                # U0 enters the compaction window instead of being permanently
                # preserved.
                sys_tok = await _tokenize_with_retry_messages(
                    http_client, f"{server_url}/tokenize",
                    {"model": model, "messages": [prompt_dicts[0]], "add_generation_prompt": False},
                    model,
                )
                system_prompt_len = len(sys_tok["tokens"])
                state["_tc_system_prompt_len"] = system_prompt_len

                session_max_kv_len = max(
                    self.max_kv_len,
                    len(prompt_ids) + self.n_max_turns * (self.max_response_tokens + 300),
                ) if self.n_max_turns > 0 else self.max_kv_len

                # Client-managed modes: no automatic server-side turn compaction
                client_managed = self.compaction_mode in ("summary", "markovian_pure")
                if self.compaction_mode in ("kv_summary", "kv_summary_grad"):
                    # kv_summary: server uses markovian delete, but client triggers it
                    server_compaction_mode = "markovian"
                    server_n_max_turns = -1
                    server_n_preserved = self.n_preserved_turns
                elif client_managed:
                    server_compaction_mode = "attention_matching"
                    server_n_max_turns = -1
                    server_n_preserved = 0
                else:
                    _MODE_TO_SERVER = {"kv_markovian_grad": "markovian"}
                    server_compaction_mode = _MODE_TO_SERVER.get(self.compaction_mode, self.compaction_mode)
                    server_n_max_turns = self.n_max_turns
                    server_n_preserved = self.n_preserved_turns

                create_body = {
                    "session_id": session_id,
                    "prompt_ids": prompt_ids,
                    "max_kv_len": session_max_kv_len,
                    "max_response_tokens": self.max_response_tokens,
                    "compact_target_ratio": self.compact_target_ratio,
                    "compact_window": self.compact_window,
                    "temperature": temperature,
                    "top_p": top_p,
                    "compaction_mode": server_compaction_mode,
                    "use_suffix_queries": self.use_suffix_queries,
                    "n_max_turns": server_n_max_turns,
                    "n_preserved_turns": server_n_preserved,
                    "system_prompt_len": system_prompt_len,
                }
                while True:
                    resp = await http_client.post(
                        f"{server_url}/compact_session/create", json=create_body,
                    )
                    if resp.status_code < 500:
                        break
                    logger.warning(
                        "compact_session/create returned %d (no blocks?), retrying in 5s: %s",
                        resp.status_code, resp.text[:200],
                    )
                    await asyncio.sleep(5)
                resp.raise_for_status()
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

            # Retry on 500s (server not ready or transient error)
            if resp.status_code >= 500:
                for retry in range(19):
                    logger.warning(
                        "%s returned %d (attempt %d/20), retrying in 10s: %s",
                        resp.request.url, resp.status_code, retry + 2, resp.text[:300],
                    )
                    await asyncio.sleep(10)
                    resp = await http_client.post(
                        str(resp.request.url),
                        json=json.loads(resp.request.content),
                    )
                    if resp.status_code < 500:
                        break
            if resp.status_code >= 400:
                logger.error(
                    "%s returned %d: %s",
                    resp.request.url, resp.status_code, resp.text[:500],
                )
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
            completion_mask=_build_completion_mask(all_token_ids, model),
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
            sys_len = state.get("_tc_system_prompt_len")
            if sys_len is not None:
                extras["system_prompt_len"] = sys_len

        state["_tc_cumulative"] = cumulative
        state["_tc_seg_boundaries"] = seg_boundaries
        state["_tc_compaction_indices"] = all_compaction_indices
        state["_tc_compact_windows"] = all_compact_windows
        state["_tc_prev_prompt_ids"] = prompt_ids
        state["_tc_prev_completion_ids"] = completion_ids
        state["_tc_last_token"] = completion_ids[-1] if completion_ids else 0
        state["_tc_turn"] = turn + 1

        if self.compaction_mode in ("summary", "markovian_pure", "kv_summary", "kv_summary_grad"):
            state["_tc_summary_turn_count"] = state.get("_tc_summary_turn_count", 0) + 1

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
    summary_max_tokens: int = 512,
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
        summary_max_tokens=summary_max_tokens,
    )
