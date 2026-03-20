"""Segmented forward pass with KV cache compaction replay for training.

When segment_boundaries is present in a micro batch, the trainer replays
compaction during training to compute correct logprobs for post-compaction tokens.

The trainer's own KV cache is used (not inference's), making the importance
ratio pi_theta / pi_old unbiased under View B (compaction as part of policy).

Boundary handling: after compaction, the inference worker reprocesses the
boundary token (last of previous segment) with compacted context. The trainer
must do the same — each segment after the first starts from the boundary token,
and its first logit (predicting the first new token) replaces the stale
pre-compaction logit from the previous segment.

Beta correction: when compute_beta=True, the NNLS solver produces per-key
additive biases that correct the partition function mismatch between full and
compacted attention. In inference, BetaAttentionWrapper applies these via custom
SDPA. In training, we achieve the same effect model-agnostically via
forward_pre_hooks on attention layers that modify the attention_mask to include
per-head beta at compacted positions. All HF attention implementations (eager,
SDPA) broadcast attention_mask into attention logits, so a 4D mask with per-head
beta at compacted positions achieves identical behavior.
"""

import logging

import torch
from torch import Tensor
from transformers import DynamicCache

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from prime_rl.inference.compaction.algorithm import compact_kv
from prime_rl.trainer.models.layers.lora import base as lora_base

logger = logging.getLogger(__name__)


# ── Beta attention hooks for training ──────────────────────────────────────


class _BetaTrainingState:
    """Per-layer beta bias for forward_pre_hooks during training.

    After compaction, stores (1, num_heads, 1, kv_len) bias tensors per layer.
    Hooks on attention modules add these to the attention_mask, matching the
    inference BetaAttentionWrapper behavior.
    """

    def __init__(self, num_kv_heads: int, num_heads: int):
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.heads_per_group = num_heads // num_kv_heads
        self.active = False
        self.beta_per_layer: dict[int, Tensor] = {}

    def set_beta(
        self,
        beta_list: list[Tensor],
        prompt_len: int,
        compacted_len: int,
        total_kv_len: int,
        device: torch.device,
    ):
        """Update beta from compact_kv output.

        Args:
            beta_list: Per-layer tensors of shape (target_len, num_kv_heads)
            prompt_len: Number of prompt tokens
            compacted_len: Number of compacted prefix tokens
            total_kv_len: Total KV cache length after compaction
        """
        self.active = True
        for layer_idx, beta in enumerate(beta_list):
            # beta: (target_len, num_kv_heads)
            bias = torch.zeros(1, self.num_heads, 1, total_kv_len,
                               device=device, dtype=torch.float32)
            # Expand from kv_heads to num_heads for GQA: each kv-head serves
            # heads_per_group query heads (interleaved layout)
            beta_clamped = beta.T.clamp(-30.0, 30.0)  # (num_kv_heads, target_len)
            beta_expanded = beta_clamped.repeat_interleave(
                self.heads_per_group, dim=0)  # (num_heads, target_len)
            bias[0, :, 0, prompt_len:prompt_len + compacted_len] = beta_expanded
            self.beta_per_layer[layer_idx] = bias

    def clear(self):
        self.active = False
        self.beta_per_layer.clear()


def _find_attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Find attention modules by looking for q_proj/k_proj/v_proj.

    Works for Llama, Qwen2, Qwen3, Mistral, Gemma, etc.
    """
    modules = []
    for _name, module in model.named_modules():
        if (hasattr(module, 'q_proj') and hasattr(module, 'k_proj')
                and hasattr(module, 'v_proj')):
            modules.append(module)
    return modules


def _make_beta_hook(layer_idx: int, beta_state: _BetaTrainingState):
    """Create a forward_pre_hook that adds beta bias to attention_mask."""

    def hook(module, args, kwargs):
        if not beta_state.active or layer_idx not in beta_state.beta_per_layer:
            return args, kwargs

        attention_mask = kwargs.get('attention_mask')
        if attention_mask is None:
            return args, kwargs

        beta_bias = beta_state.beta_per_layer[layer_idx]
        KV = attention_mask.shape[-1]
        bias_slice = beta_bias[:, :, :, :KV].to(attention_mask.dtype)

        # (B, 1, Q, KV) + (1, H, 1, KV) broadcasts to (B, H, Q, KV)
        kwargs = {**kwargs, 'attention_mask': attention_mask + bias_slice}
        return args, kwargs

    return hook


def _register_beta_hooks(
    model: torch.nn.Module,
    beta_state: _BetaTrainingState,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register forward_pre_hooks on attention layers for beta bias injection."""
    hooks = []
    attn_modules = _find_attention_modules(model)
    for layer_idx, module in enumerate(attn_modules):
        handle = module.register_forward_pre_hook(
            _make_beta_hook(layer_idx, beta_state),
            with_kwargs=True,
        )
        hooks.append(handle)
    logger.debug("Registered beta hooks on %d attention layers", len(hooks))
    return hooks


def _make_query_capture_hook(layer_idx: int, buffer: dict[int, Tensor]):
    """Create a forward_pre_hook that captures Q after q_proj + q_norm + RoPE.

    Works with HuggingFace Qwen3Attention which has separate q_proj, q_norm,
    and receives position_embeddings as a kwarg for RoPE.
    """

    def hook(module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        if hidden_states is None:
            return

        head_dim = module.head_dim
        hidden_shape = (*hidden_states.shape[:-1], -1, head_dim)

        q = module.q_proj(hidden_states).view(hidden_shape)
        if hasattr(module, 'q_norm'):
            q = module.q_norm(q)
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

        position_embeddings = kwargs.get('position_embeddings')
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # apply_rotary_pos_emb expects (batch, heads, seq, dim)
            # Import from the model's own module to match its RoPE implementation
            model_module = type(module).__module__
            import importlib
            mod = importlib.import_module(model_module)
            rope_fn = getattr(mod, 'apply_rotary_pos_emb')
            q, _ = rope_fn(q, q, cos, sin)

        # Store detached: (batch, num_heads, seq_len, head_dim)
        buffer[layer_idx] = q.detach()

    return hook


def _extract_suffix_queries_from_buffer(
    buffer: dict[int, Tensor],
    num_layers: int,
    suffix_start_in_segment: int,
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
) -> list[Tensor]:
    """Extract suffix queries from captured buffer, grouped into KV-head space.

    Args:
        buffer: {layer_idx: (1, num_attn_heads, seg_len, head_dim)} captured queries
        suffix_start_in_segment: index within the segment where the suffix begins
        num_kv_heads: number of KV heads for GQA grouping
        head_size: dimension per head

    Returns:
        suffix_queries[layer]: (num_kv_heads, suffix_len * heads_per_group, head_size)
    """
    result = []
    for l in range(num_layers):
        if l not in buffer:
            result.append(torch.zeros(num_kv_heads, 0, head_size,
                                      device=device, dtype=torch.float32))
            continue
        q = buffer[l]  # (1, num_attn_heads, seg_len, head_dim)
        q = q[0, :, suffix_start_in_segment:, :]  # (num_attn_heads, suffix_len, head_dim)
        num_attn_heads = q.shape[0]
        suffix_len = q.shape[1]
        if suffix_len == 0:
            result.append(torch.zeros(num_kv_heads, 0, head_size,
                                      device=device, dtype=torch.float32))
            continue
        heads_per_group = num_attn_heads // num_kv_heads
        # (num_kv_heads, heads_per_group, suffix_len, head_dim)
        q = q.view(num_kv_heads, heads_per_group, suffix_len, head_size)
        # Concat GQA group heads as separate queries
        q = q.reshape(num_kv_heads, heads_per_group * suffix_len, head_size).float()
        result.append(q)
    return result


def _get_kv_from_cache(cache: DynamicCache) -> tuple[list[Tensor], list[Tensor], int, int]:
    """Extract key/value tensors and dimensions from a DynamicCache.

    Handles both old API (cache.key_cache) and new API (cache.layers[i].keys).

    Returns:
        keys: list of [seq, heads, dim] per layer
        values: list of [seq, heads, dim] per layer
        num_layers, num_kv_heads, head_size
    """
    num_layers = len(cache)

    # Try new API first (transformers >= 4.49)
    if hasattr(cache, 'layers') and len(cache.layers) > 0:
        layer0 = cache.layers[0]
        k0 = layer0.keys  # [batch, heads, seq, dim]
        num_kv_heads = k0.shape[1]
        kv_seq_len = k0.shape[2]
        head_size = k0.shape[3]

        keys = [cache.layers[l].keys[0].permute(1, 0, 2).contiguous()
                for l in range(num_layers)]
        values = [cache.layers[l].values[0].permute(1, 0, 2).contiguous()
                  for l in range(num_layers)]
    # Fall back to old API
    elif hasattr(cache, 'key_cache'):
        k0 = cache.key_cache[0]  # [batch, heads, seq, dim]
        num_kv_heads = k0.shape[1]
        kv_seq_len = k0.shape[2]
        head_size = k0.shape[3]

        keys = [cache.key_cache[l][0].permute(1, 0, 2).contiguous()
                for l in range(num_layers)]
        values = [cache.value_cache[l][0].permute(1, 0, 2).contiguous()
                  for l in range(num_layers)]
    else:
        raise RuntimeError(f"Unknown DynamicCache API: {type(cache)}")

    return keys, values, num_kv_heads, head_size


def segmented_forward(
    model: torch.nn.Module,
    input_ids: Tensor,
    position_ids: Tensor,
    segment_boundaries: list[int],
    prompt_len: int,
    compact_target_ratio: float,
    compact_window: int | None,
    temperature: Tensor,
    max_forward_passes: int | None = None,
    compute_beta: bool = False,
    use_suffix_queries: bool = True,
    compaction_indices: list | None = None,
    compaction_mode: str = "attention_matching",
) -> dict[str, Tensor]:
    """Run segmented forward passes with compaction replay between segments.

    For each segment:
    1. Forward pass with use_cache=True to get logits and KV cache
    2. Between segments: extract KV, run compaction, detach, create new past_key_values
    3. Next segment uses compacted past_key_values, starting from boundary token

    The boundary token (last of previous segment) is re-fed as the first input
    of the next segment so that its logit (predicting the first new token) is
    computed with compacted context, matching inference behavior.

    Args:
        model: HuggingFace model (e.g. Qwen3ForCausalLM)
        input_ids: Full input_ids [1, seq_len] (prompt + all completion tokens)
        position_ids: Full position_ids [1, seq_len]
        segment_boundaries: Cumulative completion token counts at end of each segment
        prompt_len: Number of prompt tokens
        compact_target_ratio: Fraction of prefix keys to keep
        compact_window: If set, only compress first N assistant tokens per compaction
        temperature: Per-token temperatures [1, seq_len]

    Returns:
        Dict with "logits" key containing [1, seq_len, vocab] temperature-scaled logits
    """
    device = input_ids.device
    assert input_ids.shape[0] == 1, "Segmented forward only supports batch_size=1"

    # Capture past_key_values from the backbone via a hook, since the top-level
    # model output (through FSDP2 + VanillaOutputLinear) may not propagate it.
    captured_kv = {}

    def _capture_kv_hook(_module, _input, output):
        if hasattr(output, 'past_key_values'):
            captured_kv['past_key_values'] = output.past_key_values
        elif isinstance(output, dict):
            captured_kv['past_key_values'] = output.get('past_key_values')

    backbone = model.model if hasattr(model, 'model') else model
    hook_handle = backbone.register_forward_hook(_capture_kv_hook)

    # Build segment token ranges in input_ids space
    seg_input_ranges = []
    prev_boundary = 0
    for i, boundary in enumerate(segment_boundaries):
        if i == 0:
            seg_start = 0
        else:
            seg_start = prompt_len + prev_boundary - 1
        seg_end = prompt_len + boundary
        seg_input_ranges.append((seg_start, seg_end))
        prev_boundary = boundary

    all_logits_pieces = []
    past_key_values = None
    position_offset = 0

    # Beta training state (lazy-initialized on first compaction with beta)
    beta_state: _BetaTrainingState | None = None
    beta_hooks: list = []

    # Save reference to the LoRA num_tokens tensor and its original value.
    # MultiLoRALinear instances hold a direct reference to this tensor,
    # so we must modify it in-place (not replace it).
    saved_lora_num_tokens = lora_base.LORA_NUM_TOKENS
    original_lora_value = saved_lora_num_tokens[0].item() if saved_lora_num_tokens is not None else None

    # Disable activation checkpointing only when LoRA is active.
    # AC recomputes forward during backward, but LoRA offsets are global state
    # that changes per-segment, causing offset mismatches during recomputation.
    # For Full FT (no LoRA), AC is safe and critical for memory.
    saved_checkpoint_fns = {}
    if saved_lora_num_tokens is not None:
        for name, module in backbone.named_modules():
            if isinstance(module, CheckpointWrapper):
                saved_checkpoint_fns[name] = module.checkpoint_fn
                module.checkpoint_fn = lambda fn, *args, **kwargs: fn(*args, **kwargs)

    # Query capture for suffix queries (lazy-initialized)
    query_capture_hooks: list = []
    query_buffer: dict[int, Tensor] = {}

    for seg_idx, (seg_start, seg_end) in enumerate(seg_input_ranges):
        seg_ids = input_ids[:, seg_start:seg_end]
        seg_positions = position_ids[:, seg_start:seg_end]
        seg_temps = temperature[:, seg_start:seg_end]

        # LoRA expects offsets matching the segment's token count.
        # Must use in-place copy (reset_reference=False) because MultiLoRALinear
        # instances hold a reference to the original LORA_NUM_TOKENS tensor.
        if saved_lora_num_tokens is not None:
            seg_len = seg_end - seg_start
            saved_lora_num_tokens[0] = seg_len

        # Register query capture hooks for non-last segments when using suffix queries
        # (not needed for markovian mode — no importance scoring)
        is_last_segment = (seg_idx == len(seg_input_ranges) - 1)
        if use_suffix_queries and compaction_mode != "markovian" and not is_last_segment and not query_capture_hooks:
            attn_modules = _find_attention_modules(model)
            for layer_idx, module in enumerate(attn_modules):
                handle = module.register_forward_pre_hook(
                    _make_query_capture_hook(layer_idx, query_buffer),
                    with_kwargs=True,
                )
                query_capture_hooks.append(handle)

        out = model(
            input_ids=seg_ids,
            position_ids=seg_positions,
            past_key_values=past_key_values,
            use_cache=True,
        )

        raw_logits = out["logits"] if isinstance(out, dict) else out.logits
        # VanillaOutputLinear returns PrimeLmOutput(logits=tensor), which HF wraps as
        # CausalLMOutputWithPast(logits={"logits": tensor}). Unwrap if needed.
        seg_logits = raw_logits["logits"] if isinstance(raw_logits, dict) else raw_logits  # [1, seg_len, vocab]
        scaled_seg_logits = seg_logits / seg_temps.unsqueeze(-1).to(seg_logits.dtype)

        is_last_segment = (seg_idx == len(seg_input_ranges) - 1)

        if is_last_segment:
            all_logits_pieces.append(scaled_seg_logits)
        else:
            # Drop last logit — it will be recomputed post-compaction
            # by the next segment's forward pass (boundary token overlap)
            all_logits_pieces.append(scaled_seg_logits[:, :-1, :])

        if seg_idx < len(seg_input_ranges) - 1:
            kv_cache = captured_kv.get('past_key_values')
            assert kv_cache is not None, "Hook did not capture past_key_values. Check backbone hook."
            captured_kv.clear()

            keys, values, num_kv_heads, head_size = _get_kv_from_cache(kv_cache)
            num_layers = len(keys)
            kv_seq_len = keys[0].shape[0]

            asst_len = kv_seq_len - prompt_len
            window = min(compact_window or asst_len, asst_len)

            # Extract suffix queries from captured buffer
            suffix_queries = None
            if use_suffix_queries and query_buffer:
                suffix_start_in_seg = prompt_len + window - seg_start
                suffix_queries = _extract_suffix_queries_from_buffer(
                    query_buffer, num_layers, suffix_start_in_seg,
                    num_kv_heads, head_size, device,
                )
                query_buffer.clear()

            # Convert inference indices to forced_indices tensors if available
            seg_forced_indices = None
            if compaction_indices is not None and seg_idx < len(compaction_indices):
                seg_ci = compaction_indices[seg_idx]
                if seg_ci is not None:
                    seg_forced_indices = [
                        torch.tensor(layer_indices, dtype=torch.int64, device=device)
                        for layer_indices in seg_ci
                    ]

            if compaction_mode == "markovian":
                kv_dim = (0, num_kv_heads, head_size)
                c1_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in range(num_layers)]
                c2_list = [torch.empty(kv_dim, dtype=values[0].dtype, device=device) for _ in range(num_layers)]
                beta_list = None
                compacted_prefix_len = 0
            else:
                compact_seed = prompt_len * 10000 + seg_idx
                c1_list, c2_list, beta_list, _ = compact_kv(
                    keys, values, prompt_len, compact_target_ratio,
                    num_kv_heads, head_size, device,
                    compact_window=window,
                    compute_beta=compute_beta,
                    seed=compact_seed,
                    suffix_queries=suffix_queries,
                    forced_indices=seg_forced_indices,
                )
                compacted_prefix_len = c1_list[0].shape[0]
            suffix_len = asst_len - window

            compacted_cache = DynamicCache()
            for l in range(num_layers):
                orig_K = keys[l]
                orig_V = values[l]
                suffix_K = orig_K[prompt_len + window:]
                suffix_V = orig_V[prompt_len + window:]

                new_K = torch.cat([orig_K[:prompt_len], c1_list[l], suffix_K], dim=0)
                new_V = torch.cat([orig_V[:prompt_len], c2_list[l], suffix_V], dim=0)

                new_K = new_K.permute(1, 0, 2).unsqueeze(0).detach()
                new_V = new_V.permute(1, 0, 2).unsqueeze(0).detach()

                compacted_cache.update(new_K, new_V, l)

            # Register beta hooks for the next segment's forward pass
            if compute_beta and beta_list is not None:
                new_kv_len = prompt_len + compacted_prefix_len + suffix_len
                if beta_state is None:
                    num_heads = model.config.num_attention_heads
                    beta_state = _BetaTrainingState(num_kv_heads, num_heads)
                    beta_hooks = _register_beta_hooks(model, beta_state)
                beta_state.set_beta(
                    beta_list, prompt_len, compacted_prefix_len, new_kv_len, device)

            tokens_removed = kv_seq_len - (prompt_len + compacted_prefix_len + suffix_len)
            position_offset += tokens_removed
            del keys, values, c1_list, c2_list, beta_list, kv_cache
            past_key_values = compacted_cache
            torch.cuda.empty_cache()

            logger.debug(
                "Compaction replay seg %d: kv_len %d -> %d (removed %d), "
                "window=%d, prefix_after=%d, suffix=%d",
                seg_idx, kv_seq_len,
                prompt_len + compacted_prefix_len + suffix_len,
                tokens_removed, window, compacted_prefix_len, suffix_len,
            )

    # Restore AC, LoRA offsets, use_cache, and beta/query hooks
    hook_handle.remove()
    for h in beta_hooks:
        h.remove()
    for h in query_capture_hooks:
        h.remove()
    if beta_state is not None:
        beta_state.clear()
    for name, module in backbone.named_modules():
        if name in saved_checkpoint_fns:
            module.checkpoint_fn = saved_checkpoint_fns[name]
    if saved_lora_num_tokens is not None:
        saved_lora_num_tokens[0] = original_lora_value
    model.config.use_cache = False

    torch.cuda.empty_cache()
    full_logits = torch.cat(all_logits_pieces, dim=1)
    del all_logits_pieces

    # Pad with dummy forward passes to keep FSDP ranks synchronized.
    # segmented_forward calls model.forward() once per segment; different
    # samples have different segment counts. Without padding, ranks diverge
    # on NCCL all-gather counts, causing deadlock.
    actual_passes = len(seg_input_ranges)
    target_passes = max_forward_passes or actual_passes
    if target_passes > actual_passes:
        dummy_sum = torch.tensor(0.0, device=device)
        for _ in range(target_passes - actual_passes):
            d_out = model(
                input_ids=input_ids[:, :1],
                position_ids=position_ids[:, :1],
            )
            d_logits = d_out["logits"] if isinstance(d_out, dict) else d_out.logits
            if isinstance(d_logits, dict):
                d_logits = d_logits["logits"]
            # Use float().mean() to prevent bf16 sum overflow → Inf.
            # Inf * 0 = NaN (IEEE 754), which would corrupt gradients.
            dummy_sum = dummy_sum + d_logits.float().mean()
        # Multiply by 0 to zero out gradient values while preserving the
        # autograd graph so FSDP backward hooks (reduce-scatter) still fire.
        # Cast to logits dtype to avoid float32 upcast allocating 2x memory.
        full_logits = full_logits + (dummy_sum * 0).to(full_logits.dtype)

    assert full_logits.shape[1] == input_ids.shape[1], (
        f"Segmented forward logits shape {full_logits.shape[1]} != input {input_ids.shape[1]}"
    )

    return {"logits": full_logits}
