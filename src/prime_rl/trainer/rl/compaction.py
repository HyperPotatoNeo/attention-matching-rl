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
"""

import logging

import torch
from torch import Tensor
from transformers import DynamicCache

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from prime_rl.inference.compaction.algorithm import compact_kv
from prime_rl.trainer.models.layers.lora import base as lora_base

logger = logging.getLogger(__name__)


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

            c1_list, c2_list, _ = compact_kv(
                keys, values, prompt_len, compact_target_ratio,
                num_kv_heads, head_size, device,
                compact_window=window,
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

            tokens_removed = kv_seq_len - (prompt_len + compacted_prefix_len + suffix_len)
            position_offset += tokens_removed
            del keys, values, c1_list, c2_list, kv_cache
            past_key_values = compacted_cache
            torch.cuda.empty_cache()

            logger.debug(
                "Compaction replay seg %d: kv_len %d -> %d (removed %d), "
                "window=%d, prefix_after=%d, suffix=%d",
                seg_idx, kv_seq_len,
                prompt_len + compacted_prefix_len + suffix_len,
                tokens_removed, window, compacted_prefix_len, suffix_len,
            )

    # Restore AC, LoRA offsets, and use_cache
    hook_handle.remove()
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
