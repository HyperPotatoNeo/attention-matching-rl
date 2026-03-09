"""Beta-corrected attention for decode steps after KV cache compaction.

After compaction, the compacted keys have an associated per-token bias (beta)
that corrects the partition function mismatch. This module provides:

1. BetaState: manages contiguous KV mirrors + beta buffers alongside paged cache
2. monkey-patch helpers to replace FlashAttention with SDPA+beta for decode
3. Manual GQA-aware attention with additive bias

The contiguous KV buffers are maintained in parallel with vLLM's paged cache:
- New tokens: appended to both paged cache (by vLLM) and contiguous buffer (by us)
- Compaction: contiguous buffer is rebuilt from compact_kv output
- Beta: stored per-layer, zero for non-compacted positions
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BetaState:
    """Manages contiguous KV mirrors and beta buffers for all sequences.

    These buffers live alongside vLLM's paged KV cache. They enable SDPA-based
    attention with per-token bias that FlashAttention doesn't support.
    """

    def __init__(
        self,
        B: int,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        num_heads: int,
        head_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.B = B
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.heads_per_group = num_heads // num_kv_heads
        self.head_size = head_size
        self.scale = 1.0 / (head_size ** 0.5)

        # Per-layer contiguous KV: (B, max_seq_len, num_kv_heads, head_size)
        self.K = [
            torch.zeros(B, max_seq_len, num_kv_heads, head_size,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.V = [
            torch.zeros(B, max_seq_len, num_kv_heads, head_size,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        # Per-layer beta: (B, num_kv_heads, max_seq_len) — zero = no bias
        self.beta = [
            torch.zeros(B, num_kv_heads, max_seq_len,
                        device=device, dtype=torch.float32)
            for _ in range(num_layers)
        ]

        # Per-sequence valid lengths (shared across layers)
        self.seq_lens = torch.zeros(B, dtype=torch.long, device=device)

        # Validity mask: (B, 1, 1, max_seq_len) — pre-allocated, updated per step
        self.valid_mask = torch.zeros(B, 1, 1, max_seq_len,
                                      device=device, dtype=torch.bool)

        self.active = False
        self.device = device
        self.dtype = dtype

    def init_from_prefill(
        self,
        kv_caches: list[torch.Tensor],
        per_seq_blocks: list[list[int]],
        prompt_lens: list[int],
        block_size: int,
    ):
        """Copy prefilled KV from paged cache into contiguous buffers."""
        for layer_idx in range(self.num_layers):
            kv = kv_caches[layer_idx]
            for seq_idx in range(self.B):
                plen = prompt_lens[seq_idx]
                n_blocks = (plen + block_size - 1) // block_size
                bids = per_seq_blocks[seq_idx][:n_blocks]
                # Gather from paged blocks: (n_blocks * block_size, H, D)
                k_gathered = kv[0][bids].reshape(-1, self.num_kv_heads, self.head_size)[:plen]
                v_gathered = kv[1][bids].reshape(-1, self.num_kv_heads, self.head_size)[:plen]
                self.K[layer_idx][seq_idx, :plen] = k_gathered
                self.V[layer_idx][seq_idx, :plen] = v_gathered

        # Set seq_lens and masks
        for seq_idx, plen in enumerate(prompt_lens):
            self.seq_lens[seq_idx] = plen
        self._update_valid_mask()

    def append_token_kv(
        self,
        layer_idx: int,
        seq_idx: int,
        pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Append one token's K, V at position pos for a given layer and sequence."""
        self.K[layer_idx][seq_idx, pos] = k
        self.V[layer_idx][seq_idx, pos] = v

    def set_compacted(
        self,
        layer_idx: int,
        seq_idx: int,
        prompt_len: int,
        c1: torch.Tensor,
        c2: torch.Tensor,
        beta: torch.Tensor,
        suffix_K: torch.Tensor,
        suffix_V: torch.Tensor,
    ):
        """Update contiguous buffer after compaction for one sequence, one layer.

        Args:
            c1: (compacted_len, num_kv_heads, head_size) — compacted keys
            c2: (compacted_len, num_kv_heads, head_size) — compacted values
            beta: (compacted_len, num_kv_heads) — per-key bias
            suffix_K/V: (suffix_len, num_kv_heads, head_size) — preserved suffix
        """
        compacted_len = c1.shape[0]
        suffix_len = suffix_K.shape[0]
        total_new = prompt_len + compacted_len + suffix_len

        # Write compacted region
        start = prompt_len
        end = start + compacted_len
        self.K[layer_idx][seq_idx, start:end] = c1
        self.V[layer_idx][seq_idx, start:end] = c2

        # Write suffix
        if suffix_len > 0:
            suf_end = end + suffix_len
            self.K[layer_idx][seq_idx, end:suf_end] = suffix_K
            self.V[layer_idx][seq_idx, end:suf_end] = suffix_V

        # Write beta: zeros everywhere, then set compacted positions
        self.beta[layer_idx][seq_idx, :] = 0.0
        # beta: (compacted_len, num_kv_heads) → transpose to (num_kv_heads, compacted_len)
        self.beta[layer_idx][seq_idx, :, start:end] = beta.T

    def update_seq_lens(self, new_seq_lens: torch.Tensor):
        """Update valid lengths and mask."""
        self.seq_lens[:] = new_seq_lens
        self._update_valid_mask()

    def _update_valid_mask(self):
        """Rebuild validity mask from seq_lens."""
        arange = torch.arange(self.max_seq_len, device=self.device)
        # (B, 1, 1, max_seq_len): True where position < seq_len
        self.valid_mask[:] = (arange.unsqueeze(0) < self.seq_lens.unsqueeze(1)).unsqueeze(1).unsqueeze(1)

    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GQA attention with beta bias for decode (1 query per sequence).

        Args:
            layer_idx: which transformer layer
            query: (B, num_heads * head_size) — query from QKV projection

        Returns:
            output: (B, num_heads * head_size) — attention output
        """
        B = query.shape[0]
        Q = query.view(B, self.num_heads, self.head_size)

        K = self.K[layer_idx][:B]  # (B, max_seq_len, H_kv, D)
        V = self.V[layer_idx][:B]  # same
        beta = self.beta[layer_idx][:B]  # (B, H_kv, max_seq_len)

        # GQA: group query heads
        # Q: (B, H_kv, G, D) where G = heads_per_group
        Q_grouped = Q.view(B, self.num_kv_heads, self.heads_per_group, self.head_size)

        # K: (B, H_kv, S, D) — transpose from (B, S, H_kv, D)
        K_t = K.permute(0, 2, 1, 3)  # (B, H_kv, max_seq_len, D)
        V_t = V.permute(0, 2, 1, 3)  # (B, H_kv, max_seq_len, D)

        # Attention scores: Q @ K^T
        # (B, H_kv, G, D) @ (B, H_kv, D, S) → (B, H_kv, G, S)
        scores = torch.einsum('bhgd,bhsd->bhgs', Q_grouped.float(), K_t.float()) * self.scale

        # Add beta bias: (B, H_kv, 1, S) — broadcast across query groups
        scores = scores + beta.unsqueeze(2)

        # Mask invalid positions
        # valid_mask: (B, 1, 1, max_seq_len) → broadcasts to (B, H_kv, G, S)
        scores = scores.masked_fill(~self.valid_mask[:B], float('-inf'))

        # Softmax + weighted sum
        attn_weights = torch.softmax(scores, dim=-1)  # (B, H_kv, G, S)

        # Output: (B, H_kv, G, S) @ (B, H_kv, S, D) → (B, H_kv, G, D)
        out = torch.einsum('bhgs,bhsd->bhgd', attn_weights.to(V_t.dtype), V_t)

        # Reshape back to (B, num_heads * head_size)
        return out.reshape(B, self.num_heads * self.head_size)


def patch_attention_layers(model, beta_state: BetaState) -> list:
    """Monkey-patch attention layers to use SDPA+beta when beta_state.active.

    Replaces each Attention layer's impl with a wrapper that:
    - When beta inactive: delegates to original FlashAttention
    - When beta active: writes K,V to paged cache, then computes SDPA+beta

    Returns list of (layer_name, original_impl) for cleanup.
    """
    from vllm.model_executor.layers.attention.attention import Attention

    originals = []
    layer_idx = 0

    for name, module in model.named_modules():
        if not isinstance(module, Attention):
            continue

        original_impl = module.impl
        originals.append((name, original_impl))

        wrapper = _BetaAttentionWrapper(original_impl, layer_idx, beta_state)
        module.impl = wrapper
        layer_idx += 1

    logger.info("Patched %d attention layers for beta support", len(originals))
    return originals


def unpatch_attention_layers(model, originals: list):
    """Restore original attention implementations."""
    from vllm.model_executor.layers.attention.attention import Attention

    idx = 0
    for name, module in model.named_modules():
        if not isinstance(module, Attention):
            continue
        if idx < len(originals):
            _, original_impl = originals[idx]
            module.impl = original_impl
            idx += 1


class _BetaAttentionWrapper:
    """Wraps FlashAttentionImpl to optionally use SDPA+beta for decode."""

    def __init__(self, original_impl, layer_idx: int, beta_state: BetaState):
        self.original_impl = original_impl
        self.layer_idx = layer_idx
        self.beta_state = beta_state
        # Forward all attributes from original impl (needed by vLLM internals)
        for attr in dir(original_impl):
            if attr.startswith('_'):
                continue
            if attr in ('forward', 'do_kv_cache_update'):
                continue
            try:
                setattr(self, attr, getattr(original_impl, attr))
            except (AttributeError, TypeError):
                pass

    @property
    def forward_includes_kv_cache_update(self):
        return self.original_impl.forward_includes_kv_cache_update

    def do_kv_cache_update(self, *args, **kwargs):
        return self.original_impl.do_kv_cache_update(*args, **kwargs)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, **kwargs):
        if not self.beta_state.active:
            return self.original_impl.forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output=output, **kwargs)

        # Beta-active path: write K,V to paged cache, then use SDPA+beta

        # 1. Write new K,V to paged cache (so it stays consistent)
        if not self.original_impl.forward_includes_kv_cache_update:
            # KV update was already done by the caller (Attention.forward)
            pass
        else:
            # Some backends include KV update in forward — do it manually
            self.original_impl.do_kv_cache_update(
                layer, key, value, kv_cache, attn_metadata.slot_mapping)

        # 2. Write new K,V to contiguous buffer
        # key/value: (B, num_kv_heads * head_size) for decode
        B = attn_metadata.num_actual_tokens
        seq_lens = attn_metadata.seq_lens[:B].long()
        # Position = seq_len - 1 (the just-written position)
        positions = seq_lens - 1

        k_reshaped = key[:B].view(B, self.beta_state.num_kv_heads, self.beta_state.head_size)
        v_reshaped = value[:B].view(B, self.beta_state.num_kv_heads, self.beta_state.head_size)

        for i in range(B):
            pos = positions[i].item()
            if pos < self.beta_state.max_seq_len:
                self.beta_state.K[self.layer_idx][i, pos] = k_reshaped[i]
                self.beta_state.V[self.layer_idx][i, pos] = v_reshaped[i]

        # 3. Update valid mask (only needed once per step, not per layer,
        #    but it's idempotent so calling it per layer is fine)
        self.beta_state.update_seq_lens(seq_lens)

        # 4. Compute attention with beta
        result = self.beta_state.compute_attention(self.layer_idx, query[:B])

        # Write to output buffer if provided
        if output is not None:
            output[:B] = result.to(output.dtype)
            return output
        return result.to(query.dtype)
