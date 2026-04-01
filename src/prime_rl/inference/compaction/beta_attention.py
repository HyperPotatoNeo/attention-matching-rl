"""Beta-corrected attention for decode steps after KV cache compaction.

After compaction, compacted keys have an associated per-token bias (beta)
that corrects the partition function mismatch. This module provides:

1. BetaState: contiguous KV mirrors + beta buffers alongside paged cache
2. monkey-patch helpers to replace FlashAttention with SDPA+beta for decode
3. GQA-aware manual attention with additive bias

All operations use fixed-shape tensors with in-place updates for CUDA graph
compatibility. No .item() calls, no Python conditionals on tensor values.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class BetaState:
    """Manages contiguous KV mirrors and beta buffers for all sequences.

    These buffers live alongside vLLM's paged KV cache. They enable SDPA-based
    attention with per-token bias that FlashAttention doesn't support.

    Memory: ~2.6 GB for B=8, S=2048, H_kv=8, D=128, L=36 (Qwen3-4B).
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

        # Validity mask: (B, 1, 1, max_seq_len)
        self.valid_mask = torch.zeros(B, 1, 1, max_seq_len,
                                      device=device, dtype=torch.bool)

        # Pre-allocated index tensors for vectorized scatter
        self._arange_seq = torch.arange(max_seq_len, device=device)
        self._arange_B = torch.arange(B, device=device)

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
                k_gathered = kv[0][bids].reshape(-1, self.num_kv_heads, self.head_size)[:plen]
                v_gathered = kv[1][bids].reshape(-1, self.num_kv_heads, self.head_size)[:plen]
                self.K[layer_idx][seq_idx, :plen] = k_gathered
                self.V[layer_idx][seq_idx, :plen] = v_gathered

        for seq_idx, plen in enumerate(prompt_lens):
            self.seq_lens[seq_idx] = plen
        self.update_valid_mask()

    def append_token_kv_batch(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Append one token's K,V at given positions for all sequences.

        CUDA-graph-compatible: vectorized scatter, no .item() calls.

        Args:
            positions: (B,) long tensor — write position per sequence
            k: (B, num_kv_heads, head_size)
            v: (B, num_kv_heads, head_size)
        """
        safe_pos = positions.clamp(0, self.max_seq_len - 1)
        self.K[layer_idx][self._arange_B, safe_pos] = k
        self.V[layer_idx][self._arange_B, safe_pos] = v

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
        zero_beta: bool = True,
    ):
        """Update contiguous buffer after compaction for one sequence, one layer.

        Called outside CUDA graph (during compaction, which is eager anyway).

        Args:
            c1: (compacted_len, num_kv_heads, head_size)
            c2: (compacted_len, num_kv_heads, head_size)
            beta: (compacted_len, num_kv_heads)
            suffix_K/V: (suffix_len, num_kv_heads, head_size)
        """
        compacted_len = c1.shape[0]
        suffix_len = suffix_K.shape[0]

        start = prompt_len
        end = start + compacted_len
        self.K[layer_idx][seq_idx, start:end] = c1
        self.V[layer_idx][seq_idx, start:end] = c2

        if suffix_len > 0:
            suf_end = end + suffix_len
            self.K[layer_idx][seq_idx, end:suf_end] = suffix_K
            self.V[layer_idx][seq_idx, end:suf_end] = suffix_V

        if zero_beta:
            self.beta[layer_idx][seq_idx, :] = 0.0
        # beta: (compacted_len, num_kv_heads) → (num_kv_heads, compacted_len)
        self.beta[layer_idx][seq_idx, :, start:end] = beta.T

    def update_valid_mask(self):
        """Rebuild validity mask from seq_lens. CUDA-graph-compatible."""
        self.valid_mask[:] = (
            self._arange_seq.unsqueeze(0) < self.seq_lens.unsqueeze(1)
        ).unsqueeze(1).unsqueeze(1)

    def compute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """GQA attention with beta bias for decode (1 query per sequence).

        CUDA-graph-compatible: all fixed-shape tensor operations.

        Args:
            query: (B, num_heads * head_size)
        Returns:
            (B, num_heads * head_size)
        """
        B = self.B
        Q = query.view(B, self.num_heads, self.head_size)
        K = self.K[layer_idx]
        V = self.V[layer_idx]
        beta = self.beta[layer_idx]

        # GQA grouping: (B, H_kv, G, D)
        Q_grouped = Q.view(B, self.num_kv_heads, self.heads_per_group, self.head_size)

        # Transpose to (B, H_kv, S, D) for batched matmul
        K_t = K.permute(0, 2, 1, 3)
        V_t = V.permute(0, 2, 1, 3)

        # QK^T in native dtype (bf16 tensor cores), upcast for softmax
        scores = torch.einsum('bhgd,bhsd->bhgs', Q_grouped, K_t) * self.scale
        scores = scores.float() + beta.clamp(-30.0, 30.0).unsqueeze(2)

        # Mask invalid positions
        scores = scores.masked_fill(~self.valid_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)

        out = torch.einsum('bhgs,bhsd->bhgd', attn_weights.to(V_t.dtype), V_t)

        return out.reshape(B, self.num_heads * self.head_size)


def patch_attention_layers(model, beta_state: BetaState) -> list:
    """Monkey-patch attention layers to use SDPA+beta when beta_state.active.

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
    """Wraps FlashAttentionImpl to use SDPA+beta when active.

    CUDA-graph-compatible: vectorized tensor ops, no .item() calls.
    The active/inactive switch requires separate CUDA graph captures.
    """

    def __init__(self, original_impl, layer_idx: int, beta_state: BetaState):
        self.original_impl = original_impl
        self.layer_idx = layer_idx
        self.beta_state = beta_state
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
        return getattr(self.original_impl, 'forward_includes_kv_cache_update', True)

    def do_kv_cache_update(self, *args, **kwargs):
        if hasattr(self.original_impl, 'do_kv_cache_update'):
            return self.original_impl.do_kv_cache_update(*args, **kwargs)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, **kwargs):
        if not self.beta_state.active:
            return self.original_impl.forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output=output, **kwargs)

        # Beta-active path:
        # 1. Run original forward to keep paged KV cache in sync
        #    (FlashAttentionImpl.forward handles both cache update + attention)
        self.original_impl.forward(
            layer, query, key, value, kv_cache, attn_metadata,
            output=output, **kwargs)

        # 2. Update contiguous buffer for beta attention
        B = self.beta_state.B
        positions = self.beta_state.seq_lens - 1
        k_reshaped = key[:B].view(B, self.beta_state.num_kv_heads, self.beta_state.head_size)
        v_reshaped = value[:B].view(B, self.beta_state.num_kv_heads, self.beta_state.head_size)
        self.beta_state.append_token_kv_batch(self.layer_idx, positions, k_reshaped, v_reshaped)

        # 3. Override with beta-corrected attention
        result = self.beta_state.compute_attention(self.layer_idx, query[:B])

        if output is not None:
            output[:B] = result.to(output.dtype).view_as(output[:B])
            return output
        return result.to(query.dtype).view_as(query[:B])
