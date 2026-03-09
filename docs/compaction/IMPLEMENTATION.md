# Compaction Implementation: Technical Details

## Overview

KV cache compaction compresses the key-value cache mid-generation to allow longer effective
context within a fixed memory budget. Generation is split into segments. After each segment,
the oldest portion of the assistant's KV cache is compressed using Attention Matching, and
generation continues on the compacted cache.

Based on [Attention Matching](https://arxiv.org/abs/2602.16284) (Zweiger et al., 2026):
select a subset of keys (C1), solve for replacement values (C2) that preserve attention output,
and optionally compute beta bias to correct the partition function mismatch.

## File Layout

```
src/prime_rl/inference/compaction/
  algorithm.py      # Attention Matching + NNLS beta solver (pure tensors, no vLLM dep)
  beta_attention.py  # BetaState contiguous mirrors + SDPA decode with per-token bias
  routes.py          # /compact_generate endpoint + _RequestBatcher (auto-batching B=8)
  worker.py          # CompactionWorker: generation, compaction, CUDA graphs
```

## Compaction Algorithm (`algorithm.py`)

### `compact_kv()`

Per-layer vectorized Attention Matching across all KV heads simultaneously:

1. **Attention scoring**: Random queries `Q` (H, 64, D) probe the full assistant KV.
   Importance per prefix position = RMS attention weight across queries.

2. **Top-k selection**: Keep `target_ratio` fraction of prefix keys per head.
   Each head selects independently (different keys per head).

3. **C2 via least-squares**: Solve `softmax(Q @ C1^T [+ beta]) @ C2 = prefix_attn @ V_prefix`
   so the compacted KV produces the same attention output as the original prefix.

4. **NaN/extreme value handling**: `lstsq` can produce NaN from rank-deficient softmax
   (peaked betas). NaN entries fall back to original values. C2 is clamped to 2x the
   original value range.

### Partial Compaction (`compact_window`)

Only the first `compact_window` assistant tokens are compressed. The suffix is preserved:

```
Before:  [prompt | ========= 2048 assistant tokens =========]
                   ^ window=512 ^     ^ suffix=1536          ^

After:   [prompt | C1/C2 (128) | ---- suffix (1536) --------|
```

Full-context scoring: even though only the prefix is compressed, attention scoring uses
the full KV (prefix + suffix) so the algorithm accounts for redundancy with the suffix.

### NNLS Beta Solver (`_solve_beta_nnls()`)

When `compute_beta=True`, finds per-key additive bias that corrects the partition function
mismatch between full and compacted attention:

- **Target**: `Z_full(q) = sum_k exp(q @ k / sqrt(d))` -- full partition function per query
- **Design matrix**: `M_{i,j} = exp(q_i @ c1_j / sqrt(d))` -- compacted terms
- **NNLS**: `min ||M @ B - target||^2, B >= 0` via projected gradient descent (50 iters)
- **Output**: `beta = log(B)` -- additive bias in log-space

Beta is computed over the prefix keys only (not suffix), since suffix keys remain in
attention without correction.

## Beta Attention (`beta_attention.py`)

### `BetaState`

Contiguous KV mirrors + beta buffers that live alongside vLLM's paged KV cache:

- `K[layer]`, `V[layer]`: `(B, max_seq_len, H_kv, D)` -- full contiguous copies
- `beta[layer]`: `(B, H_kv, max_seq_len)` -- per-token additive bias (zero = no correction)
- Memory: ~2.6 GB for B=8, S=2048, H_kv=8, D=128, L=36 (Qwen3-4B)

All operations use fixed-shape tensors with in-place updates for CUDA graph compatibility.

### Monkey-patched attention

`patch_attention_layers()` wraps each FlashAttentionImpl with `_BetaAttentionWrapper`:
- **Inactive** (`beta_state.active = False`): pass through to original FlashAttention
- **Active** (`beta_state.active = True`): run original forward (keeps paged cache in sync),
  then override output with GQA manual attention + beta bias

This enables two-phase CUDA graph capture:
1. Pre-compaction graph: FlashAttention (beta inactive)
2. Post-compaction graph: SDPA+beta (beta active)

### `compute_attention()`

GQA-aware manual attention with per-token additive bias:
```
scores = einsum(Q_grouped, K^T) * scale + beta
scores = masked_fill(~valid_mask, -inf)
attn = softmax(scores)
out = einsum(attn, V)
```
Operates in float32 for numerical stability, casts output to model dtype.

## Worker (`worker.py`)

### `CompactionWorker`

Extends `FileSystemWeightUpdateWorker`. Two main methods:

- `compact_generate()`: Single sequence generation with compaction
- `compact_generate_batch()`: B sequences in parallel (used by auto-batching)

### Generation flow

1. **Prefill**: Single forward pass populates KV cache for all prompt tokens
2. **Decode loop**: Per-segment token generation with CUDA graphs (TP=1)
   - Tokens accumulated in GPU buffers, EOS checked every 64 tokens
   - Batch-transfer to CPU after each segment
3. **Compaction**: Extract KV -> `compact_kv()` -> inject compacted KV
   - If `compute_beta`: also computes NNLS beta, updates `BetaState`, activates beta attention
4. **Two-phase CUDA graphs**: Pre-compaction capture uses FlashAttention; after first
   compaction with beta, recapture with SDPA+beta active

### RoPE position tracking

After compaction removes tokens, cache position and RoPE position diverge:
```
cache_position = current_seq_len - 1
rope_position  = cache_position + position_offset
```

### `current_seq_len = new_seq_len + 1`

After compaction, the boundary token's KV is stale (attended to pre-compaction context).
Setting `+1` forces the next decode to recompute it against compacted context.

## Auto-Batching (`routes.py`)

`_RequestBatcher` accumulates individual `/compact_generate` requests (up to B=8, wait
up to 1.0s) and dispatches them as a single `compact_generate_batch` collective_rpc call.

Mandatory cleanup after each batch: `torch.cuda.synchronize() + gc.collect() + empty_cache()`
to prevent CUDA memory fragmentation.

## FSDP2 Training

`segmented_forward` in the trainer calls `model.forward()` once per compaction segment.
Under FSDP, each forward triggers all-gather. Variable segment counts across ranks cause
NCCL deadlock.

Fix: all-reduce MAX segment count, pad with dummy forwards (`result * 0` to preserve
autograd graph for FSDP backward hooks). `empty_cache()` between segments prevents OOM
from CUDA memory fragmentation.

## Performance (Qwen3-4B, A100-80GB)

| Metric | Value |
|--------|-------|
| Decode speed (CUDA graph, TP=1) | ~130 tok/s per GPU |
| Aggregate throughput (DP=4) | ~480 tok/s |
| Compaction algo time | ~0.05s/event (vectorized across heads) |
| Compaction % of total time | ~1% |
