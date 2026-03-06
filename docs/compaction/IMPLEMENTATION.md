# Compaction Implementation: Technical Details

## Overview

KV cache compaction compresses the key-value cache mid-generation to allow longer effective
context within a fixed memory budget. Instead of generating one long sequence, generation is
split into segments. After each segment, the oldest portion of the assistant's KV cache is
compressed using Attention Matching, and generation continues on the compacted cache.

The method is based on [Attention Matching](https://arxiv.org/abs/2602.16284) (Zweiger et al., 2026),
which selects a subset of keys (C1) and solves for replacement values (C2) that best preserve
the attention output under random queries.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ vLLM Server (per GPU)                               │
│                                                     │
│  FastAPI ──► /compact_generate  (routes.py)         │
│                    │                                │
│                    ▼                                │
│            collective_rpc("compact_generate")       │
│                    │                                │
│                    ▼                                │
│         CompactionWorker.compact_generate()         │
│                (worker.py)                          │
│                    │                                │
│         ┌─────────┴──────────┐                      │
│         ▼                    ▼                      │
│    Manual forward       KV compaction               │
│    passes (decode)      (_compact_kv)               │
│         │                    │                      │
│         ▼                    ▼                      │
│    FlashAttention       Paged block                 │
│    metadata             manipulation                │
│    (_DecodeContext)      (_inject_compacted_kv)      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Why manual forward passes?

vLLM's scheduler manages the KV cache and model execution. To inject compacted KV entries
mid-sequence, we need direct access to the KV cache blocks during generation. The
`collective_rpc` mechanism pauses the scheduler and gives the worker exclusive access to
the model and KV cache. The worker then drives generation manually: constructing
`FlashAttentionMetadata`, running model forward passes, and sampling tokens — all without
the scheduler's involvement.

## File Layout

```
src/prime_rl/inference/compaction/
├── __init__.py
├── routes.py      # FastAPI endpoint: /compact_generate
└── worker.py      # CompactionWorker + all generation/compaction logic
```

### `worker.py` — Core Logic

#### `CompactionWorker` (line 29)

Extends `FileSystemWeightUpdateWorker` (which itself extends vLLM's worker). The only
public method is `compact_generate()`, callable via `collective_rpc`.

#### `compact_generate()` (line 36)

Main entry point. Flow:

1. **Setup**: Get model, device, KV cache shape. Allocate free paged blocks.
2. **Prefill**: Run the prompt through the model in one forward pass (`_run_prefill`).
   This populates the KV cache for all prompt tokens.
3. **Decode loop**: For each segment (up to `n_compacts + 1` segments):
   - Generate tokens one at a time using CUDA-graphed decode steps.
   - Tokens are accumulated in pre-allocated GPU buffers (`seg_token_ids`, `seg_logprobs`)
     to avoid per-token CPU syncs.
   - EOS is checked every 64 tokens in batch on GPU.
   - After the segment, batch-transfer tokens to CPU.
4. **Compaction** (between segments): Extract KV → run Attention Matching → inject
   compacted KV back into paged blocks.
5. **Return**: All token IDs, logprobs, and diagnostics.

#### RoPE Position Tracking

After compaction removes tokens from the cache, the cache position and the RoPE position
diverge. The worker tracks `position_offset` — the cumulative number of tokens removed.
Each decode step computes:

```
cache_position = current_seq_len - 1        # where to write in KV cache
rope_position  = cache_position + position_offset  # position encoding for the model
```

This ensures the model sees monotonically increasing positions even though the cache is shorter.

#### `current_seq_len = new_seq_len + 1` (line 290)

After compaction, the last generated token's KV entry is stale (it attended to pre-compaction
context). Setting `current_seq_len = new_seq_len + 1` causes the next decode step to
overwrite that entry — the model recomputes it attending to the compacted context.

### `_DecodeContext` (line 369)

Pre-allocated tensors reused across all decode steps. Avoids creating new
`FlashAttentionMetadata` objects each step. Fields are updated in-place via
`_update_decode_state()`.

Key optimization: `attn_metadata_dict` and `slot_mapping_ctx` are pre-built dicts mapping
attention layer names to the shared metadata object. Since the dict values are mutable
tensors, updating the tensors updates all dict references simultaneously.

### CUDA Graph Capture (line 127)

With TP=1, a CUDA graph is captured after warm-up, and replayed each decode step. This
eliminates Python overhead and CPU-GPU synchronization, giving ~5-10x decode speedup.
With TP>1, NCCL all-reduce ops are incompatible with raw CUDA graphs, so decode falls
back to eager mode.

The graph is captured with `max_seq_len` set to the maximum possible length. The actual
sequence length is controlled by the `seq_lens` tensor (updated in-place each step), so
FlashAttention only reads the valid portion of the KV cache.

### `_run_prefill()` (line 438)

Runs the prompt through the model in a single forward pass. Constructs slot mappings
that map each position to its paged block location:

```
slot = block_ids[pos // block_size] * block_size + pos % block_size
```

### `_sample_token()` (line 513)

Temperature + top-p sampling with deterministic seeding via `rng.manual_seed(seed)`.
The seed is `len(all_token_ids) + seg_count`, ensuring reproducible results across
TP ranks (all ranks sample the same token).

## KV Cache Compaction Algorithm

### `_compact_kv()` (line 579)

Implements Attention Matching with β=0 (no regularization). For each layer and head:

**Step 1: Attention scoring with full context**

```python
Q = torch.randn(64, head_size)  # random query probes
full_scores = Q @ K_full.T * scale
full_attn = softmax(full_scores, dim=-1)
```

Random queries probe which keys receive the most attention. Using the full assistant KV
(not just the prefix being compressed) ensures the algorithm accounts for the suffix's
presence when deciding which prefix keys to keep.

**Step 2: Select top-k prefix keys**

```python
prefix_importance = full_attn[:, :window].pow(2).mean(dim=0).sqrt()
topk_indices = prefix_importance.topk(target_len).indices.sort().values
C1 = K_prefix[topk_indices]  # selected keys
```

The importance score is the RMS attention weight across all random queries. Only prefix
positions (first `compact_window` assistant tokens) are candidates for selection.

**Step 3: Solve for replacement values (C2)**

```python
# What the original prefix contributed to the output:
Y = full_attn[:, :window] @ V_prefix

# What the selected keys would produce with new values:
X = softmax(Q @ C1.T * scale, dim=-1)

# Least-squares: find C2 such that X @ C2 ≈ Y
C2 = lstsq(X, Y).solution
```

This is the key insight from Attention Matching: C1 preserves the original keys (so
attention patterns are similar), while C2 is a new set of values optimized so that
`softmax(Q @ C1.T) @ C2 ≈ softmax(Q @ K.T) @ V` — the attention output is preserved
even though fewer KV entries remain.

### Partial Compaction (`compact_window`)

When `compact_window` is set (e.g., 512), only the first 512 assistant tokens are
compressed. The remaining tokens (suffix) are preserved unchanged.

```
Before:  [prompt | ========= 2048 assistant tokens =========]
                   ↑ window=512 ↑     ↑ suffix=1536          ↑

After:   [prompt | C1/C2 (128) | ---- suffix (1536) --------|
```

The suffix is crucial context (most recent reasoning). By only compressing the oldest
tokens, the model retains its recent chain of thought while reclaiming memory from
earlier, less critical tokens.

**Full-context scoring**: Even though only the prefix is compressed, the attention
scoring in Step 1 uses the full KV cache (prefix + suffix). This means the algorithm
can identify prefix keys that are redundant because the suffix already captures that
information.

**Regression target correction**: The regression target Y is `full_attn[:, :window] @ V_prefix`
— only the prefix's contribution to the attention output. After injection, the suffix
handles its own contribution directly; C2 only needs to approximate what the prefix
originally contributed.

### `_inject_compacted_kv()` (line 657)

Writes the compacted KV back into vLLM's paged blocks:

```python
K_new = cat([K_prompt, C1, K_suffix])
V_new = cat([V_prompt, C2, V_suffix])
```

Then writes block-by-block into the paged KV cache. Stale blocks (beyond the new
sequence length) are zeroed.

## `routes.py` — HTTP Endpoint

Thin FastAPI layer. Receives the request, calls `collective_rpc("compact_generate", ...)`
on the engine, decodes the output tokens, and returns the result. The `collective_rpc`
call blocks the vLLM scheduler for the entire duration — no other requests can be
served concurrently on that server instance.

## Performance

With TP=1 and CUDA graphs, single-server throughput is ~110 tok/s. With DP=4 (4 servers),
aggregate throughput is ~400-450 tok/s. Each compaction event takes ~0.5-0.7s
(~0.4s algorithm + ~0.1-0.2s KV extraction/injection).

## Eval Script: `scripts/eval_rg_mix.py`

Evaluates on a weighted mix of reasoning-gym tasks:
- arc_1d, sokoban (hard), countdown (7 numbers), zebra_puzzles (7 people), cryptarithm
- Tasks weighted by inverse difficulty (harder tasks sampled more)

Supports two modes:
- **compaction**: DP=4 parallel requests to ports 8000-8003 via `/compact_generate`
- **baseline**: Sequential requests to a single server via `/v1/chat/completions`

Reports overall accuracy, per-task accuracy, accuracy by number of compactions performed,
compaction statistics, and mean logprob degradation.
