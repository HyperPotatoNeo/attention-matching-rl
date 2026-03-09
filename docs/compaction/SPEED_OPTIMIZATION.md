# Compaction Inference Speed Optimization

## Current Performance (2026-03-07, Qwen3-4B, A100-80GB)

| Metric | Value |
|--------|-------|
| Decode speed (CUDA graph, TP=1) | ~130 tok/s per GPU |
| Aggregate throughput (DP=4) | ~480 tok/s |
| Per-request latency (8192 tokens) | ~73s |
| Compaction algo time | 0.36s/event → **0.05s/event** (vectorized) |
| Compaction inject time | 0.085s/event → ~0.02s/event (vectorized) |
| Compaction % of total time | 4.4% → ~1% (after vectorization) |

### Timing Breakdown (per request, 9 compactions, 8192 tokens)

| Component | Time | % of total |
|-----------|------|-----------|
| Decode (CUDA graph) | ~69s | 95.6% |
| Compaction algo | ~3.2s | 4.4% → ~0.5s after vectorization |
| KV inject | ~0.8s | 1% → ~0.2s after vectorization |
| KV extract | ~0.07s | 0.1% |
| HTTP/tokenization | ~0.5s | 0.7% |

## Completed Optimizations

### Phase 1: Algorithm Vectorization (2026-03-07)

Vectorized `compact_kv` across KV heads within each layer:
- **Before**: Python loop over 8 heads per layer (288 iterations total)
  - Per-head: randn, bmm, softmax, topk, lstsq
- **After**: Single batched operation per layer (36 iterations total)
  - Per-layer: randn(H,Q,D), bmm, softmax, batched topk, batched lstsq
- Expected speedup: ~5-8x on algo_time (0.36s → ~0.05s per event)

### Phase 2: KV Inject Vectorization (2026-03-07)

Replaced per-block Python loop in `_inject_compacted_kv` with padded
reshape + batched scatter:
- **Before**: Python loop over n_blocks per layer, with conditional logic
- **After**: Pad to block-aligned length, reshape, single scatter per layer
- Expected speedup: ~3-5x on inject_time (0.085s → ~0.02s per event)

## Planned Optimizations

### Phase 3: Batched Generation (Major — estimated 3-5x overall speedup)

**The bottleneck**: Each `collective_rpc` processes exactly one sequence.
The GPU does one token per forward pass. With standard vLLM continuous
batching, many sequences share each forward pass.

**Proposal**: Modify `compact_generate` to accept a batch of prompts and
generate them simultaneously:

1. Prefill all prompts (sequentially or with padding)
2. Decode loop processes B tokens per step (one per active sequence)
3. Each sequence tracks its own: current_seq_len, position_offset, blocks,
   compaction state
4. When a sequence hits compaction trigger, pause it, compact its KV, resume
5. Sequences that hit EOS are marked done

**Complexity**: Medium-high. Requires:
- Per-sequence state tracking (block allocations, offsets, compaction counts)
- FlashAttentionMetadata for B sequences simultaneously
- Variable-length handling (sequences at different positions)
- CUDA graph re-capture for batch decode (fixed batch size)

**Expected impact**: With batch_size=8 per GPU, decode throughput goes from
~130 tok/s to ~600-800 tok/s per GPU. For training (2048 rollouts), this
reduces inference time by 4-6x.

**Risk**: High complexity, potential for subtle bugs in multi-sequence KV
management. Should be developed incrementally with extensive testing.

### Phase 4: Async Pipeline (Low effort, moderate impact)

Queue incoming requests on the server side instead of processing one at a
time. The `collective_rpc` still processes sequentially, but HTTP overhead
is hidden.

### Phase 5: FP8 Quantization (Low effort, ~1.5x decode speedup)

Use vLLM's FP8 quantization for KV cache. Reduces memory bandwidth during
attention, directly speeds up decode.

## Benchmark Script

`scripts/bench_compaction.py` — test both algorithm-only and end-to-end:

```bash
# Algorithm only (no server needed, tests on CPU/GPU):
python scripts/bench_compaction.py --algo-only --n-trials 5

# End-to-end (needs 4 servers running):
python scripts/bench_compaction.py --n 30
```
