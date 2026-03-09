# Compaction Inference Speed Optimization

## Current Performance (2026-03-07, Qwen3-4B, A100-80GB)

| Metric | Value |
|--------|-------|
| Decode speed (CUDA graph, TP=1) | ~130 tok/s per GPU |
| Aggregate throughput (DP=4, auto-batch B=8) | ~480 tok/s |
| Per-request latency (8192 tokens) | ~73s |
| Compaction algo time | ~0.05s/event (vectorized) |
| Compaction inject time | ~0.02s/event (vectorized) |
| Compaction % of total time | ~1% |

### Timing Breakdown (per request, 9 compactions, 8192 tokens)

| Component | Time | % of total |
|-----------|------|-----------|
| Decode (CUDA graph) | ~69s | 95.6% |
| Compaction algo | ~0.5s | ~0.7% |
| KV inject | ~0.2s | ~0.3% |
| KV extract | ~0.07s | 0.1% |
| HTTP/tokenization | ~0.5s | 0.7% |

## Completed Optimizations

### Algorithm Vectorization

Vectorized `compact_kv` across KV heads within each layer:
- **Before**: Python loop over 8 heads per layer (288 iterations)
- **After**: Single batched operation per layer (36 iterations)
- Result: ~7x speedup on algo_time (0.36s -> ~0.05s per event)

### KV Inject Vectorization

Replaced per-block Python loop with padded reshape + batched scatter:
- **Before**: Python loop over n_blocks per layer
- **After**: Pad to block-aligned length, reshape, single scatter per layer
- Result: ~4x speedup on inject_time (0.085s -> ~0.02s per event)

### Auto-Batching (B=8)

`_RequestBatcher` in routes.py transparently batches individual `/compact_generate`
requests into `compact_generate_batch` calls. Decode processes B tokens per forward pass.
Training step time: ~64 min -> ~23 min (~3x speedup).

## Possible Future Optimizations

### FP8 KV Cache Quantization

Use vLLM's FP8 quantization for KV cache. Reduces memory bandwidth during attention,
directly speeds up decode. Low effort, ~1.5x expected.

## Benchmark Script

```bash
# Algorithm only (no server needed):
python scripts/bench_compaction.py --algo-only --n-trials 5

# End-to-end (needs 4 servers running):
python scripts/bench_compaction.py --n 30
```
