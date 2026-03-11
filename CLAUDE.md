@AGENTS.md

## Compaction Module

KV cache compaction lives in `src/prime_rl/inference/compaction/`. See
`docs/compaction/README.md` for usage and `docs/compaction/IMPLEMENTATION.md` for
the full technical walkthrough.

### Key files

| File | Purpose |
|------|---------|
| `src/prime_rl/inference/compaction/worker.py` | Generation + compaction logic (single & batch) |
| `src/prime_rl/inference/compaction/routes.py` | `/compact_generate` endpoint + auto-batching |
| `src/prime_rl/inference/compaction/algorithm.py` | Attention Matching + NNLS beta solver (deterministic, prompt keys, suffix queries, forced indices) |
| `src/prime_rl/inference/compaction/beta_attention.py` | BetaState mirrors + SDPA decode with per-token bias |
| `src/prime_rl/trainer/rl/compaction.py` | Beta training hooks + deterministic compaction replay + query capture hooks + forced indices |
| `src/compaction_env/env.py` | CompactionEnv (verifiers SingleTurnEnv wrapper) |
| `scripts/eval_rg_mix.py` | rg-mix-env evaluation (compaction and baseline modes) |

### Configs

| Config | Purpose |
|--------|---------|
| `configs/compaction/qwen3_4b_fullft_determ_nobeta.toml` | **Default** — 4+4 layout, deterministic random queries, no beta |
| `configs/compaction/qwen3_4b_fullft_suffix_queries.toml` | Suffix queries — real query vectors instead of random probes |
| `configs/compaction/qwen3_4b_fullft_determ_suffix.toml` | Deterministic random queries + prompt keys |
| `configs/compaction/qwen3_4b_fullft_fixed_1024q.toml` | 1024 random queries (matches new default num_queries) |
| `configs/compaction/qwen3_4b_fullft_nobeta.toml` | 4+4 layout, no beta (pre-deterministic, legacy) |
| `configs/compaction/qwen3_4b_beta_test.toml` | Beta attention test config |
| `configs/compaction/qwen3_4b_fullft_baseline.toml` | Baseline (no compaction) |
| `configs/compaction/qwen3_4b_serve_tp1.toml` | TP=1 inference server |

### How it works

The `CompactionWorker` drives model forward passes manually inside a `collective_rpc` call,
bypassing vLLM's scheduler. Generation happens in segments. Between segments, the KV cache
is compacted using Attention Matching: select top-k keys by attention importance (C1),
solve least-squares for replacement values (C2), optionally compute NNLS beta bias for
partition function correction, then inject `[prompt | C1/C2 | suffix]` back into paged blocks.

**Full-context scoring (default)**: `compact_kv()` scores key importance using the full KV
cache (prompt + assistant keys in the softmax denominator). Window keys redundant with prompt
content score lower. Default `num_queries=1024` (seeded random Gaussian probes).

**Deterministic compaction**: `compact_kv()` uses seeded random queries (`seed + layer_idx`)
to ensure identical compaction between inference and training replay. The seed is derived from
`prompt_len * 10000 + segment_idx`. This reduces Mismatch KL by ~6x (0.0077 → 0.0013).

**Compaction indices passing**: Inference returns per-event top-k indices in
`diagnostics.compaction_indices`. The trainer passes these as `forced_indices` to
`compact_kv()`, skipping importance scoring. C2 values are recomputed from the trainer's
KV cache for correct gradients. Essential for suffix queries (vLLM vs HF numerical diff).

**Beta attention**: When `compute_beta=true`, the NNLS solver finds per-key additive biases
that correct the partition function mismatch between full and compacted attention.
`BetaState` maintains contiguous KV mirrors alongside paged cache. Decode switches from
FlashAttention to SDPA+beta via monkey-patched attention layers. Two separate CUDA graph
captures handle pre-compaction (FlashAttention) and post-compaction (SDPA+beta) phases.
Beta training hooks (`compaction.py`) inject matching bias into attention_mask during FSDP2
training for consistency.

**Suffix queries**: When `use_suffix_queries=true`, the compaction algorithm uses real query
vectors from suffix tokens instead of random Gaussian probes. A prefill pass re-runs the
suffix through the model with hooks on vLLM's inner `Attention` class to capture post-RoPE
queries at every layer. Model-agnostic (works for any architecture using vLLM's Attention).
+3% accuracy over random probes at ~20% slower wall time (extra prefill per compaction event).

**Auto-batching**: Individual `/compact_generate` requests are transparently batched into
`compact_generate_batch` calls (B=8) by `_RequestBatcher` in routes.py.

### Critical invariants

- **KV extraction uses `current_seq_len - 1`**: After decode, boundary token is sampled but
  its KV isn't written yet.
- `current_seq_len = new_seq_len + 1` after compaction (recomputes boundary token's KV)
- `position_offset` tracks cumulative tokens removed (for correct RoPE positions)
- **FSDP + segmented forward**: All ranks must call model.forward() same number of times
  per micro-step. Variable segment counts require dummy padding + all-reduce MAX.
- CUDA graphs: only with TP=1; batch mode uses two-phase capture (pre/post compaction)
- `empty_cache()` between compaction segments in trainer prevents CUDA fragmentation OOM
- AC disable in segmented_forward is LoRA-only (Full FT keeps AC enabled)
- **Full-context softmax is default**: `compact_kv()` scores importance over all keys
  (prompt + assistant). Do NOT regress to assistant-only scoring.
- **NNLS target must use K_all_h (all keys)**, not Kw_h (window-only). Using window-only
  produces beta that's too small, causing 2.4x lower reward.

### Training (default: 2-node, 4+4 layout)

Node 1: 4 inference servers (TP=1, ports 8000-8003)
Node 2: 4 trainer GPUs (FSDP2) + orchestrator (CPU)

```bash
sbatch -A m5017 -C "gpu&hbm80g" --qos=premium --time 48:00:00 --gpus-per-node 4 --nodes=2 ~/compaction_determ_nobeta.sh
```

### Running evals

```bash
# Inside container:
python scripts/eval_rg_mix.py --mode compaction --n 100 \
    --max-kv-len 2048 --max-total-tokens 8192 \
    --n-compacts 99 --compact-ratio 0.25 --compact-window 1024

python scripts/eval_rg_mix.py --mode baseline --n 100

# With suffix queries:
python scripts/eval_rg_mix.py --mode compaction --n 100 \
    --max-kv-len 2048 --max-total-tokens 8192 \
    --n-compacts 99 --compact-ratio 0.25 --compact-window 1024 \
    --use-suffix-queries
```
