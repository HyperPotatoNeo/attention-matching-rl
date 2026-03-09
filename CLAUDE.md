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
| `src/prime_rl/inference/compaction/algorithm.py` | Attention Matching + NNLS beta solver |
| `src/prime_rl/inference/compaction/beta_attention.py` | BetaState mirrors + SDPA decode with per-token bias |
| `src/compaction_env/env.py` | CompactionEnv (verifiers SingleTurnEnv wrapper) |
| `scripts/eval_rg_mix.py` | rg-mix-env evaluation (compaction and baseline modes) |
| `scripts/start_4servers.sh` | Launch 4 TP=1 servers for DP=4 |
| `configs/compaction/qwen3_4b_fullft_train.toml` | **Default training config** — 2-node, mixed-mode |
| `configs/compaction/qwen3_4b_beta_test.toml` | Beta attention test config |
| `configs/compaction/qwen3_4b_serve_tp1.toml` | TP=1 server config (compaction) |

### How it works

The `CompactionWorker` drives model forward passes manually inside a `collective_rpc` call,
bypassing vLLM's scheduler. Generation happens in segments. Between segments, the KV cache
is compacted using Attention Matching: select top-k keys by attention importance (C1),
solve least-squares for replacement values (C2), optionally compute NNLS beta bias for
partition function correction, then inject `[prompt | C1/C2 | suffix]` back into paged blocks.

**Beta attention**: When `compute_beta=true`, the NNLS solver finds per-key additive biases
that correct the partition function mismatch between full and compacted attention.
`BetaState` maintains contiguous KV mirrors alongside paged cache. Decode switches from
FlashAttention to SDPA+beta via monkey-patched attention layers. Two separate CUDA graph
captures handle pre-compaction (FlashAttention) and post-compaction (SDPA+beta) phases.

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

### Training

Default: 2-node mixed mode (5 inference + 3 trainer GPUs). See the `multinode` skill for
launch instructions.

```bash
sbatch -A m5017 -C "gpu&hbm80g" --qos=premium --time 24:00:00 --gpus-per-node 4 --nodes=2 ~/compaction_multinode.sh
```

### Running evals

```bash
# Inside container:
python scripts/eval_rg_mix.py --mode compaction --n 100 \
    --max-kv-len 2048 --max-total-tokens 8192 \
    --n-compacts 99 --compact-ratio 0.25 --compact-window 1024

python scripts/eval_rg_mix.py --mode baseline --n 100
```
