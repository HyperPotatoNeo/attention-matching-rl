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
| `src/compaction_env/env.py` | CompactionEnv (verifiers SingleTurnEnv wrapper) |
| `scripts/eval_rg_mix.py` | rg-mix-env evaluation (compaction and baseline modes) |
| `scripts/start_4servers.sh` | Launch 4 TP=1 servers for DP=4 |
| `configs/compaction/qwen3_4b_fullft_train.toml` | **Default training config** — 2-node Full FT |
| `configs/compaction/qwen3_4b_serve_tp1.toml` | TP=1 server config (compaction) |
| `~/compaction_multinode.sh` | 2-node sbatch launch script |

### How it works

The `CompactionWorker` drives model forward passes manually inside a `collective_rpc` call,
bypassing vLLM's scheduler. Generation happens in segments. Between segments, the KV cache
is compacted using Attention Matching: select top-k keys by attention importance (C1),
solve least-squares for replacement values (C2), inject `[prompt | C1/C2 | suffix]` back
into paged blocks.

**Auto-batching**: Individual `/compact_generate` requests are transparently batched into
`compact_generate_batch` calls (B=8) by `_RequestBatcher` in routes.py. This gives ~5x
throughput improvement since collective_rpc only processes one request at a time per server.

### Critical invariants

- **KV extraction uses `current_seq_len - 1`**: After decode, boundary token is sampled but
  its KV isn't written yet.
- `current_seq_len = new_seq_len + 1` after compaction (recomputes boundary token's KV)
- `position_offset` tracks cumulative tokens removed (for correct RoPE positions)
- **FSDP + segmented forward**: All ranks must call model.forward() same number of times
  per micro-step. Variable segment counts → pad with dummy forwards + all-reduce MAX.
- CUDA graphs: only with TP=1 and B=1 (batch mode disables them)
- CUDA memory cleanup required after batch calls (`synchronize + gc.collect + empty_cache`)
- `empty_cache()` between compaction segments in trainer prevents CUDA fragmentation OOM
- AC disable in segmented_forward is LoRA-only (Full FT keeps AC enabled)

### Training

```bash
# Submit 2-node training job (default config):
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
