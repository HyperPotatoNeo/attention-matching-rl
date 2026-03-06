@AGENTS.md

## Compaction Module

KV cache compaction lives in `src/prime_rl/inference/compaction/`. See
`docs/compaction/README.md` for usage and `docs/compaction/IMPLEMENTATION.md` for
the full technical walkthrough.

### Key files

| File | Purpose |
|------|---------|
| `src/prime_rl/inference/compaction/worker.py` | All compaction + generation logic |
| `src/prime_rl/inference/compaction/routes.py` | `/compact_generate` FastAPI endpoint |
| `scripts/eval_rg_mix.py` | rg-mix-env evaluation (compaction and baseline modes) |
| `scripts/start_4servers.sh` | Launch 4 TP=1 servers for DP=4 |
| `configs/compaction/qwen3_4b_serve_tp1.toml` | TP=1 server config (compaction) |
| `configs/compaction/qwen3_4b_baseline.toml` | TP=4 server config (baseline) |

### How it works (summary)

The `CompactionWorker` drives model forward passes manually inside a `collective_rpc` call,
bypassing vLLM's scheduler. Generation happens in segments. Between segments, the KV cache
is compacted using Attention Matching: select top-k keys by attention importance (C1),
solve least-squares for replacement values (C2), inject `[prompt | C1/C2 | suffix]` back
into paged blocks.

Partial compaction (`compact_window`): only the first N assistant tokens are compressed;
the suffix is preserved intact. The full KV (including suffix) is used for scoring.

### Critical invariants

- `current_seq_len = new_seq_len + 1` after compaction (recomputes last token's KV)
- `position_offset` tracks cumulative tokens removed (for correct RoPE positions)
- CUDA graphs only with TP=1 (NCCL ops break graphs)
- All scripts run inside the container, not on the host (ports not exposed)

### Running evals

```bash
# Inside container:
python scripts/eval_rg_mix.py --mode compaction --n 100 \
    --max-tokens-per-segment 2048 --n-compacts 3 \
    --compact-ratio 0.25 --compact-window 512

python scripts/eval_rg_mix.py --mode baseline --n 100
```

### Results files

- `results_baseline_100.json` — 15.0% baseline accuracy
- `results_compaction_100.json` — 6.0% full compaction accuracy
- `results_compaction_window512_100.json` — 13.0% partial compaction accuracy
