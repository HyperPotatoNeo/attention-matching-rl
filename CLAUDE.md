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
| `src/compaction_env/env.py` | CompactionEnv + TurnCompactionEnv (verifiers env wrappers) |
| `src/turn_compaction_env.py` | `load_environment` shim for `vf.load_environment("turn_compaction_env")` |
| `src/balrog_bench.py` | Vendored BALROG verifiers env (not on PyPI — PyPI `balrog-bench` is wrong package) |
| `scripts/eval_rg_mix.py` | rg-mix-env evaluation (compaction, baseline, and RSA modes) |
| `scripts/eval_aime_rsa.py` | AIME benchmark for RSA vs baseline comparison |
| `scripts/eval_balrog_babyai.py` | BabyAI (MiniGrid) multi-turn eval (compaction, baseline, markovian) |
| `configs/compaction/qwen3_4b_balrog_babyai.toml` | BabyAI baseline training config |
| `configs/compaction/qwen3_4b_turn_compaction_babyai.toml` | BabyAI turn-based compaction training config |
| `scripts/start_4servers.sh` | Launch 4 TP=1 servers for DP=4 |
| `configs/compaction/qwen3_4b_fullft_train.toml` | **Default training config** — 2-node, mixed-mode |
| `configs/compaction/qwen3_4b_beta_test.toml` | Beta attention test config |
| `configs/compaction/qwen3_4b_markovian_test.toml` | Markovian mode — Qwen3-4B, 50 steps |
| `configs/compaction/qwen3_06b_markovian_test.toml` | Markovian mode — Qwen3-0.6B, fast E2E test |
| `configs/compaction/qwen3_4b_serve_tp1.toml` | TP=1 server config (compaction) |
| `configs/compaction/qwen3_06b_serve_tp1.toml` | TP=1 server config (0.6B) |

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

**Markovian mode**: When `compaction_mode="markovian"`, the window is hard-deleted instead
of compressed. The cache becomes `[prompt | suffix]` with no C1/C2. Supported in both
inference (`worker.py`) and trainer (`segmented_forward`). Config field `compaction_mode`
is auto-synced from env args to trainer config.

**Turn-based compaction (TurnCompactionEnv)**: Uses `/compact_session/create` + `/compact_session/step`
to maintain KV state across turns. Compaction fires between turns (server-side, controlled by
`n_max_turns`/`n_preserved_turns`). `segment_boundaries` are accumulated turn-by-turn and stored
on `trajectory[0]["extras"]` so `interleave_rollout` can reconstruct the full sequence for training.
Turn-based mode: boundary placed after the turn whose accumulated KV triggered compaction.
Config: `env_id="turn_compaction_env"`, trainer needs `compact_target_ratio` + `compaction_mode`.
Three critical invariants: (1) `max_turns=n_max_turns` (not inner env's `max_turns=-1`) so verifiers
stops at the right count; (2) `setup_state` preserves the outer EnvGroup `task` key after calling
inner env (BalrogEnv overwrites it with the game task name, breaking buffer routing); (3) `dataset`
falls back to `eval_dataset` when inner env has none (BalrogEnv is eval-only in verifiers terms).

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

### RSA (Recursive Self-Aggregation)

`rsa_generate` on `CompactionWorker` implements RSA V2 with persistent compacted memory.
Prefills the question, forks KV into K candidates, generates a population, then iteratively:
selects peers, builds aggregation prompt, append-prefills onto base KV, generates probe
tokens for attention patterns, compacts the aggregation region, and generates new candidates.

Key helpers: `_fork_kv_blocks` (block-level KV copy), `_prefill_append` (chunked prefill
onto existing KV), `_batch_generate` (K candidates in parallel), `_inject_compacted_range`
(range-based KV injection), `compact_kv_range` in algorithm.py (range-based compaction).

Endpoint: `/rsa_generate` in routes.py. No auto-batching (RSA uses full GPU internally).

### BALROG / BabyAI dependency

`balrog` (the BALROG benchmark, not the Paylogic ACL package on PyPI) conflicts with
`verifiers` via `google-generativeai` → `prime-sandboxes`, so it cannot go in `pyproject.toml`.
Install manually after `uv sync`:

```bash
uv pip install git+https://github.com/DavidePaglieri/BALROG.git@a5fa0e7 --no-deps
```

`balrog-bench` is vendored at `src/balrog_bench.py` — no extra install needed.
See the `installation` skill for full details.

### Running evals

```bash
# Inside container:
python scripts/eval_rg_mix.py --mode compaction --n 100 \
    --max-kv-len 2048 --max-total-tokens 8192 \
    --n-compacts 99 --compact-ratio 0.25 --compact-window 1024

python scripts/eval_rg_mix.py --mode baseline --n 100

# RSA mode
python scripts/eval_rg_mix.py --mode rsa --n 100 \
    --rsa-K 4 --rsa-T 2 --rsa-k-peers 2 --rsa-probe-tokens 512

# Markovian mode
python scripts/eval_rg_mix.py --mode compaction --n 100 \
    --max-kv-len 2048 --max-total-tokens 8192 \
    --n-compacts 99 --compact-window 1024 \
    --compaction-mode markovian

# AIME benchmark
python scripts/eval_aime_rsa.py --mode rsa --n 30 --rsa-K 4 --rsa-T 2
python scripts/eval_aime_rsa.py --mode baseline --n 30

# BabyAI multi-turn grid-world (via minigrid)
python scripts/eval_balrog_babyai.py --mode baseline --n 10
python scripts/eval_balrog_babyai.py --mode compaction --n 10 \
    --max-kv-len 2048 --compact-ratio 0.25
python scripts/eval_balrog_babyai.py --mode markovian --n 10 \
    --max-kv-len 2048
```

@import /home/mila/e/emiliano.penaloza/orchestrator/CLAUDE.md
