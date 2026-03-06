# KV Cache Compaction for Long-Context RL

Mid-generation KV cache compression using Attention Matching, implemented as a vLLM worker extension. Generates text in segments, compacting the KV cache between segments to allow longer effective context within fixed memory.

## Quick Start

### 1. Allocate a GPU node

```bash
salloc -A m4881 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 --gpus-per-node 4
```

### 2. Launch compaction servers (DP=4, one per GPU)

```bash
# On the compute node, inside the container:
cd $SCRATCH/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME
bash scripts/start_4servers.sh
```

This starts 4 independent TP=1 vLLM servers on ports 8000-8003, each with the `CompactionWorker` extension that exposes `/compact_generate`.

Or use the container launcher:

```bash
export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name skyrl_compaction \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME \
  -v "$SCRATCH":"$SCRATCH" -v "$HOME":"$HOME" \
  -w "$SCRATCH/compaction-rl" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash scripts/start_4servers.sh
```

### 3. Run evaluation

From inside the same container (use `podman-hpc exec`):

```bash
# Compaction eval (DP=4, 100 problems)
python scripts/eval_rg_mix.py \
    --mode compaction --n 100 \
    --max-tokens-per-segment 2048 \
    --n-compacts 3 \
    --compact-ratio 0.25 \
    --compact-window 512 \
    --output results.json

# Baseline eval (no compaction, single TP=4 server)
python scripts/eval_rg_mix.py \
    --mode baseline --n 100 \
    --server-url http://localhost:8000
```

### 4. Launch a baseline server (for comparison)

```bash
uv run inference @ configs/compaction/qwen3_4b_baseline.toml
```

This uses TP=4 with `max_model_len=12288` and no compaction.

## API

### POST `/compact_generate`

Generate text with mid-sequence KV cache compaction.

```json
{
    "prompt_ids": [1, 2, 3, ...],
    "max_seq_len": 8192,
    "max_tokens_per_segment": 2048,
    "n_compacts": 3,
    "compact_target_ratio": 0.25,
    "compact_window": 512,
    "temperature": 0.6,
    "top_p": 0.95
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_ids` | `list[int]` | required | Tokenized prompt |
| `max_seq_len` | `int` | 8192 | Maximum total sequence length |
| `max_tokens_per_segment` | `int` | auto | Tokens per segment before compaction |
| `n_compacts` | `int` | 3 | Maximum number of compaction events |
| `compact_target_ratio` | `float` | 0.3 | Target compression ratio for the prefix |
| `compact_window` | `int\|null` | null | Compress only first N assistant tokens (null = all) |
| `temperature` | `float` | 0.7 | Sampling temperature |
| `top_p` | `float` | 0.95 | Top-p sampling |

**Response:**

```json
{
    "all_token_ids": [101, 203, ...],
    "all_logprobs": [-0.12, -0.34, ...],
    "final_text": "decoded text...",
    "diagnostics": {
        "prompt_len": 45,
        "total_tokens": 8192,
        "total_time": 67.3,
        "compaction_events": [
            {
                "segment": 0,
                "assistant_tokens_before": 2048,
                "assistant_tokens_after": 1664,
                "prefix_compacted": 512,
                "prefix_after": 128,
                "suffix_preserved": 1536,
                "ratio": 0.81,
                "algo_time": 0.41
            }
        ],
        "segment_boundaries": [2048, 4096, 6144, 8192]
    }
}
```

## Configs

| Config | Purpose |
|--------|---------|
| `configs/compaction/qwen3_4b_serve_tp1.toml` | TP=1 compaction server (use with `start_4servers.sh`) |
| `configs/compaction/qwen3_4b_baseline.toml` | TP=4 baseline server (no compaction) |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/start_4servers.sh` | Launch 4 TP=1 servers on ports 8000-8003 |
| `scripts/eval_rg_mix.py` | Evaluate on rg-mix-env (compaction or baseline mode) |
| `scripts/run_compaction_eval.sh` | Wrapper to run compaction eval |
| `scripts/run_baseline_eval.sh` | Wrapper to run baseline eval |
| `scripts/start_baseline_server.sh` | Launch single TP=4 baseline server |

## Eval Results (Qwen3-4B, rg-mix-env, 100 problems)

| Configuration | Accuracy | Avg Tokens |
|---------------|----------|------------|
| Baseline (8192 tokens, no compaction) | 15.0% | ~5000 |
| Full compaction (ratio=0.3, window=all) | 6.0% | ~7200 |
| Partial compaction (ratio=0.25, window=512) | 13.0% | ~7400 |

Partial compaction preserves most baseline performance while allowing 3 compaction events.
