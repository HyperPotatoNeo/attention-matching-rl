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

Generate text with mid-sequence KV cache compaction. Supports two modes:

1. **Fixed segment mode** (default): Generate `max_tokens_per_segment` tokens, compact, repeat `n_compacts` times
2. **KV budget mode** (set `max_kv_len`): Generate until KV cache fills to `max_kv_len`, compact `compact_window` tokens, continue until `max_total_tokens` total completion tokens

#### Fixed segment mode

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

#### KV budget mode

```json
{
    "prompt_ids": [1, 2, 3, ...],
    "max_kv_len": 2048,
    "max_total_tokens": 4096,
    "compact_target_ratio": 0.25,
    "compact_window": 512,
    "n_compacts": 99,
    "temperature": 0.6,
    "top_p": 0.95
}
```

In KV budget mode, segment sizes are variable:
- Segment 0: generates ~(`max_kv_len` - `prompt_len`) tokens until KV fills
- Segment 1+: generates ~(`compact_window` * (1 - `ratio`)) tokens (the space freed by compaction)
- Stops when total completion tokens reach `max_total_tokens`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_ids` | `list[int]` | required | Tokenized prompt |
| `max_seq_len` | `int` | 8192 | Maximum total sequence length (fixed segment mode) |
| `max_tokens_per_segment` | `int\|null` | auto | Tokens per segment before compaction (fixed segment mode) |
| `n_compacts` | `int` | 3 | Maximum number of compaction events |
| `compact_target_ratio` | `float` | 0.3 | Target compression ratio for the compacted window |
| `compact_window` | `int\|null` | null | Compress only first N assistant tokens (null = all) |
| `temperature` | `float` | 0.7 | Sampling temperature |
| `top_p` | `float` | 0.95 | Top-p sampling |
| `max_kv_len` | `int\|null` | null | KV budget mode: max KV cache length before compaction triggers |
| `max_total_tokens` | `int\|null` | null | KV budget mode: stop after this many total completion tokens |

**Response:**

```json
{
    "all_token_ids": [101, 203, ...],
    "all_logprobs": [-0.12, -0.34, ...],
    "final_text": "decoded text...",
    "diagnostics": {
        "prompt_len": 45,
        "total_tokens": 4096,
        "total_time": 67.3,
        "compaction_events": [
            {
                "segment": 0,
                "assistant_tokens_before": 1848,
                "assistant_tokens_after": 1464,
                "prefix_compacted": 512,
                "prefix_after": 128,
                "suffix_preserved": 1336,
                "ratio": 0.79,
                "algo_time": 0.41
            }
        ],
        "segment_boundaries": [1848, 2232, 2616, ...]
    }
}
```

## Training

### Default: 2-node multi-node training

The default training config is `configs/compaction/qwen3_4b_fullft_train.toml` (multi-node, Full FT):

```bash
sbatch -A m5017 -C "gpu&hbm80g" --qos=premium --time 24:00:00 --gpus-per-node 4 --nodes=2 ~/compaction_multinode.sh
```

Architecture:
- **Node 1** (4 GPUs): 4 independent TP=1 compaction inference servers (ports 8000-8003)
- **Node 2** (4 GPUs): FSDP2 trainer (all 4 GPUs) + Orchestrator (CPU)
- Weight broadcast via Lustre filesystem

Training parameters:
- Full fine-tune (no LoRA), lr=1e-6, AdamW with CPU offload
- KV budget: max_kv_len=2048, compact_window=1024, ratio=0.25, max_total_tokens=8192
- batch_size=256, rollouts_per_example=8, seq_len=9216

Launch scripts:
- `~/compaction_multinode.sh` — sbatch entry point
- `$SCRATCH/multinode/compaction/node1_inference.sh` — starts 4 servers inside container
- `$SCRATCH/multinode/compaction/node2_rl.sh` — starts trainer + orchestrator inside container

W&B metrics logged per step:
- `compaction/avg_num_compactions` — mean compaction events per rollout
- `compaction/accuracy_N_compactions` — reward grouped by # of compactions
- `compaction/total_generated_tokens` — mean total tokens generated

## Configs

| Config | Purpose |
|--------|---------|
| `configs/compaction/qwen3_4b_fullft_train.toml` | **Default training config** — 2-node Full FT with KV budget mode |
| `configs/compaction/qwen3_4b_serve_tp1.toml` | TP=1 compaction server (used by training and `start_4servers.sh`) |
| `configs/compaction/qwen3_4b_baseline.toml` | TP=4 baseline server (no compaction) |
| `configs/compaction/qwen3_4b_countdown_train.toml` | LoRA RL training (countdown, validation) |

## Scripts

| Script | Purpose |
|--------|---------|
| `~/compaction_multinode.sh` | **Default launch script** — 2-node sbatch with container setup |
| `scripts/start_4servers.sh` | Launch 4 TP=1 servers on ports 8000-8003 (single node) |
| `scripts/eval_rg_mix.py` | Evaluate on rg-mix-env (compaction or baseline mode) |
| `scripts/run_compaction_eval.sh` | Wrapper to run compaction eval |
| `scripts/run_baseline_eval.sh` | Wrapper to run baseline eval |
| `scripts/start_baseline_server.sh` | Launch single TP=4 baseline server |

## Eval Results (Qwen3-4B base, rg-mix-env, 100 problems)

| Configuration | Accuracy | Avg Tokens | Throughput |
|---------------|----------|------------|------------|
| Baseline (8192 tokens, no compaction) | 15.0% | ~5000 | 213 tok/s |
| Full compaction (ratio=0.3, window=all) | 6.0% | ~7200 | — |
| Partial compaction (ratio=0.25, window=512) | 13.0% | ~7400 | — |
| KV budget (w=1024, ratio=0.25, 8k total) | 16.0% | ~7855 | 440 tok/s |
| KV budget batched (B=4, same params) | 9.0% | ~7890 | 616 tok/s |

The accuracy gap is what RL training aims to close — teaching the model to
reason effectively despite periodic context compression.
