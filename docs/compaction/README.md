# KV Cache Compaction for Long-Context RL

Mid-generation KV cache compression using Attention Matching, implemented as a vLLM worker extension. Generates text in segments, compacting the KV cache between segments to allow longer effective context within fixed memory.

## Quick Start

### 1. Allocate a GPU node

```bash
salloc -A m5017 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 --gpus-per-node 4
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

### 3. Run evaluation

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

## API

### POST `/compact_generate`

Generate text with mid-sequence KV cache compaction. Requests are transparently batched
(up to B=8) by the server for higher throughput. Supports two modes:

1. **Fixed segment mode** (default): Generate `max_tokens_per_segment` tokens, compact, repeat
2. **KV budget mode** (set `max_kv_len`): Generate until KV fills, compact `compact_window` tokens, continue until `max_total_tokens`

#### Fixed segment mode

```json
{
    "prompt_ids": [1, 2, 3],
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
    "prompt_ids": [1, 2, 3],
    "max_kv_len": 2048,
    "max_total_tokens": 4096,
    "compact_target_ratio": 0.25,
    "compact_window": 512,
    "n_compacts": 99,
    "compute_beta": true,
    "temperature": 0.6,
    "top_p": 0.95
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_ids` | `list[int]` | required | Tokenized prompt |
| `max_seq_len` | `int` | 8192 | Maximum total sequence length (fixed segment mode) |
| `max_tokens_per_segment` | `int\|null` | auto | Tokens per segment before compaction |
| `n_compacts` | `int` | 3 | Maximum compaction events |
| `compact_target_ratio` | `float` | 0.3 | Fraction of compacted window to keep |
| `compact_window` | `int\|null` | null | Compress only first N assistant tokens (null = all) |
| `temperature` | `float` | 0.7 | Sampling temperature |
| `top_p` | `float` | 0.95 | Top-p sampling |
| `max_kv_len` | `int\|null` | null | KV budget: compact when cache reaches this length |
| `max_total_tokens` | `int\|null` | null | KV budget: stop after this many completion tokens |
| `compute_beta` | `bool` | false | Compute NNLS beta bias for partition function correction |

**Response:**

```json
{
    "all_token_ids": [101, 203],
    "all_logprobs": [-0.12, -0.34],
    "final_text": "decoded text...",
    "diagnostics": {
        "prompt_len": 45,
        "total_tokens": 4096,
        "total_time": 67.3,
        "compaction_events": [{
            "segment": 0,
            "assistant_tokens_before": 1848,
            "assistant_tokens_after": 1464,
            "prefix_compacted": 512,
            "prefix_after": 128,
            "suffix_preserved": 1336,
            "ratio": 0.79,
            "algo_time": 0.41
        }],
        "segment_boundaries": [1848, 2232, 2616]
    }
}
```

## Training

Default: 2-node mixed mode (5 inference GPUs + 3 trainer GPUs). See the `multinode` skill
in `skills/multinode/` for step-by-step launch instructions.

```bash
sbatch -A m5017 -C "gpu&hbm80g" --qos=premium --time 24:00:00 --gpus-per-node 4 --nodes=2 ~/compaction_multinode.sh
```

Architecture:
- **Node 1** (4 GPUs): 4 TP=1 compaction inference servers (ports 8000-8003)
- **Node 2** (4 GPUs): 1 inference server (GPU 0, port 8004) + FSDP2 trainer (GPUs 1-3)
- Weight broadcast via Lustre filesystem
- Containers use `--net=host` for cross-node communication

Training parameters:
- Full fine-tune (no LoRA), lr=1e-6, AdamW with CPU offload
- KV budget: max_kv_len=2048, compact_window=1024, ratio=0.25, max_total_tokens=8192
- batch_size=256, rollouts_per_example=8, seq_len=9216

## Configs

| Config | Purpose |
|--------|---------|
| `qwen3_4b_fullft_train.toml` | **Default** -- 2-node Full FT, KV budget mode |
| `qwen3_4b_beta_test.toml` | Beta attention test (compute_beta=true) |
| `qwen3_4b_serve_tp1.toml` | TP=1 compaction server |
| `qwen3_4b_baseline.toml` | TP=4 baseline server (no compaction) |
| `qwen3_4b_countdown_train.toml` | LoRA RL training (countdown, validation) |

## Scripts

| Script | Purpose |
|--------|---------|
| `~/compaction_multinode.sh` | **Default launch** -- 2-node sbatch |
| `scripts/start_4servers.sh` | 4 TP=1 servers on ports 8000-8003 |
| `scripts/eval_rg_mix.py` | Evaluate on rg-mix-env |

## Eval Results (Qwen3-4B base, rg-mix-env, 100 problems)

| Configuration | Accuracy | Avg Tokens | Throughput |
|---------------|----------|------------|------------|
| Baseline (8192 tokens, no compaction) | 15.0% | ~5000 | 213 tok/s |
| Partial compaction (ratio=0.25, window=512) | 13.0% | ~7400 | -- |
| KV budget (w=1024, ratio=0.25, 8k total) | 16.0% | ~7855 | 440 tok/s |
| KV budget batched (B=4, same params) | 9.0% | ~7890 | 616 tok/s |

The accuracy gap is what RL training aims to close.
