# KV Cache Compaction RL

RL training with mid-generation KV cache compaction using [Attention Matching](https://arxiv.org/abs/2602.16284). This enables learning over long effective contexts (8K+ tokens) within fixed KV memory budgets (2K), by compressing the cache between generation segments.

Built on [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) for async RL and [vLLM](https://github.com/vllm-project/vllm) for inference.

## How It Works

The `CompactionWorker` drives model forward passes inside vLLM's `collective_rpc`, bypassing the scheduler for full control over the KV cache. Generation proceeds until the KV cache fills a budget (`max_kv_len`), then:

1. **Select** top-k keys by attention importance (C1)
2. **Solve** least-squares for replacement values (C2)
3. **Optionally compute** NNLS beta bias for partition function correction
4. **Inject** `[prompt | C1/C2 | suffix]` back into paged blocks
5. **Continue** generating until `max_total_tokens` or EOS

With `compute_beta=true`, an additive per-key bias corrects the softmax partition function mismatch after compaction. This uses contiguous KV mirrors (BetaState) alongside vLLM's paged cache, with SDPA decode replacing FlashAttention for the bias-corrected path.

## Setup

```bash
git clone https://github.com/HyperPotatoNeo/attention-matching-rl.git
cd attention-matching-rl
uv sync --all-extras
```

## Inference

### Single-GPU server

```bash
uv run inference @ configs/compaction/qwen3_4b_serve_tp1.toml --server.port 8000
```

### Test compaction generation

```python
import requests

resp = requests.post("http://localhost:8000/compact_generate", json={
    "prompt_ids": tokenizer.encode("Solve this problem..."),
    "max_kv_len": 2048,
    "max_total_tokens": 8192,
    "compact_target_ratio": 0.25,
    "compact_window": 1024,
    "n_compacts": 99,
    "compute_beta": True,
    "temperature": 0.6,
})
result = resp.json()
print(result["final_text"])
print(f"Tokens: {result['diagnostics']['total_tokens']}, "
      f"Compactions: {len(result['diagnostics']['compaction_events'])}")
```

### Evaluate on rg-mix-env

```bash
# Start 4 TP=1 servers
bash scripts/start_4servers.sh

# Run compaction eval
python scripts/eval_rg_mix.py --mode compaction --n 100 \
    --max-kv-len 2048 --max-total-tokens 8192 \
    --n-compacts 99 --compact-ratio 0.25 --compact-window 1024

# Run baseline eval (no compaction)
python scripts/eval_rg_mix.py --mode baseline --n 100
```

## Training

Training uses 2 nodes in mixed mode: 5 inference GPUs + 3 trainer GPUs.

**Architecture:**
- **Node 1** (4 GPUs): 4 independent TP=1 compaction servers (ports 8000-8003)
- **Node 2** (4 GPUs): 1 inference server on GPU 0 (port 8004) + FSDP2 trainer on GPUs 1-3

### Config

The training config at `configs/compaction/qwen3_4b_beta_test.toml` uses:
- `compute_beta = true` for partition function correction
- `max_kv_len = 2048`, `compact_window = 1024`, `ratio = 0.25`
- `max_total_tokens = 8192` effective context per rollout
- `batch_size = 256`, `rollouts_per_example = 8`
- Full fine-tune, lr=1e-6, AdamW with CPU offload

### Launch

1. **Create resolved config** — replace `__INFERENCE_NODE__` and `__TRAINER_NODE__` placeholders with actual hostnames
2. **Node 1**: Run `scripts/start_4servers.sh` (or `multinode/compaction/node1_inference.sh`)
3. **Node 2**: Run `multinode/compaction/node2_mixed.sh <resolved_config.toml>`

The trainer on node 2 waits for the inference server on GPU 0 to become ready, then starts the RL loop. Weight updates are broadcast via the filesystem.

## Key Files

| File | Purpose |
|------|---------|
| `src/prime_rl/inference/compaction/worker.py` | Generation + compaction (single & batch) |
| `src/prime_rl/inference/compaction/algorithm.py` | Attention Matching + NNLS beta solver |
| `src/prime_rl/inference/compaction/beta_attention.py` | BetaState mirrors + SDPA decode with bias |
| `src/prime_rl/inference/compaction/routes.py` | `/compact_generate` endpoint + auto-batching |
| `src/compaction_env/env.py` | CompactionEnv (verifiers wrapper) |
| `scripts/eval_rg_mix.py` | Evaluation script |

## Configs

| Config | Purpose |
|--------|---------|
| `qwen3_4b_beta_test.toml` | Beta attention training (5 steps, for testing) |
| `qwen3_4b_fullft_train.toml` | Full fine-tune training (production) |
| `qwen3_4b_serve_tp1.toml` | TP=1 compaction server |
| `qwen3_4b_baseline.toml` | TP=4 baseline (no compaction) |

## Docs

- [Implementation details](docs/compaction/IMPLEMENTATION.md) — algorithm, beta correction, CUDA graphs
- [Speed optimizations](docs/compaction/SPEED_OPTIMIZATION.md) — batching, graph capture, profiling

## Citation

Based on [Attention Matching](https://arxiv.org/abs/2602.16284):

```bibtex
@article{zweiger2025attention,
  title={Attention Matching: an Attention Decomposition Framework for Efficient KV Cache Compression},
  author={Zweiger, Adam},
  journal={arXiv preprint arXiv:2602.16284},
  year={2025}
}
```
