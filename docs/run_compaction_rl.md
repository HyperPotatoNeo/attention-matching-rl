# Running Compaction RL Training

## Quick Start

All training runs use 4 GPUs on a single node: 2 for inference (DP=2, TP=1) and 2 for training (FSDP2).

```bash
# From the project root:
uv run rl @ configs/compaction/<config>.toml
```

For SLURM (Mila cluster, 4x A100-80GB, 3h short-unkillable):

```bash
sbatch --partition=short-unkillable --time=3:00:00 --gres=gpu:a100l:4 \
    --mem=128G --cpus-per-task=16 \
    --output=../scratch/outputs/<run-name>/job_%j.log \
    --error=../scratch/outputs/<run-name>/job_%j.log \
    --wrap="cd $(pwd) && source .venv/bin/activate && uv run rl @ configs/compaction/<config>.toml"
```

## Available Configs

All configs are turn-based compaction on BabyAI. The key knobs are `n_max_turns` (how many turns accumulate before compaction fires) and `n_preserved_turns` (how many recent turns survive compaction).

| Config | Mode | n_max_turns | n_preserved_turns | Description |
|--------|------|-------------|-------------------|-------------|
| `babyai_am_default.toml` | attention_matching | 4 | 2 | Default AM — compacts every 4 turns, keeps last 2 |
| `babyai_am_aggressive.toml` | attention_matching | 3 | 1 | Aggressive AM — compacts every 3 turns, keeps only last 1 |
| `babyai_am_conservative.toml` | attention_matching | 6 | 4 | Conservative AM — compacts every 6 turns, keeps last 4 |
| `babyai_markovian.toml` | markovian | 4 | 2 | Hard-delete — same cadence as AM default, but drops instead of compressing |

All configs share:
- **Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Checkpoint interval**: 10 steps
- **Eval interval**: 10 steps (50 examples, temperature=0.0)
- **Environment**: BabyAI (MiniGrid) multi-turn grid-world via `balrog-bench`
- **Max turns**: 30 per episode
- **compact_target_ratio**: 0.25 (keeps 25% of KV tokens after compaction)

## Config Differences Explained

**AM Default (4t2p)** — Compaction fires after 4 turns of history accumulate. The 2 most recent turns are preserved; older turns are compressed via attention matching (C1: select top-k keys by attention importance, C2: least-squares replacement values). This is the balanced setting.

**AM Aggressive (3t1p)** — Compacts more often (every 3 turns) and retains less (only 1 turn). The model must learn with significantly less context available after compaction. Tests whether AM can preserve enough signal under high compression.

**AM Conservative (6t4p)** — Compacts less often (every 6 turns) and retains more (4 turns). More context is available at all times, but compaction covers a wider window when it does fire. Tests whether less frequent compaction helps.

**Markovian (4t2p)** — Same turn cadence as AM default, but hard-deletes the compaction window instead of compressing it. The cache becomes `[prompt | preserved_turns]` with no C1/C2. Serves as a lower bound — if AM doesn't beat this, the compression isn't adding value.

## Running on SLURM (Mila)

```bash
# AM Default (4 turns, preserve 2)
sbatch --partition=short-unkillable --time=3:00:00 --gres=gpu:a100l:4 \
    --mem=128G --cpus-per-task=16 \
    --output=../scratch/outputs/babyai-am-4t2p/job_%j.log \
    --error=../scratch/outputs/babyai-am-4t2p/job_%j.log \
    --wrap="cd $(pwd) && source .venv/bin/activate && uv run rl @ configs/compaction/babyai_am_default.toml"

# AM Aggressive (3 turns, preserve 1)
sbatch --partition=short-unkillable --time=3:00:00 --gres=gpu:a100l:4 \
    --mem=128G --cpus-per-task=16 \
    --output=../scratch/outputs/babyai-am-3t1p/job_%j.log \
    --error=../scratch/outputs/babyai-am-3t1p/job_%j.log \
    --wrap="cd $(pwd) && source .venv/bin/activate && uv run rl @ configs/compaction/babyai_am_aggressive.toml"

# AM Conservative (6 turns, preserve 4)
sbatch --partition=short-unkillable --time=3:00:00 --gres=gpu:a100l:4 \
    --mem=128G --cpus-per-task=16 \
    --output=../scratch/outputs/babyai-am-6t4p/job_%j.log \
    --error=../scratch/outputs/babyai-am-6t4p/job_%j.log \
    --wrap="cd $(pwd) && source .venv/bin/activate && uv run rl @ configs/compaction/babyai_am_conservative.toml"

# Markovian (4 turns, preserve 2)
sbatch --partition=short-unkillable --time=3:00:00 --gres=gpu:a100l:4 \
    --mem=128G --cpus-per-task=16 \
    --output=../scratch/outputs/babyai-mk-4t2p/job_%j.log \
    --error=../scratch/outputs/babyai-mk-4t2p/job_%j.log \
    --wrap="cd $(pwd) && source .venv/bin/activate && uv run rl @ configs/compaction/babyai_markovian.toml"
```

## Resuming Training

All configs set `resume_step = -1`, which means "resume from the latest checkpoint if one exists". After a SLURM timeout or crash, simply resubmit the same command — it will find the checkpoint and continue.

Checkpoints are saved every 10 steps to `<output_dir>/checkpoints/step_N/`.

To verify a checkpoint exists:
```bash
ls ../scratch/outputs/<run-name>/checkpoints/
```

## Monitoring

### Logs

Each run writes logs to `<output_dir>/logs/`:
- `orchestrator.stdout` — rollout progress, step rewards, throughput
- `trainer.stdout` — loss, gradient norms, memory usage
- `inference.stdout` — vLLM server logs

### Key metrics to watch

From `trainer.stdout`:
```
Step N | Loss: X | Entropy: X | Mismatch KL: X | Peak Mem.: X GiB
```
- **Peak Mem** should stay under ~40 GiB on A100-80GB
- **Mismatch KL** measures divergence between reference and trained policy
- **Loss** is the RL policy gradient loss

From `orchestrator.stdout`:
```
Step N | Reward: X | Throughput: X tokens/s | Seq. Length: X tokens/sample
```

### WandB

All runs log to the `balrog-rl` WandB project. Per-subtask rewards (goto, pickup, open, etc.) are logged under `subtask_reward/` and `val_subtask_reward/` panels.

To resume logging to the same WandB run across restarts, add the run IDs to the per-component wandb config:
```toml
[trainer.wandb]
id = "<trainer-run-id>"

[orchestrator.wandb]
id = "<orchestrator-run-id>"
```
Note: `id` goes under `[trainer.wandb]` / `[orchestrator.wandb]`, NOT the top-level `[wandb]`.

## Running Evals (Standalone)

Evals compare compaction methods on BabyAI tasks without training. Launch 4 TP=1 servers, then run the eval script:

```bash
# Launch servers (4 GPUs, TP=1 each)
for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    CUDA_VISIBLE_DEVICES=$gpu uv run inference \
        @ configs/compaction/qwen3_4b_serve_tp1.toml \
        --server.port $port &
done

# Wait for health
for port in 8000 8001 8002 8003; do
    until curl -s http://localhost:$port/health > /dev/null 2>&1; do sleep 1; done
done

# Attention Matching eval (vary --n-max-turns and --n-preserved-turns)
uv run python scripts/eval_balrog_babyai.py \
    --mode compaction --use-sessions --use-suffix-queries \
    --max-kv-len 3000 --compact-ratio 0.25 \
    --n-max-turns 4 --n-preserved-turns 2 \
    --n 10 --max-turns 25 --max-response-tokens 512 \
    --ports 8000,8001,8002,8003 --save-traces \
    --output results_am_4t2p.json

# Markovian eval
uv run python scripts/eval_balrog_babyai.py \
    --mode markovian --use-sessions \
    --max-kv-len 3000 \
    --n-max-turns 4 --n-preserved-turns 2 \
    --n 10 --max-turns 25 --max-response-tokens 512 \
    --ports 8000,8001,8002,8003 --save-traces \
    --output results_mk_4t2p.json

# Baseline eval (no compaction)
uv run python scripts/eval_balrog_babyai.py \
    --mode baseline \
    --n 10 --max-turns 25 --max-response-tokens 512 \
    --ports 8000,8001,8002,8003 --save-traces \
    --output results_baseline.json
```

The eval script supports resume: if interrupted, re-running with the same `--output` file skips completed episodes.

## Troubleshooting

**Config validation error `Extra inputs are not permitted`**: The top-level `[wandb]` section only supports `project`, `name`, and `offline`. Per-run fields like `id` must go under `[trainer.wandb]` or `[orchestrator.wandb]`.

**No checkpoints saved after 3h job**: Check `ckpt.interval` — with ~7 steps per 3h window, an interval of 100 means no checkpoint is ever saved. Use interval 10 or lower.

**JSONDecodeError warnings during rollouts**: Normal — ~5-8% of rollouts fail when the model generates invalid JSON actions. These are skipped gracefully and do not affect training.

**CUDA OOM**: Peak memory should be ~38 GiB. If OOM occurs, check that `optim_cpu_offload = true` and `ac.freq = 1` are set in the trainer config.
