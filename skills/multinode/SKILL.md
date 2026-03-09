---
name: multinode
description: Launch multi-node compaction RL training on NERSC Perlmutter. Use when asked to start training, allocate nodes, set up containers, or debug multi-node issues.
---

# Multi-Node Compaction RL

## Architecture

2-node mixed mode: 5 inference GPUs + 3 trainer GPUs.

- **Node 1** (4 GPUs): 4 independent TP=1 compaction inference servers (ports 8000-8003)
- **Node 2** (4 GPUs): 1 inference server on GPU 0 (port 8004) + FSDP2 trainer on GPUs 1-3
- Containers use `--net=host` for cross-node communication
- Config uses `__INFERENCE_NODE__` / `__TRAINER_NODE__` placeholders, resolved at launch

## Quick Launch (sbatch)

```bash
sbatch -A m5017 -C "gpu&hbm80g" --qos=premium --time 24:00:00 --gpus-per-node 4 --nodes=2 ~/compaction_multinode.sh
```

This script handles everything: container setup, config resolution, server health checks, and trainer launch. Logs go to `$SCRATCH/multinode/compaction/`.

## Manual Launch (salloc, step by step)

### 1. Get 2-node interactive allocation

```bash
salloc -A m5017 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 --gpus-per-node 4 --nodes=2
```

Note the two node hostnames from `scontrol show hostnames $SLURM_JOB_NODELIST`.

### 2. Create containers on both nodes

Containers must use `--net=host` for cross-node HTTP access. Run from the login node (or any node in the allocation):

```bash
export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
IMAGE=docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8

# Node 1 (inference)
ssh <NODE1> "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
  podman-hpc run --rm -d --user \$(id -u):\$(id -g) --replace --name skyrl-inf \
  --group-add keep-groups --userns keep-id --net=host --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME \
  -v $SCRATCH:$SCRATCH -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w $SCRATCH/compaction-rl $IMAGE sleep infinity"

# Node 2 (mixed: inference + trainer)
ssh <NODE2> "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
  podman-hpc run --rm -d --user \$(id -u):\$(id -g) --replace --name skyrl-train \
  --group-add keep-groups --userns keep-id --net=host --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME -e WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8 \
  -v $SCRATCH:$SCRATCH -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w $SCRATCH/compaction-rl $IMAGE sleep infinity"
```

### 3. Resolve config (replace hostname placeholders)

```bash
TOML_TEMPLATE=$SCRATCH/compaction-rl/configs/compaction/qwen3_4b_fullft_train.toml
TOML_RESOLVED=$SCRATCH/multinode/compaction/resolved.toml
sed -e "s/__INFERENCE_NODE__/<NODE1>/g" -e "s/__TRAINER_NODE__/<NODE2>/g" "$TOML_TEMPLATE" > "$TOML_RESOLVED"
```

For beta test, use `qwen3_4b_beta_test.toml` as the template instead.

### 4. Launch inference on Node 1

```bash
ssh <NODE1> "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
  podman-hpc exec skyrl-inf bash $SCRATCH/multinode/compaction/node1_inference.sh"
```

This starts 4 TP=1 servers (one per GPU, ports 8000-8003). Wait for all 4 to report ready:

```bash
# Health check from inside container:
for port in 8000 8001 8002 8003; do
  curl -s http://localhost:$port/v1/models | grep -q Qwen && echo "Port $port: OK"
done
```

### 5. Launch mixed mode on Node 2

```bash
ssh <NODE2> "export HOME=$SCRATCH && export PODMANHPC_PODMAN_BIN=$PODMANHPC_PODMAN_BIN && \
  podman-hpc exec skyrl-train bash $SCRATCH/multinode/compaction/node2_mixed.sh $TOML_RESOLVED"
```

This starts 1 inference server on GPU 0 (port 8004), waits for it, then launches the trainer on GPUs 1-3.

## Key Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `compaction_multinode.sh` | `~/` | Full sbatch launch (handles everything) |
| `node1_inference.sh` | `$SCRATCH/multinode/compaction/` | 4 TP=1 servers inside container |
| `node2_mixed.sh` | `$SCRATCH/multinode/compaction/` | 1 inference + trainer inside container |
| `node2_rl.sh` | `$SCRATCH/multinode/compaction/` | Trainer-only (4 GPU, no local inference) |
| `run_node1.sh` | `$SCRATCH/multinode/compaction/` | Container launcher for node1 |
| `run_node2.sh` | `$SCRATCH/multinode/compaction/` | Container launcher for node2 |

## Logs

| Log | Path |
|-----|------|
| Server 8000-8003 | `$SCRATCH/multinode/compaction/server_800{0,1,2,3}.log` |
| Server 8004 | `$SCRATCH/multinode/compaction/server_8004.log` |
| Orchestrator | `$SCRATCH/compaction-rl/outputs/<run>/run_default/logs/orchestrator.log` |
| Trainer | stdout of `node2_mixed.sh` |

## Common Issues

**NCCL deadlock**: If trainer hangs at a specific NCCL SeqNum, check that `segmented_forward` pads dummy forwards to match max segment count across FSDP ranks.

**OOM at ~78 GiB**: Add `torch.cuda.empty_cache()` between compaction segments. The issue is CUDA memory fragmentation, not total allocation.

**Servers unreachable cross-node**: Containers must use `--net=host`. Verify with `curl http://<NODE1>:8000/v1/models` from Node 2's container.

**Config not resolved**: The TOML uses `__INFERENCE_NODE__` / `__TRAINER_NODE__` placeholders. These must be replaced with actual hostnames before launching the trainer.
