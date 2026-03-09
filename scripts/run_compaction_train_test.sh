#!/bin/bash
# Test compaction RL training - 5 steps with countdown env.
# Run inside container on a 4-GPU compute node.
set -e
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

echo "=== Compaction RL Training Test ==="
echo "Date: $(date)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

# Run full RL pipeline (orchestrator + trainer + inference server)
uv run rl @ configs/compaction/qwen3_4b_countdown_train.toml 2>&1 | tee compaction_train_test.log
