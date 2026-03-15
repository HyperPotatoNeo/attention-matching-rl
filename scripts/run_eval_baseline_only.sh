#!/bin/bash
set -e
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

MODEL_DIR=/pscratch/sd/s/siddart2/compaction-rl/outputs/rg-mix-baseline/merged_step_600

echo "=== Running baseline eval (no compaction, 200 problems, seed=42) ==="
python scripts/eval_rg_mix.py \
    --mode baseline \
    --n 200 --seed 42 \
    --model "$MODEL_DIR" \
    --max-tokens-per-segment 8192 \
    --n-compacts 0 \
    --ports 8000,8001,8002,8003 \
    --output results_step600_baseline_200.json
echo "=== Done ==="
