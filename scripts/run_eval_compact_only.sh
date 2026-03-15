#!/bin/bash
set -e
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

MODEL_DIR=/pscratch/sd/s/siddart2/compaction-rl/outputs/rg-mix-baseline/merged_step_600

echo "=== Running compaction eval (suffix queries, 200 problems, seed=42) ==="
python scripts/eval_rg_mix.py \
    --mode compaction \
    --n 200 --seed 42 \
    --model "$MODEL_DIR" \
    --max-kv-len 2048 \
    --compact-window 1024 \
    --compact-ratio 0.03125 \
    --max-total-tokens 8192 \
    --n-compacts 99 \
    --use-suffix-queries \
    --ports 8000,8001,8002,8003 \
    --output results_step600_compaction_suffix_200.json
echo "=== Done ==="
