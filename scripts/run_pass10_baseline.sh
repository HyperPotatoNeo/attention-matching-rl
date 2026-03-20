#!/bin/bash
set -e
cd /home/mila/e/emiliano.penaloza/rsa-compaction
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME
export HF_HOME=/network/scratch/e/emiliano.penaloza/huggingface

for pass in $(seq 1 10); do
    echo "========== BASELINE PASS $pass / 10 =========="
    python scripts/eval_rg_mix.py \
        --mode baseline \
        --n 100 \
        --seed 42 \
        --max-tokens-per-segment 2048 \
        --n-compacts 3 \
        --ports 8000,8001,8002,8003 \
        --baseline-concurrency 25 \
        --output "results_baseline_pass${pass}_100.json"
    echo ""
done
echo "ALL 10 BASELINE PASSES DONE"
