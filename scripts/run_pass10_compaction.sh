#!/bin/bash
set -e
cd /home/mila/e/emiliano.penaloza/rsa-compaction
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME
export HF_HOME=/network/scratch/e/emiliano.penaloza/huggingface

# Pass 1 already exists as results_compaction_window512_r025_100.json
# but it used sampling_seed=0. Re-run as pass1 with explicit seed for consistency.
for pass in $(seq 1 10); do
    seed_offset=$(( (pass - 1) * 100000 ))
    echo "========== COMPACTION PASS $pass / 10 (sampling_seed=$seed_offset) =========="
    python scripts/eval_rg_mix.py \
        --mode compaction \
        --n 100 \
        --seed 42 \
        --max-tokens-per-segment 2048 \
        --n-compacts 3 \
        --compact-ratio 0.25 \
        --compact-window 512 \
        --sampling-seed $seed_offset \
        --ports 8000,8001,8002,8003 \
        --output "results_compaction_w512_pass${pass}_100.json"
    echo ""
done
echo "ALL 10 COMPACTION PASSES DONE"
