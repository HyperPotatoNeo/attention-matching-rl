#!/bin/bash
# Run baseline eval (no compaction) on standard vLLM server.
set -e
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv pip install reasoning-gym 2>&1 | tail -3

python scripts/eval_rg_mix.py \
    --mode baseline \
    --n 100 \
    --max-tokens-per-segment 2048 \
    --n-compacts 3 \
    --server-url http://localhost:8000 \
    --output results_baseline_100.json
