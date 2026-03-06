#!/bin/bash
# Run compaction eval on existing 4-server setup.
# Prereq: start_4servers.sh already running on this node.
set -e
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv pip install reasoning-gym 2>&1 | tail -3

python scripts/eval_rg_mix.py \
    --mode compaction \
    --n 100 \
    --max-tokens-per-segment 2048 \
    --n-compacts 3 \
    --compact-ratio 0.3 \
    --ports 8000,8001,8002,8003 \
    --output results_compaction_100.json
