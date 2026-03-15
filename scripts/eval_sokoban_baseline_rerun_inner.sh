#!/bin/bash
# Rerun: Eval rg-mix-baseline step_600 WITHOUT compaction on sokoban (full 200).
set -e
export PATH="/pscratch/sd/s/siddart2/.local/bin:$PATH"
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

MODEL_DIR=/pscratch/sd/s/siddart2/compaction-rl/outputs/rg-mix-baseline/merged_step_600
CONFIG=configs/compaction/qwen3_4b_serve_tp1_baseline.toml

echo "=== Starting 4 baseline servers ==="
for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    CUDA_VISIBLE_DEVICES=$gpu nohup uv run inference \
        @ "$CONFIG" --model.name "$MODEL_DIR" --server.port $port \
        > /tmp/server_gpu${gpu}.log 2>&1 &
done

echo "Waiting for servers..."
for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    for i in $(seq 1 600); do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null | grep -q 200; then
            echo "  GPU $gpu (port $port) ready after ${i}s"; break
        fi
        [ $i -eq 600 ] && { echo "ERROR: GPU $gpu failed"; cat /tmp/server_gpu${gpu}.log; exit 1; }
        sleep 1
    done
done
echo "All servers ready."

python scripts/eval_rg_mix.py \
    --mode baseline --n 200 --seed 42 \
    --task-filter sokoban_hard \
    --model "$MODEL_DIR" \
    --max-tokens-per-segment 8192 --n-compacts 0 \
    --ports 8000,8001,8002,8003 \
    --output results_step600_baseline_sokoban_200.json
echo "=== Done ==="
