#!/bin/bash
# Eval step_600 checkpoint WITH compaction (random queries) on sokoban only.
# Runs on a compute node with 4 GPUs.
set -e

export PATH="/pscratch/sd/s/siddart2/.local/bin:$PATH"
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache

cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

MODEL_DIR=/pscratch/sd/s/siddart2/compaction-rl/outputs/rg-mix-baseline/merged_step_600
CONFIG=configs/compaction/qwen3_4b_serve_tp1.toml

echo "=== Starting 4 compaction servers (step_600 checkpoint) ==="
for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    echo "Starting server on GPU $gpu, port $port"
    CUDA_VISIBLE_DEVICES=$gpu nohup uv run inference \
        @ "$CONFIG" \
        --model.name "$MODEL_DIR" \
        --server.port $port \
        > /tmp/server_gpu${gpu}.log 2>&1 &
done

echo "Waiting for all 4 servers..."
for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    for i in $(seq 1 600); do
        if curl -s -o /dev/null -w '%{http_code}' http://localhost:$port/health 2>/dev/null | grep -q 200; then
            echo "  GPU $gpu (port $port) ready after ${i}s"
            break
        fi
        if [ $i -eq 600 ]; then
            echo "  ERROR: GPU $gpu (port $port) failed to start"
            cat /tmp/server_gpu${gpu}.log
            exit 1
        fi
        sleep 1
    done
done
echo "All 4 servers ready."

echo ""
echo "=== Running compaction eval (random queries, sokoban_hard, 200 problems, seed=42) ==="
python scripts/eval_rg_mix.py \
    --mode compaction \
    --n 200 --seed 42 \
    --task-filter sokoban_hard \
    --model "$MODEL_DIR" \
    --max-kv-len 2048 \
    --compact-window 1024 \
    --compact-ratio 0.25 \
    --max-total-tokens 8192 \
    --n-compacts 99 \
    --ports 8000,8001,8002,8003 \
    --output results_step600_compaction_random_sokoban_200.json
echo "=== Done ==="
