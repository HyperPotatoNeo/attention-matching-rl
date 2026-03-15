#!/bin/bash
# Eval rg-mix baseline step_600 checkpoint WITH compaction (suffix queries).
# Runs INSIDE the container on a compute node with 4 GPUs.
set -e

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
    for i in $(seq 1 300); do
        if curl -s -o /dev/null -w '%{http_code}' http://localhost:$port/health 2>/dev/null | grep -q 200; then
            echo "  GPU $gpu (port $port) ready after ${i}s"
            break
        fi
        if [ $i -eq 300 ]; then
            echo "  ERROR: GPU $gpu (port $port) failed to start"
            cat /tmp/server_gpu${gpu}.log
            exit 1
        fi
        sleep 1
    done
done
echo "All 4 servers ready."

echo ""
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
