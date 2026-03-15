#!/bin/bash
# Eval suffix_seq_7500 step_600 WITH compaction (suffix queries, 1024->32) on sokoban.
set -e
export PATH="/pscratch/sd/s/siddart2/.local/bin:$PATH"
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

MODEL_DIR=/pscratch/sd/s/siddart2/compaction-rl/outputs/suffix_seq_7500/merged_step_600
CONFIG=configs/compaction/qwen3_4b_serve_tp1.toml

echo "=== Starting 4 compaction servers (suffix_7500 step_600) ==="
for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    CUDA_VISIBLE_DEVICES=$gpu nohup inference \
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
    --mode compaction --n 200 --seed 42 \
    --task-filter sokoban_hard \
    --model "$MODEL_DIR" \
    --max-kv-len 2048 --compact-window 1024 \
    --compact-ratio 0.03125 --max-total-tokens 8192 \
    --n-compacts 99 --use-suffix-queries \
    --ports 8000,8001,8002,8003 \
    --output results_suffix7500_step600_compaction_sokoban_200.json
echo "=== Done ==="
