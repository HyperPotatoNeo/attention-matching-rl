#!/bin/bash
# Launch 4 independent vLLM servers, one per GPU, for data-parallel compaction.
set -e
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    echo "Starting server on GPU $gpu, port $port..."
    CUDA_VISIBLE_DEVICES=$gpu nohup uv run inference \
        @ configs/compaction/qwen3_4b_serve_tp1.toml \
        --server.port $port \
        > server_gpu${gpu}.log 2>&1 &
done

echo "Waiting for all 4 servers..."
for gpu in 0 1 2 3; do
    port=$((8000 + gpu))
    for i in $(seq 1 180); do
        if curl -s -o /dev/null -w '%{http_code}' http://localhost:$port/health 2>/dev/null | grep -q 200; then
            echo "  GPU $gpu (port $port) ready after ${i}s"
            break
        fi
        sleep 1
    done
done

echo "All servers started. Keeping container alive..."
# Keep container running — wait for all background server processes
wait
