#!/bin/bash
set -e
export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman

# Stop any existing container
podman-hpc stop skyrl_compaction 2>/dev/null || true
podman-hpc rm skyrl_compaction 2>/dev/null || true

echo "Starting vLLM server container on $(hostname)..."
podman-hpc run -d \
  --user "$(id -u):$(id -g)" --replace --name skyrl_compaction \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH=/pscratch/sd/s/siddart2 -e HOME=/pscratch/sd/s/siddart2 \
  -p 8000:8000 \
  -v /pscratch/sd/s/siddart2:/pscratch/sd/s/siddart2 \
  -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w /pscratch/sd/s/siddart2/compaction-rl \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c 'unset NCCL_SOCKET_IFNAME && source .venv/bin/activate && export PYTHONPATH=/pscratch/sd/s/siddart2/compaction-rl/src:$PYTHONPATH && python -u -m prime_rl.entrypoints.inference @ configs/compaction/qwen3_4b_serve.toml 2>&1 | tee /pscratch/sd/s/siddart2/compaction-rl/server.log'

echo "Container started. Waiting for server health..."
for i in $(seq 1 300); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s!"
        exit 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "Still waiting (${i}s)..."
        tail -3 /pscratch/sd/s/siddart2/compaction-rl/server.log 2>/dev/null || true
    fi
    sleep 1
done

echo "Server did not start in 300s. Last 50 lines of server.log:"
tail -50 /pscratch/sd/s/siddart2/compaction-rl/server.log 2>/dev/null || echo "No log"
echo "Container logs:"
podman-hpc logs skyrl_compaction 2>&1 | tail -50
exit 1
