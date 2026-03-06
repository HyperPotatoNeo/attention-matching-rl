#!/bin/bash
set -e
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

echo "Starting baseline vLLM server (TP=4, port 8000)..."
exec uv run inference @ configs/compaction/qwen3_4b_baseline.toml --server.port 8000
