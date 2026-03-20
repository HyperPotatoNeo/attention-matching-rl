#!/bin/bash

#SBATCH --job-name=eval-markovian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:1
#SBATCH --partition=main
#SBATCH --time=3:00:00
#SBATCH --output=../scratch/outputs/compaction_markovian_mila/eval_%j.log
#SBATCH --error=../scratch/outputs/compaction_markovian_mila/eval_%j.log

export PROJECT_DIR=/home/mila/e/emiliano.penaloza/rsa-compaction

cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate

# Start inference server on GPU 0
CUDA_VISIBLE_DEVICES=0 uv run inference @ configs/compaction/qwen3_4b_serve_tp1.toml --server.port 8000 &
SERVER_PID=$!

# Wait for server to be healthy
echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server healthy after ${i}s"
        break
    fi
    sleep 5
done

# Run markovian eval on rg_mix (n=100)
echo "Running markovian eval..."
uv run python scripts/eval_rg_mix.py \
    --mode markovian \
    --n 100 \
    --max-kv-len 2048 \
    --max-total-tokens 8192 \
    --n-compacts 99 \
    --compact-window 1024 \
    --ports 8000 \
    --output ../scratch/outputs/compaction_markovian_mila/eval_markovian_results.json

echo "Done. Killing server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
