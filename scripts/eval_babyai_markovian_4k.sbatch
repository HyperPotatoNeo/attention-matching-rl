#!/bin/bash

#SBATCH --job-name=eval-babyai-markovian-4k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:2
#SBATCH --partition=main
#SBATCH --time=8:00:00
#SBATCH --output=~/scratch/outputs/babyai_evals/slurm_%j_markovian_4k.log
#SBATCH --error=~/scratch/outputs/babyai_evals/slurm_%j_markovian_4k.log

PROJECT_DIR=/home/mila/e/emiliano.penaloza/rsa-compaction
cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate

mkdir -p ~/scratch/outputs/babyai_evals

CUDA_VISIBLE_DEVICES=0 uv run inference @ configs/compaction/qwen3_4b_serve_tp1.toml \
    --server.port 8000 &
SERVER_PID=$!

echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server healthy after ${i}×5s"
        break
    fi
    sleep 5
done

echo "Running markovian 4k eval..."
uv run python scripts/eval_balrog_babyai.py \
    --mode markovian --use-sessions \
    --max-kv-len 4096 \
    --n 10 --max-turns 20 --max-response-tokens 1024 \
    --ports 8000 --save-traces \
    --output ~/scratch/outputs/babyai_evals/markovian_4k_n10.json

echo "Done. Killing server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
