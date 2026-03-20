#!/bin/bash
#SBATCH --job-name=markovian-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=1:00:00
#SBATCH --output=../scratch/outputs/compaction_markovian_mila/test_eval_%j.log
#SBATCH --error=../scratch/outputs/compaction_markovian_mila/test_eval_%j.log

export PROJECT_DIR=/home/mila/e/emiliano.penaloza/rsa-compaction
cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate

echo "=== Starting inference server ==="
uv run inference @ configs/compaction/qwen3_4b_serve_tp1.toml --server.port 8000 &
SERVER_PID=$!

echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready after $((i*5))s"
        break
    fi
    sleep 5
done

echo ""
echo "=== Running markovian eval (n=100) ==="
uv run python scripts/eval_rg_mix.py \
    --mode markovian \
    --n 100 \
    --max-kv-len 2048 \
    --max-total-tokens 8192 \
    --n-compacts 99 \
    --compact-window 1024 \
    --compact-ratio 0.25

echo ""
echo "=== Running baseline eval (n=100) for comparison ==="
uv run python scripts/eval_rg_mix.py \
    --mode baseline \
    --n 100 \
    --server-url http://localhost:8000

echo ""
echo "=== Done ==="
kill $SERVER_PID 2>/dev/null
