#!/bin/bash
# Eval 4-way: base Qwen3-4B and step-600 baseline checkpoint,
# each with and without budget injection. 300 samples, seed=42, 8192 response tokens.
# Runs INSIDE the container on a compute node with 4 GPUs.
set -e

cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv pip install reasoning-gym 2>&1 | tail -3

N=300
SEED=42
MAX_TOKENS=8192
BATCH_SIZE=16
PORTS="8000,8001,8002,8003"

BASE_MODEL="Qwen/Qwen3-4B"
STEP600_MODEL="/pscratch/sd/s/siddart2/compaction-rl/outputs/rg-mix-baseline/weights/step_600"
CONFIG="configs/compaction/qwen3_4b_serve_tp1.toml"
OUTDIR="/pscratch/sd/s/siddart2/compaction-rl/results_4way"
mkdir -p "$OUTDIR"

start_servers() {
    local model_dir="$1"
    local label="$2"
    echo ""
    echo "================================================================"
    echo "Starting 4 servers for: $label"
    echo "  Model: $model_dir"
    echo "================================================================"
    for gpu in 0 1 2 3; do
        port=$((8000 + gpu))
        echo "  GPU $gpu → port $port"
        CUDA_VISIBLE_DEVICES=$gpu nohup uv run inference \
            @ "$CONFIG" \
            --model.name "$model_dir" \
            --server.port $port \
            > /tmp/server_gpu${gpu}.log 2>&1 &
    done

    echo "Waiting for servers..."
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
}

kill_servers() {
    echo "Killing all inference servers..."
    pkill -f "uv run inference" 2>/dev/null || true
    sleep 3
    pkill -9 -f "uv run inference" 2>/dev/null || true
    pkill -f "vllm" 2>/dev/null || true
    sleep 2
    echo "Servers stopped."
}

# ================================================================
# Phase 1: Base Qwen3-4B
# ================================================================
start_servers "$BASE_MODEL" "Base Qwen3-4B"

echo ""
echo "=== [1/4] Base Qwen3-4B — NO budget injection ==="
python scripts/eval_rg_mix.py \
    --mode baseline \
    --n $N --seed $SEED \
    --model "$BASE_MODEL" \
    --max-tokens-per-segment $MAX_TOKENS \
    --n-compacts 0 \
    --ports $PORTS \
    --batch-size $BATCH_SIZE \
    --output "$OUTDIR/base_no_budget_${N}.json"

echo ""
echo "=== [2/4] Base Qwen3-4B — WITH budget injection ==="
python scripts/eval_rg_mix.py \
    --mode inject \
    --n $N --seed $SEED \
    --model "$BASE_MODEL" \
    --max-tokens-per-segment $MAX_TOKENS \
    --n-compacts 0 \
    --inject-budget-every 2048 \
    --ports $PORTS \
    --batch-size $BATCH_SIZE \
    --output "$OUTDIR/base_budget_${N}.json"

kill_servers

# ================================================================
# Phase 2: Baseline RL step 600
# ================================================================
start_servers "$STEP600_MODEL" "Baseline RL step 600"

echo ""
echo "=== [3/4] Baseline RL step 600 — NO budget injection ==="
python scripts/eval_rg_mix.py \
    --mode baseline \
    --n $N --seed $SEED \
    --model "$STEP600_MODEL" \
    --max-tokens-per-segment $MAX_TOKENS \
    --n-compacts 0 \
    --ports $PORTS \
    --batch-size $BATCH_SIZE \
    --output "$OUTDIR/step600_no_budget_${N}.json"

echo ""
echo "=== [4/4] Baseline RL step 600 — WITH budget injection ==="
python scripts/eval_rg_mix.py \
    --mode inject \
    --n $N --seed $SEED \
    --model "$STEP600_MODEL" \
    --max-tokens-per-segment $MAX_TOKENS \
    --n-compacts 0 \
    --inject-budget-every 2048 \
    --ports $PORTS \
    --batch-size $BATCH_SIZE \
    --output "$OUTDIR/step600_budget_${N}.json"

kill_servers

echo ""
echo "================================================================"
echo "ALL 4 EVALS COMPLETE"
echo "================================================================"
echo "Results:"
for f in "$OUTDIR"/*_${N}.json; do
    acc=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['accuracy']:.1%}\")")
    echo "  $(basename $f): $acc"
done
