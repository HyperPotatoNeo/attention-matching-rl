#!/bin/bash
# 4-way eval: compaction-trained and baseline models, each with and without compaction.
# Runs INSIDE the container on a compute node with 4 GPUs.
# Compaction settings match fixed_1024q training config:
#   compact_ratio=0.25, compact_window=1024, max_kv_len=2048, max_total_tokens=8192
set -e

cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv pip install reasoning-gym 2>&1 | tail -3

N=300
SEED=42
PORTS="8000,8001,8002,8003"

COMPACT_MODEL="/pscratch/sd/s/siddart2/compaction-rl/outputs/compaction_fixed_1024_q/weights/step_600"
BASELINE_MODEL="/pscratch/sd/s/siddart2/compaction-rl/outputs/rg-mix-baseline/weights/step_600"
COMPACT_CONFIG="configs/compaction/qwen3_4b_serve_tp1.toml"
BASELINE_CONFIG="configs/compaction/qwen3_4b_serve_tp1_baseline.toml"
OUTDIR="/pscratch/sd/s/siddart2/compaction-rl/results_compaction_4way"
mkdir -p "$OUTDIR"

start_servers() {
    local model_dir="$1"
    local config="$2"
    local label="$3"
    echo ""
    echo "================================================================"
    echo "Starting 4 servers for: $label"
    echo "  Model: $model_dir"
    echo "  Config: $config"
    echo "================================================================"
    for gpu in 0 1 2 3; do
        port=$((8000 + gpu))
        echo "  GPU $gpu → port $port"
        CUDA_VISIBLE_DEVICES=$gpu nohup uv run inference \
            @ "$config" \
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
# Phase 1: Compaction-trained model (fixed_1024q step 600)
# ================================================================

# --- 1a: Compaction-trained + WITH compaction ---
start_servers "$COMPACT_MODEL" "$COMPACT_CONFIG" "Compaction-trained (fixed_1024q step 600) — compaction servers"

echo ""
echo "=== [1/4] Compaction-trained + WITH compaction ==="
python scripts/eval_rg_mix.py \
    --mode compaction \
    --n $N --seed $SEED \
    --model "$COMPACT_MODEL" \
    --max-kv-len 2048 \
    --compact-window 1024 \
    --compact-ratio 0.25 \
    --max-total-tokens 8192 \
    --n-compacts 99 \
    --ports $PORTS \
    --batch-size 16 \
    --save-text \
    --output "$OUTDIR/compact_trained_with_compaction_${N}.json"

kill_servers

# --- 1b: Compaction-trained + WITHOUT compaction ---
start_servers "$COMPACT_MODEL" "$BASELINE_CONFIG" "Compaction-trained (fixed_1024q step 600) — baseline servers"

echo ""
echo "=== [2/4] Compaction-trained + WITHOUT compaction ==="
python scripts/eval_rg_mix.py \
    --mode baseline \
    --n $N --seed $SEED \
    --model "$COMPACT_MODEL" \
    --max-tokens-per-segment 8192 \
    --n-compacts 0 \
    --ports $PORTS \
    --batch-size 32 \
    --save-text \
    --output "$OUTDIR/compact_trained_no_compaction_${N}.json"

kill_servers

# ================================================================
# Phase 2: Baseline RL model (rg-mix-baseline step 600)
# ================================================================

# --- 2a: Baseline RL + WITH compaction ---
start_servers "$BASELINE_MODEL" "$COMPACT_CONFIG" "Baseline RL (step 600) — compaction servers"

echo ""
echo "=== [3/4] Baseline RL + WITH compaction ==="
python scripts/eval_rg_mix.py \
    --mode compaction \
    --n $N --seed $SEED \
    --model "$BASELINE_MODEL" \
    --max-kv-len 2048 \
    --compact-window 1024 \
    --compact-ratio 0.25 \
    --max-total-tokens 8192 \
    --n-compacts 99 \
    --ports $PORTS \
    --batch-size 16 \
    --save-text \
    --output "$OUTDIR/baseline_rl_with_compaction_${N}.json"

kill_servers

# --- 2b: Baseline RL + WITHOUT compaction ---
start_servers "$BASELINE_MODEL" "$BASELINE_CONFIG" "Baseline RL (step 600) — baseline servers"

echo ""
echo "=== [4/4] Baseline RL + WITHOUT compaction ==="
python scripts/eval_rg_mix.py \
    --mode baseline \
    --n $N --seed $SEED \
    --model "$BASELINE_MODEL" \
    --max-tokens-per-segment 8192 \
    --n-compacts 0 \
    --ports $PORTS \
    --batch-size 32 \
    --save-text \
    --output "$OUTDIR/baseline_rl_no_compaction_${N}.json"

kill_servers

# ================================================================
# Summary
# ================================================================
echo ""
echo "================================================================"
echo "ALL 4 EVALS COMPLETE"
echo "================================================================"
echo "Results:"
for f in "$OUTDIR"/*_${N}.json; do
    acc=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['accuracy']:.1%}  avg_tok={d['avg_tokens']:.0f}\")")
    echo "  $(basename $f): $acc"
done
