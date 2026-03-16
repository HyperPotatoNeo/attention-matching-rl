#!/bin/bash
# Qualitative budget injection samples: specific problem indices, both models.
# Runs INSIDE container on compute node with 4 GPUs.
set -e
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv pip install reasoning-gym 2>&1 | tail -3

# Interesting indices from the 300-sample eval:
#   0: zebra - base 8192->budget 2311 (budget forces conciseness)
#  21: zebra - base 8192->budget 813 (dramatic compression)
#  36: cryptarithm - budget helps base, hurts step600
#  68: countdown - budget hurts base, helps step600 (crossover!)
# 106: arc_1d - budget helps both models
# 172: countdown - base solves in 1423 tok, budget bloats to 8291
#  97: zebra - budget helps base, hurts step600
#  10: sokoban - base passes, budget fails; step600 fails, budget passes
INDICES="0,21,36,68,106,172,97,10"

BASE_MODEL="Qwen/Qwen3-4B"
STEP600_MODEL="/pscratch/sd/s/siddart2/compaction-rl/outputs/rg-mix-baseline/weights/step_600"
CONFIG="configs/compaction/qwen3_4b_serve_tp1.toml"
OUTDIR="/pscratch/sd/s/siddart2/compaction-rl/results_4way"

start_server() {
    local model_dir="$1"
    local label="$2"
    echo ""
    echo "================================================================"
    echo "Starting server for: $label"
    echo "================================================================"
    CUDA_VISIBLE_DEVICES=0 nohup uv run inference \
        @ "$CONFIG" \
        --model.name "$model_dir" \
        --server.port 8000 \
        > /tmp/server_qual.log 2>&1 &

    for i in $(seq 1 300); do
        if curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null | grep -q 200; then
            echo "  Server ready after ${i}s"
            break
        fi
        if [ $i -eq 300 ]; then
            echo "  ERROR: Server failed to start"
            cat /tmp/server_qual.log
            exit 1
        fi
        sleep 1
    done
}

kill_server() {
    pkill -f "uv run inference" 2>/dev/null || true
    sleep 3
    pkill -9 -f "uv run inference" 2>/dev/null || true
    pkill -f "vllm" 2>/dev/null || true
    sleep 2
}

# Phase 1: Base model
start_server "$BASE_MODEL" "Base Qwen3-4B"
python scripts/qualitative_budget_injection.py \
    --model "$BASE_MODEL" \
    --indices "$INDICES" \
    --output "$OUTDIR/qualitative_base.json"
kill_server

# Phase 2: Step 600
start_server "$STEP600_MODEL" "Baseline RL step 600"
python scripts/qualitative_budget_injection.py \
    --model "$STEP600_MODEL" \
    --indices "$INDICES" \
    --output "$OUTDIR/qualitative_step600.json"
kill_server

echo ""
echo "Done. Results in $OUTDIR/qualitative_base.json and qualitative_step600.json"
