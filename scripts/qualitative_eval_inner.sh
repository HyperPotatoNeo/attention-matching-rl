#!/bin/bash
set -e
export PATH="/pscratch/sd/s/siddart2/.local/bin:$PATH"
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache
cd /pscratch/sd/s/siddart2/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME

BASELINE_MODEL=outputs/rg-mix-baseline/merged_step_600
SUFFIX_MODEL=outputs/suffix_seq_7500/merged_step_600
N=10

# --- 1. Baseline RL model, no compaction ---
echo "=== Starting baseline server (baseline RL model) ==="
CUDA_VISIBLE_DEVICES=0 nohup inference \
    @ configs/compaction/qwen3_4b_serve_tp1_baseline.toml \
    --model.name "$BASELINE_MODEL" --server.port 8000 \
    > /tmp/server0.log 2>&1 &
for i in $(seq 1 600); do
    curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q 200 && break
    [ $i -eq 600 ] && { echo "FAIL"; cat /tmp/server0.log; exit 1; }
    sleep 1
done
echo "Server ready."

python scripts/qualitative_sample.py --mode baseline --model "$BASELINE_MODEL" \
    --n $N --port 8000 --output qualitative_baseline_rl_nocompact.json
kill %1 2>/dev/null; wait 2>/dev/null

# --- 2. Baseline RL model, WITH compaction (random queries, 1024->256) ---
echo "=== Starting compaction server (baseline RL model) ==="
CUDA_VISIBLE_DEVICES=0 nohup inference \
    @ configs/compaction/qwen3_4b_serve_tp1.toml \
    --model.name "$BASELINE_MODEL" --server.port 8000 \
    > /tmp/server0.log 2>&1 &
for i in $(seq 1 600); do
    curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q 200 && break
    [ $i -eq 600 ] && { echo "FAIL"; cat /tmp/server0.log; exit 1; }
    sleep 1
done
echo "Server ready."

python scripts/qualitative_sample.py --mode compaction --model "$BASELINE_MODEL" \
    --n $N --port 8000 --compact-ratio 0.25 --compact-window 1024 \
    --output qualitative_baseline_rl_compact.json
kill %1 2>/dev/null; wait 2>/dev/null

# --- 3. Suffix RL model, no compaction ---
echo "=== Starting baseline server (suffix RL model) ==="
CUDA_VISIBLE_DEVICES=0 nohup inference \
    @ configs/compaction/qwen3_4b_serve_tp1_baseline.toml \
    --model.name "$SUFFIX_MODEL" --server.port 8000 \
    > /tmp/server0.log 2>&1 &
for i in $(seq 1 600); do
    curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q 200 && break
    [ $i -eq 600 ] && { echo "FAIL"; cat /tmp/server0.log; exit 1; }
    sleep 1
done
echo "Server ready."

python scripts/qualitative_sample.py --mode baseline --model "$SUFFIX_MODEL" \
    --n $N --port 8000 --output qualitative_suffix_rl_nocompact.json
kill %1 2>/dev/null; wait 2>/dev/null

# --- 4. Suffix RL model, WITH compaction (suffix queries, 1024->32) ---
echo "=== Starting compaction server (suffix RL model) ==="
CUDA_VISIBLE_DEVICES=0 nohup inference \
    @ configs/compaction/qwen3_4b_serve_tp1.toml \
    --model.name "$SUFFIX_MODEL" --server.port 8000 \
    > /tmp/server0.log 2>&1 &
for i in $(seq 1 600); do
    curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q 200 && break
    [ $i -eq 600 ] && { echo "FAIL"; cat /tmp/server0.log; exit 1; }
    sleep 1
done
echo "Server ready."

python scripts/qualitative_sample.py --mode compaction --model "$SUFFIX_MODEL" \
    --n $N --port 8000 --compact-ratio 0.03125 --compact-window 1024 \
    --use-suffix-queries --output qualitative_suffix_rl_compact.json

echo "=== All qualitative samples done ==="
