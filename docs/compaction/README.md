# KV Cache Compaction for Long-Context RL

Mid-generation KV cache compression using Attention Matching, implemented as a vLLM worker extension. Generates text in segments, compacting the KV cache between segments to allow longer effective context within fixed memory.

## Quick Start

### 1. Allocate a GPU node

```bash
salloc -A m5017 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 --gpus-per-node 4
```

### 2. Launch compaction servers (DP=4, one per GPU)

```bash
# On the compute node, inside the container:
cd $SCRATCH/compaction-rl
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME
bash scripts/start_4servers.sh
```

This starts 4 independent TP=1 vLLM servers on ports 8000-8003, each with the `CompactionWorker` extension that exposes `/compact_generate`.

### 3. Run evaluation

```bash
# Compaction eval (DP=4, 100 problems)
python scripts/eval_rg_mix.py \
    --mode compaction --n 100 \
    --max-tokens-per-segment 2048 \
    --n-compacts 3 \
    --compact-ratio 0.25 \
    --compact-window 512 \
    --output results.json

# Baseline eval (no compaction, single TP=4 server)
python scripts/eval_rg_mix.py \
    --mode baseline --n 100 \
    --server-url http://localhost:8000
```

## API

### POST `/compact_generate`

Generate text with mid-sequence KV cache compaction. Requests are transparently batched
(up to B=8) by the server for higher throughput. Supports two modes:

1. **Fixed segment mode** (default): Generate `max_tokens_per_segment` tokens, compact, repeat
2. **KV budget mode** (set `max_kv_len`): Generate until KV fills, compact `compact_window` tokens, continue until `max_total_tokens`

#### Fixed segment mode

```json
{
    "prompt_ids": [1, 2, 3],
    "max_seq_len": 8192,
    "max_tokens_per_segment": 2048,
    "n_compacts": 3,
    "compact_target_ratio": 0.25,
    "compact_window": 512,
    "temperature": 0.6,
    "top_p": 0.95
}
```

#### KV budget mode

```json
{
    "prompt_ids": [1, 2, 3],
    "max_kv_len": 2048,
    "max_total_tokens": 4096,
    "compact_target_ratio": 0.25,
    "compact_window": 512,
    "n_compacts": 99,
    "compute_beta": true,
    "temperature": 0.6,
    "top_p": 0.95
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_ids` | `list[int]` | required | Tokenized prompt |
| `max_seq_len` | `int` | 8192 | Maximum total sequence length (fixed segment mode) |
| `max_tokens_per_segment` | `int\|null` | auto | Tokens per segment before compaction |
| `n_compacts` | `int` | 3 | Maximum compaction events |
| `compact_target_ratio` | `float` | 0.3 | Fraction of compacted window to keep |
| `compact_window` | `int\|null` | null | Compress only first N assistant tokens (null = all) |
| `temperature` | `float` | 0.7 | Sampling temperature |
| `top_p` | `float` | 0.95 | Top-p sampling |
| `max_kv_len` | `int\|null` | null | KV budget: compact when cache reaches this length |
| `max_total_tokens` | `int\|null` | null | KV budget: stop after this many completion tokens |
| `compute_beta` | `bool` | false | Compute NNLS beta bias for partition function correction |
| `use_suffix_queries` | `bool` | true | Use suffix token attention queries instead of random probes |
| `compaction_mode` | `str` | `"attention_matching"` | `"attention_matching"` or `"markovian"` (hard-delete window) |

**Response:**

```json
{
    "all_token_ids": [101, 203],
    "all_logprobs": [-0.12, -0.34],
    "final_text": "decoded text...",
    "diagnostics": {
        "prompt_len": 45,
        "total_tokens": 4096,
        "total_time": 67.3,
        "compaction_events": [{
            "segment": 0,
            "assistant_tokens_before": 1848,
            "assistant_tokens_after": 1464,
            "prefix_compacted": 512,
            "prefix_after": 128,
            "suffix_preserved": 1336,
            "ratio": 0.79,
            "algo_time": 0.41
        }],
        "segment_boundaries": [1848, 2232, 2616]
    }
}
```

## Session API

Persistent KV-cache sessions for multi-turn generation. Each session retains its KV cache
across steps, enabling compaction strategies that are aware of turn boundaries.

### POST `/compact_session/create`

Create a session and generate the first response.

```json
{
    "session_id": "my-session-0",
    "prompt_ids": [1, 2, 3],
    "max_kv_len": 8192,
    "compact_target_ratio": 0.5,
    "compact_window": null,
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.95,
    "compaction_mode": "attention_matching",
    "n_max_turns": 5,
    "n_preserved_turns": 3
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` | required | Unique session identifier |
| `prompt_ids` | `list[int]` | required | Tokenized system prompt + first user message |
| `max_kv_len` | `int` | required | KV cache capacity (tokens). Triggers KV-budget compaction when `n_max_turns=-1`. |
| `compact_target_ratio` | `float` | 0.3 | Fraction of the compacted window to keep |
| `compact_window` | `int\|null` | null | Compress only first N tokens of history (null = all). Ignored when `n_max_turns >= 0`. |
| `max_tokens` | `int` | 512 | Max tokens per response |
| `temperature` | `float` | 0.7 | Sampling temperature |
| `top_p` | `float` | 0.95 | Top-p sampling |
| `compaction_mode` | `str` | `"attention_matching"` | `"attention_matching"` or `"markovian"` |
| `n_max_turns` | `int` | -1 | Turn-based compaction: fire compaction when uncompacted turns reach this limit. `-1` = disabled (KV-budget mode). |
| `n_preserved_turns` | `int` | 0 | Turn-based compaction: keep last N turns verbatim after compaction fires. |

### POST `/compact_session/step`

Continue a session with a new user message.

```json
{
    "session_id": "my-session-0",
    "new_token_ids": [42, 67, 89],
    "max_tokens": 512
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` | required | Session ID from `/compact_session/create` |
| `new_token_ids` | `list[int]` | required | Tokenized user message (boundary token prepended automatically) |
| `max_tokens` | `int` | 512 | Max tokens for this response |
| `temperature` | `float` | inherited | Override sampling temperature for this step |
| `top_p` | `float` | inherited | Override top-p for this step |

**Response (both create and step):**

```json
{
    "token_ids": [101, 203, 55],
    "text": "decoded response...",
    "compaction_events": [
        {"kv_len_before": 4096, "kv_len_after": 2200}
    ]
}
```

### DELETE `/compact_session/{session_id}`

Release KV blocks and remove session state.

---

## Compaction Modes

### Attention Matching (default)

Compresses the KV window using Attention Matching: select top-k keys by importance (C1),
solve least-squares for replacement values (C2). The compacted cache is
`[prompt | C1/C2 | suffix]`. Optionally uses NNLS beta for partition function correction.

### Markovian (Delethink)

Hard-deletes the compaction window entirely, keeping only `[prompt | suffix]`. No
compressed keys, no importance scoring, no C2 solve. Set `compaction_mode = "markovian"`
in both the env args and trainer config (auto-synced via model validator).

This is useful for training "Markovian thinkers" that learn to reason in segments where
earlier thinking can be discarded. The model learns to produce self-contained reasoning
within each segment.

```bash
# Inference request
curl -X POST http://localhost:8000/compact_generate -d '{
    "prompt_ids": [1, 2, 3],
    "max_kv_len": 2048,
    "max_total_tokens": 8192,
    "compact_window": 1024,
    "n_compacts": 99,
    "compaction_mode": "markovian"
}'

# Training config
[trainer]
compaction_mode = "markovian"

[[orchestrator.env]]
args = { ..., compaction_mode = "markovian", use_suffix_queries = false }
```

### Turn-Based Compaction (`n_max_turns` + `n_preserved_turns`)

Fires compaction **after a turn completes and the uncompacted turn count reaches
`n_max_turns`**, not reactively when the KV fills. The last `n_preserved_turns` turns
are kept verbatim; everything older is compacted. Compaction then fires again every
`n_max_turns - n_preserved_turns` turns.

Example — `n_max_turns=5, n_preserved_turns=3`: turns accumulate until turn 5 → compact
oldest 2, keep turns 3,4,5 → then compact again at turns 7, 9, ... (every 2 turns).

The importance queries come from the **most recent assistant response's K-vectors** — a
lookahead that grounds importance scoring in what the model actually attended to, rather
than random Gaussian probes.

KV structure after each compaction:
```
[system prompt] | [compacted turns 1..N-n_preserved] | [turn_{N-n+1}]…[turn_N]
```

Set `n_max_turns >= 0` on the session endpoint. Default is `-1` (disabled — old
KV-budget behavior unchanged).

```python
# Session API
client.post("/compact_session/create", json={
    "session_id": "...",
    "prompt_ids": [...],
    "max_kv_len": 8192,
    "compact_target_ratio": 0.5,
    "n_max_turns": 5,        # fire compaction when 5 turns have accumulated
    "n_preserved_turns": 3,  # keep the 3 most recent turns verbatim
})
```

```bash
# Eval script
python scripts/eval_balrog_babyai.py \
    --mode compaction --use-sessions \
    --max-kv-len 8192 --compact-ratio 0.5 \
    --n-max-turns 5 --n-preserved-turns 3
```

**Comparison of compaction triggers:**

| Mode | Trigger | Queries | Protects |
|------|---------|---------|---------|
| KV-budget | `seq_len >= max_kv_len` (mid-generation) | random Gaussian | nothing |
| Turn-based | after N turns accumulate | assistant K-vectors (lookahead) | last `n_preserved_turns` turns |
| Markovian | same trigger as KV-budget | n/a (hard delete) | nothing |

## BabyAI Evaluation

Multi-turn grid-world benchmark using BabyAI (MiniGrid). The model receives text
descriptions of a grid-world state and must issue actions to complete navigation and
manipulation tasks. Evaluates the model's ability to maintain context over many turns.

### Installation

`minigrid` and `gymnasium` are included in the project dependencies:

```bash
uv sync
```

No additional setup is needed — the eval script uses MiniGrid directly and does not
require the full BALROG framework (no cmake, no NLE).

### Tasks

Eight BabyAI tasks spanning three difficulty levels:

| Task | Difficulty | Description |
|------|-----------|-------------|
| `GoToObj` | easy | Navigate to a named object |
| `GoToLocal` | easy | Navigate to a local object |
| `PickupLoc` | easy | Pick up an object at a location |
| `Open` | medium | Open a door |
| `PutNextLocal` | medium | Place an object next to another |
| `GoTo` | medium | Navigate to a goal position |
| `Unlock` | hard | Find key and unlock door |
| `UnlockLocal` | hard | Find and use local key |

### Running Evals

```bash
# Baseline — standard generation, no compaction
python scripts/eval_balrog_babyai.py --mode baseline --n 10

# KV-budget compaction — compacts reactively when cache fills
python scripts/eval_balrog_babyai.py --mode compaction --n 10 \
    --max-kv-len 4096 --compact-ratio 0.5

# Turn-based compaction — compacts when 5 turns accumulate, keeps 3 most recent
python scripts/eval_balrog_babyai.py --mode compaction --use-sessions --n 10 \
    --max-kv-len 8192 --compact-ratio 0.5 --n-max-turns 5 --n-preserved-turns 3

# Turn-based markovian — hard-deletes history when 3 turns accumulate, keeps 1
python scripts/eval_balrog_babyai.py --mode markovian --use-sessions --n 10 \
    --max-kv-len 8192 --n-max-turns 3 --n-preserved-turns 1

# Specific tasks only
python scripts/eval_balrog_babyai.py --mode baseline --n 10 \
    --envs GoToObj GoToLocal Unlock
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | required | `baseline`, `compaction`, or `markovian` |
| `--n` | 10 | Episodes per task |
| `--max-kv-len` | none | KV cache size limit (tokens) |
| `--compact-ratio` | 0.25 | Fraction to keep after compaction |
| `--use-sessions` | off | Use persistent KV session API |
| `--n-max-turns` | -1 | Turn-based: trigger compaction when uncompacted turns reach this count (-1 = KV-budget mode) |
| `--n-preserved-turns` | 0 | Turn-based: keep last N turns verbatim after compaction fires |
| `--max-turns` | 64 | Max turns per episode |
| `--max-response-tokens` | 512 | Max tokens per model response |
| `--ports` | `8000,...,8003` | Server ports (comma-separated) |
| `--output` | none | Save results to JSON |
| `--save-traces` | off | Include full action traces in output |
| `--plot` | off | Generate reward/token plots |

### Notes

- `--use-sessions` is required for turn-based compaction (`--n-max-turns >= 0`).
  KV-budget compaction works without sessions (`--use-sessions` optional).
- When `--n-max-turns >= 0`, `--compact-window` has no effect and a warning is logged.
- The model (`Qwen/Qwen3-4B-Instruct-2507`) is served via `configs/compaction/qwen3_4b_serve_tp1.toml`.

---

## Training

Default: 2-node, 4+4 layout (4 inference + 4 trainer GPUs). Uses suffix queries for
key importance scoring with forced indices passed from inference to trainer, ensuring
identical key selection despite numerical differences between vLLM and HuggingFace.

```bash
sbatch -A m5017 -C "gpu&hbm80g" --qos=premium --time 48:00:00 --gpus-per-node 4 --nodes=2 ~/compaction_suffix_queries.sh
```

Architecture:
- **Node 1** (4 GPUs): 4 TP=1 compaction inference servers (ports 8000-8003)
- **Node 2** (4 GPUs): FSDP2 trainer (4 GPUs) + orchestrator (CPU)
- Weight broadcast via Lustre filesystem
- Containers use `--net=host` for cross-node communication

Training parameters:
- Full fine-tune (no LoRA), lr=1e-6, AdamW with CPU offload
- KV budget: max_kv_len=2048, compact_window=1024, ratio=0.25, max_total_tokens=8192
- batch_size=256, rollouts_per_example=8, seq_len=9216
- Suffix queries + forced indices (default), no beta
- Checkpoints every 10 steps, auto-resume with `resume_step = -1`

## Configs

| Config | Purpose |
|--------|---------|
| `qwen3_4b_fullft_suffix_queries.toml` | **Default** — 4+4 layout, suffix queries + forced indices, no beta |
| `qwen3_4b_fullft_determ_nobeta.toml` | Random queries, deterministic compaction, no beta |
| `qwen3_4b_fullft_determ_suffix.toml` | Deterministic random queries + prompt keys |
| `qwen3_4b_fullft_fixed_1024q.toml` | 1024 random queries |
| `qwen3_4b_fullft_nobeta.toml` | 4+4 layout, no beta (pre-deterministic, legacy) |
| `qwen3_4b_beta_test.toml` | Beta attention test (compute_beta=true) |
| `qwen3_4b_fullft_baseline.toml` | Baseline training (no compaction) |
| `qwen3_4b_balrog_babyai.toml` | BabyAI RL training config (balrog-bench env) |
| `qwen3_4b_markovian_test.toml` | Markovian mode — Qwen3-4B, 50 steps |
| `qwen3_06b_markovian_test.toml` | Markovian mode — Qwen3-0.6B, fast E2E test |
| `qwen3_4b_serve_tp1.toml` | TP=1 inference server with compaction (4B Instruct) |
| `qwen3_4b_baseline_tp1.toml` | TP=1 baseline server, no compaction (4B) |
| `qwen3_06b_serve_tp1.toml` | TP=1 compaction server (0.6B) |

## Scripts

| Script | Purpose |
|--------|---------|
| `~/compaction_suffix_queries.sh` | **Default launch** — 2-node sbatch (suffix queries + forced indices) |
| `scripts/start_4servers.sh` | 4 TP=1 servers on ports 8000-8003 |
| `scripts/eval_rg_mix.py` | Evaluate on rg-mix-env (single-turn, math/reasoning) |
| `scripts/eval_balrog_babyai.py` | BabyAI multi-turn eval (baseline, compaction, markovian) |
| `scripts/eval_aime_rsa.py` | AIME benchmark for RSA vs baseline |
| `scripts/viz_babyai.py` | Visualize BabyAI episode traces |
| `scripts/viz_results.py` | Plot eval results across runs |

## Prompt Keys (Default Behavior)

The compaction algorithm scores key importance using full-context attention: the softmax
denominator includes **all** keys (prompt + assistant), not just assistant keys. This means
window keys that are redundant with prompt content score lower and are more likely to be
evicted. No flag needed — this is the default in `compact_kv()`.

## Compaction Indices Passing

During RL training, the trainer must replay compaction identically to inference. By default,
deterministic seeded random queries ensure this. For suffix queries (which produce different
Q vectors in vLLM vs HuggingFace due to numerical differences), the inference worker returns
the exact top-k indices in `diagnostics.compaction_indices`. The trainer passes these as
`forced_indices` to `compact_kv()`, skipping importance scoring entirely. C2 values are
still recomputed from the trainer's KV cache for correct gradients.

Data flow: `worker.compact_generate()` → `env.py` (extras) → `trajectories.py` →
`TrainingSample` → `MicroBatch` → `segmented_forward(compaction_indices=...)`.

## Suffix Queries

By default, compaction uses random Gaussian probes to score key importance. With
`use_suffix_queries=true`, the server instead extracts real query vectors from the
suffix tokens that already attend to every key in the compaction window. This produces
importance scores grounded in the model's actual attention patterns rather than random
projections.

### How it works

1. Before compaction, a prefill pass re-runs the suffix tokens through the model
2. Forward pre-hooks on vLLM's inner `Attention` layers capture the query tensors
   (post-projection, post-RoPE) at every layer
3. Queries are grouped by GQA KV head and used as probes in the Attention Matching algorithm

This is model-agnostic — hooks target `vllm.model_executor.layers.attention.Attention`,
which all architectures use.

### Usage

```bash
# API: add use_suffix_queries to request body
curl -X POST http://localhost:8000/compact_generate -d '{
    "prompt_ids": [1, 2, 3],
    "max_kv_len": 2048,
    "max_total_tokens": 8192,
    "n_compacts": 99,
    "compact_target_ratio": 0.25,
    "compact_window": 1024,
    "use_suffix_queries": true
}'

# Eval script
python scripts/eval_rg_mix.py \
    --mode compaction --n 100 --batch-size 4 \
    --max-kv-len 2048 --max-total-tokens 8192 \
    --n-compacts 99 --compact-ratio 0.25 --compact-window 1024 \
    --use-suffix-queries
```

## Eval Results (Qwen3-4B base, rg-mix-env)

### Compression sweep (200 problems, max_kv_len=2048, max_total_tokens=8192, window=1024)

| Compression | Random Queries | Suffix Queries | Delta |
|-------------|---------------|----------------|-------|
| 1024 → 256 (25%) | 14.5% | 14.5% | 0.0% |
| 1024 → 128 (12.5%) | 13.0% | 13.0% | 0.0% |
| 1024 → 64 (6.25%) | 13.5% | 14.0% | +0.5% |
| 1024 → 32 (3.1%) | 14.0% | **17.0%** | **+3.0%** |

Suffix queries provide increasing benefit at higher compression. At 97% compression
(1024→32), suffix queries give +3% accuracy, driven by reasoning tasks (countdown +9.4%,
sokoban +7.3%). At mild compression the two modes are equivalent.

### Baseline comparison (100 problems)

| Configuration | Accuracy | Avg Tokens | Throughput |
|---------------|----------|------------|------------|
| Baseline (8192 tokens, no compaction) | 15.0% | ~5000 | 213 tok/s |
| KV budget (w=1024, ratio=0.25, 8k total) | 16.0% | ~7855 | 440 tok/s |

See `reports/compression_sweep.md` for the full experiment report.
