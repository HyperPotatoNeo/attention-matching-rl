# RG-Mix Baseline Step 600: Compaction vs No-Compaction Eval

**Date**: 2026-03-13
**Model**: Qwen3-4B RL-trained (rg-mix baseline, step 600)
**Benchmark**: rg-mix-env (200 problems, seed=42)

## Motivation

Compare the RL-trained rg-mix baseline checkpoint (step 600) with and without KV cache compaction. This measures the accuracy cost of extreme compression (1024→32) on a model that has been specifically trained on these tasks. The previous compression sweep (2026-03-11) tested on the base model only.

## Setup

**Common:**
- 200 rg-mix problems, seed=42, inverse-difficulty weighting
- temperature=0.6, top_p=0.95
- max response tokens: 8192
- DP=4 (4 vLLM servers, TP=1, one per GPU)

**Compaction eval:**
- max_kv_len=2048 (compact when KV cache reaches this length)
- compact_window=1024 (compress oldest 1024 assistant tokens)
- compacted_size=32 (ratio=0.03125, keep 32 of 1024)
- n_compacts=99 (unlimited)
- suffix queries (real post-RoPE attention queries)

**Baseline eval:**
- Standard vLLM generation, no compaction
- max_tokens=8192

**Task distribution (identical for both):**

| Task | Count | Share |
|------|-------|-------|
| cryptarithm | 54 | 27% |
| zebra_puzzles_7 | 43 | 22% |
| sokoban_hard | 41 | 20% |
| countdown_7 | 32 | 16% |
| arc_1d | 30 | 15% |

## Results

### Overall

| Metric | Compaction (1024→32) | Baseline (no compaction) | Delta |
|--------|---------------------|-------------------------|-------|
| **Accuracy** | **40.0% (80/200)** | **46.5% (93/200)** | **-6.5%** |
| Total tokens | 1,038,822 | 919,154 | +13.0% |
| Avg tokens/problem | 5,194 | 4,596 | +13.0% |
| Wall time | 3,203s (53 min) | 6,685s (111 min) | **-52%** |
| Throughput | 324 tok/s | 137 tok/s | **+2.4x** |
| Time per problem | 16.0s | 33.4s | -52% |

### Per-task accuracy

| Task | Compaction | Baseline | Delta |
|------|-----------|----------|-------|
| arc_1d | **66.7%** (20/30) | 60.0% (18/30) | **+6.7%** |
| countdown_7 | **53.1%** (17/32) | 43.8% (14/32) | **+9.4%** |
| cryptarithm | 24.1% (13/54) | **27.8%** (15/54) | -3.7% |
| sokoban_hard | 26.8% (11/41) | **36.6%** (15/41) | -9.8% |
| zebra_puzzles_7 | 44.2% (19/43) | **72.1%** (31/43) | **-27.9%** |

### Compaction diagnostics (compaction eval only)

| Compactions | Accuracy | Count | Avg Tokens | Avg Ratio |
|-------------|----------|-------|------------|-----------|
| 0 | 95.7% | 23 | 987 | - |
| 1 | 51.7% | 29 | 2,314 | 0.46 |
| 2 | 63.3% | 30 | 3,112 | 0.40 |
| 3 | 69.2% | 13 | 4,109 | 0.38 |
| 4 | 47.4% | 19 | 5,134 | 0.42 |
| 5 | 0.0% | 9 | 6,103 | 0.42 |
| 6 | 50.0% | 6 | 7,059 | 0.43 |
| 7 | 4.2% | 71 | 8,555 | 0.40 |

Avg compaction ratio: 0.407, avg algo time: 0.219s per event, 782 total events.

## Analysis

1. **Compaction costs 6.5% overall accuracy.** This is the cost of extreme 1024→32 compression on an RL-trained model. The model trades accuracy for 2.4x throughput. At 324 tok/s aggregate (4 GPUs), compaction is significantly faster than baseline (137 tok/s) because the KV cache stays small during long generations.

2. **Compaction actually helps on arc_1d (+6.7%) and countdown_7 (+9.4%).** These tasks have structured, step-by-step reasoning where compaction may act as a form of attention focusing — removing noise from old KV entries. The model generates more tokens with compaction (5,194 vs 4,596 avg), suggesting it explores more reasoning paths within the same wall time.

3. **Zebra puzzles suffer most (-27.9%).** Zebra puzzles require maintaining complex constraint tables across long reasoning chains. Compressing the KV cache loses the fine-grained constraint state. The 72.1% → 44.2% drop is the largest across all tasks.

4. **Problems solved quickly (0 compactions) have 95.7% accuracy.** When the model generates <2048 tokens, no compaction occurs. These easy problems are essentially unaffected. The accuracy loss concentrates on harder problems that require many compactions (7 compactions = 4.2% accuracy).

5. **RL training improved both conditions vs base model.** Comparing to the base model compression sweep (2026-03-11): base model at 1024→32 with suffix queries scored 17.0%, while the RL-trained model with compaction scores 40.0% (+23.0%). The baseline (no compaction) went from ~15% (base) to 46.5% (+31.5%).

6. **Mean logprob degrades with more compactions.** From -0.094 (0 compactions) to -0.159 (6 compactions), indicating increasing model uncertainty as information is compressed away. At 7 compactions, logprob slightly recovers to -0.147, likely because those problems hit max tokens and the model is generating confidently-wrong output.

7. **KV cache goes from 2047 → 1055 per compaction event.** The target compression (32/1024 = 0.03125) is achieved correctly — 992 tokens are removed from the 1024-token window, keeping 32 compacted keys. The "ratio" field (avg 0.407) is the algorithm's internal attention fidelity metric, not the compression ratio. After compaction, the KV cache contains: prompt (~100 tokens) + 32 compacted keys + ~920 suffix tokens = ~1055.

## Comparison with base model (no RL)

| Condition | Base Model | RL Step 600 | Delta |
|-----------|-----------|-------------|-------|
| Compaction (1024→32, suffix) | 17.0% | 40.0% | +23.0% |
| No compaction | ~15%* | 46.5% | +31.5% |

*Base model no-compaction accuracy estimated from compression sweep data.

## Conclusion

The RL-trained model (step 600) is significantly better than the base model under both conditions. Compaction at extreme compression (1024→32) costs 6.5% accuracy but provides 2.4x throughput. The accuracy cost is concentrated on zebra puzzles and problems requiring many compaction events. For time-constrained settings, compaction is favorable — especially on tasks with structured step-by-step reasoning (arc_1d, countdown) where it can actually improve accuracy.

## Raw data

- Compaction: `results_step600_compaction_suffix_200.json` (in compaction-rl/)
- Baseline: `results_step600_baseline_200.json` (in compaction-rl/)
- Logs: `$SCRATCH/eval_compact_step600.log`, `$SCRATCH/eval_baseline_step600.log`
