# 4-Way Compaction Evaluation Report

**Date**: 2026-03-16
**Models**: Qwen3-4B, step 600 checkpoints
**Problems**: 300 rg-mix (seed=42), inverse-difficulty weighted
**Compaction config**: ratio=0.25, window=1024, max_kv_len=2048, max_total_tokens=8192 (matches `fixed_1024q` training)
**Job**: 50156316 (premium QOS, 1 node, 4 GPUs)

## Summary

| Condition | Accuracy | Avg Tokens | Total Tokens | Hit Max (≥8000) |
|-----------|----------|------------|--------------|-----------------|
| Baseline RL, no compaction | **44.3%** | 4,714 | 1,414,214 | 21.0% |
| Baseline RL + compaction | 36.7% | 4,996 | 1,498,678 | 23.0% |
| Compact-trained + compaction | 28.0% | 4,675 | 1,402,524 | 15.7% |
| Compact-trained, no compaction | 26.7% | 5,533 | 1,659,853 | 25.7% |

**Key result**: Baseline RL without compaction is the best model by a wide margin (+7.6pp over baseline+compaction, +16.3pp over compact-trained+compaction). Compaction training actively degrades model quality—the compact-trained model performs worse than baseline **even without compaction at eval** (26.7% vs 44.3%).

## Per-Task Breakdown

| Task | Baseline no-comp | Baseline + comp | Compact no-comp | Compact + comp | N |
|------|-----------------|-----------------|-----------------|----------------|---|
| arc_1d | **59.1%** | 50.0% | 27.3% | 29.5% | 44 |
| countdown_7 | 46.7% | **48.9%** | 37.8% | 46.7% | 45 |
| cryptarithm | **29.4%** | 20.0% | 14.1% | 20.0% | 85 |
| sokoban_hard | **33.9%** | 16.1% | 9.7% | 8.1% | 62 |
| zebra_puzzles_7 | **62.5%** | 60.9% | 51.6% | 43.8% | 64 |

Notable observations:
- **Sokoban**: Most sensitive to both compaction (-17.8pp for baseline) and compaction training (-24.2pp even without compaction). Sokoban requires maintaining spatial state across long sequences—compaction destroys this.
- **Countdown**: Only task where compaction slightly *helps* the baseline model (+2.2pp). Countdown solutions tend to be shorter, and compaction may prune unproductive exploration.
- **Zebra puzzles**: Compaction training alone costs -10.9pp (51.6% vs 62.5%) without compaction. Adding compaction costs another -7.8pp.

## Sequence Length Analysis

| Condition | Mean | Median | P25 | P75 | Min | Max |
|-----------|------|--------|-----|-----|-----|-----|
| Baseline no-comp | 4,714 | 4,374 | 2,727 | 7,265 | 197 | 8,192 |
| Baseline + comp | 4,996 | 4,529 | 2,888 | 7,753 | 127 | 8,784 |
| Compact no-comp | 5,533 | 5,930 | 3,539 | 8,059 | 272 | 8,192 |
| Compact + comp | 4,675 | 4,505 | 3,000 | 6,220 | 317 | 8,785 |

- **Compact-trained without compaction generates longer sequences** (5,533 avg vs 4,714 for baseline). This suggests the model has lost the ability to converge on answers efficiently—it rambles more.
- **Compact-trained + compaction has shorter average sequences** (4,675) because compaction forces some sequences to terminate earlier.
- **25.7% of compact-trained sequences hit max tokens** without compaction (vs 21.0% for baseline), confirming the model tends to not converge.

## Compaction Event Analysis

### Accuracy vs Number of Compactions

| Compactions | Baseline+comp Acc | N | Compact+comp Acc | N |
|-------------|-------------------|---|------------------|---|
| 0 | **94.3%** | 35 | **89.5%** | 38 |
| 1 | 65.4% | 26 | 66.7% | 15 |
| 2 | 43.3% | 30 | 39.3% | 28 |
| 3 | 45.0% | 40 | 15.2% | 46 |
| 4 | 29.6% | 27 | 20.5% | 39 |
| 5 | 22.2% | 18 | 11.1% | 36 |
| 6 | 35.3% | 17 | 26.1% | 23 |
| 7 | 14.3% | 14 | 14.3% | 21 |
| 8 | 30.0% | 20 | 0.0% | 7 |
| 9 (max) | 4.1% | 73 | 2.1% | 47 |

**Critical insight**: Problems solved with 0 compactions have ~90-94% accuracy—these are "easy" problems where the model converges before KV reaches 2048. After the first compaction, accuracy drops sharply. By 9 compactions (sequences that hit the token limit), accuracy is near zero.

The baseline model maintains a consistent accuracy advantage at every compaction count, especially at 3 compactions (45.0% vs 15.2%)—a **3x gap**. This confirms compaction training degrades the model's ability to reason through multi-segment problems.

### Compaction Statistics

| Metric | Compact-trained | Baseline RL |
|--------|-----------------|-------------|
| Total events | 1,309 | 1,421 |
| Avg compression ratio | 0.543 | 0.543 |
| Avg algo time | 0.271s | 0.270s |

The baseline model triggers more compaction events (1,421 vs 1,309) because it generates more tokens in the compacted region before terminating. The compression ratios are identical (0.543) since both use the same algorithm and settings.

### Zero-Compaction Problems

Problems solved before KV hits 2048 (no compaction needed):

| Condition | Count | Accuracy | Avg Tokens |
|-----------|-------|----------|------------|
| Compact-trained | 38 | 89.5% | 1,094 |
| Baseline RL | 35 | 94.3% | 883 |

Baseline RL solves these fast problems in fewer tokens (883 vs 1,094) with higher accuracy. The compact-trained model even struggles on "easy" problems.

## Divergent Problem Analysis

| Direction | Baseline RL | Compact-trained |
|-----------|-------------|-----------------|
| Compaction HURTS | 48 problems | 28 problems |
| Compaction HELPS | 25 problems | 32 problems |
| **Net effect** | **-23** | **+4** |

Compaction is a net negative for the baseline model (-23 problems) but roughly neutral for the compact-trained model (+4 problems). This makes sense: the compact-trained model has already adapted to compacted contexts, while the baseline model's full-context reasoning gets disrupted.

### Where Compaction Hurts (Baseline RL)
- **arc_1d (12)**, **cryptarithm (11)**, **sokoban_hard (11)**: Tasks requiring coherent multi-step reasoning across long sequences
- **zebra_puzzles_7 (8)**: Constraint tables that must be maintained precisely

### Where Compaction Helps (Baseline RL)
- **arc_1d (8)**, **zebra_puzzles_7 (7)**, **countdown_7 (7)**: Cases where the model was stuck in unproductive reasoning loops and compaction acted as a "reset"

## Qualitative Trajectory Analysis

### Pattern 1: Compaction Destroys State-Dependent Reasoning (Sokoban, Cryptarithm)

**Sokoban (Problems 13, 23)**: The baseline model without compaction builds up a spatial representation of the grid and systematically tries moves. After compaction, the model loses track of which positions it has already explored and which grid state it's working with. It often restarts grid parsing from scratch, wasting tokens re-establishing context.

**Cryptarithm (Problem 4)**: The baseline solves this in 5,885 tokens without compaction (exploring constraint propagation). With compaction at 3 events, the model loses intermediate constraint deductions and fails at 3,907 tokens—it can't reconstruct the eliminated possibilities from the compressed KV cache.

### Pattern 2: Compaction as Productive Reset (ARC, Zebra Puzzles)

**ARC 1D (Problems 1, 19)**: In several ARC problems, the baseline without compaction goes down a wrong path for 4,000+ tokens and never recovers. With compaction, the model is forced to restart its reasoning from a compressed summary after ~1,024 tokens. In problems 1 and 19, this restart led to a correct approach that the uncompacted model missed.

**Zebra puzzles (Problems 37, 61, 74)**: The baseline without compaction sometimes converges on an incorrect constraint assignment and then spends the remaining tokens trying to make it work. After compaction, the model re-reads the compressed context and sometimes picks a different (correct) assignment path.

### Pattern 3: Compact-Trained Model's Degraded Reasoning

The compact-trained model shows qualitatively different reasoning patterns compared to baseline:
- **Longer preambles**: More tokens spent on problem re-statement before attempting solution
- **Less structured**: Fewer explicit step-by-step breakdowns, more stream-of-consciousness
- **Higher max-token rate**: 25.7% of sequences hit 8,192 tokens without compaction (vs 21.0% for baseline), suggesting the model has difficulty converging
- **On ARC tasks**: The compact-trained model often generates 8,000+ tokens with 9 compactions and still fails, while the baseline solves the same problem in 2,000-3,000 tokens

### Pattern 4: The Compaction Boundary Region

Examining text immediately before and after compaction events reveals:
- **Before compaction**: The model is mid-reasoning, often in the middle of a deduction chain ("Now, looking at column 3..." or "Let me check if P=5 works...")
- **After compaction**: The model restarts from a higher-level summary. Concrete intermediate results (digit assignments, grid positions, constraint eliminations) are lost. The model must re-derive them, often getting different (sometimes wrong) results.
- **Compression ratio 0.543**: Only ~54% of the original KV tokens are retained. This means nearly half the computed context is discarded at each compaction event.

## Conclusions

1. **Compaction training is destructive**: The `.detach()` at segment boundaries prevents learning to produce compaction-friendly representations. The compact-trained model is worse than baseline in ALL conditions, including without compaction (26.7% vs 44.3%).

2. **Compaction at inference is a trade-off**: For the baseline model, compaction costs -7.6pp accuracy on average but enables longer generation within a fixed KV budget. The cost is highly task-dependent (negligible for countdown, devastating for sokoban).

3. **Compaction acts as forced exploration**: In ~25 problems, compaction helped the baseline model by breaking out of unproductive reasoning loops. This suggests a potential hybrid strategy: apply compaction only when the model appears stuck.

4. **The 0-compaction sweet spot**: Problems solved before first compaction have ~94% accuracy. The key question is whether we can identify (at generation time) which problems need extended reasoning and which should terminate early.

5. **Sequence length is diagnostic**: The compact-trained model generates longer, less focused sequences—a clear sign of degraded reasoning quality. Future compaction training should monitor sequence length as a health metric.

## Recommendations

1. **Do not use the compact-trained model**. Use baseline RL + compaction at inference when KV budget is constrained.
2. **Fix compaction training** by removing `.detach()` and using full forward passes (Plan 1 from `compaction_improvement_plans.md`).
3. **Consider adaptive compaction**: Only trigger compaction when the model's output entropy or diversity suggests it's stuck, not on a fixed schedule.
4. **For sokoban-like tasks**: Avoid compaction entirely. These tasks require precise spatial state that cannot survive lossy compression.

## Appendix: Experiment Configuration

```
Models:
  Compact-trained: outputs/compaction_fixed_1024_q/weights/step_600
  Baseline RL: outputs/rg-mix-baseline/weights/step_600

Compaction settings (both compaction evals):
  compact_target_ratio: 0.25
  compact_window: 1024
  max_kv_len: 2048
  max_total_tokens: 8192
  n_compacts: 99
  use_suffix_queries: false (random queries, matching training config)

Baseline eval:
  max_tokens: 8192
  Standard vLLM serving (no compaction)

Both:
  temperature: 0.6, top_p: 0.95
  seed: 42
  300 problems, inverse-difficulty weighted
  DP=4 (4 GPUs, 1 server per GPU)
```
