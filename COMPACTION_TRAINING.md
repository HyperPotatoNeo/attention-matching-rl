# Compaction RL Training: Implementation Plan

## Why Replay is Needed (Bias Argument)

Standard trainer forward pass is BIASED for compaction samples. Tokens after compaction
were generated conditioned on compacted KV, but a standard forward pass conditions on
the full uncompacted sequence. The importance ratio `pi_theta(a|s) / pi_old(a|s)` requires
both terms to condition on the same state `s`.

**View B (compaction as part of policy)**: Compaction is a deterministic function of the
model weights and the KV cache. Both inference and training apply compaction with their
own weights. The importance ratio becomes:

```
pi_theta(a | compact_theta(s)) / pi_old(a | compact_old(s))
```

This is **unbiased** because each policy uses its own compacted KV to compute logprobs.
The trainer must replay compaction (re-run the algorithm on its own KV cache) to get
correct logprobs for post-compaction tokens.

## Implementation Phases

### Phase 1: Data Pipeline (segment_boundaries → trainer)

1. `TrainingSample` — add `segment_boundaries: list[int] | None`
2. `CompactionEnv` — store segment_boundaries from server response, override
   `add_model_response` to inject into `TrajectoryStep.extras`
3. `interleave_rollout()` — read from extras, pass to TrainingSample
4. `MicroBatch` — add `segment_boundaries: list[int] | None`
5. `prepare_sample()` — pass through
6. `data.py` — pass through to TensorMicroBatch

### Phase 2: Shared Compaction Algorithm

7. Extract `_compact_kv` from `worker.py` into `algorithm.py` — pure tensor ops,
   no vLLM dependency. Import in both worker.py and trainer.

### Phase 3: Segmented Forward in Trainer

8. Implement `segmented_forward()` in a new file `src/prime_rl/trainer/rl/compaction.py`:
   - Forward segment 0 with `use_cache=True` → get logprobs + KV cache
   - Extract KV, run compaction algorithm, detach (no gradient through state)
   - Forward segment 1 with `past_key_values=compacted_kv` → get logprobs
   - Repeat for remaining segments
   - Return concatenated logprobs matching the full sequence

9. Modify training loop in `train.py`:
   - If `segment_boundaries` present → call `segmented_forward()` instead of `forward()`
   - Otherwise → standard single forward pass (backward compatible)

### Phase 4: Config & Test

10. Add compaction params to trainer config (compact_target_ratio, compact_window)
11. Create test config, run 5 training steps with rg-mix-env
12. Log per-segment logprob stats for debugging

## Key Design Decisions

- **No sample packing for compaction samples**: Each compaction sample is its own micro
  batch (simpler segmented forward). This is enforced by setting a flag.
- **KV detachment**: Between segments, KV cache is `.detach()` — gradient does NOT flow
  through the compacted prefix. It's the "state" in the MDP sense.
- **Position IDs**: After compaction, new tokens keep their original position IDs (RoPE
  positions are baked into past keys, so the model correctly handles non-contiguous positions).
- **Compaction algorithm determinism**: Both inference and training use the same algorithm
  (same random seed for query probes). The difference is in the KV values (different weights).

## Status

- [ ] Phase 1: Data pipeline
- [ ] Phase 2: Shared algorithm
- [ ] Phase 3: Segmented forward
- [ ] Phase 4: Config & test
