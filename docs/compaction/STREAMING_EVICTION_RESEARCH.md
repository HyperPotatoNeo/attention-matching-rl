# Streaming KV Cache Eviction: Research Report

Research into replacing the current batch compaction (Attention Matching) with
streaming/online KV cache eviction that runs inline during decode, avoiding
the stop-extract-compact-inject cycle.

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [Candidate Approaches](#candidate-approaches)
3. [The FlashAttention Problem](#the-flashattention-problem)
4. [Proxy Metrics (No Attention Scores Needed)](#proxy-metrics)
5. [CUDA Graphs Compatibility](#cuda-graphs-compatibility)
6. [Accuracy Tradeoff vs Attention Matching](#accuracy-tradeoff)
7. [Architecture Sketch for compaction-rl](#architecture-sketch)
8. [Recommendation](#recommendation)
9. [References](#references)

---

## 1. Current Architecture <a name="current-architecture"></a>

The current system in `worker.py` uses **discrete compaction events**:

```
Prefill prompt
  |
  v
Decode N tokens (segment)           <-- CUDA graph replay
  |
  v
STOP decode loop
  |-- Extract KV from paged blocks  (~extract_time)
  |-- Run Attention Matching algo   (~algo_time: importance scoring + lstsq)
  |-- Inject compacted KV back      (~inject_time)
  |
  v
Resume decode loop (next segment)   <-- Re-captured or replayed CUDA graph
```

This creates **interruptions** every `max_tokens_per_segment` tokens. The
compaction algorithm (Attention Matching from arXiv:2602.16284) is high-quality
-- it solves a least-squares problem to find optimal replacement values C2 that
reconstruct the attention output -- but it requires stopping generation, extracting
the full KV cache, running the algorithm, and re-injecting.

Profiling (Instance #40) showed the compaction algo is only ~4.4% of total time,
with decode being ~95.6%. So the interruption cost is small in wall-clock terms,
but the architectural complexity (two CUDA graph captures, position_offset tracking,
segment boundaries, etc.) is significant.

---

## 2. Candidate Approaches <a name="candidate-approaches"></a>

### 2.1 H2O (Heavy-Hitter Oracle) -- NeurIPS 2023

**Paper**: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models" (arXiv:2306.14048)

**Core idea**: Track cumulative attention scores per token across all decode steps.
When the KV cache exceeds a budget, evict the token with the lowest cumulative
attention score, retaining a mix of "heavy hitters" (high cumulative attention) and
recent tokens (sliding window).

**Key observations**:
- Attention score distribution follows a **power law**: a small fraction of tokens
  receive the majority of attention mass. These are "heavy hitters" (H2 tokens).
- H2 tokens are **persistent**: once a token becomes a heavy hitter, it tends to
  stay one. This makes cumulative scoring stable.
- The eviction is formulated as a **dynamic submodular optimization** problem with
  provable guarantees.

**Mechanism**:
- Maintain a score accumulator `S[i]` for each cached token `i`, initialized to 0.
- After each decode step, update: `S[i] += attention_weight[i]` for all cached tokens.
- When cache is full, evict `argmin(S[i])` among non-recent tokens (keep a window of
  the last W tokens unconditionally).

**Performance**: 20% heavy hitters achieves up to 29x throughput improvement with
minimal quality loss on standard benchmarks.

**Limitation**: **Requires per-step attention scores**, which FlashAttention does
not return (see Section 3).

### 2.2 ScissorHands -- NeurIPS 2023

**Paper**: Liu et al., "Scissorhands: Exploiting the Persistence of Importance
Hypothesis for LLM KV Cache Compression at Test Time" (arXiv:2305.17118)

**Core idea**: The "persistence of importance" hypothesis -- tokens that were
important (above-average attention score) at any past step will remain important
in the future. Track which tokens have been "pivotal" and retain them with higher
probability.

**Mechanism**:
- At each decode step, identify tokens with attention score above the layer average.
- Mark these as "pivotal" in a persistent bitmap.
- When cache is full, preferentially evict non-pivotal tokens.
- Achieves up to 5x KV cache reduction without quality loss, combinable with
  4-bit quantization for 20x total compression.

**Key difference from H2O**: ScissorHands uses a binary pivotal/non-pivotal
classification rather than continuous cumulative scores. This is slightly simpler
but loses the ranking granularity.

**Same limitation**: Requires per-step attention scores.

### 2.3 FastGen (Adaptive KV Cache Compression) -- ICLR 2024

**Paper**: Ge et al., "Model Tells You What to Discard: Adaptive KV Cache
Compression for LLMs" (arXiv:2310.01801)

**Core idea**: Different attention heads have fundamentally different attention
patterns (local, special-token-focused, broad). Profile each head during prefill,
then apply per-head eviction policies during decode.

**Mechanism**:
- During prefill, profile each attention head to classify its pattern:
  - **Local**: Head attends mostly to nearby tokens --> keep sliding window only.
  - **Special-token**: Head attends to punctuation/delimiters --> keep special tokens.
  - **Broad**: Head attends to everything --> keep full cache (or use attention-weighted selection).
- During decode, apply the per-head policy to decide what to keep.

**Advantage**: The profiling is done once (at prefill), so decode-time eviction is
simple index-based selection with no attention score computation needed.

**Disadvantage**: The policy is frozen after prefill. For long reasoning chains
where attention patterns evolve, this can be suboptimal.

**Performance**: 50% memory reduction with negligible quality loss (>95% attention
score recovery with 35% cache compressed).

### 2.4 StreamingLLM -- ICLR 2024

**Paper**: Xiao et al., "Efficient Streaming Language Models with Attention Sinks"
(arXiv:2309.17453)

**Core idea**: Keep first few "attention sink" tokens (which receive disproportionate
attention due to softmax normalization) plus a sliding window of recent tokens. Evict
everything in between.

**Mechanism**: Fixed policy -- keep positions [0..3] + [seq_len-W..seq_len]. No
importance scoring needed at all.

**Advantage**: Zero overhead, trivially compatible with FlashAttention and CUDA graphs.

**Disadvantage**: Discards all middle context. Unsuitable for tasks where early
reasoning steps inform later ones (chain-of-thought, multi-step math).

### 2.5 PagedEviction -- September 2025

**Paper**: "PagedEviction: Structured Block-wise KV Cache Pruning for Efficient
Large Language Model Inference" (arXiv:2509.04377)

**Core idea**: Evict entire **blocks** (pages) rather than individual tokens,
maintaining compatibility with vLLM's PagedAttention.

**Key innovations**:
- Block-level eviction avoids partially filled blocks (incompatible with vLLM).
- Uses a **proxy importance metric** derived from key/value states directly,
  no attention scores needed.
- Eviction triggers only when a new block needs allocation (once per block_size
  tokens), not every step.
- Integrates with vLLM without modifying CUDA attention kernels.

**This is the most architecturally relevant approach for our codebase.**

### 2.6 HashEvict -- December 2024

**Paper**: "HashEvict: A Pre-Attention KV Cache Eviction Strategy using
Locality-Sensitive Hashing" (arXiv:2412.16187)

**Core idea**: Use locality-sensitive hashing (LSH) to identify tokens whose
key vectors are dissimilar to the current query, then evict them **before**
attention computation.

**Mechanism**:
- Maintain binarized Gaussian projections of cached key vectors.
- For each new query, compute Hamming distance to all cached keys.
- Evict keys with high Hamming distance (= low cosine similarity to query).

**Advantage**: Pre-attention decision, no attention scores needed, lightweight
binary operations.

**Performance**: 30-70% compression with minimal quality loss on Llama 3.

### 2.7 Lethe -- November 2025

**Paper**: Zeng et al., "Lethe: Layer- and Time-Adaptive KV Cache Pruning for
Reasoning-Intensive LLM Serving" (arXiv:2511.06029)

**Core idea**: Adaptive pruning along both spatial (per-layer budgets) and
temporal (multi-round eviction during generation) dimensions, specifically
targeting reasoning tasks.

**Key feature**: Recency-Aware Selective Retention (RASR) mechanism that
considers both recency and evolving attention patterns. Up to 2.56x throughput
with 91.7% KV cache memory reduction.

### 2.8 L2-Norm Eviction (EMNLP 2024)

**Paper**: "A Simple and Effective L2 Norm-Based Strategy for KV Cache
Compression" (EMNLP 2024)

**Core idea**: Key L2-norm correlates with attention importance. Lower L2-norm
keys receive higher attention scores (empirically observed). Evict keys with
high L2-norm.

**Advantage**: Trivially computable from cached keys, no attention scores needed.

**Limitation**: Correlation is empirical and model-dependent. Does not hold
universally, especially for reasoning tasks.

---

## 3. The FlashAttention Problem <a name="the-flashattention-problem"></a>

### 3.1 Why Can't We Access Per-Step Attention Scores?

FlashAttention (used by vLLM for all attention operations) is designed to
**never materialize the full attention matrix**. The entire point of FlashAttention
is to avoid the O(n^2) memory cost of storing attention weights. The kernel
computes attention output and `softmax_lse` (log-sum-exp) in a single fused pass
by tiling over blocks of Q, K, V.

**What FlashAttention returns**:
- `output`: The attention output (Q @ softmax(Q @ K^T / sqrt(d)) @ V)
- `softmax_lse`: The log-sum-exp per query head (log(sum(exp(Q @ K^T / sqrt(d)))))

**What it does NOT return**:
- Per-token attention weights
- Per-token importance scores
- Any per-KV-position information

### 3.2 What About `softmax_lse`?

The vLLM FlashAttention backend already uses `return_softmax_lse=True` (visible
in the cascade_attention path). The `softmax_lse` gives us the log of the
partition function per query head:

```
softmax_lse[h] = log(sum_i(exp(q_h @ k_i / sqrt(d))))
```

This is a **scalar per query per head** -- it tells us the "sharpness" of
attention distribution but NOT which specific keys received high attention.
It cannot be used to score individual KV positions for eviction.

### 3.3 Can We Modify the FlashAttention Kernel?

**Theoretically yes, practically no.** The modifications needed:
- Add a per-key attention accumulator that gets atomically updated during the
  tiled computation.
- This would require significant changes to flash_attn's CUDA kernels.
- Would break compatibility with upstream flash-attn updates.
- Would add memory overhead (one float32 per KV position per head per layer).
- Would likely degrade the kernel's performance due to the atomic accumulations.

vLLM's RFC for sparse KV cache (Issue #12254, #5751) acknowledges this challenge:
implementing H2O "requires significant changes to the attention kernel to maintain
a running sum of attention scores per head and per layer across different blocks."

### 3.4 Alternatives to Modifying FlashAttention

| Approach | How it works | Overhead | Quality |
|----------|-------------|----------|---------|
| **Key L2-norm** | Evict keys with high L2-norm | Negligible (norm already computed or trivially computable) | Moderate -- empirical correlation, model-dependent |
| **HashEvict (LSH)** | Cosine similarity proxy via binary hashing | Low (binary ops on small projections) | Good for retrieval, weaker for reasoning |
| **Query-key dot product** | Compute q @ k for current query only | O(n * d) per step per head | Good (exact current-step importance) |
| **FastGen profiling** | Classify heads at prefill, apply static policies | Zero at decode time | Good for short contexts, degrades with long reasoning |
| **PagedEviction proxy** | Block-level importance from key/value statistics | Low | Good (block-level granularity reduces noise) |
| **Periodic mini-attention** | Every K steps, do a small attention pass to update scores | Medium (amortized) | High (actual attention scores, just less frequent) |

**The most promising alternative for our use case: periodic mini-attention**,
because it is closest to what we already do (extract KV, compute importance,
inject back), but at a much lighter weight than full Attention Matching.

---

## 4. Proxy Metrics (No Attention Scores Needed) <a name="proxy-metrics"></a>

### 4.1 Key L2-Norm

**Observation**: In many transformer models, keys with lower L2-norm tend to
receive higher attention scores. The intuition is that "important" keys are
well-aligned with many queries, and this alignment happens at moderate norms.
Very high-norm keys tend to be outliers that are dissimilar to most queries.

**Implementation**: After each decode step, read the new key's L2-norm from the
KV cache. Maintain a min-heap of (norm, position) tuples. When cache is full,
evict the position with the highest norm.

**Cost**: O(1) per step (one norm computation + heap update).

**Risk**: The correlation is empirical and has been shown to be model-dependent.
For Qwen3-4B specifically, this would need validation.

### 4.2 Cosine Similarity to Recent Queries

**Observation**: If a cached key has low cosine similarity to recent queries,
it is unlikely to be attended to in the near future.

**Implementation**: Maintain a running average of recent query vectors (e.g.,
exponential moving average). Score each cached key by its cosine similarity to
this average. Evict lowest-similarity keys.

**Cost**: O(n * d) per step to score all cached keys, but can be amortized by
only scoring every K steps.

### 4.3 Observation Window (SnapKV-style)

**Observation**: The attention pattern from a small window of recent queries
predicts which prefix tokens will be important for future queries.

**Implementation**: At eviction time, take the last W query vectors (from the
model's actual query outputs, not random probes), compute full attention scores
against all cached keys, and use these to select important keys.

**This is essentially what suffix_queries does in our current algorithm**,
but applied as a streaming policy rather than in batch compaction events.

**Cost**: O(W * n * d) per eviction event. If W is small (e.g., 16-64), this
is much cheaper than full Attention Matching which also solves lstsq.

---

## 5. CUDA Graphs Compatibility <a name="cuda-graphs-compatibility"></a>

### 5.1 The Core Constraint

CUDA graphs require **fixed tensor shapes** and **fixed kernel launch parameters**
across replays. The current codebase captures a decode graph with
`max_seq_len = max_possible_len` baked in, and varies only `seqused_k` (a tensor)
to control the actual sequence length.

**Streaming eviction changes cache size mid-generation**, which creates a
fundamental tension with CUDA graphs:

- After eviction, `current_seq_len` decreases.
- The block table mapping changes (evicted tokens' slots become free).
- `seqused_k` changes, but this is already a tensor and can be updated.

### 5.2 Block-Level Eviction is Compatible

**If we evict entire blocks** (PagedEviction-style), the CUDA graph can remain
valid:
- The block table is a tensor that can be updated between replays.
- `seqused_k` is a tensor that can be updated.
- `max_seq_len` is already set to the maximum possible value.
- The only change is which block IDs appear in the block table.

Evicting whole blocks just means zeroing out entries in the block table and
updating `seqused_k`. The graph replay itself doesn't need to change.

### 5.3 Token-Level Eviction is NOT Compatible

Token-level eviction within a block would leave partially filled blocks. This
causes two problems:
1. FlashAttention operates on entire blocks -- it cannot skip individual slots
   within a block.
2. The remaining tokens would need to be repacked into contiguous blocks,
   requiring data movement that cannot be part of the captured graph.

**Exception**: If we evict tokens and repack them into new blocks **between**
graph replays (as a synchronous CPU-driven operation), it works. But this is
essentially what the current compaction does, just more frequently.

### 5.4 Hybrid Approach: Periodic Block Eviction

A middle ground that preserves CUDA graphs:
1. Decode with CUDA graph replay (no eviction).
2. Every `block_size` tokens (when a new block needs allocation), check if total
   KV cache exceeds budget.
3. If over budget, evict the least-important block (one block table update +
   `seqused_k` update, no graph recapture needed).
4. Continue CUDA graph replay.

This is essentially the PagedEviction approach and is fully compatible with the
current CUDA graph infrastructure.

---

## 6. Accuracy Tradeoff vs Attention Matching <a name="accuracy-tradeoff"></a>

### 6.1 What Makes Attention Matching Superior

Attention Matching (arXiv:2602.16284, our current algorithm) is fundamentally
different from eviction approaches:

| Property | Attention Matching | Eviction (H2O et al.) |
|----------|-------------------|----------------------|
| **Key selection** | Top-k by attention importance | Same (or proxy) |
| **Value treatment** | Solve lstsq: find C2 such that softmax(Q @ C1) @ C2 = original output | Keep original values unchanged |
| **Information preservation** | Reconstructs the attention output for any query | Only preserves the specific KV pairs, loses information from evicted tokens |
| **Compression quality** | Forms Pareto frontier, outperforms all eviction methods | Good but fundamentally limited by discarding information |

The critical difference is that **Attention Matching reconstructs optimal
replacement values** via least-squares, while eviction methods simply discard
tokens. When you evict token i, its value information is permanently lost.
When Attention Matching compresses, it redistributes that information across
the remaining value vectors.

### 6.2 Empirical Quality Comparison

From the Attention Matching paper and related benchmarks:
- Attention Matching achieves "near-lossless generation quality" at 50x compression.
- H2O at 5-15% retention (roughly 7-20x compression) shows accuracy drops from
  0.482 to 0.148 on some benchmarks.
- H2O and SnapKV are the best eviction methods for reasoning tasks, but they
  "significantly lag full cache performance on nearly every reasoning dataset."

**For reasoning-intensive tasks (which compaction-rl targets), the quality gap
between Attention Matching and eviction methods is likely significant.**

### 6.3 The "Attention Drift" Problem

A critical issue for all eviction methods during long reasoning chains: information
deemed unimportant early on may become pivotal later. This is called "attention
drift" and is particularly severe in chain-of-thought reasoning.

Attention Matching partially mitigates this because:
- It preserves information from evicted tokens in the replacement values (C2).
- The lstsq solution distributes information optimally across retained positions.

Pure eviction methods permanently lose the information, with no recovery possible.

---

## 7. Architecture Sketch for compaction-rl <a name="architecture-sketch"></a>

### 7.1 Option A: Streaming Block Eviction (Low effort, lower quality)

Replace the current segment-based compaction with per-block eviction:

```python
# Inside the decode loop (currently in compact_generate_batch)
while total_tokens < max_total_tokens and not eos_hit:
    # Normal decode step (CUDA graph replay)
    _update_decode_state(last_token_gpu, position, current_seq_len, decode_ctx)
    decode_graph.replay()
    token, logprob = _sample_token(...)

    current_seq_len += 1

    # Check if we just filled a block
    if current_seq_len % block_size == 0:
        n_blocks_used = current_seq_len // block_size
        if n_blocks_used > max_kv_blocks:
            # Evict least important block (proxy: key L2-norm mean per block)
            block_to_evict = _find_least_important_block(
                kv_caches, my_blocks, prompt_blocks, recent_window_blocks
            )
            _evict_block(block_to_evict, block_table, current_seq_len)
            # No graph recapture needed -- just update block table tensor
```

**Pros**: Simple, CUDA-graph compatible, no decode interruption.
**Cons**: Block-level granularity is coarse (block_size=16 tokens). No value
reconstruction. Quality will be worse than Attention Matching.

### 7.2 Option B: Streaming Token Eviction with Periodic Repack (Medium effort)

Evict individual tokens but batch the repacking:

```python
# Maintain importance scores alongside KV cache
importance = torch.zeros(max_seq_len, num_layers, num_kv_heads)

while total_tokens < max_total_tokens and not eos_hit:
    # Decode step
    _update_decode_state(...)
    decode_graph.replay()
    token, logprob = _sample_token(...)

    # Update importance using proxy (key norm, or periodic mini-attention)
    if step % importance_update_interval == 0:
        _update_importance_scores(importance, kv_caches, recent_queries)

    current_seq_len += 1

    # Periodic repack: when cache exceeds budget
    if current_seq_len > max_kv_len:
        # Select tokens to keep using importance scores
        keep_mask = _select_topk_per_head(importance, target_len)
        # Repack KV cache (moves data, updates block table)
        _repack_kv_cache(kv_caches, my_blocks, keep_mask)
        current_seq_len = new_len
        # May need to re-capture CUDA graph if shapes changed significantly
```

**Pros**: Token-level granularity, importance-aware.
**Cons**: Still requires periodic repacking (= mini interruptions). May need
graph recapture after repack. Essentially a lighter-weight version of current
approach.

### 7.3 Option C: Hybrid -- Streaming Eviction + Periodic Attention Matching (Recommended)

Combine streaming eviction for bulk removal with periodic Attention Matching for
quality-critical value reconstruction:

```python
while total_tokens < max_total_tokens and not eos_hit:
    # Normal decode with CUDA graph
    decode_graph.replay()
    token, logprob = _sample_token(...)
    current_seq_len += 1

    # Lightweight streaming eviction (block-level, key-norm proxy)
    if current_seq_len > soft_budget:
        _evict_lowest_importance_block(...)  # fast, no interruption

    # Periodic high-quality compaction (less frequent than current)
    if current_seq_len > hard_budget or step % compaction_interval == 0:
        # Full Attention Matching with lstsq value reconstruction
        # Same as current algorithm but triggered less frequently
        keys, values = _extract_kv(...)
        c1, c2, _, indices = compact_kv(...)
        _inject_compacted_kv(...)
```

**Pros**: Best of both worlds -- streaming eviction keeps cache size bounded
between compaction events, while periodic Attention Matching preserves the
high-quality value reconstruction that makes our approach superior.
**Cons**: More complex logic. Two eviction mechanisms to tune.

### 7.4 Option D: Keep Current Architecture, Optimize Transitions (Lowest effort)

The current architecture's interruptions add ~4.4% overhead. The compaction
algorithm itself is fast. The real question is whether the engineering complexity
of segments, position_offset, two CUDA graph captures, etc. justifies a rewrite.

Arguments for keeping current approach:
- Quality is demonstrably superior to eviction methods.
- The interruption cost is small (4.4% of total time).
- The segment structure is well-tested and battle-hardened.
- Attention Matching is differentiable (C2 via lstsq has gradients for training).

**If the goal is better quality, not faster inference, the current architecture
may already be optimal.**

---

## 8. Recommendation <a name="recommendation"></a>

### For the compaction-rl use case specifically:

**Keep Attention Matching as the primary compaction algorithm.** The quality
advantage over eviction methods is significant, especially for reasoning tasks.
The 4.4% overhead is negligible compared to the accuracy benefit.

**If streaming behavior is desired, go with Option C (hybrid)**:
1. Add block-level streaming eviction using key L2-norm as a proxy.
2. Trigger this when cache size crosses a soft threshold (e.g., 90% of budget).
3. Keep periodic Attention Matching for the hard threshold compaction.
4. This reduces the frequency of full Attention Matching events (fewer
   interruptions) while maintaining quality.

**Do NOT replace Attention Matching with pure eviction.** The lstsq value
reconstruction is the key differentiator that makes our approach form the
Pareto frontier. Pure eviction permanently loses information and performs
significantly worse on reasoning tasks.

### Practical next steps (if pursued):

1. **Validate key L2-norm correlation for Qwen3-4B**: Run the existing eval
   with importance scoring logged. Compare L2-norm ranking vs attention-based
   ranking to see if the proxy is reliable for our specific model.

2. **Prototype block-level eviction**: Add a simple block eviction routine to
   `worker.py` that runs inline in the decode loop. Test quality impact on
   rg-mix-env with aggressive budgets.

3. **Benchmark**: Compare wall-clock time and accuracy of:
   - Current: N segments with Attention Matching between each
   - Hybrid: Block eviction + less-frequent Attention Matching
   - Pure eviction: Block eviction only (no Attention Matching)

### Summary table

| Approach | Quality | Complexity | CUDA Graph Compat | FlashAttention Compat |
|----------|---------|------------|-------------------|----------------------|
| Current (Attention Matching) | Best | High (segments, offsets, dual graphs) | Yes (two captures) | Yes (extracts KV externally) |
| Pure H2O/ScissorHands | Poor for reasoning | Medium | No (needs per-step attention) | No (needs attention scores) |
| StreamingLLM | Poor (loses middle context) | Low | Yes | Yes |
| Block eviction (PagedEviction-style) | Moderate | Low | Yes | Yes |
| Key L2-norm eviction | Moderate | Low | Yes | Yes |
| HashEvict | Moderate | Low | Yes | Yes |
| **Hybrid (eviction + AM)** | **High** | **Medium** | **Yes** | **Yes** |

---

## 9. References <a name="references"></a>

### Primary papers

- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) -- NeurIPS 2023
- [Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time](https://arxiv.org/abs/2305.17118) -- NeurIPS 2023
- [FastGen: Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs](https://arxiv.org/abs/2310.01801) -- ICLR 2024
- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) -- ICLR 2024
- [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284) -- arXiv 2026 (our base algorithm)

### Proxy metrics and alternatives

- [A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression](https://aclanthology.org/2024.emnlp-main.1027.pdf) -- EMNLP 2024
- [HashEvict: A Pre-Attention KV Cache Eviction Strategy using Locality-Sensitive Hashing](https://arxiv.org/abs/2412.16187) -- December 2024
- [SnapKV: LLM Knows What You Are Looking for Before Generation](https://arxiv.org/abs/2404.14469) -- NeurIPS 2024
- [NaCl: A General and Effective KV Cache Eviction Framework for LLMs](https://aclanthology.org/2024.acl-long.428.pdf) -- ACL 2024
- [CAOTE: KV Cache Eviction for LLMs via Attention Output Error-Based Token Selection](https://arxiv.org/abs/2504.14051) -- April 2025

### vLLM-compatible approaches

- [PagedEviction: Structured Block-wise KV Cache Pruning](https://arxiv.org/abs/2509.04377) -- September 2025
- [Lethe: Layer- and Time-Adaptive KV Cache Pruning for Reasoning-Intensive LLM Serving](https://arxiv.org/abs/2511.06029) -- November 2025

### Reasoning-specific analysis

- [Hold Onto That Thought: Assessing KV Cache Compression On Reasoning](https://arxiv.org/abs/2512.12008) -- December 2025
- [The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231) -- October 2025
- [SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning](https://arxiv.org/abs/2602.22603) -- February 2026

### Implementation references

- [vLLM Sparse KV Cache Framework RFC](https://github.com/vllm-project/vllm/issues/12254)
- [vLLM H2O Feature Request](https://github.com/vllm-project/vllm/issues/3532)
- [H2O Reference Implementation](https://github.com/FMInference/H2O)
- [FlashAttention softmax_lse Discussion](https://github.com/Dao-AILab/flash-attention/issues/404)
- [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/)
