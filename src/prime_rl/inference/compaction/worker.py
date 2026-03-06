"""Worker extension for KV cache compaction.

Adds compact_generate() method that drives model forward passes manually,
performs KV cache compaction between segments, and returns all tokens + logprobs.

Called via collective_rpc from the /compact_generate endpoint. Blocks the
scheduler for the entire duration — generation happens entirely within the RPC.

With TP=1, CUDA graphs are used for decode steps (~5-10x speedup).
With TP>1, falls back to eager mode (NCCL ops incompatible with raw CUDA graphs).
"""

import logging
import math
import time

import torch
from torch.nn import Module
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention.attention import Attention
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from prime_rl.inference.vllm.worker.filesystem import FileSystemWeightUpdateWorker

logger = logging.getLogger(__name__)


class CompactionWorker(FileSystemWeightUpdateWorker):
    """Worker extension with KV cache compaction for RL training.

    Inherits filesystem weight updates. Adds compact_generate() for
    generating text with mid-sequence KV cache compaction.
    """

    def compact_generate(
        self,
        prompt_ids: list[int],
        max_tokens_per_segment: int,
        n_compacts: int,
        compact_target_ratio: float,
        compact_window: int | None,
        temperature: float,
        top_p: float,
        eos_token_id: int,
    ) -> dict:
        """Generate text with KV cache compaction between segments.

        Drives model forward passes manually using FlashAttention metadata.
        Each segment generates up to max_tokens_per_segment tokens, then
        the KV cache is compacted before the next segment.

        Returns dict with all_token_ids, all_logprobs, and diagnostics.
        """
        t_start = time.time()
        model = self._get_model()
        device = self._get_device()
        vllm_config = self.model_runner.vllm_config

        kv_caches = self.model_runner.kv_caches
        num_layers = len(kv_caches)
        kv_shape = kv_caches[0].shape  # (2, num_blocks, block_size, num_kv_heads, head_size)
        num_total_blocks = kv_shape[1]
        block_size = kv_shape[2]
        num_kv_heads = kv_shape[3]
        head_size = kv_shape[4]

        logger.info(
            "compact_generate: prompt_len=%d, max_tokens_per_seg=%d, n_compacts=%d, "
            "ratio=%.2f, temp=%.2f, top_p=%.2f | "
            "KV: layers=%d, blocks=%d, block_size=%d, kv_heads=%d, head_size=%d",
            len(prompt_ids), max_tokens_per_segment, n_compacts,
            compact_target_ratio, temperature, top_p,
            num_layers, num_total_blocks, block_size, num_kv_heads, head_size,
        )

        my_blocks = self._find_free_blocks(num_total_blocks)
        max_possible_len = len(prompt_ids) + max_tokens_per_segment * (n_compacts + 1)
        max_blocks_needed = (max_possible_len + block_size - 1) // block_size
        assert len(my_blocks) >= max_blocks_needed, (
            f"Need {max_blocks_needed} blocks, only {len(my_blocks)} free"
        )
        my_blocks = my_blocks[:max_blocks_needed]

        attn_layer_names = self._get_attn_layer_names(model)
        logger.info("Found %d attention layers", len(attn_layer_names))

        # --- Pre-allocate decode buffers (Phase 1 optimization) ---
        decode_ctx = _DecodeContext.create(
            block_ids=my_blocks,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )
        rng = torch.Generator(device=device)

        # --- Prefill prompt ---
        t_prefill = time.time()
        input_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        positions_t = torch.arange(len(prompt_ids), dtype=torch.long, device=device)

        logits = _run_prefill(
            model, input_ids_t, positions_t,
            seq_len=len(prompt_ids),
            block_ids=my_blocks,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )
        logger.info("Prefill: %d tokens in %.3fs", len(prompt_ids), time.time() - t_prefill)

        current_seq_len = len(prompt_ids)
        prompt_len = len(prompt_ids)
        position_offset = 0

        # Sample first token from prefill output
        first_token, first_logprob = _sample_token(
            logits[-1:], temperature, top_p, rng, seed=0,
        )
        all_token_ids = [first_token.item()]
        all_logprobs = [first_logprob.item()]
        current_seq_len += 1
        last_token_gpu = first_token.view(1)

        # CUDA graph capture for decode (TP=1 only — NCCL ops break graphs)
        use_cuda_graph = (self.model_runner.vllm_config.parallel_config
                          .tensor_parallel_size == 1)
        decode_graph = None
        logits_buf = None

        if use_cuda_graph:
            t_graph = time.time()
            # max_seq_len is a Python int baked into the graph —
            # set to max possible so seqused_k (tensor) controls actual compute
            decode_ctx.metadata.max_seq_len = max_possible_len

            _update_decode_state(last_token_gpu, current_seq_len - 1,
                                 current_seq_len, decode_ctx)
            with set_forward_context(
                decode_ctx.attn_metadata_dict, decode_ctx.vllm_config,
                virtual_engine=0, num_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=decode_ctx.slot_mapping_ctx,
            ):
                # Warm up (populates caches, JIT, etc.)
                for _ in range(3):
                    hidden = model(input_ids=decode_ctx.input_ids,
                                   positions=decode_ctx.positions)
                    logits_buf = model.compute_logits(hidden)

                # Capture graph
                torch.cuda.synchronize()
                decode_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(decode_graph):
                    hidden = model(input_ids=decode_ctx.input_ids,
                                   positions=decode_ctx.positions)
                    logits_buf = model.compute_logits(hidden)
            logger.info("CUDA graph captured in %.3fs", time.time() - t_graph)

        # GPU buffers for segment tokens (avoid per-token .item() syncs)
        max_seg = max_tokens_per_segment
        seg_token_ids = torch.zeros(max_seg, dtype=torch.long, device=device)
        seg_logprobs = torch.zeros(max_seg, dtype=torch.float32, device=device)

        compaction_events = []
        segment_boundaries = []  # cumulative token count at end of each segment

        eos_check_interval = 64

        for segment in range(n_compacts + 1):
            t_seg = time.time()
            tokens_in_segment = 1 if segment == 0 else 0
            eos_hit = (all_token_ids[-1] == eos_token_id)
            seg_count = 0

            # Keep ForwardContext open for entire segment decode loop
            with set_forward_context(
                decode_ctx.attn_metadata_dict,
                decode_ctx.vllm_config,
                virtual_engine=0,
                num_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=decode_ctx.slot_mapping_ctx,
            ):
                while tokens_in_segment < max_seg and not eos_hit:
                    position = current_seq_len - 1 + position_offset
                    _update_decode_state(
                        last_token_gpu, position, current_seq_len, decode_ctx,
                    )

                    if decode_graph is not None:
                        decode_graph.replay()
                        decode_logits = logits_buf
                    else:
                        decode_logits = _run_decode_step(model, decode_ctx)

                    seed = len(all_token_ids) + seg_count
                    token, logprob = _sample_token(
                        decode_logits, temperature, top_p, rng, seed=seed,
                    )

                    seg_token_ids[seg_count] = token
                    seg_logprobs[seg_count] = logprob
                    seg_count += 1

                    last_token_gpu = token.view(1)
                    current_seq_len += 1
                    tokens_in_segment += 1

                    if seg_count % eos_check_interval == 0:
                        eos_positions = (seg_token_ids[:seg_count] == eos_token_id).nonzero(as_tuple=False)
                        if len(eos_positions) > 0:
                            first_eos = eos_positions[0].item()
                            extra = seg_count - first_eos - 1
                            seg_count = first_eos + 1
                            current_seq_len -= extra
                            tokens_in_segment -= extra
                            eos_hit = True

            # Final EOS check for tokens since last batch check
            if not eos_hit and seg_count > 0:
                eos_positions = (seg_token_ids[:seg_count] == eos_token_id).nonzero(as_tuple=False)
                if len(eos_positions) > 0:
                    first_eos = eos_positions[0].item()
                    extra = seg_count - first_eos - 1
                    seg_count = first_eos + 1
                    current_seq_len -= extra
                    tokens_in_segment -= extra
                    eos_hit = True

            # Batch transfer segment tokens to CPU
            if seg_count > 0:
                all_token_ids.extend(seg_token_ids[:seg_count].tolist())
                all_logprobs.extend(seg_logprobs[:seg_count].tolist())
            segment_boundaries.append(len(all_token_ids))

            seg_time = time.time() - t_seg
            logger.info(
                "Segment %d: %d tokens in %.2fs (%.1f tok/s), "
                "cache_pos=%d, rope_pos=%d, hit_eos=%s",
                segment, tokens_in_segment, seg_time,
                tokens_in_segment / max(seg_time, 1e-6),
                current_seq_len, current_seq_len - 1 + position_offset,
                eos_hit,
            )

            if eos_hit or segment == n_compacts:
                break

            # --- Compact KV cache ---
            t_compact = time.time()
            original_seq_len = current_seq_len
            assistant_len_before = current_seq_len - prompt_len

            t_extract = time.time()
            keys, values = _extract_kv(
                kv_caches, my_blocks, current_seq_len, block_size, num_layers
            )
            extract_time = time.time() - t_extract

            t_algo = time.time()
            asst_len = current_seq_len - prompt_len
            window = min(compact_window or asst_len, asst_len)
            suffix_len = asst_len - window

            c1_list, c2_list = _compact_kv(
                keys, values, prompt_len, compact_target_ratio,
                num_kv_heads, head_size, device,
                compact_window=window,
            )
            algo_time = time.time() - t_algo

            compacted_prefix_len = c1_list[0].shape[0]
            new_seq_len = prompt_len + compacted_prefix_len + suffix_len

            t_inject = time.time()
            _inject_compacted_kv(
                kv_caches, keys, values, c1_list, c2_list,
                my_blocks, block_size, prompt_len, num_layers,
                old_seq_len=current_seq_len,
                compact_window=window,
            )
            inject_time = time.time() - t_inject

            position_offset += original_seq_len - new_seq_len
            # +1 because the last generated token's KV will be recomputed
            # in the next decode step (it attends to compacted context)
            current_seq_len = new_seq_len + 1

            compact_time = time.time() - t_compact
            new_assistant_len = compacted_prefix_len + suffix_len
            event = {
                "segment": segment,
                "assistant_tokens_before": assistant_len_before,
                "assistant_tokens_after": new_assistant_len,
                "prefix_compacted": window,
                "prefix_after": compacted_prefix_len,
                "suffix_preserved": suffix_len,
                "ratio": new_assistant_len / max(assistant_len_before, 1),
                "position_offset": position_offset,
                "extract_time": round(extract_time, 3),
                "algo_time": round(algo_time, 3),
                "inject_time": round(inject_time, 3),
                "total_time": round(compact_time, 3),
            }
            compaction_events.append(event)

            logger.info(
                "Compaction %d: %d -> %d assistant tokens (ratio=%.2f), "
                "offset=%d | extract=%.3fs, algo=%.3fs, inject=%.3fs, total=%.3fs",
                segment, assistant_len_before, new_assistant_len,
                event["ratio"], position_offset,
                extract_time, algo_time, inject_time, compact_time,
            )

        total_time = time.time() - t_start
        logger.info(
            "DONE: %d tokens total, %d compactions, %.2fs total, "
            "mean_logprob=%.4f",
            len(all_token_ids), len(compaction_events), total_time,
            sum(all_logprobs) / max(len(all_logprobs), 1),
        )

        return {
            "all_token_ids": all_token_ids,
            "all_logprobs": all_logprobs,
            "diagnostics": {
                "prompt_len": prompt_len,
                "total_tokens": len(all_token_ids),
                "total_time": round(total_time, 3),
                "compaction_events": compaction_events,
                "segment_boundaries": segment_boundaries,
                "final_position_offset": position_offset,
                "mean_logprob": sum(all_logprobs) / max(len(all_logprobs), 1),
            },
        }

    def _get_model(self) -> Module:
        model = self.model_runner.model
        if hasattr(model, "runnable"):
            model = model.runnable
        assert isinstance(model, Module)
        return model

    def _get_device(self) -> torch.device:
        return next(self._get_model().parameters()).device

    def _find_free_blocks(self, num_total_blocks: int) -> list[int]:
        used = set()
        for req in self.model_runner.requests.values():
            for group_block_ids in req.block_ids:
                used.update(group_block_ids)
        return sorted(b for b in range(num_total_blocks) if b not in used)

    @staticmethod
    def _get_attn_layer_names(model: Module) -> list[str]:
        return [
            name for name, module in model.named_modules()
            if isinstance(module, Attention)
        ]


# ---------------------------------------------------------------------------
# Decode context: pre-allocated tensors reused across all decode steps
# ---------------------------------------------------------------------------

class _DecodeContext:
    """Pre-allocated tensors and metadata for fast decode steps."""

    __slots__ = (
        "input_ids", "positions", "slot_mapping", "seq_lens",
        "block_table", "metadata", "attn_metadata_dict",
        "slot_mapping_ctx", "vllm_config", "block_ids", "block_size",
    )

    @classmethod
    def create(
        cls,
        block_ids: list[int],
        block_size: int,
        attn_layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> "_DecodeContext":
        ctx = cls()
        ctx.block_ids = block_ids
        ctx.block_size = block_size
        ctx.vllm_config = vllm_config

        # Reusable scalar tensors
        ctx.input_ids = torch.zeros(1, dtype=torch.long, device=device)
        ctx.positions = torch.zeros(1, dtype=torch.long, device=device)
        ctx.slot_mapping = torch.zeros(1, dtype=torch.int64, device=device)
        ctx.seq_lens = torch.zeros(1, dtype=torch.int32, device=device)

        # Block table (fixed for entire generation)
        n_blocks = len(block_ids)
        ctx.block_table = torch.tensor(
            block_ids, dtype=torch.int32, device=device,
        ).unsqueeze(0)
        # Pad to full allocation size if needed
        if ctx.block_table.shape[1] < n_blocks:
            pad = torch.full(
                (1, n_blocks - ctx.block_table.shape[1]), -1,
                dtype=torch.int32, device=device,
            )
            ctx.block_table = torch.cat([ctx.block_table, pad], dim=1)

        # Template metadata — mutable fields updated in-place each step
        ctx.metadata = FlashAttentionMetadata(
            num_actual_tokens=1,
            max_query_len=1,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            max_seq_len=0,  # updated each step
            seq_lens=ctx.seq_lens,
            block_table=ctx.block_table,
            slot_mapping=ctx.slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )

        # Pre-built dicts (same object refs each step — no dict creation)
        ctx.attn_metadata_dict = {name: ctx.metadata for name in attn_layer_names}
        ctx.slot_mapping_ctx = {name: ctx.slot_mapping for name in attn_layer_names}

        return ctx


# ---------------------------------------------------------------------------
# Forward pass functions
# ---------------------------------------------------------------------------

def _run_prefill(
    model, input_ids, positions,
    seq_len, block_ids, block_size,
    attn_layer_names, vllm_config, device,
) -> torch.Tensor:
    """Run prefill forward pass (variable-length, not optimized for reuse)."""
    slots = []
    for pos in range(seq_len):
        block_idx = pos // block_size
        offset = pos % block_size
        slots.append(block_ids[block_idx] * block_size + offset)

    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)

    n_blocks = len(block_ids)
    block_table = torch.tensor(
        block_ids, dtype=torch.int32, device=device,
    ).unsqueeze(0)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=seq_len,
        max_query_len=seq_len,
        query_start_loc=torch.tensor(
            [0, seq_len], dtype=torch.int32, device=device
        ),
        max_seq_len=seq_len,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )

    attn_metadata = {name: metadata for name in attn_layer_names}
    slot_mapping_dict = {name: slot_mapping for name in attn_layer_names}

    with set_forward_context(
        attn_metadata,
        vllm_config,
        virtual_engine=0,
        num_tokens=seq_len,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        slot_mapping=slot_mapping_dict,
    ):
        hidden_states = model(input_ids=input_ids, positions=positions)

    return model.compute_logits(hidden_states)


def _update_decode_state(
    token_id: torch.Tensor, position: int,
    seq_len: int, ctx: _DecodeContext,
) -> None:
    """Update pre-allocated decode buffers in-place for next step."""
    ctx.input_ids[0] = token_id
    ctx.positions[0] = position

    pos_in_cache = seq_len - 1
    block_idx = pos_in_cache // ctx.block_size
    offset = pos_in_cache % ctx.block_size
    ctx.slot_mapping[0] = ctx.block_ids[block_idx] * ctx.block_size + offset

    ctx.seq_lens[0] = seq_len
    ctx.metadata.max_seq_len = seq_len


def _run_decode_step(model, ctx: _DecodeContext) -> torch.Tensor:
    """Forward pass using pre-updated decode context (caller manages ForwardContext)."""
    hidden_states = model(input_ids=ctx.input_ids, positions=ctx.positions)
    return model.compute_logits(hidden_states)


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    rng: torch.Generator,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a token with temperature + top-p. Deterministic via seed for TP consistency."""
    if temperature <= 0:
        token = logits.argmax(dim=-1).squeeze()
        logprob = torch.log_softmax(logits.float(), dim=-1).max(dim=-1).values.squeeze()
        return token, logprob

    logits_f = logits.float().squeeze(0) / temperature
    log_probs = torch.log_softmax(logits_f, dim=-1)
    probs = log_probs.exp()

    rng.manual_seed(seed)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = (cumsum - sorted_probs) > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        idx = torch.multinomial(sorted_probs, 1, generator=rng)
        token = sorted_indices[idx].squeeze()
        logprob = log_probs[token]
    else:
        token = torch.multinomial(probs, 1, generator=rng).squeeze()
        logprob = log_probs[token]

    return token, logprob


# ---------------------------------------------------------------------------
# KV cache manipulation
# ---------------------------------------------------------------------------

def _extract_kv(
    kv_caches: list[torch.Tensor],
    block_ids: list[int],
    seq_len: int,
    block_size: int,
    num_layers: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract KV from paged blocks as contiguous tensors per layer.

    Returns:
        keys[layer]: (seq_len, num_kv_heads, head_size)
        values[layer]: (seq_len, num_kv_heads, head_size)
    """
    keys, values = [], []
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    bids = block_ids[:num_blocks_needed]

    for layer_idx in range(num_layers):
        kv = kv_caches[layer_idx]
        k_gathered = kv[0][bids].reshape(-1, kv.shape[3], kv.shape[4])[:seq_len]
        v_gathered = kv[1][bids].reshape(-1, kv.shape[3], kv.shape[4])[:seq_len]
        keys.append(k_gathered.clone())
        values.append(v_gathered.clone())

    return keys, values


def _compact_kv(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    prompt_len: int,
    target_ratio: float,
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
    num_queries: int = 64,
    compact_window: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compact assistant KV prefix via Attention Matching (beta=0, random queries).

    When compact_window is set, only the first `compact_window` assistant tokens
    are compressed. The full assistant KV (including suffix beyond the window) is
    used for attention scoring so the algorithm sees the full context. The suffix
    KV is preserved unchanged by the caller.

    Returns:
        c1[layer]: (target_len, num_kv_heads, head_size) — compacted prefix keys
        c2[layer]: (target_len, num_kv_heads, head_size) — compacted prefix values
    """
    num_layers = len(keys)
    dtype = keys[0].dtype
    scale = 1.0 / math.sqrt(head_size)

    c1_list, c2_list = [], []

    for layer_idx in range(num_layers):
        asst_K = keys[layer_idx][prompt_len:]
        asst_V = values[layer_idx][prompt_len:]
        asst_len = asst_K.shape[0]
        window = min(compact_window or asst_len, asst_len)
        target_len = max(1, int(window * target_ratio))

        prefix_K = asst_K[:window]
        prefix_V = asst_V[:window]

        layer_c1, layer_c2 = [], []

        for h in range(num_kv_heads):
            # Full assistant KV for attention scoring
            k_full_f = asst_K[:, h, :].float()
            v_full_f = asst_V[:, h, :].float()
            # Prefix-only KV for selection/compression
            k_prefix_f = prefix_K[:, h, :].float()

            Q = torch.randn(num_queries, head_size, device=device, dtype=torch.float32)

            # Score using full context so suffix informs which prefix keys matter
            full_scores = Q @ k_full_f.T * scale
            full_attn = torch.softmax(full_scores, dim=-1)

            # Only select from prefix keys
            prefix_scores = full_attn[:, :window].pow(2).mean(dim=0).sqrt()
            topk_indices = prefix_scores.topk(target_len).indices.sort().values

            c1 = prefix_K[topk_indices, h, :]
            c1_f = k_prefix_f[topk_indices]

            # Regression: C2 should reproduce the prefix's contribution to output.
            # After injection the model attends to [prompt | C1 | suffix], so
            # C2 only needs to cover what the original prefix contributed.
            v_prefix_f = prefix_V[:, h, :].float()
            prefix_attn = full_attn[:, :window]
            Y = prefix_attn @ v_prefix_f
            X = torch.softmax(Q @ c1_f.T * scale, dim=-1)
            c2 = torch.linalg.lstsq(X, Y).solution.to(dtype)

            layer_c1.append(c1)
            layer_c2.append(c2)

        c1_list.append(torch.stack(layer_c1, dim=1))
        c2_list.append(torch.stack(layer_c2, dim=1))

    return c1_list, c2_list


def _inject_compacted_kv(
    kv_caches: list[torch.Tensor],
    original_keys: list[torch.Tensor],
    original_values: list[torch.Tensor],
    c1_list: list[torch.Tensor],
    c2_list: list[torch.Tensor],
    block_ids: list[int],
    block_size: int,
    prompt_len: int,
    num_layers: int,
    old_seq_len: int,
    compact_window: int | None = None,
) -> None:
    """Write [prompt | compacted_prefix | suffix] into paged blocks, zeroing stale."""
    for layer_idx in range(num_layers):
        orig_K = original_keys[layer_idx]
        orig_V = original_values[layer_idx]
        asst_len = orig_K.shape[0] - prompt_len
        window = min(compact_window or asst_len, asst_len)
        suffix_K = orig_K[prompt_len + window:]
        suffix_V = orig_V[prompt_len + window:]

        K = torch.cat([orig_K[:prompt_len], c1_list[layer_idx], suffix_K], dim=0)
        V = torch.cat([orig_V[:prompt_len], c2_list[layer_idx], suffix_V], dim=0)
        total_len = K.shape[0]

        kv = kv_caches[layer_idx]
        num_blocks_to_touch = (old_seq_len + block_size - 1) // block_size
        for i in range(min(num_blocks_to_touch, len(block_ids))):
            bid = block_ids[i]
            start = i * block_size
            end = min(start + block_size, total_len)
            if start < total_len:
                kv[0, bid, : end - start] = K[start:end]
                kv[1, bid, : end - start] = V[start:end]
                if end - start < block_size:
                    kv[0, bid, end - start :] = 0
                    kv[1, bid, end - start :] = 0
            else:
                kv[0, bid] = 0
                kv[1, bid] = 0
