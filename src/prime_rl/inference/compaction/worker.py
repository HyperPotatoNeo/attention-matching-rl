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

from prime_rl.inference.compaction.algorithm import compact_kv
from prime_rl.inference.compaction.beta_attention import (
    BetaState, patch_attention_layers, unpatch_attention_layers,
)
from prime_rl.inference.vllm.worker.filesystem import FileSystemWeightUpdateWorker

logger = logging.getLogger(__name__)

# Upper bound on inject token count (for block allocation headroom)
_MAX_INJECT_TOKENS = 40


def _build_inject_tokens(tokenizer, message: str) -> list[int]:
    """Build token IDs for ending assistant turn, user budget message, and restarting assistant.

    Produces: <|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    newline = tokenizer.encode("\n", add_special_tokens=False)
    user_ids = tokenizer.encode("user\n", add_special_tokens=False)
    asst_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
    msg_ids = tokenizer.encode(message, add_special_tokens=False)

    return (
        [im_end] + newline
        + [im_start] + user_ids + msg_ids + [im_end] + newline
        + [im_start] + asst_ids
    )


class CompactionWorker(FileSystemWeightUpdateWorker):
    """Worker extension with KV cache compaction for RL training.

    Inherits filesystem weight updates. Adds compact_generate() for
    generating text with mid-sequence KV cache compaction.
    """

    def _get_tokenizer(self):
        if not hasattr(self, '_cached_tokenizer'):
            from transformers import AutoTokenizer
            model_name = self.model_runner.vllm_config.model_config.model
            self._cached_tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)
        return self._cached_tokenizer

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
        max_kv_len: int | None = None,
        max_total_tokens: int | None = None,
        compute_beta: bool = False,
        use_suffix_queries: bool = True,
        inject_budget_message: bool = False,
        budget_message_template: str = "Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining.",
    ) -> dict:
        """Generate text with KV cache compaction between segments.

        Drives model forward passes manually using FlashAttention metadata.

        Two modes:
        - Fixed segments (default): each segment generates exactly
          max_tokens_per_segment tokens, with n_compacts compaction events.
        - KV budget mode (max_kv_len set): each segment generates until
          current_seq_len reaches max_kv_len, then compacts. Stops when
          total completion tokens reach max_total_tokens.

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

        kv_budget_mode = max_kv_len is not None
        if kv_budget_mode:
            max_total_tokens = max_total_tokens or max_kv_len * 2
            # In KV budget mode, KV cache never exceeds max_kv_len
            # Add headroom for inject tokens (temporarily exceed budget after compaction)
            inject_headroom = _MAX_INJECT_TOKENS if inject_budget_message else 0
            max_possible_len = max_kv_len + inject_headroom
        else:
            max_possible_len = len(prompt_ids) + max_tokens_per_segment * (n_compacts + 1)

        # Budget injection state
        inject_ranges: list[tuple[int, int]] = []
        injected_count = 0  # cumulative inject tokens (for accurate budget tracking)
        tokenizer = self._get_tokenizer() if inject_budget_message else None

        logger.info(
            "compact_generate: prompt_len=%d, max_tokens_per_seg=%d, n_compacts=%d, "
            "ratio=%.2f, temp=%.2f, top_p=%.2f, kv_budget_mode=%s, "
            "max_kv_len=%s, max_total_tokens=%s, inject_budget=%s | "
            "KV: layers=%d, blocks=%d, block_size=%d, kv_heads=%d, head_size=%d",
            len(prompt_ids), max_tokens_per_segment, n_compacts,
            compact_target_ratio, temperature, top_p,
            kv_budget_mode, max_kv_len, max_total_tokens, inject_budget_message,
            num_layers, num_total_blocks, block_size, num_kv_heads, head_size,
        )

        my_blocks = self._find_free_blocks(num_total_blocks)
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
        if kv_budget_mode:
            max_seg = max_kv_len  # first segment can be up to max_kv_len - prompt_len
        else:
            max_seg = max_tokens_per_segment
        seg_token_ids = torch.zeros(max_seg, dtype=torch.long, device=device)
        seg_logprobs = torch.zeros(max_seg, dtype=torch.float32, device=device)

        compaction_events = []
        segment_boundaries = []  # cumulative token count at end of each segment

        eos_check_interval = 64
        max_segments = n_compacts + 1 if not kv_budget_mode else 10000
        # After inject prefill, the first assistant token is already sampled
        injected_last_segment = False

        for segment in range(max_segments):
            t_seg = time.time()
            # First segment starts at 1 (token from prefill).
            # After inject, also 1 (first asst token sampled from inject prefill).
            tokens_in_segment = 1 if (segment == 0 or injected_last_segment) else 0
            injected_last_segment = False
            eos_hit = (all_token_ids[-1] == eos_token_id)
            seg_count = 0

            if kv_budget_mode:
                seg_limit = max_kv_len - current_seq_len
                if seg_limit <= 0:
                    break
            else:
                seg_limit = max_tokens_per_segment

            # Keep ForwardContext open for entire segment decode loop
            with set_forward_context(
                decode_ctx.attn_metadata_dict,
                decode_ctx.vllm_config,
                virtual_engine=0,
                num_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=decode_ctx.slot_mapping_ctx,
            ):
                while tokens_in_segment < seg_limit and not eos_hit:
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

            if eos_hit:
                break
            if kv_budget_mode:
                # Use assistant-only count so inject overhead doesn't reduce model's budget
                effective_tokens = len(all_token_ids) - injected_count
                if effective_tokens >= max_total_tokens:
                    break
            elif segment == n_compacts:
                break

            # --- Compact KV cache ---
            t_compact = time.time()
            # current_seq_len is 1 ahead of the last written KV position:
            # the boundary token was sampled but its KV hasn't been written.
            kv_len = current_seq_len - 1
            assistant_len_before = kv_len - prompt_len

            t_extract = time.time()
            keys, values = _extract_kv(
                kv_caches, my_blocks, kv_len, block_size, num_layers
            )
            extract_time = time.time() - t_extract

            t_algo = time.time()
            asst_len = kv_len - prompt_len
            window = min(compact_window or asst_len, asst_len)
            suffix_len = asst_len - window

            # Extract suffix query vectors for importance scoring
            suffix_queries = None
            if use_suffix_queries and suffix_len > 0:
                suffix_start = prompt_len + window
                suffix_ids = torch.tensor(
                    all_token_ids[window:window + suffix_len],
                    dtype=torch.long, device=device,
                )
                suffix_positions = torch.arange(
                    suffix_start + position_offset,
                    suffix_start + position_offset + suffix_len,
                    dtype=torch.long, device=device,
                )
                suffix_queries = _extract_suffix_queries(
                    model, suffix_ids, suffix_positions,
                    kv_caches, my_blocks, kv_len,
                    block_size, attn_layer_names,
                    vllm_config, device,
                    num_kv_heads, head_size,
                )

            compact_seed = prompt_len * 10000 + segment
            c1_list, c2_list, _, topk_indices_list = compact_kv(
                keys, values, prompt_len, compact_target_ratio,
                num_kv_heads, head_size, device,
                compact_window=window,
                compute_beta=compute_beta,
                seed=compact_seed,
                suffix_queries=suffix_queries,
            )
            algo_time = time.time() - t_algo

            compacted_prefix_len = c1_list[0].shape[0]
            new_seq_len = prompt_len + compacted_prefix_len + suffix_len

            t_inject = time.time()
            _inject_compacted_kv(
                kv_caches, keys, values, c1_list, c2_list,
                my_blocks, block_size, prompt_len, num_layers,
                old_seq_len=kv_len,
                compact_window=window,
            )
            inject_time = time.time() - t_inject

            position_offset += kv_len - new_seq_len

            if inject_budget_message:
                # Inject user budget message after compaction.
                # Prefill [boundary_token + inject_tokens] in one pass:
                #   - boundary token gets KV recomputed with compacted context
                #   - inject tokens (end asst + user msg + start asst) get their KV written
                #   - last logit predicts first token of new assistant response
                asst_tokens_generated = len(all_token_ids) - injected_count
                remaining = max(0, max_total_tokens - asst_tokens_generated)
                msg = budget_message_template.format(
                    used=asst_tokens_generated, total=max_total_tokens, remaining=remaining)
                inject_ids = _build_inject_tokens(tokenizer, msg)

                boundary_token = all_token_ids[-1]
                prefill_input = torch.tensor(
                    [boundary_token] + inject_ids, dtype=torch.long, device=device)
                prefill_positions = torch.arange(
                    new_seq_len, new_seq_len + len(prefill_input),
                    dtype=torch.long, device=device) + position_offset

                inject_logits = _run_prefill(
                    model, prefill_input, prefill_positions,
                    seq_len=new_seq_len + len(prefill_input),
                    block_ids=my_blocks,
                    block_size=block_size,
                    attn_layer_names=attn_layer_names,
                    vllm_config=vllm_config,
                    device=device,
                    prefill_start=new_seq_len,
                    prefill_len=len(prefill_input),
                )

                seed_val = len(all_token_ids) + len(inject_ids)
                first_token, first_lp = _sample_token(
                    inject_logits[-1:], temperature, top_p, rng, seed=seed_val)

                # Track injected tokens (for mask — these are not model-generated)
                inject_start = len(all_token_ids)
                all_token_ids.extend(inject_ids)
                all_logprobs.extend([0.0] * len(inject_ids))
                inject_ranges.append((inject_start, inject_start + len(inject_ids)))
                injected_count += len(inject_ids)

                # Add first assistant token (model-generated)
                all_token_ids.append(first_token.item())
                all_logprobs.append(first_lp.item())

                current_seq_len = new_seq_len + 1 + len(inject_ids) + 1
                last_token_gpu = first_token.view(1)

                injected_last_segment = True
                logger.info(
                    "Budget inject: %d tokens, msg='%s', remaining=%d",
                    len(inject_ids), msg, remaining)
            else:
                # +1 because the boundary token's KV will be recomputed
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
                "kv_len_before": kv_len,
                "kv_len_after": new_seq_len,
                "extract_time": round(extract_time, 3),
                "algo_time": round(algo_time, 3),
                "inject_time": round(inject_time, 3),
                "total_time": round(compact_time, 3),
                "compaction_indices": [
                    idx.cpu().tolist() for idx in topk_indices_list
                ],
            }
            compaction_events.append(event)

            logger.info(
                "Compaction %d: kv %d->%d, asst %d->%d (ratio=%.2f), "
                "offset=%d | extract=%.3fs, algo=%.3fs, inject=%.3fs, total=%.3fs",
                segment, kv_len, new_seq_len,
                assistant_len_before, new_assistant_len,
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
                "inject_ranges": inject_ranges,
                "final_position_offset": position_offset,
                "mean_logprob": sum(all_logprobs) / max(len(all_logprobs), 1),
            },
        }

    def compact_generate_batch(
        self,
        prompt_ids_list: list[list[int]],
        max_tokens_per_segment: int,
        n_compacts: int,
        compact_target_ratio: float,
        compact_window: int | None,
        temperature: float,
        top_p: float,
        eos_token_id: int,
        max_kv_len: int | None = None,
        max_total_tokens: int | None = None,
        compute_beta: bool = False,
        use_suffix_queries: bool = True,
        inject_budget_message: bool = False,
        budget_message_template: str = "Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining.",
    ) -> list[dict]:
        """Generate B sequences simultaneously with KV compaction.

        Same interface as compact_generate but processes a batch of prompts
        in each forward pass for higher GPU utilization.
        """
        t_start = time.time()
        B = len(prompt_ids_list)
        if B == 0:
            return []

        model = self._get_model()
        device = self._get_device()
        vllm_config = self.model_runner.vllm_config

        kv_caches = self.model_runner.kv_caches
        num_layers = len(kv_caches)
        kv_shape = kv_caches[0].shape
        num_total_blocks = kv_shape[1]
        block_size = kv_shape[2]
        num_kv_heads = kv_shape[3]
        head_size = kv_shape[4]

        kv_budget_mode = max_kv_len is not None
        if kv_budget_mode:
            max_total_tokens = max_total_tokens or max_kv_len * 2
            inject_headroom = _MAX_INJECT_TOKENS if inject_budget_message else 0
            max_possible_len = max_kv_len + inject_headroom
        else:
            max_prompt = max(len(p) for p in prompt_ids_list)
            max_possible_len = max_prompt + max_tokens_per_segment * (n_compacts + 1)

        # Budget injection state
        inject_ranges: list[list[tuple[int, int]]] = [[] for _ in range(B)]
        injected_counts = [0] * B
        tokenizer = self._get_tokenizer() if inject_budget_message else None

        max_blocks_per_seq = (max_possible_len + block_size - 1) // block_size

        free_blocks = self._find_free_blocks(num_total_blocks)
        total_needed = max_blocks_per_seq * B
        assert len(free_blocks) >= total_needed, (
            f"Need {total_needed} blocks for {B} seqs, only {len(free_blocks)} free"
        )
        per_seq_blocks = [
            free_blocks[i * max_blocks_per_seq:(i + 1) * max_blocks_per_seq]
            for i in range(B)
        ]

        attn_layer_names = self._get_attn_layer_names(model)
        rng = torch.Generator(device=device)

        logger.info(
            "compact_generate_batch: B=%d, max_kv_len=%s, max_total_tokens=%s, "
            "ratio=%.2f, blocks_per_seq=%d",
            B, max_kv_len, max_total_tokens, compact_target_ratio, max_blocks_per_seq,
        )

        # --- Prefill each prompt separately ---
        prompt_lens = []
        current_seq_lens = torch.zeros(B, dtype=torch.long, device=device)
        position_offsets = torch.zeros(B, dtype=torch.long, device=device)
        last_tokens = torch.zeros(B, dtype=torch.long, device=device)
        all_token_ids: list[list[int]] = [[] for _ in range(B)]
        all_logprobs: list[list[float]] = [[] for _ in range(B)]

        t_prefill = time.time()
        for i in range(B):
            prompt_ids = prompt_ids_list[i]
            plen = len(prompt_ids)
            prompt_lens.append(plen)

            input_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
            positions_t = torch.arange(plen, dtype=torch.long, device=device)

            logits = _run_prefill(
                model, input_ids_t, positions_t,
                seq_len=plen,
                block_ids=per_seq_blocks[i],
                block_size=block_size,
                attn_layer_names=attn_layer_names,
                vllm_config=vllm_config,
                device=device,
            )

            token, logprob = _sample_token(logits[-1:], temperature, top_p, rng, seed=i)
            all_token_ids[i].append(token.item())
            all_logprobs[i].append(logprob.item())
            current_seq_lens[i] = plen + 1
            last_tokens[i] = token

        logger.info("Prefilled %d prompts in %.3fs", B, time.time() - t_prefill)

        # --- Beta state setup (contiguous KV + beta buffers) ---
        beta_state = None
        beta_originals = None
        if compute_beta:
            num_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config)
            beta_state = BetaState(
                B=B, max_seq_len=max_possible_len, num_layers=num_layers,
                num_kv_heads=num_kv_heads, num_heads=num_heads,
                head_size=head_size, device=device, dtype=kv_caches[0].dtype,
            )
            beta_state.init_from_prefill(kv_caches, per_seq_blocks, prompt_lens, block_size)
            beta_state.seq_lens[:] = current_seq_lens
            beta_state.update_valid_mask()
            beta_originals = patch_attention_layers(model, beta_state)

        # --- Build batch decode context ---
        batch_ctx = _BatchDecodeContext.create(
            B=B,
            per_seq_blocks=per_seq_blocks,
            block_size=block_size,
            max_blocks_per_seq=max_blocks_per_seq,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )

        # --- CUDA graph capture (TP=1 only) ---
        # All batch tensors have fixed shapes (B,) with in-place value updates.
        # FA uses seqused_k (from seq_lens tensor) for variable-length sequences.
        # Compaction modifies KV data between replays — graph reads updated values.
        # Inactive sequences (post-EOS) keep decoding with fixed inputs; results ignored.
        use_cuda_graph = (self.model_runner.vllm_config.parallel_config
                          .tensor_parallel_size == 1)
        decode_graph = None
        logits_buf = None

        if use_cuda_graph:
            batch_ctx.input_ids[:] = last_tokens
            batch_ctx.positions[:] = current_seq_lens - 1
            batch_ctx.seq_lens[:] = current_seq_lens.int()
            _update_batch_slots(batch_ctx, current_seq_lens)
            decode_graph, logits_buf = _capture_decode_graph(
                model, batch_ctx, B, max_possible_len)

        # --- Per-sequence tracking ---
        active = [True] * B
        compaction_events: list[list[dict]] = [[] for _ in range(B)]
        segment_boundaries: list[list[int]] = [[] for _ in range(B)]

        max_seg = max_kv_len if kv_budget_mode else max_tokens_per_segment
        seg_tokens = torch.zeros(B, max_seg, dtype=torch.long, device=device)
        seg_lps = torch.zeros(B, max_seg, dtype=torch.float32, device=device)
        seg_counts = [0] * B

        eos_check_interval = 64
        step = 0
        arange_B = torch.arange(B, device=device)

        # --- Batched decode loop ---
        batch_ctx.metadata.max_seq_len = max_possible_len
        with set_forward_context(
            batch_ctx.attn_metadata_dict, batch_ctx.vllm_config,
            virtual_engine=0, num_tokens=B,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=batch_ctx.slot_mapping_ctx,
        ):
            while any(active):
                # Update batch decode state
                batch_ctx.input_ids[:] = last_tokens
                batch_ctx.positions[:] = current_seq_lens - 1 + position_offsets
                batch_ctx.seq_lens[:] = current_seq_lens.int()
                _update_batch_slots(batch_ctx, current_seq_lens)

                # Update beta state validity mask before model forward
                if beta_state is not None:
                    beta_state.seq_lens[:] = current_seq_lens
                    beta_state.update_valid_mask()

                # Forward pass
                if decode_graph is not None:
                    decode_graph.replay()
                    logits = logits_buf
                else:
                    hidden = model(input_ids=batch_ctx.input_ids,
                                   positions=batch_ctx.positions)
                    logits = model.compute_logits(hidden)

                # Batch sample
                rng.manual_seed(step)
                tokens, lps = _sample_batch(logits, temperature, top_p, rng)

                # Per-sequence updates
                for i in range(B):
                    if not active[i]:
                        continue
                    sc = seg_counts[i]
                    seg_tokens[i, sc] = tokens[i]
                    seg_lps[i, sc] = lps[i]
                    seg_counts[i] = sc + 1
                    last_tokens[i] = tokens[i]
                    current_seq_lens[i] += 1

                step += 1

                # Periodic EOS check
                if step % eos_check_interval == 0:
                    for i in range(B):
                        if not active[i]:
                            continue
                        sc = seg_counts[i]
                        if sc == 0:
                            continue
                        eos_pos = (seg_tokens[i, :sc] == eos_token_id).nonzero(as_tuple=False)
                        if len(eos_pos) > 0:
                            first_eos = eos_pos[0].item()
                            current_seq_lens[i] -= (sc - first_eos - 1)
                            seg_counts[i] = first_eos + 1
                            _flush_seg(i, all_token_ids, all_logprobs,
                                       seg_tokens, seg_lps, seg_counts,
                                       segment_boundaries)
                            active[i] = False

                # Check compaction triggers
                for i in range(B):
                    if not active[i]:
                        continue

                    needs_compact = False
                    if kv_budget_mode:
                        needs_compact = int(current_seq_lens[i].item()) >= max_kv_len
                    elif seg_counts[i] >= max_tokens_per_segment:
                        needs_compact = True

                    if not needs_compact:
                        continue

                    # Flush current segment
                    _flush_seg(i, all_token_ids, all_logprobs,
                               seg_tokens, seg_lps, seg_counts,
                               segment_boundaries)

                    # Check termination (assistant-only count excludes inject overhead)
                    effective_tokens_i = len(all_token_ids[i]) - injected_counts[i]
                    if kv_budget_mode and effective_tokens_i >= max_total_tokens:
                        active[i] = False
                        continue
                    if not kv_budget_mode and len(compaction_events[i]) >= n_compacts:
                        active[i] = False
                        continue

                    # Compact this sequence's KV
                    t_compact = time.time()
                    kv_len = int(current_seq_lens[i].item()) - 1

                    keys, values = _extract_kv(
                        kv_caches, per_seq_blocks[i], kv_len, block_size, num_layers
                    )

                    asst_len = kv_len - prompt_lens[i]
                    window = min(compact_window or asst_len, asst_len)
                    suffix_len_i = asst_len - window

                    # Extract suffix queries for this sequence
                    suffix_queries = None
                    if use_suffix_queries and suffix_len_i > 0:
                        suffix_start = prompt_lens[i] + window
                        suffix_ids = torch.tensor(
                            all_token_ids[i][window:window + suffix_len_i],
                            dtype=torch.long, device=device,
                        )
                        suffix_positions = torch.arange(
                            suffix_start + int(position_offsets[i].item()),
                            suffix_start + int(position_offsets[i].item()) + suffix_len_i,
                            dtype=torch.long, device=device,
                        )
                        suffix_queries = _extract_suffix_queries(
                            model, suffix_ids, suffix_positions,
                            kv_caches, per_seq_blocks[i], kv_len,
                            block_size, attn_layer_names,
                            vllm_config, device,
                            num_kv_heads, head_size,
                        )

                    compact_seed = prompt_lens[i] * 10000 + len(compaction_events[i])
                    c1_list, c2_list, beta_list, topk_indices_list = compact_kv(
                        keys, values, prompt_lens[i], compact_target_ratio,
                        num_kv_heads, head_size, device,
                        compact_window=window,
                        compute_beta=compute_beta,
                        seed=compact_seed,
                        suffix_queries=suffix_queries,
                    )

                    compacted_prefix_len = c1_list[0].shape[0]
                    suffix_len = asst_len - window
                    new_seq_len = prompt_lens[i] + compacted_prefix_len + suffix_len

                    _inject_compacted_kv(
                        kv_caches, keys, values, c1_list, c2_list,
                        per_seq_blocks[i], block_size, prompt_lens[i], num_layers,
                        old_seq_len=kv_len, compact_window=window,
                    )

                    # Update beta state contiguous buffers
                    if beta_state is not None and beta_list is not None:
                        for l in range(num_layers):
                            suffix_K = keys[l][prompt_lens[i] + window:]
                            suffix_V = values[l][prompt_lens[i] + window:]
                            beta_state.set_compacted(
                                l, i, prompt_lens[i],
                                c1_list[l], c2_list[l], beta_list[l],
                                suffix_K, suffix_V,
                            )

                        # Activate beta on first compaction, recapture CUDA graph
                        if not beta_state.active:
                            beta_state.active = True
                            if use_cuda_graph:
                                decode_graph, logits_buf = _capture_decode_graph(
                                    model, batch_ctx, B, max_possible_len)

                    position_offsets[i] += kv_len - new_seq_len

                    if inject_budget_message:
                        asst_gen = len(all_token_ids[i]) - injected_counts[i]
                        remaining = max(0, max_total_tokens - asst_gen)
                        msg = budget_message_template.format(
                            used=asst_gen, total=max_total_tokens, remaining=remaining)
                        inject_ids = _build_inject_tokens(tokenizer, msg)

                        boundary_token = all_token_ids[i][-1]
                        prefill_input = torch.tensor(
                            [boundary_token] + inject_ids, dtype=torch.long, device=device)
                        prefill_positions = torch.arange(
                            new_seq_len,
                            new_seq_len + len(prefill_input),
                            dtype=torch.long, device=device,
                        ) + position_offsets[i]

                        inject_logits = _run_prefill(
                            model, prefill_input, prefill_positions,
                            seq_len=new_seq_len + len(prefill_input),
                            block_ids=per_seq_blocks[i],
                            block_size=block_size,
                            attn_layer_names=attn_layer_names,
                            vllm_config=vllm_config,
                            device=device,
                            prefill_start=new_seq_len,
                            prefill_len=len(prefill_input),
                        )

                        seed_val = len(all_token_ids[i]) + len(inject_ids)
                        first_tok, first_lp = _sample_token(
                            inject_logits[-1:], temperature, top_p, rng, seed=seed_val)

                        inject_start = len(all_token_ids[i])
                        all_token_ids[i].extend(inject_ids)
                        all_logprobs[i].extend([0.0] * len(inject_ids))
                        inject_ranges[i].append((inject_start, inject_start + len(inject_ids)))
                        injected_counts[i] += len(inject_ids)

                        all_token_ids[i].append(first_tok.item())
                        all_logprobs[i].append(first_lp.item())

                        current_seq_lens[i] = new_seq_len + 1 + len(inject_ids) + 1
                        last_tokens[i] = first_tok
                    else:
                        current_seq_lens[i] = new_seq_len + 1

                    compact_time = time.time() - t_compact
                    new_assistant_len = compacted_prefix_len + suffix_len
                    assistant_len_before = kv_len - prompt_lens[i]
                    compaction_events[i].append({
                        "kv_len_before": kv_len,
                        "kv_len_after": new_seq_len,
                        "ratio": new_assistant_len / max(assistant_len_before, 1),
                        "algo_time": round(compact_time, 3),
                        "extract_time": 0.0,
                        "inject_time": 0.0,
                        "total_time": round(compact_time, 3),
                        "compaction_indices": [
                            idx.cpu().tolist() for idx in topk_indices_list
                        ],
                    })

        # Final flush for remaining tokens
        for i in range(B):
            sc = seg_counts[i]
            if sc > 0:
                eos_pos = (seg_tokens[i, :sc] == eos_token_id).nonzero(as_tuple=False)
                if len(eos_pos) > 0:
                    current_seq_lens[i] -= (sc - eos_pos[0].item() - 1)
                    seg_counts[i] = eos_pos[0].item() + 1
                _flush_seg(i, all_token_ids, all_logprobs,
                           seg_tokens, seg_lps, seg_counts,
                           segment_boundaries)

        total_time = time.time() - t_start
        total_tokens = sum(len(t) for t in all_token_ids)
        logger.info(
            "BATCH DONE: %d seqs, %d tokens, %.2fs, %.0f tok/s",
            B, total_tokens, total_time, total_tokens / max(total_time, 1e-6),
        )

        # Cleanup beta state
        if beta_state is not None and beta_originals is not None:
            unpatch_attention_layers(model, beta_originals)

        # Free temporary GPU memory to prevent crashes on repeated batch calls
        torch.cuda.synchronize()
        del batch_ctx, seg_tokens, seg_lps, beta_state
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return [
            {
                "all_token_ids": all_token_ids[i],
                "all_logprobs": all_logprobs[i],
                "diagnostics": {
                    "prompt_len": prompt_lens[i],
                    "total_tokens": len(all_token_ids[i]),
                    "total_time": round(total_time, 3),
                    "compaction_events": compaction_events[i],
                    "segment_boundaries": segment_boundaries[i],
                    "inject_ranges": inject_ranges[i],
                    "final_position_offset": int(position_offsets[i].item()),
                    "mean_logprob": sum(all_logprobs[i]) / max(len(all_logprobs[i]), 1),
                },
            }
            for i in range(B)
        ]

    def inject_only_generate(
        self,
        prompt_ids: list[int],
        max_total_tokens: int,
        inject_budget_every: int,
        temperature: float,
        top_p: float,
        eos_token_id: int,
        budget_message_template: str = "Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining.",
    ) -> dict:
        """Generate text with periodic budget injection (no KV compaction).

        Every inject_budget_every effective tokens, injects a user message with
        the remaining token budget. Positions are continuous (no compaction offset).
        """
        t_start = time.time()
        model = self._get_model()
        device = self._get_device()
        vllm_config = self.model_runner.vllm_config

        kv_caches = self.model_runner.kv_caches
        num_layers = len(kv_caches)
        kv_shape = kv_caches[0].shape
        num_total_blocks = kv_shape[1]
        block_size = kv_shape[2]

        tokenizer = self._get_tokenizer()
        max_injections = max_total_tokens // inject_budget_every
        # Total KV: prompt + generated + inject overhead (boundary + inject tokens per injection)
        max_possible_len = len(prompt_ids) + max_total_tokens + max_injections * (_MAX_INJECT_TOKENS + 1)

        inject_ranges: list[tuple[int, int]] = []
        injected_count = 0

        logger.info(
            "inject_only_generate: prompt_len=%d, max_total=%d, inject_every=%d, "
            "temp=%.2f, top_p=%.2f",
            len(prompt_ids), max_total_tokens, inject_budget_every,
            temperature, top_p,
        )

        my_blocks = self._find_free_blocks(num_total_blocks)
        max_blocks_needed = (max_possible_len + block_size - 1) // block_size
        assert len(my_blocks) >= max_blocks_needed, (
            f"Need {max_blocks_needed} blocks, only {len(my_blocks)} free"
        )
        my_blocks = my_blocks[:max_blocks_needed]

        attn_layer_names = self._get_attn_layer_names(model)

        decode_ctx = _DecodeContext.create(
            block_ids=my_blocks, block_size=block_size,
            attn_layer_names=attn_layer_names, vllm_config=vllm_config, device=device,
        )
        rng = torch.Generator(device=device)

        # --- Prefill ---
        input_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        positions_t = torch.arange(len(prompt_ids), dtype=torch.long, device=device)
        logits = _run_prefill(
            model, input_ids_t, positions_t,
            seq_len=len(prompt_ids), block_ids=my_blocks, block_size=block_size,
            attn_layer_names=attn_layer_names, vllm_config=vllm_config, device=device,
        )

        current_seq_len = len(prompt_ids)
        first_token, first_logprob = _sample_token(logits[-1:], temperature, top_p, rng, seed=0)
        all_token_ids = [first_token.item()]
        all_logprobs = [first_logprob.item()]
        current_seq_len += 1
        last_token_gpu = first_token.view(1)

        # CUDA graph (TP=1 only)
        use_cuda_graph = (vllm_config.parallel_config.tensor_parallel_size == 1)
        decode_graph = None
        logits_buf = None

        if use_cuda_graph:
            decode_ctx.metadata.max_seq_len = max_possible_len
            _update_decode_state(last_token_gpu, current_seq_len - 1, current_seq_len, decode_ctx)
            with set_forward_context(
                decode_ctx.attn_metadata_dict, decode_ctx.vllm_config,
                virtual_engine=0, num_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=decode_ctx.slot_mapping_ctx,
            ):
                for _ in range(3):
                    hidden = model(input_ids=decode_ctx.input_ids, positions=decode_ctx.positions)
                    logits_buf = model.compute_logits(hidden)
                torch.cuda.synchronize()
                decode_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(decode_graph):
                    hidden = model(input_ids=decode_ctx.input_ids, positions=decode_ctx.positions)
                    logits_buf = model.compute_logits(hidden)

        seg_token_ids = torch.zeros(inject_budget_every, dtype=torch.long, device=device)
        seg_logprobs = torch.zeros(inject_budget_every, dtype=torch.float32, device=device)

        segment_boundaries = []
        eos_check_interval = 64
        injected_last_segment = False

        for segment in range(max_injections + 1):
            tokens_in_segment = 1 if (segment == 0 or injected_last_segment) else 0
            injected_last_segment = False
            eos_hit = (all_token_ids[-1] == eos_token_id)
            seg_count = 0

            with set_forward_context(
                decode_ctx.attn_metadata_dict, decode_ctx.vllm_config,
                virtual_engine=0, num_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=decode_ctx.slot_mapping_ctx,
            ):
                while tokens_in_segment < inject_budget_every and not eos_hit:
                    _update_decode_state(
                        last_token_gpu, current_seq_len - 1, current_seq_len, decode_ctx,
                    )
                    if decode_graph is not None:
                        decode_graph.replay()
                        decode_logits = logits_buf
                    else:
                        decode_logits = _run_decode_step(model, decode_ctx)

                    seed = len(all_token_ids) + seg_count
                    token, logprob = _sample_token(decode_logits, temperature, top_p, rng, seed=seed)
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

            # Final EOS check
            if not eos_hit and seg_count > 0:
                eos_positions = (seg_token_ids[:seg_count] == eos_token_id).nonzero(as_tuple=False)
                if len(eos_positions) > 0:
                    first_eos = eos_positions[0].item()
                    extra = seg_count - first_eos - 1
                    seg_count = first_eos + 1
                    current_seq_len -= extra
                    eos_hit = True

            if seg_count > 0:
                all_token_ids.extend(seg_token_ids[:seg_count].tolist())
                all_logprobs.extend(seg_logprobs[:seg_count].tolist())
            segment_boundaries.append(len(all_token_ids))

            if eos_hit:
                break
            effective_tokens = len(all_token_ids) - injected_count
            if effective_tokens >= max_total_tokens:
                break
            if segment == max_injections:
                break

            # --- Inject budget message (no compaction) ---
            kv_len = current_seq_len - 1
            asst_tokens_generated = len(all_token_ids) - injected_count
            remaining = max(0, max_total_tokens - asst_tokens_generated)
            msg = budget_message_template.format(
                used=asst_tokens_generated, total=max_total_tokens, remaining=remaining)
            inject_ids = _build_inject_tokens(tokenizer, msg)

            boundary_token = all_token_ids[-1]
            prefill_input = torch.tensor(
                [boundary_token] + inject_ids, dtype=torch.long, device=device)
            prefill_positions = torch.arange(
                kv_len, kv_len + len(prefill_input), dtype=torch.long, device=device)

            inject_logits = _run_prefill(
                model, prefill_input, prefill_positions,
                seq_len=kv_len + len(prefill_input),
                block_ids=my_blocks, block_size=block_size,
                attn_layer_names=attn_layer_names, vllm_config=vllm_config, device=device,
                prefill_start=kv_len, prefill_len=len(prefill_input),
            )

            seed_val = len(all_token_ids) + len(inject_ids)
            first_tok, first_lp = _sample_token(
                inject_logits[-1:], temperature, top_p, rng, seed=seed_val)

            inject_start = len(all_token_ids)
            all_token_ids.extend(inject_ids)
            all_logprobs.extend([0.0] * len(inject_ids))
            inject_ranges.append((inject_start, inject_start + len(inject_ids)))
            injected_count += len(inject_ids)

            all_token_ids.append(first_tok.item())
            all_logprobs.append(first_lp.item())

            current_seq_len = kv_len + len(prefill_input) + 1
            last_token_gpu = first_tok.view(1)
            injected_last_segment = True

            logger.info(
                "Budget inject %d: %d tokens, remaining=%d",
                segment, len(inject_ids), remaining,
            )

        total_time = time.time() - t_start
        logger.info(
            "inject_only DONE: %d tokens, %d injections, %.2fs",
            len(all_token_ids), len(inject_ranges), total_time,
        )

        return {
            "all_token_ids": all_token_ids,
            "all_logprobs": all_logprobs,
            "diagnostics": {
                "prompt_len": len(prompt_ids),
                "total_tokens": len(all_token_ids),
                "total_time": round(total_time, 3),
                "compaction_events": [],
                "segment_boundaries": segment_boundaries,
                "inject_ranges": inject_ranges,
                "final_position_offset": 0,
                "mean_logprob": sum(all_logprobs) / max(len(all_logprobs), 1),
            },
        }

    def inject_only_generate_batch(
        self,
        prompt_ids_list: list[list[int]],
        max_total_tokens: int,
        inject_budget_every: int,
        temperature: float,
        top_p: float,
        eos_token_id: int,
        budget_message_template: str = "Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining.",
    ) -> list[dict]:
        """Batched inject-only generation. B sequences decoded in lockstep."""
        t_start = time.time()
        B = len(prompt_ids_list)
        if B == 0:
            return []

        model = self._get_model()
        device = self._get_device()
        vllm_config = self.model_runner.vllm_config

        kv_caches = self.model_runner.kv_caches
        num_layers = len(kv_caches)
        kv_shape = kv_caches[0].shape
        num_total_blocks = kv_shape[1]
        block_size = kv_shape[2]

        tokenizer = self._get_tokenizer()
        max_injections = max_total_tokens // inject_budget_every
        max_prompt = max(len(p) for p in prompt_ids_list)
        max_possible_len = max_prompt + max_total_tokens + max_injections * (_MAX_INJECT_TOKENS + 1)

        inject_ranges: list[list[tuple[int, int]]] = [[] for _ in range(B)]
        injected_counts = [0] * B

        max_blocks_per_seq = (max_possible_len + block_size - 1) // block_size
        free_blocks = self._find_free_blocks(num_total_blocks)
        total_needed = max_blocks_per_seq * B
        assert len(free_blocks) >= total_needed, (
            f"Need {total_needed} blocks for {B} seqs, only {len(free_blocks)} free"
        )
        per_seq_blocks = [
            free_blocks[i * max_blocks_per_seq:(i + 1) * max_blocks_per_seq]
            for i in range(B)
        ]

        attn_layer_names = self._get_attn_layer_names(model)
        rng = torch.Generator(device=device)

        logger.info(
            "inject_only_batch: B=%d, max_total=%d, inject_every=%d, blocks/seq=%d",
            B, max_total_tokens, inject_budget_every, max_blocks_per_seq,
        )

        # --- Prefill ---
        prompt_lens = []
        current_seq_lens = torch.zeros(B, dtype=torch.long, device=device)
        last_tokens = torch.zeros(B, dtype=torch.long, device=device)
        all_token_ids: list[list[int]] = [[] for _ in range(B)]
        all_logprobs: list[list[float]] = [[] for _ in range(B)]

        for i in range(B):
            pids = prompt_ids_list[i]
            plen = len(pids)
            prompt_lens.append(plen)

            input_ids_t = torch.tensor(pids, dtype=torch.long, device=device)
            positions_t = torch.arange(plen, dtype=torch.long, device=device)
            logits = _run_prefill(
                model, input_ids_t, positions_t,
                seq_len=plen, block_ids=per_seq_blocks[i], block_size=block_size,
                attn_layer_names=attn_layer_names, vllm_config=vllm_config, device=device,
            )
            token, logprob = _sample_token(logits[-1:], temperature, top_p, rng, seed=i)
            all_token_ids[i].append(token.item())
            all_logprobs[i].append(logprob.item())
            current_seq_lens[i] = plen + 1
            last_tokens[i] = token

        # --- Batch decode context ---
        batch_ctx = _BatchDecodeContext.create(
            B=B, per_seq_blocks=per_seq_blocks, block_size=block_size,
            max_blocks_per_seq=max_blocks_per_seq, attn_layer_names=attn_layer_names,
            vllm_config=vllm_config, device=device,
        )

        use_cuda_graph = (vllm_config.parallel_config.tensor_parallel_size == 1)
        decode_graph = None
        logits_buf = None

        if use_cuda_graph:
            batch_ctx.input_ids[:] = last_tokens
            batch_ctx.positions[:] = current_seq_lens - 1
            batch_ctx.seq_lens[:] = current_seq_lens.int()
            _update_batch_slots(batch_ctx, current_seq_lens)
            decode_graph, logits_buf = _capture_decode_graph(
                model, batch_ctx, B, max_possible_len)

        active = [True] * B
        segment_boundaries: list[list[int]] = [[] for _ in range(B)]
        # Track effective tokens since last inject per sequence
        tokens_since_inject = [1] * B  # 1 for the first token from prefill

        seg_tokens = torch.zeros(B, inject_budget_every, dtype=torch.long, device=device)
        seg_lps = torch.zeros(B, inject_budget_every, dtype=torch.float32, device=device)
        seg_counts = [0] * B

        eos_check_interval = 64
        step = 0

        batch_ctx.metadata.max_seq_len = max_possible_len
        with set_forward_context(
            batch_ctx.attn_metadata_dict, batch_ctx.vllm_config,
            virtual_engine=0, num_tokens=B,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=batch_ctx.slot_mapping_ctx,
        ):
            while any(active):
                batch_ctx.input_ids[:] = last_tokens
                batch_ctx.positions[:] = current_seq_lens - 1
                batch_ctx.seq_lens[:] = current_seq_lens.int()
                _update_batch_slots(batch_ctx, current_seq_lens)

                if decode_graph is not None:
                    decode_graph.replay()
                    logits = logits_buf
                else:
                    hidden = model(input_ids=batch_ctx.input_ids, positions=batch_ctx.positions)
                    logits = model.compute_logits(hidden)

                rng.manual_seed(step)
                tokens, lps = _sample_batch(logits, temperature, top_p, rng)

                for i in range(B):
                    if not active[i]:
                        continue
                    sc = seg_counts[i]
                    seg_tokens[i, sc] = tokens[i]
                    seg_lps[i, sc] = lps[i]
                    seg_counts[i] = sc + 1
                    last_tokens[i] = tokens[i]
                    current_seq_lens[i] += 1
                    tokens_since_inject[i] += 1

                step += 1

                if step % eos_check_interval == 0:
                    for i in range(B):
                        if not active[i]:
                            continue
                        sc = seg_counts[i]
                        if sc == 0:
                            continue
                        eos_pos = (seg_tokens[i, :sc] == eos_token_id).nonzero(as_tuple=False)
                        if len(eos_pos) > 0:
                            first_eos = eos_pos[0].item()
                            current_seq_lens[i] -= (sc - first_eos - 1)
                            seg_counts[i] = first_eos + 1
                            _flush_seg(i, all_token_ids, all_logprobs,
                                       seg_tokens, seg_lps, seg_counts, segment_boundaries)
                            active[i] = False

                # Check inject triggers
                for i in range(B):
                    if not active[i]:
                        continue
                    if tokens_since_inject[i] < inject_budget_every:
                        continue

                    # Flush segment buffer
                    _flush_seg(i, all_token_ids, all_logprobs,
                               seg_tokens, seg_lps, seg_counts, segment_boundaries)

                    # Check termination
                    effective_tokens_i = len(all_token_ids[i]) - injected_counts[i]
                    if effective_tokens_i >= max_total_tokens:
                        active[i] = False
                        continue

                    # Inject budget message
                    kv_len = int(current_seq_lens[i].item()) - 1
                    remaining = max(0, max_total_tokens - effective_tokens_i)
                    msg = budget_message_template.format(
                        used=effective_tokens_i, total=max_total_tokens, remaining=remaining)
                    inject_ids = _build_inject_tokens(tokenizer, msg)

                    boundary_token = all_token_ids[i][-1]
                    prefill_input = torch.tensor(
                        [boundary_token] + inject_ids, dtype=torch.long, device=device)
                    prefill_positions = torch.arange(
                        kv_len, kv_len + len(prefill_input), dtype=torch.long, device=device)

                    inject_logits = _run_prefill(
                        model, prefill_input, prefill_positions,
                        seq_len=kv_len + len(prefill_input),
                        block_ids=per_seq_blocks[i], block_size=block_size,
                        attn_layer_names=attn_layer_names, vllm_config=vllm_config,
                        device=device, prefill_start=kv_len, prefill_len=len(prefill_input),
                    )

                    seed_val = len(all_token_ids[i]) + len(inject_ids)
                    first_tok, first_lp = _sample_token(
                        inject_logits[-1:], temperature, top_p, rng, seed=seed_val)

                    inject_start = len(all_token_ids[i])
                    all_token_ids[i].extend(inject_ids)
                    all_logprobs[i].extend([0.0] * len(inject_ids))
                    inject_ranges[i].append((inject_start, inject_start + len(inject_ids)))
                    injected_counts[i] += len(inject_ids)

                    all_token_ids[i].append(first_tok.item())
                    all_logprobs[i].append(first_lp.item())

                    current_seq_lens[i] = kv_len + len(prefill_input) + 1
                    last_tokens[i] = first_tok
                    tokens_since_inject[i] = 1  # reset (1 for the new assistant token)

        # Final flush
        for i in range(B):
            sc = seg_counts[i]
            if sc > 0:
                eos_pos = (seg_tokens[i, :sc] == eos_token_id).nonzero(as_tuple=False)
                if len(eos_pos) > 0:
                    current_seq_lens[i] -= (sc - eos_pos[0].item() - 1)
                    seg_counts[i] = eos_pos[0].item() + 1
                _flush_seg(i, all_token_ids, all_logprobs,
                           seg_tokens, seg_lps, seg_counts, segment_boundaries)

        total_time = time.time() - t_start
        total_tokens = sum(len(t) for t in all_token_ids)
        logger.info(
            "inject_only BATCH DONE: %d seqs, %d tokens, %.2fs",
            B, total_tokens, total_time,
        )

        torch.cuda.synchronize()
        del batch_ctx, seg_tokens, seg_lps
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return [
            {
                "all_token_ids": all_token_ids[i],
                "all_logprobs": all_logprobs[i],
                "diagnostics": {
                    "prompt_len": prompt_lens[i],
                    "total_tokens": len(all_token_ids[i]),
                    "total_time": round(total_time, 3),
                    "compaction_events": [],
                    "segment_boundaries": segment_boundaries[i],
                    "inject_ranges": inject_ranges[i],
                    "final_position_offset": 0,
                    "mean_logprob": sum(all_logprobs[i]) / max(len(all_logprobs[i]), 1),
                },
            }
            for i in range(B)
        ]

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
# Batch decode context: pre-allocated tensors for B sequences
# ---------------------------------------------------------------------------

class _BatchDecodeContext:
    """Pre-allocated tensors and metadata for batched decode steps."""

    __slots__ = (
        "B", "input_ids", "positions", "slot_mapping", "seq_lens",
        "block_table", "block_table_i64", "metadata", "attn_metadata_dict",
        "slot_mapping_ctx", "vllm_config", "block_size",
    )

    @classmethod
    def create(
        cls,
        B: int,
        per_seq_blocks: list[list[int]],
        block_size: int,
        max_blocks_per_seq: int,
        attn_layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> "_BatchDecodeContext":
        ctx = cls()
        ctx.B = B
        ctx.block_size = block_size
        ctx.vllm_config = vllm_config

        ctx.input_ids = torch.zeros(B, dtype=torch.long, device=device)
        ctx.positions = torch.zeros(B, dtype=torch.long, device=device)
        ctx.slot_mapping = torch.zeros(B, dtype=torch.int64, device=device)
        ctx.seq_lens = torch.zeros(B, dtype=torch.int32, device=device)

        # Block tables — int32 for FlashAttention, int64 for slot computation
        padded = [
            blocks + [-1] * (max_blocks_per_seq - len(blocks))
            for blocks in per_seq_blocks
        ]
        ctx.block_table = torch.tensor(padded, dtype=torch.int32, device=device)
        ctx.block_table_i64 = ctx.block_table.long()

        ctx.metadata = FlashAttentionMetadata(
            num_actual_tokens=B,
            max_query_len=1,
            query_start_loc=torch.arange(B + 1, dtype=torch.int32, device=device),
            max_seq_len=0,
            seq_lens=ctx.seq_lens,
            block_table=ctx.block_table,
            slot_mapping=ctx.slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )

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
    prefill_start: int = 0,
    prefill_len: int | None = None,
) -> torch.Tensor:
    """Run prefill forward pass (variable-length, not optimized for reuse).

    Args:
        prefill_start: Cache position where these tokens begin writing KV.
            Default 0 (full prefill). For suffix-only prefill, set to kv_len
            so the suffix tokens write after existing KV.
        prefill_len: Number of tokens being prefilled. Default is seq_len.
            For suffix-only prefill, this is the suffix length while seq_len
            is the total KV length (prefix + suffix) for attention.
    """
    if prefill_len is None:
        prefill_len = seq_len

    # slot_mapping maps each INPUT token to its KV cache write position
    slots = []
    for i in range(prefill_len):
        pos = prefill_start + i
        block_idx = pos // block_size
        offset = pos % block_size
        slots.append(block_ids[block_idx] * block_size + offset)

    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)

    block_table = torch.tensor(
        block_ids, dtype=torch.int32, device=device,
    ).unsqueeze(0)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=prefill_len,
        max_query_len=prefill_len,
        query_start_loc=torch.tensor(
            [0, prefill_len], dtype=torch.int32, device=device
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
        num_tokens=prefill_len,
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
# Batch helpers
# ---------------------------------------------------------------------------

def _update_batch_slots(ctx: _BatchDecodeContext, current_seq_lens: torch.Tensor) -> None:
    """Vectorized slot_mapping update for all sequences in batch."""
    pos_in_cache = current_seq_lens - 1
    block_indices = pos_in_cache // ctx.block_size
    offsets = pos_in_cache % ctx.block_size
    arange = torch.arange(ctx.B, device=pos_in_cache.device)
    block_ids = ctx.block_table_i64[arange, block_indices]
    ctx.slot_mapping[:] = block_ids * ctx.block_size + offsets


def _capture_decode_graph(
    model: Module,
    batch_ctx: _BatchDecodeContext,
    B: int,
    max_seq_len: int,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    """Capture (or recapture) a CUDA graph for batched decode.

    Returns (graph, logits_buffer).
    """
    t = time.time()
    batch_ctx.metadata.max_seq_len = max_seq_len

    with set_forward_context(
        batch_ctx.attn_metadata_dict, batch_ctx.vllm_config,
        virtual_engine=0, num_tokens=B,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        slot_mapping=batch_ctx.slot_mapping_ctx,
    ):
        for _ in range(3):
            hidden = model(input_ids=batch_ctx.input_ids,
                           positions=batch_ctx.positions)
            logits_buf = model.compute_logits(hidden)

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hidden = model(input_ids=batch_ctx.input_ids,
                           positions=batch_ctx.positions)
            logits_buf = model.compute_logits(hidden)

    logger.info("CUDA graph captured for B=%d in %.3fs", B, time.time() - t)
    return graph, logits_buf


def _sample_batch(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per sequence from batched logits (B, vocab)."""
    B = logits.shape[0]
    arange = torch.arange(B, device=logits.device)

    if temperature <= 0:
        tokens = logits.argmax(dim=-1)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return tokens, log_probs[arange, tokens]

    logits_f = logits.float() / temperature
    log_probs = torch.log_softmax(logits_f, dim=-1)
    probs = log_probs.exp()

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = (cumsum - sorted_probs) > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(sorted_probs, 1, generator=rng).squeeze(-1)
        tokens = sorted_indices[arange, idx]
    else:
        tokens = torch.multinomial(probs, 1, generator=rng).squeeze(-1)

    return tokens, log_probs[arange, tokens]


def _flush_seg(
    i: int,
    all_token_ids: list[list[int]],
    all_logprobs: list[list[float]],
    seg_tokens: torch.Tensor,
    seg_lps: torch.Tensor,
    seg_counts: list[int],
    segment_boundaries: list[list[int]],
) -> None:
    """Transfer segment tokens from GPU buffer to CPU list for sequence i."""
    sc = seg_counts[i]
    if sc > 0:
        all_token_ids[i].extend(seg_tokens[i, :sc].tolist())
        all_logprobs[i].extend(seg_lps[i, :sc].tolist())
    segment_boundaries[i].append(len(all_token_ids[i]))
    seg_counts[i] = 0


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
    # Pre-compute block mapping once (shared across layers)
    num_blocks_to_touch = (old_seq_len + block_size - 1) // block_size
    n_blocks = min(num_blocks_to_touch, len(block_ids))
    bids = block_ids[:n_blocks]

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

        # Pad to full block-aligned length, then reshape into blocks
        padded_len = n_blocks * block_size
        K_padded = torch.zeros(padded_len, K.shape[1], K.shape[2],
                               dtype=K.dtype, device=K.device)
        V_padded = torch.zeros(padded_len, V.shape[1], V.shape[2],
                               dtype=V.dtype, device=V.device)
        K_padded[:total_len] = K
        V_padded[:total_len] = V

        # Reshape to (n_blocks, block_size, H, D) and scatter into kv_cache
        kv = kv_caches[layer_idx]
        kv[0, bids] = K_padded.view(n_blocks, block_size, K.shape[1], K.shape[2])
        kv[1, bids] = V_padded.view(n_blocks, block_size, V.shape[1], V.shape[2])


# ---------------------------------------------------------------------------
# Suffix query extraction for real-query compaction
# ---------------------------------------------------------------------------

def _extract_suffix_queries(
    model: Module,
    suffix_token_ids: torch.Tensor,
    suffix_positions: torch.Tensor,
    kv_caches: list[torch.Tensor],
    block_ids: list[int],
    kv_len: int,
    block_size: int,
    attn_layer_names: list[str],
    vllm_config,
    device: torch.device,
    num_kv_heads: int,
    head_size: int,
) -> list[torch.Tensor]:
    """Extract query vectors from suffix tokens via a single prefill pass.

    Hooks on each inner Attention layer (vllm.model_executor.layers.attention.Attention)
    to capture the query tensor after all model-specific transforms (qkv_proj, norms, RoPE).
    This is model-agnostic — works for any architecture that uses vLLM's Attention class.

    Returns:
        suffix_queries[layer]: (num_kv_heads, suffix_len * heads_per_group, head_size)
    """
    suffix_len = len(suffix_token_ids)
    if suffix_len == 0:
        return []

    from vllm.model_executor.layers.attention.attention import Attention

    # Find all inner Attention modules (the ones that receive q, k, v)
    attn_modules = []
    for _name, module in model.named_modules():
        if isinstance(module, Attention):
            attn_modules.append(module)

    if not attn_modules:
        logger.warning("No Attention modules found for suffix query extraction")
        return []

    num_layers = len(attn_modules)
    captured_queries: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            # Attention.forward(self, query, key, value, ...)
            q = args[0] if args else kwargs["query"]
            # q shape: (suffix_len, num_heads * head_dim)
            q = q.view(suffix_len, -1, head_size)
            captured_queries[layer_idx] = q.detach().float()
        return hook_fn

    hooks = []
    for layer_idx, module in enumerate(attn_modules):
        handle = module.register_forward_pre_hook(
            _make_hook(layer_idx), with_kwargs=True,
        )
        hooks.append(handle)

    # Run prefill on suffix tokens, writing at their original KV positions.
    suffix_cache_start = kv_len - suffix_len
    try:
        _run_prefill(
            model, suffix_token_ids, suffix_positions,
            seq_len=kv_len,
            block_ids=block_ids,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
            prefill_start=suffix_cache_start,
            prefill_len=suffix_len,
        )
    finally:
        for h in hooks:
            h.remove()

    # Group attention heads into KV heads for GQA
    result = []
    for l in range(num_layers):
        if l not in captured_queries:
            result.append(torch.zeros(num_kv_heads, 0, head_size,
                                      device=device, dtype=torch.float32))
            continue
        q = captured_queries[l]  # (suffix_len, num_attn_heads, head_dim)
        num_attn_heads = q.shape[1]
        heads_per_group = num_attn_heads // num_kv_heads
        q = q.view(suffix_len, num_kv_heads, heads_per_group, head_size)
        q = q.permute(1, 2, 0, 3).reshape(num_kv_heads, heads_per_group * suffix_len, head_size)
        result.append(q)

    return result
