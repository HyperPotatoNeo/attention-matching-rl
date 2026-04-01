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
import random as pyrandom
import time
from dataclasses import dataclass

import torch
from torch.nn import Module
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention.attention import Attention
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from prime_rl.inference.compaction.algorithm import compact_kv, compact_kv_range
from prime_rl.inference.compaction.beta_attention import (
    BetaState, patch_attention_layers, unpatch_attention_layers,
)
from prime_rl.inference.vllm.worker.filesystem import FileSystemWeightUpdateWorker

logger = logging.getLogger(__name__)


SESSION_TTL_SECONDS = 180


@dataclass
class SessionState:
    block_ids: list[int]
    current_seq_len: int
    prompt_len: int
    position_offset: int
    max_kv_len: int
    compact_target_ratio: float
    compact_window: int | None
    compaction_mode: str
    temperature: float
    top_p: float
    eos_token_id: int
    use_suffix_queries: bool
    turn_user_lens: list[int]
    turn_asst_lens: list[int]
    n_max_turns: int       # -1 = disabled (use KV-budget mode)
    n_preserved_turns: int  # turns to keep verbatim after compaction fires
    compaction_count: int = 0
    compacted_prefix_end: int = 0  # stacking cursor for attention_matching_full
    last_access: float = 0.0


class CompactionWorker(FileSystemWeightUpdateWorker):
    """Worker extension with KV cache compaction for RL training.

    Inherits filesystem weight updates. Adds compact_generate() for
    generating text with mid-sequence KV cache compaction.
    """

    _think_token_id: int | None = None

    def _get_think_token_id(self) -> int:
        """Get the <think> token ID, cached across calls."""
        if self._think_token_id is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                self.model_runner.vllm_config.model_config.model,
                trust_remote_code=True,
            )
            ids = tok.encode("<think>", add_special_tokens=False)
            if len(ids) != 1:
                logger.warning("<think> encodes to %d tokens, expected 1; skipping injection", len(ids))
                self._think_token_id = -1
            else:
                self._think_token_id = ids[0]
                logger.info("<think> token ID: %d", self._think_token_id)
        return self._think_token_id

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
        compaction_mode: str = "attention_matching",
        carryover_ratio: float = 0.5,
        use_suffix_queries: bool = True,
    ) -> dict:
        """Generate text with KV cache compaction between segments.

        Drives model forward passes manually using FlashAttention metadata.

        Two modes:
        - Fixed segments (default): each segment generates exactly
          max_tokens_per_segment tokens, with n_compacts compaction events.
        - KV budget mode (max_kv_len set): each segment generates until
          current_seq_len reaches max_kv_len, then compacts. Stops when
          total completion tokens reach max_total_tokens.

        compaction_mode: "attention_matching" (default) or "markovian" (keep last m tokens).
        carryover_ratio: fraction of assistant tokens to keep in markovian mode (default 0.5).

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
            # Prompt may exceed max_kv_len in multi-turn sessions; +2 because after
            # one ineffective compaction current_seq_len = prompt_len+2 (one step grows
            # it past max_kv_len, compaction restores to prompt_len+1, next step = +2).
            max_possible_len = max(max_kv_len, len(prompt_ids) + 2)
        else:
            max_possible_len = len(prompt_ids) + max_tokens_per_segment * (n_compacts + 1)

        # <think> injection disabled — forced tokens get very low logprobs
        # which cause KL explosion during training (importance ratio exp(40+))
        think_token_id = -1
        inject_think = False

        logger.info(
            "compact_generate: prompt_len=%d, max_tokens_per_seg=%d, n_compacts=%d, "
            "ratio=%.2f, temp=%.2f, top_p=%.2f, kv_budget_mode=%s, "
            "max_kv_len=%s, max_total_tokens=%s | "
            "KV: layers=%d, blocks=%d, block_size=%d, kv_heads=%d, head_size=%d",
            len(prompt_ids), max_tokens_per_segment, n_compacts,
            compact_target_ratio, temperature, top_p,
            kv_budget_mode, max_kv_len, max_total_tokens,
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
            with torch.inference_mode(), set_forward_context(
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

        for segment in range(max_segments):
            t_seg = time.time()
            tokens_in_segment = 1 if segment == 0 else 0
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

                    if inject_think and think_token_id > 0:
                        logits_f = decode_logits.float().squeeze(0) / temperature
                        log_probs_all = torch.log_softmax(logits_f, dim=-1)
                        token = torch.tensor(think_token_id, device=device)
                        logprob = log_probs_all[think_token_id]
                        inject_think = False
                    else:
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
                if len(all_token_ids) >= max_total_tokens:
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

            indices_list = None
            if compaction_mode == "markovian":
                # Drop the window entirely, keep only the suffix
                kv_dim = (0, keys[0].shape[1], keys[0].shape[2])
                c1_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in keys]
                c2_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in values]
                compacted_prefix_len = 0
                inject_think = True
            else:
                c1_list, c2_list, _, indices_list = compact_kv(
                    keys, values, prompt_len, compact_target_ratio,
                    num_kv_heads, head_size, device,
                    compact_window=window,
                    compute_beta=compute_beta,
                )
                compacted_prefix_len = c1_list[0].shape[0]

            algo_time = time.time() - t_algo
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
                "compaction_indices": [idx.cpu().tolist() for idx in indices_list] if indices_list else None,
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
        compaction_mode: str = "attention_matching",
        carryover_ratio: float = 0.5,
        use_suffix_queries: bool = True,
    ) -> list[dict]:
        """Generate B sequences simultaneously with KV compaction."""
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
            max_prompt = max(len(p) for p in prompt_ids_list)
            # +2: after one ineffective compaction current_seq_len = max_prompt+2;
            # need enough blocks so _update_batch_slots doesn't go OOB.
            max_possible_len = max(max_kv_len, max_prompt + 2)
        else:
            max_prompt = max(len(p) for p in prompt_ids_list)
            max_possible_len = max_prompt + max_tokens_per_segment * (n_compacts + 1)

        # <think> injection disabled — forced tokens cause KL explosion
        think_token_id = -1
        inject_think = [False] * B

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

                # Override with <think> for sequences that just had markovian compaction
                if think_token_id > 0:
                    for i in range(B):
                        if inject_think[i]:
                            logits_f = logits[i].float() / temperature
                            log_probs_all = torch.log_softmax(logits_f, dim=-1)
                            tokens[i] = think_token_id
                            lps[i] = log_probs_all[think_token_id]
                            inject_think[i] = False

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

                    # Check termination
                    if kv_budget_mode and len(all_token_ids[i]) >= max_total_tokens:
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
                    suffix_len = asst_len - window

                    indices_list = None
                    if compaction_mode == "markovian":
                        kv_dim = (0, keys[0].shape[1], keys[0].shape[2])
                        c1_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in keys]
                        c2_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in values]
                        beta_list = None
                        compacted_prefix_len = 0
                        inject_think[i] = True
                    else:
                        c1_list, c2_list, beta_list, indices_list = compact_kv(
                            keys, values, prompt_lens[i], compact_target_ratio,
                            num_kv_heads, head_size, device,
                            compact_window=window,
                            compute_beta=compute_beta,
                        )
                        compacted_prefix_len = c1_list[0].shape[0]
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
                        "compaction_indices": [idx.cpu().tolist() for idx in indices_list] if indices_list else None,
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
                    "final_position_offset": int(position_offsets[i].item()),
                    "mean_logprob": sum(all_logprobs[i]) / max(len(all_logprobs[i]), 1),
                },
            }
            for i in range(B)
        ]

    def rsa_generate(
        self,
        prompt_ids: list[int],
        K: int,
        T: int,
        k_peers: int,
        max_tokens_per_candidate: int,
        compact_target_ratio: float,
        probe_tokens: int,
        agg_template: str,
        temperature: float,
        top_p: float,
        eos_token_id: int,
        selection_strategy: str = "random",
        N: int | None = None,
    ) -> dict:
        """RSA V2: Recursive Self-Aggregation with persistent compacted memory."""
        if N is None:
            N = K
        t_start = time.time()
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

        attn_layer_names = self._get_attn_layer_names(model)
        rng = torch.Generator(device=device)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.model, trust_remote_code=True)

        # Block budget. Candidates get forked copies of base KV + room for generation.
        # Key insight: candidates can REUSE base block IDs for their prefix since
        # _fork_kv_blocks copies data into candidate blocks. We only allocate base
        # once, then each candidate gets base_blocks (reused) + extra generation blocks.
        max_agg_prompt = max_tokens_per_candidate * k_peers + 512
        if compact_target_ratio >= 1.0:
            base_max_len = len(prompt_ids) + T * max_agg_prompt + 64
        else:
            max_compacted_per_step = int(max_tokens_per_candidate * compact_target_ratio) + 64
            base_max_len = (len(prompt_ids) + T * max_compacted_per_step
                            + max_agg_prompt + probe_tokens + 64)

        max_blocks_base = (base_max_len + block_size - 1) // block_size
        extra_blocks_per_candidate = (max_tokens_per_candidate + block_size - 1) // block_size

        free_blocks = self._find_free_blocks(num_total_blocks)
        total_blocks_needed = max_blocks_base + N * (max_blocks_base + extra_blocks_per_candidate)
        logger.info(
            "Block budget: base=%d, extra/cand=%d, total=%d (N=%d), free=%d",
            max_blocks_base, extra_blocks_per_candidate, total_blocks_needed, N, len(free_blocks),
        )
        assert len(free_blocks) >= total_blocks_needed, (
            f"RSA needs {total_blocks_needed} blocks, only {len(free_blocks)} free"
        )

        base_blocks = free_blocks[:max_blocks_base]
        candidate_blocks = []
        offset = max_blocks_base
        cand_block_size = max_blocks_base + extra_blocks_per_candidate
        for _n in range(N):
            candidate_blocks.append(free_blocks[offset:offset + cand_block_size])
            offset += cand_block_size

        logger.info(
            "rsa_generate: prompt=%d, N=%d, K=%d, T=%d, k_peers=%d, max_tok=%d, "
            "ratio=%.2f, probe=%d, base_blocks=%d, cand_blocks=%d",
            len(prompt_ids), N, K, T, k_peers, max_tokens_per_candidate,
            compact_target_ratio, probe_tokens,
            max_blocks_base, cand_block_size,
        )

        # --- Prefill question onto base blocks ---
        input_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        positions_t = torch.arange(len(prompt_ids), dtype=torch.long, device=device)

        prefill_logits = _run_prefill(
            model, input_ids_t, positions_t,
            seq_len=len(prompt_ids),
            block_ids=base_blocks,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )
        base_seq_len = len(prompt_ids)
        base_offset = 0

        # --- Step 0: Fork & batch-generate N candidates ---
        num_base_blocks = (base_seq_len + block_size - 1) // block_size
        for n in range(N):
            _fork_kv_blocks(kv_caches, base_blocks, candidate_blocks[n],
                            num_base_blocks, num_layers)

        populations = []
        pop0, pop0_logprobs, _ = _batch_generate(
            model, kv_caches, candidate_blocks, N,
            base_seq_len, base_offset, max_tokens_per_candidate,
            block_size, num_layers, attn_layer_names, vllm_config,
            device, temperature, top_p, eos_token_id, rng,
            first_logits=prefill_logits, tokenizer=tokenizer,
        )
        populations.append(pop0)

        logger.info("Step 0: generated %d candidates, lengths %s",
                     N, [len(c) for c in pop0])

        compaction_events = []
        torch.cuda.empty_cache()

        for t in range(T):
            t_step = time.time()

            # --- Select peers ---
            prev_pop = populations[-1]
            if selection_strategy == "best" and pop0_logprobs is not None:
                mean_lps = [
                    sum(lps) / max(len(lps), 1) for lps in
                    (pop0_logprobs if t == 0 else [[] for _ in range(N)])
                ]
                ranked = sorted(range(len(prev_pop)), key=lambda i: -mean_lps[i])
                peer_indices = ranked[:k_peers]
            else:
                peer_indices = pyrandom.sample(range(len(prev_pop)), min(k_peers, len(prev_pop)))

            # --- Build aggregation prompt ---
            peer_cots = "\n\n".join(
                f"=== Response {j+1} ===\n{prev_pop[idx]}"
                for j, idx in enumerate(peer_indices)
            )
            agg_text = agg_template.format(peer_cots=peer_cots)

            agg_ids = tokenizer.encode(agg_text, add_special_tokens=False)
            agg_ids_t = torch.tensor(agg_ids, dtype=torch.long, device=device)

            # --- Append aggregation prompt to base KV ---
            agg_start = base_seq_len
            agg_logits = _prefill_append(
                model, agg_ids_t,
                existing_seq_len=base_seq_len,
                position_offset=base_offset,
                block_ids=base_blocks,
                block_size=block_size,
                attn_layer_names=attn_layer_names,
                vllm_config=vllm_config,
                device=device,
            )
            base_seq_len += len(agg_ids)

            skip_compaction = compact_target_ratio >= 1.0

            if not skip_compaction:
                # --- Generate probe (reuse candidate_blocks[0]) ---
                num_base_blocks_now = (base_seq_len + block_size - 1) // block_size
                _fork_kv_blocks(kv_caches, base_blocks, candidate_blocks[0],
                                num_base_blocks_now, num_layers)

                probe_decode_ctx = _DecodeContext.create(
                    block_ids=candidate_blocks[0],
                    block_size=block_size,
                    attn_layer_names=attn_layer_names,
                    vllm_config=vllm_config,
                    device=device,
                )

                probe_seq_len = base_seq_len
                probe_last_token, _ = _sample_token(
                    agg_logits[-1:], temperature, top_p, rng, seed=t * 1000)
                probe_last_token = probe_last_token.view(1)
                probe_seq_len += 1

                probe_decode_ctx.metadata.max_seq_len = probe_seq_len + probe_tokens

                with set_forward_context(
                    probe_decode_ctx.attn_metadata_dict, probe_decode_ctx.vllm_config,
                    virtual_engine=0, num_tokens=1,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=probe_decode_ctx.slot_mapping_ctx,
                ):
                    for _ in range(probe_tokens - 1):
                        position = probe_seq_len - 1 + base_offset
                        _update_decode_state(probe_last_token, position, probe_seq_len,
                                             probe_decode_ctx)
                        decode_logits = _run_decode_step(model, probe_decode_ctx)
                        probe_last_token, _ = _sample_token(
                            decode_logits, temperature, top_p, rng,
                            seed=t * 1000 + probe_seq_len,
                        )
                        probe_last_token = probe_last_token.view(1)
                        probe_seq_len += 1

                # --- Extract KV from probe, compact aggregation region ---
                probe_kv_len = probe_seq_len - 1
                keys, values = _extract_kv(
                    kv_caches, candidate_blocks[0], probe_kv_len, block_size, num_layers
                )
                agg_end = agg_start + len(agg_ids)

                t_algo = time.time()
                c1_list, c2_list, _, _ = compact_kv_range(
                    keys, values,
                    compact_start=agg_start,
                    compact_end=agg_end,
                    target_ratio=compact_target_ratio,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    device=device,
                )
                algo_time = time.time() - t_algo

                compacted_len = c1_list[0].shape[0]

                # --- Inject compacted KV back into BASE blocks ---
                base_kv_len = base_seq_len
                base_keys, base_values = _extract_kv(
                    kv_caches, base_blocks, base_kv_len, block_size, num_layers
                )
                new_base_seq = _inject_compacted_range(
                    kv_caches, base_keys, base_values, c1_list, c2_list,
                    base_blocks, block_size,
                    compact_start=agg_start,
                    compact_end=agg_end,
                    num_layers=num_layers,
                    old_seq_len=base_kv_len,
                )

                tokens_removed = base_seq_len - (agg_start + compacted_len)
                base_offset += tokens_removed
                base_seq_len = agg_start + compacted_len + 1

                event = {
                    "step": t,
                    "agg_len": len(agg_ids),
                    "compacted_len": compacted_len,
                    "probe_tokens": probe_tokens,
                    "base_seq_len_after": base_seq_len,
                    "base_offset_after": base_offset,
                    "algo_time": round(algo_time, 3),
                }
            else:
                event = {
                    "step": t,
                    "agg_len": len(agg_ids),
                    "compacted_len": len(agg_ids),
                    "probe_tokens": 0,
                    "base_seq_len_after": base_seq_len,
                    "base_offset_after": base_offset,
                    "algo_time": 0.0,
                }

            compaction_events.append(event)

            logger.info(
                "RSA step %d: agg=%d tok, compacted=%d, base_seq=%d, offset=%d, "
                "algo=%.3fs, step=%.2fs",
                t, event["agg_len"], event["compacted_len"],
                base_seq_len, base_offset,
                event["algo_time"], time.time() - t_step,
            )

            # --- Fork & batch-generate N new candidates ---
            torch.cuda.empty_cache()
            num_base_blocks_now = (base_seq_len + block_size - 1) // block_size
            for n in range(N):
                _fork_kv_blocks(kv_caches, base_blocks, candidate_blocks[n],
                                num_base_blocks_now, num_layers)

            pop, pop_lps, _ = _batch_generate(
                model, kv_caches, candidate_blocks, N,
                base_seq_len, base_offset, max_tokens_per_candidate,
                block_size, num_layers, attn_layer_names, vllm_config,
                device, temperature, top_p, eos_token_id, rng,
                tokenizer=tokenizer,
            )
            populations.append(pop)

            logger.info("Step %d: generated %d candidates, lengths %s",
                         t + 1, N, [len(c) for c in pop])

        # --- Select best from final population ---
        total_time = time.time() - t_start
        final_pop = populations[-1]

        logger.info(
            "RSA DONE: %d steps, %d populations, %.2fs total",
            T, len(populations), total_time,
        )

        # Clean up
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return {
            "populations": populations,
            "best": final_pop[0] if final_pop else "",
            "diagnostics": {
                "prompt_len": len(prompt_ids),
                "N": N,
                "K": K,
                "T": T,
                "k_peers": k_peers,
                "num_populations": len(populations),
                "population_sizes": [len(p) for p in populations],
                "candidate_lengths": [[len(c) for c in p] for p in populations],
                "compaction_events": compaction_events,
                "total_time": round(total_time, 3),
            },
        }

    def parallel_generate(
        self,
        coordinator_prompt_ids: list[int],
        document_ids_list: list[list[int]],
        compact_target_ratio: float,
        probe_tokens: int,
        max_gen_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_id: int,
        compute_beta: bool = True,
        summary_prompt: str | None = None,
    ) -> dict:
        """Parallel reasoning: sequentially append each document to coordinator,
        compact in-place using the same pattern as RSA (position_offset, no
        RoPE re-rotation), then decode the coordinator.
        """
        t_start = time.time()
        K = len(document_ids_list)
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

        attn_layer_names = self._get_attn_layer_names(model)
        rng = torch.Generator(device=device)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.model, trust_remote_code=True)

        prompt_len = len(coordinator_prompt_ids)
        document_ids_list = [d for d in document_ids_list if len(d) > 0]
        K = len(document_ids_list)
        assert K > 0, "All documents are empty, nothing to compress"
        doc_lens = [len(d) for d in document_ids_list]
        max_doc_len = max(doc_lens)

        summary_ids = None
        summary_len = 0
        if summary_prompt is not None:
            summary_ids = tokenizer.encode(summary_prompt, add_special_tokens=False)
            summary_len = len(summary_ids)

        # --- Block budget ---
        max_compacted_per_doc = int(max_doc_len * compact_target_ratio) + 1
        # Coordinator temporarily holds the full document before compaction.
        # Worst case: prompt + (K-1)*compacted + full_doc + summary
        max_temp_len = (prompt_len + (K - 1) * max_compacted_per_doc
                        + max_doc_len + summary_len + 16)
        final_decode_len = (prompt_len + K * max_compacted_per_doc
                            + max_gen_tokens + 16)
        coordinator_max_len = max(max_temp_len, final_decode_len)
        coordinator_blocks_needed = (coordinator_max_len + block_size - 1) // block_size

        # Probe blocks: fork of coordinator for generating probe tokens
        probe_max_len = max_temp_len + probe_tokens + 16
        probe_blocks_needed = (probe_max_len + block_size - 1) // block_size

        total_blocks_needed = coordinator_blocks_needed + probe_blocks_needed
        free_blocks = self._find_free_blocks(num_total_blocks)
        logger.info(
            "parallel_generate: K=%d, prompt=%d, max_doc=%d, ratio=%.2f, "
            "probe=%d, coord_blocks=%d, probe_blocks=%d, free=%d",
            K, prompt_len, max_doc_len, compact_target_ratio,
            probe_tokens, coordinator_blocks_needed, probe_blocks_needed,
            len(free_blocks),
        )
        assert len(free_blocks) >= total_blocks_needed, (
            f"parallel_generate needs {total_blocks_needed} blocks, "
            f"only {len(free_blocks)} free"
        )

        coordinator_block_ids = free_blocks[:coordinator_blocks_needed]
        probe_block_ids = free_blocks[
            coordinator_blocks_needed:coordinator_blocks_needed + probe_blocks_needed
        ]

        # --- Phase 1: Prefill coordinator prompt ---
        input_ids_t = torch.tensor(
            coordinator_prompt_ids, dtype=torch.long, device=device)
        positions_t = torch.arange(prompt_len, dtype=torch.long, device=device)

        _run_prefill(
            model, input_ids_t, positions_t,
            seq_len=prompt_len,
            block_ids=coordinator_block_ids,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )

        # --- Phase 2: Sequentially append + compact each document ---
        # Follows the RSA pattern: append text via _prefill_append, fork for
        # probing, compact, inject back with _inject_compacted_range, track
        # position_offset.  No RoPE re-rotation needed.
        coord_seq_len = prompt_len
        position_offset = 0
        compacted_lens = []

        for k in range(K):
            t_doc = time.time()
            doc_ids = document_ids_list[k]
            doc_len = len(doc_ids)
            doc_start = coord_seq_len

            # 2a: Append-prefill document onto coordinator blocks
            doc_ids_t = torch.tensor(doc_ids, dtype=torch.long, device=device)
            doc_logits = _prefill_append(
                model, doc_ids_t,
                existing_seq_len=coord_seq_len,
                position_offset=position_offset,
                block_ids=coordinator_block_ids,
                block_size=block_size,
                attn_layer_names=attn_layer_names,
                vllm_config=vllm_config,
                device=device,
            )
            coord_seq_len += doc_len

            # 2b: Fork coordinator → probe blocks
            num_coord_blocks_now = (coord_seq_len + block_size - 1) // block_size
            _fork_kv_blocks(kv_caches, coordinator_block_ids, probe_block_ids,
                            num_coord_blocks_now, num_layers)
            probe_seq_len = coord_seq_len

            # 2c: Optionally append summary prompt onto PROBE blocks only
            if summary_ids is not None:
                summary_ids_t = torch.tensor(
                    summary_ids, dtype=torch.long, device=device)
                doc_logits = _prefill_append(
                    model, summary_ids_t,
                    existing_seq_len=probe_seq_len,
                    position_offset=position_offset,
                    block_ids=probe_block_ids,
                    block_size=block_size,
                    attn_layer_names=attn_layer_names,
                    vllm_config=vllm_config,
                    device=device,
                )
                probe_seq_len += summary_len

            # 2d: Generate probe tokens on probe blocks
            probe_decode_ctx = _DecodeContext.create(
                block_ids=probe_block_ids,
                block_size=block_size,
                attn_layer_names=attn_layer_names,
                vllm_config=vllm_config,
                device=device,
            )

            probe_last_token, _ = _sample_token(
                doc_logits[-1:], temperature, top_p, rng, seed=k * 1000)
            probe_last_token = probe_last_token.view(1)
            probe_seq_len += 1

            probe_decode_ctx.metadata.max_seq_len = probe_seq_len + probe_tokens

            with set_forward_context(
                probe_decode_ctx.attn_metadata_dict,
                probe_decode_ctx.vllm_config,
                virtual_engine=0, num_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=probe_decode_ctx.slot_mapping_ctx,
            ):
                for p in range(probe_tokens - 1):
                    position = probe_seq_len - 1 + position_offset
                    _update_decode_state(
                        probe_last_token, position, probe_seq_len,
                        probe_decode_ctx)
                    decode_logits = _run_decode_step(model, probe_decode_ctx)
                    probe_last_token, _ = _sample_token(
                        decode_logits, temperature, top_p, rng,
                        seed=k * 1000 + probe_seq_len,
                    )
                    probe_last_token = probe_last_token.view(1)
                    probe_seq_len += 1

            # 2e: Extract KV from probe blocks, compact document region
            probe_kv_len = probe_seq_len - 1
            keys, values = _extract_kv(
                kv_caches, probe_block_ids, probe_kv_len,
                block_size, num_layers)

            suffix_start = coord_seq_len + (summary_len if summary_ids else 0)
            suffix_queries = [
                k_layer[suffix_start:probe_kv_len].permute(1, 0, 2)
                for k_layer in keys
            ]

            doc_end = doc_start + doc_len
            t_algo = time.time()
            c1_list, c2_list, _, indices_list = compact_kv_range(
                keys, values,
                compact_start=doc_start,
                compact_end=doc_end,
                target_ratio=compact_target_ratio,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                device=device,
                compute_beta=False,
                suffix_queries=suffix_queries,
            )
            algo_time = time.time() - t_algo
            compacted_len = c1_list[0].shape[0]
            compacted_lens.append(compacted_len)

            # 2f: Inject compacted KV back into coordinator blocks
            coord_keys, coord_values = _extract_kv(
                kv_caches, coordinator_block_ids, coord_seq_len,
                block_size, num_layers)

            new_coord_seq = _inject_compacted_range(
                kv_caches, coord_keys, coord_values, c1_list, c2_list,
                coordinator_block_ids, block_size,
                compact_start=doc_start,
                compact_end=doc_end,
                num_layers=num_layers,
                old_seq_len=coord_seq_len,
            )

            tokens_removed = coord_seq_len - new_coord_seq
            position_offset += tokens_removed
            coord_seq_len = new_coord_seq

            logger.info(
                "Doc %d/%d: doc=%d tok, compacted=%d tok, "
                "coord_seq=%d, offset=%d, algo=%.3fs, total=%.2fs",
                k + 1, K, doc_len, compacted_len,
                coord_seq_len, position_offset,
                algo_time, time.time() - t_doc,
            )
            torch.cuda.empty_cache()

        # --- Phase 3: Coordinator decode ---
        # Uses the same boundary-recompute pattern as compact_generate/RSA:
        # +1 reserves a slot for the boundary token whose KV gets recomputed
        # by the first decode step, attending to the full compacted context.
        coordinator_seq_len = coord_seq_len
        current_seq_len = coord_seq_len + 1

        decode_ctx = _DecodeContext.create(
            block_ids=coordinator_block_ids,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )
        decode_ctx.metadata.max_seq_len = coordinator_max_len

        # Boundary token: dummy token to trigger first decode and get logits
        # (same pattern as _batch_generate with first_logits=None in RSA)
        boundary_token = torch.zeros(1, dtype=torch.long, device=device)

        _update_decode_state(
            boundary_token,
            position=coord_seq_len + position_offset,
            seq_len=current_seq_len,
            ctx=decode_ctx,
        )

        with set_forward_context(
            decode_ctx.attn_metadata_dict, decode_ctx.vllm_config,
            virtual_engine=0, num_tokens=1,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=decode_ctx.slot_mapping_ctx,
        ):
            first_logits = _run_decode_step(model, decode_ctx)
            first_token, first_lp = _sample_token(
                first_logits, temperature, top_p, rng, seed=0)

            all_token_ids = [first_token.item()]
            all_logprobs = [first_lp.item()]
            last_token_gpu = first_token.view(1)
            current_seq_len += 1

            for step in range(max_gen_tokens - 1):
                if all_token_ids[-1] == eos_token_id:
                    break

                position = current_seq_len - 1 + position_offset
                _update_decode_state(
                    last_token_gpu, position, current_seq_len, decode_ctx)

                decode_logits = _run_decode_step(model, decode_ctx)
                token, logprob = _sample_token(
                    decode_logits, temperature, top_p, rng, seed=step + 1)

                all_token_ids.append(token.item())
                all_logprobs.append(logprob.item())
                last_token_gpu = token.view(1)
                current_seq_len += 1

        final_text = tokenizer.decode(all_token_ids, skip_special_tokens=True)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        total_time = time.time() - t_start
        logger.info(
            "parallel_generate DONE: K=%d, coord_seq=%d, offset=%d, "
            "gen=%d tok, %.2fs total",
            K, coordinator_seq_len, position_offset,
            len(all_token_ids), total_time,
        )

        return {
            "all_token_ids": all_token_ids,
            "all_logprobs": all_logprobs,
            "final_text": final_text,
            "diagnostics": {
                "prompt_len": prompt_len,
                "K": K,
                "doc_lens": doc_lens,
                "compacted_lens": compacted_lens,
                "coordinator_seq_len": coordinator_seq_len,
                "position_offset": position_offset,
                "total_tokens": len(all_token_ids),
                "total_time": round(total_time, 3),
            },
        }

    def parallel_generate_fused(
        self,
        prompt_ids: list[int],
        K: int,
        max_candidate_tokens: int,
        compact_target_ratio: float,
        max_gen_tokens: int,
        temperature: float,
        top_p: float,
        eos_token_id: int,
        compute_beta: bool = True,
        probe_tokens: int = 256,
        synthesis_prompt: str | None = None,
        coordinator_prompt_ids: list[int] | None = None,
    ) -> dict:
        """Fused parallel reasoning: generate K candidates internally, then
        re-encode each into the coordinator via _prefill_append + compact.

        Candidates are generated from the shared prompt via _batch_generate.
        The coordinator gets a separate prefill (optionally with a different
        system prompt), and each candidate's TEXT is append-prefilled and
        compacted in-place — following the RSA pattern (no RoPE re-rotation).
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
        num_kv_heads = kv_shape[3]
        head_size = kv_shape[4]

        attn_layer_names = self._get_attn_layer_names(model)
        rng = torch.Generator(device=device)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.model, trust_remote_code=True)

        prompt_len = len(prompt_ids)
        coord_prompt_ids = coordinator_prompt_ids or prompt_ids
        coord_prompt_len = len(coord_prompt_ids)

        # --- Block budget ---
        max_compacted_per_doc = int(max_candidate_tokens * compact_target_ratio) + 1
        # Coordinator temporarily holds full candidate text before compaction
        max_temp_len = (coord_prompt_len + (K - 1) * max_compacted_per_doc
                        + max_candidate_tokens + 16)
        final_decode_len = (coord_prompt_len + K * max_compacted_per_doc
                            + max_gen_tokens + 16)
        coordinator_max_len = max(max_temp_len, final_decode_len)
        coordinator_blocks = (coordinator_max_len + block_size - 1) // block_size

        candidate_max_len = prompt_len + max_candidate_tokens + 16
        candidate_blocks_each = (candidate_max_len + block_size - 1) // block_size

        # Probe blocks: fork of coordinator for probing
        probe_max_len = max_temp_len + probe_tokens + 16
        probe_blocks_needed = (probe_max_len + block_size - 1) // block_size

        total_blocks_needed = (coordinator_blocks + K * candidate_blocks_each
                               + probe_blocks_needed)
        free_blocks = self._find_free_blocks(num_total_blocks)
        logger.info(
            "parallel_generate_fused: K=%d, prompt=%d, coord_prompt=%d, "
            "max_cand_tok=%d, ratio=%.2f, coord_blocks=%d, cand_blocks=%d, "
            "probe_blocks=%d, free=%d",
            K, prompt_len, coord_prompt_len, max_candidate_tokens,
            compact_target_ratio, coordinator_blocks, candidate_blocks_each,
            probe_blocks_needed, len(free_blocks),
        )
        assert len(free_blocks) >= total_blocks_needed, (
            f"parallel_generate_fused needs {total_blocks_needed} blocks, "
            f"only {len(free_blocks)} free"
        )

        coordinator_block_ids = free_blocks[:coordinator_blocks]
        candidate_block_ids = []
        offset = coordinator_blocks
        for _k in range(K):
            candidate_block_ids.append(
                free_blocks[offset:offset + candidate_blocks_each])
            offset += candidate_blocks_each
        probe_block_ids = free_blocks[offset:offset + probe_blocks_needed]

        # --- Phase 1: Prefill shared prompt for candidate generation ---
        input_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        positions_t = torch.arange(prompt_len, dtype=torch.long, device=device)

        prefill_logits = _run_prefill(
            model, input_ids_t, positions_t,
            seq_len=prompt_len,
            block_ids=coordinator_block_ids,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )

        # --- Phase 2: Fork prefix → K candidates, batch-generate ---
        num_prefix_blocks = (prompt_len + block_size - 1) // block_size
        for k in range(K):
            _fork_kv_blocks(kv_caches, coordinator_block_ids,
                            candidate_block_ids[k],
                            num_prefix_blocks, num_layers)

        cand_texts, cand_logprobs, cand_token_ids = _batch_generate(
            model, kv_caches, candidate_block_ids, K,
            prompt_len, 0, max_candidate_tokens,
            block_size, num_layers, attn_layer_names, vllm_config,
            device, temperature, top_p, eos_token_id, rng,
            first_logits=prefill_logits, tokenizer=tokenizer,
        )
        gen_lens = [len(ids) for ids in cand_token_ids]

        logger.info(
            "Phase 2 done: generated %d candidates, lengths %s",
            K, gen_lens)
        torch.cuda.empty_cache()

        # --- Phase 3: Re-prefill coordinator with synthesis prompt ---
        coord_ids_t = torch.tensor(
            coord_prompt_ids, dtype=torch.long, device=device)
        coord_positions_t = torch.arange(
            coord_prompt_len, dtype=torch.long, device=device)
        _run_prefill(
            model, coord_ids_t, coord_positions_t,
            seq_len=coord_prompt_len,
            block_ids=coordinator_block_ids,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )

        # --- Phase 4: Re-encode each candidate text into coordinator,
        #     compact in-place (RSA pattern) ---
        coord_seq_len = coord_prompt_len
        position_offset = 0
        compacted_lens = []

        for k in range(K):
            t_doc = time.time()
            doc_ids = cand_token_ids[k]
            doc_len = len(doc_ids)
            doc_start = coord_seq_len

            # Append-prefill candidate text onto coordinator
            doc_ids_t = torch.tensor(doc_ids, dtype=torch.long, device=device)
            doc_logits = _prefill_append(
                model, doc_ids_t,
                existing_seq_len=coord_seq_len,
                position_offset=position_offset,
                block_ids=coordinator_block_ids,
                block_size=block_size,
                attn_layer_names=attn_layer_names,
                vllm_config=vllm_config,
                device=device,
            )
            coord_seq_len += doc_len

            # Fork coordinator → probe blocks for probing
            num_coord_blocks_now = (
                coord_seq_len + block_size - 1) // block_size
            _fork_kv_blocks(kv_caches, coordinator_block_ids,
                            probe_block_ids,
                            num_coord_blocks_now, num_layers)

            # Generate probe tokens on probe blocks
            probe_decode_ctx = _DecodeContext.create(
                block_ids=probe_block_ids,
                block_size=block_size,
                attn_layer_names=attn_layer_names,
                vllm_config=vllm_config,
                device=device,
            )

            probe_last_token, _ = _sample_token(
                doc_logits[-1:], temperature, top_p, rng,
                seed=k * 1000)
            probe_last_token = probe_last_token.view(1)
            probe_seq_len = coord_seq_len + 1

            probe_decode_ctx.metadata.max_seq_len = (
                probe_seq_len + probe_tokens)

            with set_forward_context(
                probe_decode_ctx.attn_metadata_dict,
                probe_decode_ctx.vllm_config,
                virtual_engine=0, num_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=probe_decode_ctx.slot_mapping_ctx,
            ):
                for p in range(probe_tokens - 1):
                    position = probe_seq_len - 1 + position_offset
                    _update_decode_state(
                        probe_last_token, position, probe_seq_len,
                        probe_decode_ctx)
                    decode_logits = _run_decode_step(
                        model, probe_decode_ctx)
                    probe_last_token, _ = _sample_token(
                        decode_logits, temperature, top_p, rng,
                        seed=k * 1000 + probe_seq_len,
                    )
                    probe_last_token = probe_last_token.view(1)
                    probe_seq_len += 1

            # Extract KV from probe blocks, compact candidate region
            probe_kv_len = probe_seq_len - 1
            keys, values = _extract_kv(
                kv_caches, probe_block_ids, probe_kv_len,
                block_size, num_layers)

            suffix_queries = [
                k_layer[coord_seq_len:probe_kv_len].permute(1, 0, 2)
                for k_layer in keys
            ]

            doc_end = doc_start + doc_len
            c1_list, c2_list, _, indices_list = compact_kv_range(
                keys, values,
                compact_start=doc_start,
                compact_end=doc_end,
                target_ratio=compact_target_ratio,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                device=device,
                compute_beta=False,
                suffix_queries=suffix_queries,
            )
            compacted_len = c1_list[0].shape[0]
            compacted_lens.append(compacted_len)

            # Inject compacted KV back into coordinator
            coord_keys, coord_values = _extract_kv(
                kv_caches, coordinator_block_ids, coord_seq_len,
                block_size, num_layers)

            new_coord_seq = _inject_compacted_range(
                kv_caches, coord_keys, coord_values, c1_list, c2_list,
                coordinator_block_ids, block_size,
                compact_start=doc_start,
                compact_end=doc_end,
                num_layers=num_layers,
                old_seq_len=coord_seq_len,
            )

            tokens_removed = coord_seq_len - new_coord_seq
            position_offset += tokens_removed
            coord_seq_len = new_coord_seq

            logger.info(
                "Candidate %d/%d: gen=%d tok, compacted=%d tok, "
                "coord_seq=%d, offset=%d, %.2fs",
                k + 1, K, doc_len, compacted_len,
                coord_seq_len, position_offset, time.time() - t_doc,
            )
            torch.cuda.empty_cache()

        # --- Phase 5: Coordinator decode ---
        coordinator_seq_len = coord_seq_len
        current_seq_len = coord_seq_len + 1

        decode_ctx = _DecodeContext.create(
            block_ids=coordinator_block_ids,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )
        decode_ctx.metadata.max_seq_len = coordinator_max_len

        boundary_token = torch.zeros(1, dtype=torch.long, device=device)
        _update_decode_state(
            boundary_token,
            position=coord_seq_len + position_offset,
            seq_len=current_seq_len,
            ctx=decode_ctx,
        )

        with set_forward_context(
            decode_ctx.attn_metadata_dict, decode_ctx.vllm_config,
            virtual_engine=0, num_tokens=1,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=decode_ctx.slot_mapping_ctx,
        ):
            first_logits = _run_decode_step(model, decode_ctx)
            first_token, first_lp = _sample_token(
                first_logits, temperature, top_p, rng, seed=0)

            synth_token_ids = [first_token.item()]
            synth_logprobs = [first_lp.item()]
            last_token_gpu = first_token.view(1)
            current_seq_len += 1

            for step in range(max_gen_tokens - 1):
                if synth_token_ids[-1] == eos_token_id:
                    break

                position = current_seq_len - 1 + position_offset
                _update_decode_state(
                    last_token_gpu, position, current_seq_len, decode_ctx)

                decode_logits = _run_decode_step(model, decode_ctx)
                token, logprob = _sample_token(
                    decode_logits, temperature, top_p, rng, seed=step + 1)

                synth_token_ids.append(token.item())
                synth_logprobs.append(logprob.item())
                last_token_gpu = token.view(1)
                current_seq_len += 1

        final_text = tokenizer.decode(synth_token_ids, skip_special_tokens=True)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        total_time = time.time() - t_start
        logger.info(
            "parallel_generate_fused DONE: K=%d, coord_seq=%d, offset=%d, "
            "gen=%d tok, %.2fs total",
            K, coordinator_seq_len, position_offset,
            len(synth_token_ids), total_time,
        )

        return {
            "all_token_ids": synth_token_ids,
            "all_logprobs": synth_logprobs,
            "final_text": final_text,
            "candidates": [
                {"text": cand_texts[k], "gen_len": gen_lens[k]}
                for k in range(K)
            ],
            "diagnostics": {
                "prompt_len": prompt_len,
                "K": K,
                "gen_lens": gen_lens,
                "compacted_lens": compacted_lens,
                "coordinator_seq_len": coordinator_seq_len,
                "position_offset": position_offset,
                "total_tokens": len(synth_token_ids),
                "total_time": round(total_time, 3),
            },
        }

    def compact_session_create(
        self,
        session_id: str,
        prompt_ids: list[int],
        max_kv_len: int,
        max_response_tokens: int,
        eos_token_id: int,
        compact_target_ratio: float = 0.25,
        compact_window: int | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        compaction_mode: str = "attention_matching",
        use_suffix_queries: bool = True,
        n_max_turns: int = -1,
        n_preserved_turns: int = 0,
    ) -> dict:
        """Create a persistent KV session, prefill prompt, generate first response."""
        if not hasattr(self, "_sessions"):
            self._sessions = {}
        assert session_id not in self._sessions, f"Session {session_id!r} already exists"
        assert n_max_turns == -1 or n_preserved_turns < n_max_turns, (
            f"n_preserved_turns ({n_preserved_turns}) must be < n_max_turns ({n_max_turns})"
        )
        if n_max_turns >= 0 and compact_window is not None:
            logger.warning(
                "compact_window=%d is ignored when n_max_turns is set. "
                "Turn-based compaction always compacts the full history before the preserved turns.",
                compact_window,
            )

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

        # Lazy allocation: only allocate blocks for the first turn (prompt +
        # response). Additional blocks are acquired at each compact_session_step.
        initial_len = len(prompt_ids) + max_response_tokens
        blocks_needed = math.ceil(initial_len / block_size)
        free_blocks = self._find_free_blocks(num_total_blocks)
        if len(free_blocks) < blocks_needed:
            self._expire_stale_sessions()
            free_blocks = self._find_free_blocks(num_total_blocks)
        if len(free_blocks) < blocks_needed:
            return {
                "error": "blocks_exhausted",
                "blocks_needed": blocks_needed,
                "blocks_free": len(free_blocks),
                "active_sessions": len(getattr(self, "_sessions", {})),
            }
        my_blocks = free_blocks[:blocks_needed]

        attn_layer_names = self._get_attn_layer_names(model)
        rng = torch.Generator(device=device)

        prompt_len = len(prompt_ids)
        input_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        positions_t = torch.arange(prompt_len, dtype=torch.long, device=device)

        logits = _run_prefill(
            model, input_ids_t, positions_t,
            seq_len=prompt_len,
            block_ids=my_blocks,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )

        current_seq_len = prompt_len
        position_offset = 0

        first_token, first_logprob = _sample_token(logits[-1:], temperature, top_p, rng, seed=0)
        all_token_ids = [first_token.item()]
        all_logprobs = [first_logprob.item()]
        current_seq_len += 1
        last_token_gpu = first_token.view(1)

        decode_ctx = _DecodeContext.create(
            block_ids=my_blocks,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )
        decode_ctx.metadata.max_seq_len = initial_len

        compaction_events = []

        with set_forward_context(
            decode_ctx.attn_metadata_dict, decode_ctx.vllm_config,
            virtual_engine=0, num_tokens=1,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=decode_ctx.slot_mapping_ctx,
        ):
            seg_count = 0
            while len(all_token_ids) < max_response_tokens:
                if all_token_ids[-1] == eos_token_id:
                    break

                if current_seq_len - 1 >= max_kv_len:
                    kv_len = current_seq_len - 1
                    asst_len = kv_len - prompt_len
                    window = min(compact_window or asst_len, asst_len)
                    suffix_len = asst_len - window

                    keys, values = _extract_kv(kv_caches, my_blocks, kv_len, block_size, num_layers)

                    if compaction_mode == "markovian":
                        kv_dim = (0, keys[0].shape[1], keys[0].shape[2])
                        c1_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in keys]
                        c2_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in values]
                        compacted_prefix_len = 0
                    else:
                        c1_list, c2_list, _, _ = compact_kv(
                            keys, values, prompt_len, compact_target_ratio,
                            num_kv_heads, head_size, device,
                            compact_window=window,
                        )
                        compacted_prefix_len = c1_list[0].shape[0]

                    new_seq_len = prompt_len + compacted_prefix_len + suffix_len
                    _inject_compacted_kv(
                        kv_caches, keys, values, c1_list, c2_list,
                        my_blocks, block_size, prompt_len, num_layers,
                        old_seq_len=kv_len, compact_window=window,
                    )
                    position_offset += kv_len - new_seq_len
                    current_seq_len = new_seq_len + 1
                    compaction_events.append({
                        "kv_len_before": kv_len, "kv_len_after": new_seq_len,
                    })

                position = current_seq_len - 1 + position_offset
                _update_decode_state(last_token_gpu, position, current_seq_len, decode_ctx)
                decode_logits = _run_decode_step(model, decode_ctx)

                seed = seg_count
                token, logprob = _sample_token(decode_logits, temperature, top_p, rng, seed=seed)
                all_token_ids.append(token.item())
                all_logprobs.append(logprob.item())
                last_token_gpu = token.view(1)
                current_seq_len += 1
                seg_count += 1

        self._shrink_blocks(my_blocks, current_seq_len, block_size)

        self._sessions[session_id] = SessionState(
            block_ids=my_blocks,
            current_seq_len=current_seq_len,
            prompt_len=prompt_len,
            position_offset=position_offset,
            max_kv_len=max_kv_len,
            compact_target_ratio=compact_target_ratio,
            compact_window=compact_window,
            compaction_mode=compaction_mode,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            use_suffix_queries=use_suffix_queries,
            turn_user_lens=[0],
            turn_asst_lens=[len(all_token_ids)],
            n_max_turns=n_max_turns,
            n_preserved_turns=n_preserved_turns,
            compacted_prefix_end=prompt_len,
            last_access=time.monotonic(),
        )

        logger.info(
            "compact_session_create: session=%s, prompt_len=%d, response_tokens=%d, "
            "current_seq_len=%d, compactions=%d",
            session_id, prompt_len, len(all_token_ids),
            current_seq_len, len(compaction_events),
        )

        return {
            "session_id": session_id,
            "all_token_ids": all_token_ids,
            "all_logprobs": all_logprobs,
            "current_seq_len": current_seq_len,
            "diagnostics": {"compaction_events": compaction_events},
        }

    def compact_session_step(
        self,
        session_id: str,
        new_token_ids: list[int],
        max_response_tokens: int,
    ) -> dict:
        """Append new_token_ids to existing session KV and generate a response.

        new_token_ids = [boundary_token_from_prev_step] + [new_user_turn_tokens]
        """
        state = self._sessions[session_id]
        state.last_access = time.monotonic()

        model = self._get_model()
        device = self._get_device()
        vllm_config = self.model_runner.vllm_config

        kv_caches = self.model_runner.kv_caches
        num_layers = len(kv_caches)
        kv_shape = kv_caches[0].shape
        block_size = kv_shape[2]
        num_kv_heads = kv_shape[3]
        head_size = kv_shape[4]

        attn_layer_names = self._get_attn_layer_names(model)
        rng = torch.Generator(device=device)
        num_total_blocks = kv_shape[1]

        # existing_seq_len is the position where boundary token's KV will be written
        existing_seq_len = state.current_seq_len - 1
        new_ids_t = torch.tensor(new_token_ids, dtype=torch.long, device=device)

        # Grow blocks to cover append + response generation for this turn
        append_target = existing_seq_len + len(new_token_ids) + 1 + max_response_tokens
        err = self._grow_blocks(state, append_target, block_size, num_total_blocks)
        if err:
            return err

        logits = _prefill_append(
            model, new_ids_t,
            existing_seq_len=existing_seq_len,
            position_offset=state.position_offset,
            block_ids=state.block_ids,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )

        current_seq_len = existing_seq_len + len(new_token_ids)
        position_offset = state.position_offset

        # Sample boundary token for start of generation
        boundary_token, boundary_logprob = _sample_token(logits[-1:], state.temperature, state.top_p, rng, seed=0)
        all_token_ids = [boundary_token.item()]
        all_logprobs = [boundary_logprob.item()]
        current_seq_len += 1
        last_token_gpu = boundary_token.view(1)

        decode_ctx = _DecodeContext.create(
            block_ids=state.block_ids,
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )
        decode_ctx.metadata.max_seq_len = append_target

        compaction_events = []

        # Pre-decode compaction: compact if already over budget after append
        if state.n_max_turns < 0 and current_seq_len - 1 >= state.max_kv_len:
            kv_len = current_seq_len - 1
            asst_len = kv_len - state.prompt_len
            window = min(state.compact_window or asst_len, asst_len)
            suffix_len = asst_len - window

            keys, values = _extract_kv(kv_caches, state.block_ids, kv_len, block_size, num_layers)

            indices_list = None
            if state.compaction_mode == "markovian":
                kv_dim = (0, keys[0].shape[1], keys[0].shape[2])
                c1_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in keys]
                c2_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in values]
                compacted_prefix_len = 0
            else:
                compact_seed = state.prompt_len * 10000 + state.compaction_count
                c1_list, c2_list, _, indices_list = compact_kv(
                    keys, values, state.prompt_len, state.compact_target_ratio,
                    num_kv_heads, head_size, device,
                    compact_window=window,
                    seed=compact_seed,
                )
                compacted_prefix_len = c1_list[0].shape[0]
            state.compaction_count += 1

            new_seq_len = state.prompt_len + compacted_prefix_len + suffix_len
            _inject_compacted_kv(
                kv_caches, keys, values, c1_list, c2_list,
                state.block_ids, block_size, state.prompt_len, num_layers,
                old_seq_len=kv_len, compact_window=window,
            )
            position_offset += kv_len - new_seq_len
            current_seq_len = new_seq_len + 1
            compaction_events.append({
                "kv_len_before": kv_len,
                "kv_len_after": new_seq_len,
                "compaction_indices": [idx.cpu().tolist() for idx in indices_list] if indices_list else None,
            })

        with set_forward_context(
            decode_ctx.attn_metadata_dict, decode_ctx.vllm_config,
            virtual_engine=0, num_tokens=1,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=decode_ctx.slot_mapping_ctx,
        ):
            seg_count = 0
            while len(all_token_ids) < max_response_tokens:
                if all_token_ids[-1] == state.eos_token_id:
                    break

                if state.n_max_turns < 0 and current_seq_len - 1 >= state.max_kv_len:
                    kv_len = current_seq_len - 1
                    asst_len = kv_len - state.prompt_len
                    window = min(state.compact_window or asst_len, asst_len)
                    suffix_len = asst_len - window

                    keys, values = _extract_kv(kv_caches, state.block_ids, kv_len, block_size, num_layers)

                    indices_list = None
                    if state.compaction_mode == "markovian":
                        kv_dim = (0, keys[0].shape[1], keys[0].shape[2])
                        c1_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in keys]
                        c2_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in values]
                        compacted_prefix_len = 0
                    else:
                        compact_seed = state.prompt_len * 10000 + state.compaction_count
                        c1_list, c2_list, _, indices_list = compact_kv(
                            keys, values, state.prompt_len, state.compact_target_ratio,
                            num_kv_heads, head_size, device,
                            compact_window=window,
                            seed=compact_seed,
                        )
                        compacted_prefix_len = c1_list[0].shape[0]
                    state.compaction_count += 1

                    new_seq_len = state.prompt_len + compacted_prefix_len + suffix_len
                    _inject_compacted_kv(
                        kv_caches, keys, values, c1_list, c2_list,
                        state.block_ids, block_size, state.prompt_len, num_layers,
                        old_seq_len=kv_len, compact_window=window,
                    )
                    position_offset += kv_len - new_seq_len
                    current_seq_len = new_seq_len + 1
                    compaction_events.append({
                        "kv_len_before": kv_len,
                        "kv_len_after": new_seq_len,
                        "compaction_indices": [idx.cpu().tolist() for idx in indices_list] if indices_list else None,
                    })

                position = current_seq_len - 1 + position_offset
                _update_decode_state(last_token_gpu, position, current_seq_len, decode_ctx)
                decode_logits = _run_decode_step(model, decode_ctx)

                seed = seg_count
                token, logprob = _sample_token(decode_logits, state.temperature, state.top_p, rng, seed=seed)
                all_token_ids.append(token.item())
                all_logprobs.append(logprob.item())
                last_token_gpu = token.view(1)
                current_seq_len += 1
                seg_count += 1

        # Record completed turn
        user_len = len(new_token_ids) - 1  # minus boundary token
        asst_len = len(all_token_ids)
        state.turn_user_lens.append(user_len)
        state.turn_asst_lens.append(asst_len)

        # Turn-based compaction: fires when accumulated turns reach n_max_turns
        if state.n_max_turns >= 0 and len(state.turn_asst_lens) >= state.n_max_turns:
            kv_len = current_seq_len - 1  # boundary token not yet in KV

            n_protect = state.n_preserved_turns
            protected_total = (
                sum(state.turn_user_lens[-n_protect:]) + sum(state.turn_asst_lens[-n_protect:])
                if n_protect > 0 else 0
            )
            compact_end = kv_len - protected_total

            # attention_matching_full: only compact NEW turns (stack compacted blocks).
            # Other modes: re-compact from prompt_len each time.
            if state.compaction_mode == "attention_matching_full":
                compact_start = state.compacted_prefix_end
            else:
                compact_start = state.prompt_len

            if compact_end > compact_start:
                keys, values = _extract_kv(kv_caches, state.block_ids, kv_len, block_size, num_layers)
                turn_compact_window = compact_end - compact_start

                indices_list = None
                if state.compaction_mode == "markovian":
                    kv_dim = (0, keys[0].shape[1], keys[0].shape[2])
                    c1_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in keys]
                    c2_list = [torch.empty(kv_dim, dtype=keys[0].dtype, device=device) for _ in values]
                    compacted_prefix_len = 0
                else:
                    # Use K-vectors of the most recent assistant response as importance queries.
                    # Subtract 1 from asst_len because EOS is counted in turn_asst_lens but
                    # not yet written to KV (it's the boundary token for the next step).
                    asst_len_in_kv = state.turn_asst_lens[-1] - 1
                    asst_start = kv_len - asst_len_in_kv
                    query_vecs = [k[asst_start:kv_len].permute(1, 0, 2) for k in keys]

                    compact_seed = state.prompt_len * 10000 + state.compaction_count
                    c1_list, c2_list, _, indices_list = compact_kv_range(
                        keys, values,
                        compact_start=compact_start,
                        compact_end=compact_end,
                        target_ratio=state.compact_target_ratio,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        device=device,
                        suffix_queries=query_vecs,
                        seed=compact_seed,
                    )
                    compacted_prefix_len = c1_list[0].shape[0]
                state.compaction_count += 1
                new_seq_len = compact_start + compacted_prefix_len + (kv_len - compact_end)

                _inject_compacted_kv(
                    kv_caches, keys, values, c1_list, c2_list,
                    state.block_ids, block_size, compact_start, num_layers,
                    old_seq_len=kv_len,
                    compact_window=turn_compact_window,
                )
                position_offset += kv_len - new_seq_len
                current_seq_len = new_seq_len + 1

                if state.compaction_mode == "attention_matching_full":
                    state.compacted_prefix_end = compact_start + compacted_prefix_len

                compaction_events.append({
                    "kv_len_before": kv_len,
                    "kv_len_after": new_seq_len,
                    "compaction_indices": [idx.cpu().tolist() for idx in indices_list] if indices_list else None,
                    "compact_window": turn_compact_window,
                })

            state.turn_user_lens = state.turn_user_lens[-n_protect:] if n_protect > 0 else []
            state.turn_asst_lens = state.turn_asst_lens[-n_protect:] if n_protect > 0 else []

        state.current_seq_len = current_seq_len
        state.position_offset = position_offset
        self._shrink_blocks(state.block_ids, current_seq_len, block_size)

        logger.info(
            "compact_session_step: session=%s, new_tokens=%d, response_tokens=%d, "
            "current_seq_len=%d, blocks=%d, compactions=%d",
            session_id, len(new_token_ids), len(all_token_ids),
            current_seq_len, len(state.block_ids), len(compaction_events),
        )

        return {
            "all_token_ids": all_token_ids,
            "all_logprobs": all_logprobs,
            "current_seq_len": current_seq_len,
            "diagnostics": {"compaction_events": compaction_events},
        }

    def compact_session_delete(self, session_id: str) -> None:
        """Free session blocks by removing the session state."""
        getattr(self, "_sessions", {}).pop(session_id, None)

    def _expire_stale_sessions(self) -> None:
        """Remove sessions that haven't been accessed within SESSION_TTL_SECONDS."""
        sessions = getattr(self, "_sessions", {})
        now = time.monotonic()
        stale = [sid for sid, s in sessions.items() if now - s.last_access > SESSION_TTL_SECONDS]
        for sid in stale:
            logger.info("Expiring stale session %s (idle %.0fs)", sid, now - sessions[sid].last_access)
            del sessions[sid]

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
        for state in getattr(self, "_sessions", {}).values():
            used.update(state.block_ids)
        return sorted(b for b in range(num_total_blocks) if b not in used)

    def _grow_blocks(
        self,
        state: SessionState,
        target_seq_len: int,
        block_size: int,
        num_total_blocks: int,
    ) -> dict | None:
        """Ensure state.block_ids covers target_seq_len. Returns error dict on failure."""
        blocks_needed = math.ceil(target_seq_len / block_size)
        if blocks_needed <= len(state.block_ids):
            return None
        deficit = blocks_needed - len(state.block_ids)
        free = self._find_free_blocks(num_total_blocks)
        if len(free) < deficit:
            self._expire_stale_sessions()
            free = self._find_free_blocks(num_total_blocks)
        if len(free) < deficit:
            return {
                "error": "blocks_exhausted",
                "blocks_needed": deficit,
                "blocks_free": len(free),
                "active_sessions": len(getattr(self, "_sessions", {})),
            }
        state.block_ids.extend(free[:deficit])
        return None

    @staticmethod
    def _shrink_blocks(block_ids: list[int], seq_len: int, block_size: int) -> None:
        """Truncate block_ids in place to release blocks beyond seq_len."""
        needed = math.ceil(seq_len / block_size) if seq_len > 0 else 1
        if len(block_ids) > needed:
            del block_ids[needed:]

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

    return model.compute_logits(hidden_states[-1:])


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


@torch.inference_mode()
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
# RoPE re-rotation
# ---------------------------------------------------------------------------

def _rerotate_rope_keys(
    c1: torch.Tensor,
    indices: torch.Tensor,
    compact_start: int,
    target_start: int,
    rope_theta: float,
    head_size: int,
) -> torch.Tensor:
    """Re-rotate compacted keys from original RoPE positions to coordinator positions.

    When compacted keys are injected into a different sequence (e.g. coordinator
    in parallel_generate), their baked-in RoPE encoding reflects the *source*
    positions, not the *target* positions.  This applies a delta rotation
    RoPE(target - original) to each (head, token) so that downstream attention
    sees correct relative distances.

    Args:
        c1: (target_len, num_kv_heads, head_size) — compacted keys with original RoPE.
        indices: (num_kv_heads, target_len) — region-relative indices returned by
            compact_kv_range (i.e. relative to compact_start).
        compact_start: Absolute sequence position where the compact region begins
            in the source sequence.
        target_start: Absolute sequence position where these keys will be placed
            in the target (coordinator) sequence.
        rope_theta: RoPE base frequency from model config.
        head_size: Dimension per attention head (= rotary_dim for Qwen3).

    Returns:
        Re-rotated keys, same shape and dtype as c1.
    """
    device = c1.device
    dtype = c1.dtype
    target_len, num_kv_heads, _ = c1.shape

    # Original absolute positions: (num_kv_heads, target_len)
    original_pos = compact_start + indices.to(device=device)
    # Target absolute positions: same for all heads
    target_pos = target_start + torch.arange(target_len, device=device)
    # Delta per (head, token)
    delta = target_pos.unsqueeze(0) - original_pos  # (H, T)

    # Standard RoPE inv_freq
    inv_freq = 1.0 / (
        rope_theta ** (
            torch.arange(0, head_size, 2, device=device, dtype=torch.float32)
            / head_size
        )
    )  # (D/2,)

    # Angles: (H, T, D/2)
    angles = delta.unsqueeze(-1).float() * inv_freq

    cos_d = torch.cos(angles)
    sin_d = torch.sin(angles)
    # Duplicate for split-half layout: [cos, cos] matching [k_first_half, k_second_half]
    cos_full = torch.cat([cos_d, cos_d], dim=-1)  # (H, T, D)
    sin_full = torch.cat([sin_d, sin_d], dim=-1)

    # Rotate: (T, H, D) -> (H, T, D)
    k = c1.permute(1, 0, 2).float()
    d = k.shape[-1]
    k1, k2 = k[..., :d // 2], k[..., d // 2:]
    k_rotated = torch.cat([-k2, k1], dim=-1)

    k_new = k * cos_full + k_rotated * sin_full
    return k_new.permute(1, 0, 2).to(dtype)


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


def _fork_kv_blocks(
    kv_caches: list[torch.Tensor],
    src_blocks: list[int],
    dst_blocks: list[int],
    num_blocks_used: int,
    num_layers: int,
) -> None:
    """Copy KV blocks from src to dst (block-level, no contiguous extraction)."""
    src = src_blocks[:num_blocks_used]
    dst = dst_blocks[:num_blocks_used]
    for layer_idx in range(num_layers):
        kv = kv_caches[layer_idx]
        kv[0, dst] = kv[0, src]
        kv[1, dst] = kv[1, src]


def _prefill_append(
    model,
    new_token_ids: torch.Tensor,
    existing_seq_len: int,
    position_offset: int,
    block_ids: list[int],
    block_size: int,
    attn_layer_names: list[str],
    vllm_config,
    device: torch.device,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Append-prefill: run new tokens through model, attending to existing KV cache.

    Like _run_prefill but positions and slots start at existing_seq_len.
    FlashAttention sees the full sequence (old + new) via seq_lens but only
    new tokens are queries.

    When n_new > chunk_size, processes in chunks to limit peak activation memory.
    Each chunk writes its KV into the cache; only the last chunk's logits are returned.
    """
    n_new = new_token_ids.shape[0]

    if n_new <= chunk_size:
        return _prefill_append_chunk(
            model, new_token_ids, existing_seq_len, position_offset,
            block_ids, block_size, attn_layer_names, vllm_config, device,
        )

    # Chunked prefill: process in pieces to avoid OOM on large sequences.
    # Each chunk writes KV into the cache. Only the last chunk's logits are kept.
    chunks = list(range(0, n_new, chunk_size))
    for i, start in enumerate(chunks):
        end = min(start + chunk_size, n_new)
        chunk_ids = new_token_ids[start:end]
        chunk_seq_start = existing_seq_len + start

        logits = _prefill_append_chunk(
            model, chunk_ids, chunk_seq_start, position_offset,
            block_ids, block_size, attn_layer_names, vllm_config, device,
        )
        if i < len(chunks) - 1:
            del logits
            torch.cuda.empty_cache()

    return logits


def _prefill_append_chunk(
    model,
    new_token_ids: torch.Tensor,
    existing_seq_len: int,
    position_offset: int,
    block_ids: list[int],
    block_size: int,
    attn_layer_names: list[str],
    vllm_config,
    device: torch.device,
) -> torch.Tensor:
    """Single-chunk append-prefill. Runs new_token_ids attending to existing KV."""
    n_new = new_token_ids.shape[0]
    total_seq_len = existing_seq_len + n_new

    positions = torch.arange(
        existing_seq_len + position_offset,
        existing_seq_len + position_offset + n_new,
        dtype=torch.long, device=device,
    )

    slots = []
    for pos in range(existing_seq_len, total_seq_len):
        block_idx = pos // block_size
        offset = pos % block_size
        slots.append(block_ids[block_idx] * block_size + offset)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)

    block_table = torch.tensor(
        block_ids, dtype=torch.int32, device=device,
    ).unsqueeze(0)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=n_new,
        max_query_len=n_new,
        query_start_loc=torch.tensor([0, n_new], dtype=torch.int32, device=device),
        max_seq_len=total_seq_len,
        seq_lens=torch.tensor([total_seq_len], dtype=torch.int32, device=device),
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
        attn_metadata, vllm_config,
        virtual_engine=0, num_tokens=n_new,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        slot_mapping=slot_mapping_dict,
    ):
        hidden_states = model(input_ids=new_token_ids, positions=positions)

    return model.compute_logits(hidden_states)


def _inject_compacted_range(
    kv_caches: list[torch.Tensor],
    original_keys: list[torch.Tensor],
    original_values: list[torch.Tensor],
    c1_list: list[torch.Tensor],
    c2_list: list[torch.Tensor],
    block_ids: list[int],
    block_size: int,
    compact_start: int,
    compact_end: int,
    num_layers: int,
    old_seq_len: int,
) -> int:
    """Write [prefix | compacted_region | suffix] into paged blocks.

    Returns the new total sequence length after compaction.
    """
    compacted_len = c1_list[0].shape[0]
    suffix_start = compact_end
    new_seq_len = compact_start + compacted_len + (old_seq_len - suffix_start)

    num_blocks_to_touch = (old_seq_len + block_size - 1) // block_size
    n_blocks = min(num_blocks_to_touch, len(block_ids))
    bids = block_ids[:n_blocks]

    for layer_idx in range(num_layers):
        orig_K = original_keys[layer_idx]
        orig_V = original_values[layer_idx]

        prefix_K = orig_K[:compact_start]
        prefix_V = orig_V[:compact_start]
        suffix_K = orig_K[suffix_start:]
        suffix_V = orig_V[suffix_start:]

        K = torch.cat([prefix_K, c1_list[layer_idx], suffix_K], dim=0)
        V = torch.cat([prefix_V, c2_list[layer_idx], suffix_V], dim=0)
        total_len = K.shape[0]

        padded_len = n_blocks * block_size
        K_padded = torch.zeros(padded_len, K.shape[1], K.shape[2],
                               dtype=K.dtype, device=K.device)
        V_padded = torch.zeros(padded_len, V.shape[1], V.shape[2],
                               dtype=V.dtype, device=V.device)
        K_padded[:total_len] = K
        V_padded[:total_len] = V

        kv = kv_caches[layer_idx]
        kv[0, bids] = K_padded.view(n_blocks, block_size, K.shape[1], K.shape[2])
        kv[1, bids] = V_padded.view(n_blocks, block_size, V.shape[1], V.shape[2])

    return new_seq_len


def _batch_generate(
    model: Module,
    kv_caches: list[torch.Tensor],
    candidate_blocks: list[list[int]],
    K: int,
    initial_seq_len: int,
    position_offset: int,
    max_tokens: int,
    block_size: int,
    num_layers: int,
    attn_layer_names: list[str],
    vllm_config,
    device: torch.device,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    rng: torch.Generator,
    first_logits: torch.Tensor | None = None,
    tokenizer=None,
) -> tuple[list[str], list[list[float]]]:
    """Generate K candidate responses from forked KV caches using batch decode.

    Args:
        first_logits: Logits from the last prefill/append step. If provided,
            used to sample first tokens without an extra forward pass.
            Shape: (1, vocab_size) or (seq, vocab_size) — last row is used.
        tokenizer: For decoding token IDs to text. If None, loaded from config.

    Returns (texts, logprobs_per_candidate).
    """
    max_blocks_per_seq = max(len(b) for b in candidate_blocks)

    batch_ctx = _BatchDecodeContext.create(
        B=K,
        per_seq_blocks=candidate_blocks,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        attn_layer_names=attn_layer_names,
        vllm_config=vllm_config,
        device=device,
    )

    current_seq_lens = torch.full((K,), initial_seq_len, dtype=torch.long, device=device)
    position_offsets = torch.full((K,), position_offset, dtype=torch.long, device=device)

    # Get first logits if not provided — do a 1-token append on candidate_blocks[0]
    # using a dummy token. All candidates share the same initial KV state.
    if first_logits is None:
        first_logits = _prefill_append(
            model,
            torch.zeros(1, dtype=torch.long, device=device),
            existing_seq_len=initial_seq_len - 1,
            position_offset=position_offset,
            block_ids=candidate_blocks[0],
            block_size=block_size,
            attn_layer_names=attn_layer_names,
            vllm_config=vllm_config,
            device=device,
        )

    # Sample K different first tokens (different seeds for diversity)
    all_token_ids: list[list[int]] = [[] for _ in range(K)]
    all_logprobs: list[list[float]] = [[] for _ in range(K)]
    last_tokens = torch.zeros(K, dtype=torch.long, device=device)

    for k in range(K):
        token, logprob = _sample_token(first_logits[-1:], temperature, top_p, rng, seed=k)
        all_token_ids[k].append(token.item())
        all_logprobs[k].append(logprob.item())
        last_tokens[k] = token

    current_seq_lens += 1
    active = [True] * K

    max_possible_len = initial_seq_len + max_tokens + 1
    batch_ctx.metadata.max_seq_len = max_possible_len

    eos_check_interval = 64
    step = 0

    with set_forward_context(
        batch_ctx.attn_metadata_dict, batch_ctx.vllm_config,
        virtual_engine=0, num_tokens=K,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        slot_mapping=batch_ctx.slot_mapping_ctx,
    ):
        while any(active) and step < max_tokens:
            batch_ctx.input_ids[:] = last_tokens
            batch_ctx.positions[:] = current_seq_lens - 1 + position_offsets
            batch_ctx.seq_lens[:] = current_seq_lens.int()
            _update_batch_slots(batch_ctx, current_seq_lens)

            hidden = model(input_ids=batch_ctx.input_ids,
                           positions=batch_ctx.positions)
            logits = model.compute_logits(hidden)

            rng.manual_seed(step + K * 31337)
            tokens, lps = _sample_batch(logits, temperature, top_p, rng)

            for k in range(K):
                if not active[k]:
                    continue
                all_token_ids[k].append(tokens[k].item())
                all_logprobs[k].append(lps[k].item())
                last_tokens[k] = tokens[k]
                current_seq_lens[k] += 1

            step += 1

            if step % eos_check_interval == 0:
                for k in range(K):
                    if not active[k]:
                        continue
                    if eos_token_id in all_token_ids[k][-eos_check_interval:]:
                        for idx, tid in enumerate(all_token_ids[k]):
                            if tid == eos_token_id:
                                all_token_ids[k] = all_token_ids[k][:idx + 1]
                                all_logprobs[k] = all_logprobs[k][:idx + 1]
                                active[k] = False
                                break

    # Final EOS truncation
    for k in range(K):
        for idx, tid in enumerate(all_token_ids[k]):
            if tid == eos_token_id:
                all_token_ids[k] = all_token_ids[k][:idx + 1]
                all_logprobs[k] = all_logprobs[k][:idx + 1]
                break

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.model, trust_remote_code=True)
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in all_token_ids]

    return texts, all_logprobs, all_token_ids
