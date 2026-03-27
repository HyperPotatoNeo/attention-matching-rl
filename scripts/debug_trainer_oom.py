"""Standalone trainer OOM diagnostic.

Loads existing rollout data, prepares micro-batches, and runs segmented_forward
+ backward per micro-step with per-step memory logging.

Usage:
    uv run torchrun --nproc-per-node 2 scripts/debug_trainer_oom.py \
        --output-dir /network/scratch/e/emiliano.penaloza/outputs/balrog-babyai-good \
        --seq-len 12000 --step 0
"""

import argparse
import gc
from pathlib import Path

import msgspec
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoConfig

from prime_rl.trainer.batch import prepare_sample
from prime_rl.transport.types import TrainingBatch
from prime_rl.trainer.rl.compaction import segmented_forward


def log_memory(tag: str, rank: int):
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[GPU {rank}] {tag}: alloc={alloc:.2f} GiB, reserved={reserved:.2f} GiB, peak={max_alloc:.2f} GiB", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=12000)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-micro-steps", type=int, default=None)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    output_dir = Path(args.output_dir)

    # Load rollout data
    rollout_path = output_dir / "run_default" / "rollouts" / f"step_{args.step}" / "rollouts.bin"
    print(f"[Rank {rank}] Loading rollouts from {rollout_path}", flush=True)
    raw = rollout_path.read_bytes()
    batch = msgspec.msgpack.Decoder(TrainingBatch).decode(raw)
    examples = batch.examples
    print(f"[Rank {rank}] Loaded {len(examples)} samples", flush=True)

    # Analyze sample lengths before truncation
    if rank == 0:
        lengths = [len(s.prompt_ids) + len(s.completion_ids) for s in examples]
        lengths_sorted = sorted(lengths)
        print(f"\nSample length stats (N={len(lengths)}):")
        print(f"  min={min(lengths)}, max={max(lengths)}, "
              f"mean={sum(lengths)/len(lengths):.0f}, "
              f"median={lengths_sorted[len(lengths)//2]}")
        for sl in [6000, 8000, 10000, 12000]:
            n = sum(1 for l in lengths if l > sl)
            print(f"  Would truncate at seq_len={sl}: {n}/{len(lengths)}")

        # Show segment info per sample
        for i, s in enumerate(examples):
            total = len(s.prompt_ids) + len(s.completion_ids)
            n_seg = len(s.segment_boundaries) if s.segment_boundaries else 0
            n_ci = len(s.compaction_indices) if s.compaction_indices else 0
            n_cw = len(s.compact_windows) if s.compact_windows else 0
            print(f"  Sample {i}: {total} tokens, {n_seg} segments, "
                  f"{n_ci} compaction_indices, {n_cw} compact_windows, "
                  f"boundaries={s.segment_boundaries}")

    # Prepare micro-batches (1 sample each, at natural length — no padding to seq_len)
    # This matches the real trainer: compaction samples are NOT packed or padded
    micro_batches = []
    for s in examples:
        mb = prepare_sample(s, args.seq_len)
        micro_batches.append(mb)

    # Split across DP ranks
    per_rank = len(micro_batches) // world_size
    my_start = rank * per_rank
    my_end = my_start + per_rank
    my_micro_batches = micro_batches[my_start:my_end]

    print(f"[Rank {rank}] {len(my_micro_batches)} micro-batches", flush=True)

    # Log seg0 sizes for my micro-batches
    for i, mb in enumerate(my_micro_batches[:5]):
        seg_b = mb.segment_boundaries
        total = len(mb.input_ids)
        if seg_b:
            prompt_len = total - seg_b[-1]
            print(f"  [Rank {rank}] MB {i}: {total} tok, boundaries={seg_b}, "
                  f"prompt_len={prompt_len}, seg0_fwd_size={prompt_len + seg_b[0]}", flush=True)
        else:
            print(f"  [Rank {rank}] MB {i}: {total} tok, no segments", flush=True)

    # Load model
    print(f"\n[Rank {rank}] Loading model...", flush=True)
    config = AutoConfig.from_pretrained(args.model_name)
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, config=config, torch_dtype=torch.bfloat16,
    ).to(device)
    model.config.use_cache = False
    model.train()

    log_memory("After model load", rank)
    torch.cuda.reset_peak_memory_stats()

    # Run micro-steps
    max_steps = min(args.max_micro_steps or len(my_micro_batches), len(my_micro_batches))

    for micro_step in range(max_steps):
        mb = my_micro_batches[micro_step]
        input_ids = torch.tensor(mb.input_ids, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = torch.tensor(mb.position_ids, dtype=torch.long, device=device).unsqueeze(0)
        temperatures = torch.tensor(mb.temperatures, dtype=torch.float, device=device).unsqueeze(0)

        seg_boundaries = list(mb.segment_boundaries) if mb.segment_boundaries else None
        compaction_indices = mb.compaction_indices
        compact_windows = mb.compact_windows

        torch.cuda.reset_peak_memory_stats()
        log_memory(f"Step {micro_step} — start", rank)

        if seg_boundaries and len(seg_boundaries) > 0:
            prompt_len = input_ids.shape[1] - seg_boundaries[-1]
            n_forwards = len(seg_boundaries)
            max_forwards_t = torch.tensor([n_forwards], device=device, dtype=torch.int32)
            dist.all_reduce(max_forwards_t, op=dist.ReduceOp.MAX)
            max_forwards = int(max_forwards_t.item())

            out = segmented_forward(
                model, input_ids, position_ids,
                segment_boundaries=seg_boundaries,
                prompt_len=prompt_len,
                compact_target_ratio=0.25,
                compact_window=None,
                temperature=temperatures,
                max_forward_passes=max_forwards,
                compute_beta=False,
                use_suffix_queries=True,
                compaction_indices=compaction_indices,
                compaction_mode="attention_matching",
                compact_windows=compact_windows,
            )
        else:
            out = model(input_ids=input_ids, position_ids=position_ids)
            raw_logits = out.logits if hasattr(out, 'logits') else out["logits"]
            out = {"logits": raw_logits}

        log_memory(f"Step {micro_step} — after forward", rank)

        logits = out["logits"]
        loss = logits.float().mean() * 1e-6  # small dummy loss
        del out, logits
        torch.cuda.empty_cache()

        loss.backward()
        log_memory(f"Step {micro_step} — after backward", rank)

        del loss, input_ids, position_ids, temperatures
        gc.collect()
        torch.cuda.empty_cache()

        log_memory(f"Step {micro_step} — after cleanup", rank)
        print(flush=True)

    log_memory("FINAL", rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
