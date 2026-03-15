"""Test unmasked mismatch KL between inference and trainer compaction replay.

Compares four trainer replay modes:
1. Random queries (baseline): deterministic seeded queries, same as inference
2. Expert indices: forced top-k indices from inference, random queries for C2
3. Suffix queries: trainer recomputes suffix queries from its own HF model
4. Expert+suffix: forced top-k indices from inference, suffix queries for C2

Run inside container on GPU node:
    # GPU 0: inference server
    CUDA_VISIBLE_DEVICES=0 uv run serve @ configs/compaction/qwen3_4b_serve_tp1.toml --server.port 8000 &
    # GPU 1: test
    CUDA_VISIBLE_DEVICES=1 python scripts/test_mismatch_kl_expert_vs_suffix.py --device cuda:0
"""

import argparse
import asyncio
import json
import sys
import time

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from prime_rl.trainer.rl.compaction import segmented_forward


PROMPTS = [
    "Write a detailed analysis of the relationship between entropy and information theory. "
    "Cover Shannon entropy, cross-entropy, KL divergence, and their applications in machine learning.",
    "Explain the mathematical foundations of quantum mechanics step by step, including "
    "the Schrödinger equation, wave functions, and the uncertainty principle.",
    "What are the key differences between TCP and UDP protocols? Explain in detail with examples "
    "of when each should be used, including edge cases and performance considerations.",
]


async def generate_with_compaction(server_url, model_name, prompt_text, **kwargs):
    async with httpx.AsyncClient(timeout=600.0) as client:
        tok_resp = await client.post(
            f"{server_url}/tokenize",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "add_generation_prompt": True,
            },
        )
        tok_resp.raise_for_status()
        prompt_ids = tok_resp.json()["tokens"]

        request = {
            "prompt_ids": prompt_ids,
            "max_kv_len": kwargs.get("max_kv_len", 2048),
            "max_total_tokens": kwargs.get("max_total_tokens", 4096),
            "compact_target_ratio": kwargs.get("compact_target_ratio", 0.25),
            "compact_window": kwargs.get("compact_window", 1024),
            "n_compacts": kwargs.get("n_compacts", 99),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "use_suffix_queries": kwargs.get("use_suffix_queries", False),
        }
        resp = await client.post(f"{server_url}/compact_generate", json=request)
        resp.raise_for_status()
        result = resp.json()

        return {
            "prompt_ids": prompt_ids,
            "all_token_ids": result["all_token_ids"],
            "all_logprobs": result["all_logprobs"],
            "diagnostics": result["diagnostics"],
        }


def compute_trainer_logprobs(
    model, prompt_ids, completion_ids, segment_boundaries, temperature,
    device, mode="random", compaction_indices=None,
    compact_target_ratio=0.25, compact_window=1024,
):
    full_ids = prompt_ids + completion_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    position_ids = torch.arange(len(full_ids), device=device).unsqueeze(0)
    temp_tensor = torch.full((1, len(full_ids)), temperature, device=device, dtype=torch.float32)
    prompt_len = len(prompt_ids)

    model.config.use_cache = True
    model.eval()

    with torch.no_grad():
        out = segmented_forward(
            model,
            input_ids,
            position_ids,
            segment_boundaries=segment_boundaries,
            prompt_len=prompt_len,
            compact_target_ratio=compact_target_ratio,
            compact_window=compact_window,
            temperature=temp_tensor,
            compute_beta=False,
            use_suffix_queries=(mode in ("suffix", "expert+suffix")),
            compaction_indices=compaction_indices if mode in ("expert", "expert+suffix") else None,
        )

    logits = out["logits"]  # [1, seq_len, vocab], temperature-scaled
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # logprobs[t] = log P(completion[t] | context)
    # The logit at position (prompt_len + t - 1) predicts completion[t]
    pred_positions = torch.arange(
        prompt_len - 1, prompt_len + len(completion_ids) - 1, device=device,
    )
    target_tokens = torch.tensor(completion_ids, device=device, dtype=torch.long)
    completion_logprobs = log_probs[0, pred_positions, target_tokens].cpu().float()
    return completion_logprobs


def compute_mismatch_kl(inference_lp, trainer_lp):
    log_ratio = trainer_lp - inference_lp
    ratio = torch.exp(log_ratio)
    kl = ratio - log_ratio - 1
    return kl


async def main():
    parser = argparse.ArgumentParser(description="Mismatch KL: expert indices vs suffix queries")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-kv-len", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--compact-ratio", type=float, default=0.25)
    parser.add_argument("--compact-window", type=int, default=1024)
    parser.add_argument("--n-compacts", type=int, default=99)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("Unmasked Mismatch KL Test: Expert Indices vs Suffix Queries")
    print("=" * 70)
    print(f"  Server:      {args.server_url}")
    print(f"  Model:       {args.model}")
    print(f"  Device:      {device}")
    print(f"  max_kv_len:  {args.max_kv_len}")
    print(f"  max_tokens:  {args.max_total_tokens}")
    print(f"  ratio:       {args.compact_ratio}")
    print(f"  window:      {args.compact_window}")
    print(f"  temperature: {args.temperature}")
    print()

    # Load HF model
    print(f"Loading model {args.model} on {device}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print()

    all_results = []

    for i, prompt in enumerate(PROMPTS):
        print(f"{'=' * 70}")
        print(f"Prompt {i+1}/{len(PROMPTS)}: {prompt[:70]}...")
        print(f"{'=' * 70}")

        # Generate with compaction (random queries = default)
        print("  Generating with inference server...")
        t0 = time.time()
        gen = await generate_with_compaction(
            args.server_url, args.model, prompt,
            max_kv_len=args.max_kv_len,
            max_total_tokens=args.max_total_tokens,
            compact_target_ratio=args.compact_ratio,
            compact_window=args.compact_window,
            n_compacts=args.n_compacts,
            temperature=args.temperature,
        )
        gen_time = time.time() - t0

        diag = gen["diagnostics"]
        seg_boundaries = diag["segment_boundaries"]
        events = diag.get("compaction_events", [])
        compaction_indices = [e.get("compaction_indices") for e in events] if events else None

        n_tokens = len(gen["all_token_ids"])
        n_compactions = len(events)
        print(f"  Generated {n_tokens} tokens, {n_compactions} compactions in {gen_time:.1f}s")
        print(f"  Segments: {len(seg_boundaries)}, boundaries: {seg_boundaries}")
        print(f"  Inference mean logprob: {diag.get('mean_logprob', 'N/A')}")

        if n_compactions == 0:
            print("  SKIP: No compactions occurred (need longer generation)")
            continue

        inference_lp = torch.tensor(gen["all_logprobs"], dtype=torch.float32)
        prompt_result = {"prompt_idx": i, "n_tokens": n_tokens, "n_compactions": n_compactions}

        for mode in ["random", "expert", "suffix", "expert+suffix"]:
            print(f"\n  --- Mode: {mode} ---")
            t0 = time.time()
            trainer_lp = compute_trainer_logprobs(
                model, gen["prompt_ids"], gen["all_token_ids"],
                seg_boundaries, args.temperature, device,
                mode=mode,
                compaction_indices=compaction_indices,
                compact_target_ratio=args.compact_ratio,
                compact_window=args.compact_window,
            )
            dt = time.time() - t0

            kl = compute_mismatch_kl(inference_lp, trainer_lp)
            mean_kl = kl.mean().item()
            max_kl = kl.max().item()
            median_kl = kl.median().item()

            # Per-segment KL breakdown
            seg_kls = []
            prev_b = 0
            for b in seg_boundaries:
                seg_kl = kl[prev_b:b].mean().item()
                seg_kls.append(seg_kl)
                prev_b = b

            prompt_result[f"{mode}_mean_kl"] = mean_kl
            prompt_result[f"{mode}_max_kl"] = max_kl
            prompt_result[f"{mode}_median_kl"] = median_kl
            prompt_result[f"{mode}_time"] = dt

            print(f"    Mean KL:   {mean_kl:.6f}")
            print(f"    Median KL: {median_kl:.6f}")
            print(f"    Max KL:    {max_kl:.6f}")
            print(f"    Time:      {dt:.1f}s")
            print(f"    Per-segment KL: {['%.6f' % s for s in seg_kls]}")

        all_results.append(prompt_result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY: Mean Unmasked Mismatch KL (averaged across prompts)")
    print("=" * 70)
    for mode in ["random", "expert", "suffix", "expert+suffix"]:
        kls = [r[f"{mode}_mean_kl"] for r in all_results if f"{mode}_mean_kl" in r]
        if kls:
            avg = sum(kls) / len(kls)
            print(f"  {mode:>14s}: {avg:.6f}  (per-prompt: {['%.6f' % k for k in kls]})")
        else:
            print(f"  {mode:>14s}: no data")

    print()
    print("Interpretation:")
    print("  random        = baseline (trainer uses same seeded queries as inference)")
    print("  expert        = inference's top-k indices forced, random queries for C2")
    print("  suffix        = trainer independently computes suffix queries + top-k")
    print("  expert+suffix = forced top-k from inference, suffix queries for C2")
    print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
