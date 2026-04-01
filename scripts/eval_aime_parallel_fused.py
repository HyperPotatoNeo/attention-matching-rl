"""Evaluate fused parallel reasoning with compaction on AIME problems.

For each problem, a single call to /parallel_generate_fused:
  1. Generates K candidate solutions internally (from shared prompt prefix)
  2. Compacts each candidate's KV in-place (no text round-trip)
  3. Injects compacted KV into coordinator and synthesizes

Usage:
    python scripts/eval_aime_parallel_fused.py --n 5 --K 4
    python scripts/eval_aime_parallel_fused.py --n 30 --K 4 --compact-ratio 0.5
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import httpx
from transformers import AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

SYSTEM_PROMPT = (
    "You are a math competition solver. Solve the problem step by step. "
    "Put your final answer as an integer between 0 and 999 inside "
    "<answer>NUMBER</answer> tags."
)

SYNTHESIZE_SYSTEM = (
    "You are a math competition solver. You have been given compressed "
    "representations of multiple candidate solutions to a problem. "
    "Analyze the approaches, identify the most promising reasoning, "
    "and provide the correct solution step by step. "
    "Put your final answer as an integer between 0 and 999 inside "
    "<answer>NUMBER</answer> tags."
)


def load_aime_problems(n: int, year: int = 2025):
    from datasets import load_dataset
    if year == 2025:
        ds = load_dataset("MathArena/aime_2025", split="train")
        problems = []
        for i, entry in enumerate(ds):
            if i >= n:
                break
            problems.append({
                "idx": i,
                "question": entry["problem"],
                "answer": entry["answer"],
                "id": f"aime2025-{entry['problem_idx']}",
            })
    elif year == 2024:
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        problems = []
        for i, entry in enumerate(ds):
            if i >= n:
                break
            problems.append({
                "idx": i,
                "question": entry["Problem"],
                "answer": entry["Answer"],
                "id": entry["ID"],
            })
    return problems[:n]


def extract_integer_answer(text: str) -> int | None:
    match = re.search(r"<answer>\s*(\d+)\s*</answer>", text, re.DOTALL)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\b(\d{1,4})\b", text)
    if numbers:
        val = int(numbers[-1])
        if 0 <= val <= 999:
            return val
    return None


def score_aime(problem, response_text):
    extracted = extract_integer_answer(response_text)
    if extracted is None:
        return 0.0
    return 1.0 if extracted == problem["answer"] else 0.0


def run_one_problem(problem, tokenizer, port, args):
    """Single call to /parallel_generate_fused — handles everything server-side."""
    t0 = time.time()

    # Candidate prompt: solve the problem
    cand_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
    ]
    result = tokenizer.apply_chat_template(
        cand_messages, add_generation_prompt=True, tokenize=True,
    )
    prompt_ids = (
        list(result["input_ids"]) if hasattr(result, "keys") else list(result)
    )

    # Coordinator prompt: synthesize from compressed representations
    coord_messages = [
        {"role": "system", "content": SYNTHESIZE_SYSTEM},
        {"role": "user", "content": problem["question"]},
    ]
    coord_result = tokenizer.apply_chat_template(
        coord_messages, add_generation_prompt=True, tokenize=True,
    )
    coordinator_prompt_ids = (
        list(coord_result["input_ids"]) if hasattr(coord_result, "keys")
        else list(coord_result)
    )

    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=1800.0)
    body = {
        "prompt_ids": prompt_ids,
        "coordinator_prompt_ids": coordinator_prompt_ids,
        "K": args.K,
        "max_candidate_tokens": args.candidate_tokens,
        "compact_target_ratio": args.compact_ratio,
        "max_gen_tokens": args.max_gen_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "compute_beta": args.compute_beta,
        "probe_tokens": args.probe_tokens,
    }
    resp = client.post("/parallel_generate_fused", json=body)
    resp.raise_for_status()
    data = resp.json()

    elapsed = time.time() - t0

    # Extract candidate answers for logging
    candidates = data.get("candidates", [])
    cand_answers = [extract_integer_answer(c["text"]) for c in candidates]

    return {
        "text": data["final_text"],
        "candidates": candidates,
        "cand_answers": cand_answers,
        "diagnostics": data.get("diagnostics", {}),
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AIME evaluation with fused parallel reasoning")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model", default=None,
                        help="Model name (for tokenizer; auto-detected if omitted)")
    parser.add_argument("--year", type=int, default=2025, choices=[2024, 2025])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ports", type=str, default=None,
                        help="Comma-separated ports for DP, e.g. '8000,8001'")
    parser.add_argument("--K", type=int, default=4, help="Number of candidates")
    parser.add_argument("--candidate-tokens", type=int, default=4096,
                        help="Max tokens per candidate solution")
    parser.add_argument("--max-gen-tokens", type=int, default=4096,
                        help="Max tokens for coordinator synthesis")
    parser.add_argument("--compact-ratio", type=float, default=0.25)
    parser.add_argument("--probe-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-beta", action="store_true",
                        help="Disable beta correction")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    args.compute_beta = not args.no_beta

    ports = ([int(p) for p in args.ports.split(",")]
             if args.ports else [args.port])

    # Auto-detect model name from first server
    if args.model is None:
        client = httpx.Client(
            base_url=f"http://localhost:{ports[0]}", timeout=10.0)
        resp = client.get("/v1/models")
        resp.raise_for_status()
        args.model = resp.json()["data"][0]["id"]
        print(f"Auto-detected model: {args.model}")

    print(f"\nLoading AIME {args.year} problems (n={args.n})...")
    problems = load_aime_problems(args.n, year=args.year)
    print(f"Loaded {len(problems)} problems")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Health check
    for port in ports:
        health = httpx.get(f"http://localhost:{port}/health", timeout=10.0)
        health.raise_for_status()
    print(f"Server health: OK ({len(ports)} server(s) on ports {ports})\n")

    print(f"Config: K={args.K}, ratio={args.compact_ratio}, "
          f"beta={args.compute_beta}")
    print(f"Candidate tokens: {args.candidate_tokens}, "
          f"Synthesis tokens: {args.max_gen_tokens}")
    print(f"Probe tokens: {args.probe_tokens}\n")

    results = [None] * len(problems)
    total_correct = 0
    completed = 0

    output_path = (args.output
                   or f"results_aime_fused_K{args.K}_{len(problems)}.json")

    t_total = time.time()

    def _save_checkpoint():
        summary = {
            "mode": "parallel_fused",
            "n_problems": len(problems),
            "accuracy": total_correct / max(completed, 1),
            "correct": total_correct,
            "completed": completed,
            "config": {
                "model": args.model,
                "K": args.K,
                "candidate_tokens": args.candidate_tokens,
                "max_gen_tokens": args.max_gen_tokens,
                "compact_ratio": args.compact_ratio,
                "probe_tokens": args.probe_tokens,
                "compute_beta": args.compute_beta,
            },
            "results": results,
        }
        Path(output_path).write_text(json.dumps(summary, indent=2, default=str))

    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures = {}
        prob_queue = list(enumerate(problems))
        pending = set()

        for port in ports:
            if not prob_queue:
                break
            i, prob = prob_queue.pop(0)
            fut = pool.submit(run_one_problem, prob, tokenizer, port, args)
            futures[fut] = (i, port)
            pending.add(fut)

        while pending:
            done_set, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done_set:
                i, port = futures[fut]
                gen = fut.result()
                prob = problems[i]
                completed += 1

                sc = score_aime(prob, gen["text"])
                correct = sc >= 0.5
                total_correct += int(correct)

                results[i] = {
                    "idx": i,
                    "id": prob["id"],
                    "correct": correct,
                    "score": sc,
                    "answer_expected": prob["answer"],
                    "answer_extracted": extract_integer_answer(gen["text"]),
                    "cand_answers": gen["cand_answers"],
                    "final_text": gen["text"][:500],
                    "time": round(gen["time"], 2),
                    "diagnostics": gen["diagnostics"],
                }

                status = "CORRECT" if correct else "WRONG"
                print(
                    f"[{completed:3d}/{len(problems)}] {prob['id']} {status} "
                    f"extracted={extract_integer_answer(gen['text'])} "
                    f"expected={prob['answer']} "
                    f"cands={gen['cand_answers']} "
                    f"time={gen['time']:.1f}s "
                    f"acc={total_correct}/{completed} "
                    f"({total_correct/completed:.1%})"
                )
                _save_checkpoint()

                if prob_queue:
                    ni, nprob = prob_queue.pop(0)
                    nfut = pool.submit(
                        run_one_problem, nprob, tokenizer, port, args)
                    futures[nfut] = (ni, port)
                    pending.add(nfut)

    wall_time = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"AIME RESULTS: FUSED PARALLEL REASONING")
    print(f"{'='*60}")
    print(f"Accuracy: {total_correct}/{len(problems)} "
          f"({total_correct/len(problems):.1%})")
    print(f"Wall time: {wall_time:.1f}s "
          f"({wall_time/len(problems):.1f}s/problem)")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
