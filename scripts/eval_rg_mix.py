"""Evaluate rg-mix-env with compaction (DP=4) and baseline (no compaction).

Usage (inside container):
    # Compaction: 4 servers on ports 8000-8003
    python scripts/eval_rg_mix.py --mode compaction --n 100 --n-compacts 3

    # Baseline: standard vLLM on port 8000
    python scripts/eval_rg_mix.py --mode baseline --n 100 --server-url http://localhost:8000

Metrics tracked:
    - Overall accuracy
    - Per-task accuracy
    - Accuracy vs number of compactions actually performed
    - Token counts, compaction ratios, timing
"""

import argparse
import concurrent.futures
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import reasoning_gym as rg
from reasoning_gym.utils import SYSTEM_PROMPTS
from transformers import AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

TASK_VARIANTS = [
    {"id": "arc_1d", "task": "arc_1d", "pass_at_1": 0.4016, "config": {}},
    {"id": "sokoban_hard", "task": "sokoban", "pass_at_1": 0.3101,
     "config": {"min_boxes": 3, "max_boxes": 4, "max_w": 9, "max_h": 9}},
    {"id": "countdown_7", "task": "countdown", "pass_at_1": 0.30,
     "config": {"min_numbers": 7, "max_numbers": 7}},
    {"id": "zebra_puzzles_7", "task": "zebra_puzzles", "pass_at_1": 0.2510,
     "config": {"num_people": 7, "num_characteristics": 5}},
    {"id": "cryptarithm", "task": "cryptarithm", "pass_at_1": 0.1882, "config": {}},
]

SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]


def generate_dataset(n: int, seed: int = 42):
    """Generate n rg-mix problems with inverse-difficulty weighting."""
    weights = [1.0 / v["pass_at_1"] for v in TASK_VARIANTS]
    rng = random.Random(seed)

    variant_datasets = {}
    for i, variant in enumerate(TASK_VARIANTS):
        ds = rg.create_dataset(
            variant["task"], seed=seed + i + 1, size=n, **variant["config"],
        )
        variant_datasets[variant["id"]] = ds

    problems = []
    for i in range(n):
        chosen_idx = rng.choices(range(len(TASK_VARIANTS)), weights=weights, k=1)[0]
        variant = TASK_VARIANTS[chosen_idx]
        vid = variant["id"]
        ds = variant_datasets[vid]
        entry = ds[i % len(ds)]

        problems.append({
            "idx": i,
            "task": vid,
            "question": entry["question"],
            "entry": entry,
            "dataset": ds,
        })

    return problems


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()


def score_problem(problem, response_text):
    extracted = extract_answer(response_text)
    score = problem["dataset"].score_answer(answer=extracted, entry=problem["entry"])
    if score < 0.5:
        score_full = problem["dataset"].score_answer(
            answer=response_text, entry=problem["entry"]
        )
        score = max(score, score_full)
    return score


def run_compaction_one(problem, port, tokenizer, args):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
    ]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)

    max_seg = args.max_tokens_per_segment
    n_comp = args.n_compacts

    t0 = time.time()
    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)
    resp = client.post("/compact_generate", json={
        "prompt_ids": prompt_ids,
        "max_seq_len": len(prompt_ids) + max_seg * (n_comp + 1),
        "max_tokens_per_segment": max_seg,
        "n_compacts": n_comp,
        "compact_target_ratio": args.compact_ratio,
        "compact_window": args.compact_window,
        "temperature": 0.6,
        "top_p": 0.95,
    })
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    return {
        "text": data["final_text"],
        "tokens": len(data["all_token_ids"]),
        "time": elapsed,
        "diagnostics": data.get("diagnostics", {}),
        "mean_logprob": data["diagnostics"].get("mean_logprob", 0),
        "logprobs": data["all_logprobs"],
    }


def run_baseline_one(problem, port, model_name, args):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
    ]

    t0 = time.time()
    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)
    resp = client.post("/v1/chat/completions", json={
        "model": model_name,
        "messages": messages,
        "max_tokens": args.max_tokens_per_segment * (args.n_compacts + 1),
        "temperature": 0.6,
        "top_p": 0.95,
    })
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    usage = data["usage"]

    return {
        "text": text,
        "tokens": usage["completion_tokens"],
        "time": elapsed,
        "diagnostics": {},
        "mean_logprob": 0,
        "logprobs": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "compaction"], required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--max-tokens-per-segment", type=int, default=2048)
    parser.add_argument("--n-compacts", type=int, default=3)
    parser.add_argument("--compact-ratio", type=float, default=0.3)
    parser.add_argument("--compact-window", type=int, default=None,
                        help="Only compact the first N assistant tokens (None=all)")
    parser.add_argument("--ports", default="8000,8001,8002,8003",
                        help="Comma-separated ports for DP (compaction mode)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ports = [int(p) for p in args.ports.split(",")]

    print(f"Generating {args.n} rg-mix problems (seed={args.seed})...")
    problems = generate_dataset(args.n, args.seed)

    task_dist = defaultdict(int)
    for p in problems:
        task_dist[p["task"]] += 1
    print("Task distribution:")
    for task, count in sorted(task_dist.items(), key=lambda x: -x[1]):
        print(f"  {task:25s}: {count:3d} ({count/len(problems)*100:.0f}%)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    max_total_tokens = args.max_tokens_per_segment * (args.n_compacts + 1)
    print(f"\nMode: {args.mode}")
    print(f"Max response tokens: {max_total_tokens}")
    if args.mode == "compaction":
        print(f"  segments of {args.max_tokens_per_segment} tokens, "
              f"{args.n_compacts} compactions, ratio={args.compact_ratio}")
        print(f"  DP={len(ports)} (ports: {ports})")
    print()

    # Health check
    check_port = ports[0] if args.mode == "compaction" else int(
        args.server_url.rsplit(":", 1)[-1].split("/")[0]
    )
    health = httpx.get(f"http://localhost:{check_port}/health", timeout=10.0)
    health.raise_for_status()
    print("Server health: OK\n")

    results = []
    total_correct = 0
    total_tokens = 0
    t_total = time.time()

    if args.mode == "compaction":
        # DP=4: distribute problems across ports in batches
        batch_size = len(ports)
        for batch_start in range(0, len(problems), batch_size):
            batch = problems[batch_start:batch_start + batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as ex:
                futures = {}
                for j, prob in enumerate(batch):
                    port = ports[j % len(ports)]
                    f = ex.submit(run_compaction_one, prob, port, tokenizer, args)
                    futures[f] = (batch_start + j, prob)

                for f in concurrent.futures.as_completed(futures):
                    idx, prob = futures[f]
                    gen = f.result()
                    score = score_problem(prob, gen["text"])
                    correct = score >= 0.5

                    n_actual_compacts = len(
                        gen["diagnostics"].get("compaction_events", [])
                    )
                    total_correct += int(correct)
                    total_tokens += gen["tokens"]

                    results.append({
                        "idx": idx,
                        "task": prob["task"],
                        "correct": correct,
                        "score": score,
                        "tokens": gen["tokens"],
                        "time": round(gen["time"], 2),
                        "n_compactions": n_actual_compacts,
                        "mean_logprob": gen["mean_logprob"],
                        "diagnostics": gen["diagnostics"],
                    })

                    status = "OK" if correct else "FAIL"
                    print(
                        f"[{len(results):3d}/{args.n}] {status} "
                        f"task={prob['task']:20s} "
                        f"tokens={gen['tokens']:5d} "
                        f"compacts={n_actual_compacts} "
                        f"time={gen['time']:.1f}s "
                        f"acc={total_correct}/{len(results)} "
                        f"({total_correct/len(results):.1%})"
                    )
    else:
        # Baseline: sequential on single server
        base_port = int(args.server_url.rsplit(":", 1)[-1].split("/")[0])
        for i, prob in enumerate(problems):
            gen = run_baseline_one(prob, base_port, args.model, args)
            score = score_problem(prob, gen["text"])
            correct = score >= 0.5
            total_correct += int(correct)
            total_tokens += gen["tokens"]

            results.append({
                "idx": i,
                "task": prob["task"],
                "correct": correct,
                "score": score,
                "tokens": gen["tokens"],
                "time": round(gen["time"], 2),
                "n_compactions": 0,
                "mean_logprob": 0,
                "diagnostics": {},
            })

            status = "OK" if correct else "FAIL"
            print(
                f"[{i+1:3d}/{args.n}] {status} "
                f"task={prob['task']:20s} "
                f"tokens={gen['tokens']:5d} "
                f"time={gen['time']:.1f}s "
                f"acc={total_correct}/{i+1} ({total_correct/(i+1):.1%})"
            )

    wall_time = time.time() - t_total

    # Sort results by original index
    results.sort(key=lambda r: r["idx"])

    # === METRICS ===
    print("\n" + "=" * 70)
    print(f"RESULTS: {args.mode.upper()} ({args.n} problems)")
    print("=" * 70)

    print(f"\nOverall accuracy: {total_correct}/{args.n} ({total_correct/args.n:.1%})")
    print(f"Total tokens: {total_tokens}")
    print(f"Avg tokens/problem: {total_tokens/args.n:.0f}")
    print(f"Wall time: {wall_time:.1f}s ({wall_time/args.n:.1f}s/problem)")
    print(f"Throughput: {total_tokens/wall_time:.0f} tok/s aggregate")

    # Per-task accuracy
    per_task = defaultdict(lambda: {"correct": 0, "total": 0, "tokens": []})
    for r in results:
        per_task[r["task"]]["correct"] += int(r["correct"])
        per_task[r["task"]]["total"] += 1
        per_task[r["task"]]["tokens"].append(r["tokens"])

    print(f"\n{'Task':<25} {'Accuracy':>10} {'Correct':>8} {'Total':>6} {'AvgTok':>8}")
    print("-" * 60)
    for task in sorted(per_task.keys()):
        t = per_task[task]
        acc = t["correct"] / t["total"]
        avg_tok = sum(t["tokens"]) / len(t["tokens"])
        print(f"{task:<25} {acc:>9.1%} {t['correct']:>8d} {t['total']:>6d} {avg_tok:>8.0f}")

    # Accuracy vs number of compactions
    if args.mode == "compaction":
        by_compacts = defaultdict(lambda: {"correct": 0, "total": 0, "tokens": [],
                                           "ratios": [], "times": []})
        for r in results:
            nc = r["n_compactions"]
            by_compacts[nc]["correct"] += int(r["correct"])
            by_compacts[nc]["total"] += 1
            by_compacts[nc]["tokens"].append(r["tokens"])
            by_compacts[nc]["times"].append(r["time"])
            for evt in r["diagnostics"].get("compaction_events", []):
                by_compacts[nc]["ratios"].append(evt["ratio"])

        print(f"\n{'Compactions':<12} {'Accuracy':>10} {'Correct':>8} {'Total':>6} "
              f"{'AvgTok':>8} {'AvgRatio':>10} {'AvgTime':>8}")
        print("-" * 70)
        for nc in sorted(by_compacts.keys()):
            b = by_compacts[nc]
            acc = b["correct"] / b["total"]
            avg_tok = sum(b["tokens"]) / len(b["tokens"])
            avg_ratio = sum(b["ratios"]) / len(b["ratios"]) if b["ratios"] else 0
            avg_time = sum(b["times"]) / len(b["times"])
            print(f"{nc:<12d} {acc:>9.1%} {b['correct']:>8d} {b['total']:>6d} "
                  f"{avg_tok:>8.0f} {avg_ratio:>10.2f} {avg_time:>7.1f}s")

        # Compaction event stats
        all_ratios = []
        all_algo_times = []
        for r in results:
            for evt in r["diagnostics"].get("compaction_events", []):
                all_ratios.append(evt["ratio"])
                all_algo_times.append(evt["algo_time"])
        if all_ratios:
            print(f"\nCompaction stats ({len(all_ratios)} events):")
            print(f"  Avg ratio: {sum(all_ratios)/len(all_ratios):.3f}")
            print(f"  Avg algo time: {sum(all_algo_times)/len(all_algo_times):.3f}s")

        # Mean logprob by compaction count
        lp_by_compact = defaultdict(list)
        for r in results:
            if r["mean_logprob"] != 0:
                lp_by_compact[r["n_compactions"]].append(r["mean_logprob"])
        if lp_by_compact:
            print(f"\nMean logprob by compaction count:")
            for nc in sorted(lp_by_compact.keys()):
                lps = lp_by_compact[nc]
                print(f"  {nc} compactions: {sum(lps)/len(lps):.4f} (n={len(lps)})")

    # Save
    output_path = args.output or f"results_{args.mode}_{args.n}.json"
    summary = {
        "mode": args.mode,
        "n_problems": args.n,
        "accuracy": total_correct / args.n,
        "correct": total_correct,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / args.n,
        "wall_time": round(wall_time, 2),
        "throughput_tok_s": round(total_tokens / wall_time, 1),
        "config": {
            "model": args.model,
            "max_tokens_per_segment": args.max_tokens_per_segment,
            "n_compacts": args.n_compacts,
            "compact_ratio": args.compact_ratio,
            "temperature": 0.6,
            "seed": args.seed,
        },
        "per_task": {
            task: {
                "accuracy": t["correct"] / t["total"],
                "correct": t["correct"],
                "total": t["total"],
                "avg_tokens": sum(t["tokens"]) / len(t["tokens"]),
            }
            for task, t in per_task.items()
        },
        "results": [
            {k: v for k, v in r.items() if k != "logprobs"}
            for r in results
        ],
    }
    Path(output_path).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
