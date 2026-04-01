"""Evaluate RSA with compaction on AIME problems.

Loads AIME 2024/2025 problems from reasoning_gym, runs RSA with compaction
and baseline (single-pass) modes, scores answers (integers 0-999), and
compares results.

Usage:
    # RSA with compaction
    python scripts/eval_aime_rsa.py --mode rsa --n 30

    # Baseline
    python scripts/eval_aime_rsa.py --mode baseline --n 30

    # Compaction only (no RSA)
    python scripts/eval_aime_rsa.py --mode compaction --n 30
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import httpx
from transformers import AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

SYSTEM_PROMPT = (
    "You are a math competition solver. Solve the problem step by step. "
    "Put your final answer as an integer between 0 and 999 inside "
    "<answer>NUMBER</answer> tags."
)

def load_aime_problems(n: int, year: int = 2025):
    """Load AIME problems from HuggingFace."""
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
    else:
        raise ValueError(f"Unsupported year: {year}")
    return problems[:n]


def extract_integer_answer(text: str) -> int | None:
    match = re.search(r"<answer>\s*(\d+)\s*</answer>", text, re.DOTALL)
    if match:
        return int(match.group(1))
    # Fallback: last number in text (cap at 4 digits to avoid huge numbers)
    numbers = re.findall(r"\b(\d{1,4})\b", text)
    if numbers:
        val = int(numbers[-1])
        if 0 <= val <= 999:
            return val
    return None


def force_answer(text: str, problem, model_name, port, temperature=0.6, top_p=0.95):
    """If text was truncated without <answer>, do a short follow-up to extract one."""
    if "<answer>" in text:
        return text
    if "<think>" in text and "</think>" not in text:
        text += "</think>\n\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
        {"role": "assistant", "content": text + "\n\nMy final answer is: <answer>"},
    ]
    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=60.0)
    resp = client.post("/v1/chat/completions", json={
        "model": model_name,
        "messages": messages,
        "max_tokens": 20,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    resp.raise_for_status()
    continuation = resp.json()["choices"][0]["message"]["content"] or ""
    return text + "\n\nMy final answer is: <answer>" + continuation


def score_aime(problem, response_text):
    """Score AIME response. Returns 1.0 for correct, 0.0 for incorrect."""
    extracted = extract_integer_answer(response_text)
    if extracted is None:
        return 0.0
    return 1.0 if extracted == problem["answer"] else 0.0


def run_rsa(problem, port, tokenizer, args):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
    ]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)

    t0 = time.time()
    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=1800.0)
    body = {
        "prompt_ids": prompt_ids,
        "K": args.rsa_K,
        "N": args.rsa_N,
        "T": args.rsa_T,
        "k_peers": args.rsa_k_peers,
        "max_tokens_per_candidate": args.max_tokens,
        "compact_target_ratio": args.compact_ratio,
        "probe_tokens": args.probe_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    resp = client.post("/rsa_generate", json=body)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    return {
        "text": data["best"],
        "populations": data.get("populations", []),
        "time": elapsed,
        "diagnostics": data.get("diagnostics", {}),
    }


def run_baseline(problem, port, model_name, args):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
    ]

    t0 = time.time()
    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)
    resp = client.post("/v1/chat/completions", json={
        "model": model_name,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    })
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    return {
        "text": data["choices"][0]["message"]["content"],
        "time": elapsed,
        "diagnostics": {},
    }


def run_compaction(problem, port, tokenizer, args):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
    ]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)

    t0 = time.time()
    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)
    body = {
        "prompt_ids": prompt_ids,
        "max_tokens_per_segment": args.max_tokens // (args.n_compacts + 1),
        "n_compacts": args.n_compacts,
        "compact_target_ratio": args.compact_ratio,
        "compact_window": args.compact_window,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.mode == "markovian":
        body["compaction_mode"] = "markovian"
    resp = client.post("/compact_generate", json=body)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    return {
        "text": data["final_text"],
        "time": elapsed,
        "diagnostics": data.get("diagnostics", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="AIME evaluation with RSA")
    parser.add_argument("--mode", choices=["baseline", "compaction", "rsa", "rsa_no_compact", "markovian"], required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--year", type=int, default=2025, choices=[2024, 2025])
    parser.add_argument("--port", type=int, default=8000, help="Single port or first port for multi-server")
    parser.add_argument("--ports", type=str, default=None, help="Comma-separated ports for DP, e.g. '8000,8002'")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--compact-ratio", type=float, default=0.25)
    parser.add_argument("--compact-window", type=int, default=None,
                        help="Only compact the first N assistant tokens (None=all)")
    parser.add_argument("--n-compacts", type=int, default=3)
    # RSA-specific
    parser.add_argument("--rsa-K", type=int, default=4)
    parser.add_argument("--rsa-N", type=int, default=None, help="Total population size per step (default: K)")
    parser.add_argument("--rsa-T", type=int, default=2)
    parser.add_argument("--rsa-k-peers", type=int, default=2)
    parser.add_argument("--probe-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--carryover-ratio", type=float, default=0.5,
                        help="Fraction of assistant tokens to keep in markovian mode")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Loading AIME {args.year} problems (n={args.n})...")
    problems = load_aime_problems(args.n, year=args.year)
    print(f"Loaded {len(problems)} problems")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.mode == "rsa_no_compact":
        args.compact_ratio = 1.0

    print(f"\nMode: {args.mode}")
    if args.mode in ("rsa", "rsa_no_compact"):
        N_display = args.rsa_N or args.rsa_K
        print(f"  N={N_display}, K={args.rsa_K}, T={args.rsa_T}, k_peers={args.rsa_k_peers}")
        print(f"  compact_ratio={args.compact_ratio}")
    print(f"  max_tokens={args.max_tokens}")
    print()

    ports = [int(p) for p in args.ports.split(",")] if args.ports else [args.port]

    # Health check all servers
    for port in ports:
        health = httpx.get(f"http://localhost:{port}/health", timeout=10.0)
        health.raise_for_status()
    print(f"Server health: OK ({len(ports)} server(s) on ports {ports})\n")

    def run_one(i, prob, port):
        if args.mode in ("rsa", "rsa_no_compact"):
            gen = run_rsa(prob, port, tokenizer, args)
        elif args.mode in ("compaction", "markovian"):
            gen = run_compaction(prob, port, tokenizer, args)
        else:
            gen = run_baseline(prob, port, args.model, args)
        gen["text"] = force_answer(
            gen["text"], prob, args.model, port, args.temperature, args.top_p)
        return i, gen

    results = [None] * len(problems)
    total_correct = 0
    completed = 0

    # Resume from checkpoint if output file exists
    output_path = args.output or f"results_aime_{args.mode}_{len(problems)}.json"
    if Path(output_path).exists():
        prev = json.loads(Path(output_path).read_text())
        for r in (prev.get("results") or []):
            if r is not None:
                idx = r["idx"]
                if idx < len(results):
                    results[idx] = r
                    completed += 1
                    total_correct += int(r["correct"])
        print(f"Resumed from {output_path}: {completed}/{len(problems)} already done "
              f"({total_correct} correct)\n")

    t_total = time.time()

    def _save_checkpoint():
        summary = {
            "mode": args.mode,
            "n_problems": len(problems),
            "accuracy": total_correct / max(completed, 1),
            "correct": total_correct,
            "completed": completed,
            "config": {
                "model": args.model,
                "max_tokens": args.max_tokens,
                "compact_ratio": args.compact_ratio,
                "rsa_N": args.rsa_N if args.mode in ("rsa", "rsa_no_compact") else None,
                "rsa_K": args.rsa_K if args.mode in ("rsa", "rsa_no_compact") else None,
                "rsa_T": args.rsa_T if args.mode in ("rsa", "rsa_no_compact") else None,
                "rsa_k_peers": args.rsa_k_peers if args.mode in ("rsa", "rsa_no_compact") else None,
                "probe_tokens": args.probe_tokens if args.mode == "rsa" else None,
            },
            "results": results,
        }
        Path(output_path).write_text(json.dumps(summary, indent=2, default=str))

    # For compaction/markovian: fire all requests concurrently, let server-side
    # _RequestBatcher auto-batch them. For RSA/baseline: one request per server
    # (RSA uses full GPU internally, baseline uses vLLM's own batching).
    use_async = args.mode in ("compaction", "markovian") and len(ports) > 0
    max_workers = len(ports) * 32 if use_async else len(ports)

    remaining = [(i, prob) for i, prob in enumerate(problems) if results[i] is None]

    if not remaining:
        print("All problems already completed.\n")
    elif use_async:
        print(f"  Async mode: {len(remaining)} requests across {len(ports)} servers\n")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for i, prob in remaining:
                port = ports[i % len(ports)]
                fut = pool.submit(run_one, i, prob, port)
                futures[fut] = i

            for fut in as_completed(futures):
                i = futures[fut]
                _, gen = fut.result()
                prob = problems[i]
                completed += 1

                sc = score_aime(prob, gen["text"])
                correct = sc >= 0.5
                total_correct += int(correct)
                results[i] = {
                    "idx": i, "correct": correct, "score": sc,
                    "answer_extracted": extract_integer_answer(gen["text"]),
                    "time": round(gen["time"], 2),
                    "diagnostics": gen.get("diagnostics", {}),
                }
                _save_checkpoint()
                status = "OK" if correct else "FAIL"
                print(
                    f"[{completed:3d}/{len(problems)}] {status} "
                    f"answer={extract_integer_answer(gen['text'])} "
                    f"time={gen['time']:.1f}s "
                    f"acc={total_correct}/{completed} ({total_correct/completed:.1%})"
                )
    else:
        # RSA/baseline: one request per server, round-robin
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            prob_queue = list(remaining)
            pending = set()

            for port in ports:
                if not prob_queue:
                    break
                i, prob = prob_queue.pop(0)
                fut = pool.submit(run_one, i, prob, port)
                futures[fut] = (i, port)
                pending.add(fut)

            while pending:
                done_set, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    i, port = futures[fut]
                    _, gen = fut.result()
                    prob = problems[i]
                    completed += 1

                    sc = score_aime(prob, gen["text"])
                    correct = sc >= 0.5
                    total_correct += int(correct)
                    results[i] = {
                        "idx": i, "correct": correct, "score": sc,
                        "answer_extracted": extract_integer_answer(gen["text"]),
                        "time": round(gen["time"], 2),
                        "diagnostics": gen.get("diagnostics", {}),
                    }
                    _save_checkpoint()
                    status = "OK" if correct else "FAIL"
                    print(
                        f"[{completed:3d}/{len(problems)}] {status} "
                        f"answer={extract_integer_answer(gen['text'])} "
                        f"time={gen['time']:.1f}s "
                        f"acc={total_correct}/{completed} ({total_correct/completed:.1%})"
                    )

                    if prob_queue:
                        ni, nprob = prob_queue.pop(0)
                        nfut = pool.submit(run_one, ni, nprob, port)
                        futures[nfut] = (ni, port)
                        pending.add(nfut)

    wall_time = time.time() - t_total

    print("\n" + "=" * 60)
    print(f"AIME RESULTS: {args.mode.upper()}")
    print("=" * 60)
    print(f"Accuracy: {total_correct}/{len(problems)} ({total_correct/len(problems):.1%})")
    print(f"Wall time: {wall_time:.1f}s ({wall_time/len(problems):.1f}s/problem)")

    _save_checkpoint()
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
