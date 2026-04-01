"""Evaluate parallel reasoning with compaction on AIME problems.

For each problem:
  1. Generate K candidate solutions in parallel (standard vLLM)
  2. Compress all K solutions via attention matching and inject into coordinator
  3. Coordinator synthesizes a better answer from compressed context
  4. Repeat for T rounds

Usage:
    python scripts/eval_aime_parallel.py --n 5 --K 4 --T 2
    python scripts/eval_aime_parallel.py --n 5 --K 4 --T 2 --no-beta
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


def force_answer(text: str, problem, model_name, port, temperature=0.6, top_p=0.95):
    """If text was truncated without <answer>, do a short follow-up to extract one."""
    if "<answer>" in text:
        return text
    # Close any open think tag
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


def generate_candidates(problem, K, tokenizer, port, args):
    """Generate K candidate solutions using standard vLLM chat completions."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
    ]
    model_name = args.model

    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)
    resp = client.post("/v1/chat/completions", json={
        "model": model_name,
        "messages": messages,
        "max_tokens": args.candidate_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": K,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    resp.raise_for_status()
    data = resp.json()

    candidates = []
    for choice in data["choices"]:
        text = choice["message"]["content"] or ""
        if not text.strip():
            text = "(no response)"
        text = force_answer(text, problem, model_name, port, args.temperature, args.top_p)
        candidates.append(text)
    return candidates


def run_parallel_round(problem, candidates, tokenizer, port, args):
    """Run one round of parallel compression + synthesis."""
    # Build coordinator prompt: the synthesis instruction + original question
    coordinator_messages = [
        {"role": "system", "content": SYNTHESIZE_SYSTEM},
        {"role": "user", "content": problem["question"]},
    ]
    result = tokenizer.apply_chat_template(
        coordinator_messages, add_generation_prompt=True, tokenize=True,
    )
    coordinator_prompt_ids = (
        list(result["input_ids"]) if hasattr(result, "keys") else list(result)
    )

    # Tokenize each candidate solution as a document
    document_ids_list = []
    for cand in candidates:
        doc_ids = tokenizer.encode(cand, add_special_tokens=False)
        document_ids_list.append(doc_ids)

    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=1800.0)
    body = {
        "coordinator_prompt_ids": coordinator_prompt_ids,
        "document_ids_list": document_ids_list,
        "compact_target_ratio": args.compact_ratio,
        "probe_tokens": args.probe_tokens,
        "max_gen_tokens": args.max_gen_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "compute_beta": args.compute_beta,
        "summary_prompt": args.summary_prompt,
    }
    resp = client.post("/parallel_generate", json=body)
    resp.raise_for_status()
    data = resp.json()
    return data


def run_one_problem(problem, tokenizer, port, args):
    """Full pipeline: generate candidates → T rounds of parallel compress+synthesize."""
    t0 = time.time()
    K = args.K
    T = args.T

    all_rounds = []

    # Initial candidate generation
    print(f"  [Problem {problem['idx']}] Generating {K} candidates...")
    candidates = generate_candidates(problem, K, tokenizer, port, args)
    all_rounds.append({
        "round": 0,
        "type": "candidates",
        "texts": candidates,
        "answers": [extract_integer_answer(c) for c in candidates],
    })
    print(f"  [Problem {problem['idx']}] Candidates generated, "
          f"answers: {all_rounds[-1]['answers']}")

    # T rounds of parallel compress + synthesize
    for t in range(T):
        print(f"  [Problem {problem['idx']}] Round {t+1}/{T}: "
              f"compressing {len(candidates)} solutions...")
        data = run_parallel_round(problem, candidates, tokenizer, port, args)
        synthesis = data["final_text"]
        synthesis = force_answer(
            synthesis, problem, args.model, port, args.temperature, args.top_p)
        diagnostics = data.get("diagnostics", {})

        all_rounds.append({
            "round": t + 1,
            "type": "synthesis",
            "text": synthesis,
            "answer": extract_integer_answer(synthesis),
            "diagnostics": diagnostics,
        })
        print(f"  [Problem {problem['idx']}] Round {t+1} synthesis answer: "
              f"{all_rounds[-1]['answer']}")

        # For subsequent rounds: generate new candidates conditioned on synthesis
        if t < T - 1:
            candidates = generate_candidates(problem, K, tokenizer, port, args)
            all_rounds.append({
                "round": t + 1,
                "type": "candidates",
                "texts": candidates,
                "answers": [extract_integer_answer(c) for c in candidates],
            })

    elapsed = time.time() - t0
    final_text = all_rounds[-1]["text"]

    return {
        "text": final_text,
        "rounds": all_rounds,
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AIME evaluation with parallel reasoning")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model", default=None,
                        help="Model name (auto-detected from server if omitted)")
    parser.add_argument("--year", type=int, default=2025, choices=[2024, 2025])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ports", type=str, default=None,
                        help="Comma-separated ports for DP, e.g. '8000,8001,8002,8003'")
    parser.add_argument("--K", type=int, default=4, help="Number of candidates")
    parser.add_argument("--T", type=int, default=2, help="Number of synthesis rounds")
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
    parser.add_argument("--summary-prompt", type=str, default=None,
                        help="Prompt appended after each document before probe "
                             "generation to steer importance scoring toward "
                             "key reasoning (e.g. 'Summarize the key steps:')")
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

    # Health check all servers
    for port in ports:
        health = httpx.get(f"http://localhost:{port}/health", timeout=10.0)
        health.raise_for_status()
    print(f"Server health: OK ({len(ports)} server(s) on ports {ports})\n")

    print(f"Config: K={args.K}, T={args.T}, ratio={args.compact_ratio}, "
          f"beta={args.compute_beta}")
    print(f"Candidate tokens: {args.candidate_tokens}, "
          f"Synthesis tokens: {args.max_gen_tokens}")
    print(f"Probe tokens: {args.probe_tokens}\n")

    results = [None] * len(problems)
    total_correct = 0
    completed = 0

    output_path = (args.output
                   or f"results_aime_parallel_K{args.K}_T{args.T}_{len(problems)}.json")

    t_total = time.time()

    def _save_checkpoint():
        summary = {
            "mode": "parallel",
            "n_problems": len(problems),
            "accuracy": total_correct / max(completed, 1),
            "correct": total_correct,
            "completed": completed,
            "config": {
                "model": args.model,
                "K": args.K,
                "T": args.T,
                "candidate_tokens": args.candidate_tokens,
                "max_gen_tokens": args.max_gen_tokens,
                "compact_ratio": args.compact_ratio,
                "probe_tokens": args.probe_tokens,
                "compute_beta": args.compute_beta,
            },
            "results": results,
        }
        Path(output_path).write_text(json.dumps(summary, indent=2, default=str))

    # Parallel execution: one problem per server, round-robin
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures = {}
        prob_queue = list(enumerate(problems))
        pending = set()

        # Seed initial batch — one problem per port
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
                    "time": round(gen["time"], 2),
                    "rounds": gen["rounds"],
                }

                status = "CORRECT" if correct else "WRONG"
                print(
                    f"[{completed:3d}/{len(problems)}] {prob['id']} {status} "
                    f"extracted={extract_integer_answer(gen['text'])} "
                    f"expected={prob['answer']} "
                    f"time={gen['time']:.1f}s "
                    f"acc={total_correct}/{completed} "
                    f"({total_correct/completed:.1%})"
                )
                _save_checkpoint()

                # Schedule next problem on the same port
                if prob_queue:
                    ni, nprob = prob_queue.pop(0)
                    nfut = pool.submit(
                        run_one_problem, nprob, tokenizer, port, args)
                    futures[nfut] = (ni, port)
                    pending.add(nfut)

    wall_time = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"AIME RESULTS: PARALLEL REASONING")
    print(f"{'='*60}")
    print(f"Accuracy: {total_correct}/{len(problems)} "
          f"({total_correct/len(problems):.1%})")
    print(f"Wall time: {wall_time:.1f}s "
          f"({wall_time/len(problems):.1f}s/problem)")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
