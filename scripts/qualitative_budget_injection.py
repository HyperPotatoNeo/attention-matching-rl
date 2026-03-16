"""Capture full response text for specific rg-mix problems across baseline and inject modes.

Usage (inside container with 4 servers running):
    python scripts/qualitative_budget_injection.py \
        --model Qwen/Qwen3-4B \
        --indices 0,21,36,68,106 \
        --output qualitative_base.json

    python scripts/qualitative_budget_injection.py \
        --model /path/to/step600 \
        --indices 0,21,36,68,106 \
        --output qualitative_step600.json
"""

import argparse
import json
import random
import sys

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


def generate_dataset(n, seed=42):
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
            "idx": i, "task": vid,
            "question": entry["question"], "entry": entry, "dataset": ds,
        })
    return problems


def extract_answer(text):
    import re
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


def run_baseline(problem, port, model_name, max_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
    ]
    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)
    resp = client.post("/v1/chat/completions", json={
        "model": model_name, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
    })
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    tokens = data["usage"]["completion_tokens"]
    return text, tokens


def run_inject(problem, port, tokenizer, max_tokens, inject_every=2048):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["question"]},
    ]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)
    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)
    resp = client.post("/inject_generate", json={
        "prompt_ids": prompt_ids,
        "max_total_tokens": max_tokens,
        "inject_budget_every": inject_every,
        "temperature": 0.6, "top_p": 0.95,
    })
    resp.raise_for_status()
    data = resp.json()
    text = data["final_text"]
    tokens = len(data["all_token_ids"])
    return text, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--indices", required=True, help="Comma-separated problem indices")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    indices = [int(x) for x in args.indices.split(",")]
    n_total = max(indices) + 1

    print(f"Generating {n_total} problems (seed=42) to access indices {indices}...")
    problems = generate_dataset(n_total, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = []
    for idx in indices:
        prob = problems[idx]
        print(f"\n--- Problem idx={idx}, task={prob['task']} ---")

        print(f"  Running baseline...")
        text_base, tok_base = run_baseline(prob, args.port, args.model, args.max_tokens)
        extracted_base = extract_answer(text_base)
        score_base = prob["dataset"].score_answer(answer=extracted_base, entry=prob["entry"])
        score_base = max(score_base, prob["dataset"].score_answer(answer=text_base, entry=prob["entry"]))
        correct_base = score_base >= 0.5

        print(f"  Running inject...")
        text_inject, tok_inject = run_inject(prob, args.port, tokenizer, args.max_tokens)
        extracted_inject = extract_answer(text_inject)
        score_inject = prob["dataset"].score_answer(answer=extracted_inject, entry=prob["entry"])
        score_inject = max(score_inject, prob["dataset"].score_answer(answer=text_inject, entry=prob["entry"]))
        correct_inject = score_inject >= 0.5

        tag_b = "OK" if correct_base else "FAIL"
        tag_i = "OK" if correct_inject else "FAIL"
        print(f"  baseline: {tag_b} ({tok_base} tok)  inject: {tag_i} ({tok_inject} tok)")

        results.append({
            "idx": idx,
            "task": prob["task"],
            "question": prob["question"],
            "baseline": {
                "text": text_base,
                "tokens": tok_base,
                "correct": correct_base,
            },
            "inject": {
                "text": text_inject,
                "tokens": tok_inject,
                "correct": correct_inject,
            },
        })

    json.dump(results, open(args.output, "w"), indent=2, default=str)
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
