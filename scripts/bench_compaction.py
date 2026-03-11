"""Mini benchmark for compaction inference speed.

Tests compaction algorithm + KV manipulation timing on 30 problems.
Run inside container with 4 servers already started:

    python scripts/bench_compaction.py --n 30

Also supports --algo-only to benchmark just the compaction algorithm
without needing servers (useful for rapid iteration):

    python scripts/bench_compaction.py --algo-only
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict

import torch

sys.stdout.reconfigure(line_buffering=True)


def bench_algo(num_kv_heads=8, head_size=128, num_layers=36,
               window=1024, target_ratio=0.25, asst_len=1500,
               prompt_len=200, num_queries=64, n_trials=5):
    """Benchmark compact_kv algorithm in isolation."""
    from prime_rl.inference.compaction.algorithm import compact_kv

    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = prompt_len + asst_len

    # Build synthetic KV cache
    keys = [torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
            for _ in range(num_layers)]
    values = [torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device=device)
              for _ in range(num_layers)]

    # Warmup
    compact_kv(keys, values, prompt_len, target_ratio,
               num_kv_heads, head_size, device,
               num_queries=num_queries, compact_window=window)
    torch.cuda.synchronize()

    times = []
    for trial in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.time()
        c1, c2, _, _ = compact_kv(keys, values, prompt_len, target_ratio,
                             num_kv_heads, head_size, device,
                             num_queries=num_queries, compact_window=window)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        target_len = c1[0].shape[0]
        print(f"  Trial {trial}: {elapsed:.3f}s "
              f"(window={window} -> {target_len}, {num_layers} layers × {num_kv_heads} heads)")

    mean_t = sum(times) / len(times)
    min_t = min(times)
    print(f"\n  Mean: {mean_t:.3f}s, Min: {min_t:.3f}s")
    return mean_t


def bench_server(n=30, ports=None):
    """Benchmark end-to-end compaction generation on running servers."""
    import concurrent.futures
    import httpx
    from transformers import AutoTokenizer
    from reasoning_gym.utils import SYSTEM_PROMPTS

    ports = ports or [8000, 8001, 8002, 8003]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    system_prompt = SYSTEM_PROMPTS["default"]

    # Simple countdown problems for consistent benchmarking
    problems = []
    for i in range(n):
        q = f"Find numbers from [1,2,3,4,5,6,7] that sum to {10 + i % 20}. Show your reasoning step by step."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)
        problems.append({"idx": i, "prompt_ids": prompt_ids})

    body_template = {
        "max_kv_len": 2048,
        "max_total_tokens": 8192,
        "n_compacts": 99,
        "compact_target_ratio": 0.25,
        "compact_window": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
    }

    def run_one(prob, port):
        t0 = time.time()
        client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)
        body = {**body_template, "prompt_ids": prob["prompt_ids"]}
        resp = client.post("/compact_generate", json=body)
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        return {
            "idx": prob["idx"],
            "tokens": len(data["all_token_ids"]),
            "time": elapsed,
            "n_compacts": len(data["diagnostics"].get("compaction_events", [])),
            "diagnostics": data["diagnostics"],
        }

    print(f"Running {n} problems across {len(ports)} servers...")
    t_start = time.time()

    results = []
    batch_size = len(ports)
    for batch_start in range(0, n, batch_size):
        batch = problems[batch_start:batch_start + batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as ex:
            futures = {}
            for j, prob in enumerate(batch):
                port = ports[j % len(ports)]
                futures[ex.submit(run_one, prob, port)] = prob["idx"]

            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                results.append(r)
                print(f"  [{len(results):3d}/{n}] tokens={r['tokens']:5d} "
                      f"compacts={r['n_compacts']} time={r['time']:.1f}s")

    wall_time = time.time() - t_start
    total_tokens = sum(r["tokens"] for r in results)

    # Timing breakdown
    algo_times, extract_times, inject_times, compact_times = [], [], [], []
    for r in results:
        for evt in r["diagnostics"].get("compaction_events", []):
            algo_times.append(evt.get("algo_time", 0))
            extract_times.append(evt.get("extract_time", 0))
            inject_times.append(evt.get("inject_time", 0))
            compact_times.append(evt.get("total_time", 0))

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({n} problems, {len(ports)} servers)")
    print(f"{'='*60}")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput: {total_tokens/wall_time:.0f} tok/s aggregate")
    print(f"Avg time/request: {sum(r['time'] for r in results)/n:.1f}s")
    print(f"Avg tokens/request: {total_tokens/n:.0f}")

    if algo_times:
        n_evt = len(algo_times)
        print(f"\nCompaction ({n_evt} events):")
        print(f"  algo:    mean={sum(algo_times)/n_evt:.3f}s, total={sum(algo_times):.1f}s")
        print(f"  extract: mean={sum(extract_times)/n_evt:.3f}s, total={sum(extract_times):.1f}s")
        print(f"  inject:  mean={sum(inject_times)/n_evt:.3f}s, total={sum(inject_times):.1f}s")
        print(f"  total:   mean={sum(compact_times)/n_evt:.3f}s, total={sum(compact_times):.1f}s")
        total_req_time = sum(r["time"] for r in results)
        compact_pct = sum(compact_times) / total_req_time * 100
        print(f"  Compaction % of total: {compact_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-only", action="store_true",
                        help="Benchmark algorithm only (no server needed)")
    parser.add_argument("--n", type=int, default=30, help="Number of problems")
    parser.add_argument("--ports", default="8000,8001,8002,8003")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Number of algo benchmark trials")
    args = parser.parse_args()

    if args.algo_only:
        print("Benchmarking compact_kv algorithm (Qwen3-4B dimensions)...")
        print("  36 layers, 8 KV heads, 128 head_size, window=1024, ratio=0.25\n")
        bench_algo(n_trials=args.n_trials)
    else:
        ports = [int(p) for p in args.ports.split(",")]
        bench_server(n=args.n, ports=ports)


if __name__ == "__main__":
    main()
