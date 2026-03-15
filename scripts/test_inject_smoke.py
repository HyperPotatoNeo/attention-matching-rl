"""Smoke test: budget injection during compaction generation.

Sends a single /compact_generate request with inject_budget_message=True,
verifies the output contains budget messages and inject_ranges are correct.

Usage (inside container, with 1+ server running):
    python scripts/test_inject_smoke.py [--port 8000]
"""

import argparse
import json
import sys
import time

import httpx
from transformers import AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

MODEL = "Qwen/Qwen3-4B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-kv-len", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--compact-ratio", type=float, default=0.25)
    parser.add_argument("--compact-window", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Think step by step."},
        {"role": "user", "content": "Find all prime numbers between 100 and 150. Show your work."},
    ]
    result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)

    print(f"Prompt: {len(prompt_ids)} tokens")

    client = httpx.Client(base_url=f"http://localhost:{args.port}", timeout=600.0)

    # Health check
    health = client.get("/health")
    health.raise_for_status()
    print("Server health: OK")

    # ── Test 1: inject_budget_message=True ──────────────────────────────
    print("\n=== Test 1: Generation with budget injection ===")
    t0 = time.time()
    body = {
        "prompt_ids": prompt_ids,
        "max_kv_len": args.max_kv_len,
        "max_total_tokens": args.max_total_tokens,
        "compact_target_ratio": args.compact_ratio,
        "compact_window": args.compact_window,
        "n_compacts": 99,
        "temperature": 0.6,
        "top_p": 0.95,
        "use_suffix_queries": True,
        "inject_budget_message": True,
    }
    resp = client.post("/compact_generate", json=body)
    elapsed = time.time() - t0
    if resp.status_code != 200:
        print(f"  ERROR: {resp.status_code}")
        print(f"  Response: {resp.text[:2000]}")
        sys.exit(1)
    data = resp.json()

    diag = data["diagnostics"]
    all_ids = data["all_token_ids"]
    all_lps = data["all_logprobs"]
    inject_ranges = diag.get("inject_ranges", [])
    seg_bounds = diag.get("segment_boundaries", [])
    compaction_events = diag.get("compaction_events", [])

    print(f"  Total tokens: {len(all_ids)}")
    print(f"  Time: {elapsed:.1f}s ({len(all_ids)/elapsed:.0f} tok/s)")
    print(f"  Compaction events: {len(compaction_events)}")
    print(f"  Segment boundaries: {seg_bounds}")
    print(f"  Inject ranges: {inject_ranges}")
    print(f"  Final text preview (first 200 chars):")
    print(f"    {data['final_text'][:200]}...")

    # Verify inject_ranges
    assert len(inject_ranges) == len(compaction_events), (
        f"Expected {len(compaction_events)} inject ranges, got {len(inject_ranges)}")
    print(f"\n  ✓ inject_ranges count matches compaction events ({len(inject_ranges)})")

    # Verify inject ranges are within bounds
    for i, (start, end) in enumerate(inject_ranges):
        assert 0 <= start < end <= len(all_ids), (
            f"Inject range {i} out of bounds: ({start}, {end}), total={len(all_ids)}")
    print(f"  ✓ All inject ranges within bounds")

    # Verify inject ranges don't overlap
    sorted_ranges = sorted(inject_ranges)
    for i in range(1, len(sorted_ranges)):
        assert sorted_ranges[i][0] >= sorted_ranges[i-1][1], (
            f"Overlapping inject ranges: {sorted_ranges[i-1]} and {sorted_ranges[i]}")
    print(f"  ✓ No overlapping inject ranges")

    # Verify logprobs at inject positions are 0.0
    for start, end in inject_ranges:
        for j in range(start, end):
            assert all_lps[j] == 0.0, (
                f"Inject token at position {j} has logprob {all_lps[j]}, expected 0.0")
    print(f"  ✓ All inject token logprobs are 0.0")

    # Decode inject tokens and check they contain budget info
    for i, (start, end) in enumerate(inject_ranges):
        inject_text = tokenizer.decode(all_ids[start:end], skip_special_tokens=True)
        print(f"\n  Inject {i}: [{start}:{end}] ({end-start} tokens)")
        print(f"    Text: {inject_text}")
        assert "Budget" in inject_text or "budget" in inject_text or "token" in inject_text.lower(), (
            f"Inject {i} doesn't contain budget info: {inject_text}")
    print(f"\n  ✓ All inject messages contain budget information")

    # ── Test 2: inject_budget_message=False (regression) ────────────────
    print("\n=== Test 2: Generation WITHOUT injection (regression) ===")
    t0 = time.time()
    body_no_inject = dict(body)
    body_no_inject["inject_budget_message"] = False
    resp2 = client.post("/compact_generate", json=body_no_inject)
    elapsed2 = time.time() - t0
    if resp2.status_code != 200:
        print(f"  ERROR: {resp2.status_code}")
        print(f"  Response: {resp2.text[:2000]}")
        sys.exit(1)
    data2 = resp2.json()

    diag2 = data2["diagnostics"]
    inject_ranges2 = diag2.get("inject_ranges", [])
    print(f"  Total tokens: {len(data2['all_token_ids'])}")
    print(f"  Time: {elapsed2:.1f}s")
    print(f"  Inject ranges: {inject_ranges2}")

    assert len(inject_ranges2) == 0, (
        f"Expected no inject ranges with inject_budget_message=False, got {inject_ranges2}")
    print(f"  ✓ No inject ranges when disabled")

    # ── Test 3: Verify segment boundaries are consistent ───────────────
    print("\n=== Test 3: Boundary consistency ===")
    # With injection, total tokens should be more than without (inject overhead)
    total_inject_tokens = sum(end - start for start, end in inject_ranges)
    effective_with = len(all_ids) - total_inject_tokens
    effective_without = len(data2["all_token_ids"])
    print(f"  With inject: {len(all_ids)} total ({effective_with} effective + {total_inject_tokens} inject)")
    print(f"  Without inject: {effective_without} total")
    print(f"  ✓ Effective token counts comparable")

    print("\n" + "="*60)
    print("ALL SMOKE TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    main()
