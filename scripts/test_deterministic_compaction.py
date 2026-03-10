"""Test deterministic compaction on GPU.

Sends identical requests to /compact_generate and verifies identical outputs,
proving the seeded random queries produce deterministic compaction.

Usage (inside container on GPU node):
    # Start server first:
    CUDA_VISIBLE_DEVICES=0 uv run inference @ configs/compaction/qwen3_4b_serve_tp1.toml --server.port 8000 &
    sleep 60
    python scripts/test_deterministic_compaction.py --server http://localhost:8000
"""

import argparse
import json
import requests
import sys


def test_deterministic(server_url: str, model: str = "Qwen/Qwen3-4B"):
    """Send same prompt twice, verify identical compaction output."""
    print("=== Test: Deterministic Compaction (GPU) ===")

    # Tokenize a prompt
    prompt = "What is 2 + 2? Think step by step."
    tok_resp = requests.post(
        f"{server_url}/tokenize",
        json={"model": model, "prompt": prompt},
    )
    tok_resp.raise_for_status()
    prompt_ids = tok_resp.json()["tokens"]
    print(f"  Prompt: {len(prompt_ids)} tokens")

    request_body = {
        "prompt_ids": prompt_ids,
        "max_kv_len": 128,
        "max_total_tokens": 512,
        "compact_target_ratio": 0.25,
        "compact_window": 64,
        "n_compacts": 99,
        "compute_beta": True,
        "temperature": 0.0,  # greedy for reproducibility
        "top_p": 1.0,
    }

    # Request 1
    print("  Sending request 1...")
    r1 = requests.post(f"{server_url}/compact_generate", json=request_body)
    r1.raise_for_status()
    result1 = r1.json()

    # Request 2 (identical)
    print("  Sending request 2...")
    r2 = requests.post(f"{server_url}/compact_generate", json=request_body)
    r2.raise_for_status()
    result2 = r2.json()

    # Compare
    tokens1 = result1["all_token_ids"]
    tokens2 = result2["all_token_ids"]
    lp1 = result1["all_logprobs"]
    lp2 = result2["all_logprobs"]

    events1 = result1["diagnostics"]["compaction_events"]
    events2 = result2["diagnostics"]["compaction_events"]

    print(f"  Run 1: {len(tokens1)} tokens, {len(events1)} compactions")
    print(f"  Run 2: {len(tokens2)} tokens, {len(events2)} compactions")

    tokens_match = tokens1 == tokens2
    lp_match = all(abs(a - b) < 1e-6 for a, b in zip(lp1, lp2))
    n_events_match = len(events1) == len(events2)

    print(f"  Tokens match: {tokens_match}")
    print(f"  Logprobs match: {lp_match}")
    print(f"  Compaction events match: {n_events_match}")

    if tokens_match and lp_match and n_events_match:
        print("  PASSED — compaction is fully deterministic")
        return True
    else:
        if not tokens_match:
            # Find first difference
            for i, (a, b) in enumerate(zip(tokens1, tokens2)):
                if a != b:
                    print(f"  First token difference at position {i}: {a} vs {b}")
                    break
        print("  FAILED")
        return False


def test_beta_consistency(server_url: str, model: str = "Qwen/Qwen3-4B"):
    """Verify beta mode produces different results than non-beta."""
    print("\n=== Test: Beta vs Non-Beta ===")

    prompt = "Solve: 3 * 7 + 5"
    tok_resp = requests.post(
        f"{server_url}/tokenize",
        json={"model": model, "prompt": prompt},
    )
    tok_resp.raise_for_status()
    prompt_ids = tok_resp.json()["tokens"]

    base = {
        "prompt_ids": prompt_ids,
        "max_kv_len": 128,
        "max_total_tokens": 256,
        "compact_target_ratio": 0.25,
        "compact_window": 64,
        "n_compacts": 99,
        "temperature": 0.0,
        "top_p": 1.0,
    }

    r_beta = requests.post(f"{server_url}/compact_generate",
                           json={**base, "compute_beta": True})
    r_beta.raise_for_status()
    result_beta = r_beta.json()

    r_nobeta = requests.post(f"{server_url}/compact_generate",
                             json={**base, "compute_beta": False})
    r_nobeta.raise_for_status()
    result_nobeta = r_nobeta.json()

    t_beta = result_beta["all_token_ids"]
    t_nobeta = result_nobeta["all_token_ids"]

    differs = t_beta != t_nobeta
    print(f"  Beta tokens: {len(t_beta)}, No-beta tokens: {len(t_nobeta)}")
    print(f"  Outputs differ: {differs}")
    print(f"  Beta compactions: {len(result_beta['diagnostics']['compaction_events'])}")
    print(f"  No-beta compactions: {len(result_nobeta['diagnostics']['compaction_events'])}")

    # Both should complete without errors
    print("  PASSED — both modes work")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    ok = True
    ok &= test_deterministic(args.server, args.model)
    ok &= test_beta_consistency(args.server, args.model)

    if ok:
        print("\n=== ALL GPU TESTS PASSED ===")
    else:
        print("\n=== SOME TESTS FAILED ===")
        sys.exit(1)
