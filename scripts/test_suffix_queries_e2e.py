"""End-to-end test of suffix queries with vLLM inference server.

Starts a server with CompactionWorker, sends /compact_generate requests
with and without use_suffix_queries, and verifies both produce valid output.

Run inside container on GPU node:
    source .venv/bin/activate
    python scripts/test_suffix_queries_e2e.py
"""

import asyncio
import sys
import time

import httpx


SERVER_URL = "http://localhost:8000"
MODEL = "Qwen/Qwen3-4B"


async def test_suffix_queries():
    print("=== Suffix Queries E2E Test ===\n")

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Tokenize a prompt
        prompt = [{"role": "user", "content": "What is 15 + 27? Think step by step."}]
        tok_resp = await client.post(
            f"{SERVER_URL}/tokenize",
            json={"model": MODEL, "messages": prompt, "add_generation_prompt": True},
        )
        tok_resp.raise_for_status()
        prompt_ids = tok_resp.json()["tokens"]
        print(f"Prompt length: {len(prompt_ids)} tokens\n")

        base_request = {
            "prompt_ids": prompt_ids,
            "max_kv_len": 2048,
            "max_total_tokens": 4096,
            "compact_target_ratio": 0.25,
            "compact_window": 1024,
            "n_compacts": 99,
            "temperature": 0.7,
            "top_p": 0.95,
        }

        # Test 1: Without suffix queries (baseline)
        print("--- Test 1: Random queries (baseline) ---")
        t0 = time.time()
        resp = await client.post(
            f"{SERVER_URL}/compact_generate",
            json={**base_request, "use_suffix_queries": False},
        )
        resp.raise_for_status()
        result_random = resp.json()
        dt_random = time.time() - t0
        diag = result_random["diagnostics"]
        print(f"  Tokens: {diag['total_tokens']}")
        print(f"  Compactions: {len(diag['compaction_events'])}")
        print(f"  Mean logprob: {diag['mean_logprob']:.4f}")
        print(f"  Time: {dt_random:.2f}s")
        print()

        # Test 2: With suffix queries
        print("--- Test 2: Suffix queries ---")
        t0 = time.time()
        resp = await client.post(
            f"{SERVER_URL}/compact_generate",
            json={**base_request, "use_suffix_queries": True},
        )
        resp.raise_for_status()
        result_suffix = resp.json()
        dt_suffix = time.time() - t0
        diag = result_suffix["diagnostics"]
        print(f"  Tokens: {diag['total_tokens']}")
        print(f"  Compactions: {len(diag['compaction_events'])}")
        print(f"  Mean logprob: {diag['mean_logprob']:.4f}")
        print(f"  Time: {dt_suffix:.2f}s")
        print()

        # Verify basic sanity
        assert len(result_random["all_token_ids"]) > 0, "Random: no tokens generated"
        assert len(result_suffix["all_token_ids"]) > 0, "Suffix: no tokens generated"
        assert len(result_random["all_logprobs"]) == len(result_random["all_token_ids"])
        assert len(result_suffix["all_logprobs"]) == len(result_suffix["all_token_ids"])

        print(f"--- Overhead: suffix queries add {dt_suffix - dt_random:.2f}s ---")
        print()

        # Test 3: Batch mode with suffix queries
        print("--- Test 3: Batch mode with suffix queries ---")
        t0 = time.time()
        resp = await client.post(
            f"{SERVER_URL}/compact_generate_batch",
            json={
                "prompt_ids_list": [prompt_ids, prompt_ids],
                "max_kv_len": 2048,
                "max_total_tokens": 2048,
                "compact_target_ratio": 0.25,
                "compact_window": 1024,
                "n_compacts": 99,
                "temperature": 0.7,
                "top_p": 0.95,
                "use_suffix_queries": True,
            },
        )
        resp.raise_for_status()
        batch_results = resp.json()["results"]
        dt_batch = time.time() - t0
        print(f"  Batch size: {len(batch_results)}")
        for i, r in enumerate(batch_results):
            d = r["diagnostics"]
            print(f"  Seq {i}: {d['total_tokens']} tokens, {len(d['compaction_events'])} compactions")
        print(f"  Time: {dt_batch:.2f}s")
        print()

        print("=== ALL E2E TESTS PASSED ===")


if __name__ == "__main__":
    asyncio.run(test_suffix_queries())
