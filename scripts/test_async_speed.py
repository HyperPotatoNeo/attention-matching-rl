"""Quick speed comparison: sequential vs async markovian on 3 problems."""

import concurrent.futures
import time

import httpx
from transformers import AutoTokenizer
from reasoning_gym.utils import SYSTEM_PROMPTS
import reasoning_gym as rg

MODEL = "Qwen/Qwen3-4B"
PORT = 8000
N = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL)
ds = rg.create_dataset("countdown", seed=42, size=10, min_numbers=7, max_numbers=7)

SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]


def make_request(entry):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": entry["question"]},
    ]
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)
    return {
        "prompt_ids": prompt_ids,
        "max_kv_len": 2048,
        "max_total_tokens": 4096,
        "n_compacts": 99,
        "compact_target_ratio": 0.25,
        "compact_window": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "compaction_mode": "markovian",
        "carryover_ratio": 0.5,
    }


def send_one(body):
    client = httpx.Client(base_url=f"http://localhost:{PORT}", timeout=600.0)
    resp = client.post("/compact_generate", json=body)
    resp.raise_for_status()
    data = resp.json()
    return len(data["all_token_ids"])


entries = [ds[i] for i in range(N)]
bodies = [make_request(e) for e in entries]

# --- Sequential ---
print(f"Sequential: {N} requests one at a time...")
t0 = time.time()
seq_tokens = 0
for i, body in enumerate(bodies):
    toks = send_one(body)
    seq_tokens += toks
    print(f"  [{i+1}/{N}] {toks} tokens")
seq_time = time.time() - t0
print(f"  Total: {seq_tokens} tokens in {seq_time:.1f}s ({seq_tokens/seq_time:.0f} tok/s)\n")

# --- Async (concurrent) ---
print(f"Async: {N} requests fired concurrently...")
t0 = time.time()
async_tokens = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
    futs = {ex.submit(send_one, body): i for i, body in enumerate(bodies)}
    for f in concurrent.futures.as_completed(futs):
        toks = f.result()
        async_tokens += toks
        print(f"  [{futs[f]+1}/{N}] {toks} tokens")
async_time = time.time() - t0
print(f"  Total: {async_tokens} tokens in {async_time:.1f}s ({async_tokens/async_time:.0f} tok/s)\n")

print(f"Speedup: {seq_time/async_time:.2f}x")
