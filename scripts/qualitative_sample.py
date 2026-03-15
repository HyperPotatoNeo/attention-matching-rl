"""Capture full text from sokoban problems for qualitative analysis."""
import argparse, json, httpx
import reasoning_gym as rg
from reasoning_gym.utils import SYSTEM_PROMPTS
from transformers import AutoTokenizer

SOKOBAN_CONFIG = {"min_boxes": 3, "max_boxes": 4, "max_w": 9, "max_h": 9}
SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "compaction"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-kv-len", type=int, default=2048)
    parser.add_argument("--compact-window", type=int, default=1024)
    parser.add_argument("--compact-ratio", type=float, default=0.03125)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--use-suffix-queries", action="store_true")
    args = parser.parse_args()

    ds = rg.create_dataset("sokoban", seed=args.seed + 1, size=args.n, **SOKOBAN_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    client = httpx.Client(base_url=f"http://localhost:{args.port}", timeout=600.0)
    
    results = []
    for i in range(args.n):
        entry = ds[i]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry["question"]},
        ]
        
        if args.mode == "compaction":
            result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)
            body = {
                "prompt_ids": prompt_ids,
                "max_seq_len": len(prompt_ids) + args.max_total_tokens,
                "max_tokens_per_segment": 0,
                "n_compacts": 99,
                "compact_target_ratio": args.compact_ratio,
                "compact_window": args.compact_window,
                "temperature": 0.6, "top_p": 0.95,
                "max_kv_len": args.max_kv_len,
                "max_total_tokens": args.max_total_tokens,
                "use_suffix_queries": args.use_suffix_queries,
            }
            resp = client.post("/compact_generate", json=body)
            resp.raise_for_status()
            data = resp.json()
            text = data["final_text"]
            diag = data.get("diagnostics", {})
        else:
            resp = client.post("/v1/chat/completions", json={
                "model": args.model, "messages": messages,
                "max_tokens": args.max_total_tokens,
                "temperature": 0.6, "top_p": 0.95,
            })
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            diag = {}

        score = ds.score_answer(answer=text, entry=entry)
        correct = score >= 0.5
        n_compacts = len(diag.get("compaction_events", []))
        tag = "OK" if correct else "FAIL"
        print(f"[{i+1}/{args.n}] {tag} tokens={len(text.split())} compacts={n_compacts}")
        
        results.append({
            "idx": i,
            "question": entry["question"][:500],
            "text": text,
            "correct": correct,
            "score": score,
            "segment_boundaries": diag.get("segment_boundaries", []),
            "compaction_events": [
                {"kv_before": e["kv_len_before"], "kv_after": e["kv_len_after"], "ratio": e["ratio"]}
                for e in diag.get("compaction_events", [])
            ],
        })

    json.dump(results, open(args.output, "w"), indent=2)
    print(f"Saved {len(results)} results to {args.output}")

if __name__ == "__main__":
    main()
