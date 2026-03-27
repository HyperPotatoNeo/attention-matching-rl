"""Measure inference-trainer log-prob mismatch for markovian or attention_matching compaction on BabyAI.

Runs n episodes per task (8 tasks = 8*n total) with max_turns=1 and compaction,
then recomputes logprobs via segmented_forward with the HF model and reports
the per-token mismatch_kl = exp(Δlogp) - Δlogp - 1.

Usage:
    # Markovian mode (default)
    python scripts/eval_markovian_mismatch.py --n 3 --port 8000 \
        --max-kv-len 512 --compact-window 256

    # Attention matching mode
    python scripts/eval_markovian_mismatch.py --mode attention_matching --n 3 --port 8000 \
        --max-kv-len 512 --compact-window 256

    # Or specify model explicitly:
    python scripts/eval_markovian_mismatch.py --n 3 --port 8000 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --max-kv-len 512 --compact-window 256
"""

import argparse
import re
import sys

import gymnasium as gym
import httpx
import minigrid
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

minigrid.register_minigrid_envs()

sys.stdout.reconfigure(line_buffering=True)

from prime_rl.trainer.rl.compaction import segmented_forward

BABYAI_TASKS = [
    "GoToObj", "GoToLocal", "PickupLoc", "Open",
    "PutNextLocal", "GoTo", "Unlock", "UnlockLocal",
]

ACTIONS = {
    "turn left": 0, "turn right": 1, "go forward": 2,
    "pick up": 3, "drop": 4, "toggle": 5,
}

OBJECTS = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door",
           5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava"}
COLORS = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}
DOOR_STATES = {0: "open", 1: "closed", 2: "locked"}

SYSTEM_PROMPT = """You are navigating a grid-world environment (BabyAI).

Available actions: {actions}

Each turn you receive a text observation showing your mission and what you can see.
Reason briefly about what to do, then output exactly one action inside <action>...</action> tags.
Example: <action>go forward</action>""".format(actions=", ".join(ACTIONS.keys()))


def render_observation(obs):
    image = obs["image"]
    mission = obs["mission"]
    grid_lines = []
    legend_items = set()
    for row in range(7):
        cells = []
        for col in range(7):
            obj_idx, color_idx, state = image[col][row]
            obj = OBJECTS.get(obj_idx, "?")
            color = COLORS.get(color_idx, "")
            if row == 6 and col == 3:
                cells.append(" @")
                continue
            if obj in ("unseen", "empty"):
                cells.append(" .")
            elif obj == "wall":
                cells.append(" #")
            elif obj == "door":
                ds = DOOR_STATES.get(state, "")
                symbol = f"{color[0]}D"
                cells.append(f"{symbol:>2}")
                legend_items.add(f"{symbol} = {color} {ds} door")
            else:
                symbol = f"{color[0]}{obj[0]}"
                cells.append(f"{symbol:>2}")
                legend_items.add(f"{symbol} = {color} {obj}")
        grid_lines.append("".join(cells))
    lines = [f"Mission: {mission}", "", "View (you are @ at bottom center, facing up):", *grid_lines]
    if legend_items:
        lines.append(f"Legend: # = wall, @ = you, {', '.join(sorted(legend_items))}")
    return "\n".join(lines)


def collect_rollouts(args, tokenizer):
    """Collect one BabyAI turn per episode via /compact_generate."""
    client = httpx.Client(base_url=f"http://localhost:{args.port}", timeout=600.0)
    rollouts = []

    seed_offset = getattr(args, "_step_offset", 0)
    for env_name in BABYAI_TASKS:
        for seed in range(seed_offset, seed_offset + args.n):
            env_id = f"BabyAI-{env_name}-v0"
            env = gym.make(env_id, render_mode=None)
            obs, _ = env.reset(seed=seed)
            obs_text = render_observation(obs)
            env.close()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ]
            result = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
            )
            prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)

            body = {
                "prompt_ids": prompt_ids,
                "max_total_tokens": args.max_total_tokens,
                "max_kv_len": args.max_kv_len,
                "compact_window": args.compact_window,
                "compact_target_ratio": 0.25,
                "n_compacts": 99,
                "temperature": args.temperature,
                "top_p": 0.95,
                "compaction_mode": args.mode,
                "use_suffix_queries": False,
            }
            resp = client.post("/compact_generate", json=body)
            resp.raise_for_status()
            data = resp.json()

            diag = data.get("diagnostics", {})
            segment_boundaries = diag.get("segment_boundaries", [len(data["all_token_ids"])])
            n_compactions = len(diag.get("compaction_events", []))

            rollouts.append({
                "env": env_name,
                "seed": seed,
                "prompt_ids": prompt_ids,
                "all_token_ids": data["all_token_ids"],
                "all_logprobs": data["all_logprobs"],
                "segment_boundaries": segment_boundaries,
                "n_compactions": n_compactions,
                "prompt_len": diag.get("prompt_len", len(prompt_ids)),
            })

            print(
                f"  {env_name}[{seed}] prompt={len(prompt_ids)} "
                f"completion={len(data['all_token_ids'])} "
                f"segments={len(segment_boundaries)} compactions={n_compactions}"
            )

    return rollouts


def compute_trainer_logprobs(model, rollout, device, temperature, compact_window, mode):
    """Run segmented_forward and return per-completion-token logprobs."""
    prompt_ids = rollout["prompt_ids"]
    all_token_ids = rollout["all_token_ids"]
    segment_boundaries = rollout["segment_boundaries"]
    prompt_len = rollout["prompt_len"]

    if not all_token_ids:
        return [], []

    full_ids = prompt_ids + all_token_ids
    seq_len = len(full_ids)

    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    temperatures = torch.full((1, seq_len), temperature, dtype=torch.float32, device=device)

    use_suffix_queries = (mode == "attention_matching")

    with torch.no_grad():
        out = segmented_forward(
            model,
            input_ids,
            position_ids,
            segment_boundaries=segment_boundaries,
            prompt_len=prompt_len,
            compact_target_ratio=0.25,
            compact_window=compact_window,
            temperature=temperatures,
            compaction_mode=mode,
            use_suffix_queries=use_suffix_queries,
        )

    # segmented_forward returns temperature-scaled logits: [1, seq_len, vocab]
    logits = out["logits"]  # already scaled by temperature

    # log_softmax and gather at each position's actual next token
    # logits[:, i, :] predicts token at position i+1
    log_probs = F.log_softmax(logits.float(), dim=-1)  # [1, seq_len, vocab]

    # For completion tokens c_k = all_token_ids[k] at full_ids[prompt_len + k]:
    # the logit predicting c_k is at position (prompt_len + k - 1)
    trainer_lps = []
    for k, tok_id in enumerate(all_token_ids):
        pred_pos = prompt_len + k - 1  # position whose logit predicts token k
        lp = log_probs[0, pred_pos, tok_id].item()
        trainer_lps.append(lp)

    return trainer_lps, rollout["all_logprobs"]


def main():
    parser = argparse.ArgumentParser(description="Inference-trainer mismatch for BabyAI compaction")
    parser.add_argument("--mode", default="markovian", choices=["markovian", "attention_matching"])
    parser.add_argument("--n", type=int, default=3, help="Episodes per task (8 tasks * n = total)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-kv-len", type=int, default=512)
    parser.add_argument("--compact-window", type=int, default=256)
    parser.add_argument("--max-total-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=1, help="Number of rollout+mismatch rounds")
    args = parser.parse_args()

    import math

    n_total = args.n * len(BABYAI_TASKS)
    print(f"Mismatch eval: mode={args.mode}, steps={args.steps}, n={args.n}, tasks={len(BABYAI_TASKS)}, total={n_total}/step")
    print(f"max_kv_len={args.max_kv_len}, compact_window={args.compact_window}, "
          f"max_total_tokens={args.max_total_tokens}, temperature={args.temperature}")

    httpx.get(f"http://localhost:{args.port}/health", timeout=10.0).raise_for_status()
    print("Server health: OK\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"\nLoading model {args.model} on {args.device} for trainer forward pass...")
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = model.to(args.device)
    model.eval()
    model.config.use_cache = True

    all_step_means = []
    global_kls = []

    for step in range(1, args.steps + 1):
        print(f"\n{'='*60}")
        print(f"STEP {step}/{args.steps} — collecting {n_total} rollouts...")
        print(f"{'='*60}")

        args._step_offset = (step - 1) * args.n
        rollouts = collect_rollouts(args, tokenizer)

        n_with_compaction = sum(1 for r in rollouts if r["n_compactions"] > 0)
        print(f"Rollouts with >=1 compaction: {n_with_compaction}/{n_total}")

        print(f"\nComputing mismatch_kl via segmented_forward (mode={args.mode})...")
        step_kls_no_compact = []
        step_kls_compacted = []

        for rollout in rollouts:
            if not rollout["all_token_ids"]:
                continue

            trainer_lps, infer_lps = compute_trainer_logprobs(
                model, rollout, args.device, args.temperature, args.compact_window, args.mode,
            )

            kls = [math.exp(t - i) - (t - i) - 1 for t, i in zip(trainer_lps, infer_lps)]
            if rollout["n_compactions"] > 0:
                step_kls_compacted.extend(kls)
            else:
                step_kls_no_compact.extend(kls)

            mean_kl = sum(kls) / len(kls) if kls else 0.0
            print(
                f"  {rollout['env']}[{rollout['seed']}]: "
                f"tokens={len(trainer_lps)} segs={len(rollout['segment_boundaries'])} "
                f"compactions={rollout['n_compactions']} mean_kl={mean_kl:.6f}"
            )

        step_kls = step_kls_no_compact + step_kls_compacted
        step_mean = sum(step_kls) / len(step_kls) if step_kls else float("nan")
        mean_no_compact = sum(step_kls_no_compact) / len(step_kls_no_compact) if step_kls_no_compact else float("nan")
        mean_compacted = sum(step_kls_compacted) / len(step_kls_compacted) if step_kls_compacted else float("nan")
        all_step_means.append((step_mean, mean_no_compact, mean_compacted))
        global_kls.extend(step_kls)

        print(f"\nStep {step}: mean_kl={step_mean:.6f}  "
              f"no_compaction={mean_no_compact:.6f} ({len(step_kls_no_compact)} tok)  "
              f"with_compaction={mean_compacted:.6f} ({len(step_kls_compacted)} tok)")

    print("\n" + "=" * 60)
    print(f"AGGREGATE MISMATCH_KL — mode={args.mode}")
    print("mismatch_kl = exp(Δlogp) - Δlogp - 1  [same as training loss.py]")
    print("=" * 60)
    print(f"  {'Step':<6}  {'overall':>10}  {'no compaction':>14}  {'with compaction':>16}")
    for s, (m, m_no, m_c) in enumerate(all_step_means, 1):
        print(f"  {s:<6}  {m:>10.6f}  {m_no:>14.6f}  {m_c:>16.6f}")
    print(f"\n  Overall mean  = {sum(global_kls)/len(global_kls):.6f}  ({len(global_kls)} tokens)")
    print(f"  Overall max   = {max(global_kls):.6f}")


if __name__ == "__main__":
    main()
