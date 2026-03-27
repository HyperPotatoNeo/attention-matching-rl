"""Measure inference-trainer mismatch for turn-based session compaction on BabyAI.

Runs multi-turn BabyAI episodes via the session API with n_max_turns=4,
n_preserved_turns=2 (stride=2). Compaction fires after every 4 accumulated
turns, preserving the last 2. Runs 6 env steps per episode so compaction fires
at turn 4 and turns 5-6 are generated with compacted context.

For attention_matching: inference uses the last turn's key vectors as suffix
queries (already implemented in compact_session_step). Training uses
use_suffix_queries=True in segmented_forward, which captures query vectors
from the suffix tokens during the forward pass.

Usage:
    python scripts/eval_session_mismatch.py --mode markovian --n 3 --port 8000
    python scripts/eval_session_mismatch.py --mode attention_matching --n 3 --port 8000
"""

import argparse
import math
import re
import sys
import uuid

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


def parse_action(text):
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip().lower()
        for name in ACTIONS:
            if candidate == name or name in candidate:
                return name
    for name in ACTIONS:
        if name in text.lower():
            return name
    return "go forward"


def new_user_turn_ids(tokenizer, messages_with_new_user, messages_without_new_user):
    """Compute token IDs for just the new user turn appended to the conversation."""
    result_curr = tokenizer.apply_chat_template(
        messages_with_new_user, add_generation_prompt=True, tokenize=True,
    )
    curr_ids = list(result_curr["input_ids"]) if hasattr(result_curr, "keys") else list(result_curr)
    result_prev = tokenizer.apply_chat_template(
        messages_without_new_user, add_generation_prompt=False, tokenize=True,
    )
    prev_ids = list(result_prev["input_ids"]) if hasattr(result_prev, "keys") else list(result_prev)
    return curr_ids[len(prev_ids):]


def collect_episode(client, tokenizer, env_name, seed, args):
    """Run one multi-turn BabyAI episode via session API.

    Returns a rollout dict with full token sequence, inference logprobs,
    per-turn accounting, and compaction events — everything needed for
    segmented_forward mismatch computation.
    """
    env_id = f"BabyAI-{env_name}-v0"
    env = gym.make(env_id, render_mode=None)
    obs, _ = env.reset(seed=seed)

    session_id = str(uuid.uuid4())[:8]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Turn 1: create session
    obs_text = render_observation(obs)
    messages.append({"role": "user", "content": obs_text})
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)

    # Allocate enough KV blocks to hold the full episode without OOM.
    # Turn-based mode doesn't use max_kv_len for compaction trigger, only for block pre-allocation.
    estimated_total = len(prompt_ids) + args.env_steps * (args.max_response_tokens + 300)
    session_max_kv_len = max(args.max_kv_len, estimated_total)

    resp = client.post("/compact_session/create", json={
        "session_id": session_id,
        "prompt_ids": prompt_ids,
        "max_kv_len": session_max_kv_len,
        "max_response_tokens": args.max_response_tokens,
        "compact_target_ratio": args.compact_ratio,
        "temperature": args.temperature,
        "top_p": 0.95,
        "compaction_mode": args.mode,
        "use_suffix_queries": args.mode == "attention_matching",
        "n_max_turns": args.n_max_turns,
        "n_preserved_turns": args.n_preserved_turns,
    })
    resp.raise_for_status()
    data = resp.json()

    # session_token_ids tracks the full KV sequence (prompt + all turn tokens)
    session_token_ids = prompt_ids + data["all_token_ids"]

    # per-turn token accounting for segment boundary computation
    turn_asst_ids = [data["all_token_ids"]]      # list of per-turn response token lists
    turn_user_ids = [[]]                          # turn 0 user = initial prompt (no extra user tokens)
    all_infer_logprobs = list(data["all_logprobs"])

    # Track compaction events: each is {"kv_len_before", "kv_len_after", "after_turn"}
    compaction_events = [
        {**e, "after_turn": 0} for e in data.get("diagnostics", {}).get("compaction_events", [])
    ]

    messages.append({"role": "assistant", "content": data["final_text"]})
    history = [(obs_text, data["final_text"])]

    action = parse_action(data["final_text"])
    obs, _, terminated, truncated, _ = env.step(ACTIONS[action])
    done = terminated or truncated

    # Subsequent turns
    for turn_idx in range(1, args.env_steps):
        if done:
            break

        obs_text = render_observation(obs)
        prev_messages = list(messages)  # snapshot before appending new user turn
        messages.append({"role": "user", "content": obs_text})

        user_ids = new_user_turn_ids(tokenizer, messages, prev_messages)
        boundary_token = session_token_ids[-1]
        new_token_ids = [boundary_token] + user_ids

        resp = client.post("/compact_session/step", json={
            "session_id": session_id,
            "new_token_ids": new_token_ids,
            "max_response_tokens": args.max_response_tokens,
        })
        resp.raise_for_status()
        data = resp.json()

        session_token_ids = session_token_ids + user_ids + data["all_token_ids"]
        turn_user_ids.append(user_ids)
        turn_asst_ids.append(data["all_token_ids"])
        all_infer_logprobs.extend(data["all_logprobs"])

        for e in data.get("diagnostics", {}).get("compaction_events", []):
            compaction_events.append({**e, "after_turn": turn_idx})

        messages.append({"role": "assistant", "content": data["final_text"]})
        history.append((obs_text, data["final_text"]))

        action = parse_action(data["final_text"])
        obs, _, terminated, truncated, _ = env.step(ACTIONS[action])
        done = terminated or truncated

    client.delete(f"/compact_session/{session_id}")
    env.close()

    n_turns = len(turn_asst_ids)

    # Reconstruct full completion token sequence (everything after initial prompt):
    # turn0_asst | user1 | turn1_asst | user2 | turn2_asst | ...
    # Note: turn_asst_ids[t][0] is the boundary token for t>0 (already prefilled
    # as new_token_ids[0] by the caller, so it IS a generated token with a logprob).
    completion_ids = []
    for t in range(n_turns):
        if t > 0:
            completion_ids.extend(turn_user_ids[t])
        completion_ids.extend(turn_asst_ids[t])

    # Segment boundaries: cumulative completion token counts at each compaction point.
    # Turn-based compaction fires after turn n_max_turns-1 (0-indexed), at the END
    # of that turn's generation. The boundary is after all tokens through that turn.
    # We compute boundaries from the actual compaction_events' "after_turn" field.
    fired_after_turns = sorted({e["after_turn"] for e in compaction_events if e.get("after_turn") is not None})

    # Build segment boundaries in completion-token space
    segment_boundaries = []
    cumulative = 0
    for t in range(n_turns):
        if t > 0:
            cumulative += len(turn_user_ids[t])
        cumulative += len(turn_asst_ids[t])
        if t in fired_after_turns:
            segment_boundaries.append(cumulative)
    if not segment_boundaries or segment_boundaries[-1] != len(completion_ids):
        segment_boundaries.append(len(completion_ids))

    # compact_window for trainer: tokens from prompt_len to compact_end.
    # Turn-based: compact_end = kv_len - protected_kv_len
    # where protected_kv_len = sum of last n_preserved_turns user+asst lens.
    # We approximate this from the actual kv_len_before and kv_len_after in the event.
    # For segmented_forward, compact_window = compacted_region_len = kv_len_before
    # - prompt_len - protected_kv_len. Use the first compaction event.
    compact_window = None
    if compaction_events:
        ev = compaction_events[0]
        kv_before = ev["kv_len_before"]
        kv_after = ev["kv_len_after"]
        # In markovian: kv_after = prompt_len + preserved_kv_len (c1=0)
        # In attention_matching: kv_after = prompt_len + compacted_len + preserved_kv_len
        # compact_window = compacted_region = kv_before - prompt_len - preserved_kv_len
        # = kv_before - kv_after  (in markovian, since compacted_len=0)
        # Approximate: use kv_before - prompt_len - (kv_after - prompt_len) = kv_before - kv_after
        # This equals "tokens removed + compacted", which is the window for the trainer.
        compact_window = kv_before - len(prompt_ids)

    return {
        "env": env_name,
        "seed": seed,
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "all_infer_logprobs": all_infer_logprobs,
        "segment_boundaries": segment_boundaries,
        "prompt_len": len(prompt_ids),
        "n_turns": n_turns,
        "n_compactions": len(compaction_events),
        "compact_window": compact_window,
        "turn_asst_lens": [len(t) for t in turn_asst_ids],
        "turn_user_lens": [len(t) for t in turn_user_ids],
    }


def compute_trainer_logprobs(model, rollout, device, temperature, mode):
    """Run segmented_forward and return per-completion-token logprobs."""
    prompt_ids = rollout["prompt_ids"]
    completion_ids = rollout["completion_ids"]
    segment_boundaries = rollout["segment_boundaries"]
    prompt_len = rollout["prompt_len"]
    compact_window = rollout["compact_window"]

    if not completion_ids:
        return []

    full_ids = prompt_ids + completion_ids
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

    logits = out["logits"]  # [1, seq_len, vocab], already temp-scaled
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # For each completion token c_k = completion_ids[k], the predicting logit
    # is at position (prompt_len + k - 1) in the full sequence.
    trainer_lps = []
    for k, tok_id in enumerate(completion_ids):
        pred_pos = prompt_len + k - 1
        trainer_lps.append(log_probs[0, pred_pos, tok_id].item())

    return trainer_lps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="markovian", choices=["markovian", "attention_matching"])
    parser.add_argument("--n", type=int, default=3, help="Episodes per task")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-kv-len", type=int, default=2048)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--compact-ratio", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n-max-turns", type=int, default=4)
    parser.add_argument("--n-preserved-turns", type=int, default=2)
    parser.add_argument("--env-steps", type=int, default=6,
                        help="Env steps per episode (need > n_max_turns for compaction to affect output)")
    parser.add_argument("--steps", type=int, default=1, help="Rollout rounds")
    args = parser.parse_args()

    n_total = args.n * len(BABYAI_TASKS)
    print(f"Session mismatch eval: mode={args.mode}, steps={args.steps}, n={args.n}, "
          f"tasks={len(BABYAI_TASKS)}, total={n_total}/step")
    print(f"n_max_turns={args.n_max_turns}, n_preserved_turns={args.n_preserved_turns}, "
          f"env_steps={args.env_steps}, max_kv_len={args.max_kv_len}")

    httpx.get(f"http://localhost:{args.port}/health", timeout=10.0).raise_for_status()
    print("Server health: OK\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model {args.model} on {args.device}...")
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    model.eval()
    model.config.use_cache = True

    all_step_means = []
    global_kls = []

    for step in range(1, args.steps + 1):
        print(f"\n{'='*60}")
        print(f"STEP {step}/{args.steps}")
        print(f"{'='*60}")

        client = httpx.Client(base_url=f"http://localhost:{args.port}", timeout=3600.0)
        seed_offset = (step - 1) * args.n
        rollouts = []

        for env_name in BABYAI_TASKS:
            for seed in range(seed_offset, seed_offset + args.n):
                rollout = collect_episode(client, tokenizer, env_name, seed, args)
                rollouts.append(rollout)
                print(
                    f"  {env_name}[{seed}] turns={rollout['n_turns']} "
                    f"compactions={rollout['n_compactions']} "
                    f"completion_tokens={len(rollout['completion_ids'])} "
                    f"segs={len(rollout['segment_boundaries'])}"
                )

        client.close()

        n_with = sum(1 for r in rollouts if r["n_compactions"] > 0)
        print(f"Rollouts with >=1 compaction: {n_with}/{n_total}")

        step_kls_no = []
        step_kls_yes = []

        print(f"\nComputing trainer mismatch (mode={args.mode})...")
        for rollout in rollouts:
            if not rollout["completion_ids"]:
                continue

            trainer_lps = compute_trainer_logprobs(
                model, rollout, args.device, args.temperature, args.mode,
            )
            infer_lps = rollout["all_infer_logprobs"]

            # Align lengths (boundary token offset can cause off-by-one)
            n = min(len(trainer_lps), len(infer_lps))
            kls = [math.exp(t - i) - (t - i) - 1 for t, i in zip(trainer_lps[:n], infer_lps[:n])]

            if rollout["n_compactions"] > 0:
                step_kls_yes.extend(kls)
            else:
                step_kls_no.extend(kls)

            mean_kl = sum(kls) / len(kls) if kls else 0.0
            print(
                f"  {rollout['env']}[{rollout['seed']}]: "
                f"turns={rollout['n_turns']} compactions={rollout['n_compactions']} "
                f"tokens={n} mean_kl={mean_kl:.6f}"
            )

        step_kls = step_kls_no + step_kls_yes
        step_mean = sum(step_kls) / len(step_kls) if step_kls else float("nan")
        m_no = sum(step_kls_no) / len(step_kls_no) if step_kls_no else float("nan")
        m_yes = sum(step_kls_yes) / len(step_kls_yes) if step_kls_yes else float("nan")
        all_step_means.append((step_mean, m_no, m_yes, len(step_kls_no), len(step_kls_yes)))
        global_kls.extend(step_kls)

        print(f"\nStep {step}: mean_kl={step_mean:.6f}  "
              f"no_compaction={m_no:.6f} ({len(step_kls_no)} tok)  "
              f"with_compaction={m_yes:.6f} ({len(step_kls_yes)} tok)")

    print("\n" + "=" * 60)
    print(f"AGGREGATE — mode={args.mode}, n_max_turns={args.n_max_turns}, "
          f"n_preserved={args.n_preserved_turns}")
    print("mismatch_kl = exp(Δlogp) - Δlogp - 1")
    print("=" * 60)
    print(f"  {'Step':<5}  {'overall':>10}  {'no compact':>12}  {'with compact':>14}  "
          f"{'no_tok':>7}  {'yes_tok':>7}")
    for s, (m, m_no, m_yes, n_no, n_yes) in enumerate(all_step_means, 1):
        print(f"  {s:<5}  {m:>10.6f}  {m_no:>12.6f}  {m_yes:>14.6f}  {n_no:>7}  {n_yes:>7}")

    if global_kls:
        print(f"\n  Overall mean = {sum(global_kls)/len(global_kls):.6f}  ({len(global_kls)} tokens)")
        print(f"  Overall max  = {max(global_kls):.6f}")


if __name__ == "__main__":
    main()
