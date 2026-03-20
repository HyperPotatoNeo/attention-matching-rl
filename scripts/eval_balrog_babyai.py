"""Evaluate Qwen3-4B on BabyAI (MiniGrid) with and without KV cache compaction.

Multi-turn grid-world game: model receives text observations, outputs actions.
Compaction compresses the growing conversation context between turns.

Uses minigrid directly (gymnasium BabyAI environments), not the BALROG wrapper
(which requires cmake + NLE). Evaluates on a representative set of BabyAI tasks
ranging from easy (GoToObj) to hard (Unlock).

Usage:
    # Baseline (standard vLLM)
    python scripts/eval_balrog_babyai.py --mode baseline --n 10

    # Compaction (KV budget mode)
    python scripts/eval_balrog_babyai.py --mode compaction --n 10 \
        --max-kv-len 2048 --compact-ratio 0.25

    # Markovian
    python scripts/eval_balrog_babyai.py --mode markovian --n 10 \
        --max-kv-len 2048

    # Specific tasks only
    python scripts/eval_balrog_babyai.py --mode baseline --n 10 \
        --envs GoToObj GoToLocal

Metrics: success rate, avg turns, token usage, timing per task.
"""

import argparse
import concurrent.futures
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import httpx
import minigrid
from transformers import AutoTokenizer

minigrid.register_minigrid_envs()

sys.stdout.reconfigure(line_buffering=True)

# BabyAI tasks: (env_id_suffix, difficulty)
BABYAI_TASKS = [
    ("GoToObj", "easy"),
    ("GoToLocal", "easy"),
    ("PickupLoc", "easy"),
    ("Open", "medium"),
    ("PutNextLocal", "medium"),
    ("GoTo", "medium"),
    ("Unlock", "hard"),
    ("UnlockLocal", "hard"),
]

ACTIONS = {
    "turn left": 0,
    "turn right": 1,
    "go forward": 2,
    "pick up": 3,
    "drop": 4,
    "toggle": 5,
}
ACTION_NAMES = list(ACTIONS.keys())

OBJECTS = {
    0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door",
    5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava",
}
COLORS = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}
DOOR_STATES = {0: "open", 1: "closed", 2: "locked"}

SYSTEM_PROMPT = """You are navigating a grid-world environment (BabyAI).

Available actions: {actions}

Each turn you receive a text observation showing your mission and what you can see.
Reason briefly about what to do, then output exactly one action inside <action>...</action> tags.
Example: <action>go forward</action>""".format(actions=", ".join(ACTION_NAMES))

SYSTEM_PROMPT_NOTHINK = "/nothink\n" + SYSTEM_PROMPT


def capture_global_state(env):
    """Capture full global grid state for visualization."""
    uw = env.unwrapped
    return {
        "grid": uw.grid.encode().tolist(),
        "agent_pos": [int(uw.agent_pos[0]), int(uw.agent_pos[1])],
        "agent_dir": int(uw.agent_dir),
    }


def render_observation(obs):
    """Convert BabyAI observation dict to a text description.

    The 7x7 partial view is agent-centric: agent at row 6 col 3,
    facing toward row 0. We render it as ASCII with a legend.
    """
    image = obs["image"]  # (7, 7, 3): object_idx, color_idx, state
    mission = obs["mission"]

    grid_lines = []
    legend_items = set()

    for row in range(7):
        cells = []
        for col in range(7):
            obj_idx, color_idx, state = image[col][row]  # image[x,y]: x=lateral, y=depth → image[col, row]
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

    lines = [
        f"Mission: {mission}",
        "",
        "View (you are @ at bottom center, facing up):",
        *grid_lines,
    ]

    if legend_items:
        lines.append(f"Legend: # = wall, @ = you, {', '.join(sorted(legend_items))}")

    return "\n".join(lines)


def parse_action(text):
    """Extract action name from model response."""
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip().lower()
        for name in ACTION_NAMES:
            if candidate == name:
                return name
        for name in ACTION_NAMES:
            if name in candidate:
                return name

    text_lower = text.lower()
    for name in ACTION_NAMES:
        if name in text_lower:
            return name

    return "go forward"


def generate_with_think_closure(client, messages, model_name, max_think_tokens, max_action_tokens=128):
    """Generate response, forcing </think> + action if the model hits the token limit mid-think.

    Two-pass: first generate up to max_think_tokens. If the think block is unclosed,
    append </think> and continue with a second call (continue_final_message) capped at
    max_action_tokens to get the <action> tag.
    """
    resp = client.post("/v1/chat/completions", json={
        "model": model_name,
        "messages": messages,
        "max_tokens": max_think_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
    })
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    total_tokens = data["usage"]["completion_tokens"]

    if "</think>" in text:
        return text, total_tokens

    # Think block was cut off — force-close it and generate the action
    forced = text.rstrip() + "\n</think>\n\n"
    resp2 = client.post("/v1/chat/completions", json={
        "model": model_name,
        "messages": messages + [{"role": "assistant", "content": forced}],
        "max_tokens": max_action_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "continue_final_message": True,
        "add_generation_prompt": False,
    })
    resp2.raise_for_status()
    data2 = resp2.json()
    total_tokens += data2["usage"]["completion_tokens"]
    return forced + data2["choices"][0]["message"]["content"], total_tokens


def build_messages(history, current_obs, no_thinking=False):
    """Build chat messages from conversation history."""
    system = SYSTEM_PROMPT_NOTHINK if no_thinking else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}]
    for obs_text, response_text, *_rest in history:
        messages.append({"role": "user", "content": obs_text})
        messages.append({"role": "assistant", "content": response_text})
    messages.append({"role": "user", "content": current_obs})
    return messages


def run_episode(env_name, idx, port, args, tokenizer=None, model_name=None, save_traces=False, run_id=None):  # noqa: C901
    """Run one BabyAI episode. Returns result dict."""
    env_id = f"BabyAI-{env_name}-v0"
    env = gym.make(env_id, render_mode=None)
    obs, info = env.reset(seed=idx)
    obs_text = render_observation(obs)

    history = []
    total_tokens = 0
    n_compactions = 0
    t0 = time.time()

    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)

    terminated = False
    truncated = False

    session_id = f"{run_id}_{idx}" if (args.mode != "baseline" and args.use_sessions) else None
    session_token_ids = None  # actual KV token IDs (not re-encoded)

    for turn in range(args.max_turns):
        messages = build_messages(history, obs_text, no_thinking=args.no_thinking)

        diag = {}
        if args.mode == "baseline":
            text, n_tokens = generate_with_think_closure(
                client, messages, model_name, max_think_tokens=args.max_response_tokens,
            )
            total_tokens += n_tokens
            turn_token_ids = tokenizer.encode(text, add_special_tokens=False) if (save_traces and tokenizer) else None
        elif session_id is not None:
            if turn == 0:
                result = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                )
                prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)
                resp = client.post("/compact_session/create", json={
                    "session_id": session_id,
                    "prompt_ids": prompt_ids,
                    "max_kv_len": args.max_kv_len,
                    "max_response_tokens": args.max_response_tokens,
                    "compact_target_ratio": args.compact_ratio,
                    "compact_window": args.compact_window,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "compaction_mode": "markovian" if args.mode == "markovian" else "attention_matching",
                    "use_suffix_queries": args.use_suffix_queries,
                    "n_protect_turns": args.n_protect_turns,
                })
                if resp.status_code != 200:
                    raise RuntimeError(f"compact_session/create HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                session_token_ids = prompt_ids + data["all_token_ids"]
            else:
                # Compute new user turn tokens as the diff between current and prev template.
                # This avoids re-encoding the response (which may have fewer tokens after
                # skip_special_tokens strips EOS/think tokens from decoded text).
                result_curr = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                )
                curr_ids = list(result_curr["input_ids"]) if hasattr(result_curr, "keys") else list(result_curr)
                result_prev = tokenizer.apply_chat_template(
                    messages[:-1], add_generation_prompt=False, tokenize=True,
                )
                prev_ids = list(result_prev["input_ids"]) if hasattr(result_prev, "keys") else list(result_prev)
                new_user_turn_tokens = curr_ids[len(prev_ids):]
                # boundary token = last actual KV token (not yet written in KV cache)
                new_token_ids = [session_token_ids[-1]] + new_user_turn_tokens
                resp = client.post("/compact_session/step", json={
                    "session_id": session_id,
                    "new_token_ids": new_token_ids,
                    "max_response_tokens": args.max_response_tokens,
                })
                if resp.status_code != 200:
                    raise RuntimeError(f"compact_session/step HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                session_token_ids = session_token_ids + new_user_turn_tokens + data["all_token_ids"]

            text = data["final_text"]
            turn_token_ids = data["all_token_ids"] if save_traces else None
            total_tokens += len(data["all_token_ids"])
            diag = data.get("diagnostics", {})
            n_compactions += len(diag.get("compaction_events", []))
        else:
            result = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
            )
            prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)

            if args.max_kv_len and len(prompt_ids) > args.max_kv_len:
                prompt_ids = prompt_ids[-args.max_kv_len:]

            body = {
                "prompt_ids": prompt_ids,
                "max_seq_len": len(prompt_ids) + args.max_response_tokens,
                "max_tokens_per_segment": args.max_response_tokens,
                "max_total_tokens": args.max_response_tokens,
                "n_compacts": args.n_compacts,
                "compact_target_ratio": args.compact_ratio,
                "temperature": 0.6,
                "top_p": 0.95,
            }
            if args.max_kv_len is not None:
                body["max_kv_len"] = args.max_kv_len
            if args.compact_window is not None:
                body["compact_window"] = args.compact_window
            if args.use_suffix_queries:
                body["use_suffix_queries"] = True
            if args.mode == "markovian":
                body["compaction_mode"] = "markovian"

            resp = client.post("/compact_generate", json=body)
            if resp.status_code != 200:
                raise RuntimeError(f"compact_generate HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            text = data["final_text"]
            turn_token_ids = data["all_token_ids"] if save_traces else None
            total_tokens += len(data["all_token_ids"])
            diag = data.get("diagnostics", {})
            n_compactions += len(diag.get("compaction_events", []))

        action_name = parse_action(text)
        action_fallback = not bool(re.search(r"<action>", text))
        action_idx = ACTIONS[action_name]
        compaction_events = diag.get("compaction_events", [])
        gs_before = capture_global_state(env) if save_traces else None
        obs, reward, terminated, truncated, info = env.step(action_idx)
        next_obs_text = render_observation(obs)
        gs_after = capture_global_state(env) if save_traces else None

        history.append((
            obs_text, text, turn_token_ids if save_traces else None,
            action_name, action_fallback, next_obs_text if save_traces else None,
            gs_before, gs_after,
            compaction_events if save_traces else [],
        ))
        obs_text = next_obs_text

        if terminated or truncated:
            break

    if session_id is not None:
        client.delete(f"/compact_session/{session_id}")

    env.close()
    elapsed = time.time() - t0

    # BabyAI: success = terminated with positive reward (reached goal)
    success = terminated and reward > 0

    result = {
        "idx": idx,
        "env": env_name,
        "success": success,
        "reward": float(reward),
        "turns": len(history),
        "tokens": total_tokens,
        "compactions": n_compactions,
        "time": round(elapsed, 2),
    }
    if save_traces:
        result["trace"] = [
            {
                "obs": obs,
                "response": resp,
                "token_ids": tids,
                "action": action,
                "action_fallback": fallback,
                "obs_after": obs_after,
                "global_state": gs,
                "global_state_after": gs_after,
                "compaction_events": compact_evts,
            }
            for obs, resp, tids, action, fallback, obs_after, gs, gs_after, compact_evts in history
        ]
    return result


def plot_results(results_path):
    """Generate comparison bar chart if multiple result files exist."""
    results_dir = Path(results_path).parent
    result_files = sorted(results_dir.glob("*babyai*.json"))
    if len(result_files) < 2:
        print("Need >=2 result files to plot comparison. Skipping.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = {}
    for f in result_files:
        data = json.loads(f.read_text())
        summaries[data["mode"]] = data

    modes = list(summaries.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868"][:len(modes)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    success_rates = [summaries[m]["success_rate"] * 100 for m in modes]
    axes[0].bar(modes, success_rates, color=colors)
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 100)

    avg_tokens = [summaries[m]["avg_tokens"] for m in modes]
    axes[1].bar(modes, avg_tokens, color=colors)
    axes[1].set_ylabel("Avg Tokens / Episode")
    axes[1].set_title("Token Usage")

    avg_turns = [summaries[m]["avg_turns"] for m in modes]
    axes[2].bar(modes, avg_turns, color=colors)
    axes[2].set_ylabel("Avg Turns / Episode")
    axes[2].set_title("Episode Length")

    fig.suptitle("BabyAI: Baseline vs Compaction", fontsize=14)
    fig.tight_layout()

    plot_path = results_dir / "babyai_comparison.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on BabyAI (MiniGrid)")
    parser.add_argument("--mode", choices=["baseline", "compaction", "markovian"], required=True)
    parser.add_argument("--n", type=int, default=10, help="Episodes per task")
    parser.add_argument("--envs", nargs="*", default=None,
                        help="BabyAI env suffixes (default: all)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--ports", default="8000,8001,8002,8003")
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    # Compaction args
    parser.add_argument("--n-compacts", type=int, default=99)
    parser.add_argument("--compact-ratio", type=float, default=0.25)
    parser.add_argument("--compact-window", type=int, default=None)
    parser.add_argument("--max-kv-len", type=int, default=None)
    parser.add_argument("--use-suffix-queries", action="store_true")
    parser.add_argument("--use-sessions", action="store_true",
                        help="Use persistent KV sessions (avoids re-prefilling full history each turn)")
    parser.add_argument("--n-protect-turns", type=int, default=-1,
                        help="Turn-based compaction: keep last N turns uncompacted (-1 = use KV-budget mode)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable Qwen3 thinking mode (faster, shorter responses)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plot (needs >=2 result files)")
    parser.add_argument("--save-traces", action="store_true",
                        help="Save full turn-by-turn conversation traces in the output JSON")
    args = parser.parse_args()

    ports = [int(p) for p in args.ports.split(",")]

    # Select tasks
    if args.envs:
        tasks = [(name, diff) for name, diff in BABYAI_TASKS if name in args.envs]
    else:
        tasks = BABYAI_TASKS

    if not tasks:
        avail = [name for name, _ in BABYAI_TASKS]
        print(f"No valid envs. Available: {avail}")
        sys.exit(1)

    # Build episode list: n episodes per task
    episodes = []
    for i in range(args.n):
        for env_name, difficulty in tasks:
            episodes.append({
                "idx": len(episodes),
                "env": env_name,
                "difficulty": difficulty,
            })

    tokenizer = None
    if args.mode != "baseline" or args.save_traces:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Mode: {args.mode}  |  thinking={'off' if args.no_thinking else 'on'}")
    print(f"Tasks ({len(tasks)}): {[t[0] for t in tasks]}")
    print(f"Episodes per task: {args.n}  |  Total: {len(episodes)}")
    print(f"Max turns: {args.max_turns}  |  Max response tokens: {args.max_response_tokens}")
    if args.mode != "baseline":
        print(f"  compact_ratio={args.compact_ratio}, max_kv_len={args.max_kv_len}")
        print(f"  suffix_queries={args.use_suffix_queries}")
    print(f"Ports: {ports}\n")

    # Health check
    health = httpx.get(f"http://localhost:{ports[0]}/health", timeout=10.0)
    health.raise_for_status()
    print("Server health: OK\n")

    results = []
    total_success = 0
    total_tokens = 0
    t_total = time.time()

    max_workers = len(ports) * 2
    run_id = f"{args.mode}_{int(time.time())}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for ep in episodes:
            port = ports[ep["idx"] % len(ports)]
            f = ex.submit(
                run_episode,
                ep["env"], ep["idx"], port, args, tokenizer, args.model,
                save_traces=args.save_traces, run_id=run_id,
            )
            futures[f] = ep

        for f in concurrent.futures.as_completed(futures):
            ep = futures[f]
            result = f.result()
            total_success += int(result["success"])
            total_tokens += result["tokens"]
            results.append(result)

            status = "OK" if result["success"] else "FAIL"
            print(
                f"[{len(results):3d}/{len(episodes)}] {status} "
                f"env={result['env']:20s} "
                f"turns={result['turns']:3d} "
                f"tokens={result['tokens']:5d} "
                f"time={result['time']:.1f}s "
                f"rate={total_success}/{len(results)} "
                f"({total_success/len(results):.1%})"
            )

    wall_time = time.time() - t_total
    results.sort(key=lambda r: r["idx"])
    n_total = len(episodes)

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print(f"RESULTS: {args.mode.upper()} (BabyAI, {n_total} episodes)")
    print("=" * 70)

    print(f"\nSuccess rate: {total_success}/{n_total} ({total_success/n_total:.1%})")
    print(f"Total tokens: {total_tokens}")
    print(f"Avg tokens/episode: {total_tokens/n_total:.0f}")
    avg_turns = sum(r["turns"] for r in results) / n_total
    print(f"Avg turns/episode: {avg_turns:.1f}")
    print(f"Wall time: {wall_time:.1f}s ({wall_time/n_total:.1f}s/episode)")

    # Per-env stats
    per_env = defaultdict(lambda: {"success": 0, "total": 0, "turns": [], "tokens": []})
    for r in results:
        per_env[r["env"]]["success"] += int(r["success"])
        per_env[r["env"]]["total"] += 1
        per_env[r["env"]]["turns"].append(r["turns"])
        per_env[r["env"]]["tokens"].append(r["tokens"])

    print(f"\n{'Env':<25} {'Success':>8} {'Total':>6} {'Rate':>8} {'AvgTurns':>9} {'AvgTok':>8}")
    print("-" * 70)
    for env_name in sorted(per_env.keys()):
        t = per_env[env_name]
        rate = t["success"] / t["total"]
        avg_t = sum(t["turns"]) / len(t["turns"])
        avg_tok = sum(t["tokens"]) / len(t["tokens"])
        print(f"{env_name:<25} {t['success']:>8d} {t['total']:>6d} {rate:>7.1%} {avg_t:>9.1f} {avg_tok:>8.0f}")

    # Per-difficulty stats
    diff_map = dict(BABYAI_TASKS)
    per_diff = defaultdict(lambda: {"success": 0, "total": 0})
    for r in results:
        d = diff_map.get(r["env"], "unknown")
        per_diff[d]["success"] += int(r["success"])
        per_diff[d]["total"] += 1

    print(f"\n{'Difficulty':<15} {'Success':>8} {'Total':>6} {'Rate':>8}")
    print("-" * 40)
    for diff in ["easy", "medium", "hard"]:
        if diff in per_diff:
            d = per_diff[diff]
            print(f"{diff:<15} {d['success']:>8d} {d['total']:>6d} {d['success']/d['total']:>7.1%}")

    if args.mode != "baseline":
        total_compactions = sum(r["compactions"] for r in results)
        print(f"\nTotal compactions: {total_compactions}")
        print(f"Avg compactions/episode: {total_compactions/n_total:.1f}")

    # Save JSON
    output_path = args.output or f"results_babyai_{args.mode}_{n_total}.json"
    summary = {
        "mode": args.mode,
        "n_episodes": n_total,
        "n_per_task": args.n,
        "success_rate": total_success / n_total,
        "successes": total_success,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / n_total,
        "avg_turns": avg_turns,
        "wall_time": round(wall_time, 2),
        "config": {
            "model": args.model,
            "max_turns": args.max_turns,
            "max_response_tokens": args.max_response_tokens,
            "compact_ratio": args.compact_ratio,
            "max_kv_len": args.max_kv_len,
            "n_compacts": args.n_compacts,
            "temperature": 0.6,
        },
        "per_env": {
            env_name: {
                "success_rate": t["success"] / t["total"],
                "successes": t["success"],
                "total": t["total"],
                "avg_turns": sum(t["turns"]) / len(t["turns"]),
                "avg_tokens": sum(t["tokens"]) / len(t["tokens"]),
            }
            for env_name, t in per_env.items()
        },
        "results": results,
    }
    Path(output_path).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    if args.plot:
        plot_results(output_path)


if __name__ == "__main__":
    main()
