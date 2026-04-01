"""Evaluate on TextWorld (BALROG) with and without KV cache compaction.

Multi-turn text adventure: model receives text descriptions, outputs freeform commands.
Compaction compresses the growing conversation context between turns.

Uses BALROG's TextWorld integration (requires `textworld` package + game files).
Evaluates on 3 tasks: treasure_hunter, the_cooking_game, coin_collector.

Usage:
    # Baseline (standard vLLM)
    python scripts/eval_balrog_textworld.py --mode baseline --n 10

    # Compaction (KV budget mode)
    python scripts/eval_balrog_textworld.py --mode compaction --n 10 \
        --max-kv-len 4096 --compact-ratio 0.25

    # Markovian
    python scripts/eval_balrog_textworld.py --mode markovian --n 10 \
        --max-kv-len 4096

    # Summary compaction
    python scripts/eval_balrog_textworld.py --mode summary --n 10 \
        --n-max-turns 6 --n-preserved-turns 3 --summary-max-tokens 300

    # Markovian pure
    python scripts/eval_balrog_textworld.py --mode markovian_pure --n 10 \
        --n-max-turns 6 --n-preserved-turns 3

    # Specific tasks only
    python scripts/eval_balrog_textworld.py --mode baseline --n 10 \
        --envs coin_collector treasure_hunter

Metrics: score (0-100), success rate, avg turns, token usage, timing per task.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import httpx
from transformers import AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

# TextWorld tasks: (task_name, max_steps, difficulty)
TEXTWORLD_TASKS = [
    ("coin_collector", 25, "easy"),
    ("treasure_hunter", 40, "medium"),
    ("the_cooking_game", 80, "hard"),
]

# Per-task instruction prompts (from BALROG)
TASK_INSTRUCTIONS = {
    "treasure_hunter": (
        "You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated "
        "maze and must find a specific object. You need to explore different rooms to find the target object.\n"
        "Available commands: look, goal, inventory, go <dir> (north/east/south/west), open, drop, take, "
        "put ... on ..., take ... from ..., insert ... into ..., unlock ... with ...\n"
        "Tips: The target object might be in a closed or locked container. Keys match locks by adjective "
        "(e.g. non-euclidean keycard matches non-euclidean safe). Take keys whenever possible. "
        "After unlocking, you still need to open.\n"
        "You have 40 steps to complete the task."
    ),
    "the_cooking_game": (
        "You are an agent playing TextWorld, a text-based adventure game where you navigate rooms, "
        "interact with objects, and solve puzzles. Your goal: find the recipe, find and prepare food "
        "according to the recipe, then prepare and eat the meal.\n"
        "Available commands: look, goal, inventory, go <dir>, examine, eat, open, drop, take, "
        "put ... on ..., take ... from ..., insert ... into ..., lock/unlock ... with ..., "
        "cook ... with ..., slice/chop/dice ... with ..., prepare meal\n"
        "Tips: Examine cookbook to see recipe. BBQ=grilling, stove=frying, oven=roasting. "
        "Process food (chop/slice/dice with knife) before cooking. "
        "Ingredients must EXACTLY match recipe colors. "
        "When all ingredients are ready, 'prepare meal' in kitchen then 'eat meal' to win.\n"
        "You have 80 steps to complete the task."
    ),
    "coin_collector": (
        "You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated "
        "maze and must find the coin. Explore different rooms to find it.\n"
        "Available commands: goal, go <dir> (north/east/south/west), take coin\n"
        "The only actions are 'go <dir>' to explore and 'take coin' when you see the coin.\n"
        "You have 25 steps to complete the task."
    ),
}


_tw_lock = threading.Lock()
_tw_factory = None
_GymV21 = None


def _get_gym_compat():
    """Load GymV21CompatibilityV0 directly (wrappers/__init__ pulls NLE)."""
    global _GymV21
    if _GymV21 is not None:
        return _GymV21
    import importlib.util as _ilu
    import balrog.environments
    _compat_path = Path(balrog.environments.__path__[0]) / "wrappers" / "gym_compatibility.py"
    _spec = _ilu.spec_from_file_location("_gym_compat", _compat_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _GymV21 = _mod.GymV21CompatibilityV0
    return _GymV21


def _get_tw_factory():
    """Thread-safe singleton for TextWorldFactory."""
    global _tw_factory
    if _tw_factory is not None:
        return _tw_factory
    with _tw_lock:
        if _tw_factory is not None:
            return _tw_factory

        import balrog.environments.textworld as _tw_mod

        # Reset the BALROG singleton so we control initialization
        _tw_mod.TEXTWORLD_FACTORY = None
        _tw_mod.TextWorldFactory._instance = None

        balrog_dir = os.environ.get("BALROG_DIR", "/tmp/balrog")
        tw_games_path = os.path.join(balrog_dir, "tw_games")

        if not os.path.exists(tw_games_path):
            raise FileNotFoundError(
                f"TextWorld game files not found at {tw_games_path}. "
                "Run the BALROG data setup first (setup_balrog_data in balrog_bench.py) "
                "or set BALROG_DIR to point to the BALROG repo."
            )

        _tw_factory = _tw_mod.global_textworld_context(
            tasks=["treasure_hunter", "the_cooking_game", "coin_collector"],
            objective=True,
            description=True,
            score=True,
            max_score=True,
            won=True,
            max_episode_steps=80,
            textworld_games_path=tw_games_path,
        )
        return _tw_factory


def make_textworld_env(task, seed=None):
    """Create and reset a TextWorld environment via BALROG.

    Both creation and reset are serialized because tatsu (the parser
    textworld uses to load game files) is not thread-safe.
    """
    GymV21CompatibilityV0 = _get_gym_compat()
    factory = _get_tw_factory()
    with _tw_lock:
        env = factory(task, seed=seed)
        env = GymV21CompatibilityV0(env=env, render_mode=None)
        obs, info = env.reset()
    return env, obs, info


def format_observation(obs):
    """Extract text from a TextWorld observation dict."""
    if isinstance(obs, dict) and "text" in obs:
        return obs["text"].get("long_term_context", str(obs["text"]))
    if isinstance(obs, str):
        return obs
    return str(obs)


def build_system_prompt(task, no_thinking=False):
    """Build the system prompt for a TextWorld task."""
    instruction = TASK_INSTRUCTIONS[task]
    prompt = (
        f"{instruction}\n\n"
        "Each turn you receive a text observation of your surroundings.\n"
        "Reason briefly about what to do, then output exactly one command inside <action>...</action> tags.\n"
        "Example: <action>go north</action>"
    )
    if no_thinking:
        prompt = "/nothink\n" + prompt
    return prompt


def parse_action(text):
    """Extract action command from model response."""
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: look for common TextWorld commands in the text
    text_lower = text.lower().strip()
    for pattern in [
        r"\b(go (?:north|south|east|west))\b",
        r"\b(take [\w\s]+)\b",
        r"\b(open [\w\s]+)\b",
        r"\b(unlock [\w\s]+ with [\w\s]+)\b",
        r"\b(examine [\w\s]+)\b",
        r"\b(cook [\w\s]+ with [\w\s]+)\b",
        r"\b(slice|chop|dice) ([\w\s]+ with [\w\s]+)\b",
        r"\b(prepare meal)\b",
        r"\b(eat meal)\b",
        r"\b(look)\b",
        r"\b(inventory)\b",
        r"\b(drop [\w\s]+)\b",
    ]:
        m = re.search(pattern, text_lower)
        if m:
            return m.group(0).strip()

    return "look"


def generate_with_think_closure(client, messages, model_name, max_think_tokens, max_action_tokens=128):
    """Generate response, forcing </think> + action if model hits token limit mid-think."""
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


def build_messages(system_prompt, history, current_obs):
    """Build chat messages from conversation history."""
    messages = [{"role": "system", "content": system_prompt}]
    for obs_text, response_text, *_ in history:
        messages.append({"role": "user", "content": obs_text})
        messages.append({"role": "assistant", "content": response_text})
    messages.append({"role": "user", "content": current_obs})
    return messages


def generate_summary(client, turns, prev_summary, model_name, max_tokens=300):
    """Generate a text summary of conversation turns for context compression."""
    parts = []
    for obs_text, response_text, *_ in turns:
        parts.append(f"Observation:\n{obs_text}\nResponse:\n{response_text}")
    interaction = "\n---\n".join(parts)

    context = f"Previous context:\n{prev_summary}\n\n" if prev_summary else ""

    resp = client.post("/v1/chat/completions", json={
        "model": model_name,
        "messages": [{"role": "user", "content": (
            "/nothink\nBriefly summarize this text adventure interaction. "
            "Focus on: the goal, rooms explored, items found/taken, "
            "doors/containers opened, and current progress toward the objective. "
            "2-3 sentences.\n\n"
            f"{context}Interaction:\n{interaction}"
        )}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    })
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    tokens = data["usage"]["completion_tokens"]
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text, tokens


def build_messages_with_summary(system_prompt, summary, recent_history, current_obs):
    """Build chat messages with a summary of old turns injected into system prompt."""
    system = system_prompt + f"\n\nContext from earlier in this episode:\n{summary}"
    messages = [{"role": "system", "content": system}]
    for obs_text, response_text, *_ in recent_history:
        messages.append({"role": "user", "content": obs_text})
        messages.append({"role": "assistant", "content": response_text})
    messages.append({"role": "user", "content": current_obs})
    return messages


def run_episode(task_name, idx, port, args, tokenizer=None, model_name=None, save_traces=False, run_id=None):  # noqa: C901
    """Run one TextWorld episode. Returns result dict."""
    task_info = next(t for t in TEXTWORLD_TASKS if t[0] == task_name)
    max_steps = task_info[1]

    env, obs, info = make_textworld_env(task_name, seed=idx)
    obs_text = format_observation(obs)

    system_prompt = build_system_prompt(task_name, no_thinking=args.no_thinking)

    history = []
    total_tokens = 0
    n_compactions = 0
    episode_return = 0.0
    t0 = time.time()

    client = httpx.Client(base_url=f"http://localhost:{port}", timeout=600.0)

    terminated = False
    truncated = False
    actual_turns = 0

    # Session API only for modes that need KV-level compaction
    use_session = args.mode not in ("baseline", "summary", "markovian_pure") and (
        args.use_sessions or args.mode in ("compaction", "markovian")
    )
    session_id = f"{run_id}_{task_name}_{idx}" if use_session else None
    session_token_ids = None
    session_turns = 0

    # Client-side turn management state (summary / markovian_pure)
    cumulative_summary = ""
    last_summary_idx = 0
    summary_tokens_used = 0
    n_summary_resets = 0
    window_turns = 0

    effective_max_turns = min(args.max_turns, max_steps)

    for turn in range(effective_max_turns):
        # Client-side turn-dropping for summary / markovian_pure
        if args.mode in ("summary", "markovian_pure") and args.n_max_turns >= 0 and window_turns >= args.n_max_turns:
            n_keep = args.n_preserved_turns
            if args.mode == "summary":
                summary_end = len(history) - n_keep if n_keep > 0 and len(history) > n_keep else len(history)
                turns_to_summarize = history[last_summary_idx:summary_end]
                if turns_to_summarize:
                    cumulative_summary, stokens = generate_summary(
                        client, turns_to_summarize, cumulative_summary,
                        model_name, max_tokens=args.summary_max_tokens,
                    )
                    summary_tokens_used += stokens
                last_summary_idx = summary_end
                n_summary_resets += 1
            history = history[-n_keep:] if n_keep > 0 else []
            last_summary_idx = 0
            window_turns = n_keep

        if args.mode == "summary" and cumulative_summary:
            messages = build_messages_with_summary(
                system_prompt, cumulative_summary, history, obs_text,
            )
        else:
            messages = build_messages(system_prompt, history, obs_text)

        diag = {}
        if args.mode in ("baseline", "summary", "markovian_pure"):
            text, n_tokens = generate_with_think_closure(
                client, messages, model_name, max_think_tokens=args.max_response_tokens,
            )
            total_tokens += n_tokens
            window_turns += 1
        elif session_id is not None:
            if (args.mode == "markovian" and args.n_max_turns >= 0
                    and session_turns >= args.n_max_turns):
                n_keep = args.n_preserved_turns

                client.delete(f"/compact_session/{session_id}")
                session_id = f"{run_id}_{task_name}_{idx}_r{turn}"
                session_token_ids = None
                session_turns = 0

                recent = history[-n_keep:] if n_keep > 0 else []
                messages = build_messages(system_prompt, recent, obs_text)

            if session_token_ids is None:
                result = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                )
                prompt_ids = list(result["input_ids"]) if hasattr(result, "keys") else list(result)
                server_max_kv_len = args.max_kv_len

                resp = client.post("/compact_session/create", json={
                    "session_id": session_id,
                    "prompt_ids": prompt_ids,
                    "max_kv_len": server_max_kv_len,
                    "max_response_tokens": args.max_response_tokens,
                    "compact_target_ratio": args.compact_ratio,
                    "compact_window": args.compact_window,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "compaction_mode": "markovian" if args.mode == "markovian" else "attention_matching",
                    "use_suffix_queries": args.use_suffix_queries,
                    "n_max_turns": args.n_max_turns,
                    "n_preserved_turns": args.n_preserved_turns,
                })
                if resp.status_code != 200:
                    raise RuntimeError(f"compact_session/create HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                session_token_ids = prompt_ids + data["all_token_ids"]
            else:
                result_curr = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                )
                curr_ids = list(result_curr["input_ids"]) if hasattr(result_curr, "keys") else list(result_curr)
                result_prev = tokenizer.apply_chat_template(
                    messages[:-1], add_generation_prompt=False, tokenize=True,
                )
                prev_ids = list(result_prev["input_ids"]) if hasattr(result_prev, "keys") else list(result_prev)
                new_user_turn_tokens = curr_ids[len(prev_ids):]
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

            session_turns += 1
            text = data["final_text"]
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
            total_tokens += len(data["all_token_ids"])
            diag = data.get("diagnostics", {})
            n_compactions += len(diag.get("compaction_events", []))

        action = parse_action(text)
        action_fallback = not bool(re.search(r"<action>", text))

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        actual_turns += 1
        next_obs_text = format_observation(obs)

        history.append((
            obs_text, text,
            action, action_fallback,
            next_obs_text if save_traces else None,
            diag.get("compaction_events", []) if save_traces else [],
        ))
        obs_text = next_obs_text

        if terminated or truncated:
            break

    if session_id is not None:
        client.delete(f"/compact_session/{session_id}")

    env.close()
    elapsed = time.time() - t0

    # TextWorld scoring: continuous score scaled to 0-100
    score = min(max(episode_return * 10.0, 0.0), 100.0)
    success = terminated and episode_return > 0

    result = {
        "idx": idx,
        "env": task_name,
        "success": success,
        "score": round(score, 1),
        "reward": float(episode_return),
        "turns": actual_turns,
        "tokens": total_tokens,
        "compactions": n_compactions,
        "time": round(elapsed, 2),
    }
    if args.mode == "summary":
        result["summary_tokens"] = summary_tokens_used
        result["n_summary_resets"] = n_summary_resets
    if save_traces:
        result["trace"] = [
            {
                "obs": obs,
                "response": resp,
                "action": action,
                "action_fallback": fallback,
                "obs_after": obs_after,
                "compaction_events": compact_evts,
            }
            for obs, resp, action, fallback, obs_after, compact_evts in history
        ]
    return result


def plot_results(results_path):
    """Generate comparison bar chart if multiple result files exist."""
    results_dir = Path(results_path).parent
    result_files = sorted(results_dir.glob("*textworld*.json"))
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

    scores = [summaries[m]["avg_score"] for m in modes]
    axes[0].bar(modes, scores, color=colors)
    axes[0].set_ylabel("Avg Score (0-100)")
    axes[0].set_title("Score")
    axes[0].set_ylim(0, 100)

    avg_tokens = [summaries[m]["avg_tokens"] for m in modes]
    axes[1].bar(modes, avg_tokens, color=colors)
    axes[1].set_ylabel("Avg Tokens / Episode")
    axes[1].set_title("Token Usage")

    avg_turns = [summaries[m]["avg_turns"] for m in modes]
    axes[2].bar(modes, avg_turns, color=colors)
    axes[2].set_ylabel("Avg Turns / Episode")
    axes[2].set_title("Episode Length")

    fig.suptitle("TextWorld: Baseline vs Compaction", fontsize=14)
    fig.tight_layout()

    plot_path = results_dir / "textworld_comparison.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


def _save_results(results, episodes, args, output_path):
    """Save current results to JSON (called incrementally)."""
    n_total = len(results)
    if n_total == 0:
        return
    total_success = sum(int(r["success"]) for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    total_score = sum(r["score"] for r in results)

    per_env = defaultdict(lambda: {"success": 0, "total": 0, "turns": [], "tokens": [], "scores": []})
    for r in results:
        per_env[r["env"]]["success"] += int(r["success"])
        per_env[r["env"]]["total"] += 1
        per_env[r["env"]]["turns"].append(r["turns"])
        per_env[r["env"]]["tokens"].append(r["tokens"])
        per_env[r["env"]]["scores"].append(r["score"])

    summary = {
        "mode": args.mode,
        "n_episodes": len(episodes),
        "n_completed": n_total,
        "n_per_task": args.n,
        "success_rate": total_success / n_total,
        "avg_score": total_score / n_total,
        "successes": total_success,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / n_total,
        "avg_turns": sum(r["turns"] for r in results) / n_total,
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
                "avg_score": sum(t["scores"]) / len(t["scores"]),
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate on TextWorld (BALROG)")
    parser.add_argument("--mode", choices=["baseline", "compaction", "markovian", "summary", "markovian_pure"], required=True)
    parser.add_argument("--n", type=int, default=10, help="Episodes per task")
    parser.add_argument("--envs", nargs="*", default=None,
                        help="TextWorld task names (default: all)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--ports", default="8000,8001,8002,8003")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    # Compaction args
    parser.add_argument("--n-compacts", type=int, default=99)
    parser.add_argument("--compact-ratio", type=float, default=0.25)
    parser.add_argument("--compact-window", type=int, default=None)
    parser.add_argument("--max-kv-len", type=int, default=None)
    parser.add_argument("--use-suffix-queries", action="store_true")
    parser.add_argument("--use-sessions", action="store_true",
                        help="Use persistent KV sessions (avoids re-prefilling full history each turn)")
    parser.add_argument("--n-max-turns", type=int, default=-1,
                        help="Turn-based: trigger compaction when uncompacted turns reach this count (-1 = KV-budget mode)")
    parser.add_argument("--n-preserved-turns", type=int, default=0,
                        help="Turn-based: keep last N turns verbatim after compaction fires")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable Qwen3 thinking mode (faster, shorter responses)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plot (needs >=2 result files)")
    parser.add_argument("--save-traces", action="store_true",
                        help="Save full turn-by-turn conversation traces in the output JSON")
    parser.add_argument("--summary-max-tokens", type=int, default=300,
                        help="Max tokens for summary generation (summary mode only)")
    args = parser.parse_args()

    if args.mode in ("summary", "markovian_pure") and args.n_max_turns < 0:
        args.n_max_turns = 6
    if args.mode in ("summary", "markovian_pure") and args.n_preserved_turns == 0:
        args.n_preserved_turns = 3

    ports = [int(p) for p in args.ports.split(",")]

    # Select tasks
    if args.envs:
        tasks = [(name, steps, diff) for name, steps, diff in TEXTWORLD_TASKS if name in args.envs]
    else:
        tasks = TEXTWORLD_TASKS

    if not tasks:
        avail = [name for name, _, _ in TEXTWORLD_TASKS]
        print(f"No valid envs. Available: {avail}")
        sys.exit(1)

    # Build episode list: n episodes per task
    episodes = []
    for i in range(args.n):
        for task_name, max_steps, difficulty in tasks:
            episodes.append({
                "idx": i,
                "env": task_name,
                "max_steps": max_steps,
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

    # Resume: load existing results and skip completed episodes
    output_path = args.output or f"results_textworld_{args.mode}_{len(episodes)}.json"
    results = []
    completed = set()
    if Path(output_path).exists():
        prev = json.loads(Path(output_path).read_text())
        results = prev.get("results", [])
        for r in results:
            completed.add((r["env"], r["idx"]))
        print(f"Resuming: {len(completed)}/{len(episodes)} episodes already done\n")

    remaining = [ep for ep in episodes if (ep["env"], ep["idx"]) not in completed]
    if not remaining:
        print("All episodes already completed. Nothing to do.")
    else:
        total_success = sum(int(r["success"]) for r in results)
        total_tokens = sum(r["tokens"] for r in results)
        t_total = time.time()

        max_workers = len(ports) * 2
        run_id = f"{args.mode}_{int(time.time())}"

        results_lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for ep_i, ep in enumerate(remaining):
                port = ports[ep_i % len(ports)]
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
                    f"score={result['score']:5.1f} "
                    f"turns={result['turns']:3d} "
                    f"tokens={result['tokens']:5d} "
                    f"time={result['time']:.1f}s "
                    f"rate={total_success}/{len(results)} "
                    f"({total_success/len(results):.1%})"
                )

                with results_lock:
                    _save_results(results, episodes, args, output_path)

    # Final save
    results.sort(key=lambda r: (r["env"], r["idx"]))
    _save_results(results, episodes, args, output_path)

    n_total = len(results)
    total_success = sum(int(r["success"]) for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    total_score = sum(r["score"] for r in results)

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print(f"RESULTS: {args.mode.upper()} (TextWorld, {n_total}/{len(episodes)} episodes)")
    print("=" * 70)

    print(f"\nSuccess rate: {total_success}/{n_total} ({total_success/n_total:.1%})")
    print(f"Avg score: {total_score/n_total:.1f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Avg tokens/episode: {total_tokens/n_total:.0f}")
    avg_turns = sum(r["turns"] for r in results) / n_total
    print(f"Avg turns/episode: {avg_turns:.1f}")

    per_env = defaultdict(lambda: {"success": 0, "total": 0, "turns": [], "tokens": [], "scores": []})
    for r in results:
        per_env[r["env"]]["success"] += int(r["success"])
        per_env[r["env"]]["total"] += 1
        per_env[r["env"]]["turns"].append(r["turns"])
        per_env[r["env"]]["tokens"].append(r["tokens"])
        per_env[r["env"]]["scores"].append(r["score"])

    print(f"\n{'Task':<25} {'Success':>8} {'Total':>6} {'Rate':>8} {'AvgScore':>9} {'AvgTurns':>9} {'AvgTok':>8}")
    print("-" * 75)
    for task_name in sorted(per_env.keys()):
        t = per_env[task_name]
        rate = t["success"] / t["total"]
        avg_s = sum(t["scores"]) / len(t["scores"])
        avg_t = sum(t["turns"]) / len(t["turns"])
        avg_tok = sum(t["tokens"]) / len(t["tokens"])
        print(f"{task_name:<25} {t['success']:>8d} {t['total']:>6d} {rate:>7.1%} {avg_s:>9.1f} {avg_t:>9.1f} {avg_tok:>8.0f}")

    diff_map = {name: diff for name, _, diff in TEXTWORLD_TASKS}
    per_diff = defaultdict(lambda: {"success": 0, "total": 0, "scores": []})
    for r in results:
        d = diff_map.get(r["env"], "unknown")
        per_diff[d]["success"] += int(r["success"])
        per_diff[d]["total"] += 1
        per_diff[d]["scores"].append(r["score"])

    print(f"\n{'Difficulty':<15} {'Success':>8} {'Total':>6} {'Rate':>8} {'AvgScore':>9}")
    print("-" * 50)
    for diff in ["easy", "medium", "hard"]:
        if diff in per_diff:
            d = per_diff[diff]
            avg_s = sum(d["scores"]) / len(d["scores"])
            print(f"{diff:<15} {d['success']:>8d} {d['total']:>6d} {d['success']/d['total']:>7.1%} {avg_s:>9.1f}")

    if args.mode != "baseline":
        total_compactions = sum(r["compactions"] for r in results)
        print(f"\nTotal compactions: {total_compactions}")
        print(f"Avg compactions/episode: {total_compactions/n_total:.1f}")

    if args.mode == "summary":
        total_summary_tokens = sum(r.get("summary_tokens", 0) for r in results)
        total_resets = sum(r.get("n_summary_resets", 0) for r in results)
        print(f"\nSummary resets: {total_resets}")
        print(f"Summary tokens: {total_summary_tokens}")
        print(f"Avg summary tokens/episode: {total_summary_tokens/n_total:.0f}")

    print(f"\nResults saved to {output_path}")

    if args.plot:
        plot_results(output_path)


if __name__ == "__main__":
    main()
