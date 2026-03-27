"""Gradio viewer for BabyAI training rollouts from wandb.

Turn-by-turn navigation with observation/response split view,
action timeline, thinking extraction, and episode-level stats.

Usage:
    uv run python scripts/viz_training_rollouts.py <wandb_run_path>
    uv run python scripts/viz_training_rollouts.py laurent-charlin/balrog-rl/udjw2p6k
    uv run python scripts/viz_training_rollouts.py --local /path/to/final-samples.table.json
"""

import argparse
import html
import json
import re
from pathlib import Path

import gradio as gr

# Chat template tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

ACTION_ICONS = {
    "turn left":  "↺  turn left",
    "turn right": "↻  turn right",
    "go forward": "⬆  go forward",
    "pick up":    "✋  pick up",
    "drop":       "⬇  drop",
    "toggle":     "🔓  toggle",
}

CSS = """
/* Message panels */
.msg-wrap { font-family: monospace; font-size: 13px; line-height: 1.5; }
.msg { margin: 6px 0; border-radius: 6px; padding: 8px 10px; }
.msg-role { font-size: 10px; font-weight: bold; letter-spacing: 1px; opacity: 0.7; margin-bottom: 3px; }
.msg-content { white-space: pre-wrap; word-break: break-word; }
.msg-system { background: #1a1a2e; border-left: 3px solid #4a4a8a; color: #ccc; }
.msg-user { background: #1a2a1a; border-left: 3px solid #27ae60; color: #ccc; }
.msg-assistant { background: #1a2a3a; border-left: 3px solid #2980b9; color: #ccc; }
.msg-tool { background: #2a1a1a; border-left: 3px solid #e74c3c; color: #ccc; }
.think { color: #888; font-style: italic; margin: 4px 0; padding: 4px 8px; border-left: 2px solid #555; }
.action-tag { color: #f39c12; font-weight: bold; font-size: 14px; margin-top: 6px; }
.tool-call { background: #2a2a1a; border: 1px dashed #f39c12; border-radius: 4px; padding: 4px 6px; margin: 4px 0; font-size: 12px; }

/* Action timeline */
.timeline-wrap { padding: 8px 4px; font-family: monospace; }
.timeline-title { font-size: 11px; color: #888; margin-bottom: 6px; }
.timeline-chips { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 8px; }
.timeline-chip {
    display: inline-flex; flex-direction: column; align-items: center;
    min-width: 36px; padding: 3px 6px; border-radius: 4px;
    border: 2px solid transparent; cursor: default; font-size: 13px;
}
.timeline-chip.active { border-color: #ffcc00 !important; }
.chip-num { font-size: 10px; font-weight: bold; opacity: 0.7; }

/* Observation panel */
.obs-wrap { font-family: monospace; font-size: 14px; line-height: 1.6; padding: 10px;
            background: #111; border-radius: 6px; border: 1px solid #333; color: #ccc; }
.obs-item { margin: 2px 0; }
.obs-object { color: #f39c12; font-weight: bold; }
.obs-wall { color: #7f8c8d; }
.obs-direction { color: #2980b9; }
"""


def parse_messages(raw_text: str) -> list[dict]:
    """Parse chat-template formatted text into structured messages."""
    messages = []
    parts = raw_text.split(IM_START)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = part.replace(IM_END, "").strip()
        newline_idx = part.find("\n")
        if newline_idx == -1:
            role = part.strip()
            content = ""
        else:
            role = part[:newline_idx].strip()
            content = part[newline_idx + 1:].strip()
        messages.append({"role": role, "content": content})
    return messages


def extract_action(content: str) -> str | None:
    """Extract action from tool_call in assistant message."""
    match = TOOL_CALL_RE.search(content)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        parsed = json.loads(raw)
        args = parsed.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return args.get("action")
    except (json.JSONDecodeError, AttributeError):
        return None


def extract_thinking(content: str) -> tuple[str, str]:
    """Split content into (thinking, rest)."""
    match = THINK_RE.search(content)
    if not match:
        return "", content
    thinking = match.group(1).strip()
    rest = content[:match.start()] + content[match.end():]
    return thinking, rest.strip()


def extract_tool_response(content: str) -> str | None:
    """Extract observation text from <tool_response> wrapper."""
    match = TOOL_RESPONSE_RE.search(content)
    return match.group(1).strip() if match else None


def messages_to_turns(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert flat message list into structured turns.

    Returns (system_prompt, turns) where each turn has:
        obs: str, response: str, action: str|None, thinking: str
    """
    system_prompt = ""
    turns = []
    i = 0

    # Extract system prompt
    if messages and messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        i = 1

    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "user":
            obs_content = msg["content"]
            # Strip tool_response wrapper if present
            tool_resp = extract_tool_response(obs_content)
            if tool_resp:
                obs_content = tool_resp

            response = ""
            action = None
            thinking = ""

            # Look for the assistant response
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                response = messages[i + 1]["content"]
                action = extract_action(response)
                thinking, _ = extract_thinking(response)
                i += 2
            else:
                i += 1

            turns.append({
                "obs": obs_content,
                "response": response,
                "action": action,
                "thinking": thinking,
            })
        else:
            i += 1

    return system_prompt, turns


def render_observation_html(obs: str) -> str:
    """Render observation text with highlighted objects and directions."""
    lines = obs.strip().splitlines()
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        escaped = html.escape(line)
        # Highlight object names (colored objects)
        for color in ("red", "green", "blue", "purple", "yellow", "grey"):
            for obj in ("key", "ball", "box", "door"):
                pattern = f"{color} {obj}"
                if pattern in escaped:
                    escaped = escaped.replace(
                        pattern,
                        f'<span class="obs-object">{pattern}</span>',
                    )
        # Highlight walls
        if "wall" in escaped:
            escaped = escaped.replace("wall", '<span class="obs-wall">wall</span>')
        # Highlight goal
        if "goal" in escaped.lower():
            escaped = re.sub(
                r"(goal)",
                r'<span class="obs-object">\1</span>',
                escaped,
                flags=re.IGNORECASE,
            )
        # Highlight directions
        for d in ("forward", "left", "right"):
            escaped = escaped.replace(d, f'<span class="obs-direction">{d}</span>')
        items.append(f'<div class="obs-item">{escaped}</div>')
    return f'<div class="obs-wrap">{"".join(items)}</div>'


def render_response_html(response: str) -> str:
    """Render assistant response with thinking and tool call sections."""
    thinking, rest = extract_thinking(response)
    action = extract_action(response)

    parts = ['<div class="msg-wrap">']

    if thinking:
        truncated = thinking[:800] + ("..." if len(thinking) > 800 else "")
        parts.append(f'<div class="think">{html.escape(truncated)}</div>')

    # Remove thinking and tool_call from the visible "reasoning" text
    rest_clean = rest
    rest_clean = TOOL_CALL_RE.sub("", rest_clean).strip()

    if rest_clean:
        parts.append(
            f'<div class="msg msg-assistant">'
            f'<div class="msg-role">REASONING</div>'
            f'<div class="msg-content">{html.escape(rest_clean)}</div>'
            f'</div>'
        )

    if action:
        icon = ACTION_ICONS.get(action, action)
        parts.append(f'<div class="action-tag">{icon}</div>')
    else:
        # Show raw tool_call if action parsing failed
        match = TOOL_CALL_RE.search(response)
        if match:
            parts.append(f'<div class="tool-call">{html.escape(match.group(1).strip())}</div>')

    parts.append('</div>')
    return "".join(parts)


def render_action_timeline(turns: list[dict], current_turn: int) -> str:
    """Render a visual timeline of actions across the episode."""
    chips = []
    for i, t in enumerate(turns):
        action = t.get("action")
        active_cls = " active" if i == current_turn else ""

        if action in ("go forward",):
            bg, fg, icon = "#1a3a1a", "#27ae60", "⬆"
        elif action in ("turn left",):
            bg, fg, icon = "#1a1a3a", "#2980b9", "↺"
        elif action in ("turn right",):
            bg, fg, icon = "#1a1a3a", "#2980b9", "↻"
        elif action in ("pick up",):
            bg, fg, icon = "#3a2a10", "#f39c12", "✋"
        elif action in ("drop",):
            bg, fg, icon = "#3a2a10", "#f39c12", "⬇"
        elif action in ("toggle",):
            bg, fg, icon = "#2a1a3a", "#8e44ad", "🔓"
        else:
            bg, fg, icon = "#2a1a1a", "#e74c3c", "?"

        chips.append(
            f'<div class="timeline-chip{active_cls}" style="background:{bg};color:{fg};">'
            f'{icon}<span class="chip-num">{i + 1}</span></div>'
        )

    return (
        '<div class="timeline-wrap">'
        '<div class="timeline-title">Action timeline — green: forward · blue: turn · orange: interact · purple: toggle · 🟡 = current</div>'
        f'<div class="timeline-chips">{"".join(chips)}</div>'
        '</div>'
    )


def render_system_prompt_html(system_prompt: str) -> str:
    """Render the system prompt (truncated)."""
    if not system_prompt:
        return ""
    truncated = system_prompt[:600] + ("\n... (truncated)" if len(system_prompt) > 600 else "")
    return (
        '<div class="msg-wrap">'
        '<div class="msg msg-system">'
        '<div class="msg-role">SYSTEM</div>'
        f'<div class="msg-content">{html.escape(truncated)}</div>'
        '</div></div>'
    )


def load_samples(run_path: str = None, local_path: str = None) -> list[dict]:
    """Load samples from wandb or local JSON."""
    if local_path:
        with open(local_path) as f:
            data = json.load(f)
    else:
        import wandb
        api = wandb.Api()
        run = api.run(run_path)
        art_name = f"run-{run.id}-final-samples:latest"
        art = api.artifact(f"{run.entity}/{run.project}/{art_name}")
        path = art.download()
        table_file = Path(path) / "final-samples.table.json"
        with open(table_file) as f:
            data = json.load(f)

    columns = data["columns"]
    samples = []
    for row in data["data"]:
        sample = dict(zip(columns, row))
        sample["messages_parsed"] = parse_messages(sample.get("messages", ""))
        system_prompt, turns = messages_to_turns(sample["messages_parsed"])
        sample["system_prompt"] = system_prompt
        sample["turns"] = turns
        samples.append(sample)
    return samples


def build_app(samples: list[dict]) -> gr.Blocks:
    # Build rollout labels
    labels = []
    for i, s in enumerate(samples):
        reward = s.get("reward", 0)
        step = s.get("step", "?")
        n_turns = len(s["turns"])
        actions = [t["action"] for t in s["turns"] if t["action"]]
        labels.append(
            f"[step {step}] #{s.get('example_id', i)} | "
            f"reward={reward:.1f} | turns={n_turns} | "
            f"actions={len(actions)}"
        )

    with gr.Blocks(title="BabyAI Training Rollout Viewer") as demo:
        gr.Markdown(
            "# BabyAI Training Rollout Viewer\n"
            "> **Observation** (left): what the agent sees each turn. "
            "**Response** (right): model reasoning + action. "
            "**Timeline** (bottom): full action sequence for the episode."
        )

        with gr.Row():
            rollout_dd = gr.Dropdown(
                choices=labels,
                value=labels[0] if labels else None,
                label="Rollout",
                scale=3,
            )
            turn_sl = gr.Slider(
                minimum=0, maximum=0, step=1, value=0,
                label="Turn", scale=2,
            )

        ep_bar = gr.Markdown("")

        with gr.Row():
            obs_html = gr.HTML(label="Observation")
            resp_html = gr.HTML(label="Model Response")

        timeline_html = gr.HTML(label="Action Timeline")
        system_html = gr.HTML(label="System Prompt")

        with gr.Accordion("Full conversation (all messages)", open=False):
            full_msgs_html = gr.HTML()

        def on_rollout(choice):
            if not choice:
                return gr.update(maximum=0, value=0), "", "", "", "", "", ""
            idx = labels.index(choice)
            s = samples[idx]
            turns = s["turns"]
            n = len(turns)

            actions = [t["action"] for t in turns if t["action"]]
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            action_summary = ", ".join(f"{v}x {k}" for k, v in action_counts.items())

            bar = (
                f"**Step:** {s.get('step', '?')} | "
                f"**Example:** {s.get('example_id', '?')} | "
                f"**Task:** {s.get('task', '?')} | "
                f"**Reward:** {s.get('reward', 0):.2f} | "
                f"**Turns:** {n} | "
                f"**Actions:** {action_summary or 'none'}"
            )

            max_turn = max(0, n - 1)

            # Render turn 0
            obs = ""
            resp = ""
            tl = ""
            if turns:
                obs = render_observation_html(turns[0]["obs"])
                resp = render_response_html(turns[0]["response"])
                tl = render_action_timeline(turns, 0)

            sys_html = render_system_prompt_html(s["system_prompt"])

            # Full conversation
            full = render_all_messages_html(s["messages_parsed"])

            return (
                gr.update(maximum=max_turn, value=0),
                bar, obs, resp, tl, sys_html, full,
            )

        def on_turn(choice, turn):
            if not choice:
                return "", "", ""
            idx = labels.index(choice)
            s = samples[idx]
            turns = s["turns"]
            t = int(turn)
            if t >= len(turns):
                return "", "", ""

            obs = render_observation_html(turns[t]["obs"])
            resp = render_response_html(turns[t]["response"])
            tl = render_action_timeline(turns, t)
            return obs, resp, tl

        rollout_dd.change(
            on_rollout, [rollout_dd],
            [turn_sl, ep_bar, obs_html, resp_html, timeline_html, system_html, full_msgs_html],
        )
        turn_sl.change(
            on_turn, [rollout_dd, turn_sl],
            [obs_html, resp_html, timeline_html],
        )
        demo.load(
            on_rollout, [rollout_dd],
            [turn_sl, ep_bar, obs_html, resp_html, timeline_html, system_html, full_msgs_html],
        )

    return demo


def render_all_messages_html(messages: list[dict]) -> str:
    """Render all messages as styled HTML (full conversation view)."""
    blocks = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        css_class = f"msg-{role}" if role in ("system", "user", "assistant", "tool") else "msg-user"

        escaped = html.escape(content)

        if role == "assistant":
            thinking, rest = extract_thinking(content)
            action = extract_action(content)

            parts = []
            if thinking:
                parts.append(
                    f'<div class="think">{html.escape(thinking[:500])}'
                    f'{"..." if len(thinking) > 500 else ""}</div>'
                )
            rest_escaped = html.escape(rest)
            rest_escaped = re.sub(
                r"&lt;tool_call&gt;(.*?)&lt;/tool_call&gt;",
                r'<div class="tool-call">\1</div>',
                rest_escaped,
                flags=re.DOTALL,
            )
            parts.append(rest_escaped)
            if action:
                parts.append(f'<div class="action-tag">Action: {html.escape(action)}</div>')
            escaped = "\n".join(parts)
        elif role == "system":
            if len(escaped) > 600:
                escaped = escaped[:600] + "\n... (truncated)"

        block = (
            f'<div class="msg {css_class}">'
            f'<div class="msg-role">{role.upper()}</div>'
            f'<div class="msg-content">{escaped}</div>'
            f'</div>'
        )
        blocks.append(block)

    return f'<div class="msg-wrap">{"".join(blocks)}</div>'


def main():
    parser = argparse.ArgumentParser(description="Visualize training rollouts from wandb")
    parser.add_argument("run_path", nargs="?", help="wandb run path (entity/project/run_id)")
    parser.add_argument("--local", help="Path to local .table.json file")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not args.run_path and not args.local:
        # Auto-discover: check output dirs, then artifacts/
        patterns = [
            Path("/network/scratch/e/emiliano.penaloza/outputs").glob("*/run_default/wandb/*/files/media/table/*.table.json"),
            Path("artifacts").glob("run-*-final-samples*/*.table.json"),
        ]
        found = []
        for pat in patterns:
            found.extend(sorted(pat, key=lambda p: p.stat().st_mtime, reverse=True))
        if found:
            args.local = str(found[0])
            print(f"Auto-discovered: {args.local}")
        else:
            print("Usage: viz_training_rollouts.py <wandb_run_path>")
            print("   or: viz_training_rollouts.py --local /path/to/samples.table.json")
            return

    print("Loading samples...")
    samples = load_samples(run_path=args.run_path, local_path=args.local)
    print(f"Loaded {len(samples)} rollout samples")

    demo = build_app(samples)
    demo.launch(server_port=args.port, share=args.share, css=CSS,
                theme=gr.themes.Soft(primary_hue="blue"))


if __name__ == "__main__":
    main()
