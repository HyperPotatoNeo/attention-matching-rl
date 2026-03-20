"""Gradio browser for BabyAI episode traces.

Loads a traces JSON produced by eval_balrog_babyai.py --save-traces
and lets you navigate episodes and turns interactively.

Usage:
    uv run python scripts/browse_babyai_traces.py results/babyai_baseline_traces.json
    uv run python scripts/browse_babyai_traces.py results/babyai_baseline_traces.json \
        results/babyai_compaction_traces.json
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

import gradio as gr
from transformers import AutoTokenizer

TOKEN_COLORS = ["#dbeafe", "#fef9c3"]  # light blue / light yellow


def load_traces(paths: list[str]) -> tuple[list[dict], str]:
    episodes = []
    model_name = "Qwen/Qwen3-4B"
    for path in paths:
        data = json.loads(Path(path).read_text())
        model_name = data.get("config", {}).get("model", model_name)
        mode = data.get("mode", Path(path).stem)
        for r in data.get("results", []):
            if "trace" not in r:
                continue
            episodes.append({
                "label": f"[{mode}] ep{r['idx']} {r['env']} {'✓' if r['success'] else '✗'}",
                "mode": mode,
                "env": r["env"],
                "success": r["success"],
                "reward": r["reward"],
                "turns": r["turns"],
                "tokens": r["tokens"],
                "compactions": r.get("compactions", 0),
                "trace": r["trace"],
            })
    return episodes, model_name


def render_episode_header(ep: dict) -> str:
    status = "✓ SUCCESS" if ep["success"] else "✗ FAIL"
    return (
        f"**Mode:** {ep['mode']}  |  **Env:** {ep['env']}  |  **{status}**  |  "
        f"**Reward:** {ep['reward']:.3f}  |  **Turns:** {ep['turns']}  |  "
        f"**Tokens:** {ep['tokens']}  |  **Compactions:** {ep['compactions']}"
    )


def parse_response(text: str) -> tuple[str, str]:
    """Split model response into (reasoning, action)."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""

    action_match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    action = action_match.group(1).strip() if action_match else ""

    if not reasoning and not action_match:
        # No tags at all — show raw text
        reasoning = text.strip()

    return reasoning, action


def tokens_to_html(token_ids: list[int], tokenizer) -> str:
    if not token_ids:
        return "<em style='color:#999'>No token IDs saved for this turn</em>"
    spans = []
    for i, tid in enumerate(token_ids):
        text = tokenizer.decode([tid], skip_special_tokens=False)
        color = TOKEN_COLORS[i % 2]
        escaped = html.escape(text).replace(" ", "&nbsp;").replace("\n", "↵<br>")
        spans.append(
            f'<span title="id={tid}" style="background:{color};border-radius:3px;'
            f'padding:1px 4px;margin:1px;display:inline;font-family:monospace;font-size:13px">'
            f"{escaped}</span>"
        )
    n = len(token_ids)
    header = f"<div style='margin-bottom:6px;font-size:12px;color:#555'><b>{n} tokens</b></div>"
    body = "<div style='line-height:2.2;word-wrap:break-word'>" + "".join(spans) + "</div>"
    return header + body


def render_turn(ep: dict, t: int, tokenizer) -> tuple[str, str, str, str, str]:
    """Return (obs_before, reasoning, action_label, obs_after, token_html) for turn t."""
    step = ep["trace"][t]
    reasoning, _ = parse_response(step["response"])

    executed = step.get("action", "")
    fallback = step.get("action_fallback", False)
    action_label = executed
    if fallback:
        action_label += "  ⚠️ FALLBACK (model didn't output <action> tag)"

    obs_after = step.get("obs_after") or ""
    tok_html = tokens_to_html(step.get("token_ids") or [], tokenizer)
    return step["obs"], reasoning, action_label, obs_after, tok_html


def build_app(episodes: list[dict], tokenizer) -> gr.Blocks:
    episode_labels = [ep["label"] for ep in episodes]

    with gr.Blocks(title="BabyAI Trace Browser") as app:
        gr.Markdown("# BabyAI Trace Browser")

        ep_idx = gr.State(value=0)

        with gr.Row():
            ep_dropdown = gr.Dropdown(
                choices=episode_labels,
                value=episode_labels[0] if episode_labels else None,
                label="Episode",
                scale=4,
            )

        ep_header = gr.Markdown(value="")

        with gr.Row():
            turn_slider = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="Turn")

        with gr.Row():
            obs_box = gr.Textbox(label="Observation (before action)", lines=18, max_lines=25, scale=1)
            with gr.Column(scale=1):
                reasoning_box = gr.Textbox(label="Reasoning  (<think>…</think>)", lines=6, max_lines=12)
                action_box = gr.Textbox(label="Executed action", lines=2, max_lines=3)
                obs_after_box = gr.Textbox(label="Observation (after action)", lines=8, max_lines=14)

        token_box = gr.HTML(label="Tokens (hover for ID)")

        # ── event handlers ──────────────────────────────────────────────────

        OUTPUTS = [ep_idx, ep_header, turn_slider, obs_box, reasoning_box, action_box, obs_after_box, token_box]

        def on_episode_change(label):
            idx = episode_labels.index(label)
            ep = episodes[idx]
            if ep["trace"]:
                obs, reasoning, action, obs_after, toks = render_turn(ep, 0, tokenizer)
            else:
                obs, reasoning, action, obs_after, toks = "", "", "", "", ""
            return idx, render_episode_header(ep), gr.update(minimum=1, maximum=len(ep["trace"]), value=1), obs, reasoning, action, obs_after, toks

        def on_turn_change(turn, ep_index):
            ep = episodes[ep_index]
            t = max(0, min(int(turn) - 1, len(ep["trace"]) - 1))
            return render_turn(ep, t, tokenizer)

        ep_dropdown.change(on_episode_change, inputs=[ep_dropdown], outputs=OUTPUTS)
        turn_slider.change(on_turn_change, inputs=[turn_slider, ep_idx],
                           outputs=[obs_box, reasoning_box, action_box, obs_after_box, token_box])

        app.load(
            fn=lambda: on_episode_change(episode_labels[0]) if episode_labels else (),
            outputs=OUTPUTS,
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Browse BabyAI episode traces")
    parser.add_argument("traces", nargs="+", help="Traces JSON file(s) from --save-traces")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    episodes, model_name = load_traces(args.traces)
    if not episodes:
        print("No traces found. Re-run eval with --save-traces.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded {len(episodes)} episodes with traces.")

    app = build_app(episodes, tokenizer)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
