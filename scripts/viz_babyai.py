"""Interactive BabyAI trace viewer in Gradio.

Shows the full global map (when available) with the agent's trajectory,
plus the egocentric 7×7 partial view alongside the model's response.

Usage:
    uv run python scripts/viz_babyai.py
    uv run python scripts/viz_babyai.py results/babyai_baseline_traces.json
"""

import argparse
import json
import re
from pathlib import Path

import gradio as gr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MiniGrid object/color indices (matches minigrid.core.constants)
IDX_TO_OBJ = {
    0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door",
    5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava",
}
IDX_TO_COLOR = {
    0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey",
}
COLOR_HEX = {
    "red": "#e74c3c", "green": "#27ae60", "blue": "#2980b9",
    "purple": "#8e44ad", "yellow": "#f39c12", "grey": "#7f8c8d",
}
DIR_ARROW = {0: "→", 1: "↓", 2: "←", 3: "↑"}  # 0=east,1=south,2=west,3=north

INPUT_HTML_CSS = """
<style>
.inp-wrap { font-family: monospace; font-size: 12px; line-height: 1.45; }
.inp-msg { margin: 5px 0; border-radius: 4px; padding: 6px 8px; }
.inp-role { font-size: 10px; font-weight: bold; letter-spacing: 1px; opacity: 0.7; margin-bottom: 2px; }
.inp-content { white-space: pre-wrap; word-break: break-word; }
.inp-system { background: #1a1a2e; border-left: 3px solid #4a4a8a; }
.inp-user   { background: #1a2a1a; border-left: 3px solid #27ae60; }
.inp-asst   { background: #1a2a3a; border-left: 3px solid #2980b9; }
.inp-prior  { opacity: 0.38; }
.inp-banner {
    font-size: 10px; text-align: center; padding: 3px 6px;
    margin: 4px 0; border-radius: 3px; border: 1px dashed;
}
</style>
"""

KV_TIMELINE_CSS = """
<style>
.kvt-wrap { padding: 8px 4px; font-family: monospace; }
.kvt-title { font-size: 11px; color: #888; margin-bottom: 6px; }
.kvt-turns { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 8px; }
.kvt-chip {
    display: inline-flex; flex-direction: column; align-items: center;
    min-width: 32px; padding: 3px 5px; border-radius: 4px;
    border: 2px solid transparent; cursor: default;
}
.kvt-chip.active { border-color: #ffcc00 !important; }
.kvt-num  { font-size: 11px; font-weight: bold; }
.kvt-cnt  { font-size: 9px; opacity: 0.8; }
.kvt-evts { font-size: 12px; padding: 4px 0; }
.kvt-evt  { color: #f39c12; margin: 1px 0; }
</style>
"""

OBJECT_ICONS = {
    "key": "🔑", "ball": "●", "box": "📦",
    "door": "🚪", "goal": "★", "lava": "🔥",
}

ACTION_ICONS = {
    "turn left":  "↺  turn left",
    "turn right": "↻  turn right",
    "go forward": "⬆  go forward",
    "pick up":    "✋  pick up",
    "drop":       "⬇  drop",
    "toggle":     "🔓  toggle",
}

GLOBAL_CSS = """
<style>
.map-wrap { display: inline-block; font-family: monospace; }
.map-caption { font-size: 11px; color: #888; margin: 2px 0 6px 0; }
.map-table { border-collapse: collapse; }
.map-table td {
    width: 36px; height: 36px;
    text-align: center; vertical-align: middle;
    font-size: 15px; font-weight: bold;
    border: 1px solid #222;
}
.m-empty  { background: #1a1a2e; color: #333; }
.m-wall   { background: #3d3d3d; color: #666; }
.m-unseen { background: #0d0d17; color: #111; }
.m-agent  { background: #1565c0; color: #fff; font-size: 18px; border: 2px solid #42a5f5 !important; }
.m-trail  { background: #1a3a5c; color: #4a9fd5; }
.m-obj    { background: #1a1a2e; }
.m-goal   { background: #1a4020; color: #27ae60; }
</style>
"""

EGO_CSS = """
<style>
.ego-wrap { display: inline-block; }
.ego-caption { font-size: 11px; color: #888; margin: 2px 0 4px 0; font-family: monospace; }
.ego-mission { font-size: 13px; font-weight: bold; margin: 0 0 6px 0; max-width: 320px; word-wrap: break-word; }
.ego-table { border-collapse: collapse; }
.ego-table td {
    width: 42px; height: 42px;
    text-align: center; vertical-align: middle;
    font-size: 18px; font-weight: bold;
    border: 1px solid #333;
}
.e-empty  { background: #12121f; color: #2a2a4a; }
.e-wall   { background: #3d3d3d; color: #666; font-size: 14px; }
.e-agent  { background: #1565c0; color: #fff; font-size: 22px; border: 2px solid #42a5f5 !important; }
.e-obj    { background: #12121f; }
.e-diff   { outline: 3px solid #ffcc00; outline-offset: -3px; }
</style>
"""


# ---------------------------------------------------------------------------
# Global map rendering
# ---------------------------------------------------------------------------

def render_global_map(global_state, trail: list[tuple] | None = None, caption: str = "") -> str:
    """Render the full global grid as HTML.

    global_state: {"grid": [[[obj,color,state],...]], "agent_pos": [x,y], "agent_dir": int}
    trail: list of (x,y) positions to highlight as visited path (earlier turns)
    """
    if not global_state:
        return "<em>No global map (re-run eval with --save-traces)</em>"

    grid = global_state["grid"]   # grid[x][y] = [obj_idx, color_idx, state]
    ax, ay = global_state["agent_pos"]
    adir = global_state["agent_dir"]
    trail_set = set(trail or [])

    W = len(grid)
    H = len(grid[0]) if W > 0 else 0

    rows_html = ""
    for y in range(H):
        cells_html = ""
        for x in range(W):
            obj_idx, color_idx, _ = grid[x][y]
            obj = IDX_TO_OBJ.get(obj_idx, "?")
            color = IDX_TO_COLOR.get(color_idx, "grey")

            if (x, y) == (ax, ay):
                arrow = DIR_ARROW.get(adir, "?")
                cells_html += f'<td class="m-agent">{arrow}</td>'
            elif (x, y) in trail_set:
                cells_html += f'<td class="m-trail">·</td>'
            elif obj == "wall":
                cells_html += '<td class="m-wall">▪</td>'
            elif obj in ("empty", "floor"):
                cells_html += '<td class="m-empty"> </td>'
            elif obj == "unseen":
                cells_html += '<td class="m-unseen"> </td>'
            elif obj == "goal":
                cells_html += '<td class="m-goal">★</td>'
            else:
                hex_color = COLOR_HEX.get(color, "#aaa")
                icon = OBJECT_ICONS.get(obj, obj[0])
                cells_html += (
                    f'<td class="m-obj" style="color:{hex_color};">{icon}</td>'
                )
        rows_html += f"<tr>{cells_html}</tr>"

    cap_html = f'<p class="map-caption">{caption}</p>' if caption else ""
    return (
        GLOBAL_CSS
        + '<div class="map-wrap">'
        + cap_html
        + f'<table class="map-table">{rows_html}</table>'
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Egocentric view rendering (fallback / secondary panel)
# ---------------------------------------------------------------------------

def _parse_ego_grid(obs_text: str) -> tuple[str, list[list[str]]]:
    lines = obs_text.strip().splitlines()
    mission, grid, in_grid = "", [], False
    for line in lines:
        if line.startswith("Mission:"):
            mission = line[len("Mission:"):].strip()
        elif "View (" in line:
            in_grid = True
        elif line.startswith("Legend:"):
            in_grid = False
        elif in_grid and line.strip():
            row = [line[i:i+2] for i in range(0, len(line), 2)]
            grid.append(row)
    return mission, grid


def _ego_cell_html(raw: str, changed: bool = False) -> str:
    s = raw.strip()
    diff_cls = " e-diff" if changed else ""
    if not s or s == ".":
        return f'<td class="e-empty{diff_cls}">·</td>'
    if s == "#":
        return f'<td class="e-wall{diff_cls}">▪</td>'
    if s == "@":
        return f'<td class="e-agent{diff_cls}">▲</td>'
    if len(s) == 2:
        color_ch, obj_ch = s[0], s[1]
        color_map = {"r": "#e74c3c", "g": "#27ae60", "b": "#2980b9",
                     "p": "#8e44ad", "y": "#f39c12", "e": "#7f8c8d"}
        obj_icons = {"k": "🔑", "K": "🔑", "b": "●", "B": "●", "x": "📦",
                     "X": "📦", "d": "🚪", "D": "🚪", "g": "★", "G": "★",
                     "l": "🔥", "L": "🔥"}
        color = color_map.get(color_ch, "#aaa")
        icon = obj_icons.get(obj_ch, obj_ch)
        return f'<td class="e-obj{diff_cls}" style="color:{color};">{icon}</td>'
    return f'<td class="e-empty{diff_cls}">{s}</td>'


def render_ego_grid(obs_text: str, caption: str = "", diff_obs: str | None = None) -> str:
    if not obs_text:
        return "<em>No observation</em>"
    mission, grid = _parse_ego_grid(obs_text)
    _, diff = _parse_ego_grid(diff_obs) if diff_obs else (None, None)

    rows_html = ""
    for r, row in enumerate(grid):
        cells_html = ""
        for c, cell in enumerate(row):
            changed = (
                diff is not None
                and r < len(diff) and c < len(diff[r])
                and cell.strip() != diff[r][c].strip()
            )
            cells_html += _ego_cell_html(cell, changed)
        rows_html += f"<tr>{cells_html}</tr>"

    cap_html = f'<p class="ego-caption">{caption}</p>' if caption else ""
    legend = (
        '<p class="ego-caption" style="margin-top:4px;">'
        '<span style="background:#1565c0;color:#fff;padding:1px 5px;border-radius:3px;">▲ = you</span>'
        '&nbsp;&nbsp;'
        '<span style="outline:2px solid #ffcc00;outline-offset:-2px;padding:1px 5px;">🟡 = changed</span>'
        '</p>'
    )
    return (
        EGO_CSS
        + '<div class="ego-wrap">'
        + cap_html
        + f'<p class="ego-mission">🎯 {mission}</p>'
        + f'<table class="ego-table">{rows_html}</table>'
        + legend
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def format_action_md(action: str, fallback: bool, compaction_events: list) -> str:
    action_icon = ACTION_ICONS.get(action, action)
    fallback_note = "  *(fallback — no `<action>` tag)*" if fallback else ""
    compact_badge = (
        f"  🟡 **`<compacted>`** ×{len(compaction_events)}" if compaction_events else ""
    )
    return f"### {action_icon}{fallback_note}{compact_badge}"


def render_kv_timeline(trace: list, current_turn: int) -> str:
    """Episode-level KV compaction timeline.

    Each turn is a chip coloured by compaction count.
    The current turn is highlighted; below the chips, per-event details are shown.
    """
    chips = []
    for i, t in enumerate(trace):
        evts = t.get("compaction_events", [])
        n = len(evts)
        if n == 0:
            bg, fg = "#1a3a5c", "#4a9fd5"
        elif n == 1:
            bg, fg = "#3a2a10", "#f39c12"
        else:
            bg, fg = "#4a1a10", "#e74c3c"
        active_cls = " active" if i == current_turn else ""
        cnt = f'<span class="kvt-cnt">×{n}</span>' if n > 0 else '<span class="kvt-cnt">·</span>'
        chips.append(
            f'<div class="kvt-chip{active_cls}" style="background:{bg};color:{fg};">'
            f'<span class="kvt-num">{i + 1}</span>{cnt}</div>'
        )

    evts = trace[current_turn].get("compaction_events", []) if current_turn < len(trace) else []
    if evts:
        items = "".join(
            f'<div class="kvt-evt">compact {j + 1}: {e["kv_len_before"]} → {e["kv_len_after"]} '
            f'(−{e["kv_len_before"] - e["kv_len_after"]})</div>'
            for j, e in enumerate(evts)
        )
        evts_html = (
            f'<div class="kvt-evts"><strong style="color:#ccc;">Turn {current_turn + 1} compactions:</strong>'
            f'{items}</div>'
        )
    else:
        evts_html = '<div class="kvt-evts" style="color:#555;font-size:12px;">No compactions this turn</div>'

    return (
        KV_TIMELINE_CSS
        + '<div class="kvt-wrap">'
        + '<div class="kvt-title">KV timeline — blue: no compact · orange: 1 compact · red: multiple · 🟡 = current turn</div>'
        + f'<div class="kvt-turns">{"".join(chips)}</div>'
        + evts_html
        + "</div>"
    )


def format_messages_html(msgs: list, current_turn: int, trace: list) -> str:
    """Render conversation messages as HTML.

    Prior turns (in KV) are dimmed. A banner at the boundary shows the KV
    summary (number of compactions so far). Current user turn is full opacity.
    """
    n_compact = sum(len(t.get("compaction_events", [])) for t in trace[: current_turn + 1])
    n_msgs = len(msgs)

    parts = [INPUT_HTML_CSS, '<div class="inp-wrap">']
    for i, m in enumerate(msgs):
        is_current_input = i == n_msgs - 1
        role = m["role"]
        cls = {"system": "inp-system", "user": "inp-user", "assistant": "inp-asst"}.get(role, "inp-user")
        prior_cls = "" if is_current_input else " inp-prior"
        content = (
            m["content"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        if is_current_input and current_turn > 0:
            if n_compact > 0:
                color = "#f39c12"
                note = f"{current_turn} prior turn(s) · {n_compact} compaction{'s' if n_compact != 1 else ''}"
            else:
                color = "#555"
                note = f"{current_turn} prior turn(s) · no compactions"
            parts.append(
                f'<div class="inp-banner" style="border-color:{color};color:{color};">'
                f"↑ KV cache ({note})"
                f"</div>"
            )

        parts.append(
            f'<div class="inp-msg {cls}{prior_cls}">'
            f'<div class="inp-role">{role.upper()}</div>'
            f'<div class="inp-content">{content}</div>'
            f"</div>"
        )

    parts.append("</div>")
    return "".join(parts)


def extract_think(response: str) -> str:
    # Closed tag
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Truncated (no closing tag)
    m = re.search(r"<think>(.*)", response, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n\n[… truncated]"
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trace_file(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    episodes = [r for r in data["results"] if r and "trace" in r and r["trace"]]
    return {"meta": data, "episodes": episodes}


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(trace_files: list[str]) -> gr.Blocks:
    loaded = {Path(f).name: load_trace_file(f) for f in trace_files}

    with gr.Blocks(title="BabyAI Trace Viewer") as demo:
        gr.Markdown(
            "# BabyAI Trace Viewer\n"
            "> **Global map** (left): full room — agent arrow shows facing direction, "
            "blue trail = visited cells. "
            "**Egocentric view** (right): what the model sees — agent `▲` always at bottom-center."
        )

        with gr.Row():
            file_dd = gr.Dropdown(
                choices=list(loaded.keys()),
                value=list(loaded.keys())[0] if loaded else None,
                label="File", scale=2,
            )
            ep_dd = gr.Dropdown(choices=[], label="Episode", scale=3)
            turn_sl = gr.Slider(minimum=0, maximum=0, step=1, value=0,
                                label="Turn", scale=2)

        ep_bar = gr.Markdown("")

        with gr.Row():
            map_before = gr.HTML(label="Global map — before action")
            map_after  = gr.HTML(label="Global map — after action")

        with gr.Row():
            ego_before = gr.HTML(label="Partial view — before")
            ego_after  = gr.HTML(label="Partial view — after (🟡 = changed)")

        action_md = gr.Markdown("")
        kv_html = gr.HTML(label="KV Compaction Timeline")

        with gr.Row():
            full_input_box = gr.HTML(label="📨 Full input to model (all messages this turn)")
            full_output_box = gr.Textbox(
                label="📤 Full model output (raw response)",
                lines=20,
                max_lines=40,
                interactive=False,
            )

        # ── callbacks ──────────────────────────────────────────────────────

        SYSTEM_PROMPT = (
            "You are navigating a grid-world environment (BabyAI).\n\n"
            "Available actions: turn left, turn right, go forward, pick up, drop, toggle\n\n"
            "Each turn you receive a text observation showing your mission and what you can see.\n"
            "Reason briefly about what to do, then output exactly one action inside "
            "<action>...</action> tags.\nExample: <action>go forward</action>"
        )

        def reconstruct_messages(trace, turn_idx):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for t in trace[:turn_idx]:
                msgs.append({"role": "user", "content": t["obs"]})
                msgs.append({"role": "assistant", "content": t["response"]})
            msgs.append({"role": "user", "content": trace[turn_idx]["obs"]})
            return msgs

        def on_file(fname):
            if not fname or fname not in loaded:
                return gr.update(choices=[], value=None), gr.update(maximum=0, value=0)
            eps = loaded[fname]["episodes"]
            choices = [
                f"ep{r['idx']} | {r['env']} | {'✓' if r['success'] else '✗'} | {r['turns']}t"
                for r in eps
            ]
            return gr.update(choices=choices, value=choices[0] if choices else None), \
                   gr.update(maximum=0, value=0)

        def on_ep(fname, ep_label):
            if not ep_label or not fname:
                return gr.update(maximum=0, value=0), ""
            eps = loaded[fname]["episodes"]
            idx = next((i for i, r in enumerate(eps)
                        if ep_label.startswith(f"ep{r['idx']} |")), 0)
            ep = eps[idx]
            bar = (
                f"**Env:** {ep['env']} | "
                f"**Result:** {'✅ Success' if ep['success'] else '❌ Failed'} | "
                f"**Turns:** {ep['turns']} | "
                f"**Tokens:** {ep['tokens']} | "
                f"**Compactions:** {ep.get('compactions', 0)}"
            )
            return gr.update(maximum=len(ep["trace"]) - 1, value=0), bar

        def on_turn(fname, ep_label, turn):
            if not ep_label or not fname:
                return "", "", "", "", "", "", "", ""
            eps = loaded[fname]["episodes"]
            idx = next((i for i, r in enumerate(eps)
                        if ep_label.startswith(f"ep{r['idx']} |")), 0)
            ep = eps[idx]
            trace = ep["trace"]
            t = trace[int(turn)]
            n = len(trace)

            trail = []
            for prev_t in trace[:int(turn)]:
                gs = prev_t.get("global_state")
                if gs:
                    trail.append(tuple(gs["agent_pos"]))

            gs = t.get("global_state")
            gs_after = t.get("global_state_after")

            map_b = render_global_map(
                gs, trail=trail,
                caption=f"Turn {int(turn)+1}/{n} — before action",
            )
            map_a = render_global_map(
                gs_after, trail=trail + ([tuple(gs["agent_pos"])] if gs else []),
                caption="After action (blue trail = visited)",
            )

            ego_b = render_ego_grid(
                t["obs"],
                caption=f"Turn {int(turn)+1}/{n} — agent sees this",
            )
            ego_a = render_ego_grid(
                t["obs_after"],
                caption="After action (🟡 = changed cells)",
                diff_obs=t["obs"],
            ) if t.get("obs_after") else "<em>Final turn</em>"

            compact_evts = t.get("compaction_events", [])
            action_line = format_action_md(t["action"], t["action_fallback"], compact_evts)

            msgs = reconstruct_messages(trace, int(turn))
            full_input = format_messages_html(msgs, int(turn), trace)
            kv_timeline = render_kv_timeline(trace, int(turn))

            return map_b, map_a, ego_b, ego_a, action_line, kv_timeline, full_input, t["response"]

        outputs = [map_before, map_after, ego_before, ego_after,
                   action_md, kv_html, full_input_box, full_output_box]
        file_dd.change(on_file, [file_dd], [ep_dd, turn_sl])
        ep_dd.change(on_ep, [file_dd, ep_dd], [turn_sl, ep_bar])
        turn_sl.change(on_turn, [file_dd, ep_dd, turn_sl], outputs)
        demo.load(lambda f: on_file(f), [file_dd], [ep_dd, turn_sl])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", default=[])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    paths = args.files or []
    if not paths:
        paths = sorted(Path("results").glob("babyai*traces*.json"))
        paths += sorted(Path(".").glob("results_babyai*traces*.json"))
        paths = [str(p) for p in paths]

    valid = [p for p in paths if Path(p).exists()]
    if not valid:
        print("No trace files found. Run eval with --save-traces first.")
        return

    print(f"Loading: {[Path(p).name for p in valid]}")
    demo = build_app(valid)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
