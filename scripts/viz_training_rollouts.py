"""Interactive training rollout viewer in Gradio.

Turn-by-turn navigation with observation/response split view,
action timeline, thinking extraction, conversation history with
dimmed prior turns, and episode-level stats.

Usage:
    uv run python scripts/viz_training_rollouts.py <wandb_run_path>
    uv run python scripts/viz_training_rollouts.py laurent-charlin/balrog-rl/udjw2p6k
    uv run python scripts/viz_training_rollouts.py --local /path/to/final-samples.table.json
    uv run python scripts/viz_training_rollouts.py --local file1.json --local file2.json
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
THINK_OPEN_RE = re.compile(r"<think>(.*)", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

ACTION_ICONS = {
    "turn left": "↺  turn left",
    "turn right": "↻  turn right",
    "go forward": "⬆  go forward",
    "pick up": "✋  pick up",
    "drop": "⬇  drop",
    "toggle": "🔓  toggle",
}

CSS = """
<style>
/* Message history panel */
.msg-wrap { font-family: monospace; font-size: 12px; line-height: 1.45; }
.msg { margin: 5px 0; border-radius: 4px; padding: 6px 8px; color: #1a1a1a; }
.msg-role { font-size: 10px; font-weight: bold; letter-spacing: 1px; opacity: 0.6; margin-bottom: 2px; color: #333; }
.msg-content { white-space: pre-wrap; word-break: break-word; color: #1a1a1a; }
.msg-system { background: #e8e8f0; border-left: 3px solid #6a6aaa; }
.msg-user { background: #e6f4e6; border-left: 3px solid #1e8a1e; }
.msg-assistant { background: #e6eef6; border-left: 3px solid #1a6aaa; }
.msg-tool { background: #f4e6e6; border-left: 3px solid #c0392b; }
.msg-prior { opacity: 0.35; }
.msg-banner {
    font-size: 10px; text-align: center; padding: 3px 6px;
    margin: 4px 0; border-radius: 3px; border: 1px dashed #888; color: #555;
}
.think { color: #555; font-style: italic; margin: 4px 0; padding: 4px 8px; border-left: 2px solid #aaa; background: #f5f5f5; }
.action-tag { color: #c07000; font-weight: bold; font-size: 14px; margin-top: 6px; }
.tool-call { background: #fdf6e3; border: 1px dashed #c07000; border-radius: 4px; padding: 4px 6px; margin: 4px 0; font-size: 12px; color: #333; }

/* Action timeline */
.timeline-wrap { padding: 8px 4px; font-family: monospace; }
.timeline-title { font-size: 11px; color: #666; margin-bottom: 6px; }
.timeline-chips { display: flex; flex-wrap: wrap; gap: 3px; margin-bottom: 8px; }
.timeline-chip {
    display: inline-flex; flex-direction: column; align-items: center;
    min-width: 36px; padding: 3px 6px; border-radius: 4px;
    border: 2px solid transparent; cursor: default; font-size: 13px;
}
.timeline-chip.active { border-color: #b08800 !important; }
.chip-num { font-size: 10px; font-weight: bold; opacity: 0.7; }

/* Observation panel */
.obs-wrap { font-family: monospace; font-size: 14px; line-height: 1.6; padding: 10px;
            background: #f8f8f8; border-radius: 6px; border: 1px solid #ddd; color: #1a1a1a; }
.obs-item { margin: 2px 0; color: #1a1a1a; }
.obs-object { color: #b06000; font-weight: bold; }
.obs-wall { color: #666; }
.obs-direction { color: #1a5a9a; font-weight: bold; }
.obs-mission { color: #1a7a1a; font-weight: bold; font-size: 15px; margin-bottom: 6px; }
.obs-carry { color: #7a3a9a; font-weight: bold; }

/* Response detail panel */
.resp-wrap { font-family: monospace; font-size: 13px; line-height: 1.5; color: #1a1a1a; }

/* Reward badge */
.reward-badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-weight: bold; font-size: 12px; font-family: monospace;
}
.reward-pos { background: #d4edda; color: #155724; border: 1px solid #28a745; }
.reward-zero { background: #fff3cd; color: #856404; border: 1px solid #ffc107; }
.reward-neg { background: #f8d7da; color: #721c24; border: 1px solid #dc3545; }

/* Egocentric grid */
.ego-wrap { display: inline-block; font-family: monospace; }
.ego-caption { font-size: 11px; color: #666; margin: 2px 0 4px 0; }
.ego-table { border-collapse: collapse; }
.ego-table td {
    width: 42px; height: 42px;
    text-align: center; vertical-align: middle;
    font-size: 16px; font-weight: bold;
    border: 1px solid #ccc;
}
.e-empty { background: #f5f5f5; color: #ddd; }
.e-wall  { background: #999; color: #fff; font-size: 14px; }
.e-agent { background: #1565c0; color: #fff; font-size: 20px; border: 2px solid #42a5f5 !important; }
.e-obj   { background: #fefefe; }
.e-door  { background: #fefefe; }
.e-score { background: #e8f5e9; }

/* Compaction banner */
.compact-banner {
    font-size: 12px; text-align: center; padding: 6px 10px;
    margin: 6px 0; border-radius: 4px; font-family: monospace;
}
.compact-dropped { background: #f8d7da; color: #721c24; border: 1px solid #dc3545; }
.compact-kv { background: #fff3cd; color: #856404; border: 1px solid #ffc107; }
</style>
"""

COLOR_HEX_LIGHT = {
    "red": "#c0392b", "green": "#1e8a1e", "blue": "#1a6aaa",
    "purple": "#7a3a9a", "yellow": "#b08800", "grey": "#666",
}
OBJECT_ICONS = {
    "key": "🔑", "ball": "●", "box": "📦", "door": "🚪", "goal": "★", "lava": "🔥",
}


# ---------------------------------------------------------------------------
# Observation → grid parser
# ---------------------------------------------------------------------------

_OBS_PATTERN = re.compile(
    r"^a\s+(?:(\w+)\s+)?"           # optional color
    r"(wall|key|ball|box|door|goal|lava)"  # object
    r"\s+"
    r"(.+)$",                        # position description
    re.IGNORECASE,
)
_DIRECTLY = re.compile(r"directly in front of you", re.IGNORECASE)
_STEPS = re.compile(r"(\d+)\s+steps?\s+(forward|left|right)", re.IGNORECASE)
_NOW_AT = re.compile(r"now at curr", re.IGNORECASE)

GRID_SIZE = 7
AGENT_ROW = GRID_SIZE - 1  # bottom row
AGENT_COL = GRID_SIZE // 2  # center column


def _parse_position(desc: str) -> tuple[int, int] | None:
    """Parse position description into (dx, dy) relative to agent.

    forward = -row (up), left = -col, right = +col.
    Returns (col_offset, row_offset) or None.
    """
    if _DIRECTLY.search(desc):
        return (0, -1)
    if _NOW_AT.search(desc):
        return (0, 0)

    dx, dy = 0, 0
    for m in _STEPS.finditer(desc):
        n = int(m.group(1))
        direction = m.group(2).lower()
        if direction == "forward":
            dy -= n
        elif direction == "left":
            dx -= n
        elif direction == "right":
            dx += n
    if dx == 0 and dy == 0 and not _STEPS.search(desc):
        return None
    return (dx, dy)


def parse_obs_to_grid(obs: str) -> tuple[list[list[tuple | None]], list[str]]:
    """Parse observation text into a 7x7 grid and inventory list.

    Each cell is (object, color) or None.
    """
    grid = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
    inventory = []

    for line in obs.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Inventory
        lower = line.lower()
        if lower.startswith("you carry"):
            inventory.append(line)
            continue
        if lower.startswith("mission:") or lower.startswith("episode"):
            continue

        m = _OBS_PATTERN.match(line)
        if not m:
            continue

        color = (m.group(1) or "grey").lower()
        obj = m.group(2).lower()
        pos = _parse_position(m.group(3))
        if pos is None:
            continue

        dx, dy = pos
        col = AGENT_COL + dx
        row = AGENT_ROW + dy
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            grid[row][col] = (obj, color)

    return grid, inventory


def render_ego_grid_html(obs: str, prev_obs: str | None = None) -> str:
    """Render an egocentric 7x7 grid from observation text."""
    if "(observation dropped" in obs:
        return (
            CSS
            + '<div class="compact-banner compact-dropped">'
            + "Observation dropped (markovian pure — no grid data)"
            + "</div>"
        )

    grid, inventory = parse_obs_to_grid(obs)
    prev_grid, _ = parse_obs_to_grid(prev_obs) if prev_obs else (None, None)

    rows_html = ""
    for r in range(GRID_SIZE):
        cells_html = ""
        for c in range(GRID_SIZE):
            # Agent cell
            if r == AGENT_ROW and c == AGENT_COL:
                cells_html += '<td class="e-agent">▲</td>'
                continue

            cell = grid[r][c]
            changed = (
                prev_grid is not None
                and cell != prev_grid[r][c]
            )
            outline = ' style="outline:2px solid #b08800;outline-offset:-2px;"' if changed else ""

            if cell is None:
                cells_html += f'<td class="e-empty"{outline}>·</td>'
            else:
                obj, color = cell
                hex_color = COLOR_HEX_LIGHT.get(color, "#666")
                icon = OBJECT_ICONS.get(obj, obj[0].upper())
                if obj == "wall":
                    cells_html += f'<td class="e-wall"{outline}>▪</td>'
                else:
                    extra_style = f"outline:2px solid #b08800;outline-offset:-2px;" if changed else ""
                    cells_html += f'<td class="e-obj" style="color:{hex_color};{extra_style}">{icon}</td>'
        rows_html += f"<tr>{cells_html}</tr>"

    inv_html = ""
    if inventory:
        inv_items = " · ".join(html.escape(i) for i in inventory)
        inv_html = f'<p class="ego-caption" style="color:#7a3a9a;margin-top:4px;">🎒 {inv_items}</p>'

    legend = (
        '<p class="ego-caption" style="margin-top:4px;">'
        '<span style="background:#1565c0;color:#fff;padding:1px 5px;border-radius:3px;">▲ = agent</span>'
        '&nbsp;'
        '<span style="background:#999;color:#fff;padding:1px 5px;border-radius:3px;">▪ = wall</span>'
        '&nbsp;'
        '<span style="outline:2px solid #b08800;outline-offset:-1px;padding:1px 5px;">gold = changed</span>'
        '</p>'
    )

    return (
        CSS
        + '<div class="ego-wrap">'
        + '<p class="ego-caption">Egocentric view (agent ▲ at bottom center, facing up)</p>'
        + f'<table class="ego-table">{rows_html}</table>'
        + legend
        + inv_html
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_VALID_ROLES = {"system", "user", "assistant", "tool"}


def parse_messages(raw_text: str) -> list[dict]:
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
        if role not in _VALID_ROLES:
            # Model-generated garbage (hallucinated <|endoftext|>, broken tags, etc.)
            # — append to previous message's content instead of creating a new one.
            if messages:
                messages[-1]["content"] += "\n" + content if content else ""
            continue
        messages.append({"role": role, "content": content})
    return messages


def extract_action(content: str) -> str | None:
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
    match = THINK_RE.search(content)
    if match:
        thinking = match.group(1).strip()
        rest = content[: match.start()] + content[match.end() :]
        return thinking, rest.strip()
    # Truncated (no closing tag)
    match = THINK_OPEN_RE.search(content)
    if match:
        return match.group(1).strip() + "\n\n[… truncated]", ""
    return "", content


def extract_tool_response(content: str) -> str | None:
    match = TOOL_RESPONSE_RE.search(content)
    return match.group(1).strip() if match else None


def messages_to_turns(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert flat message list into structured turns.

    Returns (system_prompt, turns) where each turn has:
        obs, response, action, thinking

    Handles three formats:
    - Normal: user/assistant pairs (each user msg starts a turn)
    - Markovian pure: system + consecutive assistant msgs (observations dropped)
    - Tool-calling: user msgs may contain <tool_response> wrappers
    """
    system_prompt = ""
    turns = []
    i = 0

    if messages and messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        i = 1

    # Detect if we have any user messages (markovian_pure may not)
    has_user_msgs = any(m["role"] == "user" for m in messages[i:])

    compaction_events = []  # list of (turn_index, marker_text)

    if has_user_msgs:
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "system" and "[COMPACTION:" in msg.get("content", ""):
                compaction_events.append((len(turns), msg["content"]))
                i += 1
            elif msg["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "user":
                # Consecutive user messages = compaction dropped assistant response(s).
                # Count how many user messages in a row (each one lost its assistant).
                run_start = i
                while i < len(messages) and messages[i]["role"] == "user" and (
                    i + 1 >= len(messages) or messages[i + 1]["role"] == "user"
                ):
                    i += 1
                dropped = i - run_start
                compaction_events.append(
                    (len(turns), f"[COMPACTION: {dropped} turn(s) — context reset detected]")
                )
                # The last user message in the run is the new observation after compaction
                # — fall through to normal user handling below
                # (don't increment i, the next iteration will handle messages[i])
            elif msg["role"] == "user":
                obs_content = msg["content"]
                tool_resp = extract_tool_response(obs_content)
                if tool_resp:
                    obs_content = tool_resp

                response = ""
                action = None
                thinking = ""

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
    else:
        # No user messages — treat each assistant message as a turn
        # (markovian_pure drops observations on window reset)
        for msg in messages[i:]:
            if msg["role"] != "assistant":
                continue
            content = msg["content"]
            action = extract_action(content)
            thinking, _ = extract_thinking(content)
            turns.append({
                "obs": "(observation dropped — markovian pure)",
                "response": content,
                "action": action,
                "thinking": thinking,
            })

    return system_prompt, turns, compaction_events


# ---------------------------------------------------------------------------
# Rendering — observation
# ---------------------------------------------------------------------------

def render_observation_html(obs: str) -> str:
    lines = obs.strip().splitlines()
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        escaped = html.escape(line)

        # Mission line
        if line.lower().startswith("mission:"):
            items.append(f'<div class="obs-mission">🎯 {escaped}</div>')
            continue

        # Carrying items
        if line.lower().startswith("you carry"):
            items.append(f'<div class="obs-item obs-carry">🎒 {escaped}</div>')
            continue

        # Highlight colored objects
        for color in ("red", "green", "blue", "purple", "yellow", "grey"):
            for obj in ("key", "ball", "box", "door"):
                pattern = f"{color} {obj}"
                if pattern in escaped:
                    escaped = escaped.replace(
                        pattern,
                        f'<span class="obs-object">{pattern}</span>',
                    )

        if "wall" in escaped:
            escaped = escaped.replace("wall", '<span class="obs-wall">wall</span>')
        if "goal" in escaped.lower():
            escaped = re.sub(
                r"(goal)", r'<span class="obs-object">\1</span>', escaped, flags=re.IGNORECASE,
            )

        for d in ("forward", "left", "right"):
            escaped = escaped.replace(d, f'<span class="obs-direction">{d}</span>')

        items.append(f'<div class="obs-item">{escaped}</div>')

    return CSS + f'<div class="obs-wrap">{"".join(items)}</div>'


# ---------------------------------------------------------------------------
# Rendering — response
# ---------------------------------------------------------------------------

def render_response_html(response: str) -> str:
    thinking, rest = extract_thinking(response)
    action = extract_action(response)

    parts = [CSS, '<div class="resp-wrap">']

    if thinking:
        escaped_think = html.escape(thinking)
        parts.append(f'<div class="think">{escaped_think}</div>')

    rest_clean = TOOL_CALL_RE.sub("", rest).strip()
    if rest_clean:
        parts.append(
            f'<div class="msg msg-assistant">'
            f'<div class="msg-role">REASONING</div>'
            f'<div class="msg-content">{html.escape(rest_clean)}</div>'
            f"</div>"
        )

    if action:
        icon = ACTION_ICONS.get(action, action)
        parts.append(f'<div class="action-tag">{icon}</div>')
    else:
        match = TOOL_CALL_RE.search(response)
        if match:
            parts.append(f'<div class="tool-call">{html.escape(match.group(1).strip()[:300])}</div>')

    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Rendering — action timeline
# ---------------------------------------------------------------------------

def render_action_timeline(turns: list[dict], current_turn: int,
                           compacted: set[int] | None = None) -> str:
    compacted = compacted or set()
    chips = []
    for i, t in enumerate(turns):
        action = t.get("action")
        active_cls = " active" if i == current_turn else ""

        if action == "go forward":
            bg, fg, icon = "#d4edda", "#155724", "⬆"
        elif action == "turn left":
            bg, fg, icon = "#d6eaf8", "#1a4a7a", "↺"
        elif action == "turn right":
            bg, fg, icon = "#d6eaf8", "#1a4a7a", "↻"
        elif action == "pick up":
            bg, fg, icon = "#fff3cd", "#7a5a00", "✋"
        elif action == "drop":
            bg, fg, icon = "#fff3cd", "#7a5a00", "⬇"
        elif action == "toggle":
            bg, fg, icon = "#e8daef", "#5a2a7a", "🔓"
        else:
            bg, fg, icon = "#f8d7da", "#721c24", "?"

        dim = "opacity:0.3;" if i in compacted else ""
        chips.append(
            f'<div class="timeline-chip{active_cls}" style="background:{bg};color:{fg};{dim}">'
            f'{icon}<span class="chip-num">{i + 1}</span></div>'
        )

    return (
        CSS
        + '<div class="timeline-wrap">'
        + '<div class="timeline-title">Action timeline — '
        + "green: forward · blue: turn · orange: interact · purple: toggle · 🟡 = current · dim = compacted</div>"
        + f'<div class="timeline-chips">{"".join(chips)}</div>'
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Rendering — conversation history (viz_babyai style)
# ---------------------------------------------------------------------------

def _find_compacted_turns(turns: list[dict], messages: list[dict],
                          current_turn: int,
                          compaction_events: list[tuple[int, str]] | None = None) -> set[int]:
    """Detect which turns have been dropped from context by the current turn.

    Uses compaction_events (turn_index, marker_text) produced by load_samples.
    Each event at turn_index T means compaction fired just before turn T,
    dropping turns prior to T except for the last n_preserved_turns.

    Returns set of turn indices no longer in the model's context at current_turn.
    """
    if not turns or not compaction_events:
        return set()

    dropped: set[int] = set()
    preserved_start = 0

    for event_turn, marker in compaction_events:
        if event_turn > current_turn:
            break
        m = re.search(r"window=(\d+), preserved=(\d+)", marker)
        if not m:
            continue
        n_preserved = int(m.group(2))
        # Everything before the preserved window gets dropped
        new_preserved_start = event_turn - n_preserved
        for t in range(preserved_start, new_preserved_start):
            dropped.add(t)
        preserved_start = new_preserved_start

    return dropped


def render_conversation_html(messages: list[dict], current_turn: int,
                             system_prompt: str, turns: list[dict] | None = None,
                             compaction_events: list[tuple[int, str]] | None = None) -> str:
    """Render conversation up to the current turn.

    Only dims messages that were compacted (dropped from model context).
    Prior turns that are still in KV are shown at full opacity.
    """
    compacted = _find_compacted_turns(turns or [], messages, current_turn, compaction_events) if turns else set()
    has_user = any(m["role"] == "user" for m in messages if m["role"] != "system")

    # Build set of turn indices where compaction events should be rendered
    compact_event_map: dict[int, list[str]] = {}
    for turn_i, marker_text in (compaction_events or []):
        compact_event_map.setdefault(turn_i, []).append(marker_text)

    parts = [CSS, '<div class="msg-wrap">']

    # System prompt (always visible but compact)
    if system_prompt:
        parts.append(
            '<div class="msg msg-system">'
            '<div class="msg-role">SYSTEM</div>'
            f'<div class="msg-content">{html.escape(system_prompt)}</div>'
            "</div>"
        )

    # Detect compaction markers in messages to know which turns were dropped
    compaction_points = set()  # turn indices right after a compaction event
    _compact_marker_re = re.compile(r"\[COMPACTION: (\d+) messages? dropped")
    _ti = 0
    for msg in messages:
        if msg["role"] == "system":
            if _compact_marker_re.search(msg.get("content", "")):
                compaction_points.add(_ti)
            continue
        if msg["role"] == "assistant":
            _ti += 1

    turn_idx = 0
    for msg in messages:
        if msg["role"] == "system":
            # Render compaction markers as banners
            content = msg.get("content", "")
            m_compact = _compact_marker_re.search(content)
            if m_compact:
                parts.append(
                    '<div class="compact-banner compact-dropped">'
                    f"⚡ {content}"
                    "</div>"
                )
            continue

        role = msg["role"]
        msg_turn = turn_idx

        # Stop after current turn's response
        if msg_turn > current_turn:
            break

        # Render compaction event banners from structured data
        if role == "user" and msg_turn in compact_event_map:
            for marker in compact_event_map[msg_turn]:
                parts.append(
                    '<div class="compact-banner compact-dropped">'
                    f"⚡ {html.escape(marker)}"
                    "</div>"
                )

        # Compaction banner (heuristic fallback when no markers present)
        if not compaction_points and not compact_event_map and has_user and role == "user" and msg_turn == current_turn and current_turn > 0:
            n_compacted = len([t for t in range(current_turn) if t in compacted])
            if n_compacted > 0:
                parts.append(
                    '<div class="compact-banner compact-kv">'
                    f"↑ {n_compacted} turn(s) compacted out of context"
                    "</div>"
                )

        cls = {
            "system": "msg-system",
            "user": "msg-user",
            "assistant": "msg-assistant",
            "tool": "msg-tool",
        }.get(role, "msg-user")

        # Dim if compacted (heuristic) or if before a compaction marker
        if compaction_points:
            # With real markers: dim turns before the latest compaction point up to current turn
            latest_compact = max((cp for cp in compaction_points if cp <= msg_turn), default=-1)
            prior_cls = " msg-prior" if latest_compact >= 0 and msg_turn < latest_compact else ""
        else:
            prior_cls = " msg-prior" if msg_turn in compacted else ""
        content = msg["content"]

        # Rich rendering for current turn's assistant message
        if role == "assistant" and msg_turn == current_turn:
            thinking, rest = extract_thinking(content)
            action = extract_action(content)
            inner_parts = []
            if thinking:
                inner_parts.append(f'<div class="think">{html.escape(thinking[:600])}'
                                   f'{"…" if len(thinking) > 600 else ""}</div>')
            rest_clean = TOOL_CALL_RE.sub("", rest).strip()
            if rest_clean:
                inner_parts.append(f'<div class="msg-content">{html.escape(rest_clean)}</div>')
            if action:
                icon = ACTION_ICONS.get(action, action)
                inner_parts.append(f'<div class="action-tag">{icon}</div>')
            elif TOOL_CALL_RE.search(content):
                raw_tc = TOOL_CALL_RE.search(content).group(1).strip()
                inner_parts.append(f'<div class="tool-call">{html.escape(raw_tc[:200])}</div>')
            rendered_content = "\n".join(inner_parts)
        else:
            rendered_content = f'<div class="msg-content">{html.escape(content)}</div>'

        parts.append(
            f'<div class="msg {cls}{prior_cls}">'
            f'<div class="msg-role">{role.upper()}</div>'
            f"{rendered_content}"
            f"</div>"
        )

        # Advance turn counter
        if role == "assistant":
            turn_idx += 1

    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(run_path: str = None, local_path: str = None,
                  n_max_turns: int = 0, n_preserved_turns: int = 0) -> list[dict]:
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
        system_prompt, turns, compaction_events = messages_to_turns(sample["messages_parsed"])
        sample["system_prompt"] = system_prompt
        sample["turns"] = turns
        # Synthesize compaction events from sliding window config if not already present.
        # Compaction fires when accumulated turns exceed n_max_turns, dropping
        # all but n_preserved_turns. Then the window refills until it hits n_max_turns again.
        if not compaction_events and n_max_turns > 0 and len(turns) > n_max_turns:
            window_count = 0
            for t in range(len(turns)):
                window_count += 1
                if window_count > n_max_turns:
                    dropped = window_count - 1 - n_preserved_turns
                    compaction_events.append(
                        (t, f"[COMPACTION: {dropped} turns dropped — "
                            f"window={n_max_turns}, preserved={n_preserved_turns}]")
                    )
                    window_count = n_preserved_turns + 1
        sample["compaction_events"] = compaction_events
        samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def _reward_badge(reward: float) -> str:
    if reward > 0:
        cls = "reward-pos"
    elif reward == 0:
        cls = "reward-zero"
    else:
        cls = "reward-neg"
    return f'<span class="reward-badge {cls}">{reward:.2f}</span>'


def build_app(all_sources: dict[str, list[dict]]) -> gr.Blocks:
    with gr.Blocks(title="Training Rollout Viewer", theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.HTML(
            CSS
            + '<h1 style="margin:0;">Training Rollout Viewer</h1>'
            + '<p style="color:#555;font-size:13px;margin:4px 0 12px 0;">'
            + "Observation (left) · Model response (right) · "
            + "Conversation history (bottom left) · Action timeline (bottom right)"
            + "</p>"
        )

        # Controls row
        with gr.Row():
            source_dd = gr.Dropdown(
                choices=list(all_sources.keys()),
                value=list(all_sources.keys())[0] if all_sources else None,
                label="Source",
                scale=2,
            )
            step_dd = gr.Dropdown(choices=[], label="Step", scale=1)
            task_dd = gr.Dropdown(choices=[], label="Task / Example", scale=2)
            rollout_dd = gr.Dropdown(choices=[], label="Rollout", scale=3)
            turn_sl = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Turn", scale=1)

        ep_bar = gr.Markdown("")

        # Main view: grid + observation + response
        with gr.Row():
            grid_html = gr.HTML(label="Grid View")
            obs_html = gr.HTML(label="Observation")
            resp_html = gr.HTML(label="Model Response")

        # Bottom: conversation history + action timeline
        with gr.Row():
            conv_html = gr.HTML(label="Conversation History")
            with gr.Column(scale=1):
                timeline_html = gr.HTML(label="Action Timeline")
                system_html = gr.HTML(label="System Prompt")

        # ── helpers ───────────────────────────────────────────────────

        def _get_samples(source_name: str) -> list[dict]:
            return all_sources.get(source_name, [])

        def _steps_for_source(source_name: str) -> list[str]:
            samples = _get_samples(source_name)
            steps = sorted({s.get("step", 0) for s in samples})
            return ["all"] + [str(s) for s in steps]

        def _rollout_labels(samples: list[dict]) -> list[str]:
            labels = []
            for i, s in enumerate(samples):
                reward = s.get("reward", 0)
                n_turns = len(s["turns"])
                task = s.get("task", "?")
                if task and len(task) > 30:
                    task = task[:27] + "..."
                step = s.get("step", "?")
                badge = "+" if reward > 0 else ("0" if reward == 0 else "-")
                labels.append(
                    f"[s{step}] #{s.get('example_id', i)} | "
                    f"r={reward:.1f} ({badge}) | "
                    f"t={n_turns} | {task}"
                )
            return labels

        def _has_multiple_tasks(source_name: str) -> bool:
            samples = _get_samples(source_name)
            tasks = {s.get("task", "?") or "?" for s in samples}
            return len(tasks) > 1

        def _tasks_for_source(source_name: str, step_filter: str) -> list[str]:
            samples = _get_samples(source_name)
            if step_filter and step_filter != "all":
                step_val = int(step_filter)
                samples = [s for s in samples if s.get("step") == step_val]
            if _has_multiple_tasks(source_name):
                values = sorted({s.get("task", "?") or "?" for s in samples})
            else:
                values = sorted({str(s.get("example_id", "?")) for s in samples}, key=lambda x: int(x) if x.isdigit() else x)
            return ["all"] + values

        def _filter_samples(source_name: str, step_filter: str, task_filter: str = "all") -> list[dict]:
            samples = _get_samples(source_name)
            if step_filter and step_filter != "all":
                step_val = int(step_filter)
                samples = [s for s in samples if s.get("step") == step_val]
            if task_filter and task_filter != "all":
                if _has_multiple_tasks(source_name):
                    samples = [s for s in samples if (s.get("task") or "?") == task_filter]
                else:
                    samples = [s for s in samples if str(s.get("example_id", "?")) == task_filter]
            return samples

        # ── callbacks ─────────────────────────────────────────────────

        def on_source(source_name):
            if not source_name:
                return (
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    gr.update(maximum=0, value=0),
                    "",
                )
            steps = _steps_for_source(source_name)
            tasks = _tasks_for_source(source_name, "all")
            samples = _get_samples(source_name)
            labels = _rollout_labels(samples)
            return (
                gr.update(choices=steps, value="all"),
                gr.update(choices=tasks, value="all"),
                gr.update(choices=labels, value=labels[0] if labels else None),
                gr.update(maximum=0, value=0),
                "",
            )

        def on_step(source_name, step_filter):
            if not source_name:
                return gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(maximum=0, value=0), ""
            tasks = _tasks_for_source(source_name, step_filter)
            samples = _filter_samples(source_name, step_filter)
            labels = _rollout_labels(samples)
            return (
                gr.update(choices=tasks, value="all"),
                gr.update(choices=labels, value=labels[0] if labels else None),
                gr.update(maximum=0, value=0),
                "",
            )

        def on_task(source_name, step_filter, task_filter):
            if not source_name:
                return gr.update(choices=[], value=None), gr.update(maximum=0, value=0), ""
            samples = _filter_samples(source_name, step_filter, task_filter)
            labels = _rollout_labels(samples)
            return (
                gr.update(choices=labels, value=labels[0] if labels else None),
                gr.update(maximum=0, value=0),
                "",
            )

        def on_rollout(source_name, step_filter, task_filter, choice):
            if not choice or not source_name:
                return gr.update(maximum=0, value=0), "", "", "", "", "", "", ""
            samples = _filter_samples(source_name, step_filter, task_filter)
            labels = _rollout_labels(samples)
            if choice not in labels:
                return gr.update(maximum=0, value=0), "", "", "", "", "", "", ""
            idx = labels.index(choice)
            s = samples[idx]
            turns = s["turns"]
            n = len(turns)

            actions = [t["action"] for t in turns if t["action"]]
            action_counts = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            action_summary = ", ".join(f"{v}x {k}" for k, v in action_counts.items())

            # Detect mode from content
            has_user = any(m["role"] == "user" for m in s["messages_parsed"][1:])
            mode = "normal" if has_user else "markovian_pure"

            reward = s.get("reward", 0)
            n_compactions = len(s.get("compaction_events", []))
            bar = (
                f"**Step:** {s.get('step', '?')} | "
                f"**Example:** {s.get('example_id', '?')} | "
                f"**Task:** {s.get('task', '?')} | "
                f"**Reward:** {reward:.2f} | "
                f"**Turns:** {n} | "
                f"**Valid actions:** {len(actions)}/{n} | "
                f"**Mode:** {mode} | "
                f"**Compactions:** {n_compactions} | "
                f"**Actions:** {action_summary or 'none'}"
            )

            max_turn = max(0, n - 1)

            grid = ""
            obs = ""
            resp = ""
            tl = ""
            conv = ""
            sys_html = ""
            if turns:
                compacted = _find_compacted_turns(turns, s["messages_parsed"], 0, s.get("compaction_events"))
                obs = render_observation_html(turns[0]["obs"])
                resp = render_response_html(turns[0]["response"])
                tl = render_action_timeline(turns, 0, compacted)
                conv = render_conversation_html(
                    s["messages_parsed"], 0, s["system_prompt"], turns,
                    s.get("compaction_events"),
                )
                grid = render_ego_grid_html(turns[0]["obs"])
            if s["system_prompt"]:
                sys_html = (
                    CSS
                    + '<div class="msg-wrap"><div class="msg msg-system">'
                    + '<div class="msg-role">SYSTEM</div>'
                    + f'<div class="msg-content">{html.escape(s["system_prompt"])}</div>'
                    + "</div></div>"
                )

            return gr.update(maximum=max_turn, value=0), bar, grid, obs, resp, conv, tl, sys_html

        def on_turn(source_name, step_filter, task_filter, choice, turn):
            if not choice or not source_name:
                return "", "", "", "", ""
            samples = _filter_samples(source_name, step_filter, task_filter)
            labels = _rollout_labels(samples)
            if choice not in labels:
                return "", "", "", "", ""
            idx = labels.index(choice)
            s = samples[idx]
            turns = s["turns"]
            t = int(turn)
            if t >= len(turns):
                return "", "", "", "", ""

            prev_obs = turns[t - 1]["obs"] if t > 0 else None
            compacted = _find_compacted_turns(turns, s["messages_parsed"], t, s.get("compaction_events"))
            grid = render_ego_grid_html(turns[t]["obs"], prev_obs)
            obs = render_observation_html(turns[t]["obs"])
            resp = render_response_html(turns[t]["response"])
            tl = render_action_timeline(turns, t, compacted)
            conv = render_conversation_html(
                s["messages_parsed"], t, s["system_prompt"], s["turns"],
                s.get("compaction_events"),
            )
            return grid, obs, resp, conv, tl

        # ── wiring ────────────────────────────────────────────────────

        source_dd.change(
            on_source,
            [source_dd],
            [step_dd, task_dd, rollout_dd, turn_sl, ep_bar],
        )
        step_dd.change(
            on_step,
            [source_dd, step_dd],
            [task_dd, rollout_dd, turn_sl, ep_bar],
        )
        task_dd.change(
            on_task,
            [source_dd, step_dd, task_dd],
            [rollout_dd, turn_sl, ep_bar],
        )
        rollout_dd.change(
            on_rollout,
            [source_dd, step_dd, task_dd, rollout_dd],
            [turn_sl, ep_bar, grid_html, obs_html, resp_html, conv_html, timeline_html, system_html],
        )
        turn_sl.change(
            on_turn,
            [source_dd, step_dd, task_dd, rollout_dd, turn_sl],
            [grid_html, obs_html, resp_html, conv_html, timeline_html],
        )
        demo.load(
            on_source,
            [source_dd],
            [step_dd, task_dd, rollout_dd, turn_sl, ep_bar],
        )

    return demo


def _load_compaction_config(run_dir: Path) -> tuple[int, int]:
    """Extract n_max_turns and n_preserved_turns from an experiment's orchestrator config.

    Searches for configs/orchestrator.toml in the run_dir or its parents.
    Returns (n_max_turns, n_preserved_turns), defaulting to (0, 0) if not found.
    """
    candidates = [
        run_dir / "configs" / "orchestrator.toml",
        run_dir / "orchestrator.toml",
        run_dir.parent / "configs" / "orchestrator.toml",
    ]
    for config_path in candidates:
        if config_path.exists():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            # env args can be in [[env]] array or [env.args]
            for env in config.get("env", []):
                args = env.get("args", {})
                n_max = args.get("n_max_turns", 0)
                n_pres = args.get("n_preserved_turns", 0)
                if n_max:
                    mode = args.get("compaction_mode", "unknown")
                    print(f"  Auto-detected from {config_path}: "
                          f"n_max_turns={n_max}, n_preserved_turns={n_pres}, "
                          f"compaction_mode={mode}")
                    return n_max, n_pres
    return 0, 0


def main():
    parser = argparse.ArgumentParser(description="Visualize training rollouts")
    parser.add_argument("run_paths", nargs="*", help="wandb run path(s) (entity/project/run_id)")
    parser.add_argument("--local", action="append", default=[], help="Path(s) to local .table.json files")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--run-dir", default=None,
                        help="Experiment run directory (contains configs/orchestrator.toml). "
                             "Auto-extracts n_max_turns, n_preserved_turns, compaction_mode.")
    parser.add_argument("--n-max-turns", type=int, default=0,
                        help="Sliding window size (overrides auto-detected value from config).")
    parser.add_argument("--n-preserved-turns", type=int, default=0,
                        help="Turns preserved after compaction (overrides auto-detected value).")
    args = parser.parse_args()

    # Auto-detect compaction config from run directory
    n_max = args.n_max_turns
    n_pres = args.n_preserved_turns
    if args.run_dir:
        detected_max, detected_pres = _load_compaction_config(Path(args.run_dir))
        n_max = n_max or detected_max
        n_pres = n_pres or detected_pres

    all_sources: dict[str, list[dict]] = {}

    for local_path in args.local:
        p = Path(local_path)
        if p.is_dir():
            files = sorted(p.glob("**/*.table.json"), key=lambda f: f.stat().st_mtime)
            if not files:
                print(f"No .table.json files found in {p}")
            for f in files:
                run_dir = next(
                    (part for part in f.parts if part.startswith("run-")), None
                )
                label = run_dir or f.name
                print(f"Loading local: {f}")
                new_samples = load_samples(local_path=str(f), n_max_turns=n_max, n_preserved_turns=n_pres)
                if label in all_sources:
                    all_sources[label].extend(new_samples)
                else:
                    all_sources[label] = new_samples
        elif p.exists():
            print(f"Loading local: {p}")
            all_sources[p.name] = load_samples(local_path=str(p), n_max_turns=n_max, n_preserved_turns=n_pres)

    for run_path in args.run_paths:
        print(f"Loading wandb: {run_path}")
        all_sources[run_path] = load_samples(run_path=run_path, n_max_turns=n_max, n_preserved_turns=n_pres)

    # Auto-discover if nothing specified
    if not all_sources:
        patterns = [
            Path("/network/scratch/e/emiliano.penaloza/outputs").glob(
                "*/run_default/wandb/*/files/media/table/*.table.json"
            ),
            Path("artifacts").glob("run-*-final-samples*/*.table.json"),
        ]
        found = []
        for pat in patterns:
            found.extend(sorted(pat, key=lambda p: p.stat().st_mtime, reverse=True))
        for f in found[:5]:
            print(f"Auto-discovered: {f}")
            all_sources[f.name] = load_samples(local_path=str(f), n_max_turns=n_max, n_preserved_turns=n_pres)

    if not all_sources:
        print("No data found. Usage:")
        print("  viz_training_rollouts.py <wandb_run_path>")
        print("  viz_training_rollouts.py --local /path/to/samples.table.json")
        return

    total = sum(len(v) for v in all_sources.values())
    print(f"Loaded {total} rollouts from {len(all_sources)} source(s)")

    demo = build_app(all_sources)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
