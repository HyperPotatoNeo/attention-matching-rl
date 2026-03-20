"""Gradio dashboard for browsing eval results (BabyAI and AIME).

Usage:
    uv run python scripts/viz_results.py
    uv run python scripts/viz_results.py --port 7860
"""

import argparse
import json
from pathlib import Path

import gradio as gr
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _detect_schema(data: dict) -> str:
    if "per_env" in data:
        return "babyai"
    if "accuracy" in data and "n_problems" in data:
        return "aime"
    return "unknown"


def _load_babyai(data: dict, path: str) -> dict:
    per_env = data["per_env"]
    env_rows = [
        {
            "env": env,
            "success_rate": f"{v['success_rate']:.1%}",
            "successes": v["successes"],
            "total": v["total"],
            "avg_turns": f"{v['avg_turns']:.1f}",
            "avg_tokens": f"{v['avg_tokens']:.0f}",
        }
        for env, v in per_env.items()
    ]
    episode_rows = [
        {
            "idx": r["idx"],
            "env": r["env"],
            "success": "✓" if r["success"] else "✗",
            "turns": r["turns"],
            "tokens": r["tokens"],
            "compactions": r.get("compactions", 0),
            "time_s": f"{r['time']:.1f}",
        }
        for r in data["results"]
    ]
    summary = {
        "file": Path(path).name,
        "mode": data.get("mode", "?"),
        "episodes": data["n_episodes"],
        "success_rate": f"{data['success_rate']:.1%}",
        "avg_tokens": f"{data['avg_tokens']:.0f}",
        "avg_turns": f"{data['avg_turns']:.1f}",
        "wall_time_s": f"{data.get('wall_time', 0):.0f}",
    }
    return {"schema": "babyai", "summary": summary, "env_df": pd.DataFrame(env_rows), "episode_df": pd.DataFrame(episode_rows)}


def _load_aime(data: dict, path: str) -> dict:
    episode_rows = [
        {
            "idx": r.get("idx", i),
            "problem_id": r.get("problem_id", r.get("idx", i)),
            "correct": "✓" if r.get("correct") else "✗",
            "tokens": r.get("tokens", r.get("total_tokens", 0)),
            "time_s": f"{r.get('time', 0):.1f}",
        }
        for i, r in enumerate(data["results"])
        if r is not None
    ]
    summary = {
        "file": Path(path).name,
        "mode": data.get("mode", "?"),
        "problems": data["n_problems"],
        "accuracy": f"{data['accuracy']:.1%}",
        "correct": data.get("correct", "?"),
        "wall_time_s": f"{data.get('wall_time', 0):.0f}",
    }
    return {"schema": "aime", "summary": summary, "episode_df": pd.DataFrame(episode_rows)}


def load_result_file(path: str) -> dict | None:
    try:
        data = json.loads(Path(path).read_text())
        schema = _detect_schema(data)
        if schema == "babyai":
            return _load_babyai(data, path)
        if schema == "aime":
            return _load_aime(data, path)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_result_files(roots: list[str]) -> list[str]:
    files = []
    for root in roots:
        p = Path(root)
        if p.is_file() and p.suffix == ".json":
            files.append(str(p))
        elif p.is_dir():
            files.extend(str(f) for f in sorted(p.glob("*.json")))
    files.extend(str(f) for f in Path(".").glob("results_babyai*.json"))
    return sorted(set(files))


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(result_files: list[str]) -> gr.Blocks:
    loaded = {}
    for f in result_files:
        r = load_result_file(f)
        if r:
            loaded[Path(f).name] = r

    with gr.Blocks(title="Eval Results") as demo:
        gr.Markdown("# Eval Results Dashboard")

        with gr.Tab("Overview"):
            gr.Markdown("### All loaded result files")
            summary_rows = [v["summary"] for v in loaded.values()]
            if summary_rows:
                gr.DataFrame(pd.DataFrame(summary_rows))
            else:
                gr.Markdown("_No result files found._")

        for fname, result in loaded.items():
            schema = result["schema"]
            with gr.Tab(fname[:40]):
                summary = result["summary"]
                gr.Markdown(f"**Mode:** {summary.get('mode', '?')}  |  **File:** {fname}")

                if schema == "babyai":
                    gr.Markdown(f"**Success rate:** {summary['success_rate']}  |  **Avg tokens:** {summary['avg_tokens']}  |  **Avg turns:** {summary['avg_turns']}")
                    gr.Markdown("#### Per-environment breakdown")
                    gr.DataFrame(result["env_df"])
                    gr.Markdown("#### Episode details")
                    gr.DataFrame(result["episode_df"])

                elif schema == "aime":
                    gr.Markdown(f"**Accuracy:** {summary['accuracy']}  |  **Correct:** {summary['correct']}/{summary['problems']}")
                    gr.Markdown("#### Episode details")
                    gr.DataFrame(result["episode_df"])

        with gr.Tab("Compare"):
            gr.Markdown("### Side-by-side comparison")
            babyai = {k: v for k, v in loaded.items() if v["schema"] == "babyai"}
            aime = {k: v for k, v in loaded.items() if v["schema"] == "aime"}

            if babyai:
                gr.Markdown("#### BabyAI")
                comp_rows = [v["summary"] for v in babyai.values()]
                gr.DataFrame(pd.DataFrame(comp_rows))

                # Per-env success rates across files
                env_comp = {}
                for fname, result in babyai.items():
                    for _, row in result["env_df"].iterrows():
                        env = row["env"]
                        if env not in env_comp:
                            env_comp[env] = {"env": env}
                        env_comp[env][fname[:20]] = row["success_rate"]
                if env_comp:
                    gr.Markdown("##### Success rate by env")
                    gr.DataFrame(pd.DataFrame(list(env_comp.values())))

            if aime:
                gr.Markdown("#### AIME")
                comp_rows = [v["summary"] for v in aime.values()]
                gr.DataFrame(pd.DataFrame(comp_rows))

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("paths", nargs="*", default=["results"])
    args = parser.parse_args()

    files = find_result_files(args.paths)
    print(f"Found {len(files)} result files: {[Path(f).name for f in files]}")

    demo = build_app(files)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
