"""Qualitative analysis of 4-way compaction eval trajectories.

Focuses on:
1. Comparing same-problem trajectories across conditions
2. Analyzing text around compaction events
3. Sequence length distributions
4. Per-task trajectory differences
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

OUTDIR = Path("/pscratch/sd/s/siddart2/compaction-rl/results_compaction_4way")

FILES = {
    "compact_trained_compact": OUTDIR / "compact_trained_with_compaction_300.json",
    "compact_trained_baseline": OUTDIR / "compact_trained_no_compaction_300.json",
    "baseline_rl_compact": OUTDIR / "baseline_rl_with_compaction_300.json",
    "baseline_rl_baseline": OUTDIR / "baseline_rl_no_compaction_300.json",
}


def load_results():
    data = {}
    for name, path in FILES.items():
        with open(path) as f:
            d = json.load(f)
        results_by_idx = {}
        for r in d["results"]:
            results_by_idx[r["idx"]] = r
        data[name] = {
            "summary": {k: v for k, v in d.items() if k != "results"},
            "results": results_by_idx,
        }
    return data


def print_summary(data):
    print("=" * 90)
    print("SUMMARY: 4-Way Compaction Evaluation (300 problems, seed=42)")
    print("=" * 90)
    print()
    print(f"{'Condition':<35} {'Accuracy':>10} {'AvgTok':>8} {'TotalTok':>10}")
    print("-" * 70)
    for name, d in data.items():
        s = d["summary"]
        print(f"{name:<35} {s['accuracy']:>9.1%} {s['avg_tokens']:>8.0f} {s['total_tokens']:>10}")
    print()


def per_task_comparison(data):
    print("=" * 90)
    print("PER-TASK COMPARISON")
    print("=" * 90)

    tasks = sorted(list(data["baseline_rl_baseline"]["summary"]["per_task"].keys()))

    for task in tasks:
        print(f"\n  {task}:")
        print(f"  {'Condition':<35} {'Accuracy':>10} {'AvgTok':>8}")
        print(f"  {'-'*55}")
        for name, d in data.items():
            pt = d["summary"]["per_task"].get(task, {})
            if pt:
                print(f"  {name:<35} {pt['accuracy']:>9.1%} {pt['avg_tokens']:>8.0f}")
    print()


def analyze_compaction_events(data):
    print("=" * 90)
    print("COMPACTION EVENT ANALYSIS")
    print("=" * 90)
    print()

    for condition in ["compact_trained_compact", "baseline_rl_compact"]:
        results = data[condition]["results"]
        events_by_count = defaultdict(list)
        all_events = []

        for idx, r in results.items():
            nc = r.get("n_compactions", 0)
            events_by_count[nc].append(r)
            for evt in r.get("diagnostics", {}).get("compaction_events", []):
                all_events.append(evt)

        print(f"  {condition}:")
        print(f"  Total compaction events: {len(all_events)}")

        if all_events:
            ratios = [e["ratio"] for e in all_events if "ratio" in e]
            algo_times = [e["algo_time"] for e in all_events if "algo_time" in e]
            positions = [e.get("position", 0) for e in all_events if "position" in e]

            if ratios:
                print(f"  Avg compression ratio: {sum(ratios)/len(ratios):.3f}")
            if algo_times:
                print(f"  Avg algo time: {sum(algo_times)/len(algo_times):.3f}s")
            if positions:
                print(f"  Avg compaction position: {sum(positions)/len(positions):.0f} tokens")
                print(f"  Min/Max position: {min(positions)}/{max(positions)}")

        # Distribution of compaction counts
        print(f"  Compaction count distribution:")
        for nc in sorted(events_by_count.keys()):
            items = events_by_count[nc]
            correct = sum(1 for r in items if r.get("correct", False))
            total = len(items)
            avg_tok = sum(r["tokens"] for r in items) / total if total else 0
            print(f"    {nc} compactions: {total:3d} problems, "
                  f"acc={correct}/{total} ({correct/total:.1%}), avg_tok={avg_tok:.0f}")
        print()


def analyze_divergent_problems(data):
    """Find problems where outcomes differ between conditions."""
    print("=" * 90)
    print("DIVERGENT PROBLEM ANALYSIS")
    print("=" * 90)
    print()

    # Find problems where baseline_rl succeeds without compaction but fails with
    baseline_hurt = []
    baseline_help = []
    compact_hurt = []
    compact_help = []

    all_idxs = sorted(data["baseline_rl_baseline"]["results"].keys())

    for idx in all_idxs:
        bl_base = data["baseline_rl_baseline"]["results"].get(idx, {})
        bl_comp = data["baseline_rl_compact"]["results"].get(idx, {})
        ct_base = data["compact_trained_baseline"]["results"].get(idx, {})
        ct_comp = data["compact_trained_compact"]["results"].get(idx, {})

        # Baseline RL: compaction hurts
        if bl_base.get("correct") and not bl_comp.get("correct"):
            baseline_hurt.append(idx)
        # Baseline RL: compaction helps
        if not bl_base.get("correct") and bl_comp.get("correct"):
            baseline_help.append(idx)
        # Compact-trained: compaction hurts
        if ct_base.get("correct") and not ct_comp.get("correct"):
            compact_hurt.append(idx)
        # Compact-trained: compaction helps
        if not ct_base.get("correct") and ct_comp.get("correct"):
            compact_help.append(idx)

    print(f"  Baseline RL: compaction HURTS on {len(baseline_hurt)} problems, "
          f"HELPS on {len(baseline_help)} problems")
    print(f"  Compact-trained: compaction HURTS on {len(compact_hurt)} problems, "
          f"HELPS on {len(compact_help)} problems")
    print()

    # Task breakdown of divergent problems
    for label, idxs in [("Baseline RL hurt by compaction", baseline_hurt),
                         ("Baseline RL helped by compaction", baseline_help),
                         ("Compact-trained hurt by compaction", compact_hurt),
                         ("Compact-trained helped by compaction", compact_help)]:
        if not idxs:
            continue
        task_counts = defaultdict(int)
        for idx in idxs:
            task = data["baseline_rl_baseline"]["results"][idx]["task"]
            task_counts[task] += 1
        print(f"  {label} ({len(idxs)} problems):")
        for task in sorted(task_counts.keys(), key=lambda t: -task_counts[t]):
            print(f"    {task}: {task_counts[task]}")
        print()

    return baseline_hurt, baseline_help, compact_hurt, compact_help


def analyze_compaction_region_text(data, sample_idxs, label, n_samples=5):
    """Qualitatively analyze text around compaction events."""
    print("=" * 90)
    print(f"TRAJECTORY ANALYSIS: {label}")
    print("=" * 90)

    count = 0
    for idx in sample_idxs:
        if count >= n_samples:
            break

        bl_comp = data["baseline_rl_compact"]["results"].get(idx, {})
        bl_base = data["baseline_rl_baseline"]["results"].get(idx, {})
        ct_comp = data["compact_trained_compact"]["results"].get(idx, {})
        ct_base = data["compact_trained_baseline"]["results"].get(idx, {})

        # Need text to analyze
        if not bl_comp.get("text") and not ct_comp.get("text"):
            continue

        task = bl_comp.get("task") or ct_comp.get("task", "unknown")
        question = bl_comp.get("question") or ct_comp.get("question", "N/A")

        print(f"\n{'─'*80}")
        print(f"Problem {idx} ({task})")
        print(f"Question (first 200 chars): {question[:200]}...")
        print()

        for cond_name, r in [("baseline_rl + compaction", bl_comp),
                              ("baseline_rl + baseline", bl_base),
                              ("compact_trained + compaction", ct_comp),
                              ("compact_trained + baseline", ct_base)]:
            text = r.get("text", "")
            correct = r.get("correct", False)
            tokens = r.get("tokens", 0)
            n_comp = r.get("n_compactions", 0)
            status = "CORRECT" if correct else "WRONG"

            print(f"  [{cond_name}] {status}, {tokens} tokens, {n_comp} compactions")

            if text and n_comp > 0:
                events = r.get("diagnostics", {}).get("compaction_events", [])
                if events:
                    # Show text around first compaction event
                    first_event = events[0]
                    pos = first_event.get("position", 0)

                    # Show 200 chars before and after the compaction position
                    # Token position to char position is approximate (3-4 chars per token)
                    char_pos = pos * 3
                    start = max(0, char_pos - 300)
                    end = min(len(text), char_pos + 300)

                    if start < len(text):
                        snippet = text[start:end]
                        print(f"    First compaction at ~token {pos}:")
                        print(f"    ...{snippet[:150]}...")
                        print(f"    [COMPACTION EVENT: ratio={first_event.get('ratio', '?'):.3f}]")
                        print(f"    ...{snippet[150:300]}...")

                    # If there are multiple compactions, show last one too
                    if len(events) > 2:
                        last_event = events[-1]
                        last_pos = last_event.get("position", 0)
                        char_pos = last_pos * 3
                        start = max(0, char_pos - 200)
                        end = min(len(text), char_pos + 200)
                        if start < len(text):
                            print(f"    Last compaction at ~token {last_pos}:")
                            print(f"    ...{text[start:end][:200]}...")

            elif text and n_comp == 0:
                # Show first 200 chars of the response
                print(f"    Response start: {text[:200]}...")

            print()

        count += 1


def sequence_length_analysis(data):
    """Analyze how compaction affects sequence length distributions."""
    print("=" * 90)
    print("SEQUENCE LENGTH ANALYSIS")
    print("=" * 90)
    print()

    for condition in data:
        results = data[condition]["results"]
        tokens_list = [r["tokens"] for r in results.values()]
        tokens_list.sort()

        n = len(tokens_list)
        p25 = tokens_list[n // 4]
        p50 = tokens_list[n // 2]
        p75 = tokens_list[3 * n // 4]
        mean = sum(tokens_list) / n
        max_tok = max(tokens_list)
        min_tok = min(tokens_list)

        # Count sequences hitting max tokens (8192 or close)
        maxed_out = sum(1 for t in tokens_list if t >= 8000)

        print(f"  {condition}:")
        print(f"    Mean: {mean:.0f}, Median: {p50}, P25: {p25}, P75: {p75}")
        print(f"    Min: {min_tok}, Max: {max_tok}")
        print(f"    Hit max tokens (>=8000): {maxed_out}/{n} ({maxed_out/n:.1%})")
        print()


def zero_compaction_analysis(data):
    """Analyze problems that had 0 compactions even in compaction mode."""
    print("=" * 90)
    print("ZERO-COMPACTION ANALYSIS (problems solved before first compaction)")
    print("=" * 90)
    print()

    for condition in ["compact_trained_compact", "baseline_rl_compact"]:
        results = data[condition]["results"]
        zero = [r for r in results.values() if r.get("n_compactions", 0) == 0]
        nonzero = [r for r in results.values() if r.get("n_compactions", 0) > 0]

        zero_correct = sum(1 for r in zero if r["correct"])
        nonzero_correct = sum(1 for r in nonzero if r["correct"])
        zero_avg_tok = sum(r["tokens"] for r in zero) / len(zero) if zero else 0
        nonzero_avg_tok = sum(r["tokens"] for r in nonzero) / len(nonzero) if nonzero else 0

        print(f"  {condition}:")
        print(f"    0 compactions: {len(zero)} problems, "
              f"acc={zero_correct}/{len(zero)} ({zero_correct/len(zero):.1%}), "
              f"avg_tok={zero_avg_tok:.0f}")
        print(f"    1+ compactions: {len(nonzero)} problems, "
              f"acc={nonzero_correct}/{len(nonzero)} ({nonzero_correct/len(nonzero):.1%}), "
              f"avg_tok={nonzero_avg_tok:.0f}")

        # Per-task for zero compaction
        zero_by_task = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in zero:
            zero_by_task[r["task"]]["total"] += 1
            if r["correct"]:
                zero_by_task[r["task"]]["correct"] += 1
        print(f"    Zero-compaction by task:")
        for task in sorted(zero_by_task.keys()):
            t = zero_by_task[task]
            print(f"      {task}: {t['correct']}/{t['total']} ({t['correct']/t['total']:.1%})")
        print()


def main():
    data = load_results()

    print_summary(data)
    per_task_comparison(data)
    sequence_length_analysis(data)
    analyze_compaction_events(data)
    zero_compaction_analysis(data)
    hurt, help_, compact_hurt, compact_help = analyze_divergent_problems(data)

    # Qualitative trajectory analysis
    if hurt:
        analyze_compaction_region_text(
            data, hurt, "Baseline RL: compaction HURT accuracy", n_samples=5)
    if help_:
        analyze_compaction_region_text(
            data, help_, "Baseline RL: compaction HELPED accuracy", n_samples=5)

    # Problems where all 4 conditions have different outcomes
    interesting = []
    all_idxs = sorted(data["baseline_rl_baseline"]["results"].keys())
    for idx in all_idxs:
        outcomes = tuple(
            data[cond]["results"].get(idx, {}).get("correct", False)
            for cond in data
        )
        # Baseline correct, all others wrong = compaction training is purely destructive
        if outcomes == (False, False, True, True):  # compact-trained works, baseline fails
            interesting.append(idx)

    if interesting:
        analyze_compaction_region_text(
            data, interesting,
            "Compact-trained succeeds where baseline fails", n_samples=3)

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
