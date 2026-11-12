from __future__ import annotations

import csv
from collections import defaultdict
import random
import statistics

from .common import project_path, read_json, read_jsonl, write_json

MAIN = ("next_action_canonical_repeat", "next_action_exact_repeat", "any_target_repeat", "first_action_valid", "task_success")


def clustered_interval(values_by_cluster, seed=42, n_boot=2000):
    values = [statistics.mean(values) for _, values in sorted(values_by_cluster.items())]
    if not values:
        return None, None, None
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, None, None
    rng = random.Random(seed)
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return mean, estimates[int(0.025 * n_boot)], estimates[min(int(0.975 * n_boot), n_boot - 1)]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(output_dir):
    out = project_path(output_dir)
    manifest = read_json(out / "manifest.json")
    rows = read_jsonl(out / "trajectories.jsonl")
    if not rows:
        raise ValueError("No completed trajectories")
    planned_config = manifest["config"]
    summary = []
    depths = ["all"] + sorted({str(row["history_depth"]) for row in rows if row["history_depth"] is not None})
    for depth in depths:
        for condition in planned_config["conditions"]:
            selected = [r for r in rows if r["condition"] == condition and (depth == "all" or str(r["history_depth"]) == depth)]
            for metric in MAIN + ("n_generated_actions", "n_generated_failures", "ftb_canonical_repeat_rate"):
                groups = defaultdict(list)
                for row in selected:
                    value = row["metrics"][metric]
                    if value is not None:
                        groups[row["cluster_id"]].append(value)
                mean, lo, hi = clustered_interval(groups)
                if mean is None:
                    continue
                summary.append({"condition": condition, "history_depth": depth, "metric": metric,
                                "mean": mean, "ci_lo": lo, "ci_hi": hi, "task_clusters": len(groups),
                                "trajectories": sum(map(len, groups.values()))})
    # All comparisons are paired on identical checkpoint + sample. Differences
    # are averaged within a task before resampling independent task clusters.
    comparisons = [("abstract", "verbatim"), ("positive_history", "verbatim"),
                   ("positive_history", "abstract"), ("positive_history", "positive_current"),
                   ("positive_history", "verbatim_positive")]
    contrasts = []
    for treatment, baseline in comparisons:
        if not {treatment, baseline} <= set(planned_config["conditions"]):
            continue
        for depth in depths:
            selected = [r for r in rows if depth == "all" or str(r["history_depth"]) == depth]
            left = {(r["case_id"], r["sample"]): r for r in selected if r["condition"] == treatment}
            right = {(r["case_id"], r["sample"]): r for r in selected if r["condition"] == baseline}
            matched = left.keys() & right.keys()
            for metric in MAIN:
                groups = defaultdict(list)
                for key in sorted(matched):
                    a, b = left[key]["metrics"][metric], right[key]["metrics"][metric]
                    if a is not None and b is not None:
                        groups[left[key]["cluster_id"]].append(a - b)
                mean, lo, hi = clustered_interval(groups)
                if mean is None:
                    continue
                contrasts.append({"treatment": treatment, "baseline": baseline, "history_depth": depth,
                                  "metric": metric, "difference": mean, "ci_lo": lo, "ci_hi": hi,
                                  "task_clusters": len(groups), "paired_trajectories": sum(map(len, groups.values())),
                                  "unpaired_trajectories": len(left.keys() ^ right.keys())})
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_contrasts.csv", contrasts)
    events = [turn for row in rows for turn in row["turns"]]
    card_events = [turn for turn in events if turn["card"]]
    token_counts = {}
    for key in ("prompt_tokens", "completion_tokens"):
        values = [turn["response"].get(key) for turn in events]
        token_counts[key] = {"reported_sum": sum(v for v in values if v is not None),
                             "missing_calls": sum(v is None for v in values)}
    stats = {"completed_trajectories": len(rows), "completed_model_calls": len(events),
             "logged_request_attempts": len(read_jsonl(out / "requests.jsonl")),
             "correction_cards": len(card_events), "tokens": token_counts,
             "full_failed_call_card_leaks": sum(t["literal_audit"]["full_failed_call_present"] for t in card_events),
             "cards_with_argument_overlap": sum(bool(t["literal_audit"]["argument_string_overlap"]) for t in card_events)}
    write_json(out / "cost_and_audit.json", stats)
    kind = manifest["result_kind"]
    lines = ["# PinkElephant experiment report", "",
             f"Result type: **{kind}**. Backend: `{manifest['resolved_backend']['kind']}`.", "",
             "Smoke-reference results validate plumbing only and provide no evidence about model behavior."
             if kind == "plumbing_only" else "These are exploratory model results. Confirm on frozen, disjoint task worlds before making a general claim.", "",
             "Rates are means over task clusters; intervals use task-cluster bootstrap (2,000 resamples).",
             "Injected-history variants and sampling repetitions are correlated within a task and are not independent samples.", "",
             "| Condition | History | Next canonical repeat | First action valid | Task success | Task clusters |",
             "|---|---:|---:|---:|---:|---:|"]
    index = {(s["condition"], s["history_depth"], s["metric"]): s for s in summary}
    def fmt(cell):
        if not cell:
            return "n/a"
        return f"{cell['mean']:.3f}" + (f" [{cell['ci_lo']:.3f}, {cell['ci_hi']:.3f}]" if cell['ci_lo'] is not None else "")
    for depth in depths:
        for condition in planned_config["conditions"]:
            cells = [index.get((condition, depth, metric)) for metric in ("next_action_canonical_repeat", "first_action_valid", "task_success")]
            if any(cells):
                n = max(cell["task_clusters"] for cell in cells if cell)
                lines.append(f"| {condition} | {depth} | " + " | ".join(fmt(cell) for cell in cells) + f" | {n} |")
    lines += ["", "Primary history contrast at depth 2 (treatment minus baseline):", "",
              "| Treatment | Baseline | Metric | Difference | 95% interval | Paired tasks |",
              "|---|---|---|---:|---|---:|"]
    for cell in contrasts:
        if cell["history_depth"] == "2" and cell["metric"] in ("next_action_canonical_repeat", "task_success"):
            interval = f"[{cell['ci_lo']:.3f}, {cell['ci_hi']:.3f}]" if cell['ci_lo'] is not None else "n/a"
            lines.append(f"| {cell['treatment']} | {cell['baseline']} | {cell['metric']} | {cell['difference']:+.3f} | {interval} | {cell['task_clusters']} |")
    incomplete = not (out / "completion.json").exists()
    lines += ["", f"Complete batch: **{not incomplete}**. Completed trajectories: {len(rows)}. Model calls: {len(events)}.",
              f"Full-call literal card leaks: {stats['full_failed_call_card_leaks']}. Cards flagged for argument overlap: {stats['cards_with_argument_overlap']}.",
              "", "A valid next call can still be irrelevant. Task success is checked separately by the environment's goal predicate.",
              "Canonical repetition normalizes tool arguments; it does not detect every semantically equivalent failed strategy.",
              "Zero literal card overlap does not prove that a failed action cannot be inferred. Review audit samples manually.",
              "Incomplete batches and unmatched checkpoints must not support a final comparison.",
              "Prompt/output token totals and missing-usage counts are in cost_and_audit.json. No API price is assumed."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Empty template for qualitative review, linked to actual failed checkpoints.
    review_rows = [{"case_id": row["case_id"], "condition": row["condition"], "sample": row["sample"],
                    "repeated_strategy": "", "goal_progress": "", "card_discloses_failed_values": "", "notes": ""}
                   for row in rows[:min(len(rows), 30)]]
    review_path = out / "manual_review.csv"
    if not review_path.exists():
        write_csv(review_path, review_rows)
    return {"report": str(out / "report.md"), **stats}

