"""Validate the complete paired probe and summarize by independent task."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from collections import defaultdict
import csv
import json
import numpy as np

from pinkelephant.common import digest, project_path, read_json, read_jsonl, write_json

METRICS = ("failed_logp", "gain_vs_clean", "gold_minus_fail_logp", "p_repeat_candidate_set", "gold_top_candidate", "fail_top_candidate")
COMPARISONS = (("verbatim", "pre_clean"), ("abstract", "verbatim"),
               ("positive_history", "verbatim"), ("positive_history", "abstract"),
               ("positive_history", "positive_current"), ("positive_history", "verbatim_positive"))


def interval(groups):
    values = np.array([np.mean(v) for _, v in sorted(groups.items())], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite clustered observations")
    rng = np.random.default_rng(42)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi), len(values)


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/probability-pilot-cuda-v1")
    args = parser.parse_args()
    out = project_path(args.output)
    manifest = read_json(out / "manifest.json")
    cases = read_jsonl(project_path(manifest["config"]["dataset_path"]))
    if manifest["limit"]:
        cases = cases[:manifest["limit"]]
    conditions = manifest["conditions"]
    raw = read_jsonl(out / "scores.jsonl")
    index = {(r["case_id"], r["condition"]): r for r in raw}
    expected = {(c["case_id"], condition) for c in cases for condition in conditions}
    if len(index) != len(raw) or set(index) != expected:
        raise RuntimeError(f"Incomplete/duplicate scores: have {len(raw)}, expected {len(expected)}")
    metrics = list(csv.DictReader((out / "metrics.csv").open(encoding="utf-8-sig")))
    metric_index = {(r["case_id"], r["condition"]): r for r in metrics}
    if len(metrics) != len(raw) or set(metric_index) != expected:
        raise RuntimeError("Derived metrics do not match raw records")
    checks = []
    rows = []
    for case in cases:
        paired = [index[(case["case_id"], c)] for c in conditions]
        if len({digest(r["candidates"]) for r in paired}) != 1:
            raise RuntimeError("Candidate set changes between conditions")
        for r in paired:
            item = dict(metric_index[(r["case_id"], r["condition"])])
            for key in METRICS[:4]:
                item[key] = float(item[key])
            logps = [s["logp"] for s in r["scores"]]
            if not np.isfinite(logps).all() or max(logps) > 0:
                raise RuntimeError("Invalid candidate log probabilities")
            item["gold_top_candidate"] = int(np.argmax(logps) == 1)
            item["fail_top_candidate"] = int(np.argmax(logps) == 0)
            item["history_depth"] = int(item["history_depth"])
            item["error_family"] = case["error_family"]
            rows.append(item)
        if case["history_depth"] == 1:
            a = index[(case["case_id"], "positive_current")]
            b = index[(case["case_id"], "positive_history")]
            equal = a["messages"] == b["messages"]
            delta = max(abs(x["logp"] - y["logp"]) for x, y in zip(a["scores"], b["scores"]))
            checks.append({"case_id": case["case_id"], "same_messages": equal, "max_logp_difference": delta})
            if not equal or delta > 1e-4:
                raise RuntimeError("History=1 control mismatch")
    summary, contrasts, subgroups = [], [], []
    for depth in ("all", 1, 2):
        selected = [r for r in rows if depth == "all" or r["history_depth"] == depth]
        for condition in conditions:
            for metric in METRICS:
                groups = defaultdict(list)
                for r in selected:
                    if r["condition"] == condition:
                        groups[r["cluster_id"]].append(r[metric])
                mean, lo, hi, n = interval(groups)
                summary.append({"history_depth": depth, "condition": condition, "metric": metric,
                                "mean": mean, "ci_lo": lo, "ci_hi": hi, "task_clusters": n})
        for treatment, baseline in COMPARISONS:
            a = {r["case_id"]: r for r in selected if r["condition"] == treatment}
            b = {r["case_id"]: r for r in selected if r["condition"] == baseline}
            for metric in METRICS:
                groups = defaultdict(list)
                for key in sorted(a):
                    groups[a[key]["cluster_id"]].append(a[key][metric] - b[key][metric])
                mean, lo, hi, n = interval(groups)
                contrasts.append({"history_depth": depth, "treatment": treatment, "baseline": baseline,
                                  "metric": metric, "difference": mean, "ci_lo": lo, "ci_hi": hi, "task_clusters": n})
    for family in sorted({r["error_family"] for r in rows}):
        for metric in ("gold_minus_fail_logp", "p_repeat_candidate_set"):
            selected = [r for r in rows if r["history_depth"] == 2 and r["error_family"] == family]
            a = {r["case_id"]: r for r in selected if r["condition"] == "positive_history"}
            b = {r["case_id"]: r for r in selected if r["condition"] == "abstract"}
            groups = {a[key]["cluster_id"]: [a[key][metric] - b[key][metric]] for key in a}
            mean, lo, hi, n = interval(groups)
            subgroups.append({"error_family": family, "metric": metric, "difference": mean,
                              "ci_lo": lo, "ci_hi": hi, "task_clusters": n})
    csv_write(out / "summary.csv", summary)
    csv_write(out / "paired_contrasts.csv", contrasts)
    csv_write(out / "error_family_contrasts.csv", subgroups)
    write_json(out / "validation.json", {"complete": True, "cases": len(cases),
               "task_clusters": len({c["cluster_id"] for c in cases}), "contexts": len(raw),
               "candidate_scores": sum(len(r["scores"]) for r in raw), "history1_identity_checks": checks,
               "note": "10,000 task-cluster bootstrap resamples; exploratory intervals without multiplicity adjustment"})
    lines = ["# 48个失败点的概率实验", "",
             "模型：Qwen2.5-0.5B-Instruct；本批次使用GPU BF16权重、FP32 log_softmax。",
             "这是候选动作的教师强制评分，不是自由生成重复率或任务成功率。",
             f"完整性检查：{len(cases)}个失败点，{len({c['cluster_id'] for c in cases})}个任务，{len(raw)}组上下文。",
             "一次和两次失败是同一任务的相关变体。区间按任务聚类bootstrap，10,000次重采样。",
             "候选概率不含EOS；候选间长度不同；只解释固定候选下的条件差值。", ""]
    for depth in ("all", 1, 2):
        lines += [f"## 历史深度：{depth}", "", "| 条件 | 纠错增益 | 正确−失败 logP | 四候选失败份额 | 正确候选排名第一 |", "|---|---:|---:|---:|---:|"]
        for condition in conditions:
            cells = {r["metric"]: r["mean"] for r in summary if r["history_depth"] == depth and r["condition"] == condition}
            lines.append(f"| {condition} | {cells['gain_vs_clean']:.3f} | {cells['gold_minus_fail_logp']:.3f} | {cells['p_repeat_candidate_set']:.1%} | {cells['gold_top_candidate']:.1%} |")
        lines.append("")
    lines += ["## 两次历史失败的配对差值", "", "差值均为前者减后者；正确−失败分数越高越好，失败候选份额越低越好。", "",
              "| 前者 | 后者 | 指标 | 差值 | 95%区间 |", "|---|---|---|---:|---|"]
    for r in contrasts:
        if r["history_depth"] == 2 and r["metric"] in ("gold_minus_fail_logp", "p_repeat_candidate_set"):
            lines.append(f"| {r['treatment']} | {r['baseline']} | {r['metric']} | {r['difference']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |")
    lines += ["", "完整数值与各错误类型差值见同目录CSV文件。",
              "history=1 的两个正向条件已逐项检查输入相同、评分一致。",
              "pre_clean 在history=2仍表示两次失败前的干净上下文；其gain是整段历史的总影响。",
              "FTB abstract使用上游原版模板，其中包含that call is not repeatable措辞。",
              "本实验为pilot：单模型、三类人工错误，同一模板的24个任务；不能作为独立确认或行为成功证据。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out / "report.md"), "contexts": len(raw), "cases": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
