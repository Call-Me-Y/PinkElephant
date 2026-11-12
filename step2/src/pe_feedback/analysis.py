import csv
from collections import defaultdict
import numpy as np

from .common import read_json, read_rows, write_json
from .bridge import Task, canonical, make_env, strict_score

METRICS = ("strict_success", "upstream_success", "first_action_valid", "any_failure_repeat", "n_steps", "truncated_any")


def interval(groups):
    a = np.array([np.mean(v) for _, v in sorted(groups.items())])
    if len(a) < 2:
        return float(a.mean()), None, None
    rng = np.random.default_rng(42)
    samples = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(a.mean()), float(lo), float(hi)


def save_csv(p, rows):
    if not rows:
        return
    with p.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(out):
    manifest = read_json(out / "manifest.json")
    rows = read_rows(out / "trajectories.jsonl")
    conditions = manifest["config"]["conditions"]
    expected = {(c, m) for c in manifest["case_ids"] for m in conditions}
    index = {(r["case_id"], r["condition"]): r for r in rows}
    if set(index) != expected or len(index) != len(rows) or not (out / "completion.json").exists():
        raise RuntimeError("Batch incomplete; no final paired summary generated")
    for row in rows:
        env, task = make_env(), Task(**row["task"])
        env.reset(task)
        for step in row["initial_steps"] + [t["executed_step"] for t in row["turns"]]:
            outcome = env.step(task, step["action"])
            if outcome.ok != step["ok"] or outcome.observation != step["observation"]:
                raise RuntimeError("Independent environment replay mismatch")
        if int(strict_score(env, task)) != row["metrics"]["strict_success"]:
            raise RuntimeError("Strict score replay mismatch")
    summary, contrasts = [], []
    for condition in conditions:
        selected = [r for r in rows if r["condition"] == condition]
        for metric in METRICS:
            groups = defaultdict(list)
            for row in selected:
                groups[row["cluster_id"]].append(row["metrics"][metric])
            mean, lo, hi = interval(groups)
            summary.append({"condition": condition, "metric": metric, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                            "task_clusters": len(groups), "trajectories": len(selected),
                            "sum": sum(r["metrics"][metric] for r in selected)})
    comparisons = [(m, "abstract") for m in conditions if m != "abstract"]
    if {"adaptive", "grounded"} <= set(conditions):
        comparisons.append(("adaptive", "grounded"))
    if {"grounded", "facts_only"} <= set(conditions):
        comparisons.append(("grounded", "facts_only"))
    for pair in (("facts_retry", "facts_control"), ("facts_diagnosis", "facts_retry"),
                 ("facts_positive", "facts_control"), ("facts_positive", "facts_retry"),
                 ("facts_positive", "facts_diagnosis"), ("facts_positive", "positive_history")):
        if set(pair) <= set(conditions):
            comparisons.append(pair)
    for treatment, baseline in comparisons:
        for metric in METRICS:
            groups = defaultdict(list)
            for case_id in manifest["case_ids"]:
                a, b = index[(case_id, treatment)], index[(case_id, baseline)]
                groups[a["cluster_id"]].append(a["metrics"][metric] - b["metrics"][metric])
            mean, lo, hi = interval(groups)
            contrasts.append({"treatment": treatment, "baseline": baseline, "metric": metric,
                              "difference": mean, "ci_lo": lo, "ci_hi": hi, "task_clusters": len(groups)})
    save_csv(out / "summary.csv", summary)
    save_csv(out / "paired_contrasts.csv", contrasts)
    lines = ["# Step2 开发实验", "", "主指标为预先固定的目标核对通过；本轮按该判据停止，和step1仅事后核对的运行不可直接当作同一批数据。",
             f"完整验证{len(rows)}条轨迹；按任务聚类bootstrap 10,000次。单任务检查不产生可靠置信区间。", "",
             "| 条件 | 目标核对通过 | 上游判据通过 | 首步有效 | 任意失败调用重复 | 任务数 |", "|---|---:|---:|---:|---:|---:|"]
    for condition in conditions:
        cells = {r["metric"]: r for r in summary if r["condition"] == condition}
        values = [f"{cells[key]['mean']:.1%} ({int(cells[key]['sum'])}/{cells[key]['trajectories']})" for key in METRICS[:4]]
        lines.append(f"| {condition} | " + " | ".join(values) + f" | {cells['strict_success']['task_clusters']} |")
    lines += ["", "所有条件共享工具失败触发；新监控器只读取用户目标、公共schema和执行轨迹，未读取隐藏payload或参考动作。",
              "这是开发/连通检查；当前规则仅支持ToolShed六种目标表达。尚未实现LLM建议强基线、学习式选择器或Sentry/AppWorld集成。",
              "事实与子目标来自显式用户目标和工具契约，是规则提供的领域知识；不能将其收益全归因于反馈措辞。",
              "调用成功不清空未完成要求；但阶段一只在工具失败时发卡片，合法调用但目标偏离的共享触发器留待后续对照。",
              "完整调用、卡片、来源和原始模型文本保存在trajectories.jsonl；差值区间见paired_contrasts.csv。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "validation.json", {"complete": True, "all_actions_replayed": True, "trajectories": len(rows),
               "task_clusters": len({r["cluster_id"] for r in rows})})
    return {"report": str(out / "report.md"), "trajectories": len(rows)}
