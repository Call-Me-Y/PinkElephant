"""Audit actual actions and report task-paired intervention effects."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
from collections import defaultdict
import csv
import json
import numpy as np
from pinkelephant.common import digest, project_path, read_json, read_jsonl, write_json
from pinkelephant.ftb import Task, make_env, parse_action, trajectory_step
from pinkelephant.experiment import metrics
from pinkelephant.feedback import render

METRICS = ("task_success", "audited_goal_success", "first_action_valid", "next_action_canonical_repeat", "any_target_repeat",
           "any_failure_repeat", "n_canonical_repeats", "first_same_error_family", "same_error_family_any",
           "truncated_any", "n_generated_failures", "n_generated_actions")


def interval(groups):
    values = np.array([np.mean(v) for _, v in sorted(groups.items())], dtype=float)
    rng = np.random.default_rng(42)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extra_goal_audit(world, task):
    """Post-hoc check of explicit requirements omitted by upstream predicates.

    This scores only the recorded end state; it does not rerun with a stricter
    stopping policy and is not an exhaustive side-effect audit.
    """
    p, template = task.payload, task.payload["template"]
    if template == "email_file":
        reference = next(parse_action(a).kwargs for a in task.gold_actions if a.startswith("send_message("))
        return any(m["to"] == p["contact_email"] and m["subject"] == reference["subject"] and p["file_content"] in m["body"] for m in world.sent)
    if template == "schedule":
        return any(not e["cancelled"] and e["title"] == "1:1" and e["date"] == p["date"] and e["time"] == p["time"] and p["contact_email"] in e["attendees"] for e in world.events)
    if template == "roster":
        content = world.files.get(p["out_path"], {}).get("content", "")
        return sorted(s.strip() for s in content.split(",")) == sorted(p["team_names"])
    if template == "archive":
        return p["path"] not in world.files and any(r["date"] == p["date"] and r["text"] == f"{p['path']} was archived" for r in world.reminders)
    if template == "convert":
        try:
            value = float(world.files[p["out_path"]]["content"].strip())
            return abs(value - p["expected"]) <= 0.00005
        except (KeyError, ValueError, TypeError):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/interventions-batched-fp16-v2")
    args = parser.parse_args()
    out = project_path(args.output)
    manifest = read_json(out / "manifest.json")
    cfg = manifest["config"]
    cases = {c["case_id"]: c for c in read_jsonl(project_path(cfg["evaluation_data"]))}
    rows = read_jsonl(out / "trajectories.jsonl")
    index = {(r["case_id"], r["arm"]): r for r in rows}
    expected = {(case_id, arm) for case_id in cases for arm in cfg["arms"]}
    if len(rows) != len(index) or set(index) != expected or not read_json(out / "completion.json")["complete"]:
        raise RuntimeError("Incomplete or duplicate batch")
    training_rows = read_jsonl(project_path(cfg["training_config"]["dataset"]["path"]))
    if digest(training_rows) != read_json(out / "training.json")["training_sha256"]:
        raise RuntimeError("Training data has changed")
    def prompt_hash(case):
        e, t = make_env(), Task(**case["task"])
        e.reset(t)
        steps = []
        for action in case["prefix_actions"] + [case["failed_action"]] * case["history_depth"]:
            steps.append(trajectory_step(action, e.step(t, action)))
        return digest(render(e, t, steps, "abstract")[0])
    train_hashes = {prompt_hash(case) for case in training_rows}
    eval_hashes = {prompt_hash(case) for case in cases.values()}
    overlap = train_hashes & eval_hashes
    write_json(out / "train_test_overlap_audit.json", {
        "exact_initial_abstract_prompt_overlap": len(overlap), "training_cases": len(training_rows),
        "evaluation_cases": len(cases), "identical_goal_strings": len({c['task']['goal'] for c in training_rows} & {c['task']['goal'] for c in cases.values()})})
    if overlap:
        raise RuntimeError("Exact train/test model inputs overlap")
    # Replay every generated action independently; stored outcomes and successes
    # must agree with the executable environment, with no LLM judge.
    for row in rows:
        row["metrics"]["any_failure_repeat"] = int(row["metrics"]["n_canonical_repeats"] > 0)
        case = cases[row["case_id"]]
        task = Task(**case["task"])
        env = make_env()
        env.reset(task)
        initial = []
        for action in case["prefix_actions"] + [case["failed_action"]] * case["history_depth"]:
            step = trajectory_step(action, env.step(task, action))
            if not step.ok:
                initial.append(step)
        generated = []
        for turn in row["turns"]:
            saved = turn["executed_step"]
            actual = trajectory_step(saved["action"], env.step(task, saved["action"]))
            if actual.ok != saved["ok"] or actual.error_type != saved["error_type"] or actual.observation != saved["observation"]:
                raise RuntimeError("Saved outcome disagrees with independent replay")
            generated.append(actual)
        checked = metrics(generated, initial, "controlled", env.is_solved(task))
        for key, value in checked.items():
            if value != row["metrics"][key]:
                raise RuntimeError(f"Metric disagreement: {key}")
        row["metrics"]["audited_goal_success"] = int(env.is_solved(task) and extra_goal_audit(env.world(task), task))
    summary, contrasts, families = [], [], []
    comparisons = [(arm, "abstract") for arm in cfg["arms"] if arm != "abstract"]
    comparisons += [("ul_head", "sft_head"), ("local_constraints", "abstract_ban")]
    for depth in ("all", 1, 2):
        selected = [r for r in rows if depth == "all" or r["history_depth"] == depth]
        for arm in cfg["arms"]:
            matching = [r for r in selected if r["arm"] == arm]
            for metric in METRICS:
                groups = defaultdict(list)
                for row in matching:
                    groups[row["cluster_id"]].append(row["metrics"][metric])
                mean, lo, hi = interval(groups)
                summary.append({"history_depth": depth, "arm": arm, "metric": metric, "mean": mean,
                                "ci_lo": lo, "ci_hi": hi, "task_clusters": len(groups),
                                "numerator": sum(r["metrics"][metric] for r in matching), "trajectories": len(matching)})
        for treatment, baseline in comparisons:
            for metric in METRICS:
                groups = defaultdict(list)
                for case_id in cases:
                    a, b = index[(case_id, treatment)], index[(case_id, baseline)]
                    if depth != "all" and a["history_depth"] != depth:
                        continue
                    groups[a["cluster_id"]].append(a["metrics"][metric] - b["metrics"][metric])
                mean, lo, hi = interval(groups)
                contrasts.append({"history_depth": depth, "treatment": treatment, "baseline": baseline,
                                  "metric": metric, "difference": mean, "ci_lo": lo, "ci_hi": hi,
                                  "task_clusters": len(groups)})
    for family in sorted({r["error_family"] for r in rows}):
        for arm in cfg["arms"]:
            selected = [r for r in rows if r["error_family"] == family and r["arm"] == arm]
            for metric in ("task_success", "same_error_family_any", "any_target_repeat"):
                groups = defaultdict(list)
                for row in selected:
                    groups[row["cluster_id"]].append(row["metrics"][metric])
                mean, lo, hi = interval(groups)
                families.append({"family": family, "arm": arm, "metric": metric, "mean": mean,
                                 "ci_lo": lo, "ci_hi": hi, "task_clusters": len(groups)})
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_contrasts.csv", contrasts)
    write_csv(out / "error_families.csv", families)
    costs = []
    for arm in cfg["arms"]:
        subset = [r for r in rows if r["arm"] == arm]
        turns = [t for r in subset for t in r["turns"]]
        costs.append({"arm": arm, "model_turns": len(turns), "generated_tokens": sum(t["completion_tokens"] for t in turns),
                      "amortized_compute_seconds": sum(r["seconds"] for r in subset),
                      "forward_branches_per_token": max(t["forward_branches"] for t in turns),
                      "masked_token_attempts": sum(t["masked_token_attempts"] for t in turns)})
    write_csv(out / "costs.csv", costs)
    error_rows = []
    for arm in cfg["arms"]:
        counts = defaultdict(int)
        subset = [r for r in rows if r["arm"] == arm]
        for row in subset:
            for turn in row["turns"]:
                counts[turn["executed_step"]["error_type"] or "valid_call"] += 1
        for error, count in sorted(counts.items()):
            error_rows.append({"arm": arm, "outcome": error, "calls": count,
                               "fraction_of_calls": count / sum(counts.values())})
    write_csv(out / "error_distribution.csv", error_rows)
    write_json(out / "validation.json", {"complete": True, "cases": len(cases), "task_clusters": len({c["cluster_id"] for c in cases.values()}),
               "arms": len(cfg["arms"]), "trajectories": len(rows), "all_actions_replayed": True,
               "generated_actions": sum(len(r["turns"]) for r in rows), "training": read_json(out / "training.json")})
    labels = {"drop": "无反馈/删除失败记录", "verbatim": "原始回显", "abstract": "FTB抽象",
              "positive_history": "历史正向提示", "ftb_ban": "原始回显＋FTB字符串禁用",
              "abstract_ban": "抽象＋FTB字符串禁用", "local_constraints": "抽象＋局部约束",
              "contrast_025": "对比解码 λ=0.25", "contrast_05": "对比解码 λ=0.5", "contrast_1": "对比解码 λ=1",
              "sft_head": "低秩输出头：仅SFT", "ul_head": "低秩输出头：SFT＋UL"}
    lines = ["# 48个失败点的实际生成干预实验", "",
             f"模型Qwen2.5-0.5B-Instruct；{len(cases)}个失败点，24个任务，{len(cfg['arms'])}个条件，共{len(rows)}条轨迹。",
             "每条最多三步，每步最多96 tokens，greedy解码；GPU FP16主干/FP32分数，batch=4，所有成功由环境判定，逐动作独立重放验证通过。",
             "这是实际生成与执行的结果，区别于之前四候选内的概率份额。",
             "两种历史深度不独立，区间按24个任务聚类bootstrap，10,000次；多比较未校正，属于探索性pilot。", ""]
    lookup = {(r["history_depth"], r["arm"], r["metric"]): r for r in summary}
    for depth in ("all", 1, 2):
        lines += [f"## 历史深度 {depth}", "", "| 方法 | 上游判据成功 | 补充目标核对通过 | 首步调用有效 | 原失败动作再次出现 | 任意失败动作重复 | 同工具同类错误再现 | 达到输出上限 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in cfg["arms"]:
            cells = [lookup[(depth, arm, m)] for m in ("task_success", "audited_goal_success", "first_action_valid", "any_target_repeat", "any_failure_repeat", "same_error_family_any", "truncated_any")]
            values = [f"{r['mean']:.1%} ({int(r['numerator'])}/{r['trajectories']})" for r in cells]
            lines.append("| " + labels[arm] + " | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## 成功率配对差值（两个历史深度合并，任务内平均）", "",
              "| 前者 | 后者 | 成功率差值 | 95%区间 |", "|---|---|---:|---|"]
    for row in contrasts:
        if row["history_depth"] == "all" and row["metric"] == "task_success":
            lines.append(f"| {labels[row['treatment']]} | {labels[row['baseline']]} | {row['difference']:+.1%} | [{row['ci_lo']:+.1%}, {row['ci_hi']:+.1%}] |")
    lines += ["", "## 实现边界", "",
              "局部约束只在未解决窗口中观测到对应工具的缺参数、参数名、格式错误后触发；成功调用后撤销。",
              "它利用公开schema，不读取测试参考动作；不保证正文非空、目标选择或任务语义正确。",
              "同工具同类错误是规则代理指标，不能覆盖所有语义相同的失败策略。",
              "对比解码以FTB抽象上下文为基础，扣除中性回显相对删除历史的正logP增量，并保留基础分布top-relative 0.1的合理候选。",
              "λ的三个值预先固定；它需要三份上下文/KV缓存，步数相同但计算量更高。",
              "训练为冻结0.5B主干、仅训练rank=8输出头适配器；不是完整LoRA或原论文UL复现。",
              "训练世界seed=20261114，48个训练任务、96个训练失败点，四轮；评测世界seed=20261113，24个任务。",
              "初始抽象反馈提示的训练/评测完全重合数为0；有重复的任务目标文字，因而不是模板或领域分布外测试。",
              "SFT与SFT＋UL使用相同初始适配器、顺序、正确动作监督；UL仅额外惩罚失败调用首个分歧token，权重1。",
              "两种训练条件都基于FTB抽象反馈解码，训练参考解只用于独立训练数据。",
              "本批48个失败点此前已用于探索，不能作新方案的独立确认。未评测9B、大模型API或自然发生的失败。",
              "适配器在BF16冻结主干上训练，完整评测统一使用FP16主干、FP32分数/适配器；所有方法使用同一精度和批量runner。",
              "数值检查要求9上下文首token一致、max logit差<0.2、下一token分布KL<0.001，以及8token缓存续写一致；实际偏差见batch_validation.json。",
              "CPU、早期串行及FP32性能诊断轨迹未混入本批；实际数值偏差见batch_validation.json。",
              "耗时是批量计算摊销值，不是单请求延迟；详细tokens与计算分支见costs.csv。",
              "生成调用的错误分布见error_distribution.csv。训练成本不包含在推理costs.csv内。",
              "补充目标核对是事后审计：上游邮件判据忽略主题、日程忽略标题、花名册只核对名字前半部分、提醒忽略文本、数值文件允许子串。",
              "审计增加邮件主题、日程标题、完整逗号分隔姓名、提醒原文、文件只包含数值等要求；不是独立预注册指标，也未全面核对副作用。",
              "运行仍按原上游成功判据停止；补充核对只评价已记录终态，不代表使用更严格停止策略重新运行后的结果。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out / "report.md"), "validated_trajectories": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
