from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from pe_feedback.common import path, read_json, read_rows, write_json


def main():
    parser = argparse.ArgumentParser(description="Step2 evidence-grounded feedback experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--data", default="../step1/data/pilot.jsonl")
    demo.add_argument("--output", default="results/cards")
    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/development.json")
    run.add_argument("--limit", type=int)
    run.add_argument("--output")
    collect = sub.add_parser("collect")
    collect.add_argument("--source", required=True)
    collect.add_argument("--output", default="data/natural_dev.jsonl")
    collect.add_argument("--condition", default="abstract")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        from pe_feedback.experiment import replay
        from pe_feedback.feedback import FeedbackEngine
        cards = []
        for case in read_rows(path(args.data)):
            env, task, steps = replay(case)
            for mode in ("facts_only", "grounded", "adaptive"):
                card, evidence = FeedbackEngine().issue(task.goal, steps, mode)
                cards.append({"case_id": case["case_id"], "mode": mode, "card": card, "evidence": evidence})
        target = path(args.output)
        write_json(target / "cards.json", cards)
        lines = ["# Step2 开发集纠错卡片", "", "规则生成示例，不是模型效果实验。", ""]
        for row in cards:
            if row["mode"] == "grounded":
                lines += [f"## {row['case_id']}", "", row["card"]["text"] if row["card"] else "Unsupported public goal or no trigger", ""]
        (target / "cards.md").write_text("\n".join(lines), encoding="utf-8")
        result = {"cards": len(cards), "output": str(target)}
    elif args.command == "run":
        from pe_feedback.experiment import run
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        result = run(read_json(path(args.config)), args.limit, args.output)
    elif args.command == "collect":
        from pe_feedback.experiment import collect
        result = collect(path(args.source), args.output, args.condition)
    else:
        from pe_feedback.analysis import analyze
        result = analyze(path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

