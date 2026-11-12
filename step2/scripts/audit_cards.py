"""Counterfactual error-surface audit over the full development checkpoint set."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import asdict
from pe_feedback.bridge import parse_action, trajectory_step
from pe_feedback.common import path, read_rows, write_json
from pe_feedback.experiment import replay
from pe_feedback.feedback import FeedbackEngine


def mutate(value):
    if isinstance(value, str):
        return "UNVERIFIED_ERROR_VALUE"
    if isinstance(value, list):
        return [mutate(v) for v in value]
    return value


records = []
for case in read_rows(path("../step1/data/pilot.jsonl")):
    env, task, steps = replay(case)
    action = parse_action(case["failed_action"])
    parts = [repr(mutate(v)) for v in action.positional]
    parts += [f"{k}={mutate(v)!r}" for k, v in action.kwargs.items()]
    variant = f"{action.name}({', '.join(parts)})"
    before = asdict(env.world(task))
    outcome = env.step(task, variant)
    if outcome.ok or outcome.error_type != steps[-1].error_type or before != asdict(env.world(task)):
        raise RuntimeError(f"Perturbation changed the failure cause/state: {case['case_id']}")
    altered = steps[:-1] + [trajectory_step(variant, outcome)]
    for mode in ("facts_only", "grounded", "adaptive"):
        a, _ = FeedbackEngine().issue(task.goal, steps, mode)
        b, _ = FeedbackEngine().issue(task.goal, altered, mode)
        if a != b:
            raise RuntimeError(f"Error-value leakage: {case['case_id']} {mode}")
    known, _ = FeedbackEngine().issue(task.goal, steps, "grounded")
    unknown, _ = FeedbackEngine().issue(task.goal, [steps[-1]], "grounded")
    records.append({"case_id": case["case_id"], "error_type_preserved": True, "world_unchanged": True,
                    "three_cards_identical": True,
                    "next_tool_with_evidence": known["next_tool"], "next_tool_without_evidence": unknown["next_tool"]})
summary = {"cases": len(records), "error_surface_comparisons": 3 * len(records), "all_passed": True,
           "evidence_changed_next_tool": sum(r["next_tool_with_evidence"] != r["next_tool_without_evidence"] for r in records),
           "scope": "Fixed public goal/successful prefix/error class, altered failed argument values; not a semantic non-reconstructability proof",
           "records": records}
write_json(path("results/mechanism_audit.json"), summary)
print({k: v for k, v in summary.items() if k != "records"})
