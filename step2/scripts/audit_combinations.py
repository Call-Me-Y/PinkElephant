"""Audit exact append semantics, original fact block, budgets, and context reuse."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from collections import defaultdict
from tokenizers import Tokenizer
from pe_feedback.common import path, read_json, read_rows, write_json, digest
from pe_feedback.feedback import FeedbackEngine
from pinkelephant.feedback import correction_card, render_card
from slmecho.harness import TrajStep

out = path("results/facts-positive-v1")
manifest = read_json(out / "manifest.json")
tok = Tokenizer.from_file(str(Path(manifest["resolved_actor"]["model_path"]) / "tokenizer.json"))
measure = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
rows = read_rows(out / "trajectories.jsonl")
counts = defaultdict(lambda: {"trajectories": 0, "repeated_context_cases": 0, "cards": 0, "max_tokens": 0, "total_feedback_tokens": 0})
for row in rows:
    entry = counts[row["condition"]]
    entry["trajectories"] += 1
    hashes = [digest(turn["messages"]) for turn in row["turns"]]
    entry["repeated_context_cases"] += len(set(hashes)) < len(hashes)
    steps = [TrajStep(**s) for s in row["initial_steps"]]
    for turn in row["turns"]:
        card = turn["card"]
        if card and card.get("schema_version") == "step2-combination-v1":
            base, _ = FeedbackEngine().issue(row["task"]["goal"], steps, "facts_only", measure, 256)
            if base["text"] != card["facts_text"]:
                raise RuntimeError("Fact block changed")
            legacy = correction_card(steps, True)
            expected_addition = {
                "facts_control": "", "facts_retry": f"Retry index: {legacy['matching_failures']}.",
                "facts_diagnosis": render_card(legacy).split("\n", 1)[0], "facts_positive": render_card(legacy),
            }[row["condition"]]
            if card["addition"] != expected_addition or card["text"] != base["text"] + ("\n" + expected_addition if expected_addition else ""):
                raise RuntimeError("Appended text differs from registered intervention")
            length = measure(card["text"])
            if length != card["measured_units"] or length > 384:
                raise RuntimeError("Budget mismatch")
            entry["cards"] += 1
            entry["max_tokens"] = max(entry["max_tokens"], length)
            entry["total_feedback_tokens"] += length
        steps.append(TrajStep(**turn["executed_step"]))
# Check re-run anchors against the earlier batch, rather than silently borrowing
# their scores. All conditions in the new table are freshly generated.
old = read_rows(path("results/development-v1/trajectories.jsonl"))
old_index = {(r["case_id"], r["condition"]): r for r in old}
anchors = {}
for new_name, old_name in (("facts_control", "facts_only"), ("abstract", "abstract"), ("positive_history", "positive_history")):
    selected = [r for r in rows if r["condition"] == new_name]
    anchors[new_name] = {"cases": len(selected),
        "same_metrics": sum(r["metrics"] == old_index[(r["case_id"], old_name)]["metrics"] for r in selected),
        "same_executed_actions": sum([t["executed_step"]["action"] for t in r["turns"]] == [t["executed_step"]["action"] for t in old_index[(r["case_id"], old_name)]["turns"]] for r in selected)}
result = {"all_append_checks_passed": True, "conditions": dict(counts), "rerun_anchors": anchors,
          "scope": "All comparisons use new runs; text length is observed, not matched."}
write_json(out / "combination_audit.json", result)
print(result)
