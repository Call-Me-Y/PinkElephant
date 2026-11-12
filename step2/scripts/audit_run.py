"""Check logged fact origins and feedback lengths without loading model weights."""
from pathlib import Path
import sys
import os
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from tokenizers import Tokenizer
from pe_feedback.common import path, read_json, read_rows, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="results/development-v1")
args = parser.parse_args()
out = path(args.output)
manifest = read_json(out / "manifest.json")
cfg = manifest["config"]
model_dir = Path(manifest["resolved_actor"].get("model_path", ""))
tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json")) if model_dir.is_dir() else None
cards = facts = maximum = 0
costs = {}
for row in read_rows(out / "trajectories.jsonl"):
    cost = costs.setdefault(row["condition"], {"actor_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "missing_usage_calls": 0})
    history = list(row["initial_steps"])
    for turn in row["turns"]:
        response = turn["response"]
        cost["actor_calls"] += 1
        for name in ("prompt_tokens", "completion_tokens"):
            if response.get(name) is not None:
                cost[name] += response[name]
        if response.get("prompt_tokens") is None or response.get("completion_tokens") is None:
            cost["missing_usage_calls"] += 1
        card = turn["card"]
        if card and card.get("schema_version") == "step2-v1":
            cards += 1
            length = len(tokenizer.encode(card["text"], add_special_tokens=False).ids) if tokenizer else len(card["text"].encode("utf-8"))
            budget = cfg["max_feedback_tokens"] if tokenizer else cfg["max_feedback_bytes"]
            if length != card["measured_units"] or length > budget:
                raise RuntimeError("Feedback length or registered budget mismatch")
            maximum = max(maximum, length)
            for fact in card["verified_facts"]:
                facts += 1
                if fact["source_kind"] == "user_goal":
                    if fact["source_id"] != "goal":
                        raise RuntimeError("Goal fact has invalid source")
                elif fact["source_kind"] == "tool_success":
                    position = int(fact["source_id"].split(":")[1]) - 1
                    if not 0 <= position < len(history) or not history[position]["ok"]:
                        raise RuntimeError("Feedback fact cites failed or nonexistent observation")
                else:
                    raise RuntimeError("Unregistered fact source")
        history.append(turn["executed_step"])
summary = {"cards_checked": cards, "fact_source_links_checked": facts, "max_feedback_units": maximum,
           "all_source_links_valid": True, "all_feedback_within_budget": True,
           "scope": "Source links and token budget, not full semantic fact verification", "costs": costs}
write_json(out / "feedback_audit.json", summary)
print(summary)
