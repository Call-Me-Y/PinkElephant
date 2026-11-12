"""Compile public evidence into feedback; no environment or oracle access."""
from dataclasses import asdict
import json

from .bridge import HARNESSES, build_messages
from .evidence import Evidence
from .common import digest
from pinkelephant.feedback import correction_card, render_card

MODES = ("drop", "abstract", "positive_history", "facts_only", "grounded", "adaptive",
         "facts_control", "facts_retry", "facts_diagnosis", "facts_positive")


class FeedbackEngine:
    def __init__(self):
        self.previous = None

    def issue(self, goal, steps, mode, measure=len, budget=1600):
        evidence = Evidence(goal, steps)
        public = evidence.snapshot()
        if evidence.goal.kind == "unsupported":
            return None, public
        # Phase 1: the same failure-only trigger for every feedback condition.
        if not steps or steps[-1].ok or evidence.completed:
            return None, public
        target_tool, subgoal = evidence.next_step()
        status = "no_previous_hint"
        if self.previous and len(steps) > self.previous["history_length"]:
            following = steps[self.previous["history_length"]:]
            current_keys = {key for key, fact in evidence.facts.items() if fact.active and not key.startswith("goal.")}
            if current_keys - set(self.previous["fact_keys"]) or evidence.completed and not self.previous["completed"] or evidence.deleted and not self.previous["deleted"]:
                status = "observable_progress"
            elif following[0].tool != self.previous["target_tool"]:
                status = "requested_step_not_observed"
            else:
                status = "requested_step_without_observable_progress"
        facts = [asdict(f) for f in evidence.facts.values() if f.active and f.key != "completion_receipt"]
        kind = evidence.goal.kind
        requirements = {
            "email_file": "Requested recipient, exact subject, and full source-file text sent together.",
            "schedule": "Event with requested title, date, time, and verified attendee.",
            "cancel_event": "The uniquely identified requested event is cancelled.",
            "roster": "All verified full names, comma separated, stored at the requested path.",
            "archive": "Requested file deleted, followed by reminder with exact date and text.",
            "convert": "Verified conversion result alone stored at the requested output path.",
        }[kind]
        diagnosis = steps[-1].error_type or "execution_failure"
        lines = [f"Request requirements: {requirements}"]
        for fact in facts:
            value = json.dumps(fact["value"], ensure_ascii=False)
            lines.append(f"{fact['key']} = {value} [source {fact['source_id']}]")
        if mode != "facts_only":
            lines += [f"Diagnosis: {diagnosis}.", f"Next subgoal: {subgoal}",
                      "Check completion against the stated request requirements and successful tool results."]
        if mode == "adaptive" and status != "no_previous_hint":
            if status == "observable_progress":
                lines.append("Retain the verified progress and address the remaining subgoal.")
            elif status == "requested_step_not_observed":
                lines.append("Focus the next step on the stated subgoal before moving to another operation.")
            else:
                lines.append("Establish successful evidence for this requirement before proceeding.")
        text = "\n".join(lines)
        budget_omissions = []
        # Keep exact evidence values whole. Large values become source pointers,
        # never a truncated value that could be mistaken for the full content.
        for fact in sorted(facts, key=lambda f: len(json.dumps(f["value"])), reverse=True):
            if measure(text) <= budget:
                break
            original = f"{fact['key']} = {json.dumps(fact['value'], ensure_ascii=False)} [source {fact['source_id']}]"
            replacement = f"{fact['key']}: use the full value in {fact['source_id']}."
            if measure(replacement) < measure(original):
                text = text.replace(original, replacement)
                budget_omissions.append(fact["key"])
        if measure(text) > budget:
            raise ValueError("Feedback budget too small; choose a new registered budget, without silent truncation")
        card = {"schema_version": "step2-v1", "mode": mode, "text": text,
                "verified_facts": facts, "remaining_requirement": requirements,
                "next_tool": target_tool if mode != "facts_only" else None,
                "next_subgoal": subgoal if mode != "facts_only" else None,
                "previous_hint_outcome": status if mode == "adaptive" else None,
                "budget_omissions": budget_omissions, "measured_units": measure(text)}
        if mode == "adaptive":
            self.previous = {"history_length": len(steps), "target_tool": target_tool,
                             "fact_keys": [key for key, fact in evidence.facts.items() if fact.active and not key.startswith("goal.")],
                             "completed": evidence.completed, "deleted": evidence.deleted}
        return card, public


def compose(system_prompt, goal, steps, mode, engine, measure=len, budget=1600):
    if mode not in MODES:
        raise ValueError(mode)
    if mode in {"facts_control", "facts_retry", "facts_diagnosis", "facts_positive"}:
        from .combinations import compose_combination
        return compose_combination(system_prompt, goal, steps, mode, engine, measure, budget)
    if mode in {"drop", "abstract"}:
        return build_messages(system_prompt, goal, steps, HARNESSES[mode]), None, None
    if mode == "positive_history":
        msgs = build_messages(system_prompt, goal, steps, HARNESSES["drop"])
        card = correction_card(steps, use_history=True)
        if card:
            msgs.append({"role": "user", "content": render_card(card)})
        return msgs, card, None
    card, evidence = engine.issue(goal, steps, mode, measure, budget)
    policy = "drop" if evidence["supported"] else "abstract"
    msgs = build_messages(system_prompt, goal, steps, HARNESSES[policy])
    if card:
        msgs.append({"role": "user", "content": card["text"]})
    return msgs, card, evidence

