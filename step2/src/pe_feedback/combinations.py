"""Keep the fact block fixed, append count/diagnosis/unchanged old guidance."""
from .bridge import HARNESSES, build_messages
from pinkelephant.feedback import correction_card, render_card

COMBO_MODES = ("facts_control", "facts_retry", "facts_diagnosis", "facts_positive")
FACT_BLOCK_BUDGET = 256


def compose_combination(system, goal, steps, mode, engine, measure, total_budget):
    base, evidence = engine.issue(goal, steps, "facts_only", measure, min(FACT_BLOCK_BUDGET, total_budget))
    msgs = build_messages(system, goal, steps, HARNESSES["drop" if evidence["supported"] else "abstract"])
    if base is None:
        return msgs, None, evidence
    legacy = correction_card(steps, use_history=True)
    if legacy is None:
        raise RuntimeError("Shared failure trigger disagrees with legacy guidance")
    original = render_card(legacy)
    header = original.split("\n", 1)[0]
    addition = {
        "facts_control": "",
        "facts_retry": f"Retry index: {legacy['matching_failures']}.",
        "facts_diagnosis": header,
        "facts_positive": original,
    }[mode]
    text = base["text"] + ("\n" + addition if addition else "")
    if measure(text) > total_budget:
        raise RuntimeError("Combination exceeds the registered total budget; fact block is never silently altered")
    card = {**base, "schema_version": "step2-combination-v1", "mode": mode,
            "text": text, "facts_text": base["text"], "addition": addition,
            "legacy_card": legacy if mode == "facts_positive" else None,
            "matching_count": legacy["matching_failures"], "measured_units": measure(text),
            "fact_block_budget": FACT_BLOCK_BUDGET}
    msgs.append({"role": "user", "content": text})
    return msgs, card, evidence
