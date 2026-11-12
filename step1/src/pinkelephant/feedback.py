"""Monitor sees executed steps and public schemas, never a gold continuation."""
from __future__ import annotations

from .ftb import HARNESSES, TOOLS_BY_NAME, build_messages, parse_action

CONDITIONS = ("verbatim", "abstract", "positive_current", "positive_history", "verbatim_positive")

ADVICE = {
    "parse_error": "Express the next action as one complete Python-style tool call with keyword arguments.",
    "unknown_tool": "Choose a tool name from the available tool documentation and use its declared arguments.",
    "missing_argument": "Supply every required field in the chosen tool's signature, using the request and observed results.",
    "unexpected_argument": "Use argument names exactly as declared in the chosen tool's signature.",
    "arity_error": "Bind each declared argument once, using the keyword form shown in the tool documentation.",
    "type_error": "Match each value to its declared type; use lists for list fields and numbers for numeric fields.",
    "format_error": "Match the documented field formats: dates YYYY-MM-DD, times HH:MM, and recipient email addresses when required.",
    "permission_error": "Select a writable target consistent with the user's request and the observed workspace.",
    "state_error": "Inspect the current target state and choose a step that advances the remaining request.",
    "unsupported_conversion": "Select a unit pair supported by the conversion tool's documentation.",
}


def active_failures(steps):
    """Current unresolved window, cleared by a successful tool call."""
    start = 0
    for index, step in enumerate(steps):
        if step.ok:
            start = index + 1
    return [step for step in steps[start:] if not step.ok]


def correction_card(steps, use_history):
    failures = active_failures(steps)
    if not failures:
        return None
    latest = failures[-1]
    family = latest.error_type or "unknown_error"
    tool = latest.tool if latest.tool in TOOLS_BY_NAME else "documented tool"
    count = sum((s.error_type, s.tool) == (latest.error_type, latest.tool) for s in failures) if use_history else 1
    guidance = ADVICE.get(family, "Use a lookup tool to establish the target from observed results, then continue the user's request.")
    required = []
    if family == "missing_argument" and tool in TOOLS_BY_NAME:
        try:
            parsed = parse_action(latest.action)
            supplied = set(parsed.kwargs) | {arg.name for arg in TOOLS_BY_NAME[tool].args[:len(parsed.positional)]}
            required = [arg.name for arg in TOOLS_BY_NAME[tool].args if arg.required and arg.name not in supplied]
        except Exception:
            pass
    if required:
        guidance += " Required fields to complete: " + ", ".join(required) + "."
    if count >= 2:
        guidance += " First reconcile the required fields and their formats with the tool signature; use a lookup if a required value is still unknown."
    return {
        "schema_version": "1.0", "tool": tool, "error_family": family,
        "matching_failures": count, "guidance_level": "check_then_act" if count >= 2 else "repair",
        "positive_guidance": guidance, "evidence_source": "executed_errors_and_public_tool_schema",
    }


def render_card(card):
    return (f"Recovery guidance | tool: {card['tool']} | category: {card['error_family']} | "
            f"matching failures: {card['matching_failures']}\n{card['positive_guidance']}")


def render(env, task, steps, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition in ("verbatim", "abstract"):
        return build_messages(env.system_prompt(task), task.goal, steps, HARNESSES[condition]), None
    # The current/history variants have identical visible successful history.
    # Earlier raw failed turns and observations are removed as one unit.
    policy = "verbatim" if condition == "verbatim_positive" else "drop"
    messages = build_messages(env.system_prompt(task), task.goal, steps, HARNESSES[policy])
    card = correction_card(steps, use_history=condition != "positive_current")
    if card:
        messages.append({"role": "user", "content": render_card(card)})
    return messages, card


def literal_leak_audit(steps, card):
    """Only audit the newly injected card. Schema words/tool names are allowed.

    This function conservatively flags overlapping string values of length >=4,
    including values also present in the goal or successful history. A flag is a
    review item, not proof of semantic disclosure. Zero is not a semantic proof.
    """
    if not card:
        return None
    text = render_card(card)
    failed = [step for step in steps if not step.ok]
    full = any(step.action.strip() and step.action.strip() in text for step in failed)
    values = []
    for step in failed:
        try:
            parsed = parse_action(step.action)
        except Exception:
            continue
        def strings(value):
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)):
                return [v for item in value for v in strings(item)]
            return []
        for value in list(parsed.kwargs.values()) + list(parsed.positional):
            values.extend(strings(value))
    return {"full_failed_call_present": full,
            "argument_string_overlap": sorted({value for value in values if len(value) >= 4 and value in text}),
            "scope": "card_only_literal_check_not_semantic_reconstruction"}

