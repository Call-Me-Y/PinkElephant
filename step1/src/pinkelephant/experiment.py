from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .backends import make_backend
from .common import PROJECT, append_jsonl, digest, project_path, read_json, read_jsonl, write_json
from .feedback import CONDITIONS, literal_leak_audit, render
from .ftb import Task, canonical, extract_action, make_env, revision, trajectory_step


def metrics(generated, initial_failures, mode, solved):
    seen_exact = {step.action.strip() for step in initial_failures}
    seen_canon = {canonical(step.action) for step in initial_failures}
    target = canonical(initial_failures[-1].action) if initial_failures else None
    exact = canon = failed = 0
    for step in generated:
        if step.ok:
            continue
        failed += 1
        exact += step.action.strip() in seen_exact
        canon += canonical(step.action) in seen_canon
        seen_exact.add(step.action.strip())
        seen_canon.add(canonical(step.action))
    return {
        "next_action_exact_repeat": int(generated[0].action.strip() in {s.action.strip() for s in initial_failures}) if mode == "controlled" and generated else None,
        "next_action_canonical_repeat": int(canonical(generated[0].action) == target) if mode == "controlled" and generated else None,
        "any_target_repeat": int(any(canonical(s.action) == target for s in generated)) if mode == "controlled" else None,
        "first_action_valid": int(generated[0].ok) if generated else None,
        "n_generated_actions": len(generated), "n_generated_failures": failed,
        "n_exact_repeats": exact, "n_canonical_repeats": canon,
        "ftb_exact_repeat_rate": exact / failed if failed else None,
        "ftb_canonical_repeat_rate": canon / failed if failed else None,
        "task_success": int(solved),
    }


def run(config, limit=None, output_override=None):
    if len(set(config["conditions"])) != len(config["conditions"]) or not set(config["conditions"]) <= set(CONDITIONS):
        raise ValueError("Unknown or duplicate condition")
    mode = config.get("mode", "controlled")
    if mode not in {"controlled", "free"}:
        raise ValueError(mode)
    cases = read_jsonl(project_path(config["dataset"]["path"]))
    if mode == "free":
        cases = list({case["cluster_id"]: case for case in cases}.values())
    if limit:
        cases = cases[:limit]
    out = project_path(output_override or config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # Fingerprint code/config/data/upstream so resume cannot mix revisions.
    code = {str(path.relative_to(PROJECT)): path.read_text(encoding="utf-8") for path in sorted((PROJECT / "src").rglob("*.py"))}
    resolved_backend = {**config["backend"]}
    if resolved_backend["kind"] == "chat_api":
        for field, env_name in (("model", "model_env"), ("base_url", "base_url_env")):
            key = resolved_backend.get(env_name, "PE_MODEL" if field == "model" else "PE_BASE_URL")
            if key in os.environ:
                resolved_backend[field] = os.environ[key]
        # URLs must contain no credentials or query parameters; never persist secrets.
        from urllib.parse import urlparse
        parsed = urlparse(resolved_backend.get("base_url", ""))
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use a base URL without credentials/query/fragment; store the key only in an environment variable")
    upstream = revision()
    signature = digest({"config": config, "resolved_backend": resolved_backend, "data": cases, "code": code, "upstream": upstream})
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path)["signature"] != signature:
            raise RuntimeError("Output folder belongs to a different config/data/code/model revision. Choose a new output folder.")
    else:
        write_json(manifest_path, {"signature": signature, "config": config, "resolved_backend": resolved_backend,
                                  "case_limit": limit, "upstream": upstream, "python": sys.version,
                                  "platform": platform.platform(), "created_utc": datetime.now(timezone.utc).isoformat(),
                                  "result_kind": "plumbing_only" if config["backend"]["kind"] == "smoke" else "model_experiment"})
    rows_path = out / "trajectories.jsonl"
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {row["unit_id"] for row in existing}
    attempt_path = out / "requests.jsonl"
    calls_used = len(read_jsonl(attempt_path)) if attempt_path.exists() else 0
    seed = config.get("seed", 42)
    plan = []
    for case in cases:
        for sample in range(config.get("samples", 1)):
            conditions = list(config["conditions"])
            stable = int(digest([seed, case["case_id"], sample])[:8], 16)
            random.Random(stable).shuffle(conditions)
            for condition in conditions:
                identity = case["case_id"] if mode == "controlled" else case["cluster_id"]
                unit = f"{mode}|{identity}|{sample}|{condition}"
                plan.append((unit, case, sample, condition, stable % (2**31 - 1)))
    # Avoid loading a local model again when a batch already finished.
    if all(unit in completed for unit, *_ in plan):
        return {"output_dir": str(out), "completed": len(existing), "new_runs": 0}
    backend = make_backend(resolved_backend)
    new_runs = 0
    for unit, case, sample, condition, paired_seed in plan:
        if unit in completed:
            continue
        env = make_env()
        task = Task(**case["task"])
        env.reset(task)
        steps = []
        if mode == "controlled":
            for action in case["prefix_actions"] + [case["failed_action"]] * case["history_depth"]:
                outcome = env.step(task, action)
                steps.append(trajectory_step(action, outcome))
            if [s.ok for s in steps] != [True] * len(case["prefix_actions"]) + [False] * case["history_depth"]:
                raise RuntimeError("Frozen checkpoint no longer replays; inspect upstream changes")
        initial_failures = [s for s in steps if not s.ok]
        generated, turns = [], []
        t0 = time.monotonic()
        stop_reason = "step_limit"
        budget = config["max_steps"]
        for index in range(budget):
            if calls_used >= config["max_requests"]:
                raise RuntimeError(f"Request cap {config['max_requests']} reached. Completed trajectories are saved; current partial trajectory is not scored.")
            messages, card = render(env, task, steps, condition)
            audit = literal_leak_audit(steps, card)
            if audit and audit["full_failed_call_present"]:
                raise RuntimeError("Full failed call leaked into correction card")
            calls_used += 1
            append_jsonl(attempt_path, {"request_index": calls_used, "unit_id": unit, "step": index,
                                        "seed": paired_seed + index, "time_utc": datetime.now(timezone.utc).isoformat()})
            # Only the smoke double receives reference actions.
            context = {"generated_index": index}
            if resolved_backend["kind"] == "smoke":
                context["reference_tail"] = case["private_reference_tail"] if mode == "controlled" else task.gold_actions
            try:
                response = backend.generate(messages, paired_seed + index, context)
            except Exception as exc:
                append_jsonl(out / "errors.jsonl", {"unit_id": unit, "step": index, "error_type": type(exc).__name__})
                raise
            action = extract_action(response["text"])
            outcome = env.step(task, action)
            step = trajectory_step(action, outcome)
            generated.append(step)
            steps.append(step)
            turns.append({"messages": messages, "card": card, "literal_audit": audit,
                          "response": response, "executed_step": asdict(step)})
            if env.is_solved(task):
                stop_reason = "solved"
                break
            if outcome.info.get("finish"):
                stop_reason = "finish_called"
                break
        solved = env.is_solved(task)
        row = {"unit_id": unit, "case_id": case["case_id"] if mode == "controlled" else case["cluster_id"],
               "cluster_id": case["cluster_id"], "task_template": task.payload["template"],
               "error_family": case["error_family"] if mode == "controlled" else None,
               "history_depth": case["history_depth"] if mode == "controlled" else None,
               "condition": condition, "sample": sample, "seed": paired_seed, "mode": mode,
               "result_kind": "plumbing_only" if resolved_backend["kind"] == "smoke" else "model_experiment",
               "stop_reason": stop_reason, "seconds": time.monotonic() - t0, "turns": turns,
               "injected_failures": [asdict(s) for s in initial_failures],
               "metrics": metrics(generated, initial_failures, mode, solved)}
        append_jsonl(rows_path, row)
        new_runs += 1
        completed.add(unit)
        print(f"[{len(completed)}/{len(plan)}] {condition} {case['case_id']} success={int(solved)}", flush=True)
    write_json(out / "completion.json", {"planned": len(plan), "completed": len(completed), "requests": calls_used})
    return {"output_dir": str(out), "completed": len(completed), "new_runs": new_runs, "requests": calls_used}

