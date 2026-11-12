"""Shared-trigger paired behavior experiments; monitor receives public data."""
from dataclasses import asdict
from datetime import datetime, timezone
import random
import time

from .bridge import Task, canonical, dependencies, extract_action, make_env, strict_score, trajectory_step
from .common import ROOT, append, digest, path, read_json, read_rows, write_json
from .feedback import FeedbackEngine, compose


def replay(case, free=False):
    env, task = make_env(), Task(**case["task"])
    env.reset(task)
    steps = []
    if not free:
        actions = case["replay_actions"] if "replay_actions" in case else case.get("prefix_actions", []) + [case["failed_action"]] * case["history_depth"]
        for action in actions:
            steps.append(trajectory_step(action, env.step(task, action)))
        if not steps or steps[-1].ok:
            raise RuntimeError("Checkpoint does not end in the registered failure")
        if "expected_steps" in case:
            for a, b in zip(steps, case["expected_steps"], strict=True):
                if a.ok != b["ok"] or a.observation != b["observation"]:
                    raise RuntimeError("Natural failure replay mismatch")
    return env, task, steps


def run(cfg, limit=None, output=None):
    from .models import Actor
    cases = read_rows(path(cfg["data_path"]))
    free = cfg.get("mode", "controlled") == "free"
    if free:
        cases = [{**c, "case_id": c["cluster_id"]} for c in {c["cluster_id"]: c for c in cases}.values()]
    if limit:
        cases = cases[:limit]
    out = path(output or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    code = {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in sorted((ROOT / "src").rglob("*.py"))}
    # Resolve public endpoint/model overrides; keys are never stored.
    backend = dict(cfg["actor"])
    if backend["kind"] == "transformers":
        import os
        from pathlib import Path
        checkpoint = Path(backend["model_path"])
        if not checkpoint.is_absolute():
            checkpoint = Path(os.environ.get("MODEL_DIR", "C:/Projects/AI/models")) / checkpoint
        backend["model_path"] = str(checkpoint.resolve())
        model_metadata = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in checkpoint.glob("*") if p.suffix in {".json", ".safetensors"}}
    else:
        model_metadata = {}
    if backend["kind"] == "chat_api":
        import os
        from urllib.parse import urlparse
        for field, name in (("model", "model_env"), ("base_url", "base_url_env")):
            key = backend.get(name, "PE_MODEL" if field == "model" else "PE_BASE_URL")
            if key in os.environ:
                backend[field] = os.environ[key]
        parsed = urlparse(backend.get("base_url", ""))
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Credentials must be in environment variables, not the endpoint URL")
    signature = digest({"config": cfg, "resolved_actor": backend, "model_metadata": model_metadata, "cases": cases, "code": code, "dependencies": dependencies()})
    manifest = out / "manifest.json"
    if manifest.exists() and read_json(manifest)["signature"] != signature:
        raise RuntimeError("Config/data/code changed; use a new output directory")
    if not manifest.exists():
        write_json(manifest, {"signature": signature, "config": cfg, "resolved_actor": backend, "limit": limit,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "dependencies": dependencies(),
                    "protocol": "shared_failure_only_trigger_strict_goal_stop", "case_ids": [c["case_id"] for c in cases]})
    raw = out / "trajectories.jsonl"
    done = {(r["case_id"], r["condition"]) for r in read_rows(raw)} if raw.exists() else set()
    expected = {(c["case_id"], m) for c in cases for m in cfg["conditions"]}
    if expected <= done:
        return {"output": str(out), "completed": len(done), "new_runs": 0}
    actor = Actor(backend)
    is_token_budget = actor.tokenizer is not None
    budget = cfg["max_feedback_tokens"] if is_token_budget else cfg["max_feedback_bytes"]
    arms = list(cfg["conditions"])
    random.Random(cfg["seed"]).shuffle(arms)
    attempts = out / "requests.jsonl"
    calls = len(read_rows(attempts)) if attempts.exists() else 0
    for mode in arms:
        todo = [c for c in cases if (c["case_id"], mode) not in done]
        random.Random(cfg["seed"]).shuffle(todo)
        for start in range(0, len(todo), cfg["batch_size"]):
            batch = []
            for case in todo[start:start + cfg["batch_size"]]:
                env, task, steps = replay(case, free)
                batch.append({"case": case, "env": env, "task": task, "steps": steps,
                              "initial": list(steps), "engine": FeedbackEngine(), "turns": [], "done": False,
                              "seconds": 0.0, "stop_reason": "step_limit"})
            for index in range(cfg["max_steps"]):
                active = [r for r in batch if not r["done"]]
                if not active:
                    break
                packed = []
                for item in active:
                    if calls >= cfg["max_calls"]:
                        raise RuntimeError("Model-call cap reached; completed trajectories retained")
                    msgs, card, ledger = compose(item["env"].system_prompt(item["task"]), item["task"].goal,
                                                item["steps"], mode, item["engine"], actor.measure, budget)
                    packed.append((msgs, card, ledger))
                    calls += 1
                    append(attempts, {"case_id": item["case"]["case_id"], "condition": mode, "step": index, "call_index": calls})
                t0 = time.monotonic()
                responses = actor.generate_batch([p[0] for p in packed], cfg["seed"])
                share = (time.monotonic() - t0) / len(active)
                for item, (msgs, card, ledger), response in zip(active, packed, responses, strict=True):
                    action = extract_action(response["text"])
                    outcome = item["env"].step(item["task"], action)
                    step = trajectory_step(action, outcome)
                    item["steps"].append(step)
                    item["seconds"] += share
                    item["turns"].append({"messages": msgs, "card": card, "evidence": ledger, "response": response,
                                          "executed_step": asdict(step), "feedback_budget_unit": "tokens" if is_token_budget else "utf8_bytes"})
                    if strict_score(item["env"], item["task"]):
                        item["done"], item["stop_reason"] = True, "strict_goal_satisfied"
                    elif outcome.info.get("finish"):
                        item["done"], item["stop_reason"] = True, "finish_called"
            for item in batch:
                initial_failed = [s for s in item["initial"] if not s.ok]
                seen = {canonical(s.action) for s in initial_failed}
                repeats = 0
                for turn in item["turns"]:
                    s = turn["executed_step"]
                    if not s["ok"]:
                        key = canonical(s["action"])
                        repeats += key in seen
                        seen.add(key)
                metrics = {"strict_success": int(strict_score(item["env"], item["task"])),
                           "upstream_success": int(item["env"].is_solved(item["task"])),
                           "first_action_valid": int(item["turns"][0]["executed_step"]["ok"]),
                           "any_failure_repeat": int(repeats > 0), "repeat_count": repeats,
                           "n_steps": len(item["turns"]),
                           "truncated_any": int(any(t["response"]["finish_reason"] == "length" for t in item["turns"]))}
                case_id = item["case"]["cluster_id"] if free else item["case"]["case_id"]
                row = {"case_id": case_id, "cluster_id": item["case"]["cluster_id"], "condition": mode,
                       "mode": "free" if free else "controlled", "task": asdict(item["task"]),
                       "history_depth": None if free else item["case"]["history_depth"],
                       "initial_steps": [asdict(s) for s in item["initial"]], "turns": item["turns"],
                       "metrics": metrics, "stop_reason": item["stop_reason"], "seconds_amortized": item["seconds"]}
                append(raw, row)
                done.add((item["case"]["case_id"], mode))
            print(f"{len(done)}/{len(expected)} completed: {mode}", flush=True)
    write_json(out / "completion.json", {"complete": True, "trajectories": len(done), "calls": calls})
    return {"output": str(out), "completed": len(done), "calls": calls}


def collect(source, output, condition="abstract"):
    """First native failure per task from FROM-SCRATCH free runs only."""
    selected = {}
    for row in read_rows(source):
        if row["mode"] != "free" or row["condition"] != condition:
            continue
        actions, expected = [], []
        for turn in row["turns"]:
            step = turn["executed_step"]
            actions.append(step["action"])
            expected.append(step)
            if not step["ok"]:
                cluster = row["cluster_id"]
                selected.setdefault(cluster, {"case_id": cluster + ":native", "cluster_id": cluster,
                    "task": row["task"], "replay_actions": list(actions), "expected_steps": list(expected),
                    "history_depth": sum(not s["ok"] for s in expected), "error_family": step["error_type"],
                    "source_kind": "first_model_generated_failure_from_initial_state", "source_condition": condition})
                break
    if not selected:
        raise ValueError("No eligible from-scratch model failures; injected continuation logs are not accepted")
    target = path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(__import__("json").dumps(r, ensure_ascii=False) + "\n" for r in selected.values())
    if target.exists() and target.read_text(encoding="utf-8") != body:
        raise RuntimeError("Frozen natural-failure file already exists with different content")
    target.write_text(body, encoding="utf-8")
    write_json(target.with_suffix(".manifest.json"), {"source": str(source), "source_sha": digest(read_rows(source)),
               "condition": condition, "cases": len(selected), "selection": "first_failure_per_task_no_recovery_based_selection"})
    return {"natural_failures": len(selected), "output": str(target)}
