from __future__ import annotations

import json
from dataclasses import asdict

from .common import digest, project_path, write_json
from .ftb import generate_tasks, make_env, perturbations, revision


def prepare(config):
    """One verified failed checkpoint per distinct task, balanced by family.

    Each checkpoint has variants with one/two injected failures. Gold actions
    are used only here to establish a reachable prefix and reference feasibility.
    """
    spec = config["dataset"]
    seed = spec["world_seed"]
    quota = spec["per_family"]
    families = spec["families"]
    counts = dict.fromkeys(families, 0)
    cases = []
    chosen_tasks = set()
    env = make_env()
    candidates = generate_tasks(spec.get("candidate_tasks", 600), seed)
    for task in candidates:
        if all(value == quota for value in counts.values()):
            break
        for family in families:
            if counts[family] >= quota:
                continue
            found = None
            for index, action in enumerate(task.gold_actions):
                env.reset(task)
                if not all(env.step(task, previous).ok for previous in task.gold_actions[:index]):
                    continue
                if env.is_solved(task):
                    continue
                for perturbation in perturbations(action, env.world(task)):
                    if perturbation.expected_error != family:
                        continue
                    env.reset(task)
                    for previous in task.gold_actions[:index]:
                        env.step(task, previous)
                    before = asdict(env.world(task))
                    outcome = env.step(task, perturbation.action)
                    if outcome.ok or outcome.error_type != family or before != asdict(env.world(task)):
                        continue
                    # Same action can be injected twice without changing the world.
                    second = env.step(task, perturbation.action)
                    if second.ok or second.error_type != family or before != asdict(env.world(task)):
                        continue
                    # Verify the original remaining trajectory still solves the task.
                    tail_ok = all(env.step(task, a).ok for a in task.gold_actions[index:])
                    if not tail_ok or not env.is_solved(task):
                        continue
                    found = (index, perturbation)
                    break
                if found:
                    break
            if not found:
                continue
            index, perturbation = found
            cluster_id = f"{seed}:{task.task_id}"
            for depth in spec["history_depths"]:
                cases.append({
                    "case_id": f"{cluster_id}:{family}:h{depth}", "cluster_id": cluster_id,
                    "split": spec["split"], "task": asdict(task),
                    "prefix_actions": task.gold_actions[:index],
                    "failed_action": perturbation.action, "error_family": family,
                    "operator": perturbation.operator, "history_depth": depth,
                    "private_reference_tail": task.gold_actions[index:],
                })
            chosen_tasks.add(task.task_id)
            counts[family] += 1
            break
    if any(value != quota for value in counts.values()):
        raise RuntimeError(f"Insufficient valid cases: {counts}; increase candidate_tasks or reduce quota")
    path = project_path(spec["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    # Freeze the generated file: never silently replace it after tuning.
    if path.exists() and path.read_text(encoding="utf-8") != body:
        raise RuntimeError(f"Dataset already exists with different content: {path}; choose a new path")
    path.write_text(body, encoding="utf-8")
    write_json(path.with_suffix(".manifest.json"), {
        "dataset_spec": spec, "sha256": digest(cases), "cases": len(cases),
        "task_clusters": len(chosen_tasks), "counts_by_family_tasks": counts,
        "task_templates": sorted({case['task']['payload']['template'] for case in cases}),
        "upstream": revision(), "synthetic_failures": True,
        "warning": "Private reference actions are for replay/smoke validation only; never send them to a real actor or monitor.",
    })
    return {"path": str(path), "cases": len(cases), "task_clusters": len(chosen_tasks), "counts": counts}

