"""Run twelve behavior arms from the same 48 frozen failure checkpoints."""
from intervention_lib import Decoder, messages
from train_intervention_heads import train
from pinkelephant.common import PROJECT, append_jsonl, digest, project_path, read_json, read_jsonl, write_json
from pinkelephant.ftb import Task, canonical, extract_action, make_env, revision, trajectory_step
from pinkelephant.experiment import metrics
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import random
import time
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/interventions_v1.json")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    cfg = read_json(project_path(args.config))
    cases = read_jsonl(project_path(cfg["evaluation_data"]))
    out = project_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if args.train_only and (out / "training.json").exists():
        previous = read_json(out / "training.json")
        training_data = read_jsonl(project_path(cfg["training_config"]["dataset"]["path"]))
        if previous["complete"] and previous["config"] == cfg["training"] and previous["training_sha256"] == digest(training_data):
            print("Both trained heads already exist and match the training protocol/data")
            return
        raise RuntimeError("Existing heads do not match the training protocol")
    source_paths = [PROJECT / "scripts" / name for name in ("intervention_lib.py", "run_interventions.py", "train_intervention_heads.py")]
    source_paths += list((PROJECT / "src").rglob("*.py"))
    source = {str(p.relative_to(PROJECT)): p.read_text(encoding="utf-8") for p in sorted(source_paths)}
    signature = digest({"config": cfg, "data": cases, "sources": source, "upstream": revision()})
    manifest = out / "manifest.json"
    if manifest.exists() and read_json(manifest)["signature"] != signature:
        raise RuntimeError("Experiment changed: select a new output folder")
    if not manifest.exists():
        write_json(manifest, {"signature": signature, "config": cfg, "upstream": revision(),
                   "sources": source, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                   "protocol": "greedy_actual_generation_max_3_steps_no_test_gold_in_decoder"})
    raw = out / "trajectories.jsonl"
    existing = read_jsonl(raw) if raw.exists() else []
    done = {(r["case_id"], r["arm"]) for r in existing}
    plan = []
    for case in cases:
        arms = list(cfg["arms"])
        random.Random(int(digest(case["case_id"])[:8], 16)).shuffle(arms)
        plan += [(case, arm) for arm in arms if (case["case_id"], arm) not in done]
    if not plan:
        print(f"All {len(existing)} trajectories already completed")
        return
    decoder = Decoder(cfg)
    if not (out / "training.json").exists():
        train(decoder, cfg, out)
    if args.train_only:
        print("Training completed; use run_interventions_batched.py for evaluation")
        return
    decoder.load_adapters(out)
    for case, arm in plan:
        env = make_env()
        task = Task(**case["task"])
        env.reset(task)
        steps = []
        for action in case["prefix_actions"] + [case["failed_action"]] * case["history_depth"]:
            steps.append(trajectory_step(action, env.step(task, action)))
        n_prefix = len(case["prefix_actions"])
        if [s.ok for s in steps] != [True] * n_prefix + [False] * case["history_depth"]:
            raise RuntimeError("Checkpoint failed to replay")
        initial = [s for s in steps if not s.ok]
        generated = []
        turns = []
        t0 = time.monotonic()
        reason = "step_limit"
        for _ in range(cfg["max_steps"]):
            output = decoder.generate(env, task, steps, arm)
            action = extract_action(output["text"])
            outcome = env.step(task, action)
            step = trajectory_step(action, outcome)
            steps.append(step)
            generated.append(step)
            turns.append({**output, "executed_step": asdict(step)})
            if env.is_solved(task):
                reason = "solved"
                break
            if outcome.info.get("finish"):
                reason = "finish_called"
                break
        values = metrics(generated, initial, "controlled", env.is_solved(task))
        values["first_same_error_family"] = int(generated[0].tool == initial[-1].tool and generated[0].error_type == initial[-1].error_type)
        values["same_error_family_any"] = int(any(s.tool == initial[-1].tool and s.error_type == initial[-1].error_type for s in generated))
        values["truncated_any"] = int(any(t["finish_reason"] == "length" for t in turns))
        # A shape-changing repeat that still has the same tool/error is measured
        # separately from canonical equality. Neither is a full semantic judge.
        row = {"case_id": case["case_id"], "cluster_id": case["cluster_id"], "arm": arm,
               "history_depth": case["history_depth"], "error_family": case["error_family"],
               "task_template": task.payload["template"], "stop_reason": reason,
               "seconds": time.monotonic() - t0, "initial_failures": [asdict(s) for s in initial],
               "metrics": values, "turns": turns}
        append_jsonl(raw, row)
        done.add((case["case_id"], arm))
        print(f"[{len(done)}/{len(cases) * len(cfg['arms'])}] {arm} {case['case_id']} success={values['task_success']} repeat={values['any_target_repeat']}", flush=True)
    write_json(out / "completion.json", {"complete": True, "cases": len(cases), "arms": len(cfg["arms"]),
               "trajectories": len(done), "signature": signature})


if __name__ == "__main__":
    main()
