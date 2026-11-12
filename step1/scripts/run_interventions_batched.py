"""Complete, batched GPU rerun; previous sequential diagnostics are separate."""
from intervention_lib import Decoder, messages
from pinkelephant.common import PROJECT, append_jsonl, digest, project_path, read_json, read_jsonl, write_json
from pinkelephant.ftb import Task, extract_action, make_env, revision, trajectory_step
from pinkelephant.experiment import metrics
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import random
import shutil
import time
import torch


def checkpoint(case):
    env = make_env()
    task = Task(**case["task"])
    env.reset(task)
    steps = []
    for action in case["prefix_actions"] + [case["failed_action"]] * case["history_depth"]:
        steps.append(trajectory_step(action, env.step(task, action)))
    assert [s.ok for s in steps] == [True] * len(case["prefix_actions"]) + [False] * case["history_depth"]
    return env, task, steps


@torch.inference_mode()
def validate_batch(decoder, cases):
    contexts = []
    for case in (cases[0], cases[len(cases) // 2], cases[-1]):
        env, task, steps = checkpoint(case)
        for policy in ("abstract", "neutral", "drop"):
            contexts.append(decoder.encode(messages(env, task, steps, policy)))
    length = max(map(len, contexts))
    ids = torch.full((len(contexts), length), decoder.tok.pad_token_id, dtype=torch.long, device=decoder.net.device)
    mask = torch.zeros_like(ids)
    for i, context in enumerate(contexts):
        ids[i, -len(context):] = torch.tensor(context, device=decoder.net.device)
        mask[i, -len(context):] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    output = decoder.net.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
    batched = decoder.net.lm_head(output.last_hidden_state[:, -1]).float()
    max_difference = 0.0
    max_kl = 0.0
    for i, context in enumerate(contexts):
        _, single = decoder.state(context)
        difference = (single - batched[i]).abs().max().item()
        max_difference = max(max_difference, difference)
        a, b = single.log_softmax(-1), batched[i].log_softmax(-1)
        kl = float((a.exp() * (a - b)).sum())
        max_kl = max(max_kl, kl)
        if int(single.argmax()) != int(batched[i].argmax()) or difference > 0.2 or kl > 0.001:
            raise RuntimeError(f"Batch validation failed: difference={difference}, KL={kl}, single={int(single.argmax())}, batch={int(batched[i].argmax())}")
    # Exercise growing caches and masks too; tolerate normal BF16 rounding,
    # but require the first eight greedy tokens to match in an identical batch.
    tiny = dict(decoder.cfg)
    decoder.cfg = {**tiny, "max_new_tokens": 8}
    env, task, steps = checkpoint(cases[0])
    single_text = decoder.generate(env, task, steps, "abstract")["text"]
    batch_texts = decoder.generate_batch([(env, task, steps)] * 2, "abstract")
    decoder.cfg = tiny
    if any(row["text"] != single_text for row in batch_texts):
        raise RuntimeError("Growing KV-cache validation failed")
    return {"contexts_checked": len(contexts), "max_logit_difference": max_difference,
            "max_next_token_KL": max_kl,
            "first_argmax_all_match": True, "eight_token_cache_check": True,
            "note": "BF16 batch sizes can change rounding; all reported conditions use this batch runner."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = read_json(project_path("configs/interventions_v1.json"))
    cfg["output_dir"] = "results/interventions-batched-fp16-v2"
    cfg["batch_size"] = 4
    cfg["inference_dtype"] = "float16"
    cfg["cuda_matmul_tf32"] = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg["trained_heads_from"] = "results/interventions-v1"
    out = project_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(project_path(cfg["evaluation_data"]))
    files = [PROJECT / "scripts" / name for name in ("intervention_lib.py", "run_interventions_batched.py", "train_intervention_heads.py")]
    files += list((PROJECT / "src").rglob("*.py"))
    source = {str(p.relative_to(PROJECT)): p.read_text(encoding="utf-8") for p in sorted(files)}
    signature = digest({"cfg": cfg, "cases": cases, "source": source, "upstream": revision()})
    manifest = out / "manifest.json"
    if manifest.exists() and read_json(manifest)["signature"] != signature:
        raise RuntimeError("Run changed: choose a new version")
    if not manifest.exists():
        write_json(manifest, {"signature": signature, "config": cfg, "sources": source,
                    "upstream": revision(), "created_utc": datetime.now(timezone.utc).isoformat(),
                    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "protocol": "greedy_batch_4_actual_actions_3_steps"})
    raw = out / "trajectories.jsonl"
    existing = read_jsonl(raw) if raw.exists() else []
    done = {(r["case_id"], r["arm"]) for r in existing}
    if len(done) == len(cases) * len(cfg["arms"]) and not args.validate_only:
        print(f"All {len(done)} trajectories already complete")
        return
    decoder = Decoder(cfg)
    decoder.net.half()
    write_json(out / "batch_validation.json", validate_batch(decoder, cases))
    print("Batch numerical and KV-cache validation passed", flush=True)
    if args.validate_only:
        return
    trained_from = project_path(cfg["trained_heads_from"])
    training = read_json(trained_from / "training.json")
    if not training["complete"] or training["config"] != cfg["training"]:
        raise RuntimeError("Trained heads do not match registered training protocol")
    eval_seeds = {r["task"]["payload"]["world_seed"] for r in cases}
    if eval_seeds & set(training["training_world_seeds"]):
        raise RuntimeError("Train/test world overlap")
    for name in ("training.json", "sft_head.pt", "ul_head.pt"):
        if not (out / name).exists():
            shutil.copy2(trained_from / name, out / name)
    decoder.load_adapters(out)
    arms = list(cfg["arms"])
    random.Random(cfg["seed"]).shuffle(arms)
    for arm in arms:
        selected = [case for case in cases if (case["case_id"], arm) not in done]
        random.Random(cfg["seed"]).shuffle(selected)
        for start in range(0, len(selected), cfg["batch_size"]):
            chunk = selected[start:start + cfg["batch_size"]]
            runs = []
            for case in chunk:
                env, task, steps = checkpoint(case)
                runs.append({"case": case, "env": env, "task": task, "steps": steps,
                             "initial": [s for s in steps if not s.ok], "generated": [], "turns": [],
                             "seconds": 0.0, "stop_reason": "step_limit", "done": False})
            for step_index in range(cfg["max_steps"]):
                active = [r for r in runs if not r["done"]]
                if not active:
                    break
                t0 = time.monotonic()
                responses = decoder.generate_batch([(r["env"], r["task"], r["steps"]) for r in active], arm)
                share = (time.monotonic() - t0) / len(active)
                for run, response in zip(active, responses):
                    run["seconds"] += share
                    action = extract_action(response["text"])
                    outcome = run["env"].step(run["task"], action)
                    step = trajectory_step(action, outcome)
                    run["steps"].append(step)
                    run["generated"].append(step)
                    run["turns"].append({**response, "executed_step": asdict(step)})
                    if run["env"].is_solved(run["task"]):
                        run["done"], run["stop_reason"] = True, "solved"
                    elif outcome.info.get("finish"):
                        run["done"], run["stop_reason"] = True, "finish_called"
            for run in runs:
                case = run["case"]
                generated, initial = run["generated"], run["initial"]
                values = metrics(generated, initial, "controlled", run["env"].is_solved(run["task"]))
                values["first_same_error_family"] = int(generated[0].tool == initial[-1].tool and generated[0].error_type == initial[-1].error_type)
                values["same_error_family_any"] = int(any(s.tool == initial[-1].tool and s.error_type == initial[-1].error_type for s in generated))
                values["truncated_any"] = int(any(t["finish_reason"] == "length" for t in run["turns"]))
                row = {"case_id": case["case_id"], "cluster_id": case["cluster_id"], "arm": arm,
                       "history_depth": case["history_depth"], "error_family": case["error_family"],
                       "task_template": run["task"].payload["template"], "stop_reason": run["stop_reason"],
                       "seconds": run["seconds"], "timing": "amortized_batch_compute_not_single_request_latency",
                       "initial_failures": [asdict(s) for s in initial], "metrics": values, "turns": run["turns"]}
                append_jsonl(raw, row)
                done.add((case["case_id"], arm))
            print(f"[{len(done)}/{len(cases) * len(cfg['arms'])}] {arm} last_batch_solved={sum(r['metrics']['task_success'] for r in [row])}", flush=True)
    write_json(out / "completion.json", {"complete": True, "cases": len(cases), "arms": len(cfg["arms"]),
              "trajectories": len(done), "signature": signature})


if __name__ == "__main__":
    main()
