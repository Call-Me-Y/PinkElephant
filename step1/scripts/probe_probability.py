"""Teacher-forced candidate scoring. Does not generate actions or update models."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
import statistics

from pinkelephant.backends import TransformersBackend
from pinkelephant.common import append_jsonl, digest, project_path, read_json, read_jsonl, write_json
from pinkelephant.feedback import CONDITIONS, render
from pinkelephant.ftb import Task, make_env, perturbations, revision, trajectory_step


def score_tokens(model, prefix, suffix, device):
    """Causal next-token alignment; the prefix is never included in the score.

    Returns log probability of the candidate token PREFIX, without an EOS term.
    It is not the probability of generating that action and then stopping.
    """
    import torch
    if not prefix or not suffix:
        raise ValueError("Both context and candidate must contain tokens")
    ids = torch.tensor([prefix + suffix], device=device)
    with torch.inference_mode():
        output = model(input_ids=ids, use_cache=False)
        logits = output.logits[0, len(prefix) - 1:-1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite candidate logits")
        log_probs = logits.log_softmax(-1)
        targets = ids[0, len(prefix):]
        selected = log_probs.gather(-1, targets[:, None]).squeeze(-1)
        if not torch.isfinite(selected).all() or (selected > 0).any():
            raise RuntimeError("Invalid token log probabilities")
        return {"logp": selected.sum().item(), "n_tokens": len(suffix),
                "mean_logp": selected.mean().item(),
                "all_candidate_tokens_greedy": bool((logits.argmax(-1) == targets).all())}


def replay(case):
    env = make_env()
    task = Task(**case["task"])
    env.reset(task)
    prefix = []
    for action in case["prefix_actions"]:
        outcome = env.step(task, action)
        if not outcome.ok:
            raise RuntimeError("Prefix replay failed")
        prefix.append(trajectory_step(action, outcome))
    return env, task, prefix


def candidates_for(case, count):
    env, task, _ = replay(case)
    gold = case["private_reference_tail"][0]
    candidates = [case["failed_action"], gold]
    for item in perturbations(gold, env.world(task)):
        if item.action in candidates:
            continue
        check, checked_task, _ = replay(case)
        if not check.step(checked_task, item.action).ok:
            candidates.append(item.action)
        if len(candidates) == count:
            break
    if len(candidates) != count:
        raise RuntimeError(f"Need {count} distinct candidates for {case['case_id']}")
    # Check the evaluator's reference on the same state. It never enters messages.
    check, checked_task, _ = replay(case)
    if not check.step(checked_task, gold).ok:
        raise RuntimeError("Reference action is not valid")
    return candidates


def messages_for(case, condition):
    env, task, prefix = replay(case)
    if condition == "pre_clean":
        return render(env, task, prefix, "verbatim")[0]
    steps = list(prefix)
    for _ in range(case["history_depth"]):
        outcome = env.step(task, case["failed_action"])
        if outcome.ok or outcome.error_type != case["error_family"]:
            raise RuntimeError("Failure replay mismatch")
        step = trajectory_step(case["failed_action"], outcome)
        if condition == "neutral":
            step.observation = env.neutral_observation(step.action, outcome)
        steps.append(step)
    return render(env, task, steps, "verbatim" if condition == "neutral" else condition)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/probability_pilot.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--conditions", nargs="+")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    cfg = read_json(project_path(args.config))
    conditions = args.conditions or cfg["conditions"]
    if not set(conditions) <= set(CONDITIONS) | {"pre_clean", "neutral"}:
        parser.error("Unknown condition")
    cases = read_jsonl(project_path(cfg["dataset_path"]))
    if args.limit:
        cases = cases[:args.limit]
    # Resolve and store the actual checkpoint path, not credentials.
    model_spec = dict(cfg["model"])
    model_path = Path(model_spec["model_path"])
    if not model_path.is_absolute():
        model_path = Path(os.environ.get("MODEL_DIR", "C:/Projects/AI/models")) / model_path
    model_spec["model_path"] = str(model_path.resolve())
    weights = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
               for p in model_path.glob("*") if p.suffix in {".json", ".safetensors"}}
    sources = {str(p.relative_to(project_path("."))): p.read_text(encoding="utf-8")
               for p in sorted(project_path("src").rglob("*.py"))}
    signature = digest({"config": cfg, "resolved_model": model_spec, "weights_metadata": weights,
                        "cases": cases, "conditions": conditions, "upstream": revision(), "sources": sources,
                        "probe_source": Path(__file__).read_text(encoding="utf-8")})
    out = project_path(args.output or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists() and read_json(manifest)["signature"] != signature:
        raise RuntimeError("Config/data/code changed; select a new output folder")
    write_json(manifest, {"signature": signature, "config": cfg, "resolved_model": model_spec,
                          "conditions": conditions, "limit": args.limit, "upstream": revision(),
                          "measurement": "teacher_forced_candidate_token_prefix_without_EOS"})
    raw = out / "scores.jsonl"
    existing = read_jsonl(raw) if raw.exists() else []
    done = {(r["case_id"], r["condition"]) for r in existing}
    todo = [(case, condition) for case in cases for condition in conditions if (case["case_id"], condition) not in done]
    if todo:
        backend = TransformersBackend(model_spec)
        tok = backend.tokenizer
        for case, condition in todo:
            candidates = candidates_for(case, cfg.get("candidate_count", 4))
            messages = messages_for(case, condition)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **model_spec.get("chat_template_kwargs", {}))
            prefix = tok.encode(prompt, add_special_tokens=False)
            scores = []
            for action in candidates:
                suffix = tok.encode(action, add_special_tokens=False)
                if len(prefix) + len(suffix) > model_spec.get("context_limit", 8192):
                    raise RuntimeError("Context exceeds budget; no truncation permitted")
                scores.append(score_tokens(backend.model, prefix, suffix, backend.model.device))
            maximum = max(s["logp"] for s in scores)
            denominator = maximum + math.log(sum(math.exp(s["logp"] - maximum) for s in scores))
            row = {"case_id": case["case_id"], "cluster_id": case["cluster_id"],
                   "history_depth": case["history_depth"], "error_family": case["error_family"],
                   "condition": condition, "messages": messages, "candidates": candidates, "scores": scores,
                   "p_repeat_candidate_set": math.exp(scores[0]["logp"] - denominator),
                   "gold_minus_fail_logp": scores[1]["logp"] - scores[0]["logp"]}
            append_jsonl(raw, row)
            print(f"{condition} {case['case_id']} margin={row['gold_minus_fail_logp']:.3f}", flush=True)
    rows = read_jsonl(raw)
    index = {(r["case_id"], r["condition"]): r for r in rows}
    derived = []
    for row in rows:
        pre = index.get((row["case_id"], "pre_clean"))
        neutral = index.get((row["case_id"], "neutral"))
        fail = index.get((row["case_id"], "verbatim"))
        logp = row["scores"][0]["logp"]
        derived.append({"case_id": row["case_id"], "cluster_id": row["cluster_id"],
                        "history_depth": row["history_depth"], "condition": row["condition"],
                        "failed_logp": logp, "failed_tokens": row["scores"][0]["n_tokens"],
                        "gain_vs_clean": pre["scores"][0]["logp"] - logp if pre else None,
                        "suppression_vs_verbatim": fail["scores"][0]["logp"] - logp if fail else None,
                        "copy_delta": neutral["scores"][0]["logp"] - pre["scores"][0]["logp"] if neutral and pre else None,
                        "semantic_delta": fail["scores"][0]["logp"] - neutral["scores"][0]["logp"] if fail and neutral else None,
                        "p_repeat_candidate_set": row["p_repeat_candidate_set"],
                        "gold_minus_fail_logp": row["gold_minus_fail_logp"]})
    with (out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(derived[0]))
        writer.writeheader()
        writer.writerows(derived)
    print(json.dumps({"output": str(out), "scored_contexts": len(rows), "new_contexts": len(todo)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
