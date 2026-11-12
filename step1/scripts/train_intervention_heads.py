"""Frozen-backbone low-rank output-head SFT and SFT+UL controls."""
from intervention_lib import HeadAdapter, messages
from pinkelephant.common import digest, read_jsonl, project_path, write_json
from pinkelephant.ftb import Task, make_env, trajectory_step
from pinkelephant.dataset import prepare
import random
import torch


def feature(decoder, msgs, action):
    prefix = decoder.encode(msgs)
    suffix = decoder.tok.encode(action, add_special_tokens=False)
    if len(prefix) + len(suffix) > decoder.cfg["model"]["context_limit"]:
        raise RuntimeError("Training context too long")
    with torch.no_grad():
        ids = torch.tensor([prefix + suffix], device=decoder.net.device)
        hidden = decoder.net.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        result = hidden[len(prefix) - 1:len(prefix) + len(suffix) - 1].cpu().clone()
    return result, suffix


def train(decoder, cfg, out):
    prepare(cfg["training_config"])
    training = read_jsonl(project_path(cfg["training_config"]["dataset"]["path"]))
    evaluation = read_jsonl(project_path(cfg["evaluation_data"]))
    train_seeds = {r["task"]["payload"]["world_seed"] for r in training}
    test_seeds = {r["task"]["payload"]["world_seed"] for r in evaluation}
    if train_seeds & test_seeds:
        raise RuntimeError("Training and evaluation worlds overlap")
    spec = cfg["training"]
    pairs = []
    for index, case in enumerate(training):
        env = make_env()
        task = Task(**case["task"])
        env.reset(task)
        steps = []
        for action in case["prefix_actions"] + [case["failed_action"]] * case["history_depth"]:
            steps.append(trajectory_step(action, env.step(task, action)))
        context = messages(env, task, steps, "abstract")
        good_h, good_ids = feature(decoder, context, case["private_reference_tail"][0])
        bad_h, bad_ids = feature(decoder, context, case["failed_action"])
        # Penalize only the first divergent decision, preserving the shared
        # correct prefix. This is a small UL variant, not a paper reproduction.
        divergence = next((i for i, (a, b) in enumerate(zip(bad_ids, good_ids)) if a != b), None)
        if divergence is None:
            raise RuntimeError("No divergent failure token; inspect tokenization")
        pairs.append((good_h, good_ids, bad_h[divergence:divergence + 1], bad_ids[divergence]))
        if (index + 1) % 16 == 0:
            print(f"training features {index + 1}/{len(training)}", flush=True)
    history = []
    # Both arms start with identical adapter parameters, order and CE targets.
    for name, ul_weight in (("sft_head", 0.0), ("ul_head", spec["ul_weight"])):
        torch.manual_seed(spec["seed"])
        adapter = HeadAdapter(decoder.net.config.hidden_size, decoder.net.config.vocab_size,
                              spec["rank"], spec["alpha"]).to(decoder.net.device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=spec["learning_rate"], weight_decay=0)
        for epoch in range(spec["epochs"]):
            order = list(range(len(pairs)))
            random.Random(spec["seed"] + epoch).shuffle(order)
            ce_sum = ul_sum = 0.0
            for index in order:
                good_h, good_ids, bad_h, bad_id = pairs[index]
                good_h = good_h.to(decoder.net.device)
                bad_h = bad_h.to(decoder.net.device)
                optimizer.zero_grad(set_to_none=True)
                good_logits = decoder.net.lm_head(good_h).float() + adapter(good_h)
                ce = torch.nn.functional.cross_entropy(good_logits, torch.tensor(good_ids, device=decoder.net.device))
                bad_logits = decoder.net.lm_head(bad_h).float() + adapter(bad_h)
                bad_logp = bad_logits.log_softmax(-1)[0, bad_id]
                ul = -torch.log1p(-bad_logp.exp().clamp(max=1 - 1e-6))
                loss = ce + ul_weight * ul
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                ce_sum += float(ce.detach())
                ul_sum += float(ul.detach())
            history.append({"arm": name, "epoch": epoch + 1, "ce_loss": ce_sum / len(pairs), "ul_loss": ul_sum / len(pairs)})
            print(f"{name} epoch {epoch + 1} CE={ce_sum / len(pairs):.4f} UL={ul_sum / len(pairs):.4f}", flush=True)
        torch.save({key: value.detach().cpu() for key, value in adapter.state_dict().items()}, out / f"{name}.pt")
        del optimizer, adapter
    write_json(out / "training.json", {"complete": True, "training_cases": len(training),
              "training_tasks": len({r["cluster_id"] for r in training}), "training_world_seeds": sorted(train_seeds),
              "evaluation_world_seeds": sorted(test_seeds), "training_sha256": digest(training), "config": spec,
              "backbone_frozen": True, "adapter_location": "output_head_only",
              "negative_target": "first_divergent_failure_token", "losses": history,
              "note": "SFT control isolates adding UL from supervised training; not full-model or online UL"})

