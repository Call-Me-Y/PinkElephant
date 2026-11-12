"""Inference interventions and independent-data low-rank head training.

Kept outside src so prior experiment fingerprints remain valid.
"""
from pathlib import Path
import sys
import os
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import re
import torch
from pinkelephant.backends import TransformersBackend
from pinkelephant.feedback import render
from pinkelephant.ftb import HARNESSES, TOOLS_BY_NAME, build_messages, parse_action
from slmecho.envs.base import Outcome
from slmecho.decoding import tokenize_ban_list


def messages(env, task, steps, policy):
    if policy == "neutral":
        from dataclasses import replace
        altered = [s if s.ok else replace(s, observation=env.neutral_observation(s.action, Outcome(False, s.observation, s.error_type))) for s in steps]
        return build_messages(env.system_prompt(task), task.goal, altered, HARNESSES["verbatim"])
    if policy == "drop":
        return build_messages(env.system_prompt(task), task.goal, steps, HARNESSES["drop"])
    return render(env, task, steps, policy)[0]


def fields(text):
    """Split top-level arguments while keeping strings and containers intact."""
    match = re.match(r"\s*([A-Za-z_]\w*)\s*\(", text)
    if not match:
        return None, [], False
    quote = None
    escaped = False
    nesting = 0
    start = match.end()
    parts = []
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[({":
            nesting += 1
        elif ch == ")" and nesting == 0:
            parts.append(text[start:i])
            return match.group(1), parts, True
        elif ch in "]})":
            nesting -= 1
        elif ch == "," and nesting == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return match.group(1), parts, False


def violates(text, rules, ending=False):
    """Rules contain only observed (tool, error_type), never gold values."""
    name, parts, closed = fields(text)
    kinds = rules.get(name, set())
    if not kinds or name not in TOOLS_BY_NAME:
        return False
    spec = TOOLS_BY_NAME[name]
    declared = {a.name for a in spec.args}
    supplied = set()
    for part in parts:
        if not part.strip():
            continue
        match = re.match(r"\s*([A-Za-z_]\w*)\s*(=)?", part)
        if not match:
            continue
        key = match.group(1)
        if "unexpected_argument" in kinds:
            if match.group(2) and key not in declared:
                return True
            if not match.group(2) and not any(d.startswith(key) for d in declared):
                return True
        if match.group(2):
            supplied.add(key)
        if "format_error" in kinds and match.group(2) and key in {"date", "time"}:
            value = part[match.end():].lstrip()
            if value.startswith(("'", '"')):
                quote = value[0]
                content = value[1:]
                complete = quote in content
                content = content.split(quote, 1)[0]
                pattern = "dddd-dd-dd" if key == "date" else "dd:dd"
                if len(content) > len(pattern) or (complete and len(content) != len(pattern)):
                    return True
                if any(not ch.isdigit() if expected == "d" else ch != expected for ch, expected in zip(content, pattern)):
                    return True
    if closed or ending:
        if "missing_argument" in kinds and any(a.required and a.name not in supplied for a in spec.args):
            return True
        if ending and not closed:
            return True
        if closed and "format_error" in kinds:
            try:
                parsed = parse_action(text)
                if name == "send_message" and "to" in parsed.kwargs and "@" not in str(parsed.kwargs["to"]):
                    return True
            except Exception:
                pass
    return False


def active_rules(steps):
    rules = {}
    # Successful calls clear the unresolved window. State-dependent rules are
    # therefore not carried into a new state without fresh error evidence.
    for step in steps:
        if step.ok:
            rules = {}
        elif step.tool:
            rules.setdefault(step.tool, set()).add(step.error_type)
    return rules


class HeadAdapter(torch.nn.Module):
    def __init__(self, hidden, vocab, rank, alpha):
        super().__init__()
        self.down = torch.nn.Linear(hidden, rank, bias=False)
        self.up = torch.nn.Linear(rank, vocab, bias=False)
        torch.nn.init.normal_(self.down.weight, std=0.02)
        torch.nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, hidden):
        return self.up(self.down(hidden.float())) * self.scale


class Decoder:
    def __init__(self, cfg):
        self.cfg = cfg
        backend = TransformersBackend(cfg["model"])
        self.net, self.tok = backend.model, backend.tokenizer
        self.net.requires_grad_(False)
        self.adapters = {}
        self.eos = {self.tok.eos_token_id}
        for name in ("<|im_end|>", "<|endoftext|>"):
            if name in self.tok.get_vocab():
                self.eos.add(self.tok.convert_tokens_to_ids(name))

    def encode(self, msgs):
        return self.tok.encode(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)

    @torch.inference_mode()
    def state(self, ids, cache=None, adapter=None):
        output = self.net.model(input_ids=torch.tensor([ids], device=self.net.device), past_key_values=cache, use_cache=True)
        hidden = output.last_hidden_state[:, -1, :]
        logits = self.net.lm_head(hidden).float()[0]
        if adapter is not None:
            logits = logits + adapter(hidden)[0]
        return output.past_key_values, logits

    def load_adapters(self, folder):
        for name in ("sft_head", "ul_head"):
            module = HeadAdapter(self.net.config.hidden_size, self.net.config.vocab_size,
                                 self.cfg["training"]["rank"], self.cfg["training"]["alpha"]).to(self.net.device)
            module.load_state_dict(torch.load(folder / f"{name}.pt", map_location=self.net.device, weights_only=True))
            self.adapters[name] = module.eval()

    @torch.inference_mode()
    def generate_batch(self, items, arm):
        policy = "verbatim" if arm in {"verbatim", "ftb_ban"} else "drop" if arm == "drop" else "positive_history" if arm == "positive_history" else "abstract"
        contrast = arm.startswith("contrast_")
        branches = 3 if contrast else 1
        msg_list = [messages(env, task, steps, policy) for env, task, steps in items]
        contexts = []
        for (env, task, steps), msgs in zip(items, msg_list):
            contexts.append(self.encode(msgs))
            if contrast:
                contexts += [self.encode(messages(env, task, steps, "neutral")), self.encode(messages(env, task, steps, "drop"))]
        longest = max(map(len, contexts))
        if longest + self.cfg["max_new_tokens"] > self.cfg["model"]["context_limit"]:
            raise RuntimeError("Context budget exceeded")
        ids = torch.full((len(contexts), longest), self.tok.pad_token_id, dtype=torch.long, device=self.net.device)
        mask = torch.zeros_like(ids)
        for row, context in enumerate(contexts):
            ids[row, -len(context):] = torch.tensor(context, device=self.net.device)
            mask[row, -len(context):] = 1
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        adapter = self.adapters.get(arm)
        cache = None
        generated = [[] for _ in items]
        finished = [False] * len(items)
        reasons = ["length"] * len(items)
        masked = [0] * len(items)
        bans = [tokenize_ban_list(self.tok, [s.action for s in steps if not s.ok]) if arm.endswith("ban") else [] for _, _, steps in items]
        rules = [active_rules(steps) if arm == "local_constraints" else {} for _, _, steps in items]
        lam = {"contrast_025": 0.25, "contrast_05": 0.5, "contrast_1": 1.0}.get(arm, 0)
        for _ in range(self.cfg["max_new_tokens"]):
            output = self.net.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                    past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            hidden = output.last_hidden_state[:, -1, :]
            all_logits = self.net.lm_head(hidden).float()
            if adapter is not None:
                all_logits = all_logits + adapter(hidden)
            if not torch.isfinite(all_logits).all():
                raise RuntimeError("Nonfinite batch logits")
            chosen = []
            keep = []
            for row in range(len(items)):
                if finished[row]:
                    chosen.extend([self.tok.eos_token_id] * branches)
                    keep.extend([0] * branches)
                    continue
                logits = all_logits[row * branches].clone()
                if contrast:
                    base = logits.log_softmax(-1)
                    delta = (all_logits[row * branches + 1].log_softmax(-1) - all_logits[row * branches + 2].log_softmax(-1)).clamp_min(0)
                    logits = base - lam * delta
                    logits[base < base.max() + math_log(self.cfg["plausibility_ratio"])] = -torch.inf
                for sequence in bans[row]:
                    k = len(sequence) - 1
                    if k == 0 or len(generated[row]) >= k and generated[row][-k:] == sequence[:k]:
                        logits[sequence[-1]] = -torch.inf
                        masked[row] += 1
                token = None
                if rules[row]:
                    for size in (64, 512, 4096, len(logits)):
                        for candidate in logits.topk(min(size, len(logits))).indices.tolist():
                            proposed = self.tok.decode(generated[row] + [candidate], skip_special_tokens=True)
                            ending = candidate in self.eos or "\n" in proposed
                            if not violates(proposed.split("\n", 1)[0], rules[row], ending):
                                token = candidate
                                break
                            masked[row] += 1
                        if token is not None:
                            break
                    if token is None:
                        raise RuntimeError("Constraint exhausted vocabulary")
                else:
                    token = int(logits.argmax())
                if token in self.eos:
                    finished[row], reasons[row] = True, "eos"
                else:
                    generated[row].append(token)
                    if "\n" in self.tok.decode(generated[row], skip_special_tokens=True):
                        finished[row], reasons[row] = True, "newline"
                chosen.extend([token] * branches)
                keep.extend([0 if finished[row] else 1] * branches)
            if all(finished):
                break
            ids = torch.tensor(chosen, device=self.net.device)[:, None]
            mask = torch.cat((mask, torch.tensor(keep, device=self.net.device)[:, None]), dim=1)
            positions = (mask.sum(-1) - 1).clamp_min(0)[:, None]
        return [{"text": self.tok.decode(tokens, skip_special_tokens=True), "messages": msg_list[i],
                 "completion_tokens": len(tokens), "prompt_tokens": len(contexts[i * branches]),
                 "forward_branches": branches, "masked_token_attempts": masked[i],
                 "finish_reason": reasons[i], "lambda": lam}
                for i, tokens in enumerate(generated)]

    @torch.inference_mode()
    def generate(self, env, task, steps, arm):
        policy = "verbatim" if arm in {"verbatim", "ftb_ban"} else "drop" if arm == "drop" else "positive_history" if arm == "positive_history" else "abstract"
        msgs = messages(env, task, steps, policy)
        contexts = [self.encode(msgs)]
        contrast = arm.startswith("contrast_")
        if contrast:
            contexts += [self.encode(messages(env, task, steps, "neutral")), self.encode(messages(env, task, steps, "drop"))]
        if max(map(len, contexts)) + self.cfg["max_new_tokens"] > self.cfg["model"]["context_limit"]:
            raise RuntimeError("Context limit exceeded")
        adapter = self.adapters.get(arm)
        states = [self.state(ids, adapter=adapter) for ids in contexts]
        ban = tokenize_ban_list(self.tok, [s.action for s in steps if not s.ok]) if arm.endswith("ban") else []
        rules = active_rules(steps) if arm == "local_constraints" else {}
        generated = []
        masked = 0
        finish_reason = "length"
        lam = {"contrast_025": 0.25, "contrast_05": 0.5, "contrast_1": 1.0}.get(arm, 0)
        for _ in range(self.cfg["max_new_tokens"]):
            logits = states[0][1].clone()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite decoding logits")
            if contrast:
                base = logits.log_softmax(-1)
                delta = (states[1][1].log_softmax(-1) - states[2][1].log_softmax(-1)).clamp_min(0)
                logits = base - lam * delta
                plausible = base >= base.max() + math_log(self.cfg["plausibility_ratio"])
                logits[~plausible] = -torch.inf
            for sequence in ban:
                k = len(sequence) - 1
                if k == 0 or len(generated) >= k and generated[-k:] == sequence[:k]:
                    logits[sequence[-1]] = -torch.inf
                    masked += 1
            token = None
            if rules:
                for size in (64, 512, 4096, len(logits)):
                    for candidate in logits.topk(min(size, len(logits))).indices.tolist():
                        proposed = self.tok.decode(generated + [candidate], skip_special_tokens=True)
                        ending = candidate in self.eos or "\n" in proposed
                        first_line = proposed.split("\n", 1)[0]
                        if not violates(first_line, rules, ending):
                            token = candidate
                            break
                        masked += 1
                    if token is not None:
                        break
                if token is None:
                    raise RuntimeError("Constraint removed every continuation")
            else:
                token = int(logits.argmax())
            if token in self.eos:
                finish_reason = "eos"
                break
            generated.append(token)
            raw = self.tok.decode(generated, skip_special_tokens=True)
            if "\n" in raw:
                finish_reason = "newline"
                break
            states = [self.state([token], cache=cache, adapter=adapter) for cache, _ in states]
        return {"text": self.tok.decode(generated, skip_special_tokens=True), "messages": msgs,
                "completion_tokens": len(generated), "prompt_tokens": len(contexts[0]),
                "forward_branches": len(contexts), "masked_token_attempts": masked,
                "finish_reason": finish_reason, "lambda": lam}


def math_log(value):
    import math
    return math.log(value)
