"""Existing local checkpoints or a compatible chat endpoint; no downloads."""
import os
os.environ.setdefault("USE_TF", "0")

from .bridge import PE


class Actor:
    def __init__(self, spec):
        self.spec = spec
        if spec["kind"] == "chat_api":
            from pinkelephant.backends import ChatAPIBackend
            self.api = ChatAPIBackend(spec)
            self.tokenizer = None
            return
        import torch
        from pinkelephant.backends import TransformersBackend
        backend = TransformersBackend(spec)
        self.net, self.tokenizer = backend.model, backend.tokenizer
        if spec.get("dtype") == "float16":
            self.net.half()
        self.net.requires_grad_(False)
        self.api = None
        self.torch = torch

    def measure(self, text):
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Explicit conservative byte budget for APIs without local tokenizers.
        return len(text.encode("utf-8"))

    def generate_batch(self, messages_list, seed=42):
        if self.api:
            return [self.api.generate(msgs, seed, {}) for msgs in messages_list]
        torch, tok = self.torch, self.tokenizer
        contexts = [tok.encode(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True), add_special_tokens=False) for m in messages_list]
        longest = max(map(len, contexts))
        maximum = self.spec.get("max_new_tokens", 96)
        if longest + maximum > self.spec.get("context_limit", 8192):
            raise RuntimeError("Context budget exceeded")
        with torch.inference_mode():
            ids = torch.full((len(contexts), longest), tok.pad_token_id, dtype=torch.long, device=self.net.device)
            mask = torch.zeros_like(ids)
            for i, ctx in enumerate(contexts):
                ids[i, -len(ctx):] = torch.tensor(ctx, device=self.net.device)
                mask[i, -len(ctx):] = 1
            position = mask.cumsum(-1) - 1
            position.masked_fill_(mask == 0, 0)
            cache = None
            done = [False] * len(contexts)
            tokens = [[] for _ in contexts]
            reasons = ["length"] * len(contexts)
            for _ in range(maximum):
                output = self.net.model(input_ids=ids, attention_mask=mask, position_ids=position, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = self.net.lm_head(output.last_hidden_state[:, -1]).float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite generation logits")
                chosen = logits.argmax(-1).tolist()
                keep = []
                for i, token in enumerate(chosen):
                    if not done[i]:
                        if token == tok.eos_token_id:
                            done[i], reasons[i] = True, "eos"
                        else:
                            tokens[i].append(token)
                            if "\n" in tok.decode(tokens[i], skip_special_tokens=True):
                                done[i], reasons[i] = True, "newline"
                    keep.append(0 if done[i] else 1)
                if all(done):
                    break
                ids = torch.tensor(chosen, device=self.net.device)[:, None]
                mask = torch.cat((mask, torch.tensor(keep, device=self.net.device)[:, None]), -1)
                position = (mask.sum(-1) - 1).clamp_min(0)[:, None]
        return [{"text": tok.decode(row, skip_special_tokens=True), "prompt_tokens": len(contexts[i]),
                 "completion_tokens": len(row), "finish_reason": reasons[i]} for i, row in enumerate(tokens)]
