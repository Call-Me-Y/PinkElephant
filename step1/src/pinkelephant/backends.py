from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


class SmokeBackend:
    """Test double: follows a private reference tail, independent of feedback.

    Its perfect behavior is useful for checking accounting, never for research.
    """
    name = "smoke_reference"

    def generate(self, messages, seed, context):
        tail = context["reference_tail"]
        index = context["generated_index"]
        text = tail[index] if index < len(tail) else "finish()"
        return {"text": text, "prompt_tokens": None, "completion_tokens": None,
                "finish_reason": "smoke", "model": self.name, "usage_available": False}


class ChatAPIBackend:
    """A small text-action client for OpenAI-compatible chat-completions servers."""
    def __init__(self, spec):
        self.spec = spec
        self.base_url = os.environ.get(spec.get("base_url_env", "PE_BASE_URL"), spec.get("base_url", "")).rstrip("/")
        self.model = os.environ.get(spec.get("model_env", "PE_MODEL"), spec.get("model", ""))
        if not self.base_url.startswith(("https://", "http://")) or not self.model:
            raise ValueError("Set PE_BASE_URL (ending in /v1) and PE_MODEL, or fill the backend config.")
        self.key = os.environ.get(spec.get("api_key_env", "PE_API_KEY"), "")
        host = urllib.parse.urlparse(self.base_url).hostname
        self.local = host in {"127.0.0.1", "localhost", "::1"}
        if not self.local and not self.key:
            raise ValueError("Hosted endpoint requires the configured API key environment variable.")
        # A desktop proxy commonly breaks localhost. Remote endpoints retain env proxies.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if self.local else urllib.request.build_opener()

    def generate(self, messages, seed, context):
        spec = self.spec
        payload = {"model": self.model, "messages": messages,
                   "temperature": spec.get("temperature", 0),
                   spec.get("token_limit_field", "max_tokens"): spec.get("max_new_tokens", 128)}
        if spec.get("send_seed", True):
            payload["seed"] = seed
        payload.update(spec.get("extra_body", {}))
        # Keys that would change the action task are explicitly reserved.
        if payload["messages"] != messages or payload["model"] != self.model:
            raise ValueError("extra_body cannot replace messages/model")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        try:
            with self.opener.open(request, timeout=spec.get("timeout_seconds", 120)) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}; check model, credentials, seed and token-limit field. No automatic retry.") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model endpoint connection failed: {exc.reason}") from None
        choice = result["choices"][0]
        text = choice["message"].get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Endpoint returned no text action. Check the model's thinking/output settings and output token budget.")
        usage = result.get("usage") or {}
        return {"text": text, "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"), "finish_reason": choice.get("finish_reason"),
                "model": result.get("model", self.model), "response_id": result.get("id"),
                "system_fingerprint": result.get("system_fingerprint"), "usage_available": bool(usage)}


class TransformersBackend:
    def __init__(self, spec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.spec = spec
        torch.set_num_threads(spec.get("cpu_threads", 4))
        model_dir = Path(os.environ.get("MODEL_DIR", "C:/Projects/AI/models"))
        path = Path(spec["model_path"])
        if not path.is_absolute():
            path = model_dir / path
        if not path.is_dir():
            raise ValueError(f"Local model directory not found: {path}")
        self.path = str(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype="auto")
        self.model.to(spec.get("device", "cpu")).eval()

    def generate(self, messages, seed, context):
        torch, spec, tok = self.torch, self.spec, self.tokenizer
        torch.manual_seed(seed)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **spec.get("chat_template_kwargs", {}))
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        length = inputs.input_ids.shape[1]
        if length + spec.get("max_new_tokens", 96) > spec.get("context_limit", 8192):
            raise RuntimeError("Context budget exceeded; no silent truncation is allowed.")
        kwargs = {"max_new_tokens": spec.get("max_new_tokens", 96), "pad_token_id": tok.eos_token_id,
                  "do_sample": spec.get("temperature", 0) > 0}
        if kwargs["do_sample"]:
            kwargs["temperature"] = spec["temperature"]
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        new = output[0, length:]
        return {"text": tok.decode(new, skip_special_tokens=True), "prompt_tokens": length,
                "completion_tokens": len(new), "finish_reason": "length" if len(new) == kwargs["max_new_tokens"] else "stop",
                "model": self.path, "usage_available": True}


def make_backend(spec):
    kinds = {"smoke": SmokeBackend, "chat_api": ChatAPIBackend, "transformers": TransformersBackend}
    kind = spec["kind"]
    return kinds[kind]() if kind == "smoke" else kinds[kind](spec)

