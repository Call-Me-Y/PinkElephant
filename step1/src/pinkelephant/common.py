from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PE_ROOT = PROJECT.parent


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

