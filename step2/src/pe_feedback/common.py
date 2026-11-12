import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PE = ROOT.parent


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read_json(value):
    return json.loads(Path(value).read_text(encoding="utf-8-sig"))


def read_rows(value):
    return [json.loads(line) for line in Path(value).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(value, data):
    p = Path(value)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append(value, data):
    with Path(value).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

