"""Read-only adapter to the checked-out FTB environment; no model dependencies."""
from __future__ import annotations

import re
import subprocess
import sys

from .common import PE_ROOT, digest

UPSTREAM = PE_ROOT / "literature" / "feedback-that-backfires"
if not (UPSTREAM / "src" / "slmecho").is_dir():
    raise RuntimeError(f"FTB checkout not found: {UPSTREAM}")
sys.path.insert(0, str(UPSTREAM / "src"))

from slmecho.envs.base import Task
from slmecho.envs.toolshed import TOOLS_BY_NAME, ToolShedEnv, parse_action
from slmecho.envs.toolshed_tasks import generate_tasks, perturbations
from slmecho.harness import HARNESSES, TrajStep, build_messages


def revision():
    result = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    paths = ["envs/base.py", "envs/toolshed.py", "envs/toolshed_tasks.py", "seeding.py", "harness.py"]
    source_hash = digest({p: (UPSTREAM / "src" / "slmecho" / p).read_text(encoding="utf-8") for p in paths})
    return {"path": str(UPSTREAM), "commit": result.stdout.strip(), "tracked_changes": status.stdout.splitlines(),
            "reused_source_sha256": source_hash}


def canonical(action):
    # Matches slmecho.agent.canonical_action without importing torch/scoring.
    try:
        parsed = parse_action(action)
        args = ",".join(f"{key}={value!r}" for key, value in sorted(parsed.kwargs.items()))
        pos = ",".join(repr(value) for value in parsed.positional)
        return f"{parsed.name}({pos}|{args})"
    except Exception:
        return re.sub(r"\s+", " ", action.strip())


def trajectory_step(action, outcome):
    try:
        tool = parse_action(action).name
    except Exception:
        tool = None
    return TrajStep(action, outcome.observation, outcome.ok, outcome.error_type, tool)


def extract_action(text):
    # Keep malformed/prose output as a genuine failed action. Never repair it.
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        return re.sub(r"^(?:Action|Tool|Call)\s*:\s*", "", line, flags=re.I)
    return text.strip()


def make_env():
    return ToolShedEnv(verbosity="standard", echo_action=True, demo=True)

