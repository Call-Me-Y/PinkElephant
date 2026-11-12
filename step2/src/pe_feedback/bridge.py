"""Read-only reuse of step1 adapters and the upstream ToolShed environment."""
import importlib.util
import sys

from .common import PE

sys.dont_write_bytecode = True
sys.path.insert(0, str(PE / "step1" / "src"))
from pinkelephant.ftb import Task, HARNESSES, TOOLS_BY_NAME, build_messages, canonical, extract_action, make_env, parse_action, revision, trajectory_step
from pinkelephant.feedback import render as legacy_render

_audit = None


def strict_score(env, task):
    """Evaluator only. Its private Task/world never enters the monitor."""
    global _audit
    if _audit is None:
        spec = importlib.util.spec_from_file_location("pe_step1_goal_audit", PE / "step1" / "scripts" / "summarize_interventions.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _audit = module.extra_goal_audit
    return bool(env.is_solved(task) and _audit(env.world(task), task))


def dependencies():
    from .common import digest
    files = [PE / "step1" / "src" / "pinkelephant" / name for name in ("ftb.py", "common.py", "feedback.py", "backends.py", "experiment.py")]
    files += [PE / "step1" / "scripts" / "summarize_interventions.py"]
    return {"upstream": revision(), "step1_sources": {str(p.relative_to(PE)): digest(p.read_text(encoding="utf-8")) for p in files}}

