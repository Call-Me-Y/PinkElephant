from __future__ import annotations

import argparse
import json

from .analysis import analyze
from .common import project_path, read_json
from .dataset import prepare
from .experiment import run


def main():
    parser = argparse.ArgumentParser(description="PinkElephant minimal paired feedback experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, help="JSON config, relative to pinkelephant/ or absolute")
        if name == "run":
            command.add_argument("--limit", type=int, help="First N checkpoints; use a separate output folder")
            command.add_argument("--output", help="Override result directory")
    report = sub.add_parser("analyze")
    report.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "analyze":
        result = analyze(args.output)
    else:
        config = read_json(project_path(args.config))
        if args.command == "prepare":
            result = prepare(config)
        else:
            if args.limit is not None and args.limit < 1:
                parser.error("--limit must be positive")
            result = run(config, args.limit, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))

