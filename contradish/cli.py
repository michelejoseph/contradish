"""
contradish CLI

    contradish run evals.yaml --app mymodule:my_app_function
"""

import sys
import os
import argparse
import importlib
import json
from typing import Callable


def _load_callable(path: str) -> Callable:
    if ":" not in path:
        raise ValueError(f"App must be specified as 'module:function', got: {path!r}")
    module_str, func_str = path.rsplit(":", 1)
    sys.path.insert(0, os.getcwd())
    module = importlib.import_module(module_str)
    return getattr(module, func_str)


def _load_cases(path: str):
    from contradish import TestCase
    with open(path) as f:
        raw = f.read()

    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
            data = yaml.safe_load(raw)
        except ImportError:
            sys.exit("Install pyyaml to use YAML files:  pip install pyyaml")
        items = data.get("test_cases", data) if isinstance(data, dict) else data
    else:
        items = json.loads(raw)
        if isinstance(items, dict):
            items = items.get("test_cases", [])

    return [
        TestCase(
            input=tc["input"],
            name=tc.get("name"),
            expected_traits=tc.get("expected_traits", []),
        )
        for tc in items
    ]


def cmd_run(args):
    from contradish import Suite
    app    = _load_callable(args.app)
    cases  = _load_cases(args.eval_file)
    suite  = Suite(app=app)
    for tc in cases:
        suite.add(tc)
    report = suite.run(paraphrases=args.paraphrases)
    sys.exit(1 if report.failed else 0)


def main():
    parser = argparse.ArgumentParser(
        prog="contradish",
        description="Reasoning stability testing for LLM applications.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run eval suite against an app")
    run_p.add_argument("eval_file",    help="Path to YAML or JSON eval file")
    run_p.add_argument("--app",        required=True, help="App callable as module:function")
    run_p.add_argument("--paraphrases", type=int, default=5)

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
