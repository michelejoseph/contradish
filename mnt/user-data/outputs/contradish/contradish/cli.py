"""
contradish CLI

Usage:
    contradish run evals.yaml --app mymodule:my_app
    contradish compare evals.yaml --baseline mymodule:old_app --candidate mymodule:new_app
"""

import sys
import os
import argparse
import importlib
from typing import Callable


def _load_callable(path: str) -> Callable:
    """Load a callable from 'module:function' string."""
    if ":" not in path:
        raise ValueError(f"App path must be 'module:function', got: {path}")
    module_path, func_name = path.rsplit(":", 1)
    sys.path.insert(0, os.getcwd())
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def cmd_run(args):
    from contradish import RegressionSuite

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    suite = RegressionSuite.load(args.eval_file, api_key=api_key)
    app = _load_callable(args.app)

    # Build a one-sided suite (just run, no comparison)
    from contradish import Suite
    s = Suite(api_key=api_key, app=app)
    for tc in suite.test_cases:
        s.add_test(tc)

    report = s.run(paraphrases=args.paraphrases, verbose=True)
    print(report)

    if report.failed:
        sys.exit(1)


def cmd_compare(args):
    from contradish import RegressionSuite

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    suite = RegressionSuite.load(args.eval_file, api_key=api_key)
    baseline_app = _load_callable(args.baseline)
    candidate_app = _load_callable(args.candidate)

    result = suite.compare(
        baseline_app=baseline_app,
        candidate_app=candidate_app,
        baseline_label=args.baseline_label or args.baseline,
        candidate_label=args.candidate_label or args.candidate,
        paraphrases=args.paraphrases,
    )

    print(result)

    if args.fail_below_consistency or args.fail_below_grounding:
        try:
            result.fail_if_below(
                consistency=args.fail_below_consistency or 0.80,
                grounding=args.fail_below_grounding or 0.80,
            )
        except AssertionError as e:
            print(str(e))
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="contradish",
        description="Reasoning stability testing for LLM applications.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # contradish run
    run_parser = subparsers.add_parser("run", help="Run eval suite against an app")
    run_parser.add_argument("eval_file", help="Path to YAML or JSON eval file")
    run_parser.add_argument("--app", required=True, help="App callable as 'module:function'")
    run_parser.add_argument("--paraphrases", type=int, default=5)

    # contradish compare
    compare_parser = subparsers.add_parser("compare", help="Compare baseline vs candidate")
    compare_parser.add_argument("eval_file", help="Path to YAML or JSON eval file")
    compare_parser.add_argument("--baseline", required=True, help="Baseline callable as 'module:function'")
    compare_parser.add_argument("--candidate", required=True, help="Candidate callable as 'module:function'")
    compare_parser.add_argument("--baseline-label", default=None)
    compare_parser.add_argument("--candidate-label", default=None)
    compare_parser.add_argument("--paraphrases", type=int, default=5)
    compare_parser.add_argument("--fail-below-consistency", type=float, default=None)
    compare_parser.add_argument("--fail-below-grounding", type=float, default=None)

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
