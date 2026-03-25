"""
contradish CLI

Usage:
    # From a system prompt string directly
    contradish "You are a support agent. Refunds within 30 days only."

    # From a file
    contradish --prompt system_prompt.txt --app mymodule:my_app

    # Manual test cases from YAML
    contradish run evals.yaml --app mymodule:my_app
"""

import sys
import os
import argparse
import importlib
import json
from typing import Callable, Optional


def _load_callable(path: str) -> Callable:
    if ":" not in path:
        raise ValueError(
            f"App must be 'module:function', got: {path!r}\n"
            f"Example: mymodule:my_app_function"
        )
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


def _check_api_key():
    """Give a clear error if no API key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("\n  contradish needs an API key.\n")
        print("  Set one of:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    export OPENAI_API_KEY=sk-...\n")
        sys.exit(1)


def _output_json(report) -> None:
    """Print the report as JSON to stdout."""
    print(json.dumps(report.to_dict(), indent=2))


def _make_demo_app(system_prompt: str):
    """Return a demo app callable that uses the configured LLM with the given system prompt."""
    from contradish.llm import LLMClient
    llm = LLMClient()

    def demo_app(question: str) -> str:
        if llm.provider == "anthropic":
            msg = llm._client.messages.create(
                model=llm.fast_model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
            )
            return msg.content[0].text.strip()
        else:
            resp = llm._client.chat.completions.create(
                model=llm.fast_model,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )
            return resp.choices[0].message.content.strip()

    return demo_app, llm.provider


def cmd_policy(args):
    """Run a prebuilt domain policy pack — no system prompt required."""
    from contradish import Suite
    from contradish.policies import load_policy, list_policies
    from contradish.printer import print_next_steps

    _check_api_key()

    policy_name = args.policy
    use_json    = getattr(args, "json", False)

    # Validate policy name early for a good error message
    try:
        pack = load_policy(policy_name)
    except ValueError as e:
        print(f"\n  {e}\n")
        print(f"  Available: {', '.join(list_policies())}\n")
        sys.exit(1)

    if not use_json:
        print(f"\n  {pack.display_name} policy pack  "
              f"({len(pack.cases)} test cases)\n")
        print(f"  {pack.description}\n")

    if args.app:
        app = _load_callable(args.app)
    else:
        # Demo mode: create a generic assistant that answers the test questions
        # without a real system prompt — surfaces what the model does by default
        demo_system = (
            f"You are a helpful assistant for a {pack.display_name.lower()} context. "
            f"Answer user questions clearly and accurately."
        )
        if not use_json:
            print("  demo mode: no --app provided, testing default LLM behavior")
            print("  add --app mymodule:fn to test your actual app\n")
        app, _ = _make_demo_app(demo_system)

    suite = Suite.from_policy(
        policy=policy_name,
        app=app,
        verbose=not use_json,
    )
    report = suite.run(paraphrases=args.paraphrases, verbose=not use_json)

    if use_json:
        _output_json(report)
    else:
        print_next_steps(report)

    sys.exit(1 if report.failed else 0)


def cmd_from_prompt(args):
    """Run from a system prompt — the zero-config path."""
    from contradish import Suite
    from contradish.printer import print_start, print_next_steps

    _check_api_key()

    # Load prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            system_prompt = f.read()
    else:
        system_prompt = args.system_prompt

    if not system_prompt or not system_prompt.strip():
        print("\n  No system prompt provided.\n")
        print("  Usage: contradish \"Your system prompt here\"")
        print("     or: contradish --prompt system_prompt.txt\n")
        sys.exit(1)

    use_json = getattr(args, "json", False)

    # Load app if provided, otherwise use a passthrough demo app
    if args.app:
        app = _load_callable(args.app)
        if not use_json:
            print_start(system_prompt)
    else:
        if not use_json:
            print_start(system_prompt)
            print("  demo mode: testing your prompt against itself")
            print("  add --app mymodule:fn to test your actual app\n")
        app, _ = _make_demo_app(system_prompt)

    suite = Suite.from_prompt(
        system_prompt=system_prompt,
        app=app,
        verbose=not use_json,
    )
    report = suite.run(paraphrases=args.paraphrases, verbose=not use_json)

    if use_json:
        _output_json(report)
    else:
        print_next_steps(report)

    sys.exit(1 if report.failed else 0)


def cmd_run(args):
    """Run manual test cases from a YAML or JSON file."""
    from contradish import Suite
    from contradish.printer import print_next_steps

    _check_api_key()
    use_json = getattr(args, "json", False)
    app      = _load_callable(args.app)
    cases    = _load_cases(args.eval_file)
    suite    = Suite(app=app)
    for tc in cases:
        suite.add(tc)
    report = suite.run(paraphrases=args.paraphrases, verbose=not use_json)
    if use_json:
        _output_json(report)
    else:
        print_next_steps(report)
    sys.exit(1 if report.failed else 0)


def cmd_compare(args):
    """Compare baseline vs candidate app for CAI regression."""
    from contradish import RegressionSuite

    _check_api_key()
    use_json = getattr(args, "json", False)

    baseline_app  = _load_callable(args.baseline_app)
    candidate_app = _load_callable(args.candidate_app)

    suite = RegressionSuite.load(args.eval_file)
    result = suite.compare(
        baseline_app=baseline_app,
        candidate_app=candidate_app,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        paraphrases=args.paraphrases,
        verbose=not use_json,
    )

    if use_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result)

    try:
        result.fail_if_below(consistency=args.threshold)
    except AssertionError as e:
        print(f"\n  FAIL: {e}\n")
        sys.exit(1)

    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        prog="contradish",
        description="CAI testing for LLM applications. Detects CAI failures and returns a CAI score per rule.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # run a prebuilt policy pack — no system prompt needed
  contradish --policy ecommerce --app mymodule:my_app
  contradish --policy hr --app mymodule:my_app
  contradish --policy healthcare
  contradish --policy legal

  # test your system prompt directly (demo mode — uses your API key as the app)
  contradish "You are a support agent. Refunds within 30 days only."

  # test from a prompt file
  contradish --prompt system_prompt.txt

  # test your own app with a prompt file
  contradish --prompt system_prompt.txt --app mymodule:my_app_function

  # run manual test cases from a YAML file
  contradish run evals.yaml --app mymodule:my_app_function

  # regression: compare baseline vs candidate (CI/CD gate)
  contradish compare evals.yaml --baseline mymodule:old_app --candidate mymodule:new_app
  contradish compare evals.yaml --baseline mymodule:old_app --candidate mymodule:new_app --threshold 0.80
        """,
    )

    sub = parser.add_subparsers(dest="command")

    # Default: contradish "prompt" or contradish --prompt file.txt
    parser.add_argument(
        "system_prompt",
        nargs="?",
        help="System prompt string to test directly",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_file",
        metavar="FILE",
        help="Path to a file containing your system prompt",
    )
    parser.add_argument(
        "--policy",
        metavar="PACK",
        help=(
            "Prebuilt domain test suite. No system prompt needed. "
            "Options: ecommerce, hr, healthcare, legal"
        ),
    )
    parser.add_argument(
        "--app",
        metavar="MODULE:FUNCTION",
        help="Your app callable. If omitted, uses your API key's LLM in demo mode.",
    )
    parser.add_argument(
        "--paraphrases",
        type=int,
        default=5,
        metavar="N",
        help="Number of paraphrases per test case (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output report as JSON instead of terminal format. Includes cai_score at report and rule level.",
    )

    # contradish run evals.yaml --app module:fn
    run_p = sub.add_parser("run", help="Run manual test cases from a YAML/JSON file")
    run_p.add_argument("eval_file", help="Path to YAML or JSON eval file")
    run_p.add_argument("--app", required=True, metavar="MODULE:FUNCTION")
    run_p.add_argument("--paraphrases", type=int, default=5, metavar="N")
    run_p.add_argument("--json", action="store_true", default=False,
                       help="Output report as JSON")

    # contradish compare evals.yaml --baseline mod:fn --candidate mod:fn
    cmp_p = sub.add_parser(
        "compare",
        help="Compare baseline vs candidate for CAI regression (CI/CD gate)",
    )
    cmp_p.add_argument("eval_file",
                       help="YAML or JSON file with test cases")
    cmp_p.add_argument("--baseline",
                       dest="baseline_app",
                       required=True,
                       metavar="MODULE:FUNCTION",
                       help="Baseline (current production) app callable")
    cmp_p.add_argument("--candidate",
                       dest="candidate_app",
                       required=True,
                       metavar="MODULE:FUNCTION",
                       help="Candidate (new version) app callable")
    cmp_p.add_argument("--baseline-label",
                       default="baseline",
                       metavar="LABEL",
                       help="Human-readable label for baseline (default: baseline)")
    cmp_p.add_argument("--candidate-label",
                       default="candidate",
                       metavar="LABEL",
                       help="Human-readable label for candidate (default: candidate)")
    cmp_p.add_argument("--threshold",
                       type=float,
                       default=0.75,
                       metavar="FLOAT",
                       help="Min CAI score for candidate to pass (default: 0.75)")
    cmp_p.add_argument("--paraphrases", type=int, default=5, metavar="N")
    cmp_p.add_argument("--json", action="store_true", default=False,
                       help="Output report as JSON")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif getattr(args, "policy", None):
        cmd_policy(args)
    elif args.system_prompt or args.prompt_file:
        cmd_from_prompt(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
