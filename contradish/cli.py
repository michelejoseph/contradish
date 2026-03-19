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


def cmd_from_prompt(args):
    """Run from a system prompt — the zero-config path."""
    from contradish import Suite

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

    # Load app if provided, otherwise use a passthrough demo app
    if args.app:
        app = _load_callable(args.app)
    else:
        # Demo mode: use the judge LLM as the app itself
        # so developers can test contradish without wiring up their own app
        from contradish.llm import LLMClient
        llm = LLMClient()

        def demo_app(question: str) -> str:
            """Uses the configured LLM with the provided system prompt."""
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

        app = demo_app

    suite = Suite.from_prompt(
        system_prompt=system_prompt,
        app=app,
        verbose=True,
    )
    report = suite.run(paraphrases=args.paraphrases)
    sys.exit(1 if report.failed else 0)


def cmd_run(args):
    """Run manual test cases from a YAML or JSON file."""
    from contradish import Suite

    _check_api_key()
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # test your system prompt directly (demo mode — uses your API key as the app)
  contradish "You are a support agent. Refunds within 30 days only."

  # test from a prompt file
  contradish --prompt system_prompt.txt

  # test your own app with a prompt file
  contradish --prompt system_prompt.txt --app mymodule:my_app_function

  # run manual test cases from a YAML file
  contradish run evals.yaml --app mymodule:my_app_function
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

    # contradish run evals.yaml --app module:fn
    run_p = sub.add_parser("run", help="Run manual test cases from a YAML/JSON file")
    run_p.add_argument("eval_file", help="Path to YAML or JSON eval file")
    run_p.add_argument("--app", required=True, metavar="MODULE:FUNCTION")
    run_p.add_argument("--paraphrases", type=int, default=5, metavar="N")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.system_prompt or args.prompt_file:
        cmd_from_prompt(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
