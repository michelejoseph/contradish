"""
CAI Benchmark Evaluation Script
================================
Runs the CAI Semantic Equivalence Benchmark against any supported LLM.
Produces a JSON result file ready to submit to the leaderboard.

Usage:
    # Test Anthropic models
    export ANTHROPIC_API_KEY=sk-ant-...
    python evaluate.py --provider anthropic --model claude-sonnet-4-6
    python evaluate.py --provider anthropic --model claude-haiku-4-5-20251001
    python evaluate.py --provider anthropic --model claude-opus-4-6

    # Test OpenAI models
    export OPENAI_API_KEY=sk-...
    python evaluate.py --provider openai --model gpt-4o
    python evaluate.py --provider openai --model gpt-4o-mini

    # Run all available Anthropic models
    python evaluate.py --provider anthropic --all

    # Fewer paraphrases for a quick check (default is 5)
    python evaluate.py --provider anthropic --model claude-sonnet-4-6 --paraphrases 3

Results are saved to results/<model>_<date>.json
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path


ANTHROPIC_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o1-preview",
    "o3-mini",
]

POLICIES = ["ecommerce", "hr", "healthcare", "legal"]


def make_anthropic_app(model: str, api_key: str):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def app(question: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": question}],
        )
        return msg.content[0].text.strip()

    return app


def make_openai_app(model: str, api_key: str):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def app(question: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": question}],
        )
        return resp.choices[0].message.content.strip()

    return app


def run_benchmark(model: str, provider: str, paraphrases: int = 5, verbose: bool = True) -> dict:
    from contradish import Suite

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY", "")
        if provider == "anthropic"
        else os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        print(f"\n  set {provider.upper()}_API_KEY to run this benchmark\n")
        sys.exit(1)

    if provider == "anthropic":
        app = make_anthropic_app(model, api_key)
    else:
        app = make_openai_app(model, api_key)

    results_by_policy = {}
    all_scores = []
    start = time.time()

    for policy in POLICIES:
        if verbose:
            print(f"\n  policy: {policy}")

        try:
            suite = Suite.from_policy(
                policy=policy,
                app=app,
                api_key=api_key,
                provider=provider,
                verbose=verbose,
            )
            report = suite.run(paraphrases=paraphrases, verbose=verbose)

            policy_results = {
                "cai_score": report.cai_score,
                "cai_strain": round(1 - report.cai_score, 3) if report.cai_score is not None else None,
                "passed": len(report.passed),
                "failed": len(report.failed),
                "total": len(report.results),
                "details": report.to_dict()["results"],
            }
            results_by_policy[policy] = policy_results

            if report.cai_score is not None:
                all_scores.append(report.cai_score)

        except Exception as e:
            print(f"  policy {policy} failed: {e}")
            results_by_policy[policy] = {"error": str(e)}

    elapsed = round(time.time() - start, 1)
    avg_cai_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    avg_cai_strain = round(1 - avg_cai_score, 4) if avg_cai_score is not None else None

    return {
        "model": model,
        "provider": provider,
        "date": str(date.today()),
        "paraphrases": paraphrases,
        "policies_tested": POLICIES,
        "avg_cai_score": avg_cai_score,
        "avg_cai_strain": avg_cai_strain,
        "elapsed_seconds": elapsed,
        "results": results_by_policy,
    }


def print_summary(result: dict) -> None:
    model = result["model"]
    strain = result.get("avg_cai_strain")
    score = result.get("avg_cai_score")

    print(f"\n{'=' * 56}")
    print(f"  model:      {model}")
    print(f"  CAI score:  {score:.4f}" if score else "  CAI score:  n/a")
    print(f"  CAI strain: {strain:.4f}" if strain else "  CAI strain: n/a")
    print(f"  elapsed:    {result['elapsed_seconds']}s")
    print()

    for policy, res in result["results"].items():
        if "error" in res:
            print(f"  {policy:<16} ERROR: {res['error'][:50]}")
        else:
            s = res.get("cai_strain")
            f = res.get("failed", 0)
            t = res.get("total", 0)
            bar = "good" if s < 0.25 else ("ok" if s < 0.50 else "high")
            print(f"  {policy:<16} strain {s:.3f}  [{bar}]  {f}/{t} failures")

    print(f"{'=' * 56}\n")


def save_result(result: dict) -> str:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    safe_model = result["model"].replace("/", "-").replace(":", "-")
    filename = f"{safe_model}_{result['date']}.json"
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return str(path)


def main():
    parser = argparse.ArgumentParser(
        description="Run the CAI benchmark against an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python evaluate.py --provider anthropic --model claude-sonnet-4-6
  python evaluate.py --provider openai --model gpt-4o --paraphrases 3
  python evaluate.py --provider anthropic --all
        """,
    )
    parser.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    parser.add_argument("--model", help="Model name to test")
    parser.add_argument("--all", action="store_true", help="Run all models for this provider")
    parser.add_argument("--paraphrases", type=int, default=5, help="Adversarial phrasings per rule (default: 5)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.print_help()
        sys.exit(1)

    models_to_run = (
        (ANTHROPIC_MODELS if args.provider == "anthropic" else OPENAI_MODELS)
        if args.all
        else [args.model]
    )

    for model in models_to_run:
        print(f"\n  running CAI benchmark: {model}")
        result = run_benchmark(
            model=model,
            provider=args.provider,
            paraphrases=args.paraphrases,
            verbose=not args.quiet,
        )
        print_summary(result)
        path = save_result(result)
        print(f"  result saved: {path}")
        print(f"  submit to leaderboard: open a PR at github.com/michelejoseph1/contradish\n")


if __name__ == "__main__":
    main()
