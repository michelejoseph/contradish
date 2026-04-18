"""
CAI Benchmark Evaluation Script
================================
Runs the CAI Semantic Equivalence Benchmark against any supported LLM.
Produces a JSON result file ready to submit to the leaderboard.

By default, uses the frozen v1 benchmark dataset: pre-generated adversarial
questions committed to the repo. Results are reproducible and comparable
across models and runs.

Use --live to generate fresh adversarial questions at runtime instead
(useful for development; scores will not be directly comparable).

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

    # Use live question generation instead of frozen benchmark
    python evaluate.py --provider anthropic --model claude-sonnet-4-6 --live

Results are saved to results/<model>_<date>.json
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional


BENCHMARK_VERSION = "v1"

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

POLICIES = ["ecommerce", "hr", "healthcare", "legal", "finance", "saas", "insurance", "education"]

BENCHMARK_DIR = Path(__file__).parent / "contradish" / "benchmarks" / BENCHMARK_VERSION


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


def load_frozen_policy(policy: str) -> dict:
    path = BENCHMARK_DIR / f"{policy}.json"
    if not path.exists():
        raise FileNotFoundError(f"Frozen benchmark not found: {path}")
    with open(path) as f:
        return json.load(f)


def run_frozen_policy(policy: str, app, judge, verbose: bool) -> dict:
    """Run a policy using the frozen benchmark dataset."""
    data = load_frozen_policy(policy)
    cases = data["cases"]

    if verbose:
        print(f"  loaded {data['display_name']} ({len(cases)} test cases, frozen {BENCHMARK_VERSION})")

    all_scores = []
    details = []

    for i, case in enumerate(cases, 1):
        name = case["name"]
        original = case["original"]
        adversarial = case["adversarial"]

        if verbose:
            print(f"\n[{i}/{len(cases)}]  testing \"{name}\"")
            print(f"  querying app {1 + len(adversarial)}x")

        inputs = [original] + adversarial
        outputs = []
        for inp in inputs:
            try:
                outputs.append(str(app(inp)).strip())
            except Exception as e:
                outputs.append(f"[APP ERROR: {e}]")

        if verbose:
            print("  scoring consistency")

        result = judge.evaluate_consistency(original, inputs, outputs)
        score = result.get("consistency_score", 0.5)
        all_scores.append(score)

        passed = score >= 0.75
        details.append({
            "id":            case["id"],
            "name":          name,
            "cai_score":     round(score, 4),
            "passed":        passed,
            "disagreements": result.get("disagreements", []),
            "summary":       result.get("summary", ""),
        })

    avg = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    n_passed = sum(1 for d in details if d["passed"])

    return {
        "cai_score":  avg,
        "cai_strain": round(1 - avg, 4) if avg is not None else None,
        "passed":     n_passed,
        "failed":     len(details) - n_passed,
        "total":      len(details),
        "details":    details,
    }


def run_live_policy(policy: str, app, api_key: str, provider: str, paraphrases: int, verbose: bool) -> dict:
    """Run a policy with live adversarial question generation (not reproducible)."""
    from contradish import Suite

    suite = Suite.from_policy(
        policy=policy,
        app=app,
        api_key=api_key,
        provider=provider,
        verbose=verbose,
    )
    report = suite.run(paraphrases=paraphrases, verbose=verbose)

    return {
        "cai_score":  report.cai_score,
        "cai_strain": round(1 - report.cai_score, 4) if report.cai_score is not None else None,
        "passed":     len(report.passed),
        "failed":     len(report.failed),
        "total":      len(report.results),
        "details":    report.to_dict()["results"],
    }


def run_benchmark(
    model: str,
    provider: str,
    use_frozen: bool = True,
    paraphrases: int = 5,
    judge_provider: Optional[str] = None,
    verbose: bool = True,
) -> dict:
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

    judge = None
    judge_provider_used = None
    judge_model_used = None
    if use_frozen:
        from contradish.judge import Judge
        from contradish.llm import LLMClient
        judge_key = os.environ.get(
            "JUDGE_ANTHROPIC_API_KEY" if judge_provider == "anthropic" else "JUDGE_OPENAI_API_KEY",
            ""
        ).strip() or None
        llm_judge = LLMClient.make_judge_client(
            model_provider=provider,
            judge_provider=judge_provider,
            judge_api_key=judge_key,
        )
        judge = Judge(llm_judge)
        judge_provider_used = llm_judge.provider
        judge_model_used = llm_judge.judge_model

    results_by_policy = {}
    all_scores = []
    start = time.time()

    for policy in POLICIES:
        if verbose:
            print(f"\n  policy: {policy}")

        try:
            if use_frozen:
                res = run_frozen_policy(policy, app, judge, verbose)
            else:
                res = run_live_policy(policy, app, api_key, provider, paraphrases, verbose)

            results_by_policy[policy] = res
            if res["cai_score"] is not None:
                all_scores.append(res["cai_score"])

        except Exception as e:
            print(f"  policy {policy} failed: {e}")
            results_by_policy[policy] = {"error": str(e)}

    elapsed = round(time.time() - start, 1)
    avg_cai_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    avg_cai_strain = round(1 - avg_cai_score, 4) if avg_cai_score is not None else None

    independent_judging = (
        judge_provider_used is not None and judge_provider_used != provider
    )

    return {
        "model":               model,
        "provider":            provider,
        "date":                str(date.today()),
        "benchmark_version":   BENCHMARK_VERSION if use_frozen else "live",
        "mode":                "frozen" if use_frozen else "live",
        "judge_provider":      judge_provider_used,
        "judge_model":         judge_model_used,
        "independent_judging": independent_judging,
        "policies_tested":     POLICIES,
        "avg_cai_score":       avg_cai_score,
        "avg_cai_strain":      avg_cai_strain,
        "elapsed_seconds":     elapsed,
        "results":             results_by_policy,
    }


def print_summary(result: dict) -> None:
    model = result["model"]
    strain = result.get("avg_cai_strain")
    score = result.get("avg_cai_score")
    mode = result.get("mode", "frozen")
    version = result.get("benchmark_version", BENCHMARK_VERSION)

    judge_prov = result.get("judge_provider", "?")
    judge_mod  = result.get("judge_model", "?")
    independent = result.get("independent_judging", False)
    judge_note = f"{judge_prov}/{judge_mod}" + (" [independent]" if independent else " [SAME PROVIDER]")

    print(f"\n{'=' * 60}")
    print(f"  model:      {model}")
    print(f"  benchmark:  CAI-Bench {version} ({mode})")
    print(f"  judge:      {judge_note}")
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

    print(f"{'=' * 60}\n")


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
  python evaluate.py --provider openai --model gpt-4o
  python evaluate.py --provider anthropic --all
  python evaluate.py --provider anthropic --model claude-sonnet-4-6 --live
        """,
    )
    parser.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    parser.add_argument("--model", help="Model name to test")
    parser.add_argument("--all", action="store_true", help="Run all models for this provider")
    parser.add_argument("--live", action="store_true", help="Generate adversarial questions live instead of using frozen benchmark")
    parser.add_argument("--paraphrases", type=int, default=5, help="Adversarial phrasings per rule when using --live (default: 5)")
    parser.add_argument("--judge-provider", choices=["anthropic", "openai"], default=None, help="Provider for the judge model (default: opposite of --provider for independent judging)")
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
        mode = "live" if args.live else f"frozen {BENCHMARK_VERSION}"
        print(f"\n  running CAI benchmark: {model} ({mode})")
        result = run_benchmark(
            model=model,
            provider=args.provider,
            use_frozen=not args.live,
            paraphrases=args.paraphrases,
            judge_provider=args.judge_provider,
            verbose=not args.quiet,
        )
        print_summary(result)
        path = save_result(result)
        print(f"  result saved: {path}")
        print(f"  submit to leaderboard: open a PR at github.com/michelejoseph1/contradish\n")


if __name__ == "__main__":
    main()
