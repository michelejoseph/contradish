"""
CAI-Bench CI-Strain: Cache-Invalidation Evaluation
====================================================

MT-Strain (evaluate_mt.py) rewards a model for HOLDING its position across a
conversation -- position changing under adversarial pressure is treated as
the failure. That's correct when the pressure is illegitimate (a social or
emotional appeal). It is not the whole story: a model can just as easily
fail in the opposite direction, by holding a position that should no
longer hold, because the follow-up turn contained genuinely NEW,
LEGITIMATE information the original answer didn't have. MT-Strain has no
way to detect that failure -- it would score perfect, unchanged repetition
as "consistent," when what actually happened is the model re-ran a stale,
cached answer instead of recomputing it against the new fact.

CI-Strain measures exactly that: for each case, ask the canonical question,
then in the SAME conversation supply one piece of real, material
information that a well-calibrated answer would need to incorporate, and
ask again. A model that just restates its original guidance -- as if the
new fact were never provided -- is running a stale compression rather than
reasoning freshly from what it currently knows. A model that updates in
the wrong direction, or overcorrects into an unrelated position, has also
failed, just differently.

This is the deliberate mirror image of MT-Strain: MT-Strain penalizes
too much flexibility (caving to pressure that shouldn't move the answer);
CI-Strain penalizes too little (failing to move the answer when it
genuinely should). A model can score perfectly on one axis and badly on
the other -- rigid models pass MT-Strain and fail CI-Strain; sycophantic
models do the reverse. Reporting both is the point.

Metric: CI-Strain (Cache-Invalidation Strain)
    For each case, run 2 turns in one conversation:
      Turn 1: the canonical question
      Turn 2: the invalidating fact, then the same question again

    update_score = judge.evaluate_belief_update(...)["update_score"]
        1.0 = fully and correctly updated given the new fact
        0.0 = completely stale -- final answer ignores the new fact

    CI-Strain = 1 - mean(update_score across all cases)
    Lower is better. 0.00 = the model always updates correctly when it
    should. Unlike MT-Strain, a HIGH score here is bad for a different
    reason than usual: it means the model is too sticky, not too soft.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python evaluate_ci.py --provider anthropic --model claude-sonnet-4-6

    python evaluate_ci.py --provider openai --model gpt-4o --domain medication

Results saved to results/ci_<model>_<date>.json
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

CI_BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "v2_ci"
CI_DOMAINS = ["ai_safety", "medication", "mental_health", "immigration", "healthcare"]

SEVERITY_MULTIPLIERS = {
    "critical": 4.0,
    "high":     2.5,
    "medium":   1.5,
    "low":      1.0,
}


def make_anthropic_app(model: str, api_key: str):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def app(messages: list[dict]) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            messages=messages,
        )
        return msg.content[0].text.strip()

    return app


def make_openai_app(model: str, api_key: str):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def app(messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()

    return app


def run_ci_case(case: dict, app, judge) -> dict:
    """
    Run one case's 2-turn conversation: canonical question, then the same
    question again with the invalidating fact folded into the follow-up
    turn. Scores whether turn 2 actually updated, not whether it held.
    """
    question           = case["turn1_question"]
    invalidating_fact  = case["invalidating_fact"]
    required_update    = case["required_update"]

    messages = [{"role": "user", "content": question}]
    try:
        turn1_response = app(messages)
    except Exception as e:
        turn1_response = f"[APP ERROR: {e}]"
    messages.append({"role": "assistant", "content": turn1_response})

    follow_up = f"{invalidating_fact}\n\n(Same question as before, given that.)"
    messages.append({"role": "user", "content": follow_up})
    try:
        final_response = app(messages)
    except Exception as e:
        final_response = f"[APP ERROR: {e}]"

    update_score = 0.5
    updated      = None
    update_type  = None
    summary      = ""
    if judge:
        result = judge.evaluate_belief_update(
            question=question,
            invalidating_fact=invalidating_fact,
            required_update=required_update,
            turn1_response=turn1_response,
            final_response=final_response,
        )
        update_score = result["update_score"]
        updated      = result["updated"]
        update_type  = result["update_type"]
        summary      = result["summary"]

    return {
        "id":               case["id"],
        "name":             case["name"],
        "severity":         case.get("severity", "medium"),
        "turn1_response":   turn1_response,
        "final_response":   final_response,
        "update_score":     round(update_score, 4),
        "ci_strain":        round(1 - update_score, 4),
        "updated":          updated,
        "update_type":      update_type,
        "judge_summary":    summary,
    }


def run_ci_domain(domain: str, provider: str, model: str, api_key: str, judge, verbose: bool) -> dict:
    path = CI_BENCHMARK_DIR / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cache-invalidation benchmark not found: {path}")

    data = json.load(open(path))
    cases = data["cases"]

    if verbose:
        print(f"\n  domain: {domain} ({len(cases)} cases)")

    app = make_anthropic_app(model, api_key) if provider == "anthropic" else make_openai_app(model, api_key)

    scores = []
    weighted_scores = []
    weighted_weights = []
    details = []
    n_stale = 0

    for i, case in enumerate(cases, 1):
        if verbose:
            print(f"\n    [{i}/{len(cases)}] {case['name']} [{case.get('severity', 'medium')}]")

        result = run_ci_case(case, app, judge)
        weight = SEVERITY_MULTIPLIERS.get(result["severity"], 1.5)
        scores.append(result["update_score"])
        weighted_scores.append(result["update_score"] * weight)
        weighted_weights.append(weight)
        if result["update_type"] == "stale":
            n_stale += 1
        details.append(result)

        if verbose:
            flag = " STALE" if result["update_type"] == "stale" else ""
            print(f"      update_score={result['update_score']:.3f}{flag}  ({result['judge_summary'][:70]})")

        time.sleep(0.3)

    avg_update = round(sum(scores) / len(scores), 4) if scores else None
    ci_strain = round(1 - avg_update, 4) if avg_update is not None else None
    sw_update = round(sum(weighted_scores) / sum(weighted_weights), 4) if weighted_weights else None
    sw_ci_strain = round(1 - sw_update, 4) if sw_update is not None else None

    return {
        "avg_update_score":        avg_update,
        "ci_strain":               ci_strain,
        "severity_weighted_ci":    sw_ci_strain,
        "n_stale":                 n_stale,
        "n_cases":                 len(cases),
        "details":                 details,
    }


def run_ci_benchmark(
    model: str,
    provider: str,
    domains: list[str],
    judge_provider: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY", "")
        if provider == "anthropic"
        else os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        print(f"\n  set {provider.upper()}_API_KEY\n")
        sys.exit(1)

    judge = None
    judge_provider_used = None
    judge_model_used = None

    try:
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
    except Exception as e:
        if verbose:
            print(f"  WARNING: judge not available ({e}).")

    results_by_domain = {}
    all_ci_strain = []
    total_stale = 0
    total_cases = 0
    start = time.time()

    for d in domains:
        try:
            res = run_ci_domain(d, provider, model, api_key, judge, verbose)
            results_by_domain[d] = res
            if res["ci_strain"] is not None:
                all_ci_strain.append(res["ci_strain"])
            total_stale += res.get("n_stale", 0)
            total_cases += res.get("n_cases", 0)
        except Exception as e:
            print(f"  domain {d} failed: {e}")
            results_by_domain[d] = {"error": str(e)}

    elapsed = round(time.time() - start, 1)
    avg_ci_strain = round(sum(all_ci_strain) / len(all_ci_strain), 4) if all_ci_strain else None
    stale_rate = round(total_stale / total_cases, 4) if total_cases else None
    independent_judging = judge_provider_used is not None and judge_provider_used != provider

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  model:            {model}")
        print(f"  benchmark:        CAI-Bench CI-Strain (cache-invalidation)")
        print(f"  judge:            {judge_provider_used}/{judge_model_used}" + (" [independent]" if independent_judging else ""))
        print(f"  overall CI-Strain: {avg_ci_strain:.4f}" if avg_ci_strain is not None else "  overall CI-Strain: n/a")
        print(f"  stale rate:       {stale_rate*100:.0f}% of cases ignored the new fact entirely" if stale_rate is not None else "")
        print()
        for d, res in results_by_domain.items():
            if "error" in res:
                print(f"  {d:<20} ERROR")
            else:
                ci = res.get("ci_strain")
                stale = res.get("n_stale", 0)
                total = res.get("n_cases", 0)
                print(f"  {d:<20} CI-Strain {ci:.3f}  {stale}/{total} stale")
        print(f"{'=' * 60}\n")

    result = {
        "model":               model,
        "provider":            provider,
        "date":                str(date.today()),
        "benchmark_version":   "v2-ci",
        "test_type":           "cache_invalidation",
        "domains_tested":      domains,
        "judge_provider":      judge_provider_used,
        "judge_model":         judge_model_used,
        "independent_judging": independent_judging,
        "avg_ci_strain":       avg_ci_strain,
        "stale_rate":          stale_rate,
        "elapsed_seconds":     elapsed,
        "results":             results_by_domain,
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    safe_model = model.replace("/", "-").replace(":", "-")
    filename = f"ci_{safe_model}_{date.today()}.json"
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"  result saved: {path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run CAI-Bench CI-Strain (cache-invalidation) against an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python evaluate_ci.py --provider anthropic --model claude-sonnet-4-6
  python evaluate_ci.py --provider openai --model gpt-4o --domain medication
        """,
    )
    parser.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", choices=CI_DOMAINS, default=None,
                        help="Run only one domain (default: all)")
    parser.add_argument("--judge-provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else CI_DOMAINS

    print(f"\n  running CAI-Bench CI-Strain: {args.model}")
    print(f"  domains: {', '.join(domains)}")
    print(f"  {len(domains) * 3} scenarios x 2 turns -- does the model update when it legitimately should?\n")

    run_ci_benchmark(
        model=args.model,
        provider=args.provider,
        domains=domains,
        judge_provider=args.judge_provider,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
