"""
Commitment Invariant Generation
================================
Enriches frozen benchmark cases with explicit commitment_invariant fields.

Without commitment_invariants, evaluate_consistency() performs inter-response
comparison — a proxy that conflates the commitment axis (did the decision hold?)
with the explanation axis (did the phrasing change?). This produces a single
similarity score that cannot distinguish appropriate adaptation from contradiction.

With commitment_invariants, evaluate_commitment_invariance() evaluates each
response independently against a specified predicate. The commitment axis and
explanation axis are separate. Commitment Strain is a well-defined quantity:
"fraction of variants that violated the specified commitment."

This script generates invariants for all v2 benchmark cases that don't have
one yet, writes enriched JSON files, and produces a summary report.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_invariants.py

    # Generate for specific domains only
    python generate_invariants.py --domains mental_health medication

    # Dry run (show what would be generated, don't write)
    python generate_invariants.py --dry-run

    # Validate generated invariants against sample responses
    python generate_invariants.py --validate

Output:
    Enriches files in-place at benchmarks/v2/*.json (adds commitment_invariant,
    critical_commitment, common_violation_pattern fields to each case).
    Writes generate_invariants_report.json with per-case output for review.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "v2"
REPORT_PATH   = Path(__file__).resolve().parents[1] / "bench" / "generate_invariants_report.json"


def load_cases(domain: str) -> tuple[dict, Path]:
    p = BENCHMARK_DIR / f"{domain}.json"
    data = json.loads(p.read_text())
    return data, p


def make_judge(provider: str = "auto") -> "Judge":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from contradish.judge import Judge
    from contradish.llm import LLMClient

    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            print("  set ANTHROPIC_API_KEY or OPENAI_API_KEY")
            sys.exit(1)

    llm = LLMClient.make_judge_client(
        model_provider=provider,
        judge_provider=None,
        judge_api_key=None,
    )
    return Judge(llm)


def generate_for_domain(
    domain: str,
    judge,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> list[dict]:
    data, path = load_cases(domain)
    cases = data["cases"]
    results = []
    changed = 0

    for i, case in enumerate(cases):
        cid  = case["id"]
        name = case["name"]

        # Skip if already has a commitment_invariant (unless --force)
        if not force and "commitment_invariant" in case and case["commitment_invariant"]:
            if verbose:
                print(f"  [{cid}] skip (already has invariant)")
            results.append({"id": cid, "status": "skipped", "invariant": case["commitment_invariant"]})
            continue

        print(f"  [{cid}] generating: {name}")

        adv_samples = case.get("adversarial", [])[:4]
        result = judge.generate_commitment_invariant(
            domain=domain,
            name=name,
            original=case["original"],
            adversarial_samples=adv_samples,
        )

        inv = result.get("commitment_invariant", "")
        crit = result.get("critical_commitment", "")
        viol = result.get("common_violation_pattern", "")

        if verbose:
            print(f"         invariant: {inv[:120]}...")
            print(f"         critical:  {crit[:100]}")

        if not dry_run:
            case["commitment_invariant"]      = inv
            case["critical_commitment"]       = crit
            case["common_violation_pattern"]  = viol

        results.append({
            "id":                     cid,
            "status":                 "generated",
            "commitment_invariant":   inv,
            "critical_commitment":    crit,
            "common_violation_pattern": viol,
        })
        changed += 1
        time.sleep(0.3)  # Rate limiting

    if not dry_run and changed > 0:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  wrote {path.name} ({changed} cases enriched)")

    return results


def validate_invariants(domain: str, judge, n_samples: int = 3) -> list[dict]:
    """
    Spot-check generated invariants by evaluating real adversarial responses
    against them. This confirms the invariant is specific enough to catch violations.
    """
    from bench.evaluate import make_anthropic_app, make_openai_app
    import os

    # Build a simple test app
    if os.environ.get("ANTHROPIC_API_KEY"):
        app = make_anthropic_app("claude-haiku-4-5-20251001", os.environ["ANTHROPIC_API_KEY"])
    elif os.environ.get("OPENAI_API_KEY"):
        app = make_openai_app("gpt-4o-mini", os.environ["OPENAI_API_KEY"])
    else:
        print("  set API key for validation")
        return []

    data, _ = load_cases(domain)
    cases = data["cases"][:n_samples]
    validation_results = []

    for case in cases:
        inv = case.get("commitment_invariant")
        if not inv:
            continue

        cid  = case["id"]
        name = case["name"]
        print(f"\n  validating [{cid}] {name}")

        # Evaluate original response
        orig_response = app(case["original"])
        orig_result = judge.evaluate_constraint_satisfaction(
            commitment_invariant=inv,
            question=case["original"],
            technique="original",
            response=orig_response,
            domain=domain,
        )

        # Evaluate first adversarial response
        adv_response = app(case["adversarial"][0]) if case.get("adversarial") else ""
        adv_result = judge.evaluate_constraint_satisfaction(
            commitment_invariant=inv,
            question=case["adversarial"][0] if case.get("adversarial") else "",
            technique="adversarial",
            response=adv_response,
            domain=domain,
        ) if adv_response else None

        print(f"    original:    commitment_axis={orig_result['commitment_axis']}  "
              f"explanation_axis={orig_result['explanation_axis']}")
        if adv_result:
            print(f"    adversarial: commitment_axis={adv_result['commitment_axis']}  "
                  f"explanation_axis={adv_result['explanation_axis']}")
            if adv_result["commitment_axis"] in ("weakens", "violates"):
                print(f"    VIOLATION: {adv_result.get('violation_type', 'unspecified')}")

        validation_results.append({
            "id": cid, "name": name,
            "original_result": orig_result,
            "adversarial_result": adv_result,
        })

    return validation_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate commitment invariants for v2 benchmark")
    parser.add_argument("--domains",  nargs="+", help="Specific domains to process (default: all)")
    parser.add_argument("--provider", default="auto", choices=["auto", "anthropic", "openai"])
    parser.add_argument("--dry-run",  action="store_true", help="Generate but don't write to files")
    parser.add_argument("--force",    action="store_true", help="Regenerate even if invariant exists")
    parser.add_argument("--validate", action="store_true", help="Spot-check invariants after generation")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    all_domains = [p.stem for p in sorted(BENCHMARK_DIR.glob("*.json"))]
    domains = args.domains if args.domains else all_domains

    print(f"\n  Commitment Invariant Generation")
    print(f"  Benchmark: {BENCHMARK_DIR}")
    print(f"  Domains:   {len(domains)}")
    print(f"  Mode:      {'dry-run' if args.dry_run else 'write'}")

    judge = make_judge(args.provider)
    all_results = {}

    for domain in domains:
        print(f"\n  domain: {domain}")
        results = generate_for_domain(
            domain=domain,
            judge=judge,
            dry_run=args.dry_run,
            force=args.force,
            verbose=args.verbose,
        )
        all_results[domain] = results

        if args.validate:
            validate_invariants(domain, judge, n_samples=3)

    # Summary
    total      = sum(len(v) for v in all_results.values())
    generated  = sum(1 for v in all_results.values() for r in v if r["status"] == "generated")
    skipped    = sum(1 for v in all_results.values() for r in v if r["status"] == "skipped")

    print(f"\n  Total cases processed: {total}")
    print(f"  Generated: {generated}")
    print(f"  Skipped (already had invariant): {skipped}")

    if not args.dry_run:
        REPORT_PATH.write_text(json.dumps(all_results, indent=2))
        print(f"\n  Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
