"""
Validates result JSON files submitted to the CAI-Bench leaderboard.
Run automatically on PRs that add files to results/.

Checks:
  - Required fields are present
  - mode is "frozen" (not live)
  - benchmark_version is a known version
  - independent_judging is true
  - avg_cai_score and avg_cai_strain are in range [0, 1]
  - avg_cai_score + avg_cai_strain == 1.0 (within float tolerance)
  - policies_tested matches the expected list for the benchmark version
  - All policy results have required fields and no errors
"""

import json
import os
import sys
import glob

REQUIRED_FIELDS = [
    "model", "provider", "date", "benchmark_version", "mode",
    "judge_provider", "judge_model", "independent_judging",
    "policies_tested", "avg_cai_score", "avg_cai_strain",
    "elapsed_seconds", "results",
]

KNOWN_VERSIONS = {"v1"}

V1_POLICIES = [
    "ecommerce", "hr", "healthcare", "legal",
    "finance", "saas", "insurance", "education",
]

KNOWN_PROVIDERS = {"anthropic", "openai"}

ERRORS = []


def err(msg: str):
    ERRORS.append(msg)
    print(f"  ERROR: {msg}")


def validate_file(path: str):
    print(f"\nValidating: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        err(f"invalid JSON: {e}")
        return

    # Required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            err(f"missing required field: {field!r}")

    if ERRORS:
        return

    # Mode must be frozen
    if data["mode"] != "frozen":
        err(f"mode must be 'frozen' for leaderboard submission (got {data['mode']!r}). Use --live only for development.")

    # Benchmark version
    if data["benchmark_version"] not in KNOWN_VERSIONS:
        err(f"unknown benchmark_version {data['benchmark_version']!r}. Known: {KNOWN_VERSIONS}")

    # Independent judging required
    if not data.get("independent_judging"):
        err(
            "independent_judging must be true. "
            "The judge must be a different provider than the model under test. "
            "Set both ANTHROPIC_API_KEY and OPENAI_API_KEY before running evaluate.py."
        )

    # Judge and model must be different providers
    if data.get("judge_provider") == data.get("provider"):
        err(
            f"judge_provider ({data.get('judge_provider')!r}) matches model provider ({data.get('provider')!r}). "
            "Cross-provider judging is required."
        )

    # Providers must be known
    if data["provider"] not in KNOWN_PROVIDERS:
        err(f"unknown provider {data['provider']!r}. Known: {KNOWN_PROVIDERS}")
    if data.get("judge_provider") not in KNOWN_PROVIDERS:
        err(f"unknown judge_provider {data.get('judge_provider')!r}. Known: {KNOWN_PROVIDERS}")

    # Score range
    score = data.get("avg_cai_score")
    strain = data.get("avg_cai_strain")
    if score is not None and not (0.0 <= score <= 1.0):
        err(f"avg_cai_score out of range: {score}")
    if strain is not None and not (0.0 <= strain <= 1.0):
        err(f"avg_cai_strain out of range: {strain}")
    if score is not None and strain is not None:
        if abs((score + strain) - 1.0) > 0.01:
            err(f"avg_cai_score + avg_cai_strain should equal 1.0 (got {score} + {strain} = {score + strain:.4f})")

    # Policies tested
    version = data["benchmark_version"]
    expected_policies = V1_POLICIES if version == "v1" else []
    if sorted(data["policies_tested"]) != sorted(expected_policies):
        err(f"policies_tested mismatch. Expected {expected_policies}, got {data['policies_tested']}")

    # Per-policy results
    results = data.get("results", {})
    for policy in data.get("policies_tested", []):
        if policy not in results:
            err(f"missing result for policy: {policy!r}")
            continue
        res = results[policy]
        if "error" in res:
            err(f"policy {policy!r} has error: {res['error'][:80]}")
        else:
            for field in ["cai_score", "cai_strain", "passed", "failed", "total"]:
                if field not in res:
                    err(f"policy {policy!r} missing field: {field!r}")

    if not ERRORS:
        print(f"  OK -- model={data['model']}, strain={data.get('avg_cai_strain')}, judge={data.get('judge_provider')}/{data.get('judge_model')}")


def main():
    result_files = glob.glob("results/*.json")
    if not result_files:
        print("No result files found in results/")
        sys.exit(0)

    for path in sorted(result_files):
        validate_file(path)

    if ERRORS:
        print(f"\n{len(ERRORS)} validation error(s). Fix before merging.\n")
        sys.exit(1)
    else:
        print(f"\nAll {len(result_files)} result file(s) valid.\n")


if __name__ == "__main__":
    main()
