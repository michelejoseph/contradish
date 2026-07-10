#!/usr/bin/env python3
"""
compare_models.py — Multi-model Reality Strain comparison for the contradish
admissibility experiment.

Runs Reality Strain scoring across multiple models and produces:
  1. A JSON results file with per-model, per-domain scores
  2. A 2D scatter table: CAI Strain × Reality Strain
  3. A convergence order prediction (load-bearing weight ranking)

This is Phase 3 of the admissibility experiment:
  Phase 1 — Ground truth construction (done)
  Phase 2 — Reality Strain scoring per model (reality_strain.py)
  Phase 3 — Cross-model comparison and structural prediction test (this file)

Usage
-----
    # Compare GPT-4o and Claude Sonnet across all domains
    python compare_models.py --models gpt-4o claude-sonnet-4-6

    # Include CAI Strain from existing benchmark results
    python compare_models.py --models gpt-4o claude-sonnet-4-6 \\
        --cai-results results/cai_gpt-4o.json results/cai_claude.json

    # From cached outputs (no model API calls)
    python compare_models.py --models gpt-4o claude-sonnet-4-6 \\
        --from-caches cache/gpt-4o.json cache/claude.json

Environment
-----------
    OPENAI_API_KEY     — for OpenAI models
    ANTHROPIC_API_KEY  — for Anthropic models
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from reality_strain import (
    DOMAIN_FILE_MAP, DEFAULT_DOMAINS, DEFAULT_ALPHA,
    score_domain, admissibility_distance,
)
from contradish.llm import LLMClient
from contradish.judge import Judge


# ── formatting helpers ─────────────────────────────────────────────────────────

def _bar(score: float, width: int = 12) -> str:
    n = int(score * width)
    return "█" * n + "░" * (width - n)


def print_comparison_table(model_results: list[dict], alpha: float):
    """Print a 2D comparison: models as rows, domains as columns."""
    domains = DEFAULT_DOMAINS
    W = 80

    print("\n" + "=" * W)
    print("  contradish · Reality Strain — Multi-Model Comparison")
    print("=" * W)

    # Header
    hdr = f"  {'model':<22}"
    for d in domains:
        hdr += f"  {d[:10]:<10}"
    hdr += f"  {'overall':<8}"
    print(hdr)
    print("-" * W)

    for mr in sorted(model_results, key=lambda m: m["overall_reality_strain"]):
        name   = mr["model"]
        by_dom = {r["domain"]: r["domain_reality_strain"] for r in mr["domain_results"]}
        overall = mr["overall_reality_strain"]
        row = f"  {name:<22}"
        for d in domains:
            rs = by_dom.get(d)
            if rs is not None:
                row += f"  {rs:.3f}     "
            else:
                row += f"  {'n/a':<10}"
        row += f"  {overall:.3f}"
        print(row)

    print("-" * W)

    # Best per domain
    best_row = f"  {'best':<22}"
    for d in domains:
        vals = [(mr["model"], r["domain_reality_strain"])
                for mr in model_results
                for r in mr["domain_results"]
                if r["domain"] == d]
        if vals:
            best_model, best_rs = min(vals, key=lambda x: x[1])
            best_row += f"  {best_rs:.3f}({best_model[:3]:<3})"
        else:
            best_row += f"  {'n/a':<10}"
    print(best_row)
    print("=" * W)


def print_scatter_table(model_results: list[dict]):
    """Print CAI Strain × Reality Strain 2D scatter table."""
    models_with_cai = [mr for mr in model_results if mr.get("cai_strain") is not None]
    if not models_with_cai:
        return

    W = 62
    print("\n" + "=" * W)
    print("  CAI Strain × Reality Strain  (lower = closer to fixed point)")
    print("=" * W)
    print(f"  {'model':<22}  {'cai_strain':>10}  {'reality_strain':>14}  {'adist':>6}")
    print("-" * W)

    sorted_m = sorted(models_with_cai, key=lambda m: m.get("admissibility_distance", 1.0))
    for mr in sorted_m:
        cai  = mr["cai_strain"]
        rs   = mr["overall_reality_strain"]
        adst = mr.get("admissibility_distance")
        print(f"  {mr['model']:<22}  {cai:>10.3f}  {rs:>14.3f}  "
              f"  {adst:.3f}" if adst else "")
    print("=" * W)

    # Correlation direction check
    cai_vals  = [m["cai_strain"] for m in sorted_m]
    rs_vals   = [m["overall_reality_strain"] for m in sorted_m]
    n         = len(cai_vals)
    if n >= 3:
        mean_c = sum(cai_vals) / n
        mean_r = sum(rs_vals)  / n
        num    = sum((c - mean_c) * (r - mean_r) for c, r in zip(cai_vals, rs_vals))
        den_c  = sum((c - mean_c) ** 2 for c in cai_vals) ** 0.5
        den_r  = sum((r - mean_r) ** 2 for r in rs_vals)  ** 0.5
        if den_c * den_r > 0:
            corr = num / (den_c * den_r)
            direction = "positive" if corr > 0.1 else ("negative" if corr < -0.1 else "flat")
            print(f"\n  CAI×Reality correlation: {corr:+.3f}  ({direction})")
            if direction == "positive":
                print("  ✓ Matches structural prediction: consistency and correctness co-vary.")
            elif direction == "negative":
                print("  ✗ Negative correlation: models that are more consistent are LESS correct.")
            else:
                print("  ~ Flat: no clear relationship detected.")


def print_convergence_prediction(model_results: list[dict]):
    """Print load-bearing-weighted convergence order prediction."""
    print("\n  Convergence order prediction")
    print("  (Cases with low load_bearing_weight converge first when repair loop is applied.)")
    print()

    # Collect all cases with mean Reality Strain across models
    case_rs: dict[str, list[float]] = {}
    case_meta: dict[str, dict] = {}
    for mr in model_results:
        for dr in mr["domain_results"]:
            for c in dr["cases"]:
                cid = c["id"]
                case_rs.setdefault(cid, []).append(c["reality_strain"])
                case_meta[cid] = {
                    "load_bearing_weight": c["load_bearing_weight"],
                    "domain": dr["domain"],
                    "adversarial_pressure_type": c.get("adversarial_pressure_type", ""),
                }

    all_cases = []
    for cid, rs_list in case_rs.items():
        mean_rs = sum(rs_list) / len(rs_list)
        all_cases.append({
            "id": cid,
            "mean_reality_strain": round(mean_rs, 3),
            **case_meta[cid],
        })

    # Sort by load_bearing_weight ascending (converges first)
    sorted_cases = sorted(all_cases, key=lambda c: c["load_bearing_weight"])

    print(f"  {'case_id':<14}  {'lbw':>5}  {'mean_rs':>8}  {'domain':<22}  pressure_type")
    print("  " + "-" * 70)
    for c in sorted_cases:
        converge_note = " ← converges first" if c == sorted_cases[0] else (
                        " ← converges last"  if c == sorted_cases[-1] else "")
        print(f"  {c['id']:<14}  {c['load_bearing_weight']:>5.2f}  "
              f"{c['mean_reality_strain']:>8.3f}  {c['domain']:<22}  "
              f"{c['adversarial_pressure_type']}{converge_note}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model Reality Strain comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models", nargs="+", required=True,
        help="Models to compare (e.g. --models gpt-4o claude-sonnet-4-6)")
    parser.add_argument("--domains", nargs="+",
        choices=list(DOMAIN_FILE_MAP.keys()), default=DEFAULT_DOMAINS,
        help="Domains to score. Default: all five.")
    parser.add_argument("--from-caches", nargs="+",
        help="Cached output JSON files, one per model (parallel to --models).")
    parser.add_argument("--cai-results", nargs="+",
        help="CAI Strain result JSONs, one per model (parallel to --models).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--output-json", default="results/comparison.json")
    parser.add_argument("--judge-provider", choices=["openai", "anthropic"])
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    models       = args.models
    caches       = args.from_caches  or [None] * len(models)
    cai_files    = args.cai_results  or [None] * len(models)

    if len(caches) != len(models):
        print("Error: --from-caches must have one entry per model.", file=sys.stderr)
        sys.exit(1)
    if len(cai_files) != len(models):
        print("Error: --cai-results must have one entry per model.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    model_results = []

    for model_name, cache_path, cai_path in zip(models, caches, cai_files):
        model_provider = "anthropic" if "claude" in model_name.lower() else "openai"

        if verbose:
            print(f"\n{'='*50}")
            print(f"Model: {model_name}")
            print(f"{'='*50}")

        try:
            model_client = LLMClient(provider=model_provider)
            judge_client = LLMClient.make_judge_client(
                model_provider=model_provider,
                judge_provider=args.judge_provider,
            )
            judge = Judge(judge_client)
        except EnvironmentError as e:
            print(f"  Error: {e}", file=sys.stderr)
            continue

        # Load cache
        cached_outputs = None
        if cache_path:
            with open(cache_path) as f:
                cached_outputs = json.load(f)

        # Score domains
        domain_results = []
        for domain in args.domains:
            if verbose:
                print(f"\n▸ {domain}")
            try:
                result = score_domain(
                    domain=domain,
                    model_client=model_client,
                    model_name=model_name,
                    judge=judge,
                    cached_outputs=cached_outputs,
                    verbose=verbose,
                )
                domain_results.append(result)
                if verbose:
                    print(f"  → rs={result['domain_reality_strain']:.3f}")
            except FileNotFoundError as e:
                print(f"  Warning: {e}", file=sys.stderr)

        if not domain_results:
            continue

        # Aggregate
        all_rs = [r["domain_reality_strain"] for r in domain_results]
        overall_rs = round(sum(all_rs) / len(all_rs), 4)

        # CAI Strain
        cai_strain, adist = None, None
        if cai_path:
            try:
                with open(cai_path) as f:
                    cai_data = json.load(f)
                raw = cai_data.get("judgment_strain") or cai_data.get("cai_strain")
                if raw is not None:
                    cai_strain = float(raw)
                    adist = admissibility_distance(overall_rs, cai_strain, args.alpha)
            except Exception as e:
                print(f"  Warning: {e}", file=sys.stderr)

        model_results.append({
            "model":                  model_name,
            "model_provider":         model_provider,
            "overall_reality_strain": overall_rs,
            "cai_strain":             cai_strain,
            "admissibility_distance": adist,
            "domain_results":         domain_results,
        })

    if not model_results:
        print("No models scored.", file=sys.stderr)
        sys.exit(1)

    elapsed = round(time.time() - t0, 1)

    # Print tables
    if verbose:
        print_comparison_table(model_results, args.alpha)
        print_scatter_table(model_results)
        print_convergence_prediction(model_results)

    # Write JSON
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "models": [m["model"] for m in model_results],
            "alpha":  args.alpha,
            "elapsed_seconds": elapsed,
            "model_results": model_results,
        }, f, indent=2)

    if verbose:
        print(f"\n  Results → {out_path}  ({elapsed}s)\n")


if __name__ == "__main__":
    main()
