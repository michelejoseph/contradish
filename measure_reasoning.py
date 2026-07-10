#!/usr/bin/env python3
"""
measure_reasoning.py — Produce a complete Reasoning Profile for a model.

Combines CAI Strain (from contradish benchmark) and Reality Strain (from
reality_strain.py) into a full ReasoningProfile with law validation,
uncertainty estimation, and natural-language interpretation.

Usage
-----
    # From existing result files
    python measure_reasoning.py \\
        --model gpt-4o \\
        --cai-results results/cai_gpt-4o.json \\
        --reality-results results/reality_strain_gpt-4o.json

    # Run full measurement from scratch (both CAI and Reality Strain)
    python measure_reasoning.py --model gpt-4o --run-all

    # Compare two models
    python measure_reasoning.py \\
        --model gpt-4o \\
        --cai-results results/cai_gpt-4o.json \\
        --reality-results results/reality_gpt-4o.json \\
        --compare-model claude-sonnet-4-6 \\
        --compare-cai results/cai_claude.json \\
        --compare-reality results/reality_claude.json

    # Write profile as JSON
    python measure_reasoning.py --model gpt-4o ... --output-json results/profile_gpt-4o.json

Environment
-----------
    OPENAI_API_KEY    — required if --run-all with OpenAI models
    ANTHROPIC_API_KEY — required if --run-all with Anthropic models
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from contradish.measurement import (
    DIMENSIONS, LAWS, ReasoningProfile, MeasurementUncertainty,
    profile_from_results, compare,
)


def load_json(path: Optional[str]) -> Optional[dict]:  # noqa: F821
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  Warning: {path} not found", file=sys.stderr)
        return None
    with open(p) as f:
        return json.load(f)


def run_cai_benchmark(model: str, output_path: str) -> Optional[dict]:
    """Run contradish benchmark and return results."""
    print(f"\n  Running CAI Strain benchmark for {model}...")
    cmd = [
        sys.executable, "-m", "contradish",
        "benchmark", "--model", model,
        "--output-json", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_HERE))
    if result.returncode != 0:
        print(f"  CAI benchmark failed: {result.stderr[:200]}", file=sys.stderr)
        return None
    return load_json(output_path)


def run_reality_strain(model: str, output_path: str) -> Optional[dict]:
    """Run reality_strain.py and return results."""
    print(f"\n  Running Reality Strain scorer for {model}...")
    cmd = [sys.executable, str(_HERE / "reality_strain.py"),
           "--model", model, "--output-json", output_path, "--quiet"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_HERE))
    if result.returncode != 0:
        print(f"  Reality Strain failed: {result.stderr[:200]}", file=sys.stderr)
        return None
    return load_json(output_path)


def print_dimension_reference():
    """Print the complete dimension taxonomy."""
    W = 72
    print("\n" + "═" * W)
    print("  contradish · Reasoning Dimensions — Reference")
    print("═" * W)
    for key, dim in DIMENSIONS.items():
        if dim.ideal is not None:
            ideal_str = f"  ideal={dim.ideal}"
        else:
            ideal_str = "  structural property"
        print(f"\n  {dim.symbol}  {dim.name}  [{dim.unit}]{ideal_str}")
        print(f"     {dim.measures}")
    print("\n" + "═" * W + "\n")


def print_law_reference():
    """Print all measurement laws."""
    W = 72
    print("\n" + "═" * W)
    print("  contradish · Measurement Laws")
    print("═" * W)
    for law in LAWS:
        tag = "  [FALSIFIABLE PREDICTION]" if law.falsifiable else ""
        print(f"\n  {law.name}{tag}")
        print(f"    {law.formula}")
        # wrap description
        words = law.description.split()
        line, prefix = "    ", "    "
        for w in words:
            if len(line) + len(w) + 1 > 70:
                print(line)
                line = prefix + w
            else:
                line += (" " if line != prefix else "") + w
        if line.strip():
            print(line)
    print("\n" + "═" * W + "\n")


def build_profile(
    model:            str,
    cai_results:      Optional[dict],
    reality_results:  Optional[dict],
    judge_floor:      Optional[float] = None,
    alpha:            float = 0.5,
) -> ReasoningProfile:
    """Determine provider and build a ReasoningProfile."""
    model_provider = "anthropic" if "claude" in model.lower() else "openai"

    # Extract judge info from results
    judge_provider = None
    judge_model    = None
    if reality_results:
        judge_provider = reality_results.get("judge_provider")
        judge_model    = reality_results.get("judge_model")
    elif cai_results:
        judge_provider = cai_results.get("judge_provider")

    return profile_from_results(
        model=model,
        model_provider=model_provider,
        judge_provider=judge_provider,
        judge_model=judge_model,
        cai_results=cai_results,
        reality_results=reality_results,
        alpha=alpha,
        judge_floor=judge_floor,
    )


def main():
    from typing import Optional  # local import for clarity

    parser = argparse.ArgumentParser(
        description="Produce a complete Reasoning Profile for a model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True,
        help="Model to profile (e.g. gpt-4o, claude-sonnet-4-6)")
    parser.add_argument("--cai-results",
        help="JSON from contradish benchmark (judgment_strain key)")
    parser.add_argument("--reality-results",
        help="JSON from reality_strain.py")
    parser.add_argument("--run-all", action="store_true",
        help="Run both CAI and Reality Strain benchmarks from scratch")
    parser.add_argument("--judge-floor", type=float,
        help="Judge's own CAI Strain from judge_calibration (used in uncertainty model)")
    parser.add_argument("--alpha", type=float, default=0.5,
        help="Weight of CAI Strain in Admissibility Distance. Default 0.5")
    parser.add_argument("--compare-model",
        help="Second model to compare against")
    parser.add_argument("--compare-cai",
        help="CAI results for comparison model")
    parser.add_argument("--compare-reality",
        help="Reality Strain results for comparison model")
    parser.add_argument("--output-json",
        help="Write profile(s) to this JSON file")
    parser.add_argument("--dimensions", action="store_true",
        help="Print dimension taxonomy and exit")
    parser.add_argument("--laws", action="store_true",
        help="Print measurement laws and exit")

    # Make --model optional when only printing reference info
    if '--dimensions' in sys.argv or '--laws' in sys.argv:
        parser.set_defaults(model="__reference__")
        if '--model' not in sys.argv:
            sys.argv.insert(1, '--model')
            sys.argv.insert(2, '__reference__')

    args = parser.parse_args()

    if args.dimensions:
        print_dimension_reference()
        return
    if args.laws:
        print_law_reference()
        return

    # Load or run measurements
    if args.run_all:
        cai_path     = f"results/cai_{args.model}.json"
        reality_path = f"results/reality_{args.model}.json"
        cai_data     = run_cai_benchmark(args.model, cai_path)
        reality_data = run_reality_strain(args.model, reality_path)
    else:
        cai_data     = load_json(args.cai_results)
        reality_data = load_json(args.reality_results)

    if not cai_data and not reality_data:
        print(
            "No measurement data available. "
            "Pass --cai-results and/or --reality-results, or use --run-all.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build primary profile
    profile = build_profile(
        model=args.model,
        cai_results=cai_data,
        reality_results=reality_data,
        judge_floor=args.judge_floor,
        alpha=args.alpha,
    )

    print(profile)

    # Build comparison profile if requested
    comparison = None
    if args.compare_model:
        compare_cai     = load_json(args.compare_cai)
        compare_reality = load_json(args.compare_reality)
        compare_profile = build_profile(
            model=args.compare_model,
            cai_results=compare_cai,
            reality_results=compare_reality,
            judge_floor=args.judge_floor,
            alpha=args.alpha,
        )
        print(compare_profile)

        comparison = compare(profile, compare_profile)
        W = 66
        print("═" * W)
        print("  COMPARISON")
        print("─" * W)
        print(f"  Min detectable difference (MDD): {comparison['mdd']:.4f}")
        print()
        for r in comparison["dimensions"]:
            if r.get("status") == "insufficient data":
                print(f"  {r['symbol']:<4}  {r['dimension']:<24}  insufficient data")
                continue
            ma = args.model
            mb = args.compare_model
            va = r.get(ma, "—")
            vb = r.get(mb, "—")
            det = "✓" if r["detectable"] else "~"
            win = r["better"]
            print(f"  {r['symbol']:<4}  {r['dimension']:<24}  "
                  f"{ma[:10]}={va:.3f}  {mb[:10]}={vb:.3f}  "
                  f"{det}  {win}")
        print()
        print(f"  Conclusion: {comparison['overall']}")
        print("═" * W)

    # Write JSON output
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"primary": profile.to_dict()}
        if comparison:
            payload["comparison"] = comparison
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Profile written → {out_path}\n")


if __name__ == "__main__":
    main()
