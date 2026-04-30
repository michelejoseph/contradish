"""
contradish repair — diagnose drift cases and generate a full repair package.

Takes any contradish result JSON (SRA or main benchmark), finds all drifted/
failed cases, runs LLM-powered diagnosis on each, and produces:

  Per case:
    failure_mode            which category of adversarial failure occurred
    why_it_worked           one sentence on the exact trigger
    counterfactual_response what the model should have said instead
    system_prompt_fix       language to add to your system prompt
    training_example        fine-tuning pair ready to use

  Aggregated:
    failure_mode_distribution  which patterns dominate
    aggregate_fixes            deduplicated system prompt additions, ranked
    training_examples          JSONL ready for fine-tuning

Usage:
    python evaluate_repair.py \\
        --input results/sra_claude-sonnet-4-6.json \\
        --judge-provider openai --judge-model gpt-4o

    python evaluate_repair.py \\
        --input results/benchmark_claude-sonnet-4-6.json \\
        --max-cases 10 \\
        --output-dir repair/
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────
# ANSI colours (stripped if not a terminal)
# ─────────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if not _TTY:
        return text
    return f"\033[{code}m{text}\033[0m"

RED    = lambda t: _c("31", t)
YELLOW = lambda t: _c("33", t)
GREEN  = lambda t: _c("32", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)
RESET  = lambda t: t


# ─────────────────────────────────────────────────────────
# Severity colour helper
# ─────────────────────────────────────────────────────────

def _sev_color(severity: str) -> str:
    return {
        "critical": RED,
        "high":     YELLOW,
        "medium":   lambda t: _c("33", t),
        "low":      DIM,
    }.get(severity, RESET)


# ─────────────────────────────────────────────────────────
# Terminal output
# ─────────────────────────────────────────────────────────

def _wrap(text: str, indent: int = 13, width: int = 78) -> str:
    """Wrap long text with consistent indentation."""
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix,
                         subsequent_indent=prefix).lstrip()


def _print_case(diagnosis: dict, idx: int, total: int) -> None:
    domain = diagnosis.get("domain", "unknown")
    topic  = diagnosis.get("topic", "")
    mode   = diagnosis.get("failure_mode", "UNKNOWN")
    sev    = diagnosis.get("severity", "medium")
    tech   = diagnosis.get("technique", "")
    why    = diagnosis.get("why_it_worked", "")
    actual = diagnosis.get("actual_response", "") or ""
    counter= diagnosis.get("counterfactual_response", "") or ""
    fix    = diagnosis.get("system_prompt_fix", "") or ""
    has_ft = (
        isinstance(diagnosis.get("training_example"), dict)
        and diagnosis["training_example"].get("user")
        and diagnosis["training_example"].get("assistant")
    )

    color = _sev_color(sev)
    label = f"FAIL  {domain}/{topic}"
    mode_tag  = f"[{mode}]"
    sev_tag   = f"severity: {sev}"

    print(f"\n  {RED(label)}  {CYAN(mode_tag)}  {DIM(sev_tag)}")
    if tech:
        print(f"  {'technique:':11s} {DIM(tech)}")
    if why:
        print(f"  {'why:':11s} {_wrap(why)}")

    if actual:
        # Truncate long response for display
        snippet = actual.replace("\n", " ").strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        print(f"  {'model said:':11s} {DIM(repr(snippet))}")

    if counter:
        lines = textwrap.wrap(counter, width=65)
        print(f"  {'should say:':11s} {lines[0]}")
        for l in lines[1:]:
            print(f"  {' ' * 11} {l}")

    if fix:
        fix_short = fix.replace("\n", " ").strip()
        if len(fix_short) > 100:
            fix_short = fix_short[:97] + "..."
        print(f"  {BOLD('SYSTEM PROMPT')}  {DIM(repr(fix_short))}")

    if has_ft:
        print(f"  {BOLD('TRAINING')}       1 example added  {DIM('repair_package.jsonl')}")

    print()


def print_summary(report: dict, jsonl_path: str, prompt_path: str) -> None:
    """Print full repair report to terminal."""
    drift_count = report.get("drift_count", 0)
    result_path = report.get("result_path", "?")
    source_type = report.get("source_type", "?")
    diagnoses   = report.get("diagnoses", [])
    aggregate   = report.get("aggregate") or {}

    model = Path(result_path).stem.replace("sra_", "").replace("benchmark_", "")

    print()
    print(f"  {BOLD('contradish diagnose')}")
    print(f"  {DIM('input:')}  {result_path}  {DIM(f'({source_type})')}")
    print(f"  {DIM('model:')}  {model}")
    print()

    if drift_count == 0:
        msg = report.get("message", "No drift cases found.")
        print(f"  {GREEN(msg)}\n")
        return

    print(f"  analyzing {BOLD(str(drift_count))} drift cases\n")

    # Print each case
    for i, diagnosis in enumerate(diagnoses, 1):
        _print_case(diagnosis, i, drift_count)

    # ── Failure mode distribution ──
    dist = aggregate.get("failure_mode_distribution", [])
    if dist:
        print(f"  {BOLD('FAILURE MODE DISTRIBUTION')}")
        domain_dist = {d["domain"]: d["count"] for d in aggregate.get("domain_distribution", [])}

        # Build per-mode domain breakdown
        mode_domains: dict[str, list[str]] = {}
        for d in diagnoses:
            mode = d.get("failure_mode", "UNKNOWN")
            dom  = d.get("domain", "?")
            mode_domains.setdefault(mode, []).append(dom)

        for entry in dist:
            mode  = entry["failure_mode"]
            count = entry["count"]
            doms  = mode_domains.get(mode, [])
            # Summarize domain breakdown as "domain x2, other x1"
            from collections import Counter
            dom_counter = Counter(doms)
            dom_str = ", ".join(
                f"{d} x{n}" if n > 1 else d
                for d, n in dom_counter.most_common()
            )
            bar = "#" * count
            print(f"  {CYAN(mode):<35s} {bar:<8s} {count}  {DIM(dom_str)}")
        print()

    # ── Priority cases ──
    priority = aggregate.get("priority_cases", [])
    if priority:
        print(f"  {BOLD('PRIORITY FIXES')}  {DIM('critical + high severity')}")
        for p in priority:
            color = _sev_color(p.get("severity", "high"))
            print(f"  {color('[' + p['severity'] + ']'):20s}  {p['domain']}/{p['topic']}")
            if p.get("why"):
                print(f"  {' ' * 20}  {DIM(p['why'][:80])}")
        print()

    # ── Output files ──
    ft_count    = aggregate.get("total_ft_examples", 0)
    fixes_count = len(aggregate.get("aggregate_fixes", []))

    print(f"  {GREEN(str(ft_count))} fine-tuning examples  {DIM(jsonl_path)}")
    print(f"  {GREEN(str(fixes_count))} system prompt fixes  {DIM(prompt_path)}")
    print()


# ─────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────

def save_report(report: dict, output_dir: str) -> tuple[str, str, str]:
    """
    Save all repair outputs to output_dir.
    Returns (report_json_path, jsonl_path, prompt_path).
    """
    from contradish.diagnose import export_training_jsonl, export_system_prompt_block

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result_stem = Path(report["result_path"]).stem
    prefix = f"{output_dir}/repair_{result_stem}"

    # Full report JSON
    report_path = f"{prefix}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Fine-tuning JSONL
    jsonl_path = f"{prefix}_ft.jsonl"
    export_training_jsonl(report, jsonl_path)

    # System prompt block
    prompt_path = f"{prefix}_system_prompt.txt"
    export_system_prompt_block(report, prompt_path)

    return report_path, jsonl_path, prompt_path


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_repair",
        description=(
            "Diagnose drift cases from any contradish result JSON and generate a repair package.\n"
            "Produces: per-case counterfactuals, failure mode distribution, system prompt fixes, JSONL fine-tuning set."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python evaluate_repair.py --input results/sra_claude-sonnet-4-6.json
  python evaluate_repair.py --input results/benchmark_claude-sonnet-4-6.json --max-cases 10
  python evaluate_repair.py --input results/sra_gpt-4o.json --judge-provider anthropic --judge-model claude-opus-4-6
  python evaluate_repair.py --input results/sra_claude-sonnet-4-6.json --output-dir repair/ --quiet
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="FILE",
        help="Path to a contradish result JSON (SRA or benchmark output)",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["anthropic", "openai"],
        default=None,
        metavar="PROVIDER",
        help="Provider for the judge model (default: opposite of result model's provider)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        metavar="MODEL",
        help="Judge model name (default: claude-opus-4-6 for anthropic, gpt-4o for openai)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Limit diagnosis to first N drift cases (useful for testing)",
    )
    parser.add_argument(
        "--output-dir",
        default="repair",
        metavar="DIR",
        help="Directory for output files (default: repair/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal output (still writes output files)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full report JSON to stdout instead of formatted output",
    )

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("\n  contradish diagnose needs an API key.\n")
        print("  Set one of:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    export OPENAI_API_KEY=sk-...\n")
        sys.exit(1)

    if not Path(args.input).exists():
        print(f"\n  File not found: {args.input}\n")
        sys.exit(1)

    # Default judge provider: opposite of result model's provider
    judge_provider = args.judge_provider
    if not judge_provider:
        # Try to infer from result file
        try:
            with open(args.input) as f:
                result = json.load(f)
            model_provider = result.get("provider", "")
        except Exception:
            model_provider = ""
        judge_provider = "openai" if model_provider == "anthropic" else "anthropic"

    # Default judge model
    judge_model = args.judge_model
    if not judge_model:
        judge_model = "claude-opus-4-6" if judge_provider == "anthropic" else "gpt-4o"

    if not args.quiet and not args.json:
        print(f"\n  {BOLD('contradish diagnose')}")
        print(f"  judge: {judge_provider}/{judge_model}")
        print(f"  loading drift cases...\n")

    from contradish.diagnose import analyze_result

    try:
        report = analyze_result(
            result_path=args.input,
            judge_provider=judge_provider,
            judge_model=judge_model,
            max_cases=args.max_cases,
        )
    except FileNotFoundError as e:
        print(f"\n  {RED('Error:')} {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n  {RED('Error during diagnosis:')} {e}\n")
        raise

    if args.json:
        print(json.dumps(report, indent=2))
        sys.exit(0)

    # Save outputs
    report_json, jsonl_path, prompt_path = save_report(report, args.output_dir)

    if not args.quiet:
        print_summary(report, jsonl_path, prompt_path)
        print(f"  {DIM('full report:')} {report_json}")
        print()

    # Exit 1 if any critical/high priority cases
    priority = (report.get("aggregate") or {}).get("priority_cases", [])
    critical = [p for p in priority if p.get("severity") == "critical"]
    if critical:
        sys.exit(1)


if __name__ == "__main__":
    main()
