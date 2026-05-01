"""
contradish monitor — production drift detection CLI.

Detects consistency failures across semantically equivalent inputs in real
production conversation logs. Unlike the benchmark (which tests synthetic
adversarial cases), the monitor finds drift that is already happening.

Usage:
    python evaluate_monitor.py --input production_logs.jsonl
    python evaluate_monitor.py --input logs.jsonl --min-cluster-size 3 --max 500
    python evaluate_monitor.py --input logs.jsonl --threshold 0.25 --output results/monitor_report.json
    python evaluate_monitor.py --input logs.jsonl --judge-provider openai --quiet

Input format (JSONL):
    {"input": "user message", "output": "model response"}
    {"input": "...", "output": "...", "domain": "medication", "session_id": "abc123"}

Exits 1 if drift_rate exceeds --threshold (default 0.30).
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_TTY = sys.stdout.isatty()
def _c(code, t): return f"\033[{code}m{t}\033[0m" if _TTY else t
RED    = lambda t: _c("31", t)
YELLOW = lambda t: _c("33", t)
GREEN  = lambda t: _c("32", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_monitor",
        description="Detect consistency failures in real production conversation logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python evaluate_monitor.py --input logs.jsonl
  python evaluate_monitor.py --input logs.jsonl --max 500 --min-cluster-size 3
  python evaluate_monitor.py --input logs.jsonl --judge-provider openai --threshold 0.25
  python evaluate_monitor.py --input logs.jsonl --output results/monitor_report.json --quiet

log format (JSONL, one line per conversation):
  {"input": "what is the max dose of ibuprofen?", "output": "1,200mg per day OTC."}
  {"input": "how much ibuprofen can i take daily?", "output": "For adults, up to 2,400mg."}

optional fields per line: domain, session_id, timestamp, model
        """,
    )
    parser.add_argument("--input", "-i", required=True, metavar="FILE",
                        help="Path to conversation log (JSONL, JSON, or CSV)")
    parser.add_argument("--format", choices=["auto", "jsonl", "json", "csv"], default="auto",
                        help="Log format (default: auto-detect from extension)")
    parser.add_argument("--max", type=int, default=200, metavar="N",
                        help="Maximum conversations to load (default: 200)")
    parser.add_argument("--min-cluster-size", type=int, default=3, metavar="N",
                        help="Minimum conversations per cluster to score (default: 3)")
    parser.add_argument("--batch-size", type=int, default=25, metavar="N",
                        help="Inputs per clustering batch (default: 25, max ~30)")
    parser.add_argument("--threshold", type=float, default=0.30, metavar="F",
                        help="Drift rate threshold for exit 1 (default: 0.30)")
    parser.add_argument("--judge-provider", choices=["anthropic", "openai"], default=None,
                        help="LLM provider for the judge (default: opposite of model provider)")
    parser.add_argument("--judge-model", default=None, metavar="MODEL",
                        help="Judge model (default: claude-opus-4-6 or gpt-4o)")
    parser.add_argument("--output", "-o", default=None, metavar="FILE",
                        help="Save full analysis JSON to this path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--json", action="store_true",
                        help="Print analysis JSON to stdout instead of summary")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("\n  contradish monitor needs an API key.\n")
        sys.exit(1)

    from contradish.monitor import (
        load_log, find_clusters, score_clusters,
        analyze_monitor, print_monitor_summary,
    )
    from contradish.llm   import LLMClient
    from contradish.judge import Judge

    # Load log
    if not args.quiet and not args.json:
        print(f"\n  contradish monitor  —  loading {args.input}")

    try:
        conversations = load_log(args.input, max_conversations=args.max, format=args.format)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  {e}\n")
        sys.exit(1)

    if not args.quiet and not args.json:
        print(f"  loaded {len(conversations)} conversations")

    # Set up judge
    judge_provider = args.judge_provider or (
        "openai" if os.environ.get("ANTHROPIC_API_KEY") else "anthropic"
    )
    judge_model = args.judge_model or (
        "claude-opus-4-6" if judge_provider == "anthropic" else "gpt-4o"
    )

    if not args.quiet and not args.json:
        print(f"  judge:  {judge_provider}/{judge_model}")
        print()

    llm   = LLMClient(provider=judge_provider, model=judge_model)
    judge = Judge(llm)

    # Cluster
    if not args.quiet and not args.json:
        print(f"  finding semantic clusters  (min_size={args.min_cluster_size})...")

    clusters = find_clusters(
        conversations,
        judge,
        min_cluster_size=args.min_cluster_size,
        batch_size=args.batch_size,
        quiet=args.quiet or args.json,
    )

    if not clusters:
        if not args.json:
            print(f"\n  No clusters found with {args.min_cluster_size}+ conversations.\n"
                  f"  Try --min-cluster-size 2 or load more conversations with --max.\n")
        sys.exit(0)

    if not args.quiet and not args.json:
        print(f"  found {len(clusters)} topic clusters")
        print()
        print(f"  scoring consistency within each cluster...")

    # Score
    scored = score_clusters(
        clusters,
        judge,
        quiet=args.quiet or args.json,
    )

    # Analyze
    analysis = analyze_monitor(scored, total_conversations=len(conversations))
    analysis["log_path"]      = args.input
    analysis["judge_provider"] = judge_provider
    analysis["judge_model"]   = judge_model

    if args.json:
        print(json.dumps(analysis, indent=2))
        return

    if not args.quiet:
        print_monitor_summary(analysis, args.input)

    # Save output
    out_path = args.output
    if not out_path:
        stem     = Path(args.input).stem
        out_path = f"results/monitor_{stem}.json"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)

    if not args.quiet:
        print(f"  report saved: {out_path}")
        print()

    # Exit 1 if drift rate exceeds threshold
    drift_rate = analysis.get("drift_rate", 0.0)
    if drift_rate > args.threshold:
        if not args.quiet:
            print(RED(f"  ALERT: drift rate {drift_rate:.0%} exceeds threshold {args.threshold:.0%}"))
            print()
        sys.exit(1)


if __name__ == "__main__":
    main()
