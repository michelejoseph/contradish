#!/usr/bin/env python3
"""
policy-audit/run_policy_audit.py -- CLI runner for contradish's
policy-contradiction scanner (contradish/policy_contradiction.py).

Usage:
    # Live LLM judge (needs ANTHROPIC_API_KEY or OPENAI_API_KEY):
    python3 run_policy_audit.py legal.html compliance.html --out report.json

    # No API key available -- reviewer stands in via a ManualJudgeClient
    # lookup table (see policy_audit_legal_compliance_judgments.py for the
    # worked example against contradish.com's own legal.html + compliance.html):
    python3 run_policy_audit.py legal.html compliance.html \\
        --manual-judgments policy_audit_legal_compliance_judgments.py --out report.json

Each positional argument is a path to a local HTML file; its basename is
used as the claim source label (so pass the file the way you'd want it
cited in a finding, e.g. "legal.html" not "/tmp/whatever/legal.html").
"""
import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contradish.policy_contradiction import (
    extract_policy_claims,
    PolicyContradictionJudge,
    ManualJudgeClient,
    audit_policies,
    audit_policies_manual,
)


def _load_judgments_module(path):
    spec = importlib.util.spec_from_file_location("_manual_judgments", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="+", help="HTML files to audit for cross-page (and within-page) contradictions")
    ap.add_argument("--max-pairs", type=int, default=150, help="cap on candidate pairs sent to the judge (default 150; generate_candidate_pairs naturally caps out below this for small corpora, so raising it costs nothing when there's nothing left to include)")
    ap.add_argument("--out", default=None, help="write the JSON report to this path (in addition to printing a summary)")
    ap.add_argument("--manual-judgments", default=None,
                     help="path to a .py file defining JUDGMENTS (dict keyed by pair_key) and DEFAULT (verdict dict) "
                          "-- use this when no ANTHROPIC_API_KEY/OPENAI_API_KEY is available")
    args = ap.parse_args()

    claims = []
    for page in args.pages:
        with open(page, encoding="utf-8") as f:
            html = f.read()
        source = os.path.basename(page)
        claims += extract_policy_claims(html, source=source)

    if not claims:
        print("No claim-bearing sentences extracted -- nothing to audit.", file=sys.stderr)
        sys.exit(1)

    if args.manual_judgments:
        mod = _load_judgments_module(args.manual_judgments)
        judge = ManualJudgeClient(mod.JUDGMENTS, default=getattr(mod, "DEFAULT", None))
        report = audit_policies_manual(claims, judge, max_pairs=args.max_pairs)
    else:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            print("No ANTHROPIC_API_KEY or OPENAI_API_KEY set, and no --manual-judgments given. "
                  "Either set a key or pass --manual-judgments.", file=sys.stderr)
            sys.exit(1)
        try:
            from contradish.llm import LLMClient
        except ImportError as e:
            print(f"Couldn't import contradish.llm ({e}). Run this from within the contradish "
                  "repo, or pass --manual-judgments instead.", file=sys.stderr)
            sys.exit(1)
        judge = PolicyContradictionJudge(LLMClient())
        report = audit_policies(claims, judge, max_pairs=args.max_pairs)

    report.print_summary()

    if args.out:
        with open(args.out, "w") as f:
            f.write(report.to_json())
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
