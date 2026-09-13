#!/usr/bin/env python3
"""
Turn completed r_audit_<annotator>.csv files into a per-technique
consensus report, checked against decision_relevance.py's REAL shipped
defaults (imported directly, not re-typed here, so this can't drift out of
sync with the source it's auditing).

Unlike compute_equivalence_confidence.py, this does NOT write anything
back automatically -- default_technique_drs()'s relevance labels are
executable source code, not a JSON data file, and auto-patching source
from a CSV aggregation script is a materially bigger, riskier move than
patching a data field. This script's job ends at producing a report a
human reads before deciding whether to change decision_relevance.py.

Usage:
    python3 compute_relevance_consensus.py

Run from inside r-audit/. Expects completed CSVs in the same folder,
named like:
    r_audit_<anyname>.csv
(any number of annotators, any suffix after "r_audit_")
It does NOT touch r_audit_TEMPLATE.csv, only files matching the pattern
above that are not the template.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_annotator_files():
    files = [f for f in glob.glob("r_audit_*.csv") if "TEMPLATE" not in f]
    return files


def load_votes(files):
    """Returns votes[technique] = list of 'Y'/'N'/'C' across all files/annotators,
    and notes[technique] = list of attributed note strings."""
    votes = defaultdict(list)
    notes = defaultdict(list)
    for fpath in files:
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                j = (row.get("judgment") or "").strip().upper()
                technique = row.get("technique", "").strip()
                if j not in ("Y", "N", "C"):
                    print(f"  WARNING: {fpath} technique={technique!r} case={row.get('case_id')} "
                          f"has judgment {row.get('judgment')!r}, skipping this vote")
                    continue
                votes[technique].append(j)
                note = (row.get("notes_optional") or "").strip()
                if note:
                    notes[technique].append(f"{os.path.basename(fpath)} ({row.get('case_id')}): {note}")
    return votes, notes


_VERDICT_TO_RELEVANCE = {"Y": "irrelevant", "N": "relevant", "C": "conditional"}


def majority_verdict(technique_votes):
    total = len(technique_votes)
    if total == 0:
        return None, {}
    counts = {v: technique_votes.count(v) for v in ("Y", "N", "C")}
    top, top_n = max(counts.items(), key=lambda kv: kv[1])
    if top_n / total > 0.5:
        return top, counts
    return None, counts   # no strict majority -> disputed


def main():
    files = find_annotator_files()
    if not files:
        print("No annotator CSVs found (looked for r_audit_*.csv, excluding "
              "TEMPLATE). Drop completed CSVs in this folder and re-run.")
        sys.exit(1)

    print(f"found {len(files)} annotator file(s): {files}")
    votes, notes = load_votes(files)

    from contradish.decision_relevance import default_technique_drs
    shipped = default_technique_drs("_r_audit_reference").factors

    report = []
    for technique in sorted(shipped):
        technique_votes = votes.get(technique, [])
        verdict, counts = majority_verdict(technique_votes)
        shipped_relevance = shipped[technique].relevance
        consensus_relevance = _VERDICT_TO_RELEVANCE.get(verdict) if verdict else "disputed"
        agrees = (consensus_relevance == shipped_relevance) if verdict else None
        report.append({
            "technique": technique,
            "shipped_default": shipped_relevance,
            "n_votes": len(technique_votes),
            "y": counts.get("Y", 0), "n": counts.get("N", 0), "c": counts.get("C", 0),
            "consensus": consensus_relevance,
            "agrees_with_shipped": agrees,
            "notes": notes.get(technique, []),
        })

    print("\n" + "=" * 78)
    print("R-AUDIT CONSENSUS REPORT")
    print("=" * 78)
    n_review_needed = 0
    for r in report:
        flag = ""
        if r["n_votes"] == 0:
            flag = "  [NO VOTES -- not yet reviewed]"
        elif r["consensus"] == "disputed":
            flag = "  [DISPUTED -- no majority among reviewers]"
            n_review_needed += 1
        elif not r["agrees_with_shipped"]:
            flag = f"  [REVIEW NEEDED -- shipped={r['shipped_default']!r}, reviewers said {r['consensus']!r}]"
            n_review_needed += 1
        print(f"\n{r['technique']:14s} shipped={r['shipped_default']:11s} "
              f"votes: Y={r['y']} N={r['n']} C={r['c']} (n={r['n_votes']}){flag}")
        for note in r["notes"]:
            print(f"    note: {note}")

    print("\n" + "-" * 78)
    print(f"{n_review_needed} technique(s) need review before decision_relevance.py's "
          f"defaults can be called independently validated.")
    if n_review_needed == 0 and all(r["n_votes"] > 0 for r in report):
        print("All 8 technique defaults are corroborated by this review -- consider "
              "noting the reviewer count and date in decision_relevance.py's module "
              "docstring and BENCHMARK.md.")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r_audit_consensus.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote machine-readable report to {out_path}")
    print("Nothing in contradish/decision_relevance.py was changed by this script -- "
          "that's a deliberate human decision, not an automatic one. See the module "
          "docstring at the top of this file for why.")


if __name__ == "__main__":
    main()
