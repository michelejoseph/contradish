#!/usr/bin/env python3
"""
Turn completed annotator CSVs into real equivalence_confidence values and
write them into contradish/benchmarks/v2/<domain>.json for every domain that
has annotator CSVs present, replacing the placeholder 1.0.

Usage:
    python3 compute_equivalence_confidence.py

Run this from inside the equivalence-audit/ folder. It expects the filled-in
CSVs to be in the same folder, named like:
    medication_equivalence_audit_<anyname>.csv
    immigration_equivalence_audit_<anyname>.csv
(any number of annotators per domain, any suffix after "audit_")

It does NOT touch the *_TEMPLATE.csv files, only files matching the pattern
above that are not templates.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO_ROOT, "contradish", "benchmarks", "v2")

# Every domain with a benchmarks/v2/<domain>.json file is in scope -- this
# used to be a hardcoded ["medication", "immigration"] list that had to be
# edited by hand each time a new domain got an equivalence audit. Deriving
# it from the directory means dropping a new domain's completed CSVs in
# here is enough; no code change needed.
DOMAINS = sorted(
    os.path.splitext(f)[0]
    for f in os.listdir(BENCH_DIR)
    if f.endswith(".json") and f != "eval_report_v2.json"
)


def find_annotator_files(domain):
    pattern = f"{domain}_equivalence_audit_*.csv"
    files = [f for f in glob.glob(pattern) if "TEMPLATE" not in f]
    return files


def load_votes(files):
    """Returns votes[(case_id, variant_number)] = list of 'Y'/'N' across all files/annotators."""
    votes = defaultdict(list)
    notes = defaultdict(list)
    for fpath in files:
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                j = row.get("judgment_Y_or_N", "").strip().upper()
                if j not in ("Y", "N"):
                    print(f"  WARNING: {fpath} case {row.get('case_id')} "
                          f"variant {row.get('variant_number')} has judgment "
                          f"'{row.get('judgment_Y_or_N')}', skipping this vote")
                    continue
                key = (row["case_id"], int(row["variant_number"]))
                votes[key].append(j)
                note = (row.get("notes_optional") or "").strip()
                if note:
                    notes[key].append(f"{os.path.basename(fpath)}: {note}")
    return votes, notes


def main():
    any_domain_processed = False
    for domain in DOMAINS:
        files = find_annotator_files(domain)
        if not files:
            print(f"{domain}: no annotator CSVs found (looked for "
                  f"{domain}_equivalence_audit_*.csv, excluding TEMPLATE), skipping")
            continue
        print(f"\n{domain}: found {len(files)} annotator file(s): {files}")
        votes, notes = load_votes(files)

        json_path = os.path.join(BENCH_DIR, f"{domain}.json")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        case_report = []
        for case in data["cases"]:
            case_id = case["id"]
            n_variants = len(case["adversarial"])
            all_votes = []
            case_notes = []
            missing = []
            for v in range(1, n_variants + 1):
                key = (case_id, v)
                if key not in votes:
                    missing.append(v)
                    continue
                all_votes.extend(votes[key])
                case_notes.extend(notes.get(key, []))

            if missing:
                print(f"  WARNING: {case_id} missing votes for variant(s) "
                      f"{missing}, case left unchanged")
                continue

            y_count = sum(1 for v in all_votes if v == "Y")
            total = len(all_votes)
            eq = round(y_count / total, 3) if total else None

            if eq is None:
                continue

            old_eq = case.get("equivalence_confidence")
            case["equivalence_confidence"] = eq

            bucket = ("expert-confirmed" if eq >= 0.80
                      else "contested" if eq >= 0.50
                      else "excluded")
            case_report.append((case_id, old_eq, eq, bucket, total, case_notes))

        # write the patched JSON back
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)
            f.write("\n")

        n_cases = len(case_report)
        n_confirmed = sum(1 for r in case_report if r[3] == "expert-confirmed")
        n_contested = sum(1 for r in case_report if r[3] == "contested")
        n_excluded = sum(1 for r in case_report if r[3] == "excluded")
        eq_coverage = round(n_confirmed / len(data["cases"]), 3) if data["cases"] else 0

        print(f"\n  {domain}: wrote {json_path}")
        print(f"  {n_cases}/{len(data['cases'])} cases updated "
              f"({len(data['cases']) - n_cases} left unchanged, see warnings above)")
        print(f"  expert-confirmed (EQ>=0.80): {n_confirmed}")
        print(f"  contested (0.50<=EQ<0.80):   {n_contested}")
        print(f"  excluded (EQ<0.50):          {n_excluded}")
        print(f"  eq_coverage: {eq_coverage}")

        contested_or_excluded = [r for r in case_report if r[3] != "expert-confirmed"]
        if contested_or_excluded:
            print(f"\n  Cases that did NOT clear 0.80 (review before publishing):")
            for case_id, old_eq, eq, bucket, total, case_notes in contested_or_excluded:
                print(f"    {case_id}: EQ={eq} ({bucket}, {total} votes)")
                for n in case_notes:
                    print(f"        note: {n}")

        any_domain_processed = True

    if not any_domain_processed:
        print("\nNothing processed. Drop the completed annotator CSVs in this "
              "folder and re-run.")
        sys.exit(1)

    print("\nDone. The equivalence_confidence fields in "
          "contradish/benchmarks/v2/medication.json and immigration.json "
          "are now real numbers instead of the 1.0 placeholder.")
    print("Next: update BENCHMARK.md's placeholder note for these two domains, "
          "and re-run any benchmark reports you want headline_strain/contested_strain "
          "numbers for.")


if __name__ == "__main__":
    main()
