#!/usr/bin/env python3
"""
Generate <domain>_equivalence_audit_TEMPLATE.csv for every CAI-Bench domain
that doesn't already have one, from the case/adversarial data already in
contradish/benchmarks/v2/<domain>.json -- same shape as the medication and
immigration templates (144 rows: 18 cases x 8 adversarial variants each),
with judgment_Y_or_N and notes_optional left blank for an annotator to fill.

Run from inside equivalence-audit/. Skips a domain if its TEMPLATE.csv
already exists (won't overwrite medication/immigration).
"""
import csv
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(os.path.dirname(REPO_ROOT), "contradish", "benchmarks", "v2")

FIELDS = ["domain", "case_id", "case_name", "severity", "original_question",
          "variant_number", "adversarial_variant", "judgment_Y_or_N", "notes_optional"]


def main():
    domain_files = sorted(
        os.path.basename(f)[:-5] for f in glob.glob(os.path.join(BENCH_DIR, "*.json"))
        if os.path.basename(f) != "eval_report_v2.json"
    )
    generated = []
    skipped = []
    for domain in domain_files:
        out_path = os.path.join(REPO_ROOT, f"{domain}_equivalence_audit_TEMPLATE.csv")
        if os.path.exists(out_path):
            skipped.append(domain)
            continue

        with open(os.path.join(BENCH_DIR, f"{domain}.json"), encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for case in data["cases"]:
            variants = case["adversarial"]
            assert len(variants) == 8, f"{domain}/{case['id']}: expected 8 variants, got {len(variants)}"
            for i, variant in enumerate(variants, start=1):
                rows.append({
                    "domain": domain,
                    "case_id": case["id"],
                    "case_name": case["name"],
                    "severity": case["severity"],
                    "original_question": case["original"],
                    "variant_number": i,
                    "adversarial_variant": variant,
                    "judgment_Y_or_N": "",
                    "notes_optional": "",
                })

        assert len(rows) == 144, f"{domain}: expected 144 rows, got {len(rows)}"

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        generated.append((domain, len(rows)))

    print(f"Generated {len(generated)} templates:")
    for domain, n in generated:
        print(f"  {domain}_equivalence_audit_TEMPLATE.csv  ({n} rows)")
    if skipped:
        print(f"\nSkipped (already exist): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
