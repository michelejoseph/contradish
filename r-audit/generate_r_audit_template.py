#!/usr/bin/env python3
"""
Generate r_audit_TEMPLATE.csv: real (technique, domain, case) examples for
independently reviewing decision_relevance.py's default_technique_drs()
relevance labels -- the same kind of external review the equivalence audit
already gives equivalence_confidence, applied to R instead.

Unlike equivalence_confidence (a per-case judgment), a technique's
relevance label in default_technique_drs() is a GLOBAL claim: "this
technique is categorically framing-only (or categorically fact-changing,
or conditional), across every domain." So instead of reviewing every case,
this samples several REAL examples of each technique from DIFFERENT
domains -- if a technique's relevance judgment doesn't hold up across
domains it wasn't checked against when the default was written, that's
exactly the disagreement this audit exists to surface.

Sampling: for each of the 8 techniques, 5 examples from 5 different v2
policy domains (40 rows total), chosen with a fixed seed so the sample is
reproducible and auditable, not cherry-picked. Domains rotate so each of
the 20 v2 domains contributes to about 2 techniques' worth of examples
across the full sheet, not just whichever domain happens to be first.

Usage:
    python3 generate_r_audit_template.py
    (run from inside r-audit/; reads ../contradish/benchmarks/v2/*.json,
    writes r_audit_TEMPLATE.csv in the current directory)
"""
import csv
import json
import os
import random

TECHNIQUE_NAMES = [
    "emotional", "presuppose", "casual", "sympathy",
    "authority", "hypothetical", "boundary", "indirect",
]

POLICIES_V2 = [
    "ecommerce", "hr", "healthcare", "legal", "finance",
    "saas", "insurance", "education", "ai_safety",
    "travel", "mental_health", "government", "automotive", "real_estate",
    "medication", "telecommunications", "employment_disputes",
    "immigration", "food_delivery", "financial_planning",
]

EXAMPLES_PER_TECHNIQUE = 5
SEED = 20260913   # fixed so the sample is reproducible, not re-rolled per run

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO_ROOT, "contradish", "benchmarks", "v2")


def load_all_domains():
    domains = {}
    for policy in POLICIES_V2:
        path = os.path.join(BENCH_DIR, f"{policy}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            domains[policy] = json.load(f)["cases"]
    return domains


def main():
    domains = load_all_domains()
    if not domains:
        raise SystemExit(f"No v2 benchmark files found under {BENCH_DIR}")

    rng = random.Random(SEED)
    domain_names = sorted(domains.keys())
    n = len(domain_names)

    rows = []
    for t_idx, technique in enumerate(TECHNIQUE_NAMES):
        # Rotate the starting offset per technique so every technique draws
        # from a DIFFERENT slice of the domain list, not always the same
        # first few domains -- across all 8 techniques this touches most
        # or all of the 20 domains at least once.
        offset = (t_idx * EXAMPLES_PER_TECHNIQUE) % n
        chosen_domains = [domain_names[(offset + k) % n] for k in range(EXAMPLES_PER_TECHNIQUE)]

        for domain in chosen_domains:
            cases = [c for c in domains[domain] if len(c.get("adversarial", [])) > t_idx]
            if not cases:
                continue
            case = rng.choice(cases)
            # Deliberately no "current default label" column: this is a
            # blind review (see INSTRUCTIONS.md) -- showing the shipped
            # label would anchor the reviewer toward confirming it instead
            # of judging the example on its own. The comparison against
            # decision_relevance.py's real defaults happens later, in
            # compute_relevance_consensus.py, which reads the source
            # directly rather than trusting anything written into this CSV.
            rows.append({
                "technique": technique,
                "domain": domain,
                "case_id": case["id"],
                "case_name": case["name"],
                "original_question": case["original"],
                "adversarial_variant": case["adversarial"][t_idx],
                "judgment": "",
                "notes_optional": "",
            })

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r_audit_TEMPLATE.csv")
    fieldnames = [
        "technique", "domain", "case_id", "case_name",
        "original_question", "adversarial_variant", "judgment", "notes_optional",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out_path}")
    per_technique = {}
    for r in rows:
        per_technique.setdefault(r["technique"], []).append(r["domain"])
    for t, doms in per_technique.items():
        print(f"  {t:12s} <- {doms}")


if __name__ == "__main__":
    main()
