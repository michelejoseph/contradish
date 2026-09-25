#!/usr/bin/env python3
"""
equivalence-audit/inter_rater_agreement.py -- chance-corrected inter-rater
reliability for a completed equivalence audit domain, plus a breakdown of
WHERE disagreement concentrates.

compute_equivalence_confidence.py already pools every annotator's Y/N votes
per variant into one equivalence_confidence number and buckets cases as
expert-confirmed/contested/excluded -- useful for scoring the benchmark, but
it doesn't answer a different, prior question: how reliable are these
annotators' judgments with each other in the first place, corrected for how
skewed the Y/N base rate is? Raw percent agreement overstates reliability
whenever most items get the same label anyway (if 90% of variants are truly
Y, two annotators guessing at random already "agree" 82% of the time on Y
alone). Cohen's kappa corrects for that: kappa = (p_observed - p_chance) /
(1 - p_chance), where p_chance is the agreement two annotators would show if
their Y-rates were independent. kappa near 0 means agreement no better than
the raters' own base rates predict; kappa near 1 means real, non-chance
agreement.

This is the exact number named across this project's own moat/novelty
analysis as the most urgent missing piece: "a validated taxonomy... with
inter-rater agreement from adjudicators who are not contradish." It's cheap
to compute once two annotators have independently completed the same
domain -- no new data collection, just arithmetic on files already in this
folder.

WHAT THIS SCRIPT DOES NOT DO: it doesn't decide who's "right" on a
disagreement, doesn't average votes into a score (compute_equivalence_
confidence.py does that), and doesn't flag any individual case_id as
defective on its own -- see the category breakdown below for why
per-category patterns are more informative than per-case ones.

Usage:
    python3 inter_rater_agreement.py <domain> <rater_a.csv> <rater_b.csv>

Example:
    python3 inter_rater_agreement.py medication \\
        medication_equivalence_audit_katranji.csv \\
        medication_equivalence_audit_michele.csv
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict

# Regex-tagged categories worth checking for a systematic (not random)
# disagreement pattern -- i.e. is disagreement concentrated in cases with
# some identifiable textual property, rather than spread evenly. Extend
# this dict per domain as patterns are found; it is intentionally small and
# inspectable, not learned, same design choice policy_contradiction.py and
# distinction.py already make for their own topic/claim heuristics.
DEFAULT_CATEGORIES: dict[str, re.Pattern] = {
    "hypothetical": re.compile(r"\bhypothetical", re.I),
    "emotional_distress": re.compile(
        r"\b(terrified|anxiety|anxious|afraid|scared|desperate|hate how|"
        r"worse on this|crisis|worried|stress)\b", re.I),
}


def load(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["case_id"], r["variant_number"]): r for r in rows}


def cohens_kappa(pairs: list) -> tuple:
    """pairs: list of (rater_a_label, rater_b_label) in {'Y','N'}.
    Returns (percent_agreement, chance_agreement, kappa)."""
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0, float("nan"))
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    p_y_a = sum(1 for a, _ in pairs if a == "Y") / n
    p_y_b = sum(1 for _, b in pairs if b == "Y") / n
    pe = p_y_a * p_y_b + (1 - p_y_a) * (1 - p_y_b)
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return (po, pe, kappa)


def kappa_label(k: float) -> str:
    """Landis & Koch (1977) conventional bands -- widely used, not this
    project's own invention, cited so the number has an external anchor
    rather than an arbitrary house threshold."""
    if k != k:  # nan
        return "undefined"
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def analyze(domain: str, path_a: str, path_b: str, categories: dict = DEFAULT_CATEGORIES) -> dict:
    a = load(path_a)
    b = load(path_b)
    common = sorted(set(a) & set(b))

    pairs = []
    rows = []
    for k in common:
        ja = a[k]["judgment_Y_or_N"].strip().upper()
        jb = b[k]["judgment_Y_or_N"].strip().upper()
        if ja not in ("Y", "N") or jb not in ("Y", "N"):
            continue
        pairs.append((ja, jb))
        rows.append((k, ja, jb, a[k]))

    po, pe, kappa = cohens_kappa(pairs)

    # category breakdown
    cat_stats = {}
    tagged = defaultdict(list)
    for k, ja, jb, row in rows:
        text = row.get("adversarial_variant", "")
        matched = [name for name, rx in categories.items() if rx.search(text)]
        cat = matched[0] if matched else "other"
        tagged[cat].append((k, ja, jb))

    for cat, items in tagged.items():
        n = len(items)
        dis = sum(1 for _, ja, jb in items if ja != jb)
        direction = defaultdict(int)
        for _, ja, jb in items:
            if ja != jb:
                direction[f"a={ja}/b={jb}"] += 1
        cat_stats[cat] = {
            "n": n, "disagreements": dis,
            "disagreement_rate": round(dis / n, 3) if n else None,
            "direction": dict(direction),
        }

    disagreements = [
        {"key": k, "rater_a": ja, "rater_b": jb,
         "case_name": row.get("case_name", ""),
         "variant": row.get("adversarial_variant", "")}
        for k, ja, jb, row in rows if ja != jb
    ]

    return {
        "domain": domain,
        "rater_a_file": path_a, "rater_b_file": path_b,
        "n_common": len(common), "n_scored": len(pairs),
        "percent_agreement": round(po, 3),
        "chance_agreement": round(pe, 3),
        "cohens_kappa": round(kappa, 3) if kappa == kappa else None,
        "kappa_label": kappa_label(kappa),
        "category_breakdown": cat_stats,
        "disagreements": disagreements,
    }


def print_report(result: dict) -> None:
    sep = "─" * 72
    print()
    print(f"  INTER-RATER AGREEMENT  ·  {result['domain']}")
    print(sep)
    print(f"  rater A: {result['rater_a_file']}")
    print(f"  rater B: {result['rater_b_file']}")
    print(f"  n scored: {result['n_scored']} / {result['n_common']} common items")
    print()
    print(f"  percent agreement   : {result['percent_agreement']:.1%}")
    print(f"  chance agreement    : {result['chance_agreement']:.1%}")
    k = result["cohens_kappa"]
    print(f"  Cohen's kappa       : {k}  ({result['kappa_label']})")
    if k is not None and k < 0.40 and result["percent_agreement"] > 0.70:
        print("  NOTE: percent agreement looks fine on its own, but kappa is low --")
        print("  this means the raters' Y-rates are both high/skewed, so most of the raw")
        print("  agreement is what you'd expect from chance alone, not real consensus.")
    print()
    print("  by category:")
    for cat, stats in sorted(result["category_breakdown"].items(),
                              key=lambda kv: -(kv[1]["disagreement_rate"] or 0)):
        print(f"    {cat:<20} n={stats['n']:>3}  disagreements={stats['disagreements']:>3}  "
              f"rate={stats['disagreement_rate']}  {stats['direction']}")
    print()
    if result["disagreements"]:
        print(f"  {len(result['disagreements'])} disagreement(s):")
        for d in result["disagreements"]:
            print(f"    {d['key']}  A={d['rater_a']} B={d['rater_b']}  [{d['case_name']}]")
            print(f"      {d['variant'][:100]}")
    print()


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    domain, path_a, path_b = sys.argv[1], sys.argv[2], sys.argv[3]
    result = analyze(domain, path_a, path_b)
    print_report(result)

    import json
    out_path = f"{domain}_inter_rater_agreement.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
