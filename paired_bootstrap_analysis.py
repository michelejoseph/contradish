#!/usr/bin/env python3
"""Paired bootstrap analysis for CAI-Bench v1 case-level results.

Usage:
  python paired_bootstrap_analysis.py \
    --claude claude-sonnet-4-6_2026-04-17.json \
    --gpt gpt-4o-mini_2026-04-17.json \
    --outdir analysis_output
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def paired_bootstrap(values, rng, n_boot=50000):
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[boot_idx].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    perm_means = (values * signs).mean(axis=1)
    observed = abs(values.mean())
    p = (np.sum(np.abs(perm_means) >= observed) + 1) / (n_boot + 1)
    return values.mean(), low, high, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", required=True)
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--outdir", default="analysis_output")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--bootstrap-samples", type=int, default=50000)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.claude) as f:
        claude = json.load(f)
    with open(args.gpt) as f:
        gpt = json.load(f)

    rows = []
    for domain in claude["policies_tested"]:
        c = {x["name"]: 1.0 - float(x["cai_score"])
             for x in claude["results"][domain]["details"]}
        g = {x["name"]: 1.0 - float(x["cai_score"])
             for x in gpt["results"][domain]["details"]}
        if set(c) != set(g):
            raise ValueError(f"Case mismatch in {domain}")
        for case in sorted(c):
            rows.append({
                "domain": domain,
                "case": case,
                "claude_strain": c[case],
                "gpt4o_mini_strain": g[case],
                "difference": c[case] - g[case],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "paired_case_scores.csv", index=False)
    rng = np.random.default_rng(args.seed)
    summaries = []
    for domain, group in df.groupby("domain", sort=False):
        md, lo, hi, p = paired_bootstrap(
            group["difference"].to_numpy(), rng, args.bootstrap_samples
        )
        summaries.append({
            "domain": domain,
            "n_cases": len(group),
            "claude_mean_strain": group["claude_strain"].mean(),
            "gpt4o_mini_mean_strain": group["gpt4o_mini_strain"].mean(),
            "mean_difference": md,
            "bootstrap_95_ci_low": lo,
            "bootstrap_95_ci_high": hi,
            "sign_flip_p_value": p,
        })
    pd.DataFrame(summaries).to_csv(out / "paired_bootstrap_summary.csv", index=False)

if __name__ == "__main__":
    main()
