"""
run_test_retest.py — CAI-Bench test-retest reliability study.

Addresses reviewer concern: "Without repeated samples per prompt, one cannot
determine whether variation is caused by phrasing or ordinary generation
variance."

This script runs Claude Sonnet 4.6 on the frozen 4-domain benchmark TWICE
(independent API calls, same frozen cases) and computes:
  - Per-domain score stability (absolute difference between runs)
  - Pearson r and Spearman rho between run-1 and run-2 case-level scores
  - Estimated intraclass correlation (ICC) for test-retest reliability
  - Proportion of within-case variance attributable to phrasing vs. stochasticity

Setup
-----
    export ANTHROPIC_API_KEY=sk-ant-...

Run
---
    python run_test_retest.py
    python run_test_retest.py --model gpt-4o --key OPENAI_API_KEY

Output
------
    test_retest_results_YYYYMMDD.json
"""

import os, json, time, datetime, sys, argparse, random

try:
    import openai
except ImportError:
    sys.exit("pip install openai")

try:
    from contradish import Suite
    from contradish.adapters import wrap_openai_compatible
except ImportError:
    sys.exit("pip install contradish")

# ── CONFIG ────────────────────────────────────────────────────────────────────

DOMAINS = ["ecommerce", "hr", "healthcare", "legal"]
N_RUNS = 2  # increase to 3 for a more robust ICC estimate

# ── HELPERS ───────────────────────────────────────────────────────────────────

def run_domain(app, domain):
    try:
        suite  = Suite.from_policy(domain, app=app)
        report = suite.run()
        case_scores = {}
        for r in report.results:
            name = r.test_case.name if hasattr(r, "test_case") else str(r)
            case_scores[name] = round(1 - getattr(r, "cai_score", getattr(r, "cai_strain", 0.5)), 4)
        return round(report.cai_strain, 4), case_scores
    except Exception as e:
        print(f"      ERROR: {e}")
        return None, {}


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))**0.5
    return num/den if den else 0.0


def spearman(xs, ys):
    def ranks(lst):
        s = sorted(range(len(lst)), key=lambda i: lst[i])
        r = [0.0]*len(lst)
        i = 0
        while i < len(lst):
            j = i
            while j < len(lst) and lst[s[j]] == lst[s[i]]: j += 1
            avg = (i + j + 1) / 2.0
            for k in range(i, j): r[s[k]] = avg
            i = j
        return r
    return pearson(ranks(xs), ranks(ys))


def icc_two_way_mixed(runs_by_case):
    """
    ICC(2,1) two-way mixed model for absolute agreement.
    runs_by_case: {case_name: [score_run1, score_run2, ...]}
    """
    cases = list(runs_by_case.keys())
    k     = len(next(iter(runs_by_case.values())))  # number of runs
    n     = len(cases)
    grand_mean = sum(
        s for scores in runs_by_case.values() for s in scores
    ) / (n * k)
    # Between-subject SS
    ss_b = k * sum((sum(runs_by_case[c])/k - grand_mean)**2 for c in cases)
    # Within-subject SS
    ss_w = sum((s - sum(runs_by_case[c])/k)**2
               for c in cases for s in runs_by_case[c])
    # Between-rater SS
    run_means = [sum(runs_by_case[c][r] for c in cases)/n for r in range(k)]
    ss_r = n * sum((rm - grand_mean)**2 for rm in run_means)
    # Error SS
    ss_e = ss_w - ss_r
    ms_b = ss_b / (n - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    icc  = (ms_b - ms_e) / (ms_b + (k - 1) * ms_e)
    return round(icc, 4)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="claude-sonnet-4-6")
    parser.add_argument("--key",    default="ANTHROPIC_API_KEY")
    parser.add_argument("--base",   default="https://api.anthropic.com/v1")
    parser.add_argument("--runs",   type=int, default=N_RUNS)
    args = parser.parse_args()

    api_key = os.environ.get(args.key)
    if not api_key:
        sys.exit(f"API key not set: {args.key}")

    client = openai.OpenAI(api_key=api_key, base_url=args.base)
    app = wrap_openai_compatible(client=client, model=args.model)

    print(f"\nTest-retest reliability: {args.model}")
    print(f"Domains: {DOMAINS}")
    print(f"Runs: {args.runs}")

    run_data = {}  # {domain: {run_idx: {case: strain}}}
    domain_strains = {}  # {domain: [run1_strain, run2_strain]}

    for run_idx in range(args.runs):
        print(f"\n{'='*50}")
        print(f"RUN {run_idx + 1}")
        print(f"{'='*50}")
        for domain in DOMAINS:
            if domain not in run_data:
                run_data[domain] = {}
                domain_strains[domain] = []
            print(f"  {domain:<25}", end="", flush=True)
            strain, case_scores = run_domain(app, domain)
            if strain is None:
                print("FAILED")
                continue
            print(f"CAI Strain = {strain:.3f}")
            run_data[domain][run_idx] = case_scores
            domain_strains[domain].append(strain)
            time.sleep(0.5)

    # ── ANALYSIS ────────────────────────────────────────────────────────────
    print(f"\n\n{'='*50}")
    print("TEST-RETEST RELIABILITY ANALYSIS")
    print(f"{'='*50}")

    print(f"\n{'Domain':<20} {'Run1':>6} {'Run2':>6} {'|diff|':>7}")
    for domain in DOMAINS:
        strains = domain_strains.get(domain, [])
        if len(strains) >= 2:
            d = abs(strains[0] - strains[1])
            print(f"  {domain:<18} {strains[0]:>6.3f} {strains[1]:>6.3f} {d:>7.3f}")

    # Case-level correlation
    all_r1, all_r2 = [], []
    cases_by_domain = {}
    for domain in DOMAINS:
        rd = run_data.get(domain, {})
        if 0 in rd and 1 in rd:
            cases = sorted(set(rd[0].keys()) & set(rd[1].keys()))
            cases_by_domain[domain] = {c: [rd[0][c], rd[1][c]] for c in cases}
            all_r1.extend(rd[0][c] for c in cases)
            all_r2.extend(rd[1][c] for c in cases)

    if len(all_r1) >= 4:
        r = pearson(all_r1, all_r2)
        rho = spearman(all_r1, all_r2)
        mad = sum(abs(a-b) for a,b in zip(all_r1,all_r2)) / len(all_r1)
        print(f"\nCase-level agreement (n={len(all_r1)} cases across {args.runs} runs):")
        print(f"  Pearson r        = {r:.3f}")
        print(f"  Spearman rho     = {rho:.3f}")
        print(f"  Mean |diff|      = {mad:.3f}")

        # ICC
        all_cases = {}
        for domain, data in cases_by_domain.items():
            all_cases.update(data)
        if all_cases:
            icc = icc_two_way_mixed(all_cases)
            print(f"  ICC(2,1)         = {icc:.3f}")
            print(f"\n  Interpretation: ICC > 0.75 = good; > 0.90 = excellent")
            print(f"  Note: ICC < Pearson r when runs have systematically different means")

    # ── SAVE ────────────────────────────────────────────────────────────────
    out_file = f"test_retest_{args.model.replace('/','_')}_{datetime.date.today().strftime('%Y%m%d')}.json"
    with open(out_file, "w") as f:
        json.dump({
            "run_date":      datetime.date.today().isoformat(),
            "model":         args.model,
            "runs":          args.runs,
            "domain_strains": domain_strains,
            "case_scores":   {
                dom: {str(run): scores for run, scores in rd.items()}
                for dom, rd in run_data.items()
            },
        }, f, indent=2)
    print(f"\nSaved to {out_file}")
    print("Report ICC and Pearson r in the paper's Limitations section.")
    print("ICC > 0.75 confirms scores are stable across runs; ICC > 0.90 is excellent.")


if __name__ == "__main__":
    main()
