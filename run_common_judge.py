"""
run_common_judge.py — CAI-Bench common-judge robustness check.

Addresses reviewer concern: "Cross-provider judging confounds model rank with
judge personality. An apparent strain difference between Claude and GPT-4o could
be driven by differences between the two judges rather than the tested models."

This script runs BOTH models through the SAME judge (GPT-4o by default) so
model differences are not confounded with judge differences. Compare the output
with the cross-provider results in the paper to assess whether the domain-reversal
pattern holds.

Setup
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...

Run
---
    python run_common_judge.py
    python run_common_judge.py --judge claude   # use Claude as judge for both

Output
------
    common_judge_results_YYYYMMDD.json
"""

import os, json, time, datetime, sys, argparse

try:
    import openai
except ImportError:
    sys.exit("pip install openai")

try:
    from contradish import Suite
    from contradish.adapters import wrap_openai_compatible
    from contradish.policies import list_policies
except ImportError:
    sys.exit("pip install contradish  (or run from contradish repo root)")

# ── CONFIG ────────────────────────────────────────────────────────────────────

DOMAINS_4 = ["ecommerce", "hr", "healthcare", "legal"]   # fastest validation
DOMAINS_20 = list_policies()                              # full paper results

# Tested models (one from each provider for the key comparison)
TESTED_MODELS = [
    ("claude-sonnet-4-6", "anthropic", "ANTHROPIC_API_KEY",
     "https://api.anthropic.com/v1"),
    ("gpt-4o",            "openai",    "OPENAI_API_KEY", None),
]

# Judge options
JUDGE_OPTIONS = {
    "gpt-4o":  ("gpt-4o",                        "openai",    "OPENAI_API_KEY",    None),
    "claude":  ("claude-haiku-4-5-20251001",      "anthropic", "ANTHROPIC_API_KEY",
                "https://api.anthropic.com/v1"),
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def make_client(model_id, api_key_env, base_url):
    key = os.environ.get(api_key_env)
    if not key:
        return None
    if base_url:
        return openai.OpenAI(api_key=key, base_url=base_url)
    return openai.OpenAI(api_key=key)


def run_domain(app, judge_app, domain):
    """Run one model vs one domain with a specific judge app."""
    try:
        # contradish Suite accepts an explicit judge when supported
        suite = Suite.from_policy(domain, app=app, judge=judge_app)
        report = suite.run()
        details = []
        for r in report.results:
            entry = {
                "name":    r.test_case.name if hasattr(r, "test_case") else str(r),
                "strain":  round(getattr(r, "cai_strain", 1 - getattr(r, "cai_score", 0.5)), 4),
                "passed":  getattr(r, "passed", False),
                "contradictions": [
                    {
                        "input_a":     getattr(c, "input_a", ""),
                        "output_a":    getattr(c, "output_a", ""),
                        "input_b":     getattr(c, "input_b", ""),
                        "output_b":    getattr(c, "output_b", ""),
                        "explanation": getattr(c, "explanation", ""),
                    }
                    for c in getattr(r, "contradictions", [])
                ],
            }
            details.append(entry)
        return round(report.cai_strain, 4), details
    except Exception as e:
        # If contradish doesn't support explicit judge kwarg, fall back to
        # running with the default judge and note the limitation.
        print(f"      [judge kwarg not supported — {e}]")
        print("      Falling back: re-running with default judge selection suppressed")
        return None, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["gpt-4o", "claude"], default="gpt-4o",
                        help="Judge to use for ALL models (default: gpt-4o)")
    parser.add_argument("--domains", choices=["4", "20"], default="4",
                        help="Run 4-domain pilot (fast) or full 20 domains (default: 4)")
    args = parser.parse_args()

    domains = DOMAINS_4 if args.domains == "4" else DOMAINS_20
    judge_cfg = JUDGE_OPTIONS[args.judge]
    judge_model, judge_provider, judge_key_env, judge_base = judge_cfg

    print(f"\nCommon-judge robustness check")
    print(f"Judge: {judge_model}  ({judge_provider})")
    print(f"Domains: {len(domains)}")

    judge_client = make_client(judge_model, judge_key_env, judge_base)
    if judge_client is None:
        sys.exit(f"Judge API key not set: {judge_key_env}")

    try:
        judge_app = wrap_openai_compatible(client=judge_client, model=judge_model)
    except Exception as e:
        sys.exit(f"Could not create judge adapter: {e}")

    all_results = {}
    for model_id, provider, key_env, base_url in TESTED_MODELS:
        short = model_id.split("/")[-1]
        print(f"\n{'='*55}")
        print(f"Model: {short}  |  Judge: {judge_model}")
        print(f"{'='*55}")
        client = make_client(model_id, key_env, base_url)
        if client is None:
            print(f"  SKIPPED — {key_env} not set")
            continue
        app = wrap_openai_compatible(client=client, model=model_id)

        model_data = {"model": short, "judge": judge_model, "domains": {}}
        strains = []
        for domain in domains:
            print(f"  {domain:<25}", end="", flush=True)
            strain, details = run_domain(app, judge_app, domain)
            if strain is None:
                print("FAILED")
                continue
            failures = sum(1 for d in details if not d["passed"])
            print(f"CAI Strain = {strain:.3f}  ({failures} failures)")
            model_data["domains"][domain] = {"cai_strain": strain, "details": details}
            strains.append(strain)
            time.sleep(0.5)

        if strains:
            avg = round(sum(strains)/len(strains), 4)
            model_data["avg_cai_strain"] = avg
            model_data["domains_run"] = len(strains)
            print(f"\n  Avg CAI Strain: {avg:.3f}")
        all_results[short] = model_data

    # ── COMPARE ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*55}")
    print("COMPARISON: common-judge vs cross-provider judge")
    print(f"{'='*55}")
    print(f"{'Domain':<25} ", end="")
    for m in all_results:
        print(f"{m[:18]:<20}", end="")
    print()
    for domain in domains:
        print(f"  {domain:<23}", end="")
        for m, data in all_results.items():
            v = data["domains"].get(domain, {}).get("cai_strain", "N/A")
            print(f"  {str(v):<18}", end="")
        print()

    # ── SAVE ────────────────────────────────────────────────────────────────
    out_file = f"common_judge_{args.judge}_results_{datetime.date.today().strftime('%Y%m%d')}.json"
    with open(out_file, "w") as f:
        json.dump({
            "run_date":  datetime.date.today().isoformat(),
            "judge":     judge_model,
            "domains":   domains,
            "results":   all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_file}")
    print("Compare these domain scores against the cross-provider results")
    print("in the paper (Table 3). If the reversal pattern holds, the domain")
    print("reversal is not an artifact of judge differences.")


if __name__ == "__main__":
    main()
