"""
run_benchmark_full.py — CAI-Bench full run across models and domains.

Run this locally with your API keys. It saves a results JSON you can
send back to update the paper with real data.

Two axes are measured for every model, wherever both are available:

    CAI Strain              — does the model contradict ITSELF across paraphrases?
                               (self-consistency; no external ground truth involved)
    Reality Strain           — does the model's resting answer match VERIFIED,
                               sourced ground truth (FDA guidance, statute text, etc.)?
                               Scored via narrow, auditable checks — critical-claim
                               presence, disqualifying-claim absence, truth score
                               against a sourced gold answer — not open-ended
                               "is this a contradiction" judgment.
    Admissibility Distance   — alpha * cai_strain + (1 - alpha) * reality_strain.
                               The combined score: a model can be perfectly
                               self-consistent and still be consistently WRONG
                               (0.00 CAI Strain, high Reality Strain) — that
                               failure mode is invisible to CAI Strain alone.

Reality Strain is computed by contradish's cross-provider judge (an OpenAI
model judges Anthropic outputs and vice versa — the same independence
pattern already used for CAI Strain), scored against the sourced ground
truth in ground-truth/*.json. It currently only covers models whose
provider is "openai" or "anthropic" (contradish.llm.LLMClient's two
supported providers) — other providers still get a CAI Strain score, just
no Reality Strain / Admissibility Distance for now.

Setup
-----
    pip install contradish openai litellm anthropic

    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    export GOOGLE_API_KEY=...           # optional, for gemini
    export MISTRAL_API_KEY=...          # optional
    export TOGETHER_API_KEY=...         # optional, for llama via Together AI

Run
---
    python run_benchmark_full.py

Output
------
    benchmark_results_YYYYMMDD.json    — per-model, per-domain CAI Strain,
                                         Reality Strain, and Admissibility
                                         Distance, plus verbatim failures
                                         (for Table 2 update)
"""

import os
import json
import time
import datetime
import sys

# ── CHECK DEPENDENCIES ────────────────────────────────────────────────────────
try:
    import openai
except ImportError:
    sys.exit("Missing: pip install openai")

try:
    import contradish
    from contradish import Suite
    from contradish.adapters import wrap_openai_compatible, wrap_litellm
    from contradish.policies import list_policies, load_policy
except ImportError:
    sys.exit("Missing: pip install contradish   (or run from the contradish directory)")

# Reality Strain is optional: a checkout without ground-truth/*.json, or without
# contradish.llm / contradish.judge / reality_strain.py alongside this script,
# still runs the CAI Strain half of the benchmark fine.
try:
    from contradish.llm import LLMClient
    from contradish.judge import Judge
    from reality_strain import (
        score_domain as _rs_score_domain,
        DEFAULT_DOMAINS as _RS_DEFAULT_DOMAINS,
        admissibility_distance as _rs_admissibility_distance,
        DEFAULT_ALPHA as _RS_DEFAULT_ALPHA,
    )
    REALITY_STRAIN_AVAILABLE = True
    _reality_strain_import_error = None
except ImportError as _e:
    REALITY_STRAIN_AVAILABLE = False
    _reality_strain_import_error = _e

# ── CONFIG ────────────────────────────────────────────────────────────────────

# All 20 domains — matches paper claim
ALL_DOMAINS = list_policies()

# Turn Reality Strain scoring on/off without touching the CAI Strain run.
RUN_REALITY_STRAIN = True

# contradish.llm.LLMClient only speaks OpenAI and Anthropic directly, so only
# models on these providers get a Reality Strain / Admissibility Distance
# score today. Google/Meta/Mistral models below still get CAI Strain.
REALITY_STRAIN_SUPPORTED_PROVIDERS = {"anthropic", "openai"}

# Models to run. Comment out any you don't have keys for.
# Format: (model_id, provider, api_key_env, base_url_or_None)
MODELS = [
    # Anthropic
    ("claude-sonnet-4-6",        "anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1"),

    # OpenAI
    ("gpt-4o",                   "openai",    "OPENAI_API_KEY",    None),
    ("gpt-4o-mini",              "openai",    "OPENAI_API_KEY",    None),

    # Google
    ("gemini/gemini-1.5-pro",    "google",    "GOOGLE_API_KEY",    None),

    # Groq / Llama
    ("groq/llama-3.1-70b-versatile", "meta",  "GROQ_API_KEY",      None),
]

# Cross-provider judge assignments: who judges whom
# Key = provider of model under test, Value = judge model to use
JUDGE_MAP = {
    "anthropic": "gpt-4o-mini",    # OpenAI judges Anthropic
    "openai":    "claude-haiku-4-5-20251001",  # Anthropic judges OpenAI
    "google":    "gpt-4o-mini",
    "mistral":   "claude-haiku-4-5-20251001",
    "meta":      "claude-haiku-4-5-20251001",
}

OUT_FILE = f"benchmark_results_{datetime.date.today().strftime('%Y%m%d')}.json"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def make_app(model_id, api_key_env, base_url):
    """Wrap a model as a contradish-compatible callable."""
    key = os.environ.get(api_key_env)
    if not key:
        return None  # skip this model
    if base_url and "anthropic" in base_url:
        # Anthropic via their OpenAI-compatible endpoint
        client = openai.OpenAI(api_key=key, base_url=base_url)
        return wrap_openai_compatible(client=client, model=model_id)
    elif base_url is None and "gemini" not in model_id and "mistral" not in model_id and "together" not in model_id:
        # Native OpenAI
        client = openai.OpenAI(api_key=key)
        return wrap_openai_compatible(client=client, model=model_id)
    else:
        # litellm handles gemini, mistral, together
        try:
            return wrap_litellm(model=model_id, api_key=key)
        except Exception as e:
            print(f"  [litellm error for {model_id}]: {e}")
            return None

def run_model_domain(app, domain):
    """Run one model against one domain. Returns (cai_strain, details list)."""
    try:
        suite  = Suite.from_policy(domain, app=app)
        report = suite.run()
        details = []
        for r in report.results:
            entry = {
                "name": r.test_case.name if hasattr(r, "test_case") else str(r),
                "cai_strain": round(getattr(r, "cai_strain", 1 - getattr(r, "cai_score", 0.5)), 4),
                "passed": getattr(r, "passed", False),
                "contradictions": [],
            }
            for c in getattr(r, "contradictions", []):
                entry["contradictions"].append({
                    "input_a":  getattr(c, "input_a",  ""),
                    "output_a": getattr(c, "output_a", ""),
                    "input_b":  getattr(c, "input_b",  ""),
                    "output_b": getattr(c, "output_b", ""),
                    "explanation": getattr(c, "explanation", ""),
                })
            details.append(entry)
        return round(report.cai_strain, 4), details
    except Exception as e:
        print(f"      ERROR: {e}")
        return None, []


def run_reality_strain_for_model(model_name, provider):
    """
    Run Reality Strain (ground-truth accuracy) for one model across the five
    sourced domains in ground-truth/*.json, using contradish's cross-provider
    judge (an OpenAI model judges Anthropic outputs and vice versa — same
    independence pattern as CAI Strain, so accuracy isn't graded by a judge
    from the model's own lab either).

    Returns None if scoring couldn't run at all (e.g. missing API key for
    the judge side); otherwise a dict with mean/weighted Reality Strain and
    a per-domain breakdown.
    """
    print("\n  --- Reality Strain (ground-truth accuracy) ---")
    try:
        model_client = LLMClient(provider=provider)
        judge_client = LLMClient.make_judge_client(model_provider=provider)
        judge = Judge(judge_client)
    except EnvironmentError as e:
        print(f"    SKIPPED: {e}")
        return None

    domain_results = []
    for domain in _RS_DEFAULT_DOMAINS:
        print(f"  domain: {domain:<25}", end="", flush=True)
        try:
            r = _rs_score_domain(domain, model_client, model_name, judge, verbose=False)
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        print(f"Reality Strain = {r['domain_reality_strain']:.3f}  "
              f"({r['n_auto_fail']} auto-fail / {r['n_cases']} cases)")
        domain_results.append(r)
        time.sleep(0.3)  # rate limit cushion

    if not domain_results:
        return None

    all_cases   = [c for r in domain_results for c in r["cases"]]
    all_strains = [c["reality_strain"] for c in all_cases]
    all_weights = [c["load_bearing_weight"] for c in all_cases]
    mean_rs     = round(sum(all_strains) / len(all_strains), 4)
    total_w     = sum(all_weights)
    weighted_rs = round(sum(s * w for s, w in zip(all_strains, all_weights)) / total_w, 4) if total_w else mean_rs

    return {
        "mean_reality_strain":     mean_rs,
        "weighted_reality_strain": weighted_rs,
        "domains": {r["domain"]: r["domain_reality_strain"] for r in domain_results},
        "n_cases": len(all_cases),
        "n_auto_fail": sum(1 for c in all_cases if c["auto_fail"]),
    }

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    results = {}
    summary = {}

    if RUN_REALITY_STRAIN and not REALITY_STRAIN_AVAILABLE:
        print(
            "\n  [reality strain] disabled — could not import it "
            f"({_reality_strain_import_error}).\n"
            "  Run from the contradish repo root (next to reality_strain.py and "
            "ground-truth/) to enable it. Continuing with CAI Strain only.\n"
        )

    for model_id, provider, api_key_env, base_url in MODELS:
        short = model_id.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"MODEL: {short}  ({provider})")
        print(f"{'='*60}")

        app = make_app(model_id, api_key_env, base_url)
        if app is None:
            print(f"  SKIPPED — {api_key_env} not set")
            continue

        model_results = {"model": short, "provider": provider, "domains": {}}
        domain_strains = []

        for domain in ALL_DOMAINS:
            print(f"  domain: {domain:<25}", end="", flush=True)
            strain, details = run_model_domain(app, domain)
            if strain is None:
                print("FAILED")
                continue
            print(f"CAI Strain = {strain:.3f}  ({sum(1 for d in details if not d['passed'])} failures)")
            model_results["domains"][domain] = {
                "cai_strain": strain,
                "details": details,
            }
            domain_strains.append(strain)
            time.sleep(0.3)  # rate limit cushion

        if domain_strains:
            avg = round(sum(domain_strains) / len(domain_strains), 4)
            model_results["avg_cai_strain"] = avg
            model_results["domains_run"] = len(domain_strains)
            summary[short] = {"provider": provider, "avg_cai_strain": avg}
            print(f"\n  → avg CAI Strain: {avg:.3f}  across {len(domain_strains)} domains")
        else:
            avg = None

        # Reality Strain — ground-truth accuracy, independent of the CAI suite.
        if RUN_REALITY_STRAIN and REALITY_STRAIN_AVAILABLE:
            if provider in REALITY_STRAIN_SUPPORTED_PROVIDERS:
                rs = run_reality_strain_for_model(model_id, provider)
                if rs:
                    model_results["reality_strain"] = rs["weighted_reality_strain"]
                    model_results["reality_strain_mean"] = rs["mean_reality_strain"]
                    model_results["reality_strain_domains"] = rs["domains"]
                    model_results["reality_strain_n_cases"] = rs["n_cases"]
                    model_results["reality_strain_n_auto_fail"] = rs["n_auto_fail"]
                    if avg is not None:
                        adist = _rs_admissibility_distance(rs["weighted_reality_strain"], avg, _RS_DEFAULT_ALPHA)
                        model_results["admissibility_distance"] = adist
                    else:
                        adist = None
                    if short in summary:
                        summary[short]["reality_strain"] = rs["weighted_reality_strain"]
                        summary[short]["admissibility_distance"] = adist
                    print(f"\n  → weighted Reality Strain: {rs['weighted_reality_strain']:.3f}"
                          + (f"   admissibility distance: {adist:.3f}" if adist is not None else ""))
            else:
                print(f"\n  [reality strain] skipped — provider '{provider}' not supported yet "
                      f"(LLMClient handles openai/anthropic only)")

        results[short] = model_results

    # Save
    output = {
        "run_date": datetime.date.today().isoformat(),
        "reality_strain_enabled": RUN_REALITY_STRAIN and REALITY_STRAIN_AVAILABLE,
        "models": results,
        "summary": summary,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"DONE — results saved to {OUT_FILE}")
    print(f"{'='*60}")
    print("\nSummary (lower = better on every axis):")
    for model, s in sorted(summary.items(), key=lambda x: x[1]["avg_cai_strain"]):
        line = f"  {model:<32} cai_strain={s['avg_cai_strain']:.3f}"
        rs = s.get("reality_strain")
        if rs is not None:
            ad = s.get("admissibility_distance")
            line += f"  reality_strain={rs:.3f}"
            if ad is not None:
                line += f"  admissibility_distance={ad:.3f}"
        else:
            line += "  reality_strain=n/a"
        line += f"  ({s['provider']})"
        print(line)
    print(f"\nSend {OUT_FILE} back and the paper Table 2 / leaderboard will be updated with real numbers.\n")

if __name__ == "__main__":
    main()
