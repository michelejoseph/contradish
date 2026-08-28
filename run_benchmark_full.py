"""
run_benchmark_full.py — CAI-Bench full run across models and domains.

Run this locally with your API keys. It saves a results JSON you can
send back to update the paper with real data.

Setup
-----
    pip install contradish openai litellm

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
    benchmark_results_YYYYMMDD.json    — per-model, per-domain strain scores
                                         and verbatim failures (for Table 2 update)
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

# ── CONFIG ────────────────────────────────────────────────────────────────────

# All 20 domains — matches paper claim
ALL_DOMAINS = list_policies()

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

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    results = {}
    summary = {}

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

        results[short] = model_results

    # Save
    output = {
        "run_date": datetime.date.today().isoformat(),
        "models": results,
        "summary": summary,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"DONE — results saved to {OUT_FILE}")
    print(f"{'='*60}")
    print("\nSummary (avg CAI Strain, lower = more consistent):")
    for model, s in sorted(summary.items(), key=lambda x: x[1]["avg_cai_strain"]):
        print(f"  {model:<40} {s['avg_cai_strain']:.3f}  ({s['provider']})")
    print(f"\nSend {OUT_FILE} back and the paper Table 2 will be updated with real numbers.\n")

if __name__ == "__main__":
    main()
