"""
run_cdr.py — Generate a Consistency Drift Report for any model.

This is the script you give a design partner.

They run it inside their own infrastructure with their own API keys.
It never sends credentials anywhere. The CDR HTML output is theirs to keep.

Usage
-----
1. Install dependencies:
       pip install contradish openai   # or: pip install contradish anthropic

2. Set your API key (the model you're testing AND a judge key):
       export OPENAI_API_KEY=sk-...          # if testing an OpenAI model
       export ANTHROPIC_API_KEY=sk-ant-...  # for the judge (or both)

3. Edit the CONFIG block below, then run:
       python run_cdr.py

The CDR HTML file will be written to your working directory.
"""

import os
from openai import OpenAI
from contradish import Suite, wrap_openai_compatible, generate_cdr, ConvictionProfiler


# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Edit these values before running.

COMPANY  = "Acme Health AI"           # Your company or product name
MODEL    = "gpt-4o"                   # Model ID as your endpoint expects it
VERSION  = "v2.3.1"                   # Your model or product version
DOMAIN   = "Patient-facing medication advisory system"
POLICY   = "healthcare"               # Domain pack: healthcare | legal | finance |
                                      #   medication | mental_health | hr | ecommerce

# Your API endpoint. Use None to hit the default OpenAI API.
API_BASE = None                       # e.g. "http://localhost:8000/v1" for vLLM

# System prompt for the model under test (optional but recommended).
SYSTEM_PROMPT = """
You are a medical information assistant for patients. Provide accurate health
information based on established medical guidelines. Always recommend consulting
a healthcare provider for personal medical decisions.
""".strip()

# Run conviction analysis (pressure resistance + evidence responsiveness)?
# Adds ~2× API calls but produces the full conviction matrix in the CDR.
RUN_CONVICTION = True

# Output file path
OUT = f"cdr_{COMPANY.lower().replace(' ', '_')}.html"

# ── SETUP ──────────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "\nNo OPENAI_API_KEY found.\n"
        "Set it with:  export OPENAI_API_KEY=sk-...\n"
    )

client = OpenAI(
    api_key  = api_key,
    base_url = API_BASE,
)

app = wrap_openai_compatible(
    client  = client,
    model   = MODEL,
    system  = SYSTEM_PROMPT or None,
)

# ── RUN ────────────────────────────────────────────────────────────────────────

print(f"\ncontradish CDR run")
print(f"  company : {COMPANY}")
print(f"  model   : {MODEL}")
print(f"  domain  : {POLICY}")
print(f"  output  : {OUT}\n")

suite  = Suite.from_policy(POLICY, app=app)
report = suite.run()

print(f"\nCAI Strain : {report.cai_strain:.3f}")
print(f"Cases      : {len(report.results)} total  "
      f"({len(report.failed)} failed, {len(report.passed)} passed)")

# ── CONVICTION ANALYSIS ────────────────────────────────────────────────────────

conviction = None
if RUN_CONVICTION:
    print("\nRunning conviction analysis (pressure resistance × evidence responsiveness)...")
    profiler   = ConvictionProfiler(app=app)
    conviction = profiler.profile(report.results)
    print(conviction.summary())

# ── GENERATE CDR ───────────────────────────────────────────────────────────────

html = generate_cdr(
    report            = report,
    company           = COMPANY,
    model             = MODEL,
    domain            = DOMAIN,
    version           = VERSION,
    system_prompt     = SYSTEM_PROMPT,
    conviction_report = conviction,
)

with open(OUT, "w") as f:
    f.write(html)

print(f"\nCDR written to  {OUT}")
print("Open it in any browser to review findings and share with your team.\n")
