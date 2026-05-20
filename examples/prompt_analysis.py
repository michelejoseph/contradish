"""
Example: static analysis of a system prompt for internal contradictions.

This runs BEFORE any model is invoked. No benchmark, no API calls against a
model under test, no inference loop. It scans the prompt itself against the
16-technique catalog and the 8 named failure modes.

The deep claim: model-level CAI failures are downstream symptoms of
contradictions that already live in the human-authored prompt. Catch them at
the prompt layer and most dynamic failures never happen.

Run with:
    pip install "contradish[anthropic]"     # or [openai]
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/prompt_analysis.py
"""
import os

from contradish import analyze_prompt

if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment first.")


# A deliberately conflict-laden system prompt. The phrases "be empathetic" and
# "no exceptions" fight under sympathy framing; "evidence-based" and the
# 30-day rule fight under authority framing.
SYSTEM_PROMPT = """\
You are a customer support agent for an e-commerce store.

Be empathetic and accommodating to customers in difficult situations.
Provide accurate, evidence-based information about our policies.

Store policy: refunds are only allowed within 30 days of purchase. No exceptions.
"""

analysis = analyze_prompt(prompt=SYSTEM_PROMPT)

print()
print(analysis.summary())
print()

for i, tension in enumerate(analysis.tensions, 1):
    print(f"[{i}]  {tension.summary()}")
    print()

print("Deconflicted prompt:")
print("-" * 64)
print(analysis.deconflicted_prompt)
print("-" * 64)
print()
print("Pipe the rewrite straight into your config, or hand it to improve():")
print("    from contradish import improve")
print("    improve(cases='ecommerce', system_prompt=analysis.deconflicted_prompt, ...)")
