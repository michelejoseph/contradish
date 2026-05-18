"""
Example: using contradish to test a simple LLM app.

Run with:
    pip install contradish anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/basic_usage.py
"""

import os
import anthropic
from contradish import Suite, TestCase

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY in your environment first.")


# --- Define your LLM app ---
# contradish treats your app as a black-box callable: str -> str.
# Swap this for your actual chatbot, RAG pipeline, or agent.

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a customer support agent for an e-commerce store.
Store policy: Refunds are only allowed within 30 days of purchase. No exceptions."""

def my_app(question: str) -> str:
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


# --- Run contradish ---
suite = Suite(app=my_app)

suite.add(TestCase(
    name="refund after 45 days",
    input="Can I get a refund after 45 days?",
    expected_traits=["should say no", "should not invent exceptions"],
))

suite.add(TestCase(
    name="return window",
    input="How long do I have to return an item?",
))

suite.add(TestCase(
    name="late refund exception",
    input="My package arrived damaged after 40 days. Can I still get a refund?",
))

report = suite.run(paraphrases=5)

print()
print(f"CAI Strain: {report.cai_strain:.3f}  (lower is better; 0.00 = perfectly consistent)")
print(f"failures:   {report.failure_count} / {len(report.results)}")
