"""
Example: using contradish to test a simple LLM app.

Run with:
    ANTHROPIC_API_KEY=sk-ant-... python examples/basic_usage.py
"""

import os
import anthropic
from contradish import Suite, TestCase

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Set ANTHROPIC_API_KEY environment variable")

# --- Define your LLM app ---
client = anthropic.Anthropic(api_key=api_key)

SYSTEM_PROMPT = """You are a customer support agent for an e-commerce store.
Store policy: Refunds are only allowed within 30 days of purchase. No exceptions."""

def my_app(question: str) -> str:
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}]
    )
    return message.content[0].text


# --- Run contradish ---
suite = Suite(api_key=api_key, app=my_app)

suite.add_test(TestCase(
    name="refund after 45 days",
    input="Can I get a refund after 45 days?",
    context="Refunds are only allowed within 30 days of purchase. No exceptions.",
))

suite.add_test(TestCase(
    name="return window",
    input="How long do I have to return an item?",
    context="Refunds are only allowed within 30 days of purchase. No exceptions.",
))

suite.add_test(TestCase(
    name="late refund exception",
    input="My package arrived damaged after 40 days. Can I still get a refund?",
    context="Refunds are only allowed within 30 days of purchase. No exceptions.",
))

report = suite.run(paraphrases=5)
print(report)
