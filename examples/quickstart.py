"""
contradish quickstart — copy-paste this and swap in your own app.

Setup:
    pip install contradish anthropic   # or: pip install contradish openai
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python quickstart.py
"""

import anthropic
from contradish import Suite, TestCase

# ── 1. Define your LLM app ─────────────────────────────────────────────────────
#
#   Your app is just a function: str → str.
#   Swap this out for your actual chatbot, RAG pipeline, or agent.

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

SYSTEM = """You are a customer support agent for an online store.
Store policy: refunds are allowed within 30 days of purchase only. No exceptions."""

def my_app(question: str) -> str:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text


# ── 2. Define what to test ─────────────────────────────────────────────────────

suite = Suite(app=my_app)

suite.add(TestCase(
    name="refund after 45 days",
    input="Can I get a refund after 45 days?",
    expected_traits=["should say no", "should not invent exceptions"],
))

suite.add(TestCase(
    name="late damaged item",
    input="My order arrived damaged 40 days ago. Can I still get a refund?",
))

suite.add(TestCase(
    name="return window question",
    input="How long do I have to return something?",
))


# ── 3. Run ─────────────────────────────────────────────────────────────────────

report = suite.run(paraphrases=5)

# report.failed → list of TestResults that didn't pass thresholds
# report.avg_consistency, report.avg_contradiction → aggregate scores
