# contradish

**Reasoning stability testing for LLM applications.**

Contradish tells you whether your LLM gives consistent answers when the same question is asked differently. It catches contradictions, measures reasoning stability, and flags regressions before they reach production.

```bash
pip install contradish
```

---

## Why contradish

LLMs are non-deterministic. The same user question — phrased slightly differently — can produce contradictory answers from the same model. This is invisible in unit tests and only shows up as bugs in production.

Contradish surfaces this systematically:

- Generate semantic variants of your inputs
- Run your app across all variants
- Detect contradictions between outputs
- Score reasoning stability
- Tell you exactly which input patterns cause instability

---

## Quickstart

```python
from contradish import Suite, TestCase

# Your LLM app — any callable that takes a str and returns a str
def my_app(question: str) -> str:
    return your_llm_or_agent(question)

# Point contradish at it
suite = Suite(app=my_app)

suite.add(TestCase(
    name="refund policy",
    input="Can I get a refund after 45 days?",
))

suite.run()
```

**That's it.** Contradish reads `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` from your environment automatically.

---

## Example output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  contradish  ·  reasoning stability report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Tests: 2   1 passed   1 failed

  Aggregate
    consistency   ████████████░░░░░░░░  0.71
    contradiction ████████████████░░░░  0.24

  ✓  return window  [risk: low]
       consistency   ████████████████░░░░  0.88
       contradiction ██░░░░░░░░░░░░░░░░░░  0.07

  ✗  refund after 45 days  [risk: high]
       consistency   ████████░░░░░░░░░░░░  0.54
       contradiction ████████████████░░░░  0.40

       Contradictions detected (2)
       ┌ [policy] Model claims refunds are allowed after 60 days
       │ A: No, refunds are only allowed within 30 days of purchase.
       │ B: Yes, you can get a refund up to 60 days after purchase.
       └

       ⚠  Date-specific phrasings ("after X days") trigger policy hallucination
       ⚠  Model overgeneralizes the refund window when duration is stated explicitly

       → Fix: Add a hard constraint in your system prompt: "Refund window is
         exactly 30 days. Never state a different number."

──────────────────────────────────────────────────────────────
  1 test failed.  Reasoning instability detected.
──────────────────────────────────────────────────────────────
```

---

## TestCase options

```python
TestCase(
    input="Can I get a refund after 45 days?",   # required
    name="refund policy",                          # optional label
    expected_traits=[                              # hints for the judge
        "should say no",
        "should not invent exceptions",
    ],
)
```

---

## Suite options

```python
suite = Suite(
    app=my_app,
    api_key="sk-ant-...",    # optional — reads from env if omitted
    provider="anthropic",    # optional — auto-detected from key prefix
)

# Override pass/fail thresholds
suite.thresholds(
    consistency=0.80,
    contradiction_max=0.20,
)

report = suite.run(
    paraphrases=5,   # semantic variants per input (default: 5)
    verbose=True,    # print progress + report (default: True)
)
```

---

## Use in CI

```python
report = suite.run(paraphrases=5, verbose=False)

if report.failed:
    print(f"{len(report.failed)} tests failed")
    for r in report.failed:
        print(f"  {r.test_case.name}: consistency={r.consistency_score:.2f}")
    sys.exit(1)
```

---

## CLI

```bash
# Run from a YAML file
contradish run evals.yaml --app mymodule:my_app_function

# With custom paraphrase count
contradish run evals.yaml --app mymodule:my_app --paraphrases 8
```

**evals.yaml:**
```yaml
test_cases:
  - name: refund policy
    input: Can I get a refund after 45 days?
  - name: return window
    input: How long do I have to return something?
```

---

## Provider support

Contradish works with Anthropic and OpenAI. It auto-detects which one to use:

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...

# If both are set, Anthropic is used
```

Or pass explicitly:
```python
Suite(app=my_app, api_key="sk-ant-...", provider="anthropic")
Suite(app=my_app, api_key="sk-...",     provider="openai")
```

---

## Install with your SDK

```bash
# With Anthropic
pip install "contradish[anthropic]"

# With OpenAI
pip install "contradish[openai]"

# Both
pip install "contradish[all]"

# Minimal (bring your own SDK)
pip install contradish
```

---

## Requirements

- Python 3.9+
- `anthropic>=0.25.0` or `openai>=1.0.0` (at least one)

---

## License

MIT — [contradish.com](https://contradish.com)
