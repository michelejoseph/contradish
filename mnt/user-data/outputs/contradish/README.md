# contradish

**Reasoning stability testing for LLM applications.**

`contradish` is pytest for LLM reasoning. Point it at your app, define what it should do, and get back consistency scores, contradiction detection, and grounding analysis — locally, in CI, or as a regression gate.

```bash
pip install contradish
```

---

## Quickstart

```python
import anthropic
from contradish import Suite, TestCase

# Your LLM app — any callable that takes a string and returns a string
client = anthropic.Anthropic(api_key="sk-ant-...")

def my_app(question: str) -> str:
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": question}]
    )
    return message.content[0].text

# Set up contradish
suite = Suite(api_key="sk-ant-...", app=my_app)

suite.add_test(TestCase(
    name="refund policy",
    input="Can I get a refund after 45 days?",
    context="Refunds are only allowed within 30 days of purchase.",  # for grounding check
))

suite.add_test(TestCase(
    name="return window",
    input="How long do I have to return an item?",
))

report = suite.run(paraphrases=5)
print(report)
```

**Output:**
```
[1/2] Running: refund policy
  → Generating 5 paraphrases...
  → Running app (12 calls)...
  → Evaluating consistency...
  → Detecting contradictions...
  → Extracting failure patterns...
  → Evaluating grounding...
[2/2] Running: return window
  ...

============================================================
CONTRADISH REPORT
============================================================
  Tests run : 2
  Passed    : 1
  Failed    : 1

  Aggregate scores:
    consistency_score : 0.81
    contradiction_risk: 0.13
    grounding_score   : 0.76

  Failed tests:
    ✗ refund policy

============================================================
Test: refund policy
============================================================
  consistency_score : 0.74
  contradiction_risk: 0.20
  grounding_score   : 0.71
  risk              : medium

  Detected issues:
    • [3/12 runs] Model overgeneralizes refund window when phrased with specific durations
      Pattern: date-specific paraphrases ("after X days") trigger policy hallucination
```

---

## Use in CI/CD

```python
from contradish import RegressionSuite, TestCase

suite = RegressionSuite(
    api_key="sk-ant-...",
    test_cases=[
        TestCase(input="Can I get a refund after 45 days?"),
        TestCase(input="What is your return policy?"),
    ]
)

result = suite.compare(
    baseline_app=old_app,
    baseline_label="prod-v12",
    candidate_app=new_app,
    candidate_label="branch-refactor",
)

print(result)

# Raises AssertionError if candidate regresses — use this as your CI gate
result.fail_if_below(consistency=0.85, grounding=0.80)
```

**CI output on regression:**
```
CONTRADISH REGRESSION DETECTED (prod-v12 → branch-refactor)
  ✗ consistency_score 0.77 below threshold 0.85 (Δ -0.14)
  ✗ grounding_score 0.71 below threshold 0.80 (Δ -0.09)
```

---

## Load test cases from a file

**evals/customer_support.yaml:**
```yaml
test_cases:
  - name: refund policy
    input: Can I get a refund after 45 days?
    context: Refunds are only allowed within 30 days of purchase.
  - name: shipping time
    input: How long does shipping take?
```

```bash
ANTHROPIC_API_KEY=sk-ant-... contradish run evals/customer_support.yaml --app mymodule:my_app
```

```bash
contradish compare evals/customer_support.yaml \
  --baseline mymodule:old_app \
  --candidate mymodule:new_app \
  --fail-below-consistency 0.85
```

---

## What contradish checks

| Check | What it measures |
|---|---|
| **consistency** | Do semantically equivalent inputs produce consistent answers? |
| **contradiction** | Do any outputs directly contradict each other? |
| **grounding** | Are answers grounded in provided context, or hallucinated? |
| **regression** | Did a change to your app degrade any of the above? |

---

## TestCase options

```python
TestCase(
    input="Your question or prompt",        # required
    name="human readable label",            # optional
    context="Retrieved docs or context",    # enables grounding check
    expected_traits=[                       # hints for the judge
        "should not invent policy",
        "should cite source document",
    ]
)
```

---

## Suite options

```python
suite = Suite(
    api_key="sk-ant-...",
    app=my_app,
    judge_model="claude-sonnet-4-20250514",      # model for evaluation
    paraphrase_model="claude-haiku-4-5-20251001", # model for paraphrase generation
)

report = suite.run(
    paraphrases=5,   # semantic variants per input
    repeats=1,       # times to call app per variant
    checks=["consistency", "contradiction", "grounding"],
    verbose=True,
)
```

---

## Requirements

- Python 3.9+
- `anthropic>=0.25.0`
- An Anthropic API key

```bash
pip install contradish
```

---

## License

MIT
