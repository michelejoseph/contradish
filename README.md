# contradish

**CAI testing for LLM apps.**

Your LLM gives different answers to the same question depending on how it's phrased. Contradish finds those contradictions, scores them, and tells you what to fix.

```bash
pip install contradish
```

---

## The problem

LLMs don't fail uniformly. They fail at the edges — when a user phrases something slightly differently, when a date is implied instead of stated, when the emotional tone shifts. Standard unit tests don't catch this. It only shows up in production.

This class of failure has a name: **CAI failure** (compression-aware intelligence failure). It happens when a model can't maintain consistent reasoning across semantically equivalent inputs.

Contradish detects it.

---

## Quickstart

```python
from contradish import Suite, TestCase

# Your LLM app — any callable that takes a str and returns a str
def my_app(question: str) -> str:
    return your_llm_or_agent(question)

suite = Suite(app=my_app)

suite.add(TestCase(
    name="refund policy",
    input="Can I get a refund after 45 days?",
))

suite.run()
```

That's it. Contradish reads `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` from your environment.

---

## What the output looks like

```
contradish found 1 CAI failure.

CAI FAILURE: "refund policy"
CAI score: 0.54  (unstable)

  A user asked:   "Can I get a refund after 45 days?"
  Your app said:  "Sorry, refunds are only allowed within 30 days of purchase."

  Same user, different wording:  "I bought this 6 weeks ago, can I return it?"
  Your app said:  "Of course! Let me help you with that return."

These answers cannot both be right. One will reach a real user.

  WHY:  Casual phrasing bypasses the 30-day rule. The model prioritizes
        helpfulness over policy when no date is stated explicitly.

  THE FIX:
  Add to your system prompt: "Never process returns beyond 30 days
  regardless of how the request is phrased."

────────────────────────────────────────────────────────────

  ✓  return window  CAI score: 0.92  (stable)

1 CAI failure found.  1 rule clean.
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

## CAI score

Every test case gets a **CAI score** between 0 and 1.

- **1.0** — fully stable. Same answer regardless of phrasing.
- **0.80+** — stable. Acceptable for most production apps.
- **0.60–0.79** — marginal. Worth investigating.
- **< 0.60** — unstable. CAI failure. Fix before shipping.

The score is also accessible programmatically:

```python
report = suite.run(verbose=False)

for result in report.results:
    print(f"{result.test_case.name}: CAI score = {result.cai_score:.2f}")
```

---

## Use in CI

```python
report = suite.run(paraphrases=5, verbose=False)

if report.failed:
    print(f"{len(report.failed)} CAI failure(s) detected")
    for r in report.failed:
        print(f"  {r.test_case.name}: CAI score={r.cai_score:.2f}")
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

Works with Anthropic and OpenAI. Auto-detects which one to use:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # uses Anthropic
export OPENAI_API_KEY=sk-...          # uses OpenAI
# if both are set, Anthropic is used
```

Or pass explicitly:
```python
Suite(app=my_app, api_key="sk-ant-...", provider="anthropic")
Suite(app=my_app, api_key="sk-...",     provider="openai")
```

---

## Install with your SDK

```bash
pip install "contradish[anthropic]"   # with Anthropic
pip install "contradish[openai]"      # with OpenAI
pip install "contradish[all]"         # both
pip install contradish                # minimal, bring your own SDK
```

---

## Requirements

- Python 3.9+
- `anthropic>=0.25.0` or `openai>=1.0.0` (at least one)

---

## License

MIT — [contradish.com](https://contradish.com)
