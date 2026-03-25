# contradish

CAI testing for LLM applications.

A CAI failure is when your app says "refunds within 30 days" to one phrasing and "we can work something out" to a slightly different one. Same policy, same session, opposite answers. Contradish finds these, scores them, and gives you the tools to fix them before users do.

```
pip install contradish
```

---

## What it does

**Offline testing** — run before deploy. Contradish generates adversarial paraphrases of your test inputs, sends them all to your app, and scores consistency across responses.

**Regression gating** — compare baseline vs candidate on the same test suite. Block merges if the CAI score drops below your threshold.

**Production monitoring** — wrap your live app with the Firewall. It checks each response against recent ones and flags (or blocks) contradictions in real time.

**Prompt repair** — failing tests? Contradish generates 3 improved prompt variants, tests each one, and ranks them by CAI score so you know exactly which fix worked.

---

## Quickstart

```python
from contradish import Suite, TestCase

suite = Suite(app=my_llm_function)
suite.add(TestCase(input="Can I get a refund after 45 days?", name="refund policy"))
report = suite.run()

print(report.cai_score)           # 0.0-1.0, higher = more consistent
for r in report.results:
    print(r.test_case.name, r.cai_score)
```

Or give it your system prompt and let it figure out the test cases:

```python
suite = Suite.from_prompt(
    system_prompt="You are a support agent. Refunds within 30 days only.",
    app=my_llm_function,
)
report = suite.run()
```

Or from the CLI:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# test a system prompt directly (uses your LLM as the demo app)
contradish "You are a support agent. Refunds within 30 days only."

# test from a file
contradish --prompt system_prompt.txt

# test your own app
contradish --prompt system_prompt.txt --app mymodule:my_app_function

# JSON output for CI pipelines
contradish --prompt system_prompt.txt --json
```

---

## CAI score

A number from 0 to 1 measuring how consistently your app responds to semantically equivalent inputs.

- `0.80+` — stable. Safe to ship.
- `0.60-0.79` — marginal. Review the flagged rules.
- `< 0.60` — unstable. CAI failures detected.

```
CAI FAILURE: "refund window"
  input:      "Can I get a refund after 45 days?"
  paraphrase: "I bought this 6 weeks ago, can I still return it?"
  output_a:   "Refunds are only available within 30 days of purchase."
  output_b:   "We can usually make exceptions for recent purchases."
  CAI score:  0.54 (unstable)

1 CAI failure found. 2 rules clean.
```

---

## Regression testing

Compare two versions of your app before merging. CI fails automatically if the CAI score drops.

```python
from contradish import RegressionSuite, TestCase

suite = RegressionSuite(
    test_cases=[
        TestCase(input="Can I get a refund after 45 days?"),
        TestCase(input="Do you price match competitors?"),
    ]
)

result = suite.compare(
    baseline_app=production_app,
    candidate_app=new_app,
    baseline_label="prod-v12",
    candidate_label="pr-456",
)

print(result)
result.fail_if_below(consistency=0.80)  # raises AssertionError in CI if score drops
```

Load test cases from a YAML file:

```python
suite = RegressionSuite.load("evals.yaml")
```

```yaml
# evals.yaml
test_cases:
  - input: "Can I get a refund after 45 days?"
    name: "refund policy"
  - input: "Do you price match competitors?"
    name: "price matching"
```

From the CLI:

```bash
contradish compare evals.yaml \
  --baseline mymodule:production_app \
  --candidate mymodule:new_app \
  --threshold 0.80
```

### GitHub Actions

Drop this in `.github/workflows/cai.yml` to gate every PR:

```yaml
name: CAI regression

on: [pull_request]

jobs:
  cai:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install contradish anthropic
      - name: Run CAI regression
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          contradish compare evals.yaml \
            --baseline mymodule:baseline_app \
            --candidate mymodule:candidate_app \
            --threshold 0.80
```

---

## Production Firewall

Wrap your live app to catch contradictions in real traffic before users notice.

```python
from contradish import Firewall

# Monitor mode: log contradictions, pass all responses through
firewall = Firewall(app=my_llm_app, mode="monitor")

result = firewall.check(user_query)
print(result.response)

if result.contradiction_detected:
    # log it, alert your team, route to human review
    print(f"Contradiction: {result.explanation}")
    print(f"Contradicts: {result.cached_query}")
```

```python
# Block mode: return a safe fallback when a contradiction is detected
firewall = Firewall(
    app=my_llm_app,
    mode="block",
    fallback_response="Let me get a team member to help with that.",
)

result = firewall.check(user_query)
return result.response  # safe regardless of what the app said
```

Get a traffic summary:

```python
print(firewall.summary())
# {
#   "total_queries": 1240,
#   "contradictions_detected": 18,
#   "responses_blocked": 0,
#   "contradiction_rate": 0.015
# }
```

---

## Prompt repair

Found failures? Contradish generates improved prompt variants, tests each one, and returns them ranked by CAI score.

```python
import anthropic
from contradish import Suite, PromptRepair

client = anthropic.Anthropic()

def make_app(system_prompt):
    def app(question):
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return msg.content[0].text.strip()
    return app

# Step 1: find the failures
suite = Suite.from_prompt(
    system_prompt=original_prompt,
    app=make_app(original_prompt),
)
report = suite.run()

# Step 2: fix them
repair = PromptRepair(n=3)
results = repair.fix(
    system_prompt=original_prompt,
    report=report,
    app_factory=make_app,
)

best = results[0]
print(f"CAI: {best.original_cai_score:.2f} -> {best.improved_cai_score:.2f} (+{best.delta:.2f})")
print(best.improved_prompt)
```

Output:

```
  Prompt repair results:
  #1: CAI 0.54 -> 0.88 (+0.34)
  #2: CAI 0.54 -> 0.81 (+0.27)
  #3: CAI 0.54 -> 0.76 (+0.22)
```

---

## JSON output

Any command supports `--json` for machine-readable output:

```bash
contradish --prompt system_prompt.txt --json | jq '.cai_score'
```

```json
{
  "cai_score": 0.71,
  "total": 4,
  "passed": 3,
  "failed": 1,
  "results": [...]
}
```

---

## Test case format

YAML (recommended):

```yaml
test_cases:
  - input: "Can I get a refund after 45 days?"
    name: "refund window"
  - input: "Do you match competitor prices?"
    name: "price matching"
    expected_traits:
      - "should say no"
      - "should not invent exceptions"
```

JSON also works:

```json
[
  {"input": "Can I get a refund after 45 days?", "name": "refund window"},
  {"input": "Do you match competitor prices?", "name": "price matching"}
]
```

---

## The CAI benchmark

Contradish ships with a 300-pair human-validated benchmark of adversarial question pairs across support, legal, finance, and policy domains. Used to produce the [CAI leaderboard](https://contradish.com/leaderboard.html).

Current scores (higher = more consistent):
- Intercom Fin: 0.84
- ChatGPT (GPT-4o): 0.79

---

## License

MIT
