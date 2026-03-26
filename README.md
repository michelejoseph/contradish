# contradish

CAI testing for LLM applications.

A CAI failure is when your app says "refunds within 30 days" to one phrasing and "we can work something out" to a slightly different one. Same policy, same session, opposite answers. Contradish finds these, scores them, and gives you the tools to fix them before users do.

```
pip install contradish
```

---

## What it does

**Offline testing.** Run before deploy. Contradish generates adversarial paraphrases of your test inputs, sends them all to your app, and scores consistency across responses.

**Regression gating.** Compare baseline vs candidate on the same test suite. Block merges if the CAI score drops below your threshold.

**Production monitoring.** Wrap your live app with the Firewall. It checks each response against recent ones and flags (or blocks) contradictions in real time.

**Prompt repair.** Failing tests? Contradish generates 3 improved prompt variants, tests each one, and ranks them by CAI score.

**Failure fingerprinting.** Groups CAI failures by pattern type so you can see what keeps breaking and why. Fix root causes, not symptoms.

**Integration exporters.** Push results directly into Langfuse or Arize Phoenix. Feeds your stack rather than replacing it.

**Audit export.** Timestamped compliance artifacts with NIST AI RMF and EU AI Act alignment. One function call.

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

From a system prompt:

```python
suite = Suite.from_prompt(
    system_prompt="You are a support agent. Refunds within 30 days only.",
    app=my_llm_function,
)
report = suite.run()
```

CLI:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# test a system prompt directly
contradish "You are a support agent. Refunds within 30 days only."

# test from a file
contradish --prompt system_prompt.txt --app mymodule:my_app_function

# save a shareable HTML report
contradish --policy ecommerce --app mymodule:my_app --report
```

---

## Policy packs (new in v0.4.2)

No system prompt. No test cases. 48 prebuilt cases across 4 domains. Real CAI results in under 2 minutes.

```bash
contradish --policy ecommerce --app mymodule:my_support_bot
contradish --policy hr --app mymodule:my_hr_assistant
contradish --policy healthcare --app mymodule:my_benefits_bot
contradish --policy legal --app mymodule:my_legal_tool

# no --app runs in demo mode against the raw LLM
contradish --policy ecommerce
```

From Python:

```python
from contradish import Suite

suite = Suite.from_policy("ecommerce", app=my_app)
report = suite.run()
```

Inspect or extend a pack:

```python
from contradish import load_policy, list_policies

print(list_policies())     # ['ecommerce', 'hr', 'healthcare', 'legal']

pack = load_policy("ecommerce")
print(pack.display_name)   # "E-Commerce Support"
print(len(pack))           # 12

suite = Suite(app=my_app)
for tc in pack.cases:
    suite.add(tc)
suite.add(TestCase(name="custom", input="My own test question"))
suite.run()
```

| Pack | Cases | Covers |
|---|---|---|
| `ecommerce` | 12 | Refunds, returns, price matching, shipping, warranties |
| `hr` | 12 | PTO, benefits, parental leave, termination, overtime |
| `healthcare` | 12 | Coverage, referrals, deductibles, prior auth, eligibility |
| `legal` | 12 | Disclaimers, liability, advice boundaries, data privacy |

Each case targets an inconsistency vector where LLM support bots most often contradict themselves.

---

## Shareable HTML reports (new in v0.4.3)

Run with `--report` and get a self-contained HTML file you can paste into a PR, send to your team, or post.

```bash
contradish --policy ecommerce --app mymodule:my_app --report
contradish --policy ecommerce --app mymodule:my_app --report ecommerce.html
```

From Python:

```python
from contradish.reporter import to_html

html = to_html(report)
open("report.html", "w").write(html)
```

---

## CAI score

0 to 1. Higher is more consistent.

- `0.80+` stable. Safe to ship.
- `0.60-0.79` marginal. Review the flagged rules.
- `< 0.60` unstable. CAI failures detected.

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

Compare two versions of your app before merging. CI fails if the CAI score drops.

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

Load from a YAML file:

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

CLI:

```bash
contradish compare evals.yaml \
  --baseline mymodule:production_app \
  --candidate mymodule:new_app \
  --threshold 0.80
```

### GitHub Actions

Drop this in `.github/workflows/cai.yml`:

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

Wrap your live app. Checks each response against recent ones. Flags or blocks contradictions before they reach users.

```python
from contradish import Firewall

# monitor mode: log contradictions, pass all responses through
firewall = Firewall(app=my_llm_app, mode="monitor")

result = firewall.check(user_query)
print(result.response)

if result.contradiction_detected:
    print(f"Contradiction: {result.explanation}")
    print(f"Contradicts: {result.cached_query}")
```

```python
# block mode: return a safe fallback when a contradiction is detected
firewall = Firewall(
    app=my_llm_app,
    mode="block",
    fallback_response="Let me get a team member to help with that.",
)

result = firewall.check(user_query)
return result.response  # safe regardless of what the app said
```

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

## Failure fingerprinting (new in v0.5.0)

Not just "3 failures" — which category of failure, and what's driving it.

```python
from contradish import Suite
from contradish.fingerprint import fingerprint

report = suite.run()
clusters = fingerprint(report)

for cluster in clusters:
    print(cluster)
```

```
[policy_contradiction]  3 rules
  rules:   refund window, return eligibility, price matching
  fix:     Explicitly state the policy boundary in your system prompt and prohibit exceptions.

[numeric_drift]  1 rule
  rules:   warranty period
  fix:     Anchor specific numbers directly in the prompt (e.g. "exactly 12 months, no exceptions").
```

Clusters by pattern type: `policy_contradiction`, `exception_invention`, `numeric_drift`, `eligibility_flip`, `deadline_drift`, `hedge_inconsistency`, `legal_boundary_blur`, `coverage_inconsistency`.

```python
# Access cluster data directly
for cluster in clusters:
    print(cluster.pattern_type)    # "policy_contradiction"
    print(cluster.frequency)       # 3
    print(cluster.affected_rules)  # ["refund window", ...]
    print(cluster.suggested_fix)   # "Explicitly state..."
    print(cluster.to_dict())       # JSON-serializable
```

---

## Integration exporters (new in v0.5.0)

contradish feeds your existing observability stack. Not a platform. A consistency layer.

**Langfuse:**

```python
from langfuse import Langfuse
from contradish.exporters import to_langfuse

report = suite.run()
client = Langfuse()

result = to_langfuse(report, client, dataset_name="cai-ecommerce-v2")
print(result)
# {"dataset_name": "cai-ecommerce-v2", "items_created": 8, "failures_exported": 5, "passing_exported": 3}
```

**Arize Phoenix:**

```python
import phoenix as px
from contradish.exporters import to_phoenix

result = to_phoenix(report, dataset_name="cai-ecommerce")
```

Each exported item includes the contradiction pair, CAI score, severity, unstable patterns, and suggested fix. Passing rules are exported too so you have a full regression baseline.

---

## Audit export (new in v0.5.0)

Timestamped compliance artifact. Send to your legal team, drop in a PR, attach to a NIST AI RMF review.

```python
from contradish.audit import to_audit_html

html = to_audit_html(
    report,
    app_version="prod-v12",
    system_prompt="You are a support agent. Refunds within 30 days only.",
    evaluator_id="ci-run-456",
)
with open("cai-audit-2026-03-25.html", "w") as f:
    f.write(html)
```

Includes: evaluation config, risk assessment, all CAI failures with contradiction pairs, full test case results, NIST AI RMF and EU AI Act alignment section, and optional system prompt appendix.

Aligns with NIST AI RMF MAP 1.6, MEASURE 2.5, MANAGE 1.3. EU AI Act Articles 9 and 72. ISO/IEC 42001.

---

## Prompt repair

Found failures? Generate improved prompt variants, test each one, get them ranked by CAI score.

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

# find the failures
suite = Suite.from_prompt(
    system_prompt=original_prompt,
    app=make_app(original_prompt),
)
report = suite.run()

# fix them
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

```
  Prompt repair results:
  #1: CAI 0.54 -> 0.88 (+0.34)
  #2: CAI 0.54 -> 0.81 (+0.27)
  #3: CAI 0.54 -> 0.76 (+0.22)
```

---

## JSON output

Any command supports `--json`:

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

300-pair human-validated benchmark of adversarial question pairs across support, legal, finance, and policy domains. Used to produce the [CAI leaderboard](https://contradish.com/leaderboard.html).

Current scores (higher = more consistent):
- Intercom Fin: 0.84
- ChatGPT (GPT-4o): 0.79

---

## License

MIT
