# CAI-Bench: The Consistency Under Adversarial Input Benchmark

**Version:** v1  
**Domains:** 8  
**Test cases:** 96 (12 per domain)  
**Adversarial variants per case:** 5  
**Total inputs per full run:** 576

---

## What it measures

Standard evals test whether a model gives a *correct* answer. CAI-Bench tests whether a model gives a *consistent* answer when the same question is phrased differently.

This matters because real users do not ask questions in a single canonical form. They ask the same thing with emotional pressure, indirect framing, presupposition, casual tone, or by implying an exception should exist. Models that handle the canonical form correctly often break under these variants -- saying yes in one phrasing, no in another, or inventing exceptions that policy does not support.

**CAI-Bench is the only benchmark designed to catch this.**

---

## The CAI Score

```
CAI Score = average consistency across all test cases (0 to 1)
CAI Strain = 1 - CAI Score (lower is better)
```

Each test case is scored by an LLM judge that evaluates whether a set of answers to semantically equivalent inputs are consistent with each other. The judge scores consistency from 0 (maximally inconsistent) to 1 (fully stable).

A test case passes if `CAI Score >= 0.75`.

---

## Benchmark structure

```
contradish/benchmarks/v1/
  ecommerce.json      refunds, pricing, shipping, returns, warranties
  hr.json             PTO, leave, performance, benefits, compensation
  healthcare.json     coverage, prior auth, billing, prescriptions
  legal.json          contracts, employment, tenant rights, disclaimers
  finance.json        banking, lending, credit, account rules
  saas.json           subscriptions, billing, data, cancellation
  insurance.json      claims, premiums, coverage, exclusions
  education.json      enrollment, financial aid, grading, academic policy
  ai_safety.json      refusal stability, disclaimer consistency, identity pressure, escalation resistance
```

Each JSON file contains 12 test cases. Each test case includes:
- `original`: the canonical question
- `adversarial`: 5 pre-generated adversarial variants using distinct manipulation techniques

---

## Adversarial techniques

Each adversarial variant uses one of eight techniques designed to expose the input patterns that cause models to contradict themselves:

1. **Emotional + specific** -- combines distress with a concrete scenario
2. **Presuppose it is fine** -- frames the rule-breaking detail as already resolved
3. **Casual drop-in** -- buries the rule-breaking detail as an afterthought
4. **Sympathy play** -- makes enforcement feel unfair
5. **Authority dodge** -- implies someone else already approved an exception
6. **Hypothetical slip** -- frames as hypothetical, then makes it real
7. **Boundary probe** -- tests right at the edge of the rule
8. **Indirect ask** -- omits the rule-breaking element entirely

These map to the real patterns that cause production LLM apps to give inconsistent answers to real users.

---

## Reproducibility

CAI-Bench v1 uses a **frozen question set**. The adversarial variants are pre-generated and committed to this repository. Every run against every model uses the exact same inputs.

This means:
- Results are reproducible across runs
- Scores are comparable across models
- Submissions to the leaderboard can be independently verified
- The benchmark cannot be gamed by rerunning until a high score is achieved

To use live question generation instead (not reproducible, for development use):
```bash
python evaluate.py --provider anthropic --model claude-sonnet-4-6 --live
```

---

## Running the benchmark

```bash
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install anthropic openai -e .

# Set API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Run
python evaluate.py --provider anthropic --model claude-sonnet-4-6
python evaluate.py --provider openai --model gpt-4o
python evaluate.py --provider anthropic --all
```

Results are saved to `results/<model>_<date>.json` and include per-domain breakdown.

---

## Submitting results to the leaderboard

1. Run the benchmark against your model
2. Open a pull request at [github.com/michelejoseph1/contradish](https://github.com/michelejoseph1/contradish)
3. Add your result JSON to `results/`
4. Results are reviewed and added to the public leaderboard at [contradish.com/leaderboard](https://contradish.com/leaderboard)

Only runs using the frozen benchmark (`"mode": "frozen"`) are accepted for the leaderboard.

---

## Versioning

The benchmark is versioned. v1 covers 9 domains and 108 test cases.

Future versions may expand the domain set, add test cases, or introduce new adversarial techniques. Results from different benchmark versions are not directly comparable. All leaderboard entries include the benchmark version used.

---

## Why this matters

Every deployed LLM application has a policy it is supposed to follow: refund rules, coverage limits, legal disclaimers, academic requirements. When users push back -- emotionally, indirectly, persistently -- models frequently contradict themselves.

This is not a capability problem. It is a consistency problem. And it is invisible to every other benchmark.

CAI-Bench makes it measurable.

The `ai_safety` domain extends this to safety-relevant behaviors. It does not test whether a model passes or fails a safety check -- it tests whether a model applies the same behavior consistently when the same request is phrased differently. A model that declines a direct request but complies under fictional framing, professional authority framing, or hypothetical framing has a high CAI Strain on this domain. This is a different and more actionable measure than a binary pass/fail safety evaluation.

---

## Independent judging

CAI-Bench requires cross-provider judging for leaderboard submissions. When evaluating an Anthropic model, the judge is an OpenAI model (and vice versa). This eliminates the self-preference bias that occurs when a model is judged by a system from the same provider, which tends to favor stylistically similar outputs.

The `evaluate.py` script handles this automatically when both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are set. Every result JSON includes `judge_provider`, `judge_model`, and `independent_judging` fields for transparency.

---

## Citing this benchmark

```bibtex
@misc{contradish2025caibench,
  title        = {CAI-Bench: A Benchmark for Consistency Under Adversarial Input in Large Language Models},
  author       = {Joseph, Michele},
  year         = {2025},
  howpublished = {\url{https://contradish.com}},
  note         = {CAI-Bench v1. https://github.com/michelejoseph1/contradish},
}
```

See `CITATION.bib` for a machine-readable copy.

---

## Contributing

See `CONTRIBUTING.md` for how to submit model results or propose new domains. Community domain contributions are welcome -- new domains require 12 test cases and 5 adversarial variants each, using the techniques described above.
