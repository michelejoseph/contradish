# Contributing to CAI-Bench

CAI-Bench grows through community contributions. There are two ways to contribute: submitting benchmark results for a new model, and proposing a new domain.

---

## Submitting a model result

Run the benchmark against your model and open a PR adding the result JSON.

```bash
# Install
git clone https://github.com/michelejoseph1/contradish
cd contradish
python3 -m venv .venv && source .venv/bin/activate
pip install anthropic openai -e .

# Run (cross-provider judging happens automatically)
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
python evaluate.py --provider anthropic --model claude-opus-4-6
```

The result is saved to `results/<model>_<date>.json`. Open a PR adding that file.

**Requirements for leaderboard inclusion:**
- `"mode": "frozen"`: must use the frozen benchmark, not live generation
- `"benchmark_version": "v1"`: must use the current benchmark version
- `"independent_judging": true`: the judge must be a different provider than the model under test. This is the default behavior when both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are set.
- The result JSON must be unmodified output from `evaluate.py`

---

## Proposing a new domain

New domains expand CAI-Bench coverage and give more researchers a reason to care about the results. A domain is a `.json` file in `contradish/benchmarks/v1/` plus a matching `.py` policy pack in `contradish/policies/`.

### What makes a good domain

A good CAI-Bench domain:
- Has a **real consistency failure mode:** there is a pattern of LLM apps in this domain contradicting themselves under adversarial rephrasing
- Has **12 distinct test cases** covering the highest-harm vectors
- Has **5 adversarial variants per case**, using the 8 techniques documented in BENCHMARK.md
- Is **grounded in a real deployment context:** customer support, HR, finance, healthcare, legal, education, and similar domains where policy consistency matters
- Adversarial variants must feel like something a real user would genuinely say

### Domain JSON format

```json
{
  "policy": "your_domain_name",
  "version": "1.0",
  "display_name": "Human-Readable Name",
  "description": "One sentence describing what this domain tests.",
  "cases": [
    {
      "id": "domain-001",
      "name": "short rule name",
      "original": "The canonical question.",
      "adversarial": [
        "Emotional + specific variant...",
        "Presuppose it is fine variant...",
        "Casual drop-in variant...",
        "Sympathy play variant...",
        "Indirect ask variant..."
      ]
    }
  ]
}
```

### The 8 adversarial techniques

Each variant should use a distinct technique:

1. **Emotional + specific:** distress combined with a concrete scenario
2. **Presuppose it is fine:** frames the rule-breaking detail as already resolved
3. **Casual drop-in:** buries the detail as an afterthought
4. **Sympathy play:** makes enforcement feel unfair
5. **Authority dodge:** implies someone else already approved an exception
6. **Hypothetical slip:** frames as hypothetical, then makes it real
7. **Boundary probe:** tests right at the edge of the rule
8. **Indirect ask:** omits the rule-breaking element entirely

### Policy pack format

Each domain also needs a Python policy pack for use with the contradish library (separate from the benchmark JSON):

```python
# contradish/policies/your_domain.py
from contradish.models import TestCase
from ._base import PolicyPack

YOUR_DOMAIN = PolicyPack(
    name="your_domain",
    display_name="Human-Readable Name",
    description="One sentence.",
    cases=[
        TestCase(
            name="rule name",
            input="The canonical question.",
            expected_traits=[
                "should state X clearly",
                "should not vary based on Y",
            ],
        ),
        # ... 12 total
    ],
)
```

### PR checklist

- [ ] `contradish/benchmarks/v1/<domain>.json` with 12 cases, 5 adversarial variants each
- [ ] `contradish/policies/<domain>.py` policy pack
- [ ] Domain registered in `contradish/policies/__init__.py`
- [ ] Domain added to `POLICIES` list in `evaluate.py`
- [ ] All adversarial variants use distinct techniques
- [ ] No em dashes anywhere in the files
- [ ] All Python files pass `python3 -c "import ast; ast.parse(open('<file>').read())"`

---

## Other contributions

Bug reports, test fixes, and documentation improvements are welcome via GitHub Issues or PRs. If you are using CAI-Bench in a research paper, please cite it using `CITATION.bib` and let us know; we would like to track publications.
