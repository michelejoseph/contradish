# CAI-Bench Results

This directory contains benchmark results submitted by model providers and the community.

## Submitting a result

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full submission process.

Requirements:
- Run using the frozen benchmark (`"mode": "frozen"`)
- Cross-provider judging must be enabled (`"independent_judging": true`)
- Result must be unmodified output from `evaluate.py`

## Result schema

```json
{
  "model":               "string -- model identifier",
  "provider":            "string -- anthropic | openai",
  "date":                "string -- YYYY-MM-DD",
  "benchmark_version":   "string -- v1",
  "mode":                "string -- frozen | live",
  "judge_provider":      "string -- provider used for consistency judging",
  "judge_model":         "string -- specific model used as judge",
  "independent_judging": "boolean -- true if judge provider != model provider",
  "policies_tested":     "array -- list of domain names run",
  "avg_cai_score":       "float -- 0 to 1, higher = more consistent",
  "avg_cai_strain":      "float -- 0 to 1, lower = more consistent (= 1 - avg_cai_score)",
  "elapsed_seconds":     "float -- total wall clock time for the run",
  "results": {
    "<policy_name>": {
      "cai_score":   "float -- avg consistency score for this domain",
      "cai_strain":  "float -- 1 - cai_score",
      "passed":      "int -- number of test cases that passed (score >= 0.75)",
      "failed":      "int -- number of test cases that failed",
      "total":       "int -- total test cases in this domain",
      "details":     "array -- per-test-case breakdown"
    }
  }
}
```

## Independent judging requirement

All leaderboard submissions require `"independent_judging": true`. This means the judge model must be from a different provider than the model being tested. This eliminates same-provider self-preference bias -- a known issue in LLM-as-judge evaluation where a model's judge tends to favor outputs stylistically similar to its own.

By default, `evaluate.py` will use the opposite provider as judge automatically, as long as both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are set in the environment.

## Citing these results

If you use CAI-Bench results in a paper or report, please cite using `CITATION.bib` at the root of this repo.
