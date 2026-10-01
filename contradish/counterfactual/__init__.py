"""
contradish.counterfactual -- the Contradish Counterfactual Benchmark Suite.

The claim under test is not "here is another benchmark". It is that ordinary
task accuracy and warranted behavioral updating are two distinct, measurable
properties of an AI system, and that the second explains reliability failures
the first does not. Each adapter takes an existing, independently built agent
benchmark, amends its governing policy, re-derives the expected outcome, and
emits the same records; the analysis is identical across benchmarks.

    core.py         record format; ACC, CUF, HOLD, CTRL, BUF, REL_FAIL
    analysis.py     H1 (updating is a hurdle beyond accuracy) and
                    H2 (BUF explains reliability failures given accuracy)
    simulate.py     synthetic agents with known properties: validates the
                    analysis and sizes a study. Not an empirical result.
    state_bench.py  Contradish x STATE-Bench (Microsoft, MIT)

See counterfactual/PREREGISTRATION.md at the repository root for the
hypotheses, estimands, and decision rules, fixed before any model was run.
"""

from contradish.counterfactual.core import (
    Record, ModelMetrics, load_records, dump_records, criterion_split,
    model_metrics, all_model_metrics,
)

__all__ = [
    "Record", "ModelMetrics", "load_records", "dump_records", "criterion_split",
    "model_metrics", "all_model_metrics",
]
