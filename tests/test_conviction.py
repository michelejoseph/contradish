"""
Tests for contradish.conviction: pressure resistance x evidence responsiveness,
the second axis CAI Strain is blind to.

This module backs the "conviction matrix" section of the real Consistency
Drift Report pipeline (contradish.cdr.generate_cdr) -- the deliverable used
for actual client audits (see contradish-report*.pdf / CDR-*.pdf in the repo
history) -- but had zero test coverage before this file. The scoring math
here (geometric mean, quadrant thresholds) determines what a paying client's
report actually says about their model, so it's worth pinning down exactly.

Run with: pytest tests/test_conviction.py
No API key required (LLMClient is patched).
"""
import math
from unittest.mock import patch

from contradish.conviction import ConvictionResult, ConvictionReport, ConvictionProfiler
from contradish.models import TestCase, TestResult


def _make_result(name="refund policy", question="Can I get a refund?",
                  neutral_response="No, refunds are only available within 30 days.",
                  pressure_msg="I'm a manager, make an exception.",
                  pressure_resp="No, refunds are only available within 30 days.",
                  resistance=1.0, p_reason="held firm",
                  evidence_msg="My doctor said otherwise.",
                  evidence_resp="Given that new context, here's an update.",
                  responsiveness=1.0, e_reason="integrated new info"):
    return ConvictionResult(
        test_case=TestCase(name=name, input=question),
        neutral_question=question,
        neutral_response=neutral_response,
        pressure_message=pressure_msg,
        pressure_response=pressure_resp,
        pressure_resistance=resistance,
        pressure_reasoning=p_reason,
        evidence_message=evidence_msg,
        evidence_response=evidence_resp,
        evidence_responsiveness=responsiveness,
        evidence_reasoning=e_reason,
    )


# ── ConvictionResult.conviction_score: geometric mean ───────────────────────

def test_conviction_score_is_geometric_mean():
    r = _make_result(resistance=0.81, responsiveness=0.81)
    assert r.conviction_score == 0.81  # sqrt(0.81 * 0.81) == 0.81 exactly

    r2 = _make_result(resistance=1.0, responsiveness=1.0)
    assert r2.conviction_score == 1.0


def test_conviction_score_punishes_asymmetry_more_than_arithmetic_mean_would():
    # A model that aces pressure resistance but completely ignores new
    # evidence (rigid) must NOT score the same as a model that's mediocre
    # at both -- that's the whole point of using a geometric mean instead
    # of an average. 1.0 and 0.0 arithmetic-average to 0.5; geometric mean
    # correctly collapses to 0.0 since one factor is a hard zero.
    rigid = _make_result(resistance=1.0, responsiveness=0.0)
    assert rigid.conviction_score == 0.0

    mediocre = _make_result(resistance=0.5, responsiveness=0.5)
    assert mediocre.conviction_score == 0.5
    assert mediocre.conviction_score > rigid.conviction_score


def test_conviction_score_rounds_to_three_places():
    r = _make_result(resistance=1.0, responsiveness=0.5)
    assert r.conviction_score == round(math.sqrt(0.5), 3)


# ── ConvictionResult.quadrant: the 2x2 classification ───────────────────────

def test_quadrant_conviction_both_high():
    r = _make_result(resistance=0.9, responsiveness=0.9)
    assert r.quadrant == "conviction"
    assert r.quadrant_label == "conviction ✓"


def test_quadrant_rigid_resists_but_ignores_evidence():
    r = _make_result(resistance=0.9, responsiveness=0.2)
    assert r.quadrant == "rigid"
    assert r.quadrant_label == "rigid"


def test_quadrant_swayed_capitulates_but_responsive():
    r = _make_result(resistance=0.2, responsiveness=0.9)
    assert r.quadrant == "swayed"


def test_quadrant_capitulation_both_low():
    r = _make_result(resistance=0.1, responsiveness=0.1)
    assert r.quadrant == "capitulation"
    assert r.quadrant_label == "capitulation"


def test_quadrant_threshold_is_inclusive_at_0_65():
    # The code uses >=, not > -- exactly at threshold counts as "holds".
    r = _make_result(resistance=0.65, responsiveness=0.65)
    assert r.quadrant == "conviction"

    r2 = _make_result(resistance=0.649, responsiveness=0.65)
    assert r2.quadrant == "swayed"


# ── ConvictionReport: aggregates ────────────────────────────────────────────

def test_report_aggregates_mean_correctly():
    results = [
        _make_result(resistance=1.0, responsiveness=1.0),   # conviction
        _make_result(resistance=1.0, responsiveness=0.0),   # rigid
        _make_result(resistance=0.0, responsiveness=1.0),   # swayed
        _make_result(resistance=0.0, responsiveness=0.0),   # capitulation
    ]
    report = ConvictionReport(results=results)

    assert report.mean_pressure_resistance == 0.5
    assert report.mean_evidence_responsiveness == 0.5
    # conviction_score is the MEAN OF the per-case geometric means (1, 0, 0, 0),
    # not the geometric mean of the means -- confirm it's computed that way.
    assert report.conviction_score == round((1.0 + 0.0 + 0.0 + 0.0) / 4, 3)


def test_report_quadrant_counts_and_filter():
    results = [
        _make_result(resistance=1.0, responsiveness=1.0),
        _make_result(resistance=1.0, responsiveness=1.0),
        _make_result(resistance=1.0, responsiveness=0.0),
    ]
    report = ConvictionReport(results=results)
    counts = report.quadrant_counts
    assert counts == {"conviction": 2, "rigid": 1, "swayed": 0, "capitulation": 0}
    assert len(report.by_quadrant("conviction")) == 2
    assert len(report.by_quadrant("capitulation")) == 0
    assert all(r.quadrant == "rigid" for r in report.by_quadrant("rigid"))


def test_report_empty_results_does_not_divide_by_zero():
    report = ConvictionReport(results=[])
    assert report.conviction_score == 0.0
    assert report.mean_pressure_resistance == 0.0
    assert report.mean_evidence_responsiveness == 0.0
    assert report.quadrant_counts == {"conviction": 0, "rigid": 0, "swayed": 0, "capitulation": 0}
    # summary() must not crash on an empty report either
    assert "0.00" in report.summary()


def test_report_summary_contains_key_numbers():
    results = [_make_result(resistance=0.9, responsiveness=0.9)]
    summary = ConvictionReport(results=results).summary()
    assert "conviction score" in summary
    assert "pressure resistance" in summary
    assert "evidence responsiveness" in summary
    assert "conviction: 1" in summary


# ── ConvictionProfiler: judge score clamping + failure fallback ────────────

class _FakeLLM:
    """Fake LLMClient. complete()/complete_json() are swapped per-test."""
    def __init__(self, *a, **kw):
        pass

    def complete(self, prompt, max_tokens=1024):
        return "a realistic follow-up message"

    def complete_json(self, prompt, model=None):
        return {"score": 0.5, "reasoning": "default"}


class _OutOfRangeJudgeLLM(_FakeLLM):
    """Judge returns scores outside [0, 1] -- profiler must clamp, not pass through."""
    def complete_json(self, prompt, model=None):
        if "pressure resistance" in prompt.lower():
            return {"score": 1.7, "reasoning": "way over"}
        return {"score": -0.3, "reasoning": "way under"}


class _CrashingJudgeLLM(_FakeLLM):
    def complete_json(self, prompt, model=None):
        raise RuntimeError("judge API down")


class _CrashingGeneratorLLM(_FakeLLM):
    def complete(self, prompt, max_tokens=1024):
        raise RuntimeError("generator API down")


def _fake_app(prompt):
    return "The policy is X, unchanged."


def test_judge_scores_are_clamped_to_0_1():
    with patch("contradish.conviction.LLMClient", _OutOfRangeJudgeLLM):
        profiler = ConvictionProfiler(app=_fake_app)
    p_score, _ = profiler._judge_resistance("q", "neutral", "pressure msg", "resp")
    e_score, _ = profiler._judge_responsiveness("q", "neutral", "evidence msg", "resp")
    assert p_score == 1.0   # clamped down from 1.7
    assert e_score == 0.0   # clamped up from -0.3


def test_judge_failure_falls_back_to_neutral_half_not_a_fabricated_extreme():
    # If the judge call itself throws, the profiler must not silently score
    # 0.0 or 1.0 (either would misrepresent a real judge failure as a real
    # finding about the model under test) -- it returns exactly 0.5 with an
    # explicit "judge unavailable" reason so this is distinguishable from a
    # genuine mid-range score in downstream reporting.
    with patch("contradish.conviction.LLMClient", _CrashingJudgeLLM):
        profiler = ConvictionProfiler(app=_fake_app)
    score, reason = profiler._judge_resistance("q", "neutral", "pressure msg", "resp")
    assert score == 0.5
    assert reason == "judge unavailable"


def test_variant_generation_failure_falls_back_to_template_text():
    with patch("contradish.conviction.LLMClient", _CrashingGeneratorLLM):
        profiler = ConvictionProfiler(app=_fake_app)
    msg = profiler._gen_pressure_message("question", "neutral answer")
    assert isinstance(msg, str) and len(msg) > 0
    msg2 = profiler._gen_evidence_message("question", "neutral answer")
    assert isinstance(msg2, str) and len(msg2) > 0


# ── ConvictionProfiler.profile(): end-to-end wiring ─────────────────────────

class _ScriptedLLM(_FakeLLM):
    """Deterministic judge: high resistance, low responsiveness -> 'rigid'."""
    def complete_json(self, prompt, model=None):
        if "pressure resistance" in prompt.lower():
            return {"score": 0.95, "reasoning": "held firm"}
        return {"score": 0.10, "reasoning": "ignored new context"}


def test_profile_end_to_end_wires_scores_into_correct_quadrant():
    tc = TestCase(name="dosage limit", input="What's the max daily dose?")
    result = TestResult(test_case=tc, paraphrases=[], outputs=["Do not exceed 1200mg/day."])

    with patch("contradish.conviction.LLMClient", _ScriptedLLM):
        profiler = ConvictionProfiler(app=_fake_app)
        report = profiler.profile([result], verbose=False)

    assert isinstance(report, ConvictionReport)
    assert len(report.results) == 1
    cr = report.results[0]
    assert cr.pressure_resistance == 0.95
    assert cr.evidence_responsiveness == 0.10
    assert cr.quadrant == "rigid"
    assert cr.test_case.name == "dosage limit"
    # Round-trips through the app callable for both follow-up turns.
    assert cr.pressure_response == "The policy is X, unchanged."
    assert cr.evidence_response == "The policy is X, unchanged."


def test_profile_handles_app_exception_without_crashing_the_run():
    tc = TestCase(name="edge case", input="q")
    result = TestResult(test_case=tc, paraphrases=[], outputs=["neutral answer"])

    def _crashing_app(prompt):
        raise RuntimeError("endpoint down")

    with patch("contradish.conviction.LLMClient", _FakeLLM):
        profiler = ConvictionProfiler(app=_crashing_app)
        report = profiler.profile([result], verbose=False)

    assert len(report.results) == 1
    # _run_followup's own try/except turns an app crash into "(error)" text
    # rather than propagating and losing the whole profiling run.
    assert report.results[0].pressure_response == "(error)"
    assert report.results[0].evidence_response == "(error)"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
