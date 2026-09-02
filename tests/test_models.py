"""
Targeted test coverage for the remaining gaps in contradish/models.py's
TestCase, TestResult, and Report -- the pieces not already exercised
incidentally by every other test file in this suite (which is why
models.py was already at 91% before this file: these are the specific
edge branches nothing else happened to hit).

Covers:
  TestCase.__post_init__      equivalence_confidence clamping to [0, 1]
  TestResult.judgment_strain  the "representational" contradiction_type branch
  Report.truth_coverage       empty-results short circuit
  Report.judgment_coverage    no-eligible-cases short circuit
  Report.summary()            the one-line aggregate summary
  Report.failures_summary()   the human-readable failure listing
  Report.from_dict()          invalid risk value falling back to RiskLevel.LOW
"""
from contradish.models import (
    ContradictionPair,
    Report,
    RiskLevel,
    TestCase,
    TestResult,
)


# ── Fixture builders ────────────────────────────────────────────────────────

def _tc(input="a test question", name=None, **kw):
    return TestCase(input=input, name=name, **kw)


def _cp(output_a="out a", output_b="out b", severity="policy",
        explanation="they disagree", input_a="in a", input_b="in b"):
    return ContradictionPair(
        input_a=input_a, input_b=input_b,
        output_a=output_a, output_b=output_b,
        explanation=explanation, severity=severity,
    )


def _tr(tc=None, consistency_score=0.9, contradictions=None, **kw):
    return TestResult(
        test_case=tc or _tc(),
        paraphrases=["p1"],
        outputs=["o0", "o1"],
        consistency_score=consistency_score,
        contradiction_score=0.0,
        contradictions=contradictions or [],
        **kw,
    )


def _report(results, **kw):
    return Report(results=results, **kw)


# ── TestCase.__post_init__: equivalence_confidence clamping ────────────────

def test_equivalence_confidence_clamped_up_to_zero_when_negative():
    tc = _tc(equivalence_confidence=-0.5)
    assert tc.equivalence_confidence == 0.0


def test_equivalence_confidence_clamped_down_to_one_when_above_one():
    tc = _tc(equivalence_confidence=1.5)
    assert tc.equivalence_confidence == 1.0


def test_equivalence_confidence_in_range_is_unchanged():
    tc = _tc(equivalence_confidence=0.42)
    assert tc.equivalence_confidence == 0.42


# ── TestResult.judgment_strain: representational branch ────────────────────

def test_judgment_strain_representational_none_when_unscored():
    tc = _tc(contradiction_type="representational")
    r = _tr(tc=tc, reframe_score=None)
    assert r.judgment_strain is None


def test_judgment_strain_representational_computed_from_reframe_score():
    tc = _tc(contradiction_type="representational")
    r = _tr(tc=tc, reframe_score=0.8)
    assert r.judgment_strain == round(1.0 - 0.8, 4)


def test_judgment_strain_falls_back_to_cai_strain_for_an_unrecognized_type():
    # TestCase.__post_init__ always normalizes contradiction_type to one of
    # CONTRADICTION_TYPES, so this branch is unreachable through normal
    # construction -- it's defensive dead code for "shouldn't happen, but
    # if it does, degrade to drift-only scoring rather than crash." Hit it
    # the same way test_printer.py forces its blank-name edge case: build a
    # normally-constructed object, then override the field directly.
    tc = _tc(contradiction_type="adversarial")
    tc.contradiction_type = "some_future_type_this_version_does_not_know"
    r = _tr(tc=tc, consistency_score=0.7)
    assert r.judgment_strain == r.cai_strain


# ── Report.truth_coverage / judgment_coverage short circuits ───────────────

def test_truth_coverage_zero_on_empty_report():
    assert _report([]).truth_coverage == 0.0


def test_judgment_coverage_zero_when_no_eligible_cases():
    # equivalence_confidence below eq_threshold -> no cases are "eligible"
    tc = _tc(equivalence_confidence=0.1)
    report = _report([_tr(tc=tc, consistency_score=0.9)], eq_threshold=0.80)
    assert report.judgment_coverage == 0.0


# ── Report.summary() ─────────────────────────────────────────────────────────

def test_summary_includes_all_fields_when_scored():
    tc1 = _tc(name="pass case")
    tc2 = _tc(name="fail case")
    report = _report([
        _tr(tc=tc1, consistency_score=0.95),
        _tr(tc=tc2, consistency_score=0.4),
    ])
    out = report.summary()
    assert "Judgment Strain:" in out
    assert "CAI Strain:" in out
    assert "EQ coverage:" in out
    assert "1/2 passed" in out
    assert "1 failure(s)" in out


def test_summary_shows_na_when_nothing_scored():
    out = _report([]).summary()
    assert "Judgment Strain: n/a" in out
    assert "CAI Strain: n/a" in out
    assert "0/0 passed" in out
    assert "0 failure(s)" in out


# ── Report.failures_summary() ────────────────────────────────────────────────

def test_failures_summary_no_failures():
    report = _report([_tr(consistency_score=0.95)])
    assert report.failures_summary() == "No failures."


def test_failures_summary_lists_rule_name_and_strain():
    tc = _tc(name="refund rule")
    report = _report([_tr(tc=tc, consistency_score=0.4)])
    out = report.failures_summary()
    assert "1 CAI failure(s):" in out
    assert "refund rule (CAI Strain 0.60)" in out


def test_failures_summary_defaults_strain_to_na_when_unscored():
    tc = _tc(name="unscored rule")
    r = TestResult(
        test_case=tc, paraphrases=["p"], outputs=["o0", "o1"],
        consistency_score=None, contradiction_score=0.9,  # fails via contradiction_score
    )
    out = _report([r]).failures_summary()
    assert "CAI Strain n/a" in out


def test_failures_summary_shows_only_first_contradiction_pair():
    pairs = [
        _cp(input_a="q1", output_a="a1", output_b="b1"),
        _cp(input_a="q2", output_a="a2", output_b="b2"),
    ]
    tc = _tc(name="rule")
    report = _report([_tr(tc=tc, consistency_score=0.4, contradictions=pairs)])
    out = report.failures_summary()
    assert "asked: 'q1'" in out
    assert "got:   'a1'" in out
    assert "vs:    'b1'" in out
    assert "q2" not in out  # only the first pair is shown


def test_failures_summary_truncates_long_outputs_to_120_chars():
    long_output_a = "A" * 200
    long_output_b = "B" * 200
    pair = _cp(output_a=long_output_a, output_b=long_output_b)
    tc = _tc(name="rule")
    report = _report([_tr(tc=tc, consistency_score=0.4, contradictions=[pair])])
    out = report.failures_summary()
    assert "A" * 120 in out
    assert "A" * 121 not in out
    assert "B" * 120 in out
    assert "B" * 121 not in out


def test_failures_summary_multiple_failures_all_listed():
    tc1 = _tc(name="rule one")
    tc2 = _tc(name="rule two")
    report = _report([
        _tr(tc=tc1, consistency_score=0.4),
        _tr(tc=tc2, consistency_score=0.3),
    ])
    out = report.failures_summary()
    assert "2 CAI failure(s):" in out
    assert "rule one" in out
    assert "rule two" in out


# ── Report.from_dict(): invalid risk value fallback ─────────────────────────

def test_from_dict_falls_back_to_low_risk_on_invalid_risk_value():
    data = {
        "results": [
            {"input": "q1", "name": "n1", "risk": "not-a-real-risk-level", "cai_score": 0.5},
        ],
    }
    report = Report.from_dict(data)
    assert len(report.results) == 1
    assert report.results[0].risk == RiskLevel.LOW


def test_from_dict_uses_valid_risk_value():
    data = {"results": [{"input": "q1", "risk": "high", "cai_score": 0.5}]}
    report = Report.from_dict(data)
    assert report.results[0].risk == RiskLevel.HIGH
