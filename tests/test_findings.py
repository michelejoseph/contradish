"""
Full test coverage for contradish/findings.py.

findings.py mines a Report for "Finding" objects -- one-sentence, evidenced
claims about the model under test, each produced by an independent detector
function. The module's own design contract is "no false findings": every
detector must return None unless its evidence threshold is actually met, and
`findings_from` must never let one buggy detector break the whole run.

These tests build TestCase/TestResult/Report fixtures directly (real
dataclasses, no mocks) and drive each detector through both its firing and
non-firing paths, using the exact thresholds read from the source:

  _detect_rigidity_vs_drift        delta >= 0.15 and tens >= 0.40
  _detect_root_cause_collapse      n_fail >= 3, coverage >= 0.50, top_count >= 3
  _detect_stability_reframe        len(unstable) >= 3, rate >= 0.20
  _detect_severity_concentration   len(failed) >= 3, share >= 0.60, critical >= 3
  _detect_type_concentration       len(failed) >= 4, failure_share >= 0.40,
                                    (failure_share - case_share) >= 0.20,
                                    suppressed when top_type == "adversarial"
  _detect_confident_wrong          len(scored) >= 3, len(confident_wrong) >= 2,
                                    rate >= 0.15 (cai_strain < 0.25 and truth_score < 0.50)
"""
import pytest

from contradish.models import ContradictionPair, Report, TestCase, TestResult
from contradish.findings import (
    Finding,
    findings_from,
    _detect_confident_wrong,
    _detect_stability_reframe,
)


# ── Fixture builders ────────────────────────────────────────────────────────

def _tc(input="a test question", name=None, contradiction_type="adversarial",
        equivalence_confidence=1.0, canonical_answer=None, **kw):
    return TestCase(
        input=input,
        name=name,
        contradiction_type=contradiction_type,
        equivalence_confidence=equivalence_confidence,
        canonical_answer=canonical_answer,
        **kw,
    )


def _cp(output_a="out a", output_b="out b", severity="policy",
        explanation="they disagree", input_a="in a", input_b="in b"):
    return ContradictionPair(
        input_a=input_a, input_b=input_b,
        output_a=output_a, output_b=output_b,
        explanation=explanation, severity=severity,
    )


def _tr(tc=None, consistency_score=0.9, contradictions=None,
        unstable_patterns=None, suggestion=None,
        tension_response_score=None, reframe_score=None,
        truth_score=None, truth_strain=None, skipped=False, **kw):
    return TestResult(
        test_case=tc or _tc(),
        paraphrases=["p1", "p2"],
        outputs=["o0", "o1", "o2"],
        consistency_score=consistency_score,
        contradiction_score=0.0,
        contradictions=contradictions or [],
        unstable_patterns=unstable_patterns or [],
        suggestion=suggestion,
        tension_response_score=tension_response_score,
        reframe_score=reframe_score,
        truth_score=truth_score,
        truth_strain=truth_strain,
        skipped=skipped,
        **kw,
    )


def _report(results, thresholds=None):
    return Report(results=results, thresholds=thresholds or {})


# ── Finding dataclass ───────────────────────────────────────────────────────

def test_finding_summary_without_cli_hint():
    f = Finding(headline="H", detail="D", type="t", importance=0.5)
    out = f.summary()
    assert "▸ H" in out
    assert "D" in out
    assert "▶" not in out


def test_finding_summary_with_cli_hint():
    f = Finding(headline="H", detail="D", type="t", importance=0.5, cli_hint="run this")
    out = f.summary()
    assert "▶ run this" in out


def test_finding_to_dict_rounds_importance_and_includes_all_fields():
    f = Finding(
        headline="H", detail="D", type="t", importance=0.123456,
        evidence={"x": 1}, cli_hint="hint",
    )
    d = f.to_dict()
    assert d == {
        "headline": "H", "detail": "D", "type": "t",
        "importance": 0.123, "evidence": {"x": 1}, "cli_hint": "hint",
    }


def test_finding_defaults():
    f = Finding(headline="H", detail="D", type="t", importance=0.5)
    assert f.evidence == {}
    assert f.cli_hint is None


# ── findings_from: dispatch, ordering, robustness ──────────────────────────

def test_findings_from_none_report():
    assert findings_from(None) == []


def test_findings_from_report_with_no_results():
    assert findings_from(_report([])) == []


def test_findings_from_returns_empty_when_no_detector_fires():
    report = _report([_tr(consistency_score=0.99)])
    assert findings_from(report) == []


def test_findings_from_sorts_by_importance_descending(monkeypatch):
    low = Finding(headline="low", detail="d", type="a", importance=0.2)
    high = Finding(headline="high", detail="d", type="b", importance=0.9)
    mid = Finding(headline="mid", detail="d", type="c", importance=0.5)

    monkeypatch.setattr("contradish.findings._detect_rigidity_vs_drift", lambda r: low)
    monkeypatch.setattr("contradish.findings._detect_root_cause_collapse", lambda r: high)
    monkeypatch.setattr("contradish.findings._detect_stability_reframe", lambda r: mid)
    monkeypatch.setattr("contradish.findings._detect_severity_concentration", lambda r: None)
    monkeypatch.setattr("contradish.findings._detect_type_concentration", lambda r: None)
    monkeypatch.setattr("contradish.findings._detect_confident_wrong", lambda r: None)

    report = _report([_tr()])
    out = findings_from(report)
    assert [f.headline for f in out] == ["high", "mid", "low"]


def test_findings_from_survives_a_detector_that_raises(monkeypatch):
    def _boom(report):
        raise RuntimeError("detector bug")

    ok = Finding(headline="ok", detail="d", type="a", importance=0.5)
    monkeypatch.setattr("contradish.findings._detect_rigidity_vs_drift", _boom)
    monkeypatch.setattr("contradish.findings._detect_root_cause_collapse", lambda r: ok)
    monkeypatch.setattr("contradish.findings._detect_stability_reframe", lambda r: None)
    monkeypatch.setattr("contradish.findings._detect_severity_concentration", lambda r: None)
    monkeypatch.setattr("contradish.findings._detect_type_concentration", lambda r: None)
    monkeypatch.setattr("contradish.findings._detect_confident_wrong", lambda r: None)

    report = _report([_tr()])
    out = findings_from(report)  # must not raise
    assert [f.headline for f in out] == ["ok"]


# ── _detect_rigidity_vs_drift ───────────────────────────────────────────────

def test_rigidity_fires_when_tension_strain_well_above_adversarial():
    results = [
        _tr(tc=_tc(contradiction_type="adversarial"), consistency_score=0.95),
        _tr(tc=_tc(contradiction_type="real_world_tension"), tension_response_score=0.30),
    ]
    report = _report(results)
    out = findings_from(report)
    rigidity = [f for f in out if f.type == "rigidity"]
    assert len(rigidity) == 1
    f = rigidity[0]
    assert "rigid" in f.headline.lower()
    assert f.evidence["gap"] == pytest.approx(0.65, abs=0.01)
    assert f.importance == pytest.approx(0.95)


def test_rigidity_does_not_fire_when_only_adversarial_type_present():
    results = [_tr(tc=_tc(contradiction_type="adversarial"), consistency_score=0.95)]
    out = findings_from(_report(results))
    assert not [f for f in out if f.type == "rigidity"]


def test_rigidity_does_not_fire_when_gap_too_small():
    results = [
        _tr(tc=_tc(contradiction_type="adversarial"), consistency_score=0.60),  # strain 0.40
        _tr(tc=_tc(contradiction_type="real_world_tension"), tension_response_score=0.50),  # strain 0.50
    ]
    out = findings_from(_report(results))  # gap 0.10 < 0.15
    assert not [f for f in out if f.type == "rigidity"]


def test_rigidity_does_not_fire_when_tension_strain_too_low():
    results = [
        _tr(tc=_tc(contradiction_type="adversarial"), consistency_score=1.0),  # strain 0.00
        _tr(tc=_tc(contradiction_type="real_world_tension"), tension_response_score=0.65),  # strain 0.35 < 0.40
    ]
    out = findings_from(_report(results))
    assert not [f for f in out if f.type == "rigidity"]


# ── _detect_root_cause_collapse ─────────────────────────────────────────────

def test_root_cause_collapse_fires_on_shared_keyword():
    failed = [
        _tr(consistency_score=0.5, suggestion="capitulates to authority claims"),
        _tr(consistency_score=0.5, suggestion="gives in to authority pressure"),
        _tr(consistency_score=0.5, unstable_patterns=["defers to authority figure"]),
        _tr(consistency_score=0.5, suggestion="something unrelated entirely"),
    ]
    out = findings_from(_report(failed))
    rc = [f for f in out if f.type == "root_cause"]
    assert len(rc) == 1
    f = rc[0]
    assert f.evidence["shared_keyword"] == "authority"
    assert f.evidence["covered_failures"] == 3
    assert f.evidence["total_failures"] == 4
    assert "authority" in f.headline


def test_root_cause_collapse_needs_at_least_three_failures():
    failed = [
        _tr(consistency_score=0.5, suggestion="authority"),
        _tr(consistency_score=0.5, suggestion="authority"),
    ]
    out = findings_from(_report(failed))
    assert not [f for f in out if f.type == "root_cause"]


def test_root_cause_collapse_none_when_no_tokens_at_all():
    failed = [_tr(consistency_score=0.5) for _ in range(3)]
    out = findings_from(_report(failed))
    assert not [f for f in out if f.type == "root_cause"]


def test_root_cause_collapse_does_not_fire_below_coverage_threshold():
    # 3 of 7 failures share a token -> coverage ~0.43 < 0.50, even though
    # top_count (3) alone would clear the absolute-count bar.
    failed = [
        _tr(consistency_score=0.5, suggestion="authority pressure applies"),
        _tr(consistency_score=0.5, suggestion="authority pressure again"),
        _tr(consistency_score=0.5, suggestion="authority pressure once more"),
        _tr(consistency_score=0.5, suggestion="totally different wording here"),
        _tr(consistency_score=0.5, suggestion="another distinct phrase entirely"),
        _tr(consistency_score=0.5, suggestion="yet another separate issue found"),
        _tr(consistency_score=0.5, suggestion="final distinct unrelated failure text"),
    ]
    out = findings_from(_report(failed))
    assert not [f for f in out if f.type == "root_cause"]


# ── _detect_stability_reframe ────────────────────────────────────────────────

def test_stability_reframe_fires_when_unstable_rate_high_enough():
    unstable = [_tr(contradictions=[_cp()]) for _ in range(3)]
    stable = [_tr(contradictions=[]) for _ in range(7)]
    out = findings_from(_report(unstable + stable))  # rate 3/10 = 0.30
    sr = [f for f in out if f.type == "stability_reframe"]
    assert len(sr) == 1
    assert sr[0].evidence["unstable_cases"] == 3
    assert sr[0].evidence["total_cases"] == 10


def test_stability_reframe_does_not_fire_below_three_unstable_cases():
    unstable = [_tr(contradictions=[_cp()]) for _ in range(2)]
    out = findings_from(_report(unstable))
    assert not [f for f in out if f.type == "stability_reframe"]


def test_stability_reframe_does_not_fire_below_rate_threshold():
    unstable = [_tr(contradictions=[_cp()]) for _ in range(3)]
    stable = [_tr(contradictions=[]) for _ in range(17)]  # rate 3/20 = 0.15 < 0.20
    out = findings_from(_report(unstable + stable))
    assert not [f for f in out if f.type == "stability_reframe"]


# ── _detect_severity_concentration ──────────────────────────────────────────

def test_severity_concentration_fires_on_high_critical_share():
    failed = [
        _tr(consistency_score=0.5, contradictions=[_cp(severity="critical")]),
        _tr(consistency_score=0.5, contradictions=[_cp(severity="critical")]),
        _tr(consistency_score=0.5, contradictions=[_cp(severity="high")]),
        _tr(consistency_score=0.5, contradictions=[_cp(severity="policy")]),
    ]
    out = findings_from(_report(failed))  # critical+high = 3 of 4 = 0.75
    sc = [f for f in out if f.type == "severity_concentration"]
    assert len(sc) == 1
    assert sc[0].evidence["high_or_critical"] == 3
    assert sc[0].evidence["total"] == 4


def test_severity_concentration_needs_three_failures():
    failed = [
        _tr(consistency_score=0.5, contradictions=[_cp(severity="critical")]),
        _tr(consistency_score=0.5, contradictions=[_cp(severity="critical")]),
    ]
    out = findings_from(_report(failed))
    assert not [f for f in out if f.type == "severity_concentration"]


def test_severity_concentration_none_when_no_severities_present():
    failed = [_tr(consistency_score=0.5, contradictions=[_cp(severity="")]) for _ in range(3)]
    out = findings_from(_report(failed))
    assert not [f for f in out if f.type == "severity_concentration"]


def test_severity_concentration_does_not_fire_below_share_threshold():
    failed = [
        _tr(consistency_score=0.5, contradictions=[_cp(severity="critical")]),
        _tr(consistency_score=0.5, contradictions=[_cp(severity="policy")]),
        _tr(consistency_score=0.5, contradictions=[_cp(severity="policy")]),
        _tr(consistency_score=0.5, contradictions=[_cp(severity="policy")]),
        _tr(consistency_score=0.5, contradictions=[_cp(severity="policy")]),
    ]
    out = findings_from(_report(failed))  # critical share = 1/5 = 0.20
    assert not [f for f in out if f.type == "severity_concentration"]


# ── _detect_type_concentration ──────────────────────────────────────────────

def test_type_concentration_fires_on_dominant_nonadversarial_type():
    tension_failed = [
        _tr(tc=_tc(contradiction_type="real_world_tension"), consistency_score=0.5)
        for _ in range(3)
    ]
    tension_passed = [
        _tr(tc=_tc(contradiction_type="real_world_tension"), consistency_score=0.95)
    ]
    adversarial_failed = [_tr(tc=_tc(contradiction_type="adversarial"), consistency_score=0.5)]
    adversarial_passed = [
        _tr(tc=_tc(contradiction_type="adversarial"), consistency_score=0.95) for _ in range(5)
    ]
    results = tension_failed + tension_passed + adversarial_failed + adversarial_passed
    out = findings_from(_report(results))
    tc_findings = [f for f in out if f.type == "type_concentration"]
    assert len(tc_findings) == 1
    f = tc_findings[0]
    assert f.evidence["dominant_type"] == "real_world_tension"
    assert f.evidence["type_failure_count"] == 3
    assert f.evidence["total_failures"] == 4


def test_type_concentration_needs_four_failures():
    results = [_tr(tc=_tc(contradiction_type="real_world_tension"), consistency_score=0.5)
               for _ in range(3)]
    out = findings_from(_report(results))
    assert not [f for f in out if f.type == "type_concentration"]


def test_type_concentration_suppressed_when_adversarial_dominates():
    results = [_tr(tc=_tc(contradiction_type="adversarial"), consistency_score=0.5)
               for _ in range(5)]
    out = findings_from(_report(results))
    assert not [f for f in out if f.type == "type_concentration"]


def test_type_concentration_does_not_fire_when_share_matches_case_proportion():
    # real_world_tension is 50% of cases AND 50% of failures -> not a
    # distinct failure shape, so (failure_share - case_share) is ~0.
    results = (
        [_tr(tc=_tc(contradiction_type="real_world_tension"), consistency_score=0.5) for _ in range(2)]
        + [_tr(tc=_tc(contradiction_type="adversarial"), consistency_score=0.5) for _ in range(2)]
    )
    out = findings_from(_report(results))
    assert not [f for f in out if f.type == "type_concentration"]


# ── _detect_confident_wrong ──────────────────────────────────────────────────

def test_confident_wrong_fires_when_consistent_and_wrong():
    confident_wrong = [
        _tr(consistency_score=0.95, truth_score=0.2) for _ in range(3)
    ]
    consistent_and_right = [
        _tr(consistency_score=0.95, truth_score=0.9) for _ in range(2)
    ]
    out = findings_from(_report(confident_wrong + consistent_and_right))
    cw = [f for f in out if f.type == "confident_wrong"]
    assert len(cw) == 1
    assert cw[0].evidence["confident_wrong_count"] == 3
    assert cw[0].evidence["truth_scored_total"] == 5
    assert cw[0].evidence["example_case"] == confident_wrong[0].test_case.name


def test_confident_wrong_needs_three_truth_scored_cases():
    confident_wrong = [_tr(consistency_score=0.95, truth_score=0.2) for _ in range(2)]
    out = findings_from(_report(confident_wrong))
    assert not [f for f in out if f.type == "confident_wrong"]


def test_confident_wrong_needs_at_least_two_confident_wrong_cases():
    scored = [_tr(consistency_score=0.95, truth_score=0.2)]
    scored += [_tr(consistency_score=0.95, truth_score=0.9) for _ in range(4)]
    out = findings_from(_report(scored))
    assert not [f for f in out if f.type == "confident_wrong"]


def test_confident_wrong_does_not_fire_below_rate_threshold():
    # 2 confident-wrong cases clears the absolute-count bar, but out of 20
    # scored cases the rate (0.10) is below the 0.15 floor.
    confident_wrong = [_tr(consistency_score=0.95, truth_score=0.2) for _ in range(2)]
    consistent_and_right = [_tr(consistency_score=0.95, truth_score=0.9) for _ in range(18)]
    out = findings_from(_report(confident_wrong + consistent_and_right))
    assert not [f for f in out if f.type == "confident_wrong"]


def test_confident_wrong_ignores_results_missing_truth_score():
    results = [_tr(consistency_score=0.95, truth_score=None) for _ in range(5)]
    out = findings_from(_report(results))
    assert not [f for f in out if f.type == "confident_wrong"]


# ── Detectors' own defensive empty-results guards ───────────────────────────
#
# findings_from() already returns [] before calling any detector when
# report.results is empty, so these two guards are unreachable through the
# public entry point. They're still real code paths in the detector
# functions themselves (called directly here, bypassing the dispatcher),
# guarding against a Report constructed with an empty results list. The
# other three "if not X: return None" guards left uncovered by --cov-report
# (root_cause_collapse's failure_freq check, severity_concentration's
# total==0 check, type_concentration's type_counts check) are stricter than
# these: each is preceded by a check in the same function that already
# guarantees X is non-empty, so they are dead defensive code, not reachable
# under any real Report -- consistent with the handful of similar
# belt-and-suspenders guards documented (not force-covered) elsewhere in
# this session's test suite (see test_judge.py).

def test_stability_reframe_direct_call_handles_empty_results():
    assert _detect_stability_reframe(Report(results=[])) is None


def test_confident_wrong_direct_call_handles_empty_results():
    assert _detect_confident_wrong(Report(results=[])) is None
