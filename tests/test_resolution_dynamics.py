"""
tests/test_resolution_dynamics.py -- deterministic tests for
contradish/resolution_dynamics.py. No API key required: everything in this
module is a pure scoring core over caller-supplied data, no model or judge
calls involved.
"""

import pytest

from contradish.resolution_dynamics import (
    ContradictionEvent,
    ResolutionProbe,
    ResolutionOutcome,
    EventType,
    classify_resolution,
    check_closure,
    check_event_type_consistency,
    EntrenchmentTrial,
    score_entrenchment_fidelity,
    score_signal_separation,
    holm_bonferroni_correction,
    score_multiple_signals,
    score_resolution_dynamics,
)


# ── classify_resolution ────────────────────────────────────────────────────────

def _event(event_id="ev-1", cell_id="cell-a", turn=3, **kw):
    return ContradictionEvent(event_id=event_id, cell_id=cell_id, description="d", detected_at_turn=turn, **kw)


def test_integrated_when_primary_holds_and_transfer_cell_confirms():
    event = _event()
    probes = [
        ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=True, is_stable=True),
        ResolutionProbe("p2", "ev-1", "cell-b", turn=5, verdict_matches_domain=True, is_stable=True, is_transfer_cell=True),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.INTEGRATED


def test_behaviorally_resolved_when_no_transfer_cell_probed():
    event = _event()
    probes = [
        ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=True, is_stable=True),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.BEHAVIORALLY_RESOLVED


def test_distorted_when_transfer_cell_fails():
    event = _event()
    probes = [
        ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=True, is_stable=True),
        ResolutionProbe("p2", "ev-1", "cell-b", turn=5, verdict_matches_domain=False, is_stable=True, is_transfer_cell=True),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.DISTORTED


def test_unresolved_when_never_verified():
    event = _event()
    probes = [
        ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=False, is_stable=True),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.UNRESOLVED


def test_unresolved_when_no_probes_at_all():
    event = _event()
    verdict = classify_resolution(event, [])
    assert verdict.outcome == ResolutionOutcome.UNRESOLVED


def test_unresolved_when_most_recent_probe_unstable():
    event = _event()
    probes = [
        ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=True, is_stable=True),
        ResolutionProbe("p2", "ev-1", "cell-a", turn=6, verdict_matches_domain=False, is_stable=False),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.UNRESOLVED


def test_forgotten_when_reverted_with_no_contradicting_info():
    event = _event()
    probes = [
        ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=True, is_stable=True),
        ResolutionProbe("p2", "ev-1", "cell-a", turn=7, verdict_matches_domain=False, is_stable=True),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.FORGOTTEN


def test_superseded_when_reverted_with_contradicting_info():
    event = _event()
    probes = [
        ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=True, is_stable=True),
        ResolutionProbe(
            "p2", "ev-1", "cell-a", turn=7, verdict_matches_domain=False, is_stable=True,
            later_information_contradicts_correction=True,
        ),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.SUPERSEDED


def test_forgotten_default_when_contradiction_flag_unset():
    """later_information_contradicts_correction defaulting to None resolves
    to FORGOTTEN, not SUPERSEDED -- the field only ever moves a
    classification toward SUPERSEDED, never away from it."""
    event = _event()
    probes = [
        ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=True, is_stable=True),
        ResolutionProbe("p2", "ev-1", "cell-a", turn=7, verdict_matches_domain=False, is_stable=True,
                         later_information_contradicts_correction=None),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.FORGOTTEN


def test_unstable_transfer_probe_does_not_confirm_integration():
    """An unstable-but-momentarily-correct transfer probe must not confirm
    integration -- mirrors the real bug the sandbox rounds caught."""
    event = _event()
    probes = [
        ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=True, is_stable=True),
        ResolutionProbe("p2", "ev-1", "cell-b", turn=5, verdict_matches_domain=True, is_stable=False, is_transfer_cell=True),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.DISTORTED


def test_probes_before_detection_turn_are_ignored_for_primary_cell():
    event = _event(turn=5)
    probes = [
        # correct before the event was even detected -- shouldn't count as
        # the verifying probe.
        ResolutionProbe("p0", "ev-1", "cell-a", turn=1, verdict_matches_domain=True, is_stable=True),
        ResolutionProbe("p1", "ev-1", "cell-a", turn=6, verdict_matches_domain=True, is_stable=True),
    ]
    verdict = classify_resolution(event, probes)
    assert verdict.outcome == ResolutionOutcome.BEHAVIORALLY_RESOLVED


def test_classify_resolution_raises_on_mismatched_event_id():
    event = _event()
    probes = [ResolutionProbe("p1", "ev-OTHER", "cell-a", turn=4, verdict_matches_domain=True)]
    with pytest.raises(ValueError):
        classify_resolution(event, probes)


def test_resolution_probe_rejects_transfer_and_control_both_true():
    with pytest.raises(ValueError):
        ResolutionProbe("p1", "ev-1", "cell-a", turn=1, is_transfer_cell=True, is_control_cell=True)


# ── check_closure ──────────────────────────────────────────────────────────────

def test_check_closure_no_violation():
    event = _event()
    probes = [
        ResolutionProbe("c1", "ev-1", "cell-x", turn=1, verdict_matches_domain=True, is_stable=True, is_control_cell=True),
        ResolutionProbe("c2", "ev-1", "cell-x", turn=6, verdict_matches_domain=True, is_stable=True, is_control_cell=True),
    ]
    result = check_closure(event, probes)
    assert result.closure_violated is False
    assert result.n_control_cells_checked == 1
    assert result.contaminated_cell_ids == []


def test_check_closure_detects_contamination():
    event = _event()
    probes = [
        ResolutionProbe("c1", "ev-1", "cell-x", turn=1, verdict_matches_domain=True, is_stable=True, is_control_cell=True),
        ResolutionProbe("c2", "ev-1", "cell-x", turn=6, verdict_matches_domain=False, is_stable=True, is_control_cell=True),
    ]
    result = check_closure(event, probes)
    assert result.closure_violated is True
    assert result.contaminated_cell_ids == ["cell-x"]


def test_check_closure_excludes_cells_without_before_and_after():
    event = _event()
    probes = [
        # only measured after the event -- no before/after comparison possible.
        ResolutionProbe("c1", "ev-1", "cell-y", turn=6, verdict_matches_domain=False, is_stable=True, is_control_cell=True),
    ]
    result = check_closure(event, probes)
    assert result.n_control_cells_checked == 0
    assert result.closure_violated is False


def test_check_closure_raises_on_mismatched_event_id():
    event = _event()
    probes = [ResolutionProbe("c1", "ev-OTHER", "cell-x", turn=1, is_control_cell=True)]
    with pytest.raises(ValueError):
        check_closure(event, probes)


# ── check_event_type_consistency ──────────────────────────────────────────────

def test_event_type_warning_on_update_with_excusal():
    event = _event(event_type=EventType.UPDATE, pragmatically_excused=True)
    warning = check_event_type_consistency(event)
    assert warning is not None
    assert "UPDATE" in warning


def test_no_warning_on_revision_with_excusal():
    event = _event(event_type=EventType.REVISION, pragmatically_excused=True)
    assert check_event_type_consistency(event) is None


def test_no_warning_when_event_type_untagged():
    event = _event(event_type=None, pragmatically_excused=True)
    assert check_event_type_consistency(event) is None


def test_no_warning_on_update_without_excusal_assessed():
    event = _event(event_type=EventType.UPDATE, pragmatically_excused=None)
    assert check_event_type_consistency(event) is None


# ── EntrenchmentTrial / score_entrenchment_fidelity ───────────────────────────

def test_entrenchment_trial_faithful_and_displacement_zero():
    trial = EntrenchmentTrial(
        trial_id="t1", domain="medication",
        commitments_at_stake=["patient_safety", "convenience"],
        entrenchment_order=["patient_safety", "convenience"],
        actual_sacrifice="convenience",
    )
    assert trial.correct_sacrifice == "convenience"
    assert trial.faithful is True
    assert trial.rank_displacement == 0


def test_entrenchment_trial_unfaithful_with_positive_displacement():
    trial = EntrenchmentTrial(
        trial_id="t2", domain="medication",
        commitments_at_stake=["patient_safety", "convenience"],
        entrenchment_order=["patient_safety", "convenience"],
        actual_sacrifice="patient_safety",
    )
    assert trial.faithful is False
    assert trial.rank_displacement == 1


def test_entrenchment_trial_rejects_sacrifice_not_at_stake():
    with pytest.raises(ValueError):
        EntrenchmentTrial(
            trial_id="t3", domain="d",
            commitments_at_stake=["a", "b"],
            entrenchment_order=["a", "b"],
            actual_sacrifice="c",
        )


def test_entrenchment_trial_rejects_order_mismatch():
    with pytest.raises(ValueError):
        EntrenchmentTrial(
            trial_id="t4", domain="d",
            commitments_at_stake=["a", "b"],
            entrenchment_order=["a", "c"],
            actual_sacrifice="a",
        )


def test_entrenchment_trial_rejects_fewer_than_two_commitments():
    with pytest.raises(ValueError):
        EntrenchmentTrial(
            trial_id="t5", domain="d",
            commitments_at_stake=["a"],
            entrenchment_order=["a"],
            actual_sacrifice="a",
        )


def test_score_entrenchment_fidelity_aggregate():
    trials = [
        EntrenchmentTrial("t1", "d", ["a", "b"], ["a", "b"], "b"),  # faithful, displacement 0
        EntrenchmentTrial("t2", "d", ["a", "b"], ["a", "b"], "a"),  # unfaithful, displacement 1
    ]
    report = score_entrenchment_fidelity(trials)
    assert report.fidelity_rate == 0.5
    assert report.mean_rank_displacement == 0.5
    assert report.unfaithful_trial_ids == ["t2"]


def test_score_entrenchment_fidelity_empty():
    report = score_entrenchment_fidelity([])
    assert report.fidelity_rate is None
    assert report.mean_rank_displacement is None


# ── score_signal_separation ───────────────────────────────────────────────────

def test_signal_separation_detects_large_synthetic_effect():
    validated = [0.9, 0.92, 0.88, 0.95, 0.91, 0.89, 0.93]
    invalidated = [0.2, 0.25, 0.18, 0.22, 0.19, 0.24, 0.21]
    report = score_signal_separation(validated, invalidated, signal_name="confidence", seed=42, n_permutations=2000)
    assert report.p_value < 0.05
    assert report.cohens_d > 1.0
    assert report.underpowered is False


def test_signal_separation_no_effect_case():
    validated = [0.5, 0.51, 0.49, 0.52, 0.48, 0.50, 0.53]
    invalidated = [0.5, 0.49, 0.51, 0.48, 0.52, 0.50, 0.47]
    report = score_signal_separation(validated, invalidated, signal_name="noise", seed=7, n_permutations=2000)
    assert report.p_value > 0.05


def test_signal_separation_underpowered_flag():
    report = score_signal_separation([0.9, 0.8], [0.1, 0.2], seed=1, n_permutations=500)
    assert report.underpowered is True


def test_signal_separation_insufficient_data():
    report = score_signal_separation([0.9], [0.1, 0.2], seed=1)
    assert report.p_value is None
    assert report.underpowered is True


def test_signal_separation_deterministic_under_seed():
    a, b = [0.9, 0.85, 0.92, 0.88], [0.3, 0.35, 0.28, 0.32]
    r1 = score_signal_separation(a, b, seed=99, n_permutations=500)
    r2 = score_signal_separation(a, b, seed=99, n_permutations=500)
    assert r1.p_value == r2.p_value


# ── holm_bonferroni_correction ────────────────────────────────────────────────

def test_holm_bonferroni_known_example():
    # p = [0.01, 0.02, 0.03, 0.04]; Holm multipliers in sorted order are
    # [4, 3, 2, 1], each step taking the running max so the sequence never
    # decreases: 0.04, 0.06, 0.06 (0.03*2=0.06, tied with running max),
    # 0.06 (0.04*1=0.04 < running max 0.06, so the max is kept).
    p_values = [0.01, 0.02, 0.03, 0.04]
    adjusted = holm_bonferroni_correction(p_values)
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[1] == pytest.approx(0.06)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[3] == pytest.approx(0.06)


def test_holm_bonferroni_monotonic_in_sorted_order():
    p_values = [0.2, 0.001, 0.05, 0.4, 0.01]
    adjusted = holm_bonferroni_correction(p_values)
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    sorted_adjusted = [adjusted[i] for i in order]
    assert sorted_adjusted == sorted(sorted_adjusted)


def test_holm_bonferroni_empty():
    assert holm_bonferroni_correction([]) == []


def test_holm_bonferroni_caps_at_one():
    adjusted = holm_bonferroni_correction([0.9, 0.8, 0.7])
    assert all(v <= 1.0 for v in adjusted)


# ── score_multiple_signals ────────────────────────────────────────────────────

def test_score_multiple_signals_applies_correction():
    signals = {
        "confidence": ([0.9, 0.92, 0.88, 0.95, 0.91], [0.2, 0.25, 0.18, 0.22, 0.19]),
        "hedging": ([0.5, 0.51, 0.49, 0.52, 0.48], [0.5, 0.49, 0.51, 0.48, 0.52]),
    }
    reports = score_multiple_signals(signals, seed=3, n_permutations=1000)
    assert reports["confidence"].adjusted_p_value is not None
    assert reports["hedging"].adjusted_p_value is not None
    assert reports["confidence"].adjusted_p_value >= reports["confidence"].p_value


def test_score_multiple_signals_skips_insufficient_data():
    signals = {"too_small": ([0.9], [0.1, 0.2])}
    reports = score_multiple_signals(signals, seed=1)
    assert reports["too_small"].p_value is None
    assert reports["too_small"].adjusted_p_value is None


# ── score_resolution_dynamics (batch) ─────────────────────────────────────────

def test_score_resolution_dynamics_batch_and_counts():
    events = [_event(event_id="ev-1"), _event(event_id="ev-2")]
    probes_by_event = {
        "ev-1": [ResolutionProbe("p1", "ev-1", "cell-a", turn=4, verdict_matches_domain=True, is_stable=True)],
        "ev-2": [],
    }
    report = score_resolution_dynamics(events, probes_by_event)
    counts = report.counts()
    assert counts["behaviorally_resolved"] == 1
    assert counts["unresolved"] == 1
    assert report.rate("unresolved") == 0.5
    assert report.rate(ResolutionOutcome.BEHAVIORALLY_RESOLVED) == 0.5


def test_score_resolution_dynamics_collects_event_type_warnings():
    events = [_event(event_id="ev-1", event_type=EventType.UPDATE, pragmatically_excused=True)]
    probes_by_event = {"ev-1": []}
    report = score_resolution_dynamics(events, probes_by_event)
    assert len(report.event_type_warnings) == 1


def test_resolution_dynamics_report_rate_none_when_empty():
    from contradish.resolution_dynamics import ResolutionDynamicsReport
    report = ResolutionDynamicsReport(verdicts=[])
    assert report.rate("integrated") is None
