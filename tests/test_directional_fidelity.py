"""
Tests for contradish.directional_fidelity.

No model calls: DistinctionProfile/DistinctionMeasurement objects are built
directly with fabricated but realistic both_correct/distinction_held values,
the same pure-scoring-layer pattern used throughout this package's tests.
"""
from contradish.decision_relevance import (
    DRSFactor,
    default_technique_drs,
    score_dependency_structure,
)
from contradish.distinction import DistinctionMeasurement, DistinctionPair, DistinctionProfile
from contradish.directional_fidelity import (
    DirectionalFidelityAudit,
    DirectionalFidelityReport,
    aggregate_directional_fidelity,
    directional_fidelity_for_domain,
    drs_factor_from_distinction_pair,
    expected_effect_matches_from_reports,
    score_directional_fidelity,
    spec_with_distinction_pairs,
)


def _pair(pair_id="test_pair"):
    return DistinctionPair(
        pair_id=pair_id,
        description="state A vs state B",
        label_a="state A", label_b="state B",
        question_a="question for A?", question_b="question for B?",
        commit_a="commit-a", commit_b="commit-b",
    )


def _measurement(both_correct, distinction_held=None, framing_type="emotional", intensity=1):
    # distinction_held defaults to both_correct's implication: if both_correct
    # is True the commitments necessarily differ (commit_a != commit_b in
    # _pair()), so distinction_held is True too; a caller can still override
    # to build a "held but wrong" case (different answers, neither correct).
    held = distinction_held if distinction_held is not None else both_correct
    return DistinctionMeasurement(
        pair_id="test_pair", framing_type=framing_type, intensity=intensity,
        framing_prefix="", answer_a="a", answer_b="b",
        extracted_commit_a="commit-a" if both_correct else "commit-x",
        extracted_commit_b="commit-b" if both_correct else ("commit-x" if not held else "commit-y"),
        distinction_held=held, both_correct=both_correct,
    )


def _profile(measurements, overall_hold_rate=None):
    held_count = sum(1 for m in measurements if m.distinction_held)
    hold_rate = overall_hold_rate if overall_hold_rate is not None else (
        held_count / len(measurements) if measurements else 0.0
    )
    return DistinctionProfile(
        pair_id="test_pair", description="state A vs state B",
        label_a="state A", label_b="state B",
        hold_rate_per_framing={"emotional": hold_rate},
        overall_hold_rate=hold_rate,
        collapse_framing="emotional",
        first_collapse=None,
        measurements=measurements,
    )


# ── drs_factor_from_distinction_pair / spec_with_distinction_pairs ──────────

def test_drs_factor_from_pair_is_relevant_with_real_expected_effect():
    factor = drs_factor_from_distinction_pair(_pair())
    assert factor.relevance == "relevant"
    assert factor.expected_effect != ""
    assert "commit-a" in factor.expected_effect
    assert "commit-b" in factor.expected_effect


def test_spec_with_distinction_pairs_has_eight_technique_factors_plus_the_pair():
    spec = spec_with_distinction_pairs("c1", "test", [_pair()])
    assert len(spec.factors) == 9   # 8 techniques + 1 relevant factor
    assert spec.factors["test_pair"].relevance == "relevant"
    assert spec.factors["emotional"].relevance == "irrelevant"


# ── score_directional_fidelity: the three cells ──────────────────────────────

def test_all_correct_measurements_score_tracked_correct():
    profile = _profile([_measurement(both_correct=True) for _ in range(5)])
    report = score_directional_fidelity("c1", "test", _pair(), profile)
    assert report.cell == "tracked_correct"
    assert report.sensitivity == 1.0
    assert report.directional_correctness == 1.0


def test_sensitive_but_wrong_direction_is_distinguished_from_correct():
    # distinction held (different answers) every time, but never the RIGHT
    # pair of answers -- exactly the case DRS's plain "tracked" cell could
    # not previously distinguish from tracked_correct.
    measurements = [_measurement(both_correct=False, distinction_held=True) for _ in range(5)]
    profile = _profile(measurements)
    report = score_directional_fidelity("c1", "test", _pair(), profile)
    assert report.sensitivity == 1.0          # fully sensitive
    assert report.directional_correctness == 0.0
    assert report.cell == "tracked_wrong_direction"


def test_collapsed_distinction_scores_missed_regardless_of_direction():
    measurements = [_measurement(both_correct=False, distinction_held=False) for _ in range(5)]
    profile = _profile(measurements)
    report = score_directional_fidelity("c1", "test", _pair(), profile)
    assert report.cell == "missed"


def test_accepts_bare_pair_id_string_as_well_as_pair_object():
    profile = _profile([_measurement(both_correct=True)])
    by_string = score_directional_fidelity("c1", "test", "test_pair", profile)
    by_object = score_directional_fidelity("c1", "test", _pair(), profile)
    assert by_string.pair_id == by_object.pair_id == "test_pair"


def test_empty_measurements_scores_missed_not_a_crash():
    profile = _profile([], overall_hold_rate=0.0)
    report = score_directional_fidelity("c1", "test", _pair(), profile)
    assert report.n_measurements == 0
    assert report.cell == "missed"
    assert report.directional_correctness == 0.0


def test_threshold_is_respected_at_the_boundary():
    # 1 of 2 correct = 0.5 directional_correctness, and 0.5 >= default
    # threshold 0.5 -> tracked_correct (>=, not >, matching
    # score_dependency_structure's own >= threshold convention).
    measurements = [
        _measurement(both_correct=True, distinction_held=True),
        _measurement(both_correct=False, distinction_held=True),
    ]
    profile = _profile(measurements)
    report = score_directional_fidelity("c1", "test", _pair(), profile)
    assert report.directional_correctness == 0.5
    assert report.cell == "tracked_correct"


# ── aggregate_directional_fidelity ───────────────────────────────────────────

def test_aggregate_pools_counts_and_computes_fidelity():
    reports = {
        "p1": DirectionalFidelityReport("c1", "test", "p1", 1.0, 1.0, "tracked_correct", 5),
        "p2": DirectionalFidelityReport("c2", "test", "p2", 1.0, 0.0, "tracked_wrong_direction", 5),
        "p3": DirectionalFidelityReport("c3", "test", "p3", 0.0, 0.0, "missed", 5),
    }
    audit = aggregate_directional_fidelity("test", reports)
    assert audit.tracked_correct == ["p1"]
    assert audit.tracked_wrong_direction == ["p2"]
    assert audit.missed == ["p3"]
    assert audit.pooled_directional_fidelity == 0.5   # 1 correct of 2 tracked


def test_aggregate_with_no_reports_has_none_fidelity():
    audit = aggregate_directional_fidelity("test", {})
    assert audit.pooled_directional_fidelity is None
    assert audit.n_factors == 0


# ── directional_fidelity_for_domain: real JUNCTION_CASE_MAP-shaped wiring ───

class _FakeLossMap:
    def __init__(self, profiles):
        self.profiles = profiles


def test_domain_wrapper_only_scores_pairs_present_in_junction_case_map():
    profile_mapped = _profile([_measurement(both_correct=True) for _ in range(3)])
    profile_unmapped = _profile([_measurement(both_correct=True) for _ in range(3)])
    loss_map = _FakeLossMap({"mapped_pair": profile_mapped, "unmapped_pair": profile_unmapped})
    junction_case_map = {"mapped_pair": ["case-001"]}

    audit = directional_fidelity_for_domain("test", loss_map, junction_case_map)
    assert audit.n_factors == 1
    assert "mapped_pair" in audit.by_pair
    assert "unmapped_pair" not in audit.by_pair


def test_domain_wrapper_uses_first_case_id_when_a_pair_maps_to_several():
    profile = _profile([_measurement(both_correct=True) for _ in range(3)])
    loss_map = _FakeLossMap({"shared_pair": profile})
    junction_case_map = {"shared_pair": ["case-001", "case-002"]}

    audit = directional_fidelity_for_domain("test", loss_map, junction_case_map)
    assert audit.by_pair["shared_pair"].commitment_id == "case-001"
    # one measured pair should never be double-counted just because it maps
    # to two cases
    assert audit.n_factors == 1


def test_domain_wrapper_matches_direct_score_directional_fidelity_call():
    # No duplicated threshold logic: the wrapper's per-pair result must be
    # bit-identical to calling score_directional_fidelity directly.
    profile = _profile([
        _measurement(both_correct=True, distinction_held=True),
        _measurement(both_correct=False, distinction_held=True),
    ])
    loss_map = _FakeLossMap({"p": profile})
    junction_case_map = {"p": ["case-001"]}

    via_wrapper = directional_fidelity_for_domain("test", loss_map, junction_case_map).by_pair["p"]
    via_direct = score_directional_fidelity("case-001", "test", "p", profile)
    assert via_wrapper == via_direct


# ── expected_effect_matches_from_reports: the bridge into decision_relevance ─
# Added 2026-09-13 alongside decision_relevance.score_dependency_structure's
# new expected_effect_matches parameter -- these prove the bridge's mapping
# is exactly what that module's docstring promises, and that the two
# modules actually compose end to end, not just side by side.

def test_bridge_maps_tracked_correct_to_true():
    reports = {"p1": DirectionalFidelityReport("c1", "test", "p1", 1.0, 1.0, "tracked_correct", 5)}
    assert expected_effect_matches_from_reports(reports) == {"p1": True}


def test_bridge_maps_tracked_wrong_direction_to_false():
    reports = {"p1": DirectionalFidelityReport("c1", "test", "p1", 1.0, 0.0, "tracked_wrong_direction", 5)}
    assert expected_effect_matches_from_reports(reports) == {"p1": False}


def test_bridge_omits_missed_entirely():
    """A "missed" report has no direction to report -- it must not appear in
    the output dict at all (not even as False), since
    score_dependency_structure only ever consults this dict for factors it
    has independently classified "tracked" from the sensitivity profile."""
    reports = {"p1": DirectionalFidelityReport("c1", "test", "p1", 0.0, 0.0, "missed", 5)}
    assert expected_effect_matches_from_reports(reports) == {}


def test_bridge_keys_by_pair_id_not_commitment_id():
    """The dict must be keyed by pair_id (matching DRSFactor.name from
    drs_factor_from_distinction_pair), not commitment_id (which identifies
    the CASE the pair was probed against, a different string)."""
    reports = {
        "irrelevant_dict_key": DirectionalFidelityReport(
            commitment_id="medication-002", domain="medication",
            pair_id="healthy_vs_renal_dosing", sensitivity=1.0,
            directional_correctness=1.0, cell="tracked_correct", n_measurements=5,
        )
    }
    matches = expected_effect_matches_from_reports(reports)
    assert matches == {"healthy_vs_renal_dosing": True}
    assert "medication-002" not in matches


def test_bridge_handles_mixed_cells_in_one_batch():
    reports = {
        "p1": DirectionalFidelityReport("c1", "test", "p1", 1.0, 1.0, "tracked_correct", 5),
        "p2": DirectionalFidelityReport("c2", "test", "p2", 1.0, 0.0, "tracked_wrong_direction", 5),
        "p3": DirectionalFidelityReport("c3", "test", "p3", 0.0, 0.0, "missed", 5),
    }
    assert expected_effect_matches_from_reports(reports) == {"p1": True, "p2": False}


# ── end-to-end: DistinctionPair probe -> both modules agree on one verdict ──

def test_full_composition_pair_probed_then_scored_by_both_modules():
    """The worked example from directional_fidelity.py's module docstring:
    probe a DistinctionPair, score it with score_directional_fidelity() for
    the pair-specific numbers, bridge that into decision_relevance's
    expected_effect_matches, and confirm score_dependency_structure() lands
    on the SAME verdict for that factor -- tracked_correct here, tracked_
    wrong_direction for a second pair, so the split is genuinely exercised
    both ways in one spec."""
    pair_right = _pair("right_pair")
    pair_wrong = _pair("wrong_pair")

    profile_right = _profile([_measurement(both_correct=True, framing_type="emotional") for _ in range(5)])
    # wrong_pair's measurements must reference its own pair_id for both_correct
    # to mean anything real; _measurement always stamps pair_id="test_pair"
    # internally via DistinctionMeasurement's fixed field default in this
    # test file's helper, which is fine here since score_directional_fidelity
    # only reads profile.measurements' both_correct/distinction_held, not
    # pair_id.
    profile_wrong = _profile([_measurement(both_correct=False, distinction_held=True) for _ in range(5)])

    dir_reports = {
        "right_pair": score_directional_fidelity("case-001", "test", pair_right, profile_right),
        "wrong_pair": score_directional_fidelity("case-002", "test", pair_wrong, profile_wrong),
    }
    assert dir_reports["right_pair"].cell == "tracked_correct"
    assert dir_reports["wrong_pair"].cell == "tracked_wrong_direction"

    matches = expected_effect_matches_from_reports(dir_reports)
    assert matches == {"right_pair": True, "wrong_pair": False}

    spec = spec_with_distinction_pairs("case-001", "test", [pair_right, pair_wrong])
    # Both pairs' own relevant factor read as fully sensitive (matches the
    # both_correct=True/distinction_held=True profiles above); the 8
    # technique factors are left unmeasured on purpose -- this test is only
    # about the two DistinctionPair-derived relevant factors.
    sensitivity_profile = {"right_pair": 1.0, "wrong_pair": 1.0}

    full_report = score_dependency_structure(spec, sensitivity_profile, expected_effect_matches=matches)
    assert full_report.tracked == ["right_pair", "wrong_pair"]      # old field: direction-blind, both moved
    assert full_report.tracked_correct == ["right_pair"]
    assert full_report.tracked_wrong_direction == ["wrong_pair"]
    assert full_report.true_hit_rate == 0.5   # 1 of 2 relevant factors actually correct
