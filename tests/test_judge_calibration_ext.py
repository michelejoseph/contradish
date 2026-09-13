"""
Tests for contradish.judge_calibration_ext.

No API key required: score_calibration_votes is the pure, deterministic
scoring core (no model calls) that every measure_*_judge_floor function
feeds into after collecting real votes -- these tests exercise that core
directly with synthetic vote patterns, plus sanity-check the shape of the
three built-in calibration sets.
"""
from contradish.judge_calibration_ext import (
    HEDGE_CALIBRATION_SET,
    RESTATEMENT_CALIBRATION_SET,
    USAGE_CALIBRATION_SET,
    score_calibration_votes,
    score_calibration_votes_by_domain,
)


def test_perfectly_consistent_and_correct_judge_scores_zero_floor_strain():
    items = [{"gold": True, "domain": "x"}, {"gold": False, "domain": "x"}]
    votes = [[True, True, True], [False, False, False]]
    results, accuracy, floor_strain = score_calibration_votes(items, votes)
    assert accuracy == 1.0
    assert floor_strain == 0.0
    assert all(r["majority_correct"] for r in results)


def test_judge_that_flips_every_rephrasing_has_high_floor_strain():
    items = [{"gold": True, "domain": "x"}]
    # 3 rephrasings, judge gives 2 different answers -> agreement = 2/3
    votes = [[True, False, True]]
    results, accuracy, floor_strain = score_calibration_votes(items, votes)
    expected_agreement = round(2 / 3, 3)
    assert results[0]["agreement"] == expected_agreement
    # floor_strain is computed from the ALREADY-ROUNDED per-item agreement
    # (matching judge_calibration.py's own round-then-average order), so the
    # expected value must go through the same two-step rounding, not the
    # unrounded fraction.
    assert floor_strain == round(1 - expected_agreement, 4)


def test_confidently_wrong_judge_has_low_floor_strain_but_low_accuracy():
    """A judge can be perfectly self-consistent (low floor_strain) while
    being wrong -- floor_strain and accuracy are independent axes."""
    items = [{"gold": True, "domain": "x"}]
    votes = [[False, False, False]]   # consistent, but disagrees with gold
    results, accuracy, floor_strain = score_calibration_votes(items, votes)
    assert floor_strain == 0.0
    assert accuracy == 0.0
    assert results[0]["majority_correct"] is False


def test_none_votes_excluded_from_agreement_not_treated_as_a_verdict():
    items = [{"gold": True, "domain": "x"}]
    votes = [[True, None, True]]   # one failed call
    results, accuracy, floor_strain = score_calibration_votes(items, votes)
    assert results[0]["agreement"] == 1.0   # 2/2 clean votes agree
    assert accuracy == 1.0


def test_all_votes_none_has_no_majority_and_counts_as_incorrect():
    items = [{"gold": True, "domain": "x"}]
    votes = [[None, None, None]]
    results, accuracy, floor_strain = score_calibration_votes(items, votes)
    assert results[0]["majority_correct"] is False
    assert accuracy == 0.0


def test_empty_items_is_well_defined_not_a_crash():
    results, accuracy, floor_strain = score_calibration_votes([], [])
    assert results == []
    assert accuracy == 0.0
    assert floor_strain == 0.0


# -- calibration set shape sanity --------------------------------------------

def _check_set_shape(calib_set, required_keys):
    assert len(calib_set) >= 6, "calibration set too small to mean anything"
    n_true = sum(1 for item in calib_set if item["gold"] is True)
    n_false = sum(1 for item in calib_set if item["gold"] is False)
    assert n_true >= 2 and n_false >= 2, "calibration set should mix both gold values"
    for item in calib_set:
        assert isinstance(item["gold"], bool)
        for key in required_keys:
            assert key in item and isinstance(item[key], str) and item[key], (
                f"item missing/empty required field {key!r}: {item}"
            )


def test_hedge_calibration_set_is_well_formed():
    _check_set_shape(HEDGE_CALIBRATION_SET, ["answer"])


def test_restatement_calibration_set_is_well_formed():
    _check_set_shape(
        RESTATEMENT_CALIBRATION_SET,
        ["label_a", "commit_a", "label_b", "commit_b", "restatement"],
    )


def test_usage_calibration_set_is_well_formed():
    _check_set_shape(USAGE_CALIBRATION_SET, ["claim_content", "probe_question", "answer"])


# ── score_calibration_votes_by_domain (IRT-lite: pooling can hide heterogeneity) ──

def test_stratifies_by_domain_and_reports_per_domain_floor_strain():
    items = [
        {"gold": True, "domain": "medication"},
        {"gold": True, "domain": "billing"},
    ]
    # medication: perfectly consistent (floor_strain 0); billing: flips every time (high floor_strain)
    votes = [[True, True, True], [True, False, True]]
    breakdown = score_calibration_votes_by_domain(items, votes)
    assert breakdown["medication"]["floor_strain"] == 0.0
    assert breakdown["billing"]["floor_strain"] > 0.0
    assert breakdown["medication"]["n"] == 1
    assert breakdown["billing"]["n"] == 1


def test_heterogeneity_is_max_minus_min_floor_strain_across_domains():
    items = [
        {"gold": True, "domain": "a"},
        {"gold": True, "domain": "b"},
    ]
    votes = [[True, True, True], [True, False, True]]
    breakdown = score_calibration_votes_by_domain(items, votes)
    expected = round(breakdown["b"]["floor_strain"] - breakdown["a"]["floor_strain"], 4)
    assert breakdown["_heterogeneity"] == expected


def test_single_domain_has_none_heterogeneity():
    items = [{"gold": True, "domain": "only"}, {"gold": False, "domain": "only"}]
    votes = [[True, True], [False, False]]
    breakdown = score_calibration_votes_by_domain(items, votes)
    assert breakdown["_heterogeneity"] is None


def test_domain_breakdown_matches_pooled_computation_for_single_domain():
    items = [{"gold": True, "domain": "x"}, {"gold": False, "domain": "x"}]
    votes = [[True, True, True], [False, False, False]]
    _, pooled_accuracy, pooled_floor = score_calibration_votes(items, votes)
    breakdown = score_calibration_votes_by_domain(items, votes)
    assert breakdown["x"]["accuracy"] == pooled_accuracy
    assert breakdown["x"]["floor_strain"] == pooled_floor


def test_missing_domain_key_defaults_to_unknown():
    items = [{"gold": True}]
    votes = [[True, True]]
    breakdown = score_calibration_votes_by_domain(items, votes)
    assert "unknown" in breakdown
