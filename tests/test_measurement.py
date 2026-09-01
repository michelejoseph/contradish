"""
Tests for contradish.measurement — the reasoning measurement science module
(dimension taxonomy, measurement laws, uncertainty model, ReasoningProfile,
compare(), and profile_from_results()).
Run with: pytest tests/test_measurement.py
No API key required: everything here is pure Python over plain data.
"""
import math

import pytest

from contradish.measurement import (
    DIMENSIONS, LAWS, MeasurementUncertainty, ReasoningProfile,
    _pearson, compare, profile_from_results,
)


# ── Dimension taxonomy ──────────────────────────────────────────────────────

def test_dimensions_symbols_are_unique():
    symbols = [d.symbol for d in DIMENSIONS.values()]
    assert len(symbols) == len(set(symbols))


def test_dimensions_keys_match_dict():
    # Every DIMENSIONS entry is keyed by a snake_case identifier with a name/symbol.
    for key, dim in DIMENSIONS.items():
        assert dim.name
        assert dim.symbol
        assert dim.range[0] <= dim.range[1]
        assert dim.higher_is in ("better", "worse", "domain-dependent")
        assert dim.stability in ("stable", "experiment", "derived")


def test_dimensions_ideal_within_range_when_set():
    for dim in DIMENSIONS.values():
        if dim.ideal is not None:
            assert dim.range[0] <= dim.ideal <= dim.range[1]


def test_load_bearing_weight_has_no_ideal():
    # It's a structural property, not a performance target.
    assert DIMENSIONS["load_bearing_weight"].ideal is None


# ── LAWS sanity ──────────────────────────────────────────────────────────────

def test_laws_have_unique_names():
    names = [law.name for law in LAWS]
    assert len(names) == len(set(names))


def test_laws_are_callable_and_return_tuple():
    p = ReasoningProfile(model="m")
    for law in LAWS:
        result = law.validate(p)
        assert isinstance(result, tuple) and len(result) == 2
        passed, message = result
        assert isinstance(passed, bool)
        assert isinstance(message, str) and message


def test_falsifiable_flags_match_known_predictive_laws():
    falsifiable = {law.name for law in LAWS if law.falsifiable}
    assert falsifiable == {
        "Convergence Monotonicity",
        "Local Admissibility Trap",
        "Convergence Principle",
    }


# ── _pearson ──────────────────────────────────────────────────────────────────

def test_pearson_perfect_positive_correlation():
    pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert _pearson(pairs) == pytest.approx(1.0)


def test_pearson_perfect_negative_correlation():
    pairs = [(1, 4), (2, 3), (3, 2), (4, 1)]
    assert _pearson(pairs) == pytest.approx(-1.0)


def test_pearson_too_few_points_returns_zero():
    assert _pearson([(1, 1), (2, 2)]) == 0.0


def test_pearson_degenerate_constant_x_returns_zero():
    assert _pearson([(1, 1), (1, 2), (1, 3)]) == 0.0


# ── Law: Positivity ────────────────────────────────────────────────────────────

def test_law_positivity_passes_within_bounds():
    law = next(l for l in LAWS if l.name == "Positivity")
    p = ReasoningProfile(model="m", cai_strain=0.2, reality_strain=0.3,
                         admissibility_distance=0.25)
    passed, msg = law.validate(p)
    assert passed and "[0, 1]" in msg


def test_law_positivity_fails_out_of_bounds():
    law = next(l for l in LAWS if l.name == "Positivity")
    p = ReasoningProfile(model="m", cai_strain=1.4, reality_strain=0.3,
                         admissibility_distance=0.25)
    passed, msg = law.validate(p)
    assert not passed and "ε_c" in msg


# ── Law: Admissibility Triangle ────────────────────────────────────────────────

def test_law_triangle_passes_when_consistent():
    law = next(l for l in LAWS if l.name == "Admissibility Triangle")
    p = ReasoningProfile(model="m", cai_strain=0.4, reality_strain=0.2, alpha=0.5)
    p.compute_admissibility_distance()
    passed, _ = law.validate(p)
    assert passed


def test_law_triangle_fails_when_da_exceeds_expected():
    law = next(l for l in LAWS if l.name == "Admissibility Triangle")
    p = ReasoningProfile(model="m", cai_strain=0.2, reality_strain=0.2,
                         admissibility_distance=0.9, alpha=0.5)
    passed, msg = law.validate(p)
    assert not passed and "judge inconsistency" in msg


def test_law_triangle_insufficient_data():
    law = next(l for l in LAWS if l.name == "Admissibility Triangle")
    passed, msg = law.validate(ReasoningProfile(model="m"))
    assert passed and "insufficient data" in msg


# ── Law: Fixed Point ───────────────────────────────────────────────────────────

def test_law_fixed_point_flags_computation_error():
    law = next(l for l in LAWS if l.name == "Fixed Point")
    p = ReasoningProfile(model="m", cai_strain=0.01, reality_strain=0.01,
                         admissibility_distance=0.5)
    passed, msg = law.validate(p)
    assert not passed and "computation error" in msg


def test_law_fixed_point_ok_when_all_near_zero():
    law = next(l for l in LAWS if l.name == "Fixed Point")
    p = ReasoningProfile(model="m", cai_strain=0.01, reality_strain=0.01,
                         admissibility_distance=0.01)
    passed, _ = law.validate(p)
    assert passed


# ── Law: Judge Independence ────────────────────────────────────────────────────

def test_law_judge_independence_fails_same_provider():
    law = next(l for l in LAWS if l.name == "Judge Independence")
    p = ReasoningProfile(model="m", model_provider="openai", judge_provider="openai")
    passed, msg = law.validate(p)
    assert not passed and "Self-preference bias" in msg


def test_law_judge_independence_passes_different_provider():
    law = next(l for l in LAWS if l.name == "Judge Independence")
    p = ReasoningProfile(model="m", model_provider="openai", judge_provider="anthropic")
    passed, _ = law.validate(p)
    assert passed


def test_law_judge_independence_passes_when_unrecorded():
    law = next(l for l in LAWS if l.name == "Judge Independence")
    passed, msg = law.validate(ReasoningProfile(model="m"))
    assert passed and "not recorded" in msg


# ── Law: Awareness Covariance ──────────────────────────────────────────────────

def test_law_awareness_covariance_flags_dangerous_quadrant():
    law = next(l for l in LAWS if l.name == "Awareness Covariance")
    p = ReasoningProfile(model="m", cai_strain=0.5, csa_score=0.2)
    passed, msg = law.validate(p)
    assert not passed and "Dangerous configuration" in msg


def test_law_awareness_covariance_ok_when_aware():
    law = next(l for l in LAWS if l.name == "Awareness Covariance")
    p = ReasoningProfile(model="m", cai_strain=0.5, csa_score=0.8)
    passed, _ = law.validate(p)
    assert passed


# ── Law: Convergence Monotonicity ──────────────────────────────────────────────

def test_law_convergence_monotonicity_prediction_only_without_data():
    law = next(l for l in LAWS if l.name == "Convergence Monotonicity")
    passed, msg = law.validate(ReasoningProfile(model="m"))
    assert passed and "PREDICTION ONLY" in msg


def test_law_convergence_monotonicity_passes_when_monotone():
    law = next(l for l in LAWS if l.name == "Convergence Monotonicity")
    p = ReasoningProfile(model="m", domain_case_scores=[
        {"id": "a", "load_bearing_weight": 0.1, "convergence_order": 1},
        {"id": "b", "load_bearing_weight": 0.5, "convergence_order": 2},
        {"id": "c", "load_bearing_weight": 0.9, "convergence_order": 3},
    ])
    passed, msg = law.validate(p)
    assert passed and "monotone" in msg


def test_law_convergence_monotonicity_fails_on_violation():
    law = next(l for l in LAWS if l.name == "Convergence Monotonicity")
    p = ReasoningProfile(model="m", domain_case_scores=[
        {"id": "a", "load_bearing_weight": 0.1, "convergence_order": 3},
        {"id": "b", "load_bearing_weight": 0.9, "convergence_order": 1},
    ])
    passed, msg = law.validate(p)
    assert not passed and "Monotonicity violated" in msg


# ── Law: Strain Gradient Direction ─────────────────────────────────────────────

def test_law_strain_gradient_dangerous_when_positive():
    law = next(l for l in LAWS if l.name == "Strain Gradient Direction")
    p = ReasoningProfile(model="m", strain_gradient=0.5)
    passed, msg = law.validate(p)
    assert not passed and "Primary failure mode" in msg


def test_law_strain_gradient_benign_when_negative():
    law = next(l for l in LAWS if l.name == "Strain Gradient Direction")
    p = ReasoningProfile(model="m", strain_gradient=-0.4)
    passed, _ = law.validate(p)
    assert passed


def test_law_strain_gradient_insufficient_data():
    law = next(l for l in LAWS if l.name == "Strain Gradient Direction")
    passed, msg = law.validate(ReasoningProfile(model="m"))
    assert passed and "insufficient data" in msg


# ── Law: Local Admissibility Trap ──────────────────────────────────────────────

def test_law_trap_prediction_only_without_frustration():
    law = next(l for l in LAWS if l.name == "Local Admissibility Trap")
    passed, msg = law.validate(ReasoningProfile(model="m"))
    assert passed and "PREDICTION ONLY" in msg


def test_law_trap_detected_when_converged_and_frustrated():
    law = next(l for l in LAWS if l.name == "Local Admissibility Trap")
    p = ReasoningProfile(model="m", frustration=0.7, reality_strain=0.3,
                         repair_efficiency=0.6)
    passed, msg = law.validate(p)
    assert not passed and "LOCAL ADMISSIBILITY TRAP DETECTED" in msg


def test_law_trap_risk_without_confirmed_repair_efficiency():
    law = next(l for l in LAWS if l.name == "Local Admissibility Trap")
    p = ReasoningProfile(model="m", frustration=0.7, reality_strain=0.3)
    passed, msg = law.validate(p)
    assert not passed and "TRAP RISK" in msg


def test_law_trap_frustrated_but_low_error_is_ok():
    law = next(l for l in LAWS if l.name == "Local Admissibility Trap")
    p = ReasoningProfile(model="m", frustration=0.7, reality_strain=0.05)
    passed, msg = law.validate(p)
    assert passed and "close to the global fixed point" in msg


def test_law_trap_aligned_when_negative_frustration():
    law = next(l for l in LAWS if l.name == "Local Admissibility Trap")
    p = ReasoningProfile(model="m", frustration=-0.3)
    passed, msg = law.validate(p)
    assert passed and "aligned" in msg


def test_law_trap_near_independent_default_message():
    law = next(l for l in LAWS if l.name == "Local Admissibility Trap")
    p = ReasoningProfile(model="m", frustration=0.1)
    passed, msg = law.validate(p)
    assert passed and "near-independent" in msg


# ── Law: Convergence Principle ─────────────────────────────────────────────────

def test_law_convergence_principle_prediction_only():
    law = next(l for l in LAWS if l.name == "Convergence Principle")
    passed, msg = law.validate(ReasoningProfile(model="m"))
    assert passed and "PREDICTION ONLY" in msg


def test_law_convergence_principle_needs_per_domain_data():
    law = next(l for l in LAWS if l.name == "Convergence Principle")
    p = ReasoningProfile(model="m", inquiry_convergence=0.6)
    passed, msg = law.validate(p)
    assert passed and "per-domain ψ needed" in msg


def test_law_convergence_principle_confirmed_pattern():
    law = next(l for l in LAWS if l.name == "Convergence Principle")
    p = ReasoningProfile(model="m", inquiry_convergence=0.6, domain_inquiry_convergence={
        "hard": 0.8, "harder": 0.75, "easy": 0.3, "easier": 0.2,
    })
    passed, msg = law.validate(p)
    assert passed and "consistent with Convergence Principle" in msg


def test_law_convergence_principle_split_by_own_threshold_always_confirms():
    # The law buckets domains by their OWN psi value around 0.6 and then compares
    # bucket means, so the >0.6 bucket's mean is guaranteed to exceed the <=0.6
    # bucket's mean by construction — the "inconsistent" branch is unreachable
    # from any input. This test documents that observed behavior.
    law = next(l for l in LAWS if l.name == "Convergence Principle")
    p = ReasoningProfile(model="m", inquiry_convergence=0.6, domain_inquiry_convergence={
        "hard": 0.3, "harder": 0.2, "easy": 0.9, "easier": 0.8,
    })
    passed, msg = law.validate(p)
    assert passed and "consistent with Convergence Principle" in msg


# ── MeasurementUncertainty ─────────────────────────────────────────────────────

def test_uncertainty_estimate_defaults_to_industry_judge_floor():
    u = MeasurementUncertainty.estimate()
    assert u.judge_variance == 0.035
    assert u.paraphrase_spread == 0.0
    assert u.staleness_fraction == 0.0
    assert u.combined == pytest.approx(0.035, abs=1e-4)
    assert u.dominant_source == "judge_variance"


def test_uncertainty_estimate_paraphrase_spread_computed():
    u = MeasurementUncertainty.estimate(judge_floor=0.0, per_variant_scores=[0.2, 0.4, 0.6, 0.8])
    assert u.paraphrase_spread > 0
    assert u.combined == pytest.approx(u.paraphrase_spread, abs=1e-3)


def test_uncertainty_estimate_staleness_fraction():
    entries = [{"year_valid": 2020}, {"year_valid": 2020}, {"year_valid": 2099}]
    u = MeasurementUncertainty.estimate(judge_floor=0.0, ground_truth_entries=entries,
                                        current_year=2026)
    # 2 of 3 entries are stale, contributing up to 0.15 max
    assert u.staleness_fraction == pytest.approx((2 / 3) * 0.15, abs=1e-4)


def test_uncertainty_combined_is_root_sum_square():
    u = MeasurementUncertainty.estimate(judge_floor=0.03, per_variant_scores=[0.1, 0.9])
    expected = math.sqrt(u.judge_variance ** 2 + u.paraphrase_spread ** 2 + u.staleness_fraction ** 2)
    assert u.combined == pytest.approx(round(expected, 4), abs=1e-4)


def test_uncertainty_note_thresholds():
    low = MeasurementUncertainty.estimate(judge_floor=0.01)
    assert "Low uncertainty" in low.note

    high = MeasurementUncertainty.estimate(judge_floor=0.2)
    assert "High uncertainty" in high.note


def test_uncertainty_minimum_detectable_difference():
    u = MeasurementUncertainty(combined=0.04)
    assert u.minimum_detectable_difference() == 0.08


# ── ReasoningProfile: computed dimensions ──────────────────────────────────────

def test_compute_admissibility_distance():
    p = ReasoningProfile(model="m", cai_strain=0.4, reality_strain=0.2, alpha=0.25)
    d = p.compute_admissibility_distance()
    assert d == pytest.approx(0.25 * 0.4 + 0.75 * 0.2)
    assert p.admissibility_distance == d


def test_compute_admissibility_distance_none_when_missing_component():
    p = ReasoningProfile(model="m", cai_strain=0.4)
    assert p.compute_admissibility_distance() is None


def test_compute_frustration_from_per_case_data():
    p = ReasoningProfile(model="m", domain_case_scores=[
        {"id": f"c{i}", "reality_strain": rs} for i, rs in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    ])
    per_case_cai = {f"c{i}": ec for i, ec in enumerate([0.9, 0.7, 0.5, 0.3, 0.1])}
    gamma = p.compute_frustration(per_case_cai)
    # ec and er are perfectly anti-correlated -> pearson(ec, er) = -1 -> gamma = -(-1) = 1
    assert gamma == pytest.approx(1.0)
    assert p.frustration == gamma


def test_compute_frustration_insufficient_pairs_returns_none():
    p = ReasoningProfile(model="m", domain_case_scores=[
        {"id": "a", "reality_strain": 0.5},
        {"id": "b", "reality_strain": 0.6},
    ])
    assert p.compute_frustration({"a": 0.1, "b": 0.2}) is None


def test_compute_frustration_no_data_returns_none():
    assert ReasoningProfile(model="m").compute_frustration() is None


def test_compute_strain_gradient_positive_slope():
    p = ReasoningProfile(model="m", domain_case_scores=[
        {"load_bearing_weight": 0.1, "reality_strain": 0.1},
        {"load_bearing_weight": 0.5, "reality_strain": 0.5},
        {"load_bearing_weight": 0.9, "reality_strain": 0.9},
    ])
    slope = p.compute_strain_gradient()
    assert slope == pytest.approx(1.0)
    assert p.strain_gradient == slope


def test_compute_strain_gradient_insufficient_cases_returns_none():
    p = ReasoningProfile(model="m", domain_case_scores=[
        {"load_bearing_weight": 0.1, "reality_strain": 0.1},
    ])
    assert p.compute_strain_gradient() is None


def test_compute_strain_gradient_zero_variance_lbw_returns_zero():
    p = ReasoningProfile(model="m", domain_case_scores=[
        {"load_bearing_weight": 0.5, "reality_strain": 0.1},
        {"load_bearing_weight": 0.5, "reality_strain": 0.5},
        {"load_bearing_weight": 0.5, "reality_strain": 0.9},
    ])
    assert p.compute_strain_gradient() == 0.0


# ── ReasoningProfile: law validation / narrative helpers ───────────────────────

def test_validate_laws_returns_one_result_per_law():
    p = ReasoningProfile(model="m")
    results = p.validate_laws()
    assert len(results) == len(LAWS)
    for r in results:
        assert set(r.keys()) == {"law", "formula", "passed", "message", "falsifiable"}


def test_position_labels():
    def pos(d):
        p = ReasoningProfile(model="m", cai_strain=0.1, reality_strain=0.1,
                             admissibility_distance=d)
        return p.position()

    assert pos(0.05) == "near the joint fixed point"
    assert pos(0.20) == "moderate distance from fixed point"
    assert pos(0.40) == "significant distance from fixed point"
    assert pos(0.80) == "far from the joint fixed point"


def test_position_unknown_without_measurements():
    p = ReasoningProfile(model="m")
    assert "unknown" in p.position()


def test_primary_failure_mode_none_near_fixed_point():
    p = ReasoningProfile(model="m", cai_strain=0.05, reality_strain=0.05)
    assert "near the joint fixed point" in p.primary_failure_mode()


def test_primary_failure_mode_consistency():
    p = ReasoningProfile(model="m", cai_strain=0.6, reality_strain=0.1)
    assert "consistency" in p.primary_failure_mode()


def test_primary_failure_mode_correctness():
    p = ReasoningProfile(model="m", cai_strain=0.1, reality_strain=0.6)
    assert "correctness" in p.primary_failure_mode()


def test_primary_failure_mode_joint():
    p = ReasoningProfile(model="m", cai_strain=0.3, reality_strain=0.35)
    assert p.primary_failure_mode() == "joint (both consistency and correctness impaired)"


def test_repair_target_cai_reduction():
    p = ReasoningProfile(model="m", cai_strain=0.6, reality_strain=0.1)
    assert "CAI Strain reduction" in p.repair_target()


def test_repair_target_reality_reduction_names_worst_domain():
    p = ReasoningProfile(model="m", cai_strain=0.1, reality_strain=0.6,
                         domain_reality_strains={"refunds": 0.8, "shipping": 0.2})
    target = p.repair_target()
    assert "Reality Strain reduction in refunds" in target


def test_repair_target_joint_when_balanced():
    p = ReasoningProfile(model="m", cai_strain=0.3, reality_strain=0.3)
    assert p.repair_target() == "joint admissibility distance — both components require attention"


def test_interpret_smoke_test_minimal_profile():
    p = ReasoningProfile(model="m")
    text = p.interpret()
    assert "position unknown" in text or "The system is" in text


def test_interpret_reports_key_fields_full_profile():
    p = ReasoningProfile(
        model="gpt-x", cai_strain=0.3, reality_strain=0.25, alpha=0.5,
        domain_reality_strains={"refunds": 0.4, "shipping": 0.1},
        csa_score=0.3, strain_gradient=0.4, frustration=0.6,
        repair_efficiency=0.5, basin_depth=0.8, sag=0.1,
        inquiry_convergence=0.8, discovery_order_alignment=-0.2,
        trajectory_monotonicity=0.4,
    )
    p.compute_admissibility_distance()
    p.uncertainty = MeasurementUncertainty.estimate()
    text = p.interpret()
    assert "drifted_unaware" in text
    assert "structurally dangerous configuration" in text
    assert "LOCAL ADMISSIBILITY TRAP" in text or "Local admissibility trap" in text
    assert "Deep basin" in text
    assert "spontaneous divergence" in text
    assert "strong cross-system convergence" in text
    assert "inverted discovery order" in text
    assert "oscillating trajectory" in text
    assert "Primary repair target" in text
    assert "Measurement uncertainty" in text


def test_to_dict_shape():
    p = ReasoningProfile(model="m", cai_strain=0.2, reality_strain=0.3)
    p.compute_admissibility_distance()
    d = p.to_dict()
    assert d["model"] == "m"
    assert "dimensions" in d and d["dimensions"]["cai_strain"] == 0.2
    assert "interpretation" in d and isinstance(d["interpretation"], str)


def test_str_and_format_report_no_crash_minimal():
    p = ReasoningProfile(model="m")
    text = str(p)
    assert "contradish" in text
    assert "Reasoning Profile" in text
    assert "LAW VALIDATION" in text


def test_format_report_full_profile_contains_sections():
    p = ReasoningProfile(
        model="m", model_provider="anthropic", judge_provider="openai",
        judge_model="gpt-judge", cai_strain=0.3, reality_strain=0.2,
        domain_reality_strains={"refunds": 0.5},
        csa_score=0.7, ctr_score=0.6, sra_score=0.5, rqs=0.4,
        repair_efficiency=0.6, strain_gradient=0.3,
        frustration=0.2, basin_depth=0.4,
        sag=0.0, inquiry_convergence=0.5,
        discovery_order_alignment=0.5, trajectory_monotonicity=0.9,
        domain_case_scores=[
            {"id": "c1", "load_bearing_weight": 0.2},
            {"id": "c2", "load_bearing_weight": 0.8},
        ],
    )
    p.compute_admissibility_distance()
    p.uncertainty = MeasurementUncertainty.estimate()
    text = str(p)
    assert "cross-provider" in text
    assert "BY DOMAIN" in text
    assert "METACOGNITIVE DIMENSIONS" in text
    assert "GEOMETRY" in text
    assert "PROCESS DIMENSIONS" in text
    assert "STRUCTURAL PREDICTION" in text
    assert "MEASUREMENT UNCERTAINTY" in text
    assert "INTERPRETATION" in text


def test_format_report_same_provider_warns():
    p = ReasoningProfile(model="m", model_provider="openai", judge_provider="openai")
    assert "SAME-PROVIDER" in str(p)


# ── compare() ───────────────────────────────────────────────────────────────

def test_compare_detects_better_model_on_lower_is_better_dim():
    a = ReasoningProfile(model="A", cai_strain=0.1, reality_strain=0.1)
    b = ReasoningProfile(model="B", cai_strain=0.5, reality_strain=0.1)
    a.compute_admissibility_distance(); b.compute_admissibility_distance()
    result = compare(a, b)
    row = next(r for r in result["dimensions"] if r["dimension"] == "cai_strain")
    assert row["detectable"] is True
    assert row["better"] == "A"


def test_compare_equivalent_within_uncertainty():
    a = ReasoningProfile(model="A", cai_strain=0.30, reality_strain=0.1)
    b = ReasoningProfile(model="B", cai_strain=0.31, reality_strain=0.1)
    result = compare(a, b)
    row = next(r for r in result["dimensions"] if r["dimension"] == "cai_strain")
    assert row["detectable"] is False
    assert row["better"] == "equivalent"


def test_compare_insufficient_data_dimension():
    a = ReasoningProfile(model="A", cai_strain=0.1, reality_strain=0.1)
    b = ReasoningProfile(model="B", cai_strain=0.1, reality_strain=0.1)
    result = compare(a, b)
    row = next(r for r in result["dimensions"] if r["dimension"] == "csa_score")
    assert row["status"] == "insufficient data"


def test_compare_overall_winner_closer_to_fixed_point():
    a = ReasoningProfile(model="A", cai_strain=0.05, reality_strain=0.05)
    b = ReasoningProfile(model="B", cai_strain=0.5, reality_strain=0.5)
    a.compute_admissibility_distance(); b.compute_admissibility_distance()
    result = compare(a, b)
    assert "A is closer to the joint fixed point" in result["overall"]


def test_compare_overall_insufficient_data():
    a = ReasoningProfile(model="A")
    b = ReasoningProfile(model="B")
    result = compare(a, b)
    assert result["overall"] == "insufficient data for overall comparison"


# ── profile_from_results() ─────────────────────────────────────────────────────

def test_profile_from_results_builds_expected_shape():
    cai_results = {"judgment_strain": 0.234567}
    reality_results = {
        "reality_strain": 0.111,
        "domain_results": [
            {
                "domain": "refunds",
                "domain_reality_strain": 0.2,
                "cases": [
                    {"id": "c1", "load_bearing_weight": 0.3, "reality_strain": 0.1,
                     "truth_score": 0.9},
                    {"id": "c2", "load_bearing_weight": 0.7, "reality_strain": 0.4,
                     "truth_score": 0.5},
                    {"id": "c3", "load_bearing_weight": 0.5, "reality_strain": 0.25,
                     "truth_score": 0.7},
                ],
            },
        ],
    }
    p = profile_from_results(
        model="gpt-x", model_provider="openai", judge_provider="anthropic",
        judge_model="claude", cai_results=cai_results, reality_results=reality_results,
        alpha=0.4, judge_floor=0.02,
    )
    assert p.cai_strain == 0.2346
    assert p.reality_strain == 0.111
    assert p.domain_reality_strains == {"refunds": 0.2}
    assert len(p.domain_case_scores) == 3
    assert p.admissibility_distance == pytest.approx(0.4 * 0.2346 + 0.6 * 0.111, abs=1e-4)
    assert p.strain_gradient is not None
    assert p.uncertainty is not None and p.uncertainty.paraphrase_spread > 0


def test_profile_from_results_handles_missing_data():
    p = profile_from_results(model="bare-model")
    assert p.cai_strain is None
    assert p.reality_strain is None
    assert p.admissibility_distance is None
    assert p.domain_reality_strains == {}
    assert p.uncertainty is not None  # still computed, from defaults


def test_profile_from_results_falls_back_to_cai_strain_key():
    p = profile_from_results(model="m", cai_results={"cai_strain": 0.5})
    assert p.cai_strain == 0.5
