"""
Tests for contradish.minimal_intervention_delta: given intervention deltaI,
what is the smallest justified deltaB in behavior, and did the model
produce exactly that deltaB?

No API key required -- specs are built directly and sensitivity/direction
data are supplied as fixed dicts, exactly like test_decision_relevance.py's
pattern for score_dependency_structure.
"""
from contradish.decision_relevance import score_dependency_structure
from contradish.minimal_intervention_delta import (
    intervention_delta_spec,
    score_minimal_delta,
    aggregate_minimal_delta,
)


def _spec(justified=None, invariant=None):
    justified = justified if justified is not None else {
        "travel_advice": "advance parole required",
    }
    invariant = invariant if invariant is not None else [
        "fee_amount", "processing_office", "form_number",
    ]
    return intervention_delta_spec(
        intervention_id="i485-pending-2026-09",
        domain="immigration",
        justified_commitments=justified,
        invariant_commitments=invariant,
    )


# ── intervention_delta_spec ──────────────────────────────────────────────────

def test_justified_commitments_marked_relevant_invariant_marked_irrelevant():
    spec = _spec()
    assert spec.factors["travel_advice"].relevance == "relevant"
    assert spec.factors["fee_amount"].relevance == "irrelevant"
    assert spec.factors["processing_office"].relevance == "irrelevant"


def test_dict_form_carries_expected_effect_through():
    spec = _spec(justified={"travel_advice": "advance parole required"})
    assert spec.factors["travel_advice"].expected_effect == "advance parole required"


def test_list_form_leaves_expected_effect_blank():
    spec = intervention_delta_spec(
        intervention_id="x", domain="d",
        justified_commitments=["a", "b"], invariant_commitments=["c"],
    )
    assert spec.factors["a"].expected_effect == ""
    assert spec.factors["b"].relevance == "relevant"
    assert spec.factors["c"].relevance == "irrelevant"


def test_overlap_between_justified_and_invariant_raises():
    try:
        intervention_delta_spec(
            intervention_id="x", domain="d",
            justified_commitments=["a"], invariant_commitments=["a"],
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_commitment_id_is_the_intervention_id():
    spec = _spec()
    assert spec.commitment_id == "i485-pending-2026-09"
    assert spec.domain == "immigration"


# ── score_minimal_delta: membership ──────────────────────────────────────────

def test_perfect_model_is_exact_match():
    spec = _spec()
    # model changed exactly travel_advice, left everything else alone
    profile = {"travel_advice": 0.9, "fee_amount": 0.0, "processing_office": 0.0, "form_number": 0.0}
    report = score_dependency_structure(spec, profile)
    verdict = score_minimal_delta(report)
    assert verdict.exact_delta_match is True
    assert verdict.justified_delta == frozenset({"travel_advice"})
    assert verdict.actual_delta == frozenset({"travel_advice"})
    assert not verdict.excess_delta
    assert not verdict.deficit_delta


def test_excess_change_breaks_exact_match():
    spec = _spec()
    # model also changed fee_amount, which should have stayed invariant
    profile = {"travel_advice": 0.9, "fee_amount": 0.8, "processing_office": 0.0, "form_number": 0.0}
    report = score_dependency_structure(spec, profile)
    verdict = score_minimal_delta(report)
    assert verdict.exact_delta_match is False
    assert verdict.excess_delta == frozenset({"fee_amount"})
    assert not verdict.deficit_delta


def test_deficit_change_breaks_exact_match():
    spec = _spec()
    # model failed to update travel_advice at all
    profile = {"travel_advice": 0.1, "fee_amount": 0.0, "processing_office": 0.0, "form_number": 0.0}
    report = score_dependency_structure(spec, profile)
    verdict = score_minimal_delta(report)
    assert verdict.exact_delta_match is False
    assert verdict.deficit_delta == frozenset({"travel_advice"})
    assert not verdict.excess_delta


# ── score_minimal_delta: direction ───────────────────────────────────────────

def test_correct_direction_gives_full_exact_match():
    spec = _spec()
    profile = {"travel_advice": 0.9, "fee_amount": 0.0, "processing_office": 0.0, "form_number": 0.0}
    report = score_dependency_structure(
        spec, profile, expected_effect_matches={"travel_advice": True},
    )
    verdict = score_minimal_delta(report)
    assert verdict.exact_delta_match is True
    assert verdict.exact_delta_match_with_direction is True


def test_right_set_wrong_direction_is_membership_exact_but_not_full_exact():
    spec = _spec()
    profile = {"travel_advice": 0.9, "fee_amount": 0.0, "processing_office": 0.0, "form_number": 0.0}
    report = score_dependency_structure(
        spec, profile, expected_effect_matches={"travel_advice": False},
    )
    verdict = score_minimal_delta(report)
    assert verdict.exact_delta_match is True
    assert verdict.exact_delta_match_with_direction is False


def test_no_direction_data_leaves_direction_verdict_none():
    spec = _spec()
    profile = {"travel_advice": 0.9, "fee_amount": 0.0, "processing_office": 0.0, "form_number": 0.0}
    report = score_dependency_structure(spec, profile)  # no expected_effect_matches
    verdict = score_minimal_delta(report)
    assert verdict.exact_delta_match is True
    assert verdict.exact_delta_match_with_direction is None


# ── aggregation ───────────────────────────────────────────────────────────────

def test_aggregate_exact_match_rate_pools_across_interventions():
    spec = _spec()
    good = score_dependency_structure(
        spec, {"travel_advice": 0.9, "fee_amount": 0.0, "processing_office": 0.0, "form_number": 0.0},
    )
    bad = score_dependency_structure(
        spec, {"travel_advice": 0.1, "fee_amount": 0.8, "processing_office": 0.0, "form_number": 0.0},
    )
    verdicts = {
        "case-1": score_minimal_delta(good),
        "case-2": score_minimal_delta(bad),
    }
    audit = aggregate_minimal_delta("immigration", verdicts)
    assert audit.n_interventions == 2
    assert audit.exact_match_rate == 0.5
    assert audit.interventions_with_excess == ["case-2"]
    assert audit.interventions_with_deficit == ["case-2"]


def test_aggregate_direction_rate_is_none_without_any_direction_data():
    spec = _spec()
    good = score_dependency_structure(
        spec, {"travel_advice": 0.9, "fee_amount": 0.0, "processing_office": 0.0, "form_number": 0.0},
    )
    audit = aggregate_minimal_delta("immigration", {"case-1": score_minimal_delta(good)})
    assert audit.exact_match_rate_with_direction is None


def test_aggregate_flags_membership_exact_but_wrong_direction_separately():
    spec = _spec()
    wrong_dir = score_dependency_structure(
        spec, {"travel_advice": 0.9, "fee_amount": 0.0, "processing_office": 0.0, "form_number": 0.0},
        expected_effect_matches={"travel_advice": False},
    )
    audit = aggregate_minimal_delta("immigration", {"case-1": score_minimal_delta(wrong_dir)})
    assert audit.interventions_with_wrong_direction == ["case-1"]
    assert audit.interventions_with_excess == []
    assert audit.interventions_with_deficit == []


# ── summary() / to_dict() / report() smoke tests ─────────────────────────────

def test_verdict_summary_and_to_dict_do_not_crash():
    spec = _spec()
    report = score_dependency_structure(
        spec, {"travel_advice": 0.9, "fee_amount": 0.8, "processing_office": 0.0, "form_number": 0.0},
    )
    verdict = score_minimal_delta(report)
    assert "i485-pending-2026-09" in verdict.summary()
    d = verdict.to_dict()
    assert d["justified_delta"] == ["travel_advice"]
    assert d["excess_delta"] == ["fee_amount"]


def test_audit_report_and_to_dict_do_not_crash():
    spec = _spec()
    good = score_dependency_structure(
        spec, {"travel_advice": 0.9, "fee_amount": 0.0, "processing_office": 0.0, "form_number": 0.0},
    )
    audit = aggregate_minimal_delta("immigration", {"case-1": score_minimal_delta(good)})
    assert "immigration" in audit.report()
    d = audit.to_dict()
    assert d["n_interventions"] == 1
    assert "case-1" in d["by_intervention"]
