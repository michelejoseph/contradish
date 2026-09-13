"""
Tests for contradish.faithfulness: truth-over-coherence, as a number.

No API key required -- DistinctionLossMap/DistinctionProfile objects are
built directly with fixed values instead of measured from a model, and a
custom junction_case_map is passed explicitly so the test doesn't depend on
the real medication.json content.
"""
from contradish.distinction import DistinctionLossMap, DistinctionProfile
from contradish.faithfulness import score_faithfulness


def _make_loss_map(overall_hold_rate: float, pair_id: str = "test_pair") -> DistinctionLossMap:
    profile = DistinctionProfile(
        pair_id=pair_id, description="test distinction", label_a="a", label_b="b",
        hold_rate_per_framing={}, overall_hold_rate=overall_hold_rate,
        collapse_framing="", first_collapse=None, measurements=[],
    )
    return DistinctionLossMap(
        domain="test", profiles={pair_id: profile},
        ranked_by_fragility=[pair_id], most_fragile=pair_id,
        most_resilient=pair_id, framing_destructiveness={},
    )


def test_perfect_faithfulness_is_one():
    loss_map = _make_loss_map(overall_hold_rate=1.0)
    details = [{"id": "case-1", "cai_strain": 0.0, "passed": True}]
    report = score_faithfulness("test", loss_map, details, junction_case_map={"test_pair": ["case-1"]})
    j = report.junctions["test_pair"]
    assert j.relevant_sensitivity == 1.0
    assert j.irrelevant_sensitivity == 0.0
    assert j.faithfulness == 1.0
    assert report.mean_faithfulness == 1.0
    assert report.most_faithful == "test_pair"


def test_backwards_model_scores_negative_one():
    """Ignores real distinctions (hold_rate=0) AND is rattled by irrelevant
    phrasing (cai_strain=1.0) -- the exact-backwards signature the module
    docstring calls out as the worst case."""
    loss_map = _make_loss_map(overall_hold_rate=0.0)
    details = [{"id": "case-1", "cai_strain": 1.0, "passed": False}]
    report = score_faithfulness("test", loss_map, details, junction_case_map={"test_pair": ["case-1"]})
    j = report.junctions["test_pair"]
    assert j.faithfulness == -1.0


def test_unmapped_pair_is_reported_not_silently_scored():
    loss_map = _make_loss_map(overall_hold_rate=0.5)
    report = score_faithfulness("test", loss_map, [], junction_case_map={})
    assert report.junctions == {}
    assert report.unmapped_pairs == ["test_pair"]
    assert report.mean_faithfulness is None
    assert report.most_faithful == ""


def test_pair_mapped_to_missing_case_id_is_unmapped_not_scored_as_zero():
    loss_map = _make_loss_map(overall_hold_rate=0.5)
    report = score_faithfulness("test", loss_map, [], junction_case_map={"test_pair": ["nonexistent-case"]})
    assert "test_pair" in report.unmapped_pairs
    assert "test_pair" not in report.junctions


def test_multiple_mapped_cases_are_averaged_not_just_first():
    loss_map = _make_loss_map(overall_hold_rate=0.8)
    details = [
        {"id": "case-1", "cai_strain": 0.2, "passed": True},
        {"id": "case-2", "cai_strain": 0.6, "passed": False},
    ]
    report = score_faithfulness(
        "test", loss_map, details,
        junction_case_map={"test_pair": ["case-1", "case-2"]},
    )
    j = report.junctions["test_pair"]
    assert j.irrelevant_sensitivity == 0.4
    assert j.faithfulness == round(0.8 - 0.4, 4)
    assert j.case_strains == {"case-1": 0.2, "case-2": 0.6}


def test_defaults_to_real_junction_case_map_when_none_passed():
    """Without an explicit junction_case_map, the real one from
    predictive_validity.py is used -- so a pair_id that DOES exist there
    (healthy_vs_renal_dosing) should resolve, not end up unmapped."""
    loss_map = _make_loss_map(overall_hold_rate=0.5, pair_id="healthy_vs_renal_dosing")
    details = [{"id": "medication-002", "cai_strain": 0.25, "passed": True}]
    report = score_faithfulness("medication", loss_map, details)
    assert "healthy_vs_renal_dosing" in report.junctions
    assert report.junctions["healthy_vs_renal_dosing"].case_ids == ["medication-002"]
