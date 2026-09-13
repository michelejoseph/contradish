"""
Tests for contradish.behavioral_mapping.

No API key required: every oracle here is a deterministic Python callable
(same swappable-judge pattern as decision_boundary.py/decision_relevance.py),
so discovery, mapping, and comparison are all exercised directly.
"""
from contradish.behavioral_mapping import (
    BehavioralDependencyMap,
    CandidateVariable,
    NormativeComparisonReport,
    NormativeStructure,
    ScreeningResult,
    build_behavioral_map,
    compare_to_normative_structure,
    default_candidate_pool,
    default_normative_structure,
    screen_candidates,
)
from contradish.decision_boundary import BoundaryLadder
from contradish.decision_relevance import DecisionRelevanceSpec, DRSFactor


# ── CandidateVariable ──────────────────────────────────────────────────────

def test_invalid_kind_is_rejected():
    try:
        CandidateVariable(name="x", kind="nominal")
        assert False, "expected ValueError for an unrecognized kind"
    except ValueError:
        pass


def test_ordinal_without_ladder_is_rejected():
    try:
        CandidateVariable(name="x", kind="ordinal")
        assert False, "expected ValueError for an ordinal candidate with no ladder"
    except ValueError:
        pass


def test_categorical_probe_key_is_just_the_name():
    c = CandidateVariable(name="emotional", kind="categorical")
    assert c.probe_key() == "emotional"


def test_ordinal_probe_key_defaults_to_far_end():
    c = CandidateVariable(name="days_early", kind="ordinal", ladder=list(range(11)))
    assert c.probe_key() == "days_early::10"
    assert c.probe_key(rung=3) == "days_early::3"


def test_default_candidate_pool_is_seeded_from_known_techniques():
    from contradish.prompt_analyzer import KNOWN_TECHNIQUES
    pool = default_candidate_pool()
    assert {c.name for c in pool} == set(KNOWN_TECHNIQUES)
    assert all(c.kind == "categorical" for c in pool)


# ── screen_candidates (DISCOVER) ─────────────────────────────────────────────

def _oracle_from_map(baseline, moved_by):
    """moved_by: set of candidate names whose probe should differ from baseline."""
    def oracle(key):
        if key == "__baseline__":
            return baseline
        name = key.split("::")[0]
        return "moved" if name in moved_by else baseline
    return oracle


def test_screening_flags_only_candidates_that_actually_moved():
    candidates = [CandidateVariable(name=n, kind="categorical") for n in ("a", "b", "c")]
    oracle = _oracle_from_map("base", moved_by={"b"})
    result = screen_candidates(oracle, candidates)
    assert result.sensitive == ["b"]
    assert result.insensitive == ["a", "c"]
    assert result.baseline_decision == "base"


def test_screening_query_count_is_one_plus_n_candidates():
    candidates = [CandidateVariable(name=n, kind="categorical") for n in ("a", "b", "c", "d")]
    oracle = _oracle_from_map("base", moved_by=set())
    result = screen_candidates(oracle, candidates)
    assert result.n_queries == 5   # 1 baseline + 4 candidates


def test_screening_probes_far_end_of_ordinal_ladder():
    calls = []
    def oracle(key):
        calls.append(key)
        if key == "__baseline__":
            return "a"
        return "b"   # far end always differs from baseline here
    candidate = CandidateVariable(name="days_early", kind="ordinal", ladder=list(range(11)))
    result = screen_candidates(oracle, [candidate])
    assert "days_early::10" in calls
    assert result.sensitive == ["days_early"]


# ── build_behavioral_map (MAP) ────────────────────────────────────────────────

def test_categorical_sensitive_candidate_gets_no_boundary_recovery():
    candidates = [CandidateVariable(name="emotional", kind="categorical")]
    oracle = _oracle_from_map("base", moved_by={"emotional"})
    bmap = build_behavioral_map(oracle, candidates)
    assert bmap.sensitivity_profile["emotional"] == 1.0
    assert bmap.boundary_recoveries == {}


def test_ordinal_sensitive_candidate_gets_a_boundary_located():
    def oracle(key):
        if key == "__baseline__":
            return "approve"
        name, idx = key.split("::")
        idx = int(idx)
        return "approve" if idx < 4 else "deny"

    candidate = CandidateVariable(name="days_early", kind="ordinal", ladder=list(range(10)))
    bmap = build_behavioral_map(oracle, [candidate])
    assert bmap.sensitivity_profile["days_early"] == 1.0
    recovery = bmap.boundary_recoveries["days_early"]
    assert recovery.recovered_boundary_index == 4


def test_ordinal_insensitive_candidate_is_not_boundary_searched():
    def oracle(key):
        return "approve"   # never changes, including at the far end
    candidate = CandidateVariable(name="days_early", kind="ordinal", ladder=list(range(10)))
    bmap = build_behavioral_map(oracle, [candidate])
    assert bmap.sensitivity_profile["days_early"] == 0.0
    assert "days_early" not in bmap.boundary_recoveries


def test_map_summary_and_to_dict_render():
    candidates = [CandidateVariable(name="emotional", kind="categorical")]
    oracle = _oracle_from_map("base", moved_by={"emotional"})
    bmap = build_behavioral_map(oracle, candidates)
    assert isinstance(bmap.summary(), str)
    d = bmap.to_dict()
    assert d["sensitivity_profile"]["emotional"] == 1.0


# ── default_normative_structure ──────────────────────────────────────────────

def test_default_normative_structure_has_eight_factor_spec_no_boundaries():
    normative = default_normative_structure("c1", domain="medication")
    assert len(normative.relevance_spec.factors) == 8
    assert normative.boundaries == {}


# ── compare_to_normative_structure (COMPARE) ─────────────────────────────────

def test_compare_scores_categorical_sensitivity_against_spec():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "emotional": DRSFactor(name="emotional", relevance="irrelevant"),
        "fact": DRSFactor(name="fact", relevance="relevant"),
    })
    normative = NormativeStructure(relevance_spec=spec)
    candidates = [
        CandidateVariable(name="emotional", kind="categorical"),
        CandidateVariable(name="fact", kind="categorical"),
    ]
    oracle = _oracle_from_map("base", moved_by={"emotional"})   # reacts to emotional (bad), not fact (bad)
    bmap = build_behavioral_map(oracle, candidates)
    report = compare_to_normative_structure(bmap, normative)
    assert report.dependency_report.spurious == ["emotional"]
    assert report.dependency_report.missed == ["fact"]
    assert report.unspecified_sensitive_variables == []


def test_compare_surfaces_sensitive_variable_missing_from_spec():
    """The key new failure mode: a candidate screens sensitive but the
    normative structure never mentions it at all -- must not be silently
    dropped just because score_dependency_structure() only looks at
    spec.factors."""
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "fact": DRSFactor(name="fact", relevance="relevant"),
    })
    normative = NormativeStructure(relevance_spec=spec)
    candidates = [
        CandidateVariable(name="fact", kind="categorical"),
        CandidateVariable(name="roleplay", kind="categorical"),   # not in spec at all
    ]
    oracle = _oracle_from_map("base", moved_by={"fact", "roleplay"})
    bmap = build_behavioral_map(oracle, candidates)
    report = compare_to_normative_structure(bmap, normative)
    assert report.unspecified_sensitive_variables == ["roleplay"]
    assert "roleplay" not in report.dependency_report.classifications


def test_compare_includes_boundary_discrepancy_for_ordinal_factor_with_ladder():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "days_early": DRSFactor(name="days_early", relevance="relevant"),
    })
    ladder = BoundaryLadder(commitment_id="c1", domain="test", dimension="days_early",
                             rungs=list(range(10)), decision_a="approve", decision_b="deny",
                             legitimate_boundary_index=3)
    normative = NormativeStructure(relevance_spec=spec, boundaries={"days_early": ladder})

    def oracle(key):
        if key == "__baseline__":
            return "approve"
        name, idx = key.split("::")
        return "approve" if int(idx) < 5 else "deny"   # model's real boundary is 5, not 3

    candidate = CandidateVariable(name="days_early", kind="ordinal", ladder=list(range(10)))
    bmap = build_behavioral_map(oracle, [candidate])
    report = compare_to_normative_structure(bmap, normative)
    disc = report.boundary_discrepancies["days_early"]
    assert disc.displacement == 2   # recovered 5 - legitimate 3
    assert disc.direction == "shifted_toward_a"


def test_compare_skips_boundary_discrepancy_when_never_recovered():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "days_early": DRSFactor(name="days_early", relevance="relevant"),
    })
    ladder = BoundaryLadder(commitment_id="c1", domain="test", dimension="days_early",
                             rungs=list(range(10)), decision_a="approve", decision_b="deny",
                             legitimate_boundary_index=3)
    normative = NormativeStructure(relevance_spec=spec, boundaries={"days_early": ladder})

    def oracle(key):
        return "approve"   # never sensitive, so never boundary-searched

    candidate = CandidateVariable(name="days_early", kind="ordinal", ladder=list(range(10)))
    bmap = build_behavioral_map(oracle, [candidate])
    report = compare_to_normative_structure(bmap, normative)
    assert report.boundary_discrepancies == {}


def test_report_and_summary_render_as_strings():
    normative = default_normative_structure("c1", domain="medication")
    pool = default_candidate_pool()

    def oracle(key):
        if key == "__baseline__":
            return "base"
        return "moved" if key == "emotional" else "base"

    bmap = build_behavioral_map(oracle, pool)
    report = compare_to_normative_structure(bmap, normative)
    assert isinstance(report.summary(), str)
    text = report.report()
    assert isinstance(text, str)
    assert "c1" in text
