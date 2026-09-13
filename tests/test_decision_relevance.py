"""
Tests for contradish.decision_relevance.

No API key required: score_dependency_structure and
aggregate_dependency_structure are the pure, model-free cores -- these tests
build DecisionRelevanceSpec/sensitivity profiles directly with fixed values
instead of measuring them from a model.
"""
from contradish.decision_relevance import (
    DecisionRelevanceAudit,
    DecisionRelevanceSpec,
    DependencyStructureReport,
    DRSFactor,
    aggregate_dependency_structure,
    default_technique_drs,
    score_dependency_structure,
    sensitivity_from_consistency_score,
    sensitivity_profile_from_technique_scores,
)


# ── DRSFactor / DecisionRelevanceSpec ─────────────────────────────────────────

def test_invalid_relevance_value_is_rejected():
    try:
        DRSFactor(name="x", relevance="sometimes")
        assert False, "expected ValueError for an unrecognized relevance value"
    except ValueError:
        pass


def test_spec_summary_counts_by_relevance():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "a": DRSFactor(name="a", relevance="relevant"),
        "b": DRSFactor(name="b", relevance="irrelevant"),
        "c": DRSFactor(name="c", relevance="conditional"),
    })
    text = spec.summary()
    assert "1 relevant" in text
    assert "1 irrelevant" in text
    assert "1 conditional" in text


def test_spec_to_dict_round_trips_factor_fields():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "a": DRSFactor(name="a", relevance="relevant", expected_effect="should flip"),
    })
    d = spec.to_dict()
    assert d["factors"]["a"]["relevance"] == "relevant"
    assert d["factors"]["a"]["expected_effect"] == "should flip"


# ── default_technique_drs (seeded from the real TECHNIQUE_NAMES set) ────────

def test_default_technique_drs_covers_all_eight_techniques():
    spec = default_technique_drs("medication-002", domain="medication")
    assert set(spec.factors) == {
        "emotional", "presuppose", "casual", "sympathy",
        "authority", "hypothetical", "boundary", "indirect",
    }


def test_default_technique_drs_marks_authority_conditional_rest_irrelevant():
    spec = default_technique_drs("c1")
    assert spec.factors["authority"].relevance == "conditional"
    for name in ("emotional", "presuppose", "casual", "sympathy",
                 "hypothetical", "boundary", "indirect"):
        assert spec.factors[name].relevance == "irrelevant", name


# ── sensitivity_from_consistency_score / sensitivity_profile_from_technique_scores ──

def test_sensitivity_is_complement_of_consistency_score():
    assert sensitivity_from_consistency_score(1.0) == 0.0
    assert sensitivity_from_consistency_score(0.0) == 1.0
    assert sensitivity_from_consistency_score(0.75) == 0.25


def test_profile_from_technique_scores_drops_none_values():
    scores = {"emotional": 1.0, "authority": 0.4, "boundary": None}
    profile = sensitivity_profile_from_technique_scores(scores)
    assert profile == {"emotional": 0.0, "authority": 0.6}
    assert "boundary" not in profile


# ── score_dependency_structure: the four-cell classification ────────────────

def test_perfect_dependency_structure_scores_full_fidelity():
    """Relevant factor changed the answer (sensitive), irrelevant factor
    didn't (insensitive) -- the textbook-correct case."""
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "fact": DRSFactor(name="fact", relevance="relevant"),
        "framing": DRSFactor(name="framing", relevance="irrelevant"),
    })
    profile = {"fact": 1.0, "framing": 0.0}
    report = score_dependency_structure(spec, profile)
    assert report.tracked == ["fact"]
    assert report.invariant == ["framing"]
    assert report.missed == [] and report.spurious == []
    assert report.relevant_sensitivity == 1.0
    assert report.irrelevant_sensitivity == 0.0
    assert report.dependency_fidelity == 1.0


def test_backwards_model_scores_zero_fidelity():
    """Exactly backwards: ignores the relevant fact, reacts to the
    irrelevant framing -- faithfulness.py's worst-case signature, here as
    the two-factor degenerate case of DRS."""
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "fact": DRSFactor(name="fact", relevance="relevant"),
        "framing": DRSFactor(name="framing", relevance="irrelevant"),
    })
    profile = {"fact": 0.0, "framing": 1.0}
    report = score_dependency_structure(spec, profile)
    assert report.missed == ["fact"]
    assert report.spurious == ["framing"]
    assert report.relevant_sensitivity == 0.0
    assert report.irrelevant_sensitivity == 1.0
    assert report.dependency_fidelity == 0.0


def test_missed_and_spurious_cells_distinguished_from_tracked_and_invariant():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "r1": DRSFactor(name="r1", relevance="relevant"),
        "r2": DRSFactor(name="r2", relevance="relevant"),
        "i1": DRSFactor(name="i1", relevance="irrelevant"),
        "i2": DRSFactor(name="i2", relevance="irrelevant"),
    })
    profile = {"r1": 0.9, "r2": 0.1, "i1": 0.8, "i2": 0.2}
    report = score_dependency_structure(spec, profile)
    assert report.tracked == ["r1"]
    assert report.missed == ["r2"]
    assert report.spurious == ["i1"]
    assert report.invariant == ["i2"]
    assert report.relevant_sensitivity == 0.5
    assert report.irrelevant_sensitivity == 0.5


def test_conditional_factor_defaults_to_irrelevant_without_context():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "authority": DRSFactor(name="authority", relevance="conditional"),
    })
    report = score_dependency_structure(spec, {"authority": 0.9})
    c = report.classifications["authority"]
    assert c.effective_relevance == "irrelevant"
    assert c.cell == "spurious"   # sensitive to authority with no verified condition -> flagged


def test_conditional_factor_becomes_relevant_when_condition_context_says_so():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "authority": DRSFactor(name="authority", relevance="conditional"),
    })
    report = score_dependency_structure(spec, {"authority": 0.9}, condition_context={"authority": True})
    c = report.classifications["authority"]
    assert c.effective_relevance == "relevant"
    assert c.cell == "tracked"


def test_unmeasured_factor_is_reported_not_silently_dropped():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "a": DRSFactor(name="a", relevance="relevant"),
        "b": DRSFactor(name="b", relevance="irrelevant"),
    })
    report = score_dependency_structure(spec, {"a": 1.0})   # "b" never measured
    assert report.unmeasured_factors == ["b"]
    assert "b" not in report.classifications


def test_no_relevant_factors_measured_gives_none_relevant_sensitivity():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "i1": DRSFactor(name="i1", relevance="irrelevant"),
    })
    report = score_dependency_structure(spec, {"i1": 0.1})
    assert report.relevant_sensitivity is None
    assert report.dependency_fidelity is None   # can't compute fidelity without both rates


def test_score_dependency_structure_populates_sdt_fields():
    spec = DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "r": DRSFactor(name="r", relevance="relevant"),
        "i": DRSFactor(name="i", relevance="irrelevant"),
    })
    report = score_dependency_structure(spec, {"r": 0.95, "i": 0.05})
    assert isinstance(report.sensitivity_d_prime, float)
    assert isinstance(report.criterion, float)
    assert isinstance(report.sdt_pattern, str) and report.sdt_pattern
    assert "SDT:" in report.report()


def test_report_and_summary_render_as_strings():
    spec = default_technique_drs("c1", domain="medication")
    profile = sensitivity_profile_from_technique_scores({
        "emotional": 1.0, "presuppose": 0.9, "casual": 1.0, "sympathy": 0.95,
        "authority": 0.6, "hypothetical": 1.0, "boundary": 0.85, "indirect": 1.0,
    })
    report = score_dependency_structure(spec, profile)
    assert isinstance(report.summary(), str)
    text = report.report()
    assert isinstance(text, str)
    assert "c1" in text


# ── aggregate_dependency_structure: pooled, not averaged-of-averages ────────

def _report(commitment_id, tracked=(), missed=(), spurious=(), invariant=()):
    classifications = {}
    for n in tracked:
        classifications[n] = None
    return DependencyStructureReport(
        commitment_id=commitment_id, domain="test", classifications=classifications,
        unmeasured_factors=[], tracked=list(tracked), missed=list(missed),
        spurious=list(spurious), invariant=list(invariant),
        relevant_sensitivity=None, irrelevant_sensitivity=None, dependency_fidelity=None,
        sensitivity_d_prime=None, criterion=None, sdt_pattern="",
    )


def test_aggregate_pools_counts_across_commitments():
    reports = {
        "c1": _report("c1", tracked=["a"], invariant=["b"]),
        "c2": _report("c2", missed=["a"], spurious=["b"]),
    }
    audit = aggregate_dependency_structure("test", reports)
    assert isinstance(audit, DecisionRelevanceAudit)
    assert audit.n_commitments == 2
    # pooled: 1 tracked + 1 missed = 2 relevant measurements -> hit rate 0.5
    assert audit.pooled_relevant_sensitivity == 0.5
    # pooled: 1 spurious + 1 invariant = 2 irrelevant measurements -> FAR 0.5
    assert audit.pooled_irrelevant_sensitivity == 0.5
    assert audit.commitments_with_missed == ["c2"]
    assert audit.commitments_with_spurious == ["c2"]


def test_aggregate_with_no_measurements_gives_none_rates():
    reports = {"c1": _report("c1")}
    audit = aggregate_dependency_structure("test", reports)
    assert audit.pooled_relevant_sensitivity is None
    assert audit.pooled_dependency_fidelity is None


def test_aggregate_report_and_summary_render_as_strings():
    reports = {"c1": _report("c1", missed=["a"], spurious=["b"])}
    audit = aggregate_dependency_structure("test", reports)
    assert isinstance(audit.summary(), str)
    text = audit.report()
    assert isinstance(text, str)
    assert "c1" in text
