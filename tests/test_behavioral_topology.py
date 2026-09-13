"""
Tests for contradish.behavioral_topology.

No API key required: builds NormativeComparisonReport objects via
behavioral_mapping.py's own pure pipeline, fed by deterministic synthetic
oracles -- so this also serves as a small integration test across
decision_relevance.py, decision_boundary.py, behavioral_mapping.py, and
topology.py (reused, unmodified) all at once.
"""
from contradish.behavioral_mapping import (
    CandidateVariable,
    NormativeStructure,
    build_behavioral_map,
    compare_to_normative_structure,
)
from contradish.behavioral_topology import topology_from_behavioral_map
from contradish.decision_boundary import BoundaryLadder
from contradish.decision_relevance import DecisionRelevanceSpec, DRSFactor
from contradish.topology import FailureTopologyMap, topology_distance


def _spec():
    return DecisionRelevanceSpec(commitment_id="c1", domain="test", factors={
        "emotional": DRSFactor(name="emotional", relevance="irrelevant"),
        "fact": DRSFactor(name="fact", relevance="relevant"),
    })


def _comparison(moved_by):
    normative = NormativeStructure(relevance_spec=_spec())
    candidates = [
        CandidateVariable(name="emotional", kind="categorical"),
        CandidateVariable(name="fact", kind="categorical"),
    ]

    def oracle(key):
        if key == "__baseline__":
            return "base"
        return "moved" if key in moved_by else "base"

    bmap = build_behavioral_map(oracle, candidates)
    return compare_to_normative_structure(bmap, normative)


# ── node construction from classification cells ─────────────────────────────

def test_tracked_factor_has_zero_cai_strain():
    # fact is relevant and sensitive -> tracked -> correct, cai_strain 0
    comparison = _comparison(moved_by={"fact"})
    topo = topology_from_behavioral_map(comparison, model="m")
    assert topo.nodes["fact"].cai_strain == 0.0


def test_spurious_factor_has_full_cai_strain():
    # emotional is irrelevant and sensitive -> spurious -> wrong, cai_strain 1
    comparison = _comparison(moved_by={"emotional"})
    topo = topology_from_behavioral_map(comparison, model="m")
    assert topo.nodes["emotional"].cai_strain == 1.0


def test_missed_factor_also_has_full_cai_strain():
    # fact is relevant but NOT sensitive -> missed -> wrong, cai_strain 1
    comparison = _comparison(moved_by=set())
    topo = topology_from_behavioral_map(comparison, model="m")
    assert topo.nodes["fact"].cai_strain == 1.0


def test_invariant_factor_has_zero_cai_strain():
    # emotional is irrelevant and NOT sensitive -> invariant -> correct
    comparison = _comparison(moved_by=set())
    topo = topology_from_behavioral_map(comparison, model="m")
    assert topo.nodes["emotional"].cai_strain == 0.0


# ── unspecified sensitive variables become nodes too ─────────────────────────

def test_unspecified_sensitive_variable_becomes_a_node():
    normative = NormativeStructure(relevance_spec=DecisionRelevanceSpec(
        commitment_id="c1", domain="test",
        factors={"fact": DRSFactor(name="fact", relevance="relevant")},
    ))
    candidates = [
        CandidateVariable(name="fact", kind="categorical"),
        CandidateVariable(name="roleplay", kind="categorical"),
    ]

    def oracle(key):
        if key == "__baseline__":
            return "base"
        return "moved"

    bmap = build_behavioral_map(oracle, candidates)
    comparison = compare_to_normative_structure(bmap, normative)
    topo = topology_from_behavioral_map(comparison, model="m")
    assert "roleplay" in topo.nodes
    assert topo.nodes["roleplay"].cai_strain == 1.0
    assert topo.nodes["roleplay"].lambda_weight == 0.5
    assert "UNSPECIFIED" in topo.nodes["roleplay"].description


# ── lambda_weights override ──────────────────────────────────────────────────

def test_lambda_weights_override_replaces_the_default():
    comparison = _comparison(moved_by={"emotional"})
    topo = topology_from_behavioral_map(comparison, model="m", lambda_weights={"emotional": 0.2})
    assert topo.nodes["emotional"].lambda_weight == 0.2
    assert topo.nodes["fact"].lambda_weight == 1.0   # untouched, uses default


# ── ordinal factor: reality_strain from real boundary displacement ──────────

def test_ordinal_factor_reality_strain_comes_from_normalized_displacement():
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
        return "approve" if int(idx) < 5 else "deny"   # true boundary at 5, legitimate is 3

    candidate = CandidateVariable(name="days_early", kind="ordinal", ladder=list(range(10)))
    bmap = build_behavioral_map(oracle, [candidate])
    comparison = compare_to_normative_structure(bmap, normative)
    topo = topology_from_behavioral_map(comparison, model="m")

    expected = round(abs(2 / 10), 4)   # displacement 5-3=2, normalized 2/10
    assert topo.nodes["days_early"].reality_strain == expected
    # and it's distinguishable from cai_strain (0.0, since tracked -- correctly sensitive)
    assert topo.nodes["days_early"].cai_strain == 0.0
    assert topo.nodes["days_early"].reality_strain != topo.nodes["days_early"].cai_strain


def test_categorical_factor_reality_strain_falls_back_to_cai_strain():
    comparison = _comparison(moved_by={"emotional"})
    topo = topology_from_behavioral_map(comparison, model="m")
    node = topo.nodes["emotional"]
    assert node.reality_strain == node.cai_strain == 1.0


# ── edges: no fabricated chain by default ────────────────────────────────────

def test_no_edges_by_default():
    comparison = _comparison(moved_by={"emotional"})
    topo = topology_from_behavioral_map(comparison, model="m")
    assert topo.edges == []


def test_explicit_edges_are_honored():
    from contradish.topology import ReasoningEdge
    comparison = _comparison(moved_by={"emotional"})
    edges = [ReasoningEdge(source="fact", target="emotional", propagation=0.5)]
    topo = topology_from_behavioral_map(comparison, model="m", dependency_edges=edges)
    assert topo.edges == edges


# ── composes with topology.py's existing (unmodified) machinery ─────────────

def test_result_is_a_real_failuretopologymap_and_existing_methods_work():
    comparison = _comparison(moved_by={"emotional"})   # emotional spurious, fact missed -- both "wrong"
    topo = topology_from_behavioral_map(comparison, model="m")
    assert isinstance(topo, FailureTopologyMap)

    cp = topo.critical_path()
    assert set(cp.nodes) <= set(topo.nodes)

    influence = topo.superspreader_influence()
    assert set(influence) == set(topo.nodes)

    assert 0.0 <= topo.certification_coverage() <= 1.0
    assert isinstance(topo.gini_coefficient, float)
    assert isinstance(topo.report(), str)


def test_topology_distance_is_zero_for_a_topology_against_itself():
    comparison = _comparison(moved_by={"emotional"})
    topo = topology_from_behavioral_map(comparison, model="m")
    assert topology_distance(topo, topo) == 0.0


def test_topology_distance_is_positive_for_genuinely_different_behavior():
    # one model reacts to emotional (spurious); another doesn't (invariant)
    topo_a = topology_from_behavioral_map(_comparison(moved_by={"emotional"}), model="model-a")
    topo_b = topology_from_behavioral_map(_comparison(moved_by=set()), model="model-b")
    distance = topology_distance(topo_a, topo_b)
    assert 0.0 < distance <= 1.0
