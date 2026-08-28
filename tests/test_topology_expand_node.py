"""
Tests for contradish.topology.expand_node.

Run with: pytest tests/test_topology_expand_node.py   (no API key, pure stdlib --
PhiStarExplorer works fully offline with its default extractor/similarity_fn.)
"""
import pytest

from contradish.topology import (
    FailureTopologyMap,
    ReasoningNode,
    ReasoningEdge,
    expand_node,
)
from contradish.phi_star import PhiStarExplorer


def _make_topo():
    nodes = {
        "daily_ceiling": ReasoningNode(
            "daily_ceiling", description="max daily dose", lambda_weight=0.9,
            cai_strain=0.5, reality_strain=0.2, domain="medication",
        ),
    }
    return FailureTopologyMap(nodes=nodes, edges=[], domain="medication", model="fake-model")


def _fake_model_fn(system_prompt: str, question: str) -> str:
    # Two distinct "commitments" depending on framing, so clustering finds
    # more than one DistinctionCluster -- enough to test that expand_node
    # actually differentiates children rather than collapsing everything.
    if "difficult time" in question or "desperate" in question:
        return "Weight-based dosing can flex under distress."
    return "Weight-based dosing must be followed exactly regardless of framing."


def _make_explorer(model_fn=_fake_model_fn):
    return PhiStarExplorer(model_fn=model_fn, framing_types=["neutral", "sympathy", "emotional_appeal"])


def test_expand_node_raises_on_unknown_node():
    topo = _make_topo()
    explorer = _make_explorer()
    with pytest.raises(KeyError):
        expand_node(topo, "not_a_real_node", explorer)


def test_expand_node_adds_child_nodes_and_edges_into_parent():
    topo = _make_topo()
    explorer = _make_explorer()
    before = set(topo.nodes)

    expand_node(topo, "daily_ceiling", explorer, follow_up_question="What does dosing depend on?")

    added = set(topo.nodes) - before
    assert added, "expected at least one new child node"
    for child_id in added:
        assert child_id.startswith("daily_ceiling.child_")

    # every new edge should point a new child INTO the parent (source -> target
    # means "commit to source before target")
    new_edges = [e for e in topo.edges if e.source in added]
    assert new_edges
    for e in new_edges:
        assert e.target == "daily_ceiling"


def test_expand_node_updates_adjacency_so_graph_methods_still_work():
    topo = _make_topo()
    explorer = _make_explorer()
    expand_node(topo, "daily_ceiling", explorer, follow_up_question="What does dosing depend on?")

    # sources()/sinks()/critical_path() all depend on the internal adjacency
    # index built in __init__ -- confirm expand_node kept it in sync rather
    # than leaving nodes/edges updated but the index stale.
    assert "daily_ceiling" in topo.sinks()
    path = topo.critical_path()
    assert "daily_ceiling" in path.nodes


def test_expand_node_respects_max_children():
    topo = _make_topo()
    # A model_fn returning a different sentence per framing maximizes distinct
    # clusters, so max_children is actually the limiting factor, not the data.
    def varied_model_fn(system_prompt, question):
        return f"Distinct commitment for: {question}"

    explorer = PhiStarExplorer(
        model_fn=varied_model_fn,
        framing_types=["neutral", "sympathy", "urgency", "authority", "hypothetical"],
    )
    before = set(topo.nodes)
    expand_node(topo, "daily_ceiling", explorer, max_children=2,
                follow_up_question="What does dosing depend on?")
    added = set(topo.nodes) - before
    assert len(added) <= 2


def test_expand_node_skips_a_cluster_that_just_restates_the_parent():
    topo = _make_topo()

    def echo_model_fn(system_prompt, question):
        # Every trajectory "discovers" the parent's own description verbatim --
        # a degenerate expansion that shouldn't produce any child at all.
        return "max daily dose"

    explorer = PhiStarExplorer(model_fn=echo_model_fn, framing_types=["neutral", "sympathy"])
    before = set(topo.nodes)
    expand_node(topo, "daily_ceiling", explorer, follow_up_question="What does dosing depend on?")
    assert set(topo.nodes) == before, "a cluster identical to the parent's own claim should not become a child"


def test_expand_node_default_question_uses_parent_description():
    topo = _make_topo()
    seen_questions = []

    def recording_model_fn(system_prompt, question):
        seen_questions.append(question)
        return "some sub-distinction"

    explorer = PhiStarExplorer(model_fn=recording_model_fn, framing_types=["neutral"])
    expand_node(topo, "daily_ceiling", explorer)  # no follow_up_question passed
    assert any("max daily dose" in q for q in seen_questions)
