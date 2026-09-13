"""
contradish/behavioral_topology.py -- an AI behavioral topology: not merely
whether a model answers correctly, but the structure governing when its
answers change.

─────────────────────────────────────────────────────────────────────────────
WHERE THIS ALREADY LIVES (user-specified, 2026-09-13)
─────────────────────────────────────────────────────────────────────────────
"Contradish should produce something resembling an AI behavioral topology:
not merely whether models answer correctly, but the structure governing
when their answers change."

topology.py already exists to answer almost exactly this question -- its
own module docstring opens with "This module addresses the prior question:
WHY does a system fail where it fails? Not merely that it has high CAI
Strain in some cases -- but why those cases, and how they relate to each
other." FailureTopologyMap already has everything the request names: a
graph of junctions with load-bearing weight (lambda) and local CAI Strain
(cai_strain), a critical path, superspreader detection, certification
coverage, a Gini coefficient of failure concentration, and
topology_distance() for comparing two systems' structures. None of that
needed to be rebuilt.

What topology.py's only existing constructor (topology_from_phi_star())
builds FROM is the gap: it populates every node from PhiStarExplorer
clustering -- asking a model what a claim depends on and clustering the
free-text answer. That is the same self-report source expand_node() uses
(see decision_relevance.py's and behavioral_mapping.py's own module notes
for why this package treats self-report and behavioral measurement as
different evidence). Nothing before this module fed the topology from
CONTROLLED, BEHAVIORAL measurement -- from actually perturbing a candidate
variable and watching whether the decision changes, the way
decision_relevance.py / decision_boundary.py / behavioral_mapping.py do.

topology_from_behavioral_map() is that missing constructor: same
FailureTopologyMap, same ReasoningNode/ReasoningEdge, same
topology_distance(), reused completely unmodified -- only the data source
feeding them is new. Concretely, per factor in a NormativeComparisonReport
(behavioral_mapping.py):

    cai_strain (epsilon_c)     1.0 if the factor's four-cell classification
                               is "missed" or "spurious" (the model's
                               dependency structure is WRONG here, either
                               direction), else 0.0. This is a deliberate
                               reinterpretation, not a direct carry-over:
                               topology_from_phi_star()'s cai_strain meant
                               "does commitment shift under framing", with
                               no notion of whether shifting was correct --
                               a distinction that didn't exist before
                               decision_relevance.py's R. A relevant factor
                               the model correctly tracks should NOT read as
                               fragile just because its answer moved; this
                               mapping scores wrongness of the dependency,
                               not raw movement.

    lambda_weight (lambda)    Defaults to a uniform 1.0 for every classified
                               factor and 0.5 for a discovered-but-
                               unspecified one (see below) -- there is no
                               validated per-factor importance weighting
                               anywhere in this package yet (bench/
                               evaluate.py's SEVERITY_MULTIPLIERS weight by
                               DOMAIN, not by factor). `lambda_weights`
                               lets a caller override per node -- the same
                               parameter topology_from_phi_star() already
                               exposes -- and SEVERITY_MULTIPLIERS is a
                               natural (not wired-in-here) source for one.

    reality_strain (epsilon_r) abs(normalized_displacement) from the
                               factor's BoundaryDiscrepancyReport when one
                               exists (a real measured quantity, reused
                               directly from decision_boundary.py) --
                               otherwise falls back to cai_strain, the same
                               "no finer signal available" placeholder
                               style topology_from_phi_star() already uses
                               for cluster.stability.

A candidate that screened behaviorally sensitive but has no entry anywhere
in the normative structure (behavioral_mapping.py's
unspecified_sensitive_variables) still becomes a node -- cai_strain=1.0
(it did show real strain), lambda_weight defaulting lower (0.5, reflecting
that whether this factor is load-bearing is itself unknown, not zero) --
rather than vanishing from the topology the way it would vanish from a bare
dependency_report.

No dependency_edges are invented by default (dependency_edges=None yields
an empty edge list here, NOT topology_from_phi_star()'s linear-chain
fallback). That fallback exists there because Phi* clusters arrive in a
recurrence order that at least suggests a sequence; behavioral candidates
(rhetorical techniques, semantic ladder rungs) are probed independently and
have no such implied order -- defaulting to a chain would assert a fake
dependency between, say, "emotional" and "authority" that nothing measured.
Pass real edges when a domain-specific ordering is actually known.

Because the result is an ordinary FailureTopologyMap, everything already
built on top of one composes for free, with zero new code: critical_path(),
superspreader_influence(), certification_coverage(), gini_coefficient, and
-- notably -- topology_distance(), which now lets two behaviorally-measured
topologies (two models, or the same model at two points in time) be
compared structurally, not just by aggregate strain.

Usage::

    from contradish.behavioral_mapping import (
        default_candidate_pool, default_normative_structure,
        build_behavioral_map, compare_to_normative_structure,
    )
    from contradish.behavioral_topology import topology_from_behavioral_map

    pool = default_candidate_pool()
    normative = default_normative_structure("medication-002", domain="medication")
    behavioral_map = build_behavioral_map(model_oracle, pool)
    comparison = compare_to_normative_structure(behavioral_map, normative)

    topo = topology_from_behavioral_map(comparison, model="my-model")
    print(topo.report())
"""

from __future__ import annotations

from typing import Optional

from contradish.topology import FailureTopologyMap, ReasoningEdge, ReasoningNode

__all__ = ["topology_from_behavioral_map"]


def topology_from_behavioral_map(
    comparison,                            # NormativeComparisonReport from behavioral_mapping.py
    model: str = "",
    dependency_edges: Optional[list] = None,
    lambda_weights: Optional[dict] = None,
) -> FailureTopologyMap:
    """
    Construct a FailureTopologyMap from a NormativeComparisonReport --
    behavioral_mapping.py's discover/map/compare pipeline is what actually
    measured the model; this only reshapes its already-computed result into
    topology.py's graph representation, so it takes no oracle and makes no
    model calls itself.

    comparison:        NormativeComparisonReport, from
                        compare_to_normative_structure().
    model:              Model identifier to label the resulting map with --
                        behavioral_mapping.py doesn't track this itself, so
                        it must be supplied here.
    dependency_edges:  See module note: defaults to no edges (independent
                        nodes), not a fabricated chain.
    lambda_weights:    Optional {factor_name: float} override; see module
                        note for the uniform defaults used otherwise.
    """
    nodes: dict = {}

    for name, classification in comparison.dependency_report.classifications.items():
        cai_strain = 1.0 if classification.cell in ("missed", "spurious") else 0.0
        discrepancy = comparison.boundary_discrepancies.get(name)
        if discrepancy is not None and discrepancy.normalized_displacement is not None:
            reality_strain = abs(discrepancy.normalized_displacement)
        else:
            reality_strain = cai_strain
        lw = lambda_weights.get(name, 1.0) if lambda_weights else 1.0

        nodes[name] = ReasoningNode(
            node_id=name,
            description=f"{classification.cell} ({classification.effective_relevance}): '{name}'"[:80],
            lambda_weight=lw,
            cai_strain=cai_strain,
            reality_strain=reality_strain,
            domain=comparison.domain,
        )

    for name in comparison.unspecified_sensitive_variables:
        lw = lambda_weights.get(name, 0.5) if lambda_weights else 0.5
        nodes[name] = ReasoningNode(
            node_id=name,
            description=f"UNSPECIFIED sensitivity: '{name}' (no entry in normative R)"[:80],
            lambda_weight=lw,
            cai_strain=1.0,
            reality_strain=1.0,
            domain=comparison.domain,
        )

    return FailureTopologyMap(
        nodes=nodes,
        edges=dependency_edges if dependency_edges is not None else [],
        domain=comparison.domain,
        model=model,
    )
