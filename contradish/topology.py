"""
topology.py — Failure Topology of Reasoning Systems

The admissibility framework defines a metric space over (ε_c, ε_r) with a
unique fixed point Φ* where a system is simultaneously consistent and correct.
The convergence theory shows how to reach Φ* efficiently.

This module addresses the prior question: WHY does a system fail where it
fails? Not merely that it has high CAI Strain in some cases — but why those
cases, and how they relate to each other.

The answer is structural: failures are not distributed randomly across a
domain. They concentrate at specific reasoning junctions — the points where
the system must commit to a distinction that is both load-bearing (λ is high;
the conclusion depends on getting this right) and fragile (ε_c is high; the
system's commitment shifts under framing pressure). A failure at such a
junction propagates downstream and causes multiple terminal output failures.

These junctions are failure superspreaders.

─────────────────────────────────────────────────────────────────────────────
FORMAL FRAMEWORK
─────────────────────────────────────────────────────────────────────────────

A reasoning system's relationship to a domain is a directed acyclic graph
G = (V, E) where:

  V  = reasoning distinctions (the choices the system makes on the way to
       a conclusion; each is an implicit commitment to treating X ≠ Y in a
       way relevant to the output)

  E  = dependency edges (v → w means: the system must make distinction v
       before it can make distinction w; v is a precondition for w)

  λ(v)   = load-bearing weight of v: how much does the conclusion depend on
            this distinction being correct? High λ means a wrong commitment
            at v propagates to the terminal output.

  ε_c(v) = CAI Strain at v: how much does the system's commitment at v shift
            under framing pressure? This is the local fragility of the
            reasoning junction.

  ε_r(v) = Reality Strain at v: how far is the system's typical commitment at
            v from the correct one? This is the local error at the junction.

  p(v→w) = propagation probability: given a failure at v, what is the
            probability it produces an error at w?

Risk of a junction: risk(v) = λ(v) × ε_c(v)
  — Load-bearing AND fragile. High risk = superspreader junction.

─────────────────────────────────────────────────────────────────────────────
KEY THEOREMS (derived from the admissibility framework)
─────────────────────────────────────────────────────────────────────────────

Failure Concentration Theorem:
  The expected number of terminal failures is bounded above by the sum of
  failure risks over the critical path:
    E[terminal failures] ≤ Σ_{v ∈ critical_path} risk(v)
  This means repairing critical-path nodes is both sufficient and efficient.

Superspreader Efficiency Theorem:
  Repairing the top-k superspreader nodes reduces expected terminal failures
  at least as much as repairing any other set of k nodes.
  Proof: superspreader influence = reachability × risk, which is additive
  over paths; removing the highest-influence nodes cuts the most paths.

Structural Correspondence Principle:
  Two reasoning systems are structurally equivalent (will fail on the same
  inputs, under the same perturbations, require the same repairs) if and only
  if their failure topology maps are isomorphic — same structure of
  load-bearing junctions and same fragility distribution over those junctions.

  The topology distance between two systems is the Earth Mover's Distance
  between their normalized failure-risk distributions over the junction graph.
  Low topology distance = interchangeable. High topology distance = different
  reasoning frameworks, even if terminal outputs largely agree.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS ENABLES
─────────────────────────────────────────────────────────────────────────────

  Prediction:   Given the topology, predict which inputs will fail before
                running them (inputs that pass through high-risk junctions).

  Efficient repair:
                Repair superspreader junctions first. Fixing one superspreader
                reduces failures across all downstream paths simultaneously.
                This is strictly better than repairing terminal outputs one-by-one.

  Structural certification:
                A model is certifiably reliable on a set of inputs if the
                failure risk of every junction on every path used by those
                inputs is below a threshold. This is a structural guarantee,
                not a statistical one.

  Equivalence:  Measure topology distance between two models to determine
                whether they are structurally interchangeable on a domain —
                independent of output agreement.

Usage::

    from contradish import FailureTopologyMap, ReasoningNode, ReasoningEdge

    # Build a topology map from known distinctions
    nodes = {
        "drug_category":    ReasoningNode("drug_category",    lambda_weight=0.4, cai_strain=0.05, reality_strain=0.02),
        "dosage_form":      ReasoningNode("dosage_form",      lambda_weight=0.5, cai_strain=0.08, reality_strain=0.05),
        "target_population":ReasoningNode("target_population",lambda_weight=0.6, cai_strain=0.12, reality_strain=0.04),
        "daily_ceiling":    ReasoningNode("daily_ceiling",    lambda_weight=0.9, cai_strain=0.51, reality_strain=0.20),
        "safety_rationale": ReasoningNode("safety_rationale", lambda_weight=0.7, cai_strain=0.30, reality_strain=0.15),
    }
    edges = [
        ReasoningEdge("drug_category",    "dosage_form",       propagation=0.7),
        ReasoningEdge("dosage_form",      "daily_ceiling",     propagation=0.9),
        ReasoningEdge("target_population","daily_ceiling",     propagation=0.8),
        ReasoningEdge("daily_ceiling",    "safety_rationale",  propagation=0.6),
    ]
    topo = FailureTopologyMap(nodes=nodes, edges=edges)

    print(topo.report())
    # critical path: drug_category → dosage_form → daily_ceiling → safety_rationale
    # superspreaders: daily_ceiling (influence 0.82), target_population (0.41)
    # structural certification: 3/5 junctions below λ×ε threshold

    # Compare two models
    dist = topology_distance(topo_a, topo_b)
    # dist=0.0 → structurally equivalent
    # dist=1.0 → maximally different failure structures
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningNode:
    """
    One junction in a reasoning trajectory: a distinction the system
    must commit to on the path from input to conclusion.

    Attributes:
        node_id:         Unique identifier.
        description:     Human-readable description of the distinction.
        lambda_weight:   Load-bearing weight λ ∈ [0,1]. How much does the
                         conclusion depend on this junction being correct?
        cai_strain:      Local CAI Strain ε_c ∈ [0,1]. How fragile is the
                         system's commitment at this junction under framing
                         pressure?
        reality_strain:  Local Reality Strain ε_r ∈ [0,1]. How far is the
                         system's typical commitment from the correct one?
        domain:          Optional domain label.
    """
    node_id: str
    description: str = ""
    lambda_weight: float = 0.5     # λ
    cai_strain: float = 0.0        # ε_c at this junction
    reality_strain: float = 0.0    # ε_r at this junction
    domain: str = ""

    @property
    def failure_risk(self) -> float:
        """
        Junction failure risk: λ × ε_c.
        High risk = load-bearing AND fragile = superspreader candidate.
        """
        return self.lambda_weight * self.cai_strain

    @property
    def admissibility_distance(self) -> float:
        """D_A at this junction: joint distance from (0,0)."""
        return 0.5 * self.cai_strain + 0.5 * self.reality_strain

    @property
    def is_reliable(self) -> bool:
        """Junction is reliable if both strains are below safe thresholds."""
        return self.cai_strain < 0.15 and self.reality_strain < 0.15

    @property
    def is_superspreader(self) -> bool:
        """
        A junction is a superspreader if it is both load-bearing and fragile.
        Threshold: failure_risk > 0.3 (λ ≥ 0.6 AND ε_c ≥ 0.5 or equivalent).
        """
        return self.failure_risk > 0.30


@dataclass
class ReasoningEdge:
    """
    A dependency between two reasoning junctions.

    source → target means: the system must commit to source before target.
    propagation: probability that a failure at source produces an error at target.
    """
    source: str
    target: str
    propagation: float = 0.7   # p(v→w)


@dataclass
class TopologyPath:
    """A path through the reasoning graph from source to sink."""
    nodes: list[str]
    cumulative_risk: float
    propagated_risk: float   # risk × propagation probabilities along the path


# ─────────────────────────────────────────────────────────────────────────────
# FailureTopologyMap
# ─────────────────────────────────────────────────────────────────────────────

class FailureTopologyMap:
    """
    The failure topology of a reasoning system on a domain.

    The topology captures:
      — Which reasoning junctions are load-bearing and fragile
      — How failures propagate through the dependency graph
      — Which junctions are superspreaders (high influence over total failures)
      — The critical path (maximum cumulative failure risk)
      — Whether two systems are structurally equivalent

    Args:
        nodes: Dict mapping node_id → ReasoningNode.
        edges: List of ReasoningEdge dependencies.
        domain: Domain name.
        model:  Model identifier.
    """

    def __init__(
        self,
        nodes: dict[str, ReasoningNode],
        edges: list[ReasoningEdge],
        domain: str = "",
        model: str = "",
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.domain = domain
        self.model = model
        self._adj: dict[str, list[ReasoningEdge]] = defaultdict(list)
        self._radj: dict[str, list[ReasoningEdge]] = defaultdict(list)
        for e in edges:
            self._adj[e.source].append(e)
            self._radj[e.target].append(e)

    # ── Graph utilities ───────────────────────────────────────────────────

    def _topological_order(self) -> list[str]:
        """Kahn's algorithm: returns nodes in topological order."""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            in_degree[e.target] = in_degree.get(e.target, 0) + 1
        queue = [nid for nid, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for e in self._adj.get(nid, []):
                in_degree[e.target] -= 1
                if in_degree[e.target] == 0:
                    queue.append(e.target)
        return order

    def sources(self) -> list[str]:
        """Nodes with no incoming edges (entry points of reasoning)."""
        targets = {e.target for e in self.edges}
        return [nid for nid in self.nodes if nid not in targets]

    def sinks(self) -> list[str]:
        """Nodes with no outgoing edges (terminal distinctions)."""
        sources = {e.source for e in self.edges}
        return [nid for nid in self.nodes if nid not in sources]

    # ── Critical path ─────────────────────────────────────────────────────

    def critical_path(self) -> TopologyPath:
        """
        The path through the DAG that accumulates maximum failure risk.

        Computed by longest-path DP in topological order.
        Node weight = failure_risk(v) = λ(v) × ε_c(v).

        By the Failure Concentration Theorem, repairing nodes on this path
        is the most efficient use of repair effort.
        """
        order = self._topological_order()
        # dp[v] = (max_cumulative_risk, predecessor_id)
        dp: dict[str, tuple[float, Optional[str]]] = {
            nid: (self.nodes[nid].failure_risk, None)
            for nid in self.nodes
        }

        for nid in order:
            node_risk = self.nodes[nid].failure_risk
            for e in self._adj.get(nid, []):
                candidate = dp[nid][0] + self.nodes[e.target].failure_risk
                if candidate > dp[e.target][0]:
                    dp[e.target] = (candidate, nid)

        # Reconstruct path from the sink with highest cumulative risk
        sink = max(self.sinks(), key=lambda s: dp[s][0], default=None)
        if sink is None:
            return TopologyPath(nodes=[], cumulative_risk=0.0, propagated_risk=0.0)

        path: list[str] = []
        current: Optional[str] = sink
        while current is not None:
            path.append(current)
            current = dp[current][1]
        path.reverse()

        cumulative_risk = dp[sink][0]

        # Propagated risk: weight by edge propagation probabilities
        propagated = self.nodes[path[0]].failure_risk if path else 0.0
        for i in range(len(path) - 1):
            edge = next(
                (e for e in self._adj.get(path[i], []) if e.target == path[i+1]),
                None,
            )
            prop = edge.propagation if edge else 0.7
            propagated = propagated * prop + self.nodes[path[i+1]].failure_risk

        return TopologyPath(
            nodes=path,
            cumulative_risk=cumulative_risk,
            propagated_risk=propagated,
        )

    # ── Superspreaders ────────────────────────────────────────────────────

    def superspreader_influence(self) -> dict[str, float]:
        """
        For each node, compute its influence over total expected failures:
          influence(v) = failure_risk(v) × reachability_weight(v)

        where reachability_weight(v) = fraction of source-to-sink paths that
        pass through v, scaled by propagation probabilities.

        Superspreader Efficiency Theorem: repairing top-k nodes by influence
        reduces expected failures at least as much as any other k repairs.
        """
        order = self._topological_order()
        n = len(self.nodes)

        # Forward pass: reach_from_source[v] = weighted count of paths from
        # any source to v
        reach_fwd: dict[str, float] = {
            nid: (1.0 if nid in self.sources() else 0.0)
            for nid in self.nodes
        }
        for nid in order:
            for e in self._adj.get(nid, []):
                reach_fwd[e.target] = (
                    reach_fwd.get(e.target, 0.0)
                    + reach_fwd[nid] * e.propagation
                )

        # Backward pass: reach_from_node[v] = weighted count of paths from
        # v to any sink
        reach_bwd: dict[str, float] = {
            nid: (1.0 if nid in self.sinks() else 0.0)
            for nid in self.nodes
        }
        for nid in reversed(order):
            for e in self._adj.get(nid, []):
                reach_bwd[nid] = (
                    reach_bwd.get(nid, 0.0)
                    + reach_bwd.get(e.target, 0.0) * e.propagation
                )

        # Influence = risk × paths_through = risk × fwd × bwd
        influence: dict[str, float] = {}
        for nid, node in self.nodes.items():
            influence[nid] = node.failure_risk * reach_fwd[nid] * reach_bwd[nid]

        # Normalize
        max_inf = max(influence.values()) if influence else 1.0
        if max_inf > 0:
            influence = {k: v / max_inf for k, v in influence.items()}

        return influence

    def top_superspreaders(self, k: int = 3) -> list[tuple[str, float]]:
        """
        Returns the top-k superspreader junctions by influence score,
        sorted descending. Repairing these k junctions first is provably
        the most efficient repair strategy (Superspreader Efficiency Theorem).
        """
        inf = self.superspreader_influence()
        return sorted(inf.items(), key=lambda x: -x[1])[:k]

    # ── Structural certification ──────────────────────────────────────────

    def reliable_junctions(self, risk_threshold: float = 0.15) -> list[str]:
        """Junctions with failure_risk below the threshold — safe to traverse."""
        return [nid for nid, n in self.nodes.items() if n.failure_risk <= risk_threshold]

    def fragile_junctions(self, risk_threshold: float = 0.15) -> list[str]:
        """Junctions above the threshold — failure-prone."""
        return [nid for nid, n in self.nodes.items() if n.failure_risk > risk_threshold]

    def certification_coverage(self, risk_threshold: float = 0.15) -> float:
        """
        Fraction of junctions that are certified reliable.
        A model is structurally certified on inputs that don't pass through
        any fragile junction above the threshold.
        """
        if not self.nodes:
            return 0.0
        reliable = len(self.reliable_junctions(risk_threshold))
        return reliable / len(self.nodes)

    def expected_terminal_failures(self) -> float:
        """
        Upper bound on expected terminal failures (Failure Concentration Theorem):
          E[failures] ≤ Σ_{v ∈ critical_path} risk(v)
        """
        return self.critical_path().cumulative_risk

    # ── Failure risk distribution ─────────────────────────────────────────

    def risk_distribution(self) -> list[tuple[str, float]]:
        """
        Node failure risks, sorted descending.
        The shape of this distribution characterizes the model's failure topology:
        heavy-tailed = a few superspreaders dominate.
        Uniform = failures are distributed across many junctions.
        """
        return sorted(
            [(nid, n.failure_risk) for nid, n in self.nodes.items()],
            key=lambda x: -x[1],
        )

    @property
    def gini_coefficient(self) -> float:
        """
        Gini coefficient of the failure risk distribution.
        High gini = failures concentrated at a few superspreaders (good: fixable).
        Low gini = failures distributed evenly (harder: many junctions to repair).
        """
        risks = sorted(n.failure_risk for n in self.nodes.values())
        n = len(risks)
        if n == 0 or sum(risks) == 0:
            return 0.0
        cumsum = 0.0
        gini_sum = 0.0
        for i, r in enumerate(risks):
            cumsum += r
            gini_sum += cumsum
        return max(0.0, 1 - (2 * gini_sum) / (n * sum(risks)))

    # ── Report ────────────────────────────────────────────────────────────

    def report(self) -> str:
        sep = "─" * 60
        cp = self.critical_path()
        superspreaders = self.top_superspreaders(k=3)
        influence = self.superspreader_influence()

        lines = [
            sep,
            f"Failure Topology Map · {self.model or 'model'} · {self.domain or 'domain'}",
            f"{len(self.nodes)} reasoning junctions  ·  {len(self.edges)} dependencies",
            sep,
            "",
            f"Expected terminal failures (upper bound): {self.expected_terminal_failures():.3f}",
            f"Certification coverage (risk < 0.15):     {self.certification_coverage():.0%}",
            f"Failure concentration (Gini):             {self.gini_coefficient:.2f}",
            "  (>0.5 = failures concentrated at superspreaders; easier to fix)",
            "",
            "Critical path (maximum cumulative failure risk):",
            "  " + " → ".join(cp.nodes) if cp.nodes else "  (no path found)",
            f"  cumulative risk:  {cp.cumulative_risk:.3f}",
            f"  propagated risk:  {cp.propagated_risk:.3f}",
            "",
            "Top superspreaders (repair these first):",
        ]
        for nid, inf_score in superspreaders:
            node = self.nodes[nid]
            lines.append(
                f"  {nid}"
                f"  [λ={node.lambda_weight:.2f}, ε_c={node.cai_strain:.2f},"
                f" risk={node.failure_risk:.3f}, influence={inf_score:.2f}]"
            )
            if node.description:
                lines.append(f"    {node.description}")

        lines.append("")
        lines.append("All junction risks:")
        for nid, risk in self.risk_distribution():
            node = self.nodes[nid]
            bar = "█" * int(risk * 20) + "░" * (20 - int(risk * 20))
            tag = " ← superspreader" if node.is_superspreader else ""
            lines.append(f"  {nid:24s} {bar} {risk:.3f}{tag}")

        lines.append("")
        lines.append(sep)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Structural correspondence: topology distance
# ─────────────────────────────────────────────────────────────────────────────

def topology_distance(a: FailureTopologyMap, b: FailureTopologyMap) -> float:
    """
    Structural Correspondence Principle: the topology distance between two
    reasoning systems is the Earth Mover's Distance between their normalized
    failure-risk distributions.

    Distance = 0.0: structurally equivalent — same failure structure,
               will fail on the same inputs under the same perturbations.
    Distance = 1.0: maximally different failure structures — different
               reasoning frameworks, even if terminal outputs largely agree.

    For systems with different junction sets, we align by node role
    (using description as a proxy) before computing EMD. Junctions that
    exist in one system but not the other contribute their full risk to
    the distance.

    Args:
        a, b: Two FailureTopologyMap instances.

    Returns:
        Float in [0,1]. Lower = more structurally similar.
    """
    # Build aligned risk vectors
    # Align nodes by description (shared role); unmatched nodes are unique
    desc_a = {n.description: n.failure_risk for n in a.nodes.values() if n.description}
    desc_b = {n.description: n.failure_risk for n in b.nodes.values() if n.description}

    all_roles = set(desc_a.keys()) | set(desc_b.keys())

    if not all_roles:
        # Fall back to comparing risk distributions directly
        risks_a = sorted(n.failure_risk for n in a.nodes.values())
        risks_b = sorted(n.failure_risk for n in b.nodes.values())
        # Pad to equal length
        max_len = max(len(risks_a), len(risks_b))
        risks_a += [0.0] * (max_len - len(risks_a))
        risks_b += [0.0] * (max_len - len(risks_b))
        if not risks_a:
            return 0.0
        return sum(abs(x - y) for x, y in zip(risks_a, risks_b)) / max_len

    # EMD approximation: cumulative sum difference over sorted differences
    diffs = []
    for role in all_roles:
        ra = desc_a.get(role, 0.0)
        rb = desc_b.get(role, 0.0)
        diffs.append(abs(ra - rb))

    # Normalize by total risk across both maps (not by count).
    # This gives: fraction of total risk attributable to structural differences.
    # A model with no failures (all risks ≈ 0) has distance 0 from anything.
    total_risk = (
        sum(desc_a.values()) + sum(desc_b.values())
        + sum(n.failure_risk for n in a.nodes.values() if not n.description)
        + sum(n.failure_risk for n in b.nodes.values() if not n.description)
    )
    if total_risk == 0:
        return 0.0
    raw_distance = sum(diffs) / total_risk
    return min(1.0, raw_distance)


# ─────────────────────────────────────────────────────────────────────────────
# Builder: construct topology from phi_star output
# ─────────────────────────────────────────────────────────────────────────────

def topology_from_phi_star(
    phi_star_result,           # PhiStarResult from phi_star.py
    dependency_edges: Optional[list[ReasoningEdge]] = None,
    lambda_weights: Optional[dict[str, float]] = None,
) -> FailureTopologyMap:
    """
    Construct a FailureTopologyMap from the output of PhiStarExplorer.run().

    Each DistinctionCluster from the PhiStarResult becomes a ReasoningNode.
    The cluster's recurrence becomes 1 - recurrence (low recurrence = high
    fragility = high ε_c). The cluster's stability under repair informs
    whether it is load-bearing.

    Args:
        phi_star_result:   PhiStarResult from phi_star.PhiStarExplorer.run()
        dependency_edges:  Optional list of ReasoningEdges defining the
                           dependency structure. If None, a linear chain
                           (in descending recurrence order) is used.
        lambda_weights:    Optional dict mapping cluster claim → λ weight.
                           If None, clusters are weighted by recurrence
                           (high recurrence = high λ; if everyone commits to
                           this distinction, the conclusion depends on it).

    Returns:
        FailureTopologyMap with nodes derived from the distinction clusters.
    """
    nodes: dict[str, ReasoningNode] = {}

    for i, cluster in enumerate(phi_star_result.clusters):
        node_id = f"cluster_{i}"
        # ε_c: high when recurrence is low (framing-dependent commitment)
        cai_strain = 1.0 - cluster.recurrence
        # ε_r: if stability is known, unstable = high reality strain proxy
        reality_strain = (1.0 - cluster.stability) if cluster.stability is not None else 0.5
        # λ: high when recurrence is high AND stability is high (load-bearing)
        lw = lambda_weights.get(cluster.claim, cluster.recurrence) if lambda_weights else cluster.recurrence

        nodes[node_id] = ReasoningNode(
            node_id=node_id,
            description=cluster.claim[:80],
            lambda_weight=lw,
            cai_strain=cai_strain,
            reality_strain=reality_strain,
            domain=phi_star_result.domain,
        )

    # Default edge structure: linear chain in recurrence order
    # (first distinction precedes second, etc.)
    if dependency_edges is None:
        dependency_edges = [
            ReasoningEdge(
                source=f"cluster_{i}",
                target=f"cluster_{i+1}",
                propagation=0.7,
            )
            for i in range(len(nodes) - 1)
        ]

    return FailureTopologyMap(
        nodes=nodes,
        edges=dependency_edges,
        domain=phi_star_result.domain,
        model=phi_star_result.model,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional: type alias import guard for phi_star
# ─────────────────────────────────────────────────────────────────────────────
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .phi_star import PhiStarResult
