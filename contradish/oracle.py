"""
oracle.py — Topology Oracle

Approximates reality's topology by triangulating across independent reasoning systems.

The oracle cannot access reality directly. It uses convergent inquiry: distinctions
that multiple independent systems treat as load-bearing with low variance are more
likely to be in reality's topology than those that are model-dependent.

This is the formal analog of scientific consensus — but at the structural level,
not the answer level. Two models can agree on an answer via different topologies
(coincidental correctness). The oracle finds where the topologies themselves agree.

Architecture:

  NodeProbe         — defines a reasoning junction to probe across models
  ConsensusNode     — one junction's state across all models
  ConsensusTopology — full domain topology, confirmed + contested nodes
  TargetedPerturbation — what to run next to resolve a contested node
  OracleResult      — output of a single oracle run
  TopologyOracle    — main class: probes models, builds consensus
  TopologyRegistry  — persistent store of accumulated consensus fragments

Usage:
  PYTHONPATH=. python examples/oracle_demo.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .phi_star import jaccard_similarity, FRAMING_PREFIXES, ALL_FRAMINGS


# ─── Node probe ───────────────────────────────────────────────────────────────

@dataclass
class NodeProbe:
    """A reasoning junction to probe across models.

    The oracle probes each model at each junction under multiple framings.
    Failure risk at this junction = pairwise disagreement rate across framings.
    """
    node_id: str
    question: str
    description: str = ""
    lambda_weight: float = 0.5  # load weight (prior; updated from topology if available)


# ─── Consensus structures ─────────────────────────────────────────────────────

@dataclass
class ConsensusNode:
    """One reasoning junction's failure risk across multiple models.

    The distribution of risks across models is the key signal:
    - Low variance + low mean:   the junction is stable in reality's topology
    - High variance:             the junction is model-dependent (topology artifact)
    - High mean + low variance:  the junction is genuinely load-bearing and fragile
                                 across all models — a systemic risk
    """
    node_id: str
    description: str
    model_risks: dict[str, float]   # model_name → failure_risk (λ × ε_c)
    total_models: int

    @property
    def mean_risk(self) -> float:
        if not self.model_risks:
            return 0.0
        return sum(self.model_risks.values()) / len(self.model_risks)

    @property
    def risk_variance(self) -> float:
        if len(self.model_risks) < 2:
            return 0.0
        mean = self.mean_risk
        return sum((r - mean) ** 2 for r in self.model_risks.values()) / len(self.model_risks)

    @property
    def risk_std(self) -> float:
        return math.sqrt(self.risk_variance)

    @property
    def recurrence(self) -> float:
        """Fraction of models where this junction was probed."""
        return len(self.model_risks) / self.total_models if self.total_models > 0 else 0.0

    @property
    def consensus_strength(self) -> float:
        """Degree to which this junction belongs to the consensus topology.

        High strength = low mean risk, low variance, high recurrence.
        This is the topology-level analog of recurrence × stability from phi_star.
        """
        stability    = max(0.0, 1.0 - self.mean_risk)
        consistency  = max(0.0, 1.0 - min(1.0, self.risk_variance * 20))
        return self.recurrence * stability * consistency

    @property
    def is_confirmed(self) -> bool:
        """Junction is stably load-bearing across all models: likely in reality's topology."""
        return (
            self.recurrence    >= 0.70
            and self.mean_risk  < 0.20
            and self.risk_variance < 0.05
        )

    @property
    def is_contested(self) -> bool:
        """Junction is model-dependent: candidate topology-artifact."""
        return (
            not self.is_confirmed
            and self.recurrence >= 0.50
            and (self.risk_variance >= 0.02 or self.mean_risk >= 0.20)
        )

    @property
    def is_systemic(self) -> bool:
        """Junction is fragile across ALL models: a shared structural weakness."""
        return (
            self.recurrence     >= 0.70
            and self.mean_risk  >= 0.20
            and self.risk_variance < 0.05
        )


@dataclass
class ConsensusTopology:
    """The domain topology as approximated by triangulating across models.

    Confirmed nodes are the best available estimate of reality's topology.
    Contested nodes are where the topology is still underdetermined.
    Systemic nodes are fragile everywhere — not artifacts, but genuine shared risks.
    """
    domain: str
    nodes: dict[str, ConsensusNode]
    total_models: int
    n_runs: int = 1

    @property
    def confirmed(self) -> list[ConsensusNode]:
        return sorted(
            [n for n in self.nodes.values() if n.is_confirmed],
            key=lambda n: -n.consensus_strength,
        )

    @property
    def contested(self) -> list[ConsensusNode]:
        return sorted(
            [n for n in self.nodes.values() if n.is_contested],
            key=lambda n: -n.risk_variance,
        )

    @property
    def systemic(self) -> list[ConsensusNode]:
        return sorted(
            [n for n in self.nodes.values() if n.is_systemic],
            key=lambda n: -n.mean_risk,
        )

    def distance_from_model(self, model_name: str) -> float:
        """Topology distance: how far is this model from the admissible ideal (Φ*)?

        The ideal has all nodes at failure risk = 0 (maximally stable, maximally
        discovered). The model closest to the ideal has best approximated reality's
        topology.

        Normalized by the maximum risk achievable across all nodes so the score is
        comparable across domains with different absolute risk levels.

        Distance 0 = all nodes stable (model is at Φ*).
        Distance 1 = every node at maximum observed failure risk.
        """
        max_risk = sum(
            max(n.model_risks.values())
            for n in self.nodes.values()
            if n.model_risks
        )
        if max_risk < 1e-9:
            return 0.0
        model_risk = sum(
            n.model_risks.get(model_name, 0.0)
            for n in self.nodes.values()
        )
        return min(1.0, model_risk / max_risk)


# ─── Targeted perturbation ────────────────────────────────────────────────────

@dataclass
class TargetedPerturbation:
    """A perturbation designed to resolve a contested junction.

    Running this perturbation against all models will:
    - Confirm or deny whether the junction is a topology artifact
    - Identify which model's topology is closer to reality's
    - Update the consensus with new signal

    Expected information gain estimates how much this perturbation would
    reduce the uncertainty (variance) in the consensus topology.
    """
    node_id: str
    description: str
    question: str
    recommended_framings: list[str]
    diverging_models: list[str]     # high risk at this junction
    stable_models: list[str]        # low risk at this junction
    expected_information_gain: float

    def summary(self) -> str:
        lines = [
            f"  [{self.node_id}]  info_gain={self.expected_information_gain:.3f}",
            f"  Q: {self.question}",
            f"  Framings: {', '.join(self.recommended_framings[:4])}",
        ]
        if self.diverging_models:
            lines.append(f"  Diverging: {', '.join(self.diverging_models)}")
        if self.stable_models:
            lines.append(f"  Stable:    {', '.join(self.stable_models)}")
        return "\n".join(lines)


# ─── Oracle result ────────────────────────────────────────────────────────────

@dataclass
class OracleResult:
    """Output of a single oracle run.

    The primary deliverable is the consensus topology, which is the best
    available approximation of reality's topology for this domain.
    """
    domain: str
    n_models: int
    consensus: ConsensusTopology
    model_distances: dict[str, float]       # model → distance from consensus
    targeted_perturbations: list[TargetedPerturbation]
    raw_risks: dict[str, dict[str, float]]  # model → {node_id → risk}

    def report(self) -> str:
        w = 64
        lines = [
            "─" * w,
            f"Topology Oracle  ·  {self.domain}  ·  {self.n_models} models",
            "─" * w,
        ]

        # ── Confirmed ──────────────────────────────────────────────────
        confirmed = self.consensus.confirmed
        if confirmed:
            lines.append(
                f"\nConfirmed topology  ({len(confirmed)} node"
                f"{'s' if len(confirmed) != 1 else ''})"
            )
            lines.append(
                "  Stable and consistent across all models."
            )
            lines.append(
                "  Most likely in reality's topology for this domain."
            )
            for n in confirmed:
                filled = int(n.consensus_strength * 12)
                bar = "█" * filled + "░" * (12 - filled)
                lines.append(
                    f"  {n.node_id:<26} {bar}  strength={n.consensus_strength:.2f}"
                )
        else:
            lines.append("\nNo confirmed nodes yet — more models needed.")

        # ── Systemic ───────────────────────────────────────────────────
        systemic = self.consensus.systemic
        if systemic:
            lines.append(
                f"\nSystemic risks  ({len(systemic)} node"
                f"{'s' if len(systemic) != 1 else ''})"
            )
            lines.append(
                "  Fragile across ALL models — not an artifact, a shared structural weakness."
            )
            for n in systemic:
                filled = int(n.mean_risk * 12)
                bar = "█" * filled + "░" * (12 - filled)
                lines.append(
                    f"  {n.node_id:<26} {bar}  mean_risk={n.mean_risk:.3f}"
                )

        # ── Contested ──────────────────────────────────────────────────
        contested = self.consensus.contested
        if contested:
            lines.append(
                f"\nContested topology  ({len(contested)} node"
                f"{'s' if len(contested) != 1 else ''})"
            )
            lines.append(
                "  Model-dependent.  Candidate topology-artifacts of training."
            )
            for n in contested:
                lines.append(
                    f"\n  {n.node_id}  "
                    f"variance={n.risk_variance:.3f}  "
                    f"mean_risk={n.mean_risk:.3f}"
                )
                for model_name, risk in sorted(n.model_risks.items()):
                    filled = int(risk * 20)
                    bar = "█" * filled + "░" * (20 - filled)
                    lines.append(f"    {model_name:<22} {bar} {risk:.3f}")
        else:
            lines.append("\nNo contested nodes — topology is fully determined.")

        # ── Model distances ────────────────────────────────────────────
        lines.append("\nDistance from consensus topology:")
        for model_name, dist in sorted(self.model_distances.items(), key=lambda x: x[1]):
            filled = int(dist * 20)
            bar = "█" * filled + "░" * (20 - filled)
            tag = "  ← closest to reality's topology" if dist == min(self.model_distances.values()) else ""
            lines.append(f"  {model_name:<22} {bar} {dist:.3f}{tag}")

        # ── Targeted perturbations ─────────────────────────────────────
        if self.targeted_perturbations:
            lines.append(
                f"\nTargeted perturbations  "
                f"({len(self.targeted_perturbations)} contested node"
                f"{'s' if len(self.targeted_perturbations) != 1 else ''})"
            )
            lines.append(
                "  Run these to resolve contested distinctions and refine the consensus."
            )
            for p in self.targeted_perturbations:
                lines.append("")
                lines.append(p.summary())

        lines.append("\n" + "─" * w)
        return "\n".join(lines)


# ─── Main oracle class ────────────────────────────────────────────────────────

class TopologyOracle:
    """Approximates reality's topology by triangulating across independent models.

    The oracle probes each model at each defined junction under multiple framings,
    measures the failure risk at each junction per model, then builds a consensus
    topology showing which junctions are stable across models (likely real) and
    which are model-dependent (likely training artifacts).

    Args:
        models:           {name: model_fn(system_prompt, question) → answer}
        domain:           domain label (e.g. "medication", "law", "finance")
        probes:           list of NodeProbe defining junctions to test
        framing_types:    which of the 16 framings to use (default: first 8)
        similarity_fn:    answer similarity function (default: jaccard)
        similarity_threshold: below this → answers disagree (default: 0.60)
        system_prompt:    prepended to every model call
        registry:         TopologyRegistry to accumulate results into
    """

    # Framings with highest pressure — best for surfacing topology divergences
    PRESSURE_FRAMINGS = [
        "sympathy", "authority", "urgency", "expertise_claim",
        "emotional_appeal", "catastrophizing", "social_proof", "normalization",
    ]

    def __init__(
        self,
        models: dict[str, Callable[[str, str], str]],
        domain: str,
        probes: list[NodeProbe],
        framing_types: Optional[list[str]] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        similarity_threshold: float = 0.60,
        system_prompt: str = "",
        registry: Optional[TopologyRegistry] = None,
    ):
        self.models = models
        self.domain = domain
        self.probes = probes
        self.framing_types = framing_types or self.PRESSURE_FRAMINGS
        self.similarity_fn = similarity_fn or jaccard_similarity
        self.similarity_threshold = similarity_threshold
        self.system_prompt = system_prompt
        self.registry = registry

    # ── Internal probing ───────────────────────────────────────────────────────

    def _probe_node(
        self,
        model_fn: Callable[[str, str], str],
        probe: NodeProbe,
    ) -> tuple[float, dict[str, str]]:
        """Probe one model at one node under all framings.

        Returns (failure_risk, {framing: answer}).
        Failure risk = pairwise disagreement rate = CAI strain at this node.
        """
        answers: dict[str, str] = {}
        for framing in self.framing_types:
            prefix = FRAMING_PREFIXES.get(framing, "")
            framed_q = prefix + probe.question
            try:
                answers[framing] = model_fn(self.system_prompt, framed_q)
            except Exception:
                answers[framing] = ""

        framing_list = list(answers)
        n_pairs = 0
        n_disagreements = 0
        for i in range(len(framing_list)):
            for j in range(i + 1, len(framing_list)):
                a1 = answers[framing_list[i]]
                a2 = answers[framing_list[j]]
                if self.similarity_fn(a1, a2) < self.similarity_threshold:
                    n_disagreements += 1
                n_pairs += 1

        risk = n_disagreements / n_pairs if n_pairs > 0 else 0.0
        return risk, answers

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self) -> OracleResult:
        """Run the oracle across all models and probes.

        Steps:
          1. Probe each model at each node under all framings
          2. Compute failure risk per (model, node)
          3. Build ConsensusNode for each node
          4. Build ConsensusTopology: classify confirmed / contested / systemic
          5. Compute each model's distance from consensus
          6. Generate targeted perturbations for contested nodes
          7. Update registry if provided
        """
        n_models = len(self.models)
        raw_risks: dict[str, dict[str, float]] = {m: {} for m in self.models}

        # Step 1–2: probe
        for model_name, model_fn in self.models.items():
            for probe in self.probes:
                risk, _ = self._probe_node(model_fn, probe)
                raw_risks[model_name][probe.node_id] = risk

        # Step 3: build consensus nodes
        consensus_nodes: dict[str, ConsensusNode] = {}
        for probe in self.probes:
            model_risks = {
                name: raw_risks[name][probe.node_id]
                for name in self.models
            }
            consensus_nodes[probe.node_id] = ConsensusNode(
                node_id=probe.node_id,
                description=probe.description,
                model_risks=model_risks,
                total_models=n_models,
            )

        # Step 4: consensus topology
        consensus = ConsensusTopology(
            domain=self.domain,
            nodes=consensus_nodes,
            total_models=n_models,
        )

        # Step 5: distances
        model_distances = {
            name: consensus.distance_from_model(name)
            for name in self.models
        }

        # Step 6: targeted perturbations
        perturbations = self._generate_perturbations(consensus)

        result = OracleResult(
            domain=self.domain,
            n_models=n_models,
            consensus=consensus,
            model_distances=model_distances,
            targeted_perturbations=perturbations,
            raw_risks=raw_risks,
        )

        # Step 7: registry
        if self.registry is not None:
            self.registry.update(result)

        return result

    # ── Perturbation generation ────────────────────────────────────────────────

    def _generate_perturbations(
        self,
        consensus: ConsensusTopology,
    ) -> list[TargetedPerturbation]:
        """Generate targeted perturbations for contested nodes.

        Ranked by expected information gain = variance × recurrence.
        High variance + high recurrence = most to learn from probing here.
        """
        probe_map = {p.node_id: p for p in self.probes}
        perturbations: list[TargetedPerturbation] = []

        for node in consensus.contested:
            probe = probe_map.get(node.node_id)
            if probe is None:
                continue

            mean = node.mean_risk
            diverging = sorted(
                [m for m, r in node.model_risks.items() if r > mean + 0.05],
                key=lambda m: -node.model_risks[m],
            )
            stable = sorted(
                [m for m, r in node.model_risks.items() if r <= mean + 0.05],
                key=lambda m: node.model_risks[m],
            )

            # Recommend framings that are most likely to surface the divergence:
            # pressure framings most likely to override reasoning at this junction
            recommended = self.PRESSURE_FRAMINGS[:4]

            info_gain = node.risk_variance * node.recurrence

            perturbations.append(TargetedPerturbation(
                node_id=node.node_id,
                description=node.description,
                question=probe.question,
                recommended_framings=recommended,
                diverging_models=diverging,
                stable_models=stable,
                expected_information_gain=info_gain,
            ))

        return sorted(perturbations, key=lambda p: -p.expected_information_gain)


# ─── Topology registry ────────────────────────────────────────────────────────

class TopologyRegistry:
    """Persistent store of consensus topology fragments, indexed by domain.

    Each oracle run contributes to the registry. Over time it accumulates
    an increasingly refined approximation of reality's topology across
    multiple domains and multiple models.

    This is the civilizational component of the framework: topology discovered
    by one set of finite systems persists and becomes available to future systems.
    Without a registry, each oracle run starts from scratch. With a registry,
    every run refines a shared, growing map.

    File format: JSON, human-readable, append-safe.
    """

    def __init__(self, path: Optional[str] = None):
        self._domains: dict[str, ConsensusTopology] = {}
        self._path = path
        if path:
            try:
                self._load(path)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

    # ── Public interface ───────────────────────────────────────────────────────

    def update(self, result: OracleResult) -> None:
        """Merge a new OracleResult into the registry."""
        domain = result.domain
        if domain not in self._domains:
            self._domains[domain] = result.consensus
        else:
            existing = self._domains[domain]
            for node_id, new_node in result.consensus.nodes.items():
                if node_id in existing.nodes:
                    existing.nodes[node_id].model_risks.update(new_node.model_risks)
                    existing.nodes[node_id].total_models = max(
                        existing.nodes[node_id].total_models,
                        new_node.total_models,
                    )
                else:
                    existing.nodes[node_id] = new_node
            existing.total_models = max(existing.total_models, result.n_models)
            existing.n_runs += 1
        if self._path:
            self._save(self._path)

    def get_consensus(self, domain: str) -> Optional[ConsensusTopology]:
        return self._domains.get(domain)

    def domains(self) -> list[str]:
        return list(self._domains.keys())

    def save(self, path: str) -> None:
        self._save(path)

    @classmethod
    def load(cls, path: str) -> TopologyRegistry:
        return cls(path)

    def report(self) -> str:
        if not self._domains:
            return "Registry is empty."
        lines = ["Topology Registry", "=" * 48]
        for domain, topo in self._domains.items():
            lines.append(f"\nDomain: {domain}")
            lines.append(f"  Models seen: {topo.total_models}   Runs: {topo.n_runs}")
            lines.append(
                f"  Confirmed: {len(topo.confirmed)}   "
                f"Contested: {len(topo.contested)}   "
                f"Systemic: {len(topo.systemic)}"
            )
            for n in topo.confirmed:
                lines.append(f"    ✓ {n.node_id}  (strength={n.consensus_strength:.2f})")
            for n in topo.systemic:
                lines.append(f"    ⚠ {n.node_id}  (mean_risk={n.mean_risk:.3f}, systemic)")
            for n in topo.contested:
                lines.append(f"    ? {n.node_id}  (variance={n.risk_variance:.3f})")
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self, path: str) -> None:
        data: dict = {}
        for domain, topo in self._domains.items():
            data[domain] = {
                "total_models": topo.total_models,
                "n_runs": topo.n_runs,
                "nodes": {
                    node_id: {
                        "description": n.description,
                        "model_risks": n.model_risks,
                        "total_models": n.total_models,
                    }
                    for node_id, n in topo.nodes.items()
                },
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        for domain, topo_data in data.items():
            nodes: dict[str, ConsensusNode] = {}
            for node_id, nd in topo_data["nodes"].items():
                nodes[node_id] = ConsensusNode(
                    node_id=node_id,
                    description=nd.get("description", ""),
                    model_risks=nd["model_risks"],
                    total_models=nd["total_models"],
                )
            self._domains[domain] = ConsensusTopology(
                domain=domain,
                nodes=nodes,
                total_models=topo_data["total_models"],
                n_runs=topo_data.get("n_runs", 1),
            )
