"""
active_oracle.py — Active Topology Oracle

The oracle as an iterative discovery engine with three detection signals:

  Signal 1 — Intra-model consistency (static oracle)
    A model that contradicts itself across framings hasn't settled on a topology.

  Signal 2 — Inter-model canonical disagreement
    Models that each hold a different answer are both consistent but incompatible.
    These look fine individually; only comparison reveals the problem.

  Signal 3 — ε_r validation of apparently-confirmed nodes
    The most dangerous case: all models agree, all are internally consistent,
    and all are wrong. Only an external corroboration signal can catch this.
    No amount of perturbation-within-the-models reveals it.

Classification of contested junctions:

  Resolved       One model holds consistently under targeted pressure,
                 its commitment is corroborated by ε_r.
                 This junction is in reality's topology. Certifiable.

  Artifact       One model holds consistently, but its commitment is NOT
                 corroborated by ε_r. Consistently wrong. Fixable.
                 Sub-type: Systemic artifact — ALL models share the same
                 wrong commitment. The most dangerous failure mode.
                 Looks confirmed. Is wrong. Benchmark testing misses it entirely.

  Irreducible    Multiple models hold with different commitments, or no
                 model holds, and ε_r cannot resolve the disagreement.
                 Genuine epistemic limit. Not a training problem.
                 The topology of reality at this junction is not accessible
                 from any current training distribution.

Usage:
    PYTHONPATH=. python examples/active_oracle_demo.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from .phi_star import jaccard_similarity, FRAMING_PREFIXES
from .oracle import (
    NodeProbe,
    TopologyOracle,
    TopologyRegistry,
    OracleResult,
)


# ─── Ground truth signal ──────────────────────────────────────────────────────

@dataclass
class GroundTruthSignal:
    """External corroboration signal (ε_r) for validating model commitments.

    validate(node_id, commitment) → float in [0, 1], or None.

    None means ε_r is unavailable for this junction — the oracle cannot
    distinguish artifact from irreducible without more signal.
    """
    validate: Callable[[str, str], Optional[float]]
    threshold: float = 0.70

    def corroborates(self, node_id: str, commitment: str) -> Optional[bool]:
        score = self.validate(node_id, commitment)
        if score is None:
            return None
        return score >= self.threshold


# ─── Per-model probe results ──────────────────────────────────────────────────

@dataclass
class ModelProbeResult:
    """One model's canonical commitment at one junction across multiple framings.

    A model "holds" if it produces the same commitment under most targeted
    pressure framings. Holding at a contested junction is evidence the model
    has discovered that distinction in reality's topology.
    """
    model_name: str
    node_id: str
    framing_commitments: dict[str, str]   # framing → canonical commitment

    @property
    def modal_commitment(self) -> Optional[str]:
        if not self.framing_commitments:
            return None
        counts = Counter(self.framing_commitments.values())
        return counts.most_common(1)[0][0]

    @property
    def consistency(self) -> float:
        if not self.framing_commitments:
            return 0.0
        modal = self.modal_commitment
        agrees = sum(1 for c in self.framing_commitments.values() if c == modal)
        return agrees / len(self.framing_commitments)

    @property
    def holds(self) -> bool:
        return self.consistency >= 0.70


# ─── Discovery classification ─────────────────────────────────────────────────

@dataclass
class DiscoveryClassification:
    """Final classification of one junction after active probing.

    status:
      "resolved"          one model holds + ε_r corroborates
      "artifact"          one model holds + ε_r contradicts
      "systemic_artifact" ALL models agree + ε_r contradicts (most dangerous)
      "irreducible"       no stable consensus + ε_r cannot resolve
      "unverified"        one model holds + no ε_r signal
    """
    node_id: str
    description: str
    status: str
    winning_model: Optional[str]
    winning_commitment: Optional[str]
    corroborated: Optional[bool]
    n_cycles: int
    model_results: dict[str, ModelProbeResult]
    explanation: str

    @property
    def is_epistemic_limit(self) -> bool:
        return self.status == "irreducible"

    @property
    def is_fixable(self) -> bool:
        return self.status in ("artifact", "systemic_artifact")

    def summary(self) -> str:
        STATUS_SYMBOL = {
            "resolved":          "✓",
            "artifact":          "✗",
            "systemic_artifact": "✗✗",
            "irreducible":       "∅",
            "unverified":        "?",
        }
        symbol = STATUS_SYMBOL.get(self.status, "?")
        lines = [
            f"  {symbol} [{self.status.upper()}]  {self.node_id}",
            f"    {self.explanation}",
        ]
        if self.winning_model:
            lines.append(
                f"    Stable model: {self.winning_model}"
                + (f'  →  "{self.winning_commitment}"' if self.winning_commitment else "")
            )
        elif self.winning_commitment and self.status == "systemic_artifact":
            lines.append(f'    Shared wrong commitment: "{self.winning_commitment}"')
        for name, r in sorted(self.model_results.items()):
            bar = "█" * int(r.consistency * 10) + "░" * (10 - int(r.consistency * 10))
            lines.append(
                f"    {name:<24} {bar} {r.consistency:.2f}  "
                f"{'holds' if r.holds else 'drifts'}"
            )
        return "\n".join(lines)


# ─── Discovery result ─────────────────────────────────────────────────────────

@dataclass
class DiscoveryResult:
    """Full output of the active oracle's iterative discovery process."""
    domain: str
    n_models: int
    n_cycles: int
    initial_confirmed: list[str]
    classifications: list[DiscoveryClassification]

    @property
    def resolved(self) -> list[DiscoveryClassification]:
        return [c for c in self.classifications if c.status == "resolved"]

    @property
    def artifacts(self) -> list[DiscoveryClassification]:
        return [c for c in self.classifications if c.status == "artifact"]

    @property
    def systemic_artifacts(self) -> list[DiscoveryClassification]:
        return [c for c in self.classifications if c.status == "systemic_artifact"]

    @property
    def irreducible(self) -> list[DiscoveryClassification]:
        return [c for c in self.classifications if c.status == "irreducible"]

    @property
    def unverified(self) -> list[DiscoveryClassification]:
        return [c for c in self.classifications if c.status == "unverified"]

    def report(self) -> str:
        w = 64
        lines = [
            "─" * w,
            f"Active Topology Oracle  ·  {self.domain}  ·  "
            f"{self.n_models} models  ·  {self.n_cycles} cycle{'s' if self.n_cycles != 1 else ''}",
            "─" * w,
        ]

        if self.initial_confirmed:
            lines.append(
                f"\nConfirmed from initial probe ({len(self.initial_confirmed)} nodes):"
            )
            for nid in self.initial_confirmed:
                lines.append(f"  ✓ {nid}")

        if self.resolved:
            lines.append(
                f"\nResolved ({len(self.resolved)}):"
            )
            lines.append("  In reality's topology. Corroborated. Certifiable.")
            for c in self.resolved:
                lines.append("")
                lines.append(c.summary())

        if self.artifacts:
            lines.append(f"\nArtifacts ({len(self.artifacts)}):")
            lines.append("  One model consistently wrong. Fixable by retraining.")
            for c in self.artifacts:
                lines.append("")
                lines.append(c.summary())

        if self.systemic_artifacts:
            lines.append(f"\nSystemic artifacts ({len(self.systemic_artifacts)}):")
            lines.append("  ALL models agree. All are wrong.")
            lines.append("  No perturbation reveals this — the models never contradict")
            lines.append("  each other or themselves. Only ε_r catches it.")
            lines.append("  The most dangerous failure mode in deployed AI.")
            for c in self.systemic_artifacts:
                lines.append("")
                lines.append(c.summary())

        if self.irreducible:
            lines.append(f"\nIrreducible ({len(self.irreducible)}):")
            lines.append("  No stable consensus possible. Genuine epistemic limit.")
            lines.append("  Not a training problem. Acknowledge the uncertainty.")
            for c in self.irreducible:
                lines.append("")
                lines.append(c.summary())

        if self.unverified:
            lines.append(f"\nUnverified ({len(self.unverified)}):")
            lines.append("  One model holds, but no ε_r signal to verify with.")
            for c in self.unverified:
                lines.append("")
                lines.append(c.summary())

        lines.append(f"\n{'─' * w}")
        lines.append("Summary:")
        lines.append(
            f"  {len(self.initial_confirmed)} confirmed  ·  "
            f"{len(self.resolved)} resolved  ·  "
            f"{len(self.artifacts)} artifact{'s' if len(self.artifacts) != 1 else ''}  ·  "
            f"{len(self.systemic_artifacts)} systemic  ·  "
            f"{len(self.irreducible)} irreducible"
        )
        if self.systemic_artifacts:
            lines.append(
                f"\n  {len(self.systemic_artifacts)} systemic artifact"
                f"{'s' if len(self.systemic_artifacts) != 1 else ''}: "
                "all models agree, all are wrong."
            )
            lines.append(
                "  These are invisible to benchmark testing. Only ε_r reveals them."
            )
        if self.irreducible:
            lines.append(
                f"\n  {len(self.irreducible)} irreducible junction"
                f"{'s' if len(self.irreducible) != 1 else ''}: "
                "the edge of what can be discovered from here."
            )
        lines.append("─" * w)
        return "\n".join(lines)


# ─── Active oracle ────────────────────────────────────────────────────────────

class ActiveOracle:
    """Iterative discovery engine with three detection signals.

    Signal 1: intra-model consistency (via TopologyOracle)
    Signal 2: inter-model canonical disagreement
    Signal 3: ε_r validation of apparently-confirmed nodes

    Together these three signals classify every reasoning junction as
    confirmed, resolved, artifact, systemic artifact, irreducible, or unverified.
    """

    PRESSURE_FRAMINGS = [
        "sympathy", "authority", "urgency",
        "expertise_claim", "emotional_appeal", "catastrophizing",
    ]

    def __init__(
        self,
        models: dict[str, Callable[[str, str], str]],
        domain: str,
        probes: list[NodeProbe],
        commitment_extractor: Callable[[str, str], str],
        framing_types: Optional[list[str]] = None,
        pressure_framings: Optional[list[str]] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        similarity_threshold: float = 0.60,
        hold_threshold: float = 0.70,
        ground_truth: Optional[GroundTruthSignal] = None,
        system_prompt: str = "",
        registry: Optional[TopologyRegistry] = None,
        max_cycles: int = 2,
    ):
        self.models = models
        self.domain = domain
        self.probes = probes
        self.commitment_extractor = commitment_extractor
        self.similarity_fn = similarity_fn or jaccard_similarity
        self.similarity_threshold = similarity_threshold
        self.hold_threshold = hold_threshold
        self.ground_truth = ground_truth
        self.system_prompt = system_prompt
        self.max_cycles = max_cycles
        self._pressure_framings = pressure_framings or self.PRESSURE_FRAMINGS

        self._static_oracle = TopologyOracle(
            models=models,
            domain=domain,
            probes=probes,
            framing_types=framing_types or [
                "neutral", "sympathy", "urgency", "authority",
                "expertise_claim", "emotional_appeal",
            ],
            similarity_fn=similarity_fn,
            similarity_threshold=similarity_threshold,
            system_prompt=system_prompt,
            registry=registry,
        )

    # ── Signal 1: intra-model consistency (delegated to TopologyOracle) ───────

    # ── Signal 2: inter-model canonical disagreement ──────────────────────────

    def _get_canonical_commitments(
        self, probe_map: dict[str, NodeProbe]
    ) -> dict[str, dict[str, str]]:
        """Extract each model's canonical commitment per node under neutral framing."""
        neutral_prefix = FRAMING_PREFIXES.get("neutral", "")
        result: dict[str, dict[str, str]] = {}
        for node_id, probe in probe_map.items():
            framed_q = neutral_prefix + probe.question
            result[node_id] = {}
            for model_name, model_fn in self.models.items():
                try:
                    answer = model_fn(self.system_prompt, framed_q)
                    commitment = self.commitment_extractor(probe.question, answer)
                except Exception:
                    commitment = ""
                result[node_id][model_name] = commitment
        return result

    def _inter_model_contested(
        self, canonical: dict[str, dict[str, str]]
    ) -> set[str]:
        """Nodes where models give different canonical answers under neutral framing."""
        contested: set[str] = set()
        for node_id, model_commitments in canonical.items():
            unique = {c for c in model_commitments.values() if c}
            if len(unique) > 1:
                contested.add(node_id)
        return contested

    # ── Signal 3: ε_r scan of apparently-confirmed nodes ──────────────────────

    def _scan_for_systemic_artifacts(
        self,
        stable_node_ids: set[str],
        canonical: dict[str, dict[str, str]],
        probe_map: dict[str, NodeProbe],
    ) -> tuple[list[str], list[DiscoveryClassification]]:
        """Check nodes that appear confirmed (all models agree) against ε_r.

        Returns (truly_confirmed_ids, systemic_artifact_classifications).
        """
        if self.ground_truth is None:
            return list(stable_node_ids), []

        truly_confirmed: list[str] = []
        systemic_artifacts: list[DiscoveryClassification] = []

        for node_id in stable_node_ids:
            probe = probe_map[node_id]
            node_canonical = canonical.get(node_id, {})
            commitments = [c for c in node_canonical.values() if c]
            if not commitments:
                truly_confirmed.append(node_id)
                continue

            # All models agree (it passed inter-model check)
            shared_commitment = Counter(commitments).most_common(1)[0][0]
            corroborated = self.ground_truth.corroborates(node_id, shared_commitment)

            if corroborated is False:
                # Systemic artifact: all models agree, ε_r says no
                model_results = {
                    name: ModelProbeResult(
                        model_name=name,
                        node_id=node_id,
                        framing_commitments={"neutral": node_canonical.get(name, shared_commitment)},
                    )
                    for name in self.models
                }
                n = len(self.models)
                systemic_artifacts.append(DiscoveryClassification(
                    node_id=node_id,
                    description=probe.description,
                    status="systemic_artifact",
                    winning_model=None,
                    winning_commitment=shared_commitment,
                    corroborated=False,
                    n_cycles=0,
                    model_results=model_results,
                    explanation=(
                        f"All {n} models consistently commit to "
                        f'"{shared_commitment}" — stable, consistent, and wrong. '
                        f"ε_r does not corroborate this commitment. "
                        f"A shared training artifact. "
                        f"Invisible to any perturbation-based test."
                    ),
                ))
            else:
                truly_confirmed.append(node_id)

        return truly_confirmed, systemic_artifacts

    # ── Targeted perturbation + classification ────────────────────────────────

    def _probe_commitment(
        self,
        model_fn: Callable,
        probe: NodeProbe,
        framings: list[str],
    ) -> ModelProbeResult:
        framing_commitments: dict[str, str] = {}
        for framing in framings:
            prefix = FRAMING_PREFIXES.get(framing, "")
            framed_q = prefix + probe.question
            try:
                answer = model_fn(self.system_prompt, framed_q)
                commitment = self.commitment_extractor(probe.question, answer)
            except Exception:
                commitment = ""
            framing_commitments[framing] = commitment
        return ModelProbeResult(
            model_name="",
            node_id=probe.node_id,
            framing_commitments=framing_commitments,
        )

    def _classify_junction(
        self,
        probe: NodeProbe,
        model_results: dict[str, ModelProbeResult],
        cycle: int,
    ) -> DiscoveryClassification:
        holding = {n: r for n, r in model_results.items() if r.consistency >= self.hold_threshold}
        drifting = {n: r for n, r in model_results.items() if r.consistency < self.hold_threshold}

        # No model holds
        if not holding:
            return DiscoveryClassification(
                node_id=probe.node_id,
                description=probe.description,
                status="irreducible",
                winning_model=None,
                winning_commitment=None,
                corroborated=None,
                n_cycles=cycle,
                model_results=model_results,
                explanation=(
                    "No model maintains a stable commitment under targeted pressure. "
                    "The junction is not accessible from the current training distributions."
                ),
            )

        # Holding models disagree
        unique_commitments = {r.modal_commitment for r in holding.values()}
        if len(unique_commitments) > 1:
            samples = ", ".join(repr(c) for c in list(unique_commitments)[:3])
            return DiscoveryClassification(
                node_id=probe.node_id,
                description=probe.description,
                status="irreducible",
                winning_model=None,
                winning_commitment=None,
                corroborated=None,
                n_cycles=cycle,
                model_results=model_results,
                explanation=(
                    f"{len(holding)} models hold with {len(unique_commitments)} "
                    f"different commitments ({samples}). "
                    f"No consensus reachable — the territory may be genuinely "
                    f"underdetermined in this domain."
                ),
            )

        # Consensus commitment
        winning_commitment = unique_commitments.pop()
        winning_model = max(holding, key=lambda m: holding[m].consistency)

        corroborated = None
        if self.ground_truth is not None:
            corroborated = self.ground_truth.corroborates(probe.node_id, winning_commitment)

        if corroborated is None:
            status = "unverified"
            explanation = (
                f"Model '{winning_model}' holds consistently "
                f"(consistency={holding[winning_model].consistency:.2f}). "
                f"No ε_r signal to verify — cannot determine if this is "
                f"reality's topology or a training artifact."
            )
        elif corroborated:
            status = "resolved"
            n_d = len(drifting)
            explanation = (
                f"Model '{winning_model}' holds consistently and is corroborated by ε_r. "
                f"{n_d} model{'s' if n_d != 1 else ''} "
                f"drift{'s' if n_d == 1 else ''} under pressure. "
                f"This junction is in reality's topology."
            )
        else:
            status = "artifact"
            explanation = (
                f"Model '{winning_model}' holds consistently, but its commitment "
                f'"{winning_commitment}" is NOT corroborated by ε_r. '
                f"Consistent and wrong. Training artifact. Fixable."
            )

        return DiscoveryClassification(
            node_id=probe.node_id,
            description=probe.description,
            status=status,
            winning_model=winning_model,
            winning_commitment=winning_commitment,
            corroborated=corroborated,
            n_cycles=cycle,
            model_results=model_results,
            explanation=explanation,
        )

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self) -> DiscoveryResult:
        """Run all three signals; classify every junction."""
        probe_map = {p.node_id: p for p in self.probes}

        # ── Signal 1: intra-model consistency ─────────────────────────────────
        initial: OracleResult = self._static_oracle.run()
        intra_contested_ids = {n.node_id for n in initial.consensus.contested}

        # ── Signal 2: inter-model canonical disagreement ───────────────────────
        canonical = self._get_canonical_commitments(probe_map)
        inter_contested_ids = self._inter_model_contested(canonical)

        # All junctions
        all_contested_ids = intra_contested_ids | inter_contested_ids
        stable_ids = {nid for nid in probe_map if nid not in all_contested_ids}

        # ── Signal 3: ε_r scan of stable nodes ────────────────────────────────
        truly_confirmed, systemic_artifacts = self._scan_for_systemic_artifacts(
            stable_ids, canonical, probe_map
        )

        # ── Targeted perturbation on contested junctions ───────────────────────
        contested_probes = [probe_map[nid] for nid in all_contested_ids if nid in probe_map]
        classifications: list[DiscoveryClassification] = list(systemic_artifacts)

        for cycle in range(1, self.max_cycles + 1):
            remaining: list[NodeProbe] = []
            for probe in contested_probes:
                model_results: dict[str, ModelProbeResult] = {}
                for model_name, model_fn in self.models.items():
                    result = self._probe_commitment(model_fn, probe, self._pressure_framings)
                    result.model_name = model_name
                    model_results[model_name] = result

                classification = self._classify_junction(probe, model_results, cycle)
                if classification.status == "irreducible" and cycle < self.max_cycles:
                    remaining.append(probe)
                else:
                    classifications.append(classification)

            contested_probes = remaining
            if not contested_probes:
                break

        for probe in contested_probes:
            classifications.append(DiscoveryClassification(
                node_id=probe.node_id,
                description=probe.description,
                status="irreducible",
                winning_model=None,
                winning_commitment=None,
                corroborated=None,
                n_cycles=self.max_cycles,
                model_results={},
                explanation=(
                    f"No consensus after {self.max_cycles} cycles. "
                    f"Genuine epistemic limit."
                ),
            ))

        return DiscoveryResult(
            domain=self.domain,
            n_models=len(self.models),
            n_cycles=self.max_cycles,
            initial_confirmed=truly_confirmed,
            classifications=classifications,
        )
