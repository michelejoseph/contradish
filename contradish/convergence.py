"""
convergence.py — Trajectory similarity and convergence efficiency analysis

Implements two derived theorems:

  Efficiency Theorem:
    Repeated admissible perturbation and repair discovers structurally stable
    distinctions more efficiently than repeated repair alone. Without
    perturbation, repair is local — it addresses only failure modes visible
    from the current framing. With perturbation, each cycle covers the full
    failure space, so the expected cycles-to-threshold is strictly lower.

  Trajectory Agreement Criterion:
    Independent reasoning systems should not merely agree on answers — they
    should follow statistically similar trajectories through reasoning space.
    Answer agreement is a necessary but not sufficient condition for
    reliability. Two systems that reach the same conclusion via different
    intermediate distinctions are coincidentally correct; they will diverge
    under perturbation because their structural commitments differ.

    Trajectory agreement is the stronger criterion. It is what genuine
    convergence toward Φ* looks like at the path level, not just the
    terminal state.

Usage::

    from contradish import CrossSystemAnalyzer, convergence_efficiency

    # Trajectory similarity between two systems
    analyzer = CrossSystemAnalyzer(
        systems={"system_a": model_a, "system_b": model_b},
        step_extractor=my_step_extractor,   # extracts reasoning steps from answer
        commitment_extractor=my_extractor,
    )
    result = analyzer.run(question, domain)
    print(result.report())
    # agreement:          True  (both conclude 1200 mg)
    # trajectory_similar: False (they arrive via different reasoning paths)
    # trajectory_sim:     0.31  (low path overlap despite answer agreement)

    # Efficiency comparison
    from contradish import convergence_efficiency
    report = convergence_efficiency(explorer, question, domain, epsilon=0.10)
    print(report.summary())
    # repair_only:           avg 4.2 cycles to reach ε < 0.10
    # perturbation_repair:   avg 2.1 cycles to reach ε < 0.10
    # efficiency_gain:       2.0x
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

from .phi_star import (
    PhiStarExplorer,
    FRAMING_PREFIXES,
    ALL_FRAMINGS,
    jaccard_similarity,
    first_sentence_extractor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningTrajectory:
    """
    A sequence of intermediate distinctions on the path to a conclusion.

    steps: ordered list of intermediate claims/distinctions the system made
           before reaching the conclusion. Extract these from chain-of-thought
           output, or from a step_extractor fn that parses the answer.

    conclusion: the final commitment (same as Trajectory.commitment in phi_star)

    Two trajectories are similar if they pass through the same distinctions
    in the same order — not merely if they share a conclusion.
    """
    system: str
    framing: str
    steps: list[str]        # intermediate reasoning steps
    conclusion: str         # final commitment
    raw_answer: str = ""    # full answer text


def lcs_length(
    a: list[str],
    b: list[str],
    sim_fn: Callable[[str, str], float],
    threshold: float = 0.60,
) -> int:
    """
    Longest common subsequence length, using semantic similarity for step
    matching (rather than exact string equality).

    Two steps are considered the same if sim_fn(a, b) >= threshold.
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if sim_fn(a[i - 1], b[j - 1]) >= threshold:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def trajectory_similarity(
    a: ReasoningTrajectory,
    b: ReasoningTrajectory,
    sim_fn: Callable[[str, str], float] = jaccard_similarity,
    step_threshold: float = 0.60,
) -> float:
    """
    Trajectory similarity: fraction of reasoning path shared between a and b.

    Computed as LCS(a.steps, b.steps) / max(|a.steps|, |b.steps|).
    A score of 1.0 means identical reasoning paths. A score near 0 means
    different structural commitments even if the conclusions agree.

    This is the formal implementation of the Trajectory Agreement Criterion:
    systems should not merely agree on answers, they should follow
    statistically similar paths through reasoning space.
    """
    if not a.steps and not b.steps:
        # No intermediate steps extracted — fall back to conclusion similarity
        return sim_fn(a.conclusion, b.conclusion)
    if not a.steps or not b.steps:
        return 0.0
    lcs = lcs_length(a.steps, b.steps, sim_fn, step_threshold)
    return lcs / max(len(a.steps), len(b.steps))


def population_trajectory_similarity(
    trajectories: list[ReasoningTrajectory],
    sim_fn: Callable[[str, str], float] = jaccard_similarity,
) -> float:
    """
    Mean pairwise trajectory similarity across a population of trajectories.

    A high score indicates that independent reasoning systems are following
    statistically similar paths — the strong criterion for reliability.
    A low score despite high answer agreement indicates coincidental correctness.
    """
    if len(trajectories) < 2:
        return 1.0
    sims = [
        trajectory_similarity(trajectories[i], trajectories[j], sim_fn)
        for i in range(len(trajectories))
        for j in range(i + 1, len(trajectories))
    ]
    return statistics.mean(sims) if sims else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-system analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SystemPairResult:
    """
    Comparison of two systems on one question+framing.

    answer_agreement:    True if both reach similar conclusions.
    trajectory_similar:  True if they follow similar paths (stronger criterion).
    trajectory_sim:      Raw trajectory similarity score.
    divergence_type:     Classification of the relationship.
    """
    system_a: str
    system_b: str
    framing: str
    conclusion_a: str
    conclusion_b: str
    steps_a: list[str]
    steps_b: list[str]
    answer_agreement: bool
    trajectory_sim: float
    trajectory_similar: bool   # trajectory_sim >= threshold

    @property
    def divergence_type(self) -> str:
        if self.answer_agreement and self.trajectory_similar:
            return "convergent"            # ideal — same path, same answer
        elif self.answer_agreement and not self.trajectory_similar:
            return "coincidental"          # same answer, different path — fragile
        elif not self.answer_agreement and self.trajectory_similar:
            return "terminal_divergence"   # same path, different conclusion — unusual
        else:
            return "divergent"             # different path, different answer


@dataclass
class CrossSystemResult:
    """
    Full cross-system analysis across all framings.
    """
    question: str
    domain: str
    system_names: list[str]
    pairs: list[SystemPairResult]

    @property
    def convergent_pairs(self) -> list[SystemPairResult]:
        return [p for p in self.pairs if p.divergence_type == "convergent"]

    @property
    def coincidental_pairs(self) -> list[SystemPairResult]:
        """Answer agreement via different paths — coincidentally correct."""
        return [p for p in self.pairs if p.divergence_type == "coincidental"]

    @property
    def divergent_pairs(self) -> list[SystemPairResult]:
        return [p for p in self.pairs if p.divergence_type == "divergent"]

    @property
    def mean_trajectory_similarity(self) -> float:
        if not self.pairs:
            return 0.0
        return statistics.mean(p.trajectory_sim for p in self.pairs)

    @property
    def answer_agreement_rate(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(1 for p in self.pairs if p.answer_agreement) / len(self.pairs)

    @property
    def trajectory_agreement_rate(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(1 for p in self.pairs if p.trajectory_similar) / len(self.pairs)

    @property
    def reliability_gap(self) -> float:
        """
        Fraction of answer-agreeing pairs that are coincidental (not trajectory-similar).
        High gap = systems are agreeing for different reasons = structural unreliability.
        """
        agreeing = [p for p in self.pairs if p.answer_agreement]
        if not agreeing:
            return 0.0
        coincidental = sum(1 for p in agreeing if not p.trajectory_similar)
        return coincidental / len(agreeing)

    def report(self) -> str:
        sep = "─" * 60
        lines = [
            sep,
            f"Cross-System Analysis · {' vs '.join(self.system_names)}",
            f"Domain: {self.domain}",
            f"Question: {self.question}",
            sep,
            f"Answer agreement rate:     {self.answer_agreement_rate:.0%}",
            f"Trajectory agreement rate: {self.trajectory_agreement_rate:.0%}",
            f"Mean trajectory similarity:{self.mean_trajectory_similarity:.2f}",
            f"Reliability gap:           {self.reliability_gap:.0%}",
            "",
        ]
        if self.reliability_gap > 0:
            lines.append(
                f"⚠ {self.reliability_gap:.0%} of answer-agreeing pairs are coincidental."
            )
            lines.append(
                "  Systems reach the same answer via different reasoning paths."
            )
            lines.append(
                "  These will diverge under perturbation. Trajectory agreement required."
            )
            lines.append("")

        by_type: dict[str, list[SystemPairResult]] = {}
        for p in self.pairs:
            by_type.setdefault(p.divergence_type, []).append(p)

        for dtype, label in [
            ("convergent", "CONVERGENT (same path + same answer)"),
            ("coincidental", "COINCIDENTAL (same answer, different path)"),
            ("terminal_divergence", "TERMINAL DIVERGENCE (same path, different answer)"),
            ("divergent", "DIVERGENT"),
        ]:
            group = by_type.get(dtype, [])
            if not group:
                continue
            lines.append(f"{label}: {len(group)} framing(s)")
            for p in group[:3]:
                lines.append(f"  [{p.framing}] sim={p.trajectory_sim:.2f}")
                lines.append(f"    {p.system_a}: {p.conclusion_a[:80]}")
                lines.append(f"    {p.system_b}: {p.conclusion_b[:80]}")
            if len(group) > 3:
                lines.append(f"  ... and {len(group) - 3} more")
            lines.append("")

        lines.append(sep)
        return "\n".join(lines)


class CrossSystemAnalyzer:
    """
    Runs two or more reasoning systems on the same question across all framings
    and compares them by trajectory similarity — not just answer agreement.

    The key distinction: systems that agree on answers via different reasoning
    paths are coincidentally correct. They share the terminal state but not the
    structural commitments. Under perturbation, they will diverge.

    Args:
        systems:              Dict of {name: model_fn}. Each model_fn takes
                              (system_prompt, question) -> answer string.
        step_extractor:       Callable[[question, answer], list[str]].
                              Extracts intermediate reasoning steps from an answer.
                              Use chain-of-thought output for best results.
                              Falls back to splitting on sentence boundaries.
        commitment_extractor: Callable[[question, answer], str].
                              Extracts final conclusion. Defaults to first sentence.
        similarity_fn:        Callable[[str, str], float]. Defaults to Jaccard.
        framing_types:        Framings to test. Defaults to all 16.
        answer_threshold:     Similarity threshold for answer agreement (default 0.60).
        trajectory_threshold: Similarity threshold for trajectory agreement (default 0.55).
        system_prompt:        Shared base system prompt.
    """

    def __init__(
        self,
        systems: dict[str, Callable[[str, str], str]],
        step_extractor: Optional[Callable[[str, str], list[str]]] = None,
        commitment_extractor: Optional[Callable[[str, str], str]] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        framing_types: Optional[list[str]] = None,
        answer_threshold: float = 0.60,
        trajectory_threshold: float = 0.55,
        system_prompt: str = "",
    ) -> None:
        self.systems = systems
        self.step_extractor = step_extractor or _sentence_step_extractor
        self.commitment_extractor = commitment_extractor or first_sentence_extractor
        self.similarity_fn = similarity_fn or jaccard_similarity
        self.framing_types = framing_types or ALL_FRAMINGS
        self.answer_threshold = answer_threshold
        self.trajectory_threshold = trajectory_threshold
        self.system_prompt = system_prompt

    def _frame(self, question: str, framing: str) -> str:
        prefix = FRAMING_PREFIXES.get(framing, "")
        return f"{prefix}{question}" if prefix else question

    def run(self, question: str, domain: str) -> CrossSystemResult:
        system_names = list(self.systems)
        if len(system_names) < 2:
            raise ValueError("CrossSystemAnalyzer requires at least 2 systems.")

        # Collect trajectories for all systems and framings
        all_trajectories: dict[str, dict[str, ReasoningTrajectory]] = {
            name: {} for name in system_names
        }
        for framing in self.framing_types:
            framed = self._frame(question, framing)
            for name, model_fn in self.systems.items():
                answer = model_fn(self.system_prompt, framed)
                steps = self.step_extractor(question, answer)
                conclusion = self.commitment_extractor(question, answer)
                all_trajectories[name][framing] = ReasoningTrajectory(
                    system=name,
                    framing=framing,
                    steps=steps,
                    conclusion=conclusion,
                    raw_answer=answer,
                )

        # Compare all pairs of systems across all framings
        pairs: list[SystemPairResult] = []
        for i, name_a in enumerate(system_names):
            for name_b in system_names[i + 1:]:
                for framing in self.framing_types:
                    traj_a = all_trajectories[name_a][framing]
                    traj_b = all_trajectories[name_b][framing]
                    traj_sim = trajectory_similarity(traj_a, traj_b, self.similarity_fn)
                    answer_sim = self.similarity_fn(traj_a.conclusion, traj_b.conclusion)
                    pairs.append(SystemPairResult(
                        system_a=name_a,
                        system_b=name_b,
                        framing=framing,
                        conclusion_a=traj_a.conclusion,
                        conclusion_b=traj_b.conclusion,
                        steps_a=traj_a.steps,
                        steps_b=traj_b.steps,
                        answer_agreement=answer_sim >= self.answer_threshold,
                        trajectory_sim=traj_sim,
                        trajectory_similar=traj_sim >= self.trajectory_threshold,
                    ))

        return CrossSystemResult(
            question=question,
            domain=domain,
            system_names=system_names,
            pairs=pairs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convergence efficiency comparison
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EfficiencyResult:
    """
    Comparison of repair-only vs perturbation+repair convergence rates.

    repair_only_cycles:      Distribution of cycles-to-threshold under repair alone.
    perturb_repair_cycles:   Distribution of cycles-to-threshold under perturb+repair.
    efficiency_gain:         Ratio of mean repair_only / mean perturb_repair.
                             > 1 means perturbation+repair is faster.
    """
    epsilon: float
    repair_only_cycles: list[int]
    perturb_repair_cycles: list[int]

    @property
    def mean_repair_only(self) -> float:
        return statistics.mean(self.repair_only_cycles) if self.repair_only_cycles else 0.0

    @property
    def mean_perturb_repair(self) -> float:
        return statistics.mean(self.perturb_repair_cycles) if self.perturb_repair_cycles else 0.0

    @property
    def efficiency_gain(self) -> float:
        if self.mean_perturb_repair == 0:
            return float("inf")
        return self.mean_repair_only / self.mean_perturb_repair

    def summary(self) -> str:
        sep = "─" * 60
        lines = [
            sep,
            f"Convergence Efficiency Comparison  (ε < {self.epsilon})",
            sep,
            f"Strategy            Avg cycles    Min    Max",
            "─" * 45,
        ]
        if self.repair_only_cycles:
            lines.append(
                f"Repair only         {self.mean_repair_only:>6.1f}      "
                f"  {min(self.repair_only_cycles)}      {max(self.repair_only_cycles)}"
            )
        if self.perturb_repair_cycles:
            lines.append(
                f"Perturbation+repair {self.mean_perturb_repair:>6.1f}      "
                f"  {min(self.perturb_repair_cycles)}      {max(self.perturb_repair_cycles)}"
            )
        lines.append("")
        lines.append(f"Efficiency gain: {self.efficiency_gain:.2f}x")
        lines.append("")
        lines.append(
            "Perturbation+repair is faster because each cycle covers the full"
        )
        lines.append(
            "failure space. Repair alone is local — it addresses only failure modes"
        )
        lines.append(
            "visible from the current framing and misses failures that require"
        )
        lines.append(
            "perturbation to surface. The efficiency gap widens as the number of"
        )
        lines.append(
            "independent failure modes increases.")
        lines.append(sep)
        return "\n".join(lines)


def convergence_efficiency(
    repair_only_fn: Callable[[int], float],
    perturb_repair_fn: Callable[[int], float],
    epsilon: float = 0.10,
    n_trials: int = 10,
    max_cycles: int = 20,
) -> EfficiencyResult:
    """
    Empirically measures convergence efficiency of two strategies.

    Args:
        repair_only_fn:     Callable(cycle_number) -> cai_strain_after_repair.
                            Should simulate repair without perturbation: only
                            addresses failures visible from the neutral framing.
        perturb_repair_fn:  Callable(cycle_number) -> cai_strain_after_repair.
                            Should simulate perturbation+repair: addresses all
                            failure modes discovered across all framings.
        epsilon:            Target strain threshold (default 0.10).
        n_trials:           Number of independent trials per strategy.
        max_cycles:         Maximum cycles before declaring non-convergence.

    Returns:
        EfficiencyResult with the distribution of cycles-to-threshold
        for each strategy and the efficiency gain ratio.
    """
    def cycles_to_threshold(strain_fn: Callable[[int], float]) -> int:
        for cycle in range(1, max_cycles + 1):
            strain = strain_fn(cycle)
            if strain < epsilon:
                return cycle
        return max_cycles  # did not converge within max_cycles

    repair_only_cycles = [
        cycles_to_threshold(repair_only_fn) for _ in range(n_trials)
    ]
    perturb_repair_cycles = [
        cycles_to_threshold(perturb_repair_fn) for _ in range(n_trials)
    ]

    return EfficiencyResult(
        epsilon=epsilon,
        repair_only_cycles=repair_only_cycles,
        perturb_repair_cycles=perturb_repair_cycles,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sentence_step_extractor(question: str, answer: str) -> list[str]:
    """
    Default step extractor: split answer into sentences, treating each as one
    reasoning step. For production use, replace with an LLM call that extracts
    chain-of-thought steps from structured reasoning output.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    # Filter out very short sentences (likely not reasoning steps)
    return [s.strip() for s in sentences if len(s.strip()) > 25]
