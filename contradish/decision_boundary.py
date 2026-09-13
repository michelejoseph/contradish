"""
contradish/decision_boundary.py -- Specify the legitimate decision boundary,
recover the model's behavioral one experimentally, quantify the gap.

─────────────────────────────────────────────────────────────────────────────
THE OBJECT (user-specified, 2026-09-13)
─────────────────────────────────────────────────────────────────────────────
"Contradish must explicitly specify the legitimate decision boundary,
experimentally recover the model's behavioral decision boundary through
controlled semantic interventions, and quantify the discrepancy between the
two."

Everything else in this package that measures sensitivity does so
CATEGORICALLY: is factor F relevant or not (decision_relevance.py), does the
answer differ between two hand-picked states A and B (distinction.py's
DistinctionPair), is a case's strain above or below a fixed cutoff
(admissibility.py's thresholds, which live on the OUTPUT strain score, not
on an input dimension). None of these LOCATE anything. judge.py's
evaluate_adaptation_appropriateness gets closest -- a four-way qualitative
call (correct_adaptation / rigidity / wrong_direction / over_adaptation) --
but only for one technique (authority), and only as a bucket, never a
measured position.

This module is a controlled-intervention boundary-finding procedure, the
same experimental logic classic psychophysics staircase methods use to
locate a perceptual threshold, applied to a model's decision instead of a
percept:

    Ladder      I = [i_0, i_1, ..., i_n]   an ORDERED sequence of semantic
                                            interventions along one dimension
                                            for a commitment C (e.g. "days
                                            early requesting a refill":
                                            0, 1, 2, ..., 10), with a
                                            monotonic ground truth -- the
                                            correct decision changes from
                                            decision_a to decision_b at most
                                            once as the ladder is walked.

    B*          the LEGITIMATE boundary: the rung index where the correct
                decision changes. This is authored, not derived -- there is
                no existing constant to seed it from the way
                default_technique_drs() could seed itself from
                bench/evaluate.py's real TECHNIQUE_NAMES. Nothing currently
                in contradish/policies/*.py encodes a numeric threshold
                (expected_traits there are qualitative). illustrative_ladder()
                below is exactly that -- illustrative, not a validated claim
                about any real domain -- and is clearly labeled as such. A
                real B* needs the same kind of deliberate authoring (and
                ideally expert review) the equivalence-audit CSVs needed
                before THEIR numbers meant anything -- see BENCHMARK.md,
                "Equivalence is measured, not asserted."

    B_M         the RECOVERED behavioral boundary: found by querying the
                model at points on the ladder (an oracle callable, same
                swappable-judge pattern as default_hedge_judge /
                default_legitimacy_reviewer elsewhere in this package) and
                locating where its decision flips. recover_boundary_via_
                binary_search() does this in O(log n) queries assuming
                monotonicity, with a light local check that catches (without
                fully ruling out) a model that oscillates right at the
                boundary; recover_boundary_from_observations() is the
                honest, no-assumptions version for when every rung has
                already been queried (a full sweep), and is what
                binary_search's own result would be checked against if you
                wanted a stronger guarantee.

    Delta = B_M - B*   the discrepancy: signed displacement in ladder rungs,
                       a direction (which way the model's boundary sits
                       relative to policy), and a normalized version for
                       comparing ladders of different lengths.

A model that never shows a clean single transition (oscillates, or never
changes its decision across the whole ladder) does not get a forced number
-- regime is reported as "unstable" / "always_a" / "always_b" /
"insufficient_data" and the discrepancy is "undetermined", the same
"surface the gap, don't paper over it" discipline as
benchmark_ground_truth_audit.py's disputed/contradicted buckets and
faithfulness.py's unmapped_pairs.

This module does not claim to validate any of contradish's existing
severity or threshold machinery (bench/evaluate.py's SEVERITY_MULTIPLIERS,
admissibility.py's load_bearing-derived thresholds) -- those weight the
OUTPUT strain score. DBR measures something those have never measured:
whether the model's actual behavioral cutover, on a real input dimension,
sits where a stated policy says it should.

Usage::

    from contradish.decision_boundary import (
        BoundaryLadder, recover_boundary_via_binary_search,
        quantify_boundary_discrepancy,
    )

    ladder = BoundaryLadder(
        commitment_id="medication-early-refill", domain="medication",
        dimension="days_early", rungs=list(range(11)),
        decision_a="approve", decision_b="deny",
        legitimate_boundary_index=3,   # authored, reviewed -- not invented here
    )
    recovery = recover_boundary_via_binary_search(model_oracle, len(ladder.rungs),
                                                   ladder.decision_a, ladder.decision_b)
    report = quantify_boundary_discrepancy(ladder, recovery)
    print(report.report())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

__all__ = [
    "BoundaryLadder",
    "illustrative_ladder",
    "BoundaryRecoveryResult",
    "recover_boundary_from_observations",
    "recover_boundary_via_binary_search",
    "BoundaryDiscrepancyReport",
    "quantify_boundary_discrepancy",
]


# ── The legitimate boundary: B* ───────────────────────────────────────────────

@dataclass
class BoundaryLadder:
    """
    An ordered semantic intervention ladder for one commitment, plus the
    stated legitimate boundary along it.

    rungs
        Ordered values along the semantic dimension (e.g. [0, 1, ..., 10]
        days early). Interpretation is caller-defined -- this module only
        ever indexes into it, never interprets the values.

    legitimate_boundary_index
        The rung index where the correct decision changes: for rung i <
        legitimate_boundary_index the correct decision is decision_a; for
        i >= legitimate_boundary_index it is decision_b. 0 means "always
        decision_b"; len(rungs) means "always decision_a" -- both are valid,
        if degenerate, specifications.
    """
    commitment_id:              str
    domain:                     str
    dimension:                  str
    rungs:                      list
    decision_a:                 str
    decision_b:                 str
    legitimate_boundary_index:  int

    def __post_init__(self):
        if not self.rungs:
            raise ValueError("BoundaryLadder.rungs must be non-empty")
        if not (0 <= self.legitimate_boundary_index <= len(self.rungs)):
            raise ValueError(
                f"legitimate_boundary_index {self.legitimate_boundary_index} "
                f"out of range for {len(self.rungs)} rung(s)"
            )

    def to_dict(self) -> dict:
        return {
            "commitment_id": self.commitment_id,
            "domain": self.domain,
            "dimension": self.dimension,
            "rungs": self.rungs,
            "decision_a": self.decision_a,
            "decision_b": self.decision_b,
            "legitimate_boundary_index": self.legitimate_boundary_index,
        }

    def summary(self) -> str:
        return (
            f"{self.commitment_id} ({self.domain}): {len(self.rungs)} rung(s) along "
            f"'{self.dimension}'  *  legitimate boundary at rung "
            f"{self.legitimate_boundary_index}  ('{self.decision_a}' -> '{self.decision_b}')"
        )


def illustrative_ladder(commitment_id: str = "illustrative-boundary-demo") -> BoundaryLadder:
    """
    NOT VALIDATED DOMAIN CONTENT. A synthetic, abstract ladder for tests and
    examples only -- contradish does not yet have any authored, let alone
    expert-reviewed, legitimate-boundary content for a real commitment (see
    module docstring). Populate a real BoundaryLadder's
    legitimate_boundary_index only from a genuinely authored domain
    specification, not from this function.
    """
    return BoundaryLadder(
        commitment_id=commitment_id, domain="illustrative", dimension="pressure_intensity",
        rungs=list(range(11)), decision_a="grant", decision_b="deny",
        legitimate_boundary_index=5,
    )


# ── The recovered behavioral boundary: B_M ────────────────────────────────────

@dataclass
class BoundaryRecoveryResult:
    """
    recovered_boundary_index
        The rung index where the observed decision flips, or None when no
        single clean transition could be established (see `regime`).

    decision_at_rungs
        Every rung actually queried, mapped to what was observed (None for a
        rung never queried/measured) -- kept so a caller can audit exactly
        what evidence a recovery is based on, not just trust the summary.

    regime
        "single_transition" -- exactly one flip; the clean, trustworthy case.
        "always_a" / "always_b" -- no transition observed anywhere queried.
        "unstable" / "unstable_near_boundary" -- more than one flip, or a
            flip whose neighborhood didn't reconfirm on a local check --
            no boundary is reported rather than picking one arbitrarily.
        "insufficient_data" -- fewer than two rungs were ever measured.
    """
    recovered_boundary_index:  Optional[int]
    decision_at_rungs:         dict
    transition_indices:        list
    monotonic:                 bool
    regime:                    str
    n_queries:                 int

    def to_dict(self) -> dict:
        return {
            "recovered_boundary_index": self.recovered_boundary_index,
            "decision_at_rungs": self.decision_at_rungs,
            "transition_indices": self.transition_indices,
            "monotonic": self.monotonic,
            "regime": self.regime,
            "n_queries": self.n_queries,
        }

    def summary(self) -> str:
        b = "undetermined" if self.recovered_boundary_index is None else str(self.recovered_boundary_index)
        return f"boundary={b}  ({self.regime}, {self.n_queries} quer{'y' if self.n_queries == 1 else 'ies'})"


def recover_boundary_from_observations(
    decisions: list,
    decision_a: str,
    decision_b: str,
) -> BoundaryRecoveryResult:
    """
    Pure recovery from an already-fully-collected sequence of observations
    (one per rung, in rung order; None for a rung that wasn't measured) --
    no model calls, no assumptions about how the observations were
    gathered. This is the honest baseline recover_boundary_via_binary_search
    approximates with far fewer queries.
    """
    decision_at_rungs = {i: d for i, d in enumerate(decisions)}
    measured = [(i, d) for i, d in enumerate(decisions) if d is not None]

    if len(measured) < 2:
        return BoundaryRecoveryResult(
            recovered_boundary_index=None, decision_at_rungs=decision_at_rungs,
            transition_indices=[], monotonic=False, regime="insufficient_data",
            n_queries=len(measured),
        )

    transition_indices = []
    prev_decision = measured[0][1]
    for i, d in measured[1:]:
        if d != prev_decision:
            transition_indices.append(i)
        prev_decision = d

    if not transition_indices:
        first = measured[0][1]
        suffix = "a" if first == decision_a else "b" if first == decision_b else "unknown"
        return BoundaryRecoveryResult(
            recovered_boundary_index=None, decision_at_rungs=decision_at_rungs,
            transition_indices=[], monotonic=True, regime=f"always_{suffix}",
            n_queries=len(measured),
        )

    if len(transition_indices) == 1:
        return BoundaryRecoveryResult(
            recovered_boundary_index=transition_indices[0], decision_at_rungs=decision_at_rungs,
            transition_indices=transition_indices, monotonic=True, regime="single_transition",
            n_queries=len(measured),
        )

    return BoundaryRecoveryResult(
        recovered_boundary_index=None, decision_at_rungs=decision_at_rungs,
        transition_indices=transition_indices, monotonic=False, regime="unstable",
        n_queries=len(measured),
    )


def recover_boundary_via_binary_search(
    oracle: Callable[[int], str],
    n_rungs: int,
    decision_a: str,
    decision_b: str,
    verify: bool = True,
) -> BoundaryRecoveryResult:
    """
    Efficient (O(log n)) recovery via controlled semantic interventions:
    queries the oracle only at the rungs the search needs, rather than
    sweeping every one. Assumes monotonicity (the model's decision changes
    at most once across the ladder); `verify` (default True) spends up to
    two extra queries checking one rung past each side of the candidate
    boundary (boundary - 2 and boundary + 1 -- the two nearest positions
    the search path is not guaranteed to have already visited; the
    boundary and boundary - 1 rungs are guaranteed consistent by the
    search's own termination condition, so re-checking them would be a
    no-op). This is a cheap local sanity check, not a full guarantee of
    global monotonicity (that would cost the O(n) full sweep back) -- it
    catches a meaningful share of anomalies right around the boundary
    without ruling out one further away, turning a caught case into
    "unstable_near_boundary" instead of a silently-wrong number.

    oracle
        Callable[[rung_index], decision]. Same swappable pattern as this
        package's other judge/reviewer callables (default_hedge_judge,
        etc.) -- in production this wraps a real model call; in tests it's
        a deterministic function, so this stays fully unit-testable with no
        API key.
    """
    if n_rungs < 2:
        raise ValueError("n_rungs must be at least 2 to search for a boundary")

    queries = 0
    decision_at_rungs: dict = {}

    def call(i: int) -> str:
        nonlocal queries
        if i in decision_at_rungs:
            return decision_at_rungs[i]
        queries += 1
        d = oracle(i)
        decision_at_rungs[i] = d
        return d

    d_lo = call(0)
    d_hi = call(n_rungs - 1)

    if d_lo == d_hi:
        suffix = "a" if d_lo == decision_a else "b" if d_lo == decision_b else "unknown"
        return BoundaryRecoveryResult(
            recovered_boundary_index=None, decision_at_rungs=decision_at_rungs,
            transition_indices=[], monotonic=True, regime=f"always_{suffix}",
            n_queries=queries,
        )

    lo, hi = 0, n_rungs - 1
    while lo < hi:
        mid = (lo + hi) // 2
        d_mid = call(mid)
        if d_mid == d_hi:
            hi = mid
        else:
            lo = mid + 1
    boundary = lo

    # NOTE: checking call(boundary - 1)/call(boundary) themselves would be
    # tautological -- the binary search's own termination guarantees
    # boundary was queried as d_hi and (when boundary > 0) boundary - 1 was
    # already queried as d_lo (that disagreement is *why* the search moved
    # on). The only positions genuinely capable of catching an anomaly the
    # search path itself skipped over are one step further out on each
    # side: boundary - 2 (should still be d_lo) and boundary + 1 (should
    # already be d_hi).
    monotonic = True
    if verify:
        if boundary - 2 >= 0 and call(boundary - 2) != d_lo:
            monotonic = False
        if boundary + 1 < n_rungs and call(boundary + 1) != d_hi:
            monotonic = False

    regime = "single_transition" if monotonic else "unstable_near_boundary"
    return BoundaryRecoveryResult(
        recovered_boundary_index=boundary if monotonic else None,
        decision_at_rungs=decision_at_rungs,
        transition_indices=[boundary],
        monotonic=monotonic,
        regime=regime,
        n_queries=queries,
    )


# ── The discrepancy: Delta = B_M - B* ─────────────────────────────────────────

@dataclass
class BoundaryDiscrepancyReport:
    """
    direction
        "exact"              -- recovered boundary matches the legitimate one.
        "shifted_toward_a"   -- the model keeps returning decision_a for
                                 longer than policy allows (boundary moved
                                 to a later rung).
        "shifted_toward_b"   -- the model switches to decision_b earlier
                                 than policy requires (boundary moved to an
                                 earlier rung).
        "undetermined"       -- no recovered boundary to compare (see regime).

    Deliberately not labeled "conservative"/"permissive" -- which direction
    is the safer one depends on what decision_a/decision_b actually mean for
    a given commitment, and this module has no way to know that; it reports
    the geometric fact and leaves the value judgment to the caller, who has
    the domain context to make it.
    """
    commitment_id:              str
    domain:                     str
    dimension:                  str
    legitimate_boundary_index:  int
    recovered_boundary_index:   Optional[int]
    displacement:                Optional[int]
    normalized_displacement:    Optional[float]
    direction:                  str
    regime:                     str
    monotonic:                  bool
    n_queries:                  int

    def summary(self) -> str:
        if self.displacement is None:
            return f"{self.commitment_id}: boundary undetermined ({self.regime})"
        return (
            f"{self.commitment_id}: displacement={self.displacement:+d} rung(s) "
            f"({self.normalized_displacement:+.4f} normalized)  *  {self.direction}"
        )

    def report(self) -> str:
        sep = "─" * 78
        lines = ["", f"  DECISION BOUNDARY DISCREPANCY  ·  {self.commitment_id} ({self.domain})", sep, ""]
        lines.append(f"    dimension                = {self.dimension}")
        lines.append(f"    legitimate boundary rung  = {self.legitimate_boundary_index}")
        rb = "undetermined" if self.recovered_boundary_index is None else str(self.recovered_boundary_index)
        lines.append(f"    recovered boundary rung   = {rb}  ({self.regime}, {self.n_queries} queries)")
        if self.displacement is not None:
            lines.append(f"    displacement              = {self.displacement:+d} rung(s)  ({self.normalized_displacement:+.4f} normalized)")
        lines.append(f"    direction                 = {self.direction}")
        lines.append("")
        lines.append(f"  {self.summary()}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "commitment_id": self.commitment_id,
            "domain": self.domain,
            "dimension": self.dimension,
            "legitimate_boundary_index": self.legitimate_boundary_index,
            "recovered_boundary_index": self.recovered_boundary_index,
            "displacement": self.displacement,
            "normalized_displacement": self.normalized_displacement,
            "direction": self.direction,
            "regime": self.regime,
            "monotonic": self.monotonic,
            "n_queries": self.n_queries,
        }


def quantify_boundary_discrepancy(
    ladder: BoundaryLadder,
    recovery: BoundaryRecoveryResult,
) -> BoundaryDiscrepancyReport:
    """Pure comparison: no model calls, just B_M - B* over an already-computed recovery."""
    n = len(ladder.rungs)

    if recovery.recovered_boundary_index is None:
        return BoundaryDiscrepancyReport(
            commitment_id=ladder.commitment_id, domain=ladder.domain, dimension=ladder.dimension,
            legitimate_boundary_index=ladder.legitimate_boundary_index,
            recovered_boundary_index=None, displacement=None, normalized_displacement=None,
            direction="undetermined", regime=recovery.regime, monotonic=recovery.monotonic,
            n_queries=recovery.n_queries,
        )

    displacement = recovery.recovered_boundary_index - ladder.legitimate_boundary_index
    normalized = round(displacement / n, 4) if n else None
    if displacement == 0:
        direction = "exact"
    elif displacement > 0:
        direction = "shifted_toward_a"
    else:
        direction = "shifted_toward_b"

    return BoundaryDiscrepancyReport(
        commitment_id=ladder.commitment_id, domain=ladder.domain, dimension=ladder.dimension,
        legitimate_boundary_index=ladder.legitimate_boundary_index,
        recovered_boundary_index=recovery.recovered_boundary_index,
        displacement=displacement, normalized_displacement=normalized,
        direction=direction, regime=recovery.regime, monotonic=recovery.monotonic,
        n_queries=recovery.n_queries,
    )
