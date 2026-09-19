"""
contradish/minimal_intervention_delta.py -- given intervention delta-I, what
is the smallest justified delta-B in behavior, and did the model produce
exactly that delta-B?

─────────────────────────────────────────────────────────────────────────────
TRIGGER (user-specified, 2026-09-17, verbatim)
─────────────────────────────────────────────────────────────────────────────
"Given intervention deltaI, what is the smallest justified deltaB in
behavior, and did the model produce exactly that deltaB?"

─────────────────────────────────────────────────────────────────────────────
WHERE THIS SITS AMONG THE PACKAGE'S EXISTING DEPENDENCY-STRUCTURE MACHINERY
─────────────────────────────────────────────────────────────────────────────
decision_relevance.py already answers a structurally identical question on
a different axis. DecisionRelevanceSpec(commitment_id, factors) classifies
EVERY named factor of ONE commitment into tracked/missed/spurious/invariant
against a stated relevance spec, and score_dependency_structure() is
already fully generic over what a "factor" IS -- nothing in that module
restricts a factor to bench/evaluate.py's 8 rhetorical techniques; that is
just the one seed function (default_technique_drs) shipped so far.

Flip which side plays "commitment" and which plays "factor," and the exact
same object answers this session's question with zero changes to that
module: `commitment_id` becomes the intervention's own id, and each
"factor" becomes one of the model's downstream behavioral commitments that
MIGHT be implicated by that intervention -- relevant if deltaI logically
necessitates a change there, irrelevant (equivalently: must remain
invariant) otherwise. `intervention_delta_spec()` below is that seed
function, parallel to `default_technique_drs`, so this usage is a
first-class, documented path instead of something a caller has to
discover is possible by reading decision_relevance.py's source.

What decision_relevance.py does NOT give you, on either axis, is what this
session's question actually asks for AS ITS OWN NAMED OBJECT: not a
continuous pooled rate (hit_rate, dependency_fidelity) but the literal
minimal set the model was supposed to change (`justified_delta`), the
literal set it actually changed (`actual_delta`), and a strict,
all-or-nothing verdict -- did `actual_delta` equal `justified_delta`
exactly, with every changed commitment landing on its correct new answer,
not just "how close." That strict criterion, and the two named delta sets
it is computed from, are what this module adds -- as a pure, no-new-
scoring-math layer directly on top of an existing DependencyStructureReport,
the same "bridge module, not a parallel measurement stack" discipline
chain_fidelity.py and directional_fidelity.py both already follow.

Usage::

    from contradish.minimal_intervention_delta import (
        intervention_delta_spec, score_minimal_delta,
    )
    from contradish.decision_relevance import score_dependency_structure

    spec = intervention_delta_spec(
        intervention_id="i485-pending-2026-09",
        domain="immigration",
        justified_commitments={"travel_advice": "advance parole required"},
        invariant_commitments=["fee_amount", "processing_office", "form_number"],
    )
    profile = {...}  # measured sensitivity per commitment, same shape
                      # score_dependency_structure() already expects
    report = score_dependency_structure(spec, profile, expected_effect_matches=...)
    verdict = score_minimal_delta(report)
    print(verdict.summary())
    print(verdict.exact_delta_match)             # membership-only
    print(verdict.exact_delta_match_with_direction)  # the full claim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from contradish.decision_relevance import (
    DRSFactor,
    DecisionRelevanceSpec,
    DependencyStructureReport,
)

__all__ = [
    "intervention_delta_spec",
    "MinimalDeltaVerdict",
    "score_minimal_delta",
    "MinimalDeltaAudit",
    "aggregate_minimal_delta",
]


def intervention_delta_spec(
    intervention_id: str,
    domain: str,
    justified_commitments: Union[dict[str, str], list[str]],
    invariant_commitments: list[str],
) -> DecisionRelevanceSpec:
    """
    Seed a DecisionRelevanceSpec for the "one intervention, many downstream
    commitments" axis -- the direct counterpart to default_technique_drs's
    "one commitment, many techniques" axis, reusing the exact same spec
    object and scoring core (score_dependency_structure() in
    decision_relevance.py; this function does not duplicate any of its
    relevance/sensitivity math).

    justified_commitments
        The commitments deltaI logically necessitates a change in -- this
        session's "smallest justified deltaB," named at construction time,
        not inferred from behavior. Either a plain list of commitment names
        (expected_effect left blank, so only membership -- not direction --
        can be scored) or a {name: expected_effect} dict when a
        direction-aware verdict (exact_delta_match_with_direction) is
        wanted -- the same expected_effect text DRSFactor already carries,
        resolved downstream the same way: a judge, a DistinctionPair, or a
        hand score decides True/False and passes it as
        expected_effect_matches to score_dependency_structure().

    invariant_commitments
        Every other commitment deltaI is asserted to NOT touch -- must
        remain exactly as it was. Listing these explicitly (rather than
        "everything not in justified_commitments") keeps the spec honest
        about what was actually considered for this intervention, the same
        discipline decision_relevance.py's own `unmeasured_factors` already
        enforces one level down for measurement (as opposed to
        specification) gaps.

    Raises ValueError if a commitment name appears in both lists -- an
    intervention cannot simultaneously require and forbid a change to the
    same commitment.
    """
    if isinstance(justified_commitments, dict):
        justified_items = list(justified_commitments.items())
    else:
        justified_items = [(name, "") for name in justified_commitments]

    factors: dict[str, DRSFactor] = {
        name: DRSFactor(name=name, relevance="relevant", expected_effect=effect)
        for name, effect in justified_items
    }
    for name in invariant_commitments:
        if name in factors:
            raise ValueError(
                f"{name!r} listed as both justified_commitments and "
                f"invariant_commitments for intervention {intervention_id!r}"
            )
        factors[name] = DRSFactor(name=name, relevance="irrelevant")

    return DecisionRelevanceSpec(commitment_id=intervention_id, domain=domain, factors=factors)


@dataclass
class MinimalDeltaVerdict:
    """
    Did the model produce EXACTLY the smallest justified deltaB for one
    intervention -- no more, no less, every change correct?

    justified_delta
        The smallest correct deltaB: every commitment deltaI actually
        necessitates a change in (spec-relevant AND measured), regardless
        of what the model did. == report.tracked | report.missed.

    actual_delta
        Every commitment the model actually changed (sensitivity over
        threshold), regardless of whether it should have.
        == report.tracked | report.spurious.

    excess_delta
        actual_delta - justified_delta: changed, shouldn't have.
        == report.spurious, renamed into this question's own vocabulary.

    deficit_delta
        justified_delta - actual_delta: should have changed, didn't.
        == report.missed, renamed the same way.

    exact_delta_match
        Membership-only verdict: excess_delta and deficit_delta both empty.
        Silent on whether the changes that DID happen landed correctly --
        see exact_delta_match_with_direction for the complete answer to
        this module's own question.

    exact_delta_match_with_direction
        The full claim: exact_delta_match AND every commitment in
        justified_delta landed on its expected_effect
        (report.tracked_wrong_direction empty too). None when the
        underlying report was never given direction data
        (report.true_hit_rate is None) -- the same "don't silently compute
        a misleading number over an unmeasured gap" rule the rest of this
        cluster (unmeasured_factors, true_hit_rate itself) already follows.
    """
    intervention_id:  str
    domain:           str
    justified_delta:  frozenset
    actual_delta:     frozenset
    excess_delta:     frozenset
    deficit_delta:    frozenset
    exact_delta_match: bool
    exact_delta_match_with_direction: Optional[bool] = None

    def summary(self) -> str:
        if self.exact_delta_match_with_direction is True:
            verdict = "EXACT (correct direction)"
        elif self.exact_delta_match_with_direction is False:
            verdict = "membership exact, direction wrong" if self.exact_delta_match else "NOT EXACT"
        else:
            verdict = "EXACT (membership only, direction unchecked)" if self.exact_delta_match else "NOT EXACT"
        s = (
            f"{self.intervention_id}: justified deltaB = {sorted(self.justified_delta)}  *  "
            f"actual deltaB = {sorted(self.actual_delta)}  *  {verdict}"
        )
        if self.excess_delta:
            s += f"  *  excess={sorted(self.excess_delta)}"
        if self.deficit_delta:
            s += f"  *  deficit={sorted(self.deficit_delta)}"
        return s

    def to_dict(self) -> dict:
        return {
            "intervention_id": self.intervention_id,
            "domain": self.domain,
            "justified_delta": sorted(self.justified_delta),
            "actual_delta": sorted(self.actual_delta),
            "excess_delta": sorted(self.excess_delta),
            "deficit_delta": sorted(self.deficit_delta),
            "exact_delta_match": self.exact_delta_match,
            "exact_delta_match_with_direction": self.exact_delta_match_with_direction,
        }


def score_minimal_delta(report: DependencyStructureReport) -> MinimalDeltaVerdict:
    """
    Pure layer over an existing DependencyStructureReport -- no new model
    calls, no new relevance/sensitivity math, just the strict minimality
    verdict this module's question asks for. Build the spec with
    intervention_delta_spec() and score it with
    decision_relevance.score_dependency_structure() first, then pass the
    resulting report straight in here.
    """
    tracked = frozenset(report.tracked)
    justified_delta = tracked | frozenset(report.missed)
    actual_delta = tracked | frozenset(report.spurious)
    excess_delta = frozenset(report.spurious)
    deficit_delta = frozenset(report.missed)
    exact_delta_match = not excess_delta and not deficit_delta

    exact_with_direction: Optional[bool] = None
    if report.true_hit_rate is not None:
        exact_with_direction = exact_delta_match and not report.tracked_wrong_direction

    return MinimalDeltaVerdict(
        intervention_id=report.commitment_id,
        domain=report.domain,
        justified_delta=justified_delta,
        actual_delta=actual_delta,
        excess_delta=excess_delta,
        deficit_delta=deficit_delta,
        exact_delta_match=exact_delta_match,
        exact_delta_match_with_direction=exact_with_direction,
    )


@dataclass
class MinimalDeltaAudit:
    """
    Pooled minimality verdict across many interventions.

    exact_match_rate
        Fraction of interventions where the model's deltaB exactly equaled
        the justified deltaB (membership only).

    exact_match_rate_with_direction
        Same, additionally requiring every changed commitment landed
        correctly. None unless at least one verdict carried direction data
        -- averaged only over the verdicts that actually have it, not
        silently treated as 0 for the rest.
    """
    domain:                             str
    n_interventions:                    int
    exact_match_rate:                   Optional[float]
    exact_match_rate_with_direction:    Optional[float]
    interventions_with_excess:          list
    interventions_with_deficit:         list
    interventions_with_wrong_direction: list = field(default_factory=list)
    by_intervention:                    dict = field(default_factory=dict)

    def summary(self) -> str:
        emr = "n/a" if self.exact_match_rate is None else f"{self.exact_match_rate:.4f}"
        emrd = "n/a" if self.exact_match_rate_with_direction is None else f"{self.exact_match_rate_with_direction:.4f}"
        return (
            f"{self.domain}: {self.n_interventions} intervention(s)  *  "
            f"exact_match_rate={emr}  exact_match_rate_with_direction={emrd}  *  "
            f"{len(self.interventions_with_excess)} with excess, "
            f"{len(self.interventions_with_deficit)} with deficit"
        )

    def report(self) -> str:
        sep = "-" * 78
        lines = ["", f"  MINIMAL INTERVENTION DELTA AUDIT  *  {self.domain}", sep, "", f"  {self.summary()}", ""]
        if self.interventions_with_excess:
            lines.append(f"  changed more than justified (excess deltaB): {self.interventions_with_excess}")
        if self.interventions_with_deficit:
            lines.append(f"  changed less than justified (deficit deltaB): {self.interventions_with_deficit}")
        if self.interventions_with_wrong_direction:
            lines.append(f"  changed exactly the right set but landed WRONG: {self.interventions_with_wrong_direction}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "n_interventions": self.n_interventions,
            "exact_match_rate": self.exact_match_rate,
            "exact_match_rate_with_direction": self.exact_match_rate_with_direction,
            "interventions_with_excess": self.interventions_with_excess,
            "interventions_with_deficit": self.interventions_with_deficit,
            "interventions_with_wrong_direction": self.interventions_with_wrong_direction,
            "by_intervention": {k: v.to_dict() for k, v in self.by_intervention.items()},
        }


def aggregate_minimal_delta(domain: str, verdicts: dict) -> MinimalDeltaAudit:
    """Pure aggregation: no model calls. verdicts is {intervention_id: MinimalDeltaVerdict}."""
    n = len(verdicts)
    n_exact = sum(1 for v in verdicts.values() if v.exact_delta_match)
    exact_match_rate = round(n_exact / n, 4) if n else None

    with_direction = [v for v in verdicts.values() if v.exact_delta_match_with_direction is not None]
    exact_match_rate_with_direction = (
        round(sum(1 for v in with_direction if v.exact_delta_match_with_direction) / len(with_direction), 4)
        if with_direction else None
    )

    return MinimalDeltaAudit(
        domain=domain,
        n_interventions=n,
        exact_match_rate=exact_match_rate,
        exact_match_rate_with_direction=exact_match_rate_with_direction,
        interventions_with_excess=sorted(k for k, v in verdicts.items() if v.excess_delta),
        interventions_with_deficit=sorted(k for k, v in verdicts.items() if v.deficit_delta),
        interventions_with_wrong_direction=sorted(
            k for k, v in verdicts.items()
            if v.exact_delta_match and v.exact_delta_match_with_direction is False
        ),
        by_intervention=verdicts,
    )
