"""
contradish/directional_fidelity.py -- closing the E-gap: did a relevant
factor move the decision to the CORRECT place, not just move it.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS FIXES (self-diagnosed, 2026-09-13)
─────────────────────────────────────────────────────────────────────────────
decision_relevance.py's DRSFactor.expected_effect has existed since that
module was written as "a future E-aware scorer, rather than silently
pretended not to exist" -- but nothing in the codebase ever READS it. It is
serialized in to_dict() and never consulted anywhere else. Grepping the repo
confirms this: `grep -rn '\.expected_effect' contradish/` turns up exactly
one hit, the to_dict() line itself.

The practical consequence: score_dependency_structure()'s "tracked" cell
means "a relevant factor's sensitivity crossed the threshold" -- it does
NOT mean the model's answer actually changed to the RIGHT new answer. A
model that flips to an arbitrary, wrong different answer under a relevant
factor change scores identically to one that flips to the correct one. That
is a real gap in a framework whose stated purpose is telling right-for-the-
right-reasons apart from everything else -- it currently cannot tell
right-direction from wrong-direction, only moved from didn't-move.

There is already a real, working mechanism for exactly this check, just not
connected to DRS: distinction.py's DistinctionMeasurement.both_correct,
computed as `com_a == pair.commit_a and com_b == pair.commit_b` for every
probe of a DistinctionPair. That is a genuine directional-correctness
signal, derived from real commit_a/commit_b ground truth an author already
wrote down for each pair -- it has simply never been read back against a
DRS classification. This module is that bridge, and nothing else: it takes
distinction.py's already-computed evidence and refines a "relevant, and
sensitive" classification into "relevant, sensitive, AND correct" vs.
"relevant, sensitive, but WRONG" -- reusing DistinctionPair/DistinctionProfile/
DistinctionMeasurement exactly as they are, adding zero new model-call
machinery.

Score
-----
For a given (pair, profile) where profile came from probing that pair:

    sensitivity               = profile.overall_hold_rate (already computed)
    directional_correctness   = fraction of profile.measurements with
                                 both_correct == True
    cell:
        "missed"                  sensitivity below threshold -- the
                                   distinction collapsed; DRS's existing
                                   "missed" cell, unchanged.
        "tracked_wrong_direction" sensitivity at/above threshold, but
                                   directional_correctness below threshold --
                                   the model DID respond to the relevant
                                   change, just not correctly. Previously
                                   indistinguishable from "tracked_correct".
        "tracked_correct"         sensitivity and directional_correctness
                                   both at/above threshold -- the only cell
                                   that should actually count as right for
                                   the right reasons.

This was originally built as a NEW, separate module rather than a
modification to decision_relevance.py's DependencyStructureReport/
score_dependency_structure, specifically because those were already
load-bearing (bench/evaluate.py's production wiring reads them directly)
and this signal only exists where a DistinctionPair has actually been
probed against a real model -- a small, growing subset of cases (see
predictive_validity.JUNCTION_CASE_MAP), not something to force into every
commitment's report.

As of 2026-09-13, decision_relevance.py's score_dependency_structure()
gained a backward-compatible, opt-in `expected_effect_matches` parameter
that does the same direction-aware split (tracked_correct/
tracked_wrong_direction) this module already computed for the narrower
DistinctionPair case, but for the FULL technique-factor picture, not just
one relevant factor. expected_effect_matches_from_reports() below is the
bridge: it turns this module's own DirectionalFidelityReports into the
{factor_name: True/False} shape that parameter accepts, so the two modules
now compose into one verdict rather than only running in parallel:

    reports = {pair_id: score_directional_fidelity(...), ...}
    matches = expected_effect_matches_from_reports(reports)
    full_report = score_dependency_structure(spec, sensitivity_profile,
                                              expected_effect_matches=matches)

score_directional_fidelity()/aggregate_directional_fidelity() above are
still the right tool when only the DistinctionPair-specific numbers
(sensitivity, directional_correctness, the pooled_directional_fidelity
rate) are wanted on their own, without the other 7-8 technique factors.

drs_factor_from_distinction_pair() / spec_with_distinction_pairs() are a
smaller, secondary piece: they give DRSFactor.expected_effect real content
(from the pair's own commit_a/commit_b) for the first time, for anyone who
wants a single, complete DecisionRelevanceSpec that includes both the 8
technique factors AND the case's real relevant-fact factor in one object.
They do not feed the scoring above, which reads commit_a/commit_b straight
from the DistinctionPair -- they exist so DRSFactor.report()/summary() can
finally print something real in the expected_effect slot instead of "".

Usage::

    from contradish.distinction import BUILTIN_DISTINCTION_PAIRS
    from contradish.predictive_validity import JUNCTION_CASE_MAP
    from contradish.directional_fidelity import (
        score_directional_fidelity, aggregate_directional_fidelity,
    )

    # profile = DistinctionProber(...).measure().profiles["healthy_vs_renal_dosing"]
    # (a real DistinctionProfile from probing a real model; see distinction.py)
    pair = BUILTIN_DISTINCTION_PAIRS["medication"][0]  # healthy_vs_renal_dosing
    report = score_directional_fidelity("medication-002", "medication", pair, profile)
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from contradish.decision_relevance import (
    DRSFactor, DecisionRelevanceSpec, default_technique_drs, score_dependency_structure,
)

__all__ = [
    "drs_factor_from_distinction_pair",
    "spec_with_distinction_pairs",
    "DirectionalFidelityReport",
    "score_directional_fidelity",
    "DirectionalFidelityAudit",
    "aggregate_directional_fidelity",
    "directional_fidelity_for_domain",
    "expected_effect_matches_from_reports",
]


# ── Giving expected_effect real content ──────────────────────────────────────

def drs_factor_from_distinction_pair(pair) -> DRSFactor:
    """
    A DRSFactor for the relevant fact a DistinctionPair encodes, with
    expected_effect populated from the pair's own commit_a/commit_b instead
    of left as "". Always relevance="relevant": a DistinctionPair is by
    definition a Type I (should-distinguish) pair -- see distinction.py.
    """
    return DRSFactor(
        name=pair.pair_id,
        relevance="relevant",
        expected_effect=(
            f"state A ({pair.label_a}) -> {pair.commit_a!r}; "
            f"state B ({pair.label_b}) -> {pair.commit_b!r}"
        ),
    )


def spec_with_distinction_pairs(commitment_id: str, domain: str, pairs: list) -> DecisionRelevanceSpec:
    """
    default_technique_drs() (the 8 irrelevant/conditional technique factors)
    plus one relevant factor per given DistinctionPair -- the first
    DecisionRelevanceSpec this package can build that actually has a
    populated relevant axis alongside the irrelevant one.
    """
    spec = default_technique_drs(commitment_id, domain=domain)
    for pair in pairs:
        spec.factors[pair.pair_id] = drs_factor_from_distinction_pair(pair)
    return spec


# ── The real scored signal ───────────────────────────────────────────────────

@dataclass
class DirectionalFidelityReport:
    """
    One DistinctionPair's classification, refined with directional
    correctness. See module docstring for the four possible `cell` values;
    "spurious"/"invariant" don't apply here -- a DistinctionPair's A/B axis
    is relevant by construction, so this only ever reports "missed",
    "tracked_wrong_direction", or "tracked_correct".
    """
    commitment_id:            str
    domain:                   str
    pair_id:                  str
    sensitivity:               float   # = profile.overall_hold_rate
    directional_correctness:   float   # fraction of measurements with both_correct True
    cell:                      str     # "missed" | "tracked_wrong_direction" | "tracked_correct"
    n_measurements:             int

    def summary(self) -> str:
        return (
            f"{self.commitment_id}/{self.pair_id}: sensitivity={self.sensitivity:.4f} "
            f"directional_correctness={self.directional_correctness:.4f} "
            f"(n={self.n_measurements}) -> {self.cell}"
        )

    def to_dict(self) -> dict:
        return {
            "commitment_id": self.commitment_id,
            "domain": self.domain,
            "pair_id": self.pair_id,
            "sensitivity": self.sensitivity,
            "directional_correctness": self.directional_correctness,
            "cell": self.cell,
            "n_measurements": self.n_measurements,
        }


def score_directional_fidelity(
    commitment_id: str,
    domain: str,
    pair_id: str,              # a DistinctionPair.pair_id (or the pair object itself -- see below)
    profile,                   # DistinctionProfile, already measured against a real model
    threshold: float = 0.5,
) -> DirectionalFidelityReport:
    """
    Pure scoring step: no model calls. `profile` must already be a measured
    DistinctionProfile (e.g. from DistinctionProber.measure().profiles[...])
    for the same pair `pair_id` names -- not checked here beyond reading
    profile.measurements, so pass a matching pair_id/profile.

    `pair_id` accepts either a plain string or a DistinctionPair object (its
    `.pair_id` attribute is used) -- both call sites in this module use one
    or the other, and forcing every caller to pass only a bare string bought
    nothing but a second, easy-to-drift copy of this same threshold logic
    (that duplication used to live in directional_fidelity_for_domain below;
    it now calls this function instead).

    CONSOLIDATION (2026-09-13): the relevant/sensitive/correct -> cell
    classification below is no longer re-derived inline -- it delegates to
    decision_relevance.score_dependency_structure(), the one canonical
    implementation of that predicate, via a single-factor spec (a
    DistinctionPair's A/B axis is relevant by construction, so this always
    scores exactly one always-relevant factor). This guarantees the two
    modules cannot drift on what "tracked_correct" vs "tracked_wrong_
    direction" means, the same way expected_effect_matches_from_reports()
    already guarantees it in the other direction. Output is bit-for-bit
    identical to the prior inline threshold logic (see
    tests/test_directional_fidelity.py) -- this is a pure refactor, not a
    behavior change.
    """
    resolved_pair_id = pair_id if isinstance(pair_id, str) else pair_id.pair_id
    sensitivity = round(profile.overall_hold_rate, 4)
    n = len(profile.measurements)
    directional_correctness = (
        round(sum(1 for m in profile.measurements if m.both_correct) / n, 4) if n else 0.0
    )

    single_factor_spec = DecisionRelevanceSpec(
        commitment_id=commitment_id, domain=domain,
        factors={resolved_pair_id: DRSFactor(name=resolved_pair_id, relevance="relevant")},
    )
    dep_report = score_dependency_structure(
        single_factor_spec,
        sensitivity_profile={resolved_pair_id: sensitivity},
        threshold=threshold,
        expected_effect_matches={resolved_pair_id: directional_correctness >= threshold},
    )
    if resolved_pair_id in dep_report.missed:
        cell = "missed"
    elif resolved_pair_id in dep_report.tracked_correct:
        cell = "tracked_correct"
    else:
        cell = "tracked_wrong_direction"

    return DirectionalFidelityReport(
        commitment_id=commitment_id,
        domain=domain,
        pair_id=resolved_pair_id,
        sensitivity=sensitivity,
        directional_correctness=directional_correctness,
        cell=cell,
        n_measurements=n,
    )


# ── Pooling across pairs/cases ───────────────────────────────────────────────

@dataclass
class DirectionalFidelityAudit:
    """
    Pooled across many DirectionalFidelityReports (e.g. every pair mapped
    to a domain via JUNCTION_CASE_MAP). Counts pooled, not averaged
    per-report -- same rationale as decision_relevance.DecisionRelevanceAudit.
    """
    domain:                          str
    n_factors:                       int
    tracked_correct:                 list[str]   # pair_ids
    tracked_wrong_direction:         list[str]
    missed:                          list[str]
    pooled_directional_fidelity:     Optional[float]   # tracked_correct / (tracked_correct + tracked_wrong_direction)
    by_pair:                         dict = field(default_factory=dict)

    def summary(self) -> str:
        fid = "n/a" if self.pooled_directional_fidelity is None else f"{self.pooled_directional_fidelity:.4f}"
        return (
            f"{self.domain}: {self.n_factors} factor(s)  *  "
            f"tracked_correct={len(self.tracked_correct)} "
            f"tracked_wrong_direction={len(self.tracked_wrong_direction)} "
            f"missed={len(self.missed)}  *  pooled_directional_fidelity={fid}"
        )

    def report(self) -> str:
        sep = "-" * 78
        lines = ["", f"  DIRECTIONAL FIDELITY AUDIT  ·  {self.domain}", sep, "", f"  {self.summary()}", ""]
        if self.tracked_wrong_direction:
            lines.append(f"  moved but to the WRONG place: {self.tracked_wrong_direction}")
        if self.missed:
            lines.append(f"  did not move at all: {self.missed}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "n_factors": self.n_factors,
            "tracked_correct": self.tracked_correct,
            "tracked_wrong_direction": self.tracked_wrong_direction,
            "missed": self.missed,
            "pooled_directional_fidelity": self.pooled_directional_fidelity,
            "by_pair": {k: v.to_dict() for k, v in self.by_pair.items()},
        }


def aggregate_directional_fidelity(domain: str, reports: dict) -> DirectionalFidelityAudit:
    """Pure aggregation: no model calls. reports is {pair_id: DirectionalFidelityReport}."""
    tracked_correct = sorted(k for k, r in reports.items() if r.cell == "tracked_correct")
    tracked_wrong = sorted(k for k, r in reports.items() if r.cell == "tracked_wrong_direction")
    missed = sorted(k for k, r in reports.items() if r.cell == "missed")

    n_tracked = len(tracked_correct) + len(tracked_wrong)
    fidelity = round(len(tracked_correct) / n_tracked, 4) if n_tracked else None

    return DirectionalFidelityAudit(
        domain=domain,
        n_factors=len(reports),
        tracked_correct=tracked_correct,
        tracked_wrong_direction=tracked_wrong,
        missed=missed,
        pooled_directional_fidelity=fidelity,
        by_pair=reports,
    )


def directional_fidelity_for_domain(
    domain: str,
    loss_map,                    # DistinctionLossMap, already measured (distinction.py)
    junction_case_map: dict,     # {pair_id: [case_id, ...]} -- e.g. predictive_validity.JUNCTION_CASE_MAP
    threshold: float = 0.5,
) -> DirectionalFidelityAudit:
    """
    Convenience wrapper: scores every pair in loss_map.profiles that also
    has a JUNCTION_CASE_MAP entry, using the first mapped case_id as the
    commitment_id (a pair mapped to multiple cases -- e.g.
    schedule_ii_vs_routine_refill -- is one directional-fidelity signal
    shared by all of them, not one per case; report per-case duplication
    would double-count a single measured pair in the pooled rate).
    Pairs present in loss_map but absent from junction_case_map are skipped
    (no case to attribute them to); pairs in junction_case_map but absent
    from loss_map (not yet probed) are skipped too, not silently zeroed.
    """
    reports: dict = {}
    for pair_id, profile in loss_map.profiles.items():
        case_ids = junction_case_map.get(pair_id)
        if not case_ids:
            continue
        # A pair mapped to multiple cases (e.g. schedule_ii_vs_routine_refill
        # -> [medication-001, medication-008]) is one measured signal, not
        # one per case; case_ids[0] is used as the report's commitment_id
        # deliberately, so pooling doesn't double-count it.
        reports[pair_id] = score_directional_fidelity(
            commitment_id=case_ids[0], domain=domain, pair_id=pair_id,
            profile=profile, threshold=threshold,
        )

    return aggregate_directional_fidelity(domain, reports)


# ── Bridge into decision_relevance.py's expected_effect_matches ─────────────

def expected_effect_matches_from_reports(reports: dict) -> dict:
    """
    Converts {pair_id: DirectionalFidelityReport} into the
    {factor_name: True | False} shape decision_relevance.score_dependency_
    structure()'s `expected_effect_matches` parameter accepts, so a
    DistinctionPair probed and scored through THIS module can also drive
    the direction-aware split in the general DRS scoring core, instead of
    the two only ever being compared side by side.

    Mapping, straight from each report's own `cell` (see module docstring):
        "tracked_correct"          -> True   (relevant, sensitive, correct)
        "tracked_wrong_direction"  -> False  (relevant, sensitive, WRONG)
        "missed"                   -> omitted. A "missed" factor was never
                                       sensitive in the first place, so it
                                       never reaches score_dependency_
                                       structure()'s `cell == "tracked"`
                                       branch -- there is no direction to
                                       report for a factor that didn't move,
                                       and score_dependency_structure()
                                       already classifies it "missed" on its
                                       own from the sensitivity profile
                                       alone. Including it here would be a
                                       silently-ignored, misleading entry,
                                       not a real correctness signal.

    A DirectionalFidelityReport's `pair_id` (not `commitment_id`) is used as
    the key: it is that value that must match the factor `name` in the
    DecisionRelevanceSpec being scored (see drs_factor_from_distinction_pair,
    which sets DRSFactor.name = pair.pair_id) -- commitment_id identifies
    which CASE the pair was probed against, not which FACTOR it is.
    """
    matches = {}
    for report in reports.values():
        if report.cell == "tracked_correct":
            matches[report.pair_id] = True
        elif report.cell == "tracked_wrong_direction":
            matches[report.pair_id] = False
        # "missed" -> omitted, see docstring above.
    return matches
