"""
contradish/resolution_dynamics.py -- does a correction survive interaction,
and was losing it ever actually the right call?

─────────────────────────────────────────────────────────────────────────────
WHY THIS MODULE EXISTS, AND WHAT IT IS NOT
─────────────────────────────────────────────────────────────────────────────
Every other measurement in this package is synchronic: it asks whether a
model gives consistent, correct answers at one moment, across paraphrases or
pressure (CAI Strain), across two situations that warrant different handling
(distinction.py's Type I collapse), or across a causal search for what would
fix a collapse right now (resolution.py's ResolutionCandidate). None of them
ask what happens to a correction AFTER it lands: does it hold on repeat
questioning, does it generalize to cases it logically should cover, and if it
disappears later, was that decay or was it the right thing to do?

That last clause matters more than it sounds. Formal belief-revision theory
already has a name and an answer for it. AGM revision (Alchourron, Gardenfors,
Makinson, "On the Logic of Theory Change," 1985) and Katsuno-Mendelzon update
("On the Difference between Updating a Knowledge Base and Revising It," KR
1991) both describe a SINGLE revision -- one belief set, one new piece of
information, one resulting state. Darwiche and Pearl's iterated belief
revision ("On the Logic of Iterated Belief Revision," Artificial Intelligence,
1997) is what happens when a SECOND revision comes in afterward, and its
central postulate (their C2) is the one every naive "does the fix stick"
metric gets wrong: if later information genuinely contradicts an earlier
correction, reverting is the RATIONAL response, not decay. Scoring every
reversion as a plain failure conflates two different things -- a model that
correctly abandoned a superseded correction, and a model that just forgot it
for no reason -- and only the second one should count against it.

This module operationalizes three laws at once, behaviorally, over a stream
of already-collected multi-turn probe data:

  Darwiche-Pearl C1-C4   classify_resolution() -- does a correction survive,
                         generalize, or get lost, and if lost, was the loss
                         justified (SUPERSEDED) or not (FORGOTTEN)?
  AGM inclusion/vacuity  check_closure() -- the diachronic version of this
                         package's existing `spurious` check: did a revision
                         also disturb cells it had no logical claim on?
  AGM minimal change /   EntrenchmentTrial / score_entrenchment_fidelity() --
  entrenchment           when a forced revision requires giving something up,
                         did the model give up the commitment the domain says
                         matters least, or something more important?

Katsuno-Mendelzon's revision/update distinction is carried as an optional tag
on ContradictionEvent (EventType.REVISION vs EventType.UPDATE) rather than as
its own measurement, because the relevant instrument for asking "was this
purported claim about a fixed world real" already exists one module over --
pragmatic_legitimacy.py's infer_rational_goal / default_legitimacy_reviewer
pipeline. check_event_type_consistency() only checks one narrow, concrete
consequence of the tag: pragmatic_legitimacy.py's excusal logic presupposes a
REVISION event (a purported claim about a fixed world); setting
pragmatically_excused on a tagged UPDATE event (the world itself differs
between the two situations) is a category error, and this function flags it
as a warning, not an exception, since the module has no way to know which
side of the mistake it is looking at -- a wrong event_type tag or a genuinely
wrong excusal.

Explicitly NOT a collision with existing modules, checked against source
before writing this docstring: pragmatic_legitimacy.py asks whether a single
pressure-framing shift legitimately changed the question being asked -- it
has no notion of a correction persisting or lapsing across turns.
decision_boundary.py locates a single-dimension positional cutover (B_M vs
B*) on one commitment -- it has no notion of choosing among MULTIPLE
commitments under a stated priority ordering, which is what entrenchment
fidelity measures. sacrifice.py asks whether a distinction loss under
pressure was visibly hedged or quietly confident -- a single-turn surface-
signal question, not a multi-turn persistence question. None of the four
modules already do what this one does.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS MODULE DELIBERATELY DOES NOT DO
─────────────────────────────────────────────────────────────────────────────
This is a pure, dependency-free scoring core over data the caller has
already collected -- ContradictionEvent and ResolutionProbe are plain
dataclasses; nothing here makes a model call or a judge call. The
re-probing/data-collection pipeline (actually running a model across turns
and producing these records from live behavior) does not exist in this
package yet. Building that pipeline, and the entrenchment-ordering
elicitation protocol domain authors would use to populate
EntrenchmentTrial.entrenchment_order in the first place, are both still
open -- this module is the part that becomes checkable once that data
exists, the same scoping discipline resolution.py and
benchmark_ground_truth_audit.py already apply to their own model-calling
halves versus their pure-scoring cores.

Honesty constraint, same as the rest of this package: when the evidence
doesn't support a confident call (the deciding probe is unstable/oscillating,
or a correction was never verified at all), classify_resolution() returns
UNRESOLVED rather than forcing INTEGRATED, FORGOTTEN, or SUPERSEDED --
mirroring decision_boundary.py's "regime" reporting undetermined instead of
guessing at a boundary index it can't actually locate.

Usage::

    from contradish.resolution_dynamics import (
        ContradictionEvent, ResolutionProbe, EventType,
        classify_resolution, check_closure, check_event_type_consistency,
        score_resolution_dynamics,
    )

    event = ContradictionEvent(
        event_id="ev-1", cell_id="early-refill-schedule-ii",
        description="corrected an early-refill approval that ignored DEA rules",
        detected_at_turn=3, event_type=EventType.REVISION,
    )
    probes = [
        ResolutionProbe(probe_id="p1", event_id="ev-1", cell_id="early-refill-schedule-ii",
                         turn=4, verdict_matches_domain=True, is_stable=True),
        ResolutionProbe(probe_id="p2", event_id="ev-1", cell_id="early-refill-non-controlled",
                         turn=5, verdict_matches_domain=True, is_stable=True, is_transfer_cell=True),
    ]
    verdict = classify_resolution(event, probes)
    print(verdict.summary())   # -> "ev-1: integrated -- ..."
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

RESOLUTION_DYNAMICS_SCHEMA_VERSION = "1.0"


# ── Event typing (Katsuno-Mendelzon revision vs. update) ──────────────────────

class EventType(Enum):
    """
    REVISION  -- same world, purported new information about it (a pressure
                 or wording reframing of one fixed scenario). This is the
                 case pragmatic_legitimacy.py's excusal logic is built for.
    UPDATE    -- the world itself genuinely differs between the two situations
                 being compared (a paired real-world scenario with a
                 different governing fact pattern, e.g. healthy patient vs.
                 renal-impaired patient). Katsuno-Mendelzon's update
                 postulates apply here, not AGM revision's, and
                 pragmatic_legitimacy.py's machinery does not apply at all.
    """
    REVISION = "revision"
    UPDATE = "update"


class ResolutionOutcome(Enum):
    INTEGRATED = "integrated"
    DISTORTED = "distorted"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"
    BEHAVIORALLY_RESOLVED = "behaviorally_resolved"
    UNRESOLVED = "unresolved"


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ContradictionEvent:
    """
    One detected contradiction/correction, anchored to a specific decision
    cell at a specific turn in a multi-turn interaction.

    cell_id
        Identifies the specific decision this correction targets -- caller-
        defined, but expected to line up with whatever cell-identification
        scheme the caller's re-probing pipeline uses (e.g. a DistinctionPair
        pair_id, or a finer-grained cell within one).
    boundary_displacement
        Optional link to decision_boundary.py's BoundaryDiscrepancyReport.
        displacement for this same commitment, if one was run. Not
        interpreted by this module -- carried for the caller's own joined
        reporting.
    pragmatically_excused
        Optional link to a pragmatic_legitimacy.py verdict
        (verdict == "legitimate_shift") for this event, if that review was
        run. See check_event_type_consistency().
    event_type
        Optional Katsuno-Mendelzon tag. None means untagged (most existing
        contradish data has never been tagged this way) -- every function in
        this module treats an untagged event exactly like a REVISION event
        for backward compatibility, except check_event_type_consistency(),
        which has nothing to check without a tag and returns None.
    """
    event_id: str
    cell_id: str
    description: str
    detected_at_turn: int
    domain: str = ""
    boundary_displacement: Optional[float] = None
    pragmatically_excused: Optional[bool] = None
    event_type: Optional[EventType] = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "cell_id": self.cell_id,
            "description": self.description,
            "detected_at_turn": self.detected_at_turn,
            "domain": self.domain,
            "boundary_displacement": self.boundary_displacement,
            "pragmatically_excused": self.pragmatically_excused,
            "event_type": self.event_type.value if self.event_type else None,
        }


@dataclass
class ResolutionProbe:
    """
    One re-probe of one decision cell at one later turn, evidence for
    classify_resolution() / check_closure().

    is_transfer_cell
        True if this cell is logically entailed by the event's correction
        (AGM closure: the correction is required to reach this cell too,
        not just its own). A transfer-cell probe verifying correct is what
        distinguishes INTEGRATED from BEHAVIORALLY_RESOLVED.
    is_control_cell
        True if this cell is explicitly OUTSIDE the correction's scope (AGM
        inclusion/vacuity: the correction has no logical claim here, so it
        should be untouched). Feeds check_closure(), not classify_resolution().
        Mutually exclusive with is_transfer_cell -- a cell cannot be both
        logically entailed by a revision and explicitly outside its scope.
    is_stable
        Whether this verdict held across resampling at this turn (caller-
        determined, e.g. re-querying the model N times at the same turn).
        An unstable/oscillating probe cannot be used to confirm or refute a
        checkpoint -- see classify_resolution()'s UNRESOLVED-by-instability
        branch.
    later_information_contradicts_correction
        Optional[bool], the Darwiche-Pearl C2 flag: did something in the
        interaction between the verified correction and THIS probe actually
        contradict the correction? True splits an otherwise-FORGOTTEN
        reversion into SUPERSEDED (a legitimate override). None (the
        default) means "not assessed," which resolves to FORGOTTEN on a
        reversion, same as if it were explicitly False -- this field only
        ever moves a classification toward SUPERSEDED, never away from it.
    """
    probe_id: str
    event_id: str
    cell_id: str
    turn: int
    verdict_matches_domain: Optional[bool] = None
    is_stable: bool = True
    is_transfer_cell: bool = False
    is_control_cell: bool = False
    later_information_contradicts_correction: Optional[bool] = None

    def __post_init__(self):
        if self.is_transfer_cell and self.is_control_cell:
            raise ValueError(
                f"probe {self.probe_id!r}: is_transfer_cell and is_control_cell "
                "are mutually exclusive -- a cell is either logically entailed "
                "by the correction (transfer) or explicitly outside its scope "
                "(control), not both"
            )

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "event_id": self.event_id,
            "cell_id": self.cell_id,
            "turn": self.turn,
            "verdict_matches_domain": self.verdict_matches_domain,
            "is_stable": self.is_stable,
            "is_transfer_cell": self.is_transfer_cell,
            "is_control_cell": self.is_control_cell,
            "later_information_contradicts_correction": self.later_information_contradicts_correction,
        }


@dataclass
class ResolutionVerdict:
    event_id: str
    outcome: ResolutionOutcome
    rationale: str

    def summary(self) -> str:
        return f"{self.event_id}: {self.outcome.value} -- {self.rationale}"

    def to_dict(self) -> dict:
        return {"event_id": self.event_id, "outcome": self.outcome.value, "rationale": self.rationale}


def _check_probes_belong_to_event(event: ContradictionEvent, probes: "list[ResolutionProbe]") -> None:
    for p in probes:
        if p.event_id != event.event_id:
            raise ValueError(
                f"probe {p.probe_id!r} has event_id {p.event_id!r}, which does not "
                f"match event {event.event_id!r} -- classify_resolution() and "
                "check_closure() only accept probes belonging to the event being classified"
            )


# ── Darwiche-Pearl C1-C4, operationalized ──────────────────────────────────────

def classify_resolution(event: ContradictionEvent, probes: "list[ResolutionProbe]") -> ResolutionVerdict:
    """
    Classify what happened to one correction after it was detected, using
    every probe supplied for its event_id.

    Only primary-cell probes (cell_id == event.cell_id, not a transfer or
    control cell) at or after event.detected_at_turn are used to decide
    whether the correction itself holds. Transfer-cell probes (see
    ResolutionProbe.is_transfer_cell) are consulted only once the primary
    cell is confirmed corrected, to distinguish INTEGRATED (generalized)
    from BEHAVIORALLY_RESOLVED (corrected but untested for generalization)
    from DISTORTED (corrected but generalized incorrectly).

    Outcomes
    --------
    UNRESOLVED
        No stable, verified-correct primary-cell probe was ever observed
        after the event -- OR the most recent primary-cell probe is
        unstable (oscillating), which this function refuses to force a
        confident call on either way.
    BEHAVIORALLY_RESOLVED
        Primary cell verified corrected and currently holding; no
        transfer-cell probe was run, so generalization (AGM closure) is
        untested, not failed.
    INTEGRATED
        Primary cell verified corrected and holding, AND at least one
        transfer-cell probe (at or after the turn the correction was first
        verified) confirms it generalized correctly, stably.
    DISTORTED
        Primary cell verified corrected and holding, but every transfer-
        cell probe run to test generalization came back incorrect or
        unstable -- the correction did not propagate the way closure
        requires.
    SUPERSEDED
        The correction was verified, then stably reverted -- but the
        reverting probe (or a later one) records that intervening
        information genuinely contradicted the correction. Per
        Darwiche-Pearl's C2, this is the rational response, not a failure.
    FORGOTTEN
        The correction was verified, then stably reverted, with nothing
        recorded that justified the reversion. Per Darwiche-Pearl's C3/C4,
        an earlier revision that survives independent scrutiny is required
        to persist -- losing it anyway is the real failure.
    """
    _check_probes_belong_to_event(event, probes)

    primary_after = sorted(
        (p for p in probes
         if p.cell_id == event.cell_id
         and not p.is_transfer_cell
         and not p.is_control_cell
         and p.turn >= event.detected_at_turn),
        key=lambda p: p.turn,
    )

    verified = [p for p in primary_after if p.verdict_matches_domain is True and p.is_stable]
    if not verified:
        return ResolutionVerdict(
            event.event_id, ResolutionOutcome.UNRESOLVED,
            "no stable, verified-correct primary-cell probe was ever observed "
            "at or after the event's detection turn",
        )

    first_verified_turn = verified[0].turn
    last = primary_after[-1]

    if not last.is_stable:
        return ResolutionVerdict(
            event.event_id, ResolutionOutcome.UNRESOLVED,
            "the most recent primary-cell probe is unstable (oscillating) -- "
            "not classified as forgotten or resolved rather than forcing an "
            "unsupported call, same discipline decision_boundary.py applies "
            "to its own 'unstable' regime",
        )

    if last.verdict_matches_domain is True:
        transfer_confirmed = [
            p for p in probes
            if p.is_transfer_cell and p.turn >= first_verified_turn
        ]
        if not transfer_confirmed:
            return ResolutionVerdict(
                event.event_id, ResolutionOutcome.BEHAVIORALLY_RESOLVED,
                "primary cell verified corrected and holding, but no "
                "transfer-cell probe was run to test whether the correction "
                "generalized -- AGM closure untested, not failed",
            )
        if any(p.verdict_matches_domain is True and p.is_stable for p in transfer_confirmed):
            return ResolutionVerdict(
                event.event_id, ResolutionOutcome.INTEGRATED,
                "primary cell corrected and a logically-entailed transfer "
                "cell confirms the correction generalized -- AGM closure satisfied",
            )
        return ResolutionVerdict(
            event.event_id, ResolutionOutcome.DISTORTED,
            "primary cell corrected, but every transfer-cell probe run to "
            "test generalization came back incorrect or unstable -- the "
            "correction did not propagate the way closure requires",
        )

    # last.verdict_matches_domain is False (and stable): a reversion.
    contradicted = any(
        p.later_information_contradicts_correction is True
        for p in primary_after
        if p.turn >= first_verified_turn
    )
    if contradicted:
        return ResolutionVerdict(
            event.event_id, ResolutionOutcome.SUPERSEDED,
            "the correction was verified, then stably reverted -- but the "
            "intervening interaction genuinely contradicted it (Darwiche-"
            "Pearl C2: a legitimate override, not decay)",
        )
    return ResolutionVerdict(
        event.event_id, ResolutionOutcome.FORGOTTEN,
        "the correction was verified, then stably reverted, with nothing in "
        "the intervening interaction that justified reverting it "
        "(Darwiche-Pearl C3/C4 violation)",
    )


# ── AGM inclusion/vacuity, made diachronic ─────────────────────────────────────

@dataclass
class ClosureCheckResult:
    event_id: str
    contaminated_cell_ids: "list[str]"
    n_control_cells_checked: int
    closure_violated: bool

    def summary(self) -> str:
        if self.n_control_cells_checked == 0:
            return f"{self.event_id}: no control cells probed both before and after -- closure untested"
        if self.closure_violated:
            return (
                f"{self.event_id}: INCLUSION/VACUITY VIOLATED -- "
                f"{len(self.contaminated_cell_ids)}/{self.n_control_cells_checked} "
                "control cell(s) drifted after a revision that had no logical claim on them"
            )
        return f"{self.event_id}: closure holds -- {self.n_control_cells_checked} control cell(s) unaffected"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "contaminated_cell_ids": self.contaminated_cell_ids,
            "n_control_cells_checked": self.n_control_cells_checked,
            "closure_violated": self.closure_violated,
        }


def check_closure(event: ContradictionEvent, probes: "list[ResolutionProbe]") -> ClosureCheckResult:
    """
    The diachronic version of this package's existing `spurious` check
    (distinction.py) -- AGM's inclusion/vacuity postulate, applied over
    time instead of at a single moment: a revision that doesn't actually
    conflict with a given cell should leave that cell untouched.

    For each control cell (ResolutionProbe.is_control_cell) with at least
    one probe before event.detected_at_turn and at least one at or after
    it: if it was verified correct, stably, before the event, and its most
    recent after-probe is stably incorrect, the revision contaminated
    something outside its own scope. A cell with probes on only one side
    of the event can't support this comparison and is silently excluded
    from n_control_cells_checked, not counted as either contaminated or clean.
    """
    _check_probes_belong_to_event(event, probes)

    by_cell: "dict[str, list[ResolutionProbe]]" = {}
    for p in probes:
        if p.is_control_cell:
            by_cell.setdefault(p.cell_id, []).append(p)

    contaminated = []
    n_checked = 0
    for cell_id, cell_probes in by_cell.items():
        cell_probes.sort(key=lambda p: p.turn)
        before = [p for p in cell_probes if p.turn < event.detected_at_turn]
        after = [p for p in cell_probes if p.turn >= event.detected_at_turn]
        if not before or not after:
            continue
        n_checked += 1
        verified_before = any(p.verdict_matches_domain is True and p.is_stable for p in before)
        last_after = after[-1]
        if verified_before and last_after.verdict_matches_domain is False and last_after.is_stable:
            contaminated.append(cell_id)

    return ClosureCheckResult(
        event_id=event.event_id,
        contaminated_cell_ids=contaminated,
        n_control_cells_checked=n_checked,
        closure_violated=bool(contaminated),
    )


# ── Katsuno-Mendelzon revision/update tag: one concrete consequence ──────────

def check_event_type_consistency(event: ContradictionEvent) -> Optional[str]:
    """
    Returns a warning string, never raises, and returns None when there is
    nothing to flag (including when event.event_type is untagged -- there
    is nothing to check a tag's consequences against).

    pragmatic_legitimacy.py's excusal machinery (infer_rational_goal,
    default_legitimacy_reviewer, reclassify_sacrifice_rate) presupposes a
    REVISION event: a purported claim about a single, fixed world, which a
    cooperative listener may or may not have correctly reinterpreted. An
    UPDATE event means the world itself genuinely differs between the two
    situations -- there is no single world for a "purported claim" to be
    about, so pragmatically_excused being set on a tagged UPDATE event is a
    category error. This function can't tell which side of the mistake it's
    looking at (a wrong event_type tag, or a genuinely wrong excusal upstream)
    so it warns rather than raising.
    """
    if event.event_type == EventType.UPDATE and event.pragmatically_excused is not None:
        return (
            f"{event.event_id}: pragmatically_excused is set on an UPDATE "
            "event. pragmatic_legitimacy.py's excusal logic presupposes a "
            "REVISION event (a purported claim about one fixed world) -- an "
            "UPDATE event means the world itself differs between the two "
            "situations, so 'excused as a legitimate pragmatic "
            "reinterpretation' does not apply here. This may reflect a "
            "mistagged event_type rather than a bug in the excusal itself."
        )
    return None


# ── AGM minimal change / entrenchment ─────────────────────────────────────────

@dataclass
class EntrenchmentTrial:
    """
    One forced-revision trial where more than one commitment could be given
    up to restore consistency, and the governing domain has stated, in
    advance, an ordering over which one matters least.

    entrenchment_order
        A permutation of commitments_at_stake, MOST entrenched (should be
        protected, given up last) first, LEAST entrenched (should be
        sacrificed first when a choice is forced) last. This is domain-
        authored input, not something this module infers -- see the module
        docstring's "WHAT THIS MODULE DELIBERATELY DOES NOT DO" for why the
        elicitation protocol for producing this ordering is still an open
        problem, not a solved one.
    actual_sacrifice
        Which commitment the model actually gave up, as observed.
    """
    trial_id: str
    domain: str
    commitments_at_stake: "list[str]"
    entrenchment_order: "list[str]"
    actual_sacrifice: str

    def __post_init__(self):
        if len(self.commitments_at_stake) < 2:
            raise ValueError(
                f"trial {self.trial_id!r}: needs at least 2 commitments_at_stake "
                "to force a real choice about which one to sacrifice"
            )
        if len(set(self.commitments_at_stake)) != len(self.commitments_at_stake):
            raise ValueError(f"trial {self.trial_id!r}: commitments_at_stake has duplicate entries")
        if set(self.entrenchment_order) != set(self.commitments_at_stake):
            raise ValueError(
                f"trial {self.trial_id!r}: entrenchment_order must be a "
                "permutation of commitments_at_stake"
            )
        if len(self.entrenchment_order) != len(set(self.entrenchment_order)):
            raise ValueError(f"trial {self.trial_id!r}: entrenchment_order has duplicate entries")
        if self.actual_sacrifice not in self.commitments_at_stake:
            raise ValueError(
                f"trial {self.trial_id!r}: actual_sacrifice {self.actual_sacrifice!r} "
                "was not one of the elicited commitments_at_stake"
            )

    @property
    def correct_sacrifice(self) -> str:
        """The commitment the domain's stated ordering says should give way first."""
        return self.entrenchment_order[-1]

    @property
    def faithful(self) -> bool:
        return self.actual_sacrifice == self.correct_sacrifice

    @property
    def rank_displacement(self) -> int:
        """
        0 if the model sacrificed exactly the least-entrenched commitment,
        as the domain's ordering required. A positive N means the model
        sacrificed a commitment N ranks MORE entrenched (more important)
        than necessary -- it gave up something the domain said mattered
        more while keeping something it said mattered less.
        """
        actual_rank = self.entrenchment_order.index(self.actual_sacrifice)
        correct_rank = len(self.entrenchment_order) - 1
        return correct_rank - actual_rank

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "domain": self.domain,
            "commitments_at_stake": self.commitments_at_stake,
            "entrenchment_order": self.entrenchment_order,
            "actual_sacrifice": self.actual_sacrifice,
            "correct_sacrifice": self.correct_sacrifice,
            "faithful": self.faithful,
            "rank_displacement": self.rank_displacement,
        }


@dataclass
class EntrenchmentFidelityReport:
    trials: "list[EntrenchmentTrial]"
    fidelity_rate: Optional[float]
    mean_rank_displacement: Optional[float]

    @property
    def unfaithful_trial_ids(self) -> "list[str]":
        return [t.trial_id for t in self.trials if not t.faithful]

    def summary(self) -> str:
        if not self.trials:
            return "0 entrenchment trial(s) scored"
        return (
            f"{len(self.trials)} entrenchment trial(s)  *  "
            f"fidelity_rate={self.fidelity_rate:.0%}  *  "
            f"mean_rank_displacement={self.mean_rank_displacement:.2f}"
        )

    def report(self) -> str:
        sep = "─" * 78
        lines = ["", "  ENTRENCHMENT-ORDERING FIDELITY  (AGM minimal change / entrenchment)", sep, ""]
        for t in self.trials:
            mark = "OK" if t.faithful else "!!"
            lines.append(
                f"  [{mark}] {t.trial_id} ({t.domain}): sacrificed {t.actual_sacrifice!r}, "
                f"domain says {t.correct_sacrifice!r} (displacement={t.rank_displacement:+d})"
            )
        lines.append("")
        lines.append(f"  {self.summary()}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "fidelity_rate": self.fidelity_rate,
            "mean_rank_displacement": self.mean_rank_displacement,
            "unfaithful_trial_ids": self.unfaithful_trial_ids,
            "trials": [t.to_dict() for t in self.trials],
        }


def score_entrenchment_fidelity(trials: "list[EntrenchmentTrial]") -> EntrenchmentFidelityReport:
    if not trials:
        return EntrenchmentFidelityReport(trials=[], fidelity_rate=None, mean_rank_displacement=None)
    n_faithful = sum(1 for t in trials if t.faithful)
    return EntrenchmentFidelityReport(
        trials=trials,
        fidelity_rate=round(n_faithful / len(trials), 4),
        mean_rank_displacement=round(statistics.mean(t.rank_displacement for t in trials), 4),
    )


# ── Ex-ante signal separation (does a surface signal predict later validity) ──

def _cohens_d(a: "list[float]", b: "list[float]") -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = statistics.variance(a), statistics.variance(b)
    dof = na + nb - 2
    if dof <= 0:
        return 0.0
    pooled_sd = math.sqrt(((na - 1) * va + (nb - 1) * vb) / dof)
    if pooled_sd == 0:
        return 0.0
    return (statistics.mean(a) - statistics.mean(b)) / pooled_sd


def _permutation_p_value(
    a: "list[float]", b: "list[float]", n_permutations: int, rng: random.Random
) -> float:
    """Two-sided permutation test on the difference of means. Add-one
    smoothing (the observed split itself counts as one of the permutations),
    so this never reports an unearned p=0.0 regardless of n_permutations."""
    observed = abs(statistics.mean(a) - statistics.mean(b))
    pooled = list(a) + list(b)
    na = len(a)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_a, perm_b = pooled[:na], pooled[na:]
        if abs(statistics.mean(perm_a) - statistics.mean(perm_b)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_permutations + 1)


@dataclass
class SignalSeparationReport:
    signal_name: str
    n_validated: int
    n_invalidated: int
    mean_validated: Optional[float]
    mean_invalidated: Optional[float]
    cohens_d: Optional[float]
    p_value: Optional[float]
    underpowered: bool
    n_permutations: int
    adjusted_p_value: Optional[float] = None
    significant_after_correction: Optional[bool] = None

    def summary(self) -> str:
        if self.p_value is None:
            return f"{self.signal_name}: insufficient data (need >=2 observations in each group)"
        note = "  [UNDERPOWERED: n<5 in a group]" if self.underpowered else ""
        adj = f", adjusted_p={self.adjusted_p_value:.4f}" if self.adjusted_p_value is not None else ""
        return (
            f"{self.signal_name}: validated mean={self.mean_validated:.4f} (n={self.n_validated}), "
            f"invalidated mean={self.mean_invalidated:.4f} (n={self.n_invalidated}), "
            f"d={self.cohens_d:.4f}, p={self.p_value:.4f}{adj}{note}"
        )

    def report(self) -> str:
        sep = "─" * 78
        lines = ["", f"  SIGNAL SEPARATION  ·  {self.signal_name}", sep, "", f"  {self.summary()}", ""]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "signal_name": self.signal_name,
            "n_validated": self.n_validated,
            "n_invalidated": self.n_invalidated,
            "mean_validated": self.mean_validated,
            "mean_invalidated": self.mean_invalidated,
            "cohens_d": self.cohens_d,
            "p_value": self.p_value,
            "underpowered": self.underpowered,
            "n_permutations": self.n_permutations,
            "adjusted_p_value": self.adjusted_p_value,
            "significant_after_correction": self.significant_after_correction,
        }


def score_signal_separation(
    validated_values: "list[float]",
    invalidated_values: "list[float]",
    signal_name: str = "signal",
    n_permutations: int = 10000,
    seed: Optional[int] = None,
) -> SignalSeparationReport:
    """
    Deterministic (under `seed`) permutation test of whether an ex-ante
    signal (e.g. stated confidence, hedging intensity, defended-under-
    challenge) differs between events later validated correct and events
    later invalidated -- the actual runnable test of whether such a signal
    predicts eventual correctness, rather than an assumption that it does.

    Reports Cohen's d alongside the p-value (never significance alone, per
    this package's established discipline -- see faithfulness.py's d'/c
    split) and flags `underpowered` when either group has fewer than 5
    members.
    """
    if len(validated_values) < 2 or len(invalidated_values) < 2:
        return SignalSeparationReport(
            signal_name=signal_name,
            n_validated=len(validated_values), n_invalidated=len(invalidated_values),
            mean_validated=(round(statistics.mean(validated_values), 4) if validated_values else None),
            mean_invalidated=(round(statistics.mean(invalidated_values), 4) if invalidated_values else None),
            cohens_d=None, p_value=None, underpowered=True, n_permutations=0,
        )
    rng = random.Random(seed)
    d = _cohens_d(validated_values, invalidated_values)
    p = _permutation_p_value(validated_values, invalidated_values, n_permutations, rng)
    underpowered = len(validated_values) < 5 or len(invalidated_values) < 5
    return SignalSeparationReport(
        signal_name=signal_name,
        n_validated=len(validated_values), n_invalidated=len(invalidated_values),
        mean_validated=round(statistics.mean(validated_values), 4),
        mean_invalidated=round(statistics.mean(invalidated_values), 4),
        cohens_d=round(d, 4), p_value=round(p, 6),
        underpowered=underpowered, n_permutations=n_permutations,
    )


def holm_bonferroni_correction(p_values: "list[float]") -> "list[float]":
    """
    Standard Holm step-down correction for multiple comparisons. Returns
    adjusted p-values in the SAME order as the input (not sorted) -- index i
    of the result corresponds to p_values[i]. Monotonicity is enforced (an
    adjusted p-value is a running max over the sorted order), the standard
    fix for Holm's raw per-step formula otherwise being able to produce a
    non-monotonic sequence.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * p_values[idx])
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def score_multiple_signals(
    signals: "dict[str, tuple[list[float], list[float]]]",
    alpha: float = 0.05,
    n_permutations: int = 10000,
    seed: Optional[int] = None,
) -> "dict[str, SignalSeparationReport]":
    """
    Score several candidate ex-ante signals (e.g. confidence, hedging,
    defended-under-challenge) against the same validated/invalidated split
    in one call, then apply holm_bonferroni_correction jointly across all of
    them -- testing multiple signals against the same split without
    correcting for the resulting multiple comparisons silently inflates the
    false-positive rate.

    signals maps signal_name -> (validated_values, invalidated_values).
    Reports with no p_value (insufficient data in that signal) are excluded
    from the correction and returned with adjusted_p_value left None.
    """
    base = {
        name: score_signal_separation(v[0], v[1], signal_name=name, n_permutations=n_permutations, seed=seed)
        for name, v in signals.items()
    }
    names_with_p = [name for name, r in base.items() if r.p_value is not None]
    p_values = [base[name].p_value for name in names_with_p]
    adjusted = holm_bonferroni_correction(p_values)

    out = dict(base)
    for name, adj_p in zip(names_with_p, adjusted):
        out[name] = replace(out[name], adjusted_p_value=adj_p, significant_after_correction=adj_p < alpha)
    return out


# ── Batch entry point ──────────────────────────────────────────────────────────

@dataclass
class ResolutionDynamicsReport:
    verdicts: "list[ResolutionVerdict]"
    event_type_warnings: "list[str]" = field(default_factory=list)

    def counts(self) -> "dict[str, int]":
        out = {o.value: 0 for o in ResolutionOutcome}
        for v in self.verdicts:
            out[v.outcome.value] += 1
        return out

    def rate(self, outcome) -> Optional[float]:
        if not self.verdicts:
            return None
        key = outcome.value if isinstance(outcome, ResolutionOutcome) else outcome
        return round(self.counts()[key] / len(self.verdicts), 4)

    def summary(self) -> str:
        n = len(self.verdicts)
        c = self.counts()
        return (
            f"{n} event(s) classified  *  integrated {c['integrated']}  *  "
            f"distorted {c['distorted']}  *  superseded {c['superseded']}  *  "
            f"forgotten {c['forgotten']}  *  "
            f"behaviorally_resolved {c['behaviorally_resolved']}  *  "
            f"unresolved {c['unresolved']}"
        )

    def report(self) -> str:
        sep = "─" * 78
        lines = [
            "", "  RESOLUTION DYNAMICS  (Darwiche-Pearl iterated revision, behaviorally operationalized)",
            sep, "",
        ]
        for v in self.verdicts:
            lines.append(f"  [{v.outcome.value:<22}] {v.event_id}")
            lines.append(f"    {v.rationale}")
            lines.append("")
        lines.append(f"  {self.summary()}")
        if self.event_type_warnings:
            lines.append("")
            lines.append("  EVENT-TYPE WARNINGS:")
            for w in self.event_type_warnings:
                lines.append(f"    - {w}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema_version": RESOLUTION_DYNAMICS_SCHEMA_VERSION,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "counts": self.counts(),
            "event_type_warnings": self.event_type_warnings,
        }


def score_resolution_dynamics(
    events: "list[ContradictionEvent]",
    probes_by_event: "dict[str, list[ResolutionProbe]]",
) -> ResolutionDynamicsReport:
    """
    Classify every event in `events` via classify_resolution(), using
    probes_by_event[event.event_id] (missing/empty -> UNRESOLVED, since
    classify_resolution() requires at least one verified probe to return
    anything else), and collect check_event_type_consistency() warnings
    across all of them.
    """
    verdicts = []
    warnings_list = []
    for event in events:
        probes = probes_by_event.get(event.event_id, [])
        verdicts.append(classify_resolution(event, probes))
        w = check_event_type_consistency(event)
        if w:
            warnings_list.append(w)
    return ResolutionDynamicsReport(verdicts=verdicts, event_type_warnings=warnings_list)


__all__ = [
    "EventType",
    "ResolutionOutcome",
    "ContradictionEvent",
    "ResolutionProbe",
    "ResolutionVerdict",
    "classify_resolution",
    "ClosureCheckResult",
    "check_closure",
    "check_event_type_consistency",
    "EntrenchmentTrial",
    "EntrenchmentFidelityReport",
    "score_entrenchment_fidelity",
    "SignalSeparationReport",
    "score_signal_separation",
    "holm_bonferroni_correction",
    "score_multiple_signals",
    "ResolutionDynamicsReport",
    "score_resolution_dynamics",
]
