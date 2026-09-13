"""
contradish/behavioral_mapping.py -- the practical method: discover what an
AI system's decisions are sensitive to, map that behavior, compare it
against an independently specified normative structure.

─────────────────────────────────────────────────────────────────────────────
THE METHOD (user-specified, 2026-09-13)
─────────────────────────────────────────────────────────────────────────────
"Develop a practical method that takes an AI system, systematically
discovers the variables its decisions are sensitive to, constructs its
behavioral dependency/decision-boundary map, and compares that map with an
independently specified normative structure."

Two things already built this session do two of these four steps, against
a factor set that had to be handed to them in advance:

    decision_relevance.py    scores a FIXED set of factors (R) against a
                              measured sensitivity profile -- but it does
                              not discover which factors exist; it assumes
                              default_technique_drs()'s 8 techniques.
    decision_boundary.py     locates WHERE a boundary sits along a FIXED,
                              already-named ordinal dimension -- it does
                              not discover which dimensions matter.

Neither module discovers anything; both classify or locate. This module
adds the missing first step (DISCOVER) and the missing last step (COMPARE
against an independently specified structure that spans both relevance and
location), and composes the other two directly rather than reimplementing
them.

Note on what "discovery" means here, precisely, because this package has an
established position on the difference: topology.py's expand_node() /
phi_star.py already discover a model's reasoning dependencies by ASKING it
("what does this depend on?") and clustering its free-text answer -- a
self-report. This module discovers sensitivity BEHAVIORALLY instead: it
never asks the model what it depends on, it perturbs a candidate variable
and watches whether the decision actually changes. Those are different
epistemic sources and this package's whole premise (CAI Strain itself, not
just this module) is that behavior is the one that can't be talked around --
see witness.py / eval_awareness.py for the same discipline applied
elsewhere. This module's DISCOVER step is intentionally the cheap,
behavioral kind: one perturbed-vs-baseline probe per candidate (2 total
queries per candidate, one of which -- the baseline -- is shared across all
of them), not free-form exploration.

    DISCOVER    screen_candidates() queries a candidate pool -- by default
                seeded from prompt_analyzer.py's real KNOWN_TECHNIQUES (16
                named pressure techniques), not invented for this module --
                against a baseline, flagging which ones move the decision
                at all. This is deliberately a coarse, cheap gate: full
                classification only runs on what screens positive, so
                total cost stays close to O(n) candidates rather than
                O(n log n) from running a full boundary search on
                everything regardless of whether it matters.

    MAP         build_behavioral_map() takes what screened positive and
                builds the behavioral dependency/decision-boundary map:
                every categorical candidate gets a sensitivity value
                (0.0/1.0 from the screen) feeding directly into
                decision_relevance.py's sensitivity_profile format; every
                ordinal candidate (one given a ladder) gets its behavioral
                boundary located via decision_boundary.py's
                recover_boundary_via_binary_search(), reused unmodified.

    COMPARE     compare_to_normative_structure() takes a NormativeStructure
                -- a DecisionRelevanceSpec (R) plus, for any ordinal
                variables, a dict of BoundaryLadders (B*) -- and produces
                one report: score_dependency_structure() for the
                categorical side, quantify_boundary_discrepancy() for every
                ordinal variable with both a measured boundary and a
                stated one, reusing both functions exactly as they already
                are, tested, elsewhere in this package.

There is one failure mode neither reused function can see on its own, and
it is the reason this module does not just call them separately: a
candidate can screen as behaviorally sensitive while having NO entry at all
in the normative structure's R -- not marked relevant, not marked
irrelevant, simply never specified. score_dependency_structure(), called on
its own, silently ignores any sensitivity_profile key that isn't in
spec.factors; that would make a genuinely novel discovery -- the model
reacts to something nobody classified -- vanish exactly where it matters
most. This module surfaces that case explicitly, as
unspecified_sensitive_variables, rather than letting the reused scorer's
existing (correct, for a bounded factor set) behavior quietly swallow it.

Usage::

    from contradish.behavioral_mapping import (
        default_candidate_pool, default_normative_structure,
        build_behavioral_map, compare_to_normative_structure,
    )

    pool = default_candidate_pool()
    spec = default_normative_structure("medication-002", domain="medication")
    behavioral_map = build_behavioral_map(model_oracle, pool)
    report = compare_to_normative_structure(behavioral_map, spec)
    print(report.report())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from contradish.decision_boundary import (
    BoundaryDiscrepancyReport,
    BoundaryLadder,
    BoundaryRecoveryResult,
    quantify_boundary_discrepancy,
    recover_boundary_via_binary_search,
)
from contradish.decision_relevance import (
    DecisionRelevanceSpec,
    DependencyStructureReport,
    default_technique_drs,
    score_dependency_structure,
)
from contradish.prompt_analyzer import KNOWN_TECHNIQUES

__all__ = [
    "CandidateVariable",
    "default_candidate_pool",
    "ScreeningResult",
    "screen_candidates",
    "BehavioralDependencyMap",
    "build_behavioral_map",
    "NormativeStructure",
    "default_normative_structure",
    "NormativeComparisonReport",
    "compare_to_normative_structure",
]

_BASELINE_KEY = "__baseline__"


# ── Candidate variables: what discovery tests ────────────────────────────────

@dataclass
class CandidateVariable:
    """
    One candidate axis of variation to test the system against.

    kind
        "categorical" -- a single perturbed probe (e.g. one rhetorical
                          technique applied); screening asks only "does
                          this move the decision at all".
        "ordinal"     -- an ordered ladder (see decision_boundary.py);
                          screening probes the far end of the ladder as its
                          cheapest informative point, and a candidate that
                          screens positive gets a full boundary search.

    ladder
        Required for "ordinal": the ordered rung values (same convention as
        BoundaryLadder.rungs). Ignored for "categorical".
    """
    name:  str
    kind:  str               # "categorical" | "ordinal"
    ladder: Optional[list] = None

    def __post_init__(self):
        if self.kind not in ("categorical", "ordinal"):
            raise ValueError(f"invalid kind {self.kind!r} for candidate {self.name!r}")
        if self.kind == "ordinal" and (not self.ladder or len(self.ladder) < 2):
            raise ValueError(f"ordinal candidate {self.name!r} needs a ladder of at least 2 rungs")

    def probe_key(self, rung: Optional[int] = None) -> str:
        """The oracle key screening/recovery use to address this candidate."""
        if self.kind == "categorical":
            return self.name
        idx = self.ladder.__len__() - 1 if rung is None else rung
        return f"{self.name}::{idx}"


def default_candidate_pool() -> list:
    """
    Seeded from contradish.prompt_analyzer.KNOWN_TECHNIQUES -- the real,
    already-shipped 16-technique catalog, which is a strict superset of
    decision_relevance.py's default_technique_drs() 8 -- not invented for
    this module. All 16 are treated as categorical candidates; discovering
    sensitivity to one of the 8 NOT in default_technique_drs() (roleplay,
    third_party, incremental, social_proof, negation_trap, flattery,
    technical_reframe, persistence) is exactly the "found something the
    normative structure never specified" case this module is built to
    surface, not hide.
    """
    return [CandidateVariable(name=name, kind="categorical") for name in KNOWN_TECHNIQUES]


# ── DISCOVER: cheap behavioral screening ─────────────────────────────────────

@dataclass
class ScreeningResult:
    baseline_decision:     str
    decision_by_candidate: dict
    sensitive:              list
    insensitive:            list
    n_queries:               int

    def summary(self) -> str:
        return (
            f"screened {len(self.decision_by_candidate)} candidate(s)  *  "
            f"{len(self.sensitive)} sensitive, {len(self.insensitive)} insensitive  *  "
            f"{self.n_queries} queries"
        )

    def to_dict(self) -> dict:
        return {
            "baseline_decision": self.baseline_decision,
            "decision_by_candidate": self.decision_by_candidate,
            "sensitive": self.sensitive,
            "insensitive": self.insensitive,
            "n_queries": self.n_queries,
        }


def screen_candidates(oracle: Callable[[str], str], candidates: list) -> ScreeningResult:
    """
    One shared baseline query, then one probe per candidate (its perturbed
    form for a categorical candidate, or the far end of its ladder for an
    ordinal one) -- total queries = 1 + len(candidates), regardless of how
    many candidates end up sensitive. A candidate whose probed decision
    differs from baseline is flagged sensitive; nothing here decides
    whether that sensitivity is CORRECT -- that's compare_to_normative_
    structure()'s job, working from what this discovers.

    oracle
        Callable[[key], decision]. Same swappable-judge pattern as this
        package's other oracle-taking functions; key is _BASELINE_KEY for
        the baseline probe and candidate.probe_key() for every other call,
        so a real production oracle just needs to dispatch on that string.
    """
    queries = 0

    def call(key: str) -> str:
        nonlocal queries
        queries += 1
        return oracle(key)

    baseline = call(_BASELINE_KEY)
    decision_by_candidate = {}
    for c in candidates:
        decision_by_candidate[c.name] = call(c.probe_key())

    sensitive = sorted(name for name, d in decision_by_candidate.items() if d != baseline)
    insensitive = sorted(name for name, d in decision_by_candidate.items() if d == baseline)

    return ScreeningResult(
        baseline_decision=baseline, decision_by_candidate=decision_by_candidate,
        sensitive=sensitive, insensitive=insensitive, n_queries=queries,
    )


# ── MAP: build the behavioral dependency/decision-boundary map ──────────────

@dataclass
class BehavioralDependencyMap:
    """
    The measured object: what screening found, plus a located boundary for
    every ordinal candidate that screened sensitive. sensitivity_profile is
    kept pre-built in decision_relevance.py's format (name -> 0.0/1.0) so
    compare_to_normative_structure() can hand it straight to
    score_dependency_structure() without recomputing anything.
    """
    screening:            ScreeningResult
    sensitivity_profile:   dict
    boundary_recoveries:   dict   # {candidate_name: BoundaryRecoveryResult}, ordinal + sensitive only

    def summary(self) -> str:
        return (
            f"{self.screening.summary()}  *  "
            f"{len(self.boundary_recoveries)} boundary recover{'y' if len(self.boundary_recoveries)==1 else 'ies'} attempted"
        )

    def to_dict(self) -> dict:
        return {
            "screening": self.screening.to_dict(),
            "sensitivity_profile": self.sensitivity_profile,
            "boundary_recoveries": {n: r.to_dict() for n, r in self.boundary_recoveries.items()},
        }


def build_behavioral_map(oracle: Callable[[str], str], candidates: list) -> BehavioralDependencyMap:
    """
    DISCOVER, then MAP: screens every candidate, then runs
    recover_boundary_via_binary_search() (reused from decision_boundary.py,
    unmodified) on every ordinal candidate that screened sensitive. A
    categorical candidate never gets a boundary search -- it has no ladder
    to locate one on; its sensitivity IS the map entry, exactly the way
    decision_relevance.py already treats a technique.
    """
    screening = screen_candidates(oracle, candidates)
    sensitivity_profile = {
        name: (1.0 if name in screening.sensitive else 0.0)
        for name in screening.decision_by_candidate
    }

    by_name = {c.name: c for c in candidates}
    boundary_recoveries = {}
    for name in screening.sensitive:
        candidate = by_name[name]
        if candidate.kind != "ordinal":
            continue

        def var_oracle(i: int, _name=name) -> str:
            return oracle(f"{_name}::{i}")

        far_end_decision = screening.decision_by_candidate[name]
        recovery = recover_boundary_via_binary_search(
            var_oracle, len(candidate.ladder), screening.baseline_decision, far_end_decision,
        )
        boundary_recoveries[name] = recovery

    return BehavioralDependencyMap(
        screening=screening, sensitivity_profile=sensitivity_profile,
        boundary_recoveries=boundary_recoveries,
    )


# ── The independently specified normative structure ─────────────────────────

@dataclass
class NormativeStructure:
    """R (via a DecisionRelevanceSpec) plus, optionally, B* per ordinal factor."""
    relevance_spec:  DecisionRelevanceSpec
    boundaries:      dict = field(default_factory=dict)   # {factor_name: BoundaryLadder}


def default_normative_structure(commitment_id: str, domain: str = "") -> NormativeStructure:
    """
    default_technique_drs()'s 8-factor spec, no boundaries -- the same
    honest starting point decision_boundary.py's illustrative_ladder()
    represents: a real relevance specification (grounded in judge.py's own
    guidance), but no authored B* content, because none exists yet
    anywhere in this package. A caller with a real per-domain ladder should
    populate `.boundaries` themselves.
    """
    return NormativeStructure(relevance_spec=default_technique_drs(commitment_id, domain=domain))


# ── COMPARE: the map against the normative structure ─────────────────────────

@dataclass
class NormativeComparisonReport:
    commitment_id:                 str
    domain:                        str
    dependency_report:             DependencyStructureReport
    boundary_discrepancies:        dict          # {factor_name: BoundaryDiscrepancyReport}
    unspecified_sensitive_variables: list

    def summary(self) -> str:
        extra = (
            f"  *  {len(self.unspecified_sensitive_variables)} sensitive but UNSPECIFIED in R"
            if self.unspecified_sensitive_variables else ""
        )
        return f"{self.dependency_report.summary()}{extra}"

    def report(self) -> str:
        sep = "─" * 78
        lines = ["", f"  NORMATIVE COMPARISON  ·  {self.commitment_id} ({self.domain})", sep]
        lines.append(self.dependency_report.report())
        for name, disc in self.boundary_discrepancies.items():
            lines.append(disc.report())
        if self.unspecified_sensitive_variables:
            lines.append(
                f"  discovered sensitive to {self.unspecified_sensitive_variables}, "
                f"but the normative structure never classified them as relevant OR "
                f"irrelevant -- not scored above, surfaced here instead of silently dropped."
            )
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "commitment_id": self.commitment_id,
            "domain": self.domain,
            "dependency_report": self.dependency_report.to_dict(),
            "boundary_discrepancies": {n: d.to_dict() for n, d in self.boundary_discrepancies.items()},
            "unspecified_sensitive_variables": self.unspecified_sensitive_variables,
        }


def compare_to_normative_structure(
    behavioral_map: BehavioralDependencyMap,
    normative: NormativeStructure,
    threshold: float = 0.5,
) -> NormativeComparisonReport:
    """Pure comparison: no model calls, everything here was already measured."""
    dependency_report = score_dependency_structure(
        normative.relevance_spec, behavioral_map.sensitivity_profile, threshold=threshold,
    )

    boundary_discrepancies = {}
    for name, ladder in normative.boundaries.items():
        recovery = behavioral_map.boundary_recoveries.get(name)
        if recovery is None:
            continue
        boundary_discrepancies[name] = quantify_boundary_discrepancy(ladder, recovery)

    unspecified = sorted(
        name for name in behavioral_map.screening.sensitive
        if name not in normative.relevance_spec.factors
    )

    return NormativeComparisonReport(
        commitment_id=normative.relevance_spec.commitment_id,
        domain=normative.relevance_spec.domain,
        dependency_report=dependency_report,
        boundary_discrepancies=boundary_discrepancies,
        unspecified_sensitive_variables=unspecified,
    )
