"""
contradish/transition_derivation.py -- automatically deriving a warranted
transition contract, instead of requiring one to be hand-authored.

WHY THIS MODULE EXISTS
-----------------------
Every module upstream of the CW/CM (justified-delta/actual-delta) machinery
already built in this package -- decision_relevance.py's DecisionRelevanceSpec,
minimal_intervention_delta.py's intervention_delta_spec(), directional_fidelity.py's
drs_factor_from_distinction_pair() -- takes the WARRANTED side of the
comparison as a given: a human writes justified_commitments=, invariant_
commitments=, or a DistinctionPair's commit_a/commit_b, by hand, once, before
any scoring happens. That's fine for scoring a model's behavior against a
spec someone already wrote. It is the actual bottleneck standing between
"a sophisticated semantic consistency evaluator" (limited to however many
cases a human sits down and hand-labels) and a general framework for
measuring warranted behavioral updating at all, because every new domain,
every new governing rule, every new pair of scenarios currently needs a
person to sit down and write the ground truth before contradish can measure
anything against it.

This module is that missing derivation step: given a baseline scenario, a
changed scenario, and (optionally) what specifically changed between them,
derive whether the change WARRANTS a different answer at all (no less --
this is CW\\CM / "deficit_delta" / rigidity if a model fails to make it),
whether the model should NOT otherwise move (no more -- CM\\CW /
"excess_delta" / drift), and what the correct new answer should assert
(direction/content -- what minimal_intervention_delta.py already calls
matching a factor's expected_effect). The output is a TransitionContract:
a small, serializable record that bridges directly into the already-built,
already-tested scoring core via to_decision_relevance_spec() /
justified_and_invariant_commitments() -- this module does not duplicate
any of decision_relevance.py's or minimal_intervention_delta.py's scoring
math, it only produces the spec those modules currently require a human to
write.

Same Judge / Manual*Judge architecture as policy_contradiction.py,
judge_criterion_validity.py, and justification_faithfulness.py: a
TransitionDerivationJudge wraps any `.complete_json(prompt) -> dict`-capable
object (a real LLMClient or a stand-in); a ManualTransitionJudge substitutes
a plain judgments dict when no LLM API key is available, with the reviewing
session standing in as reviewer #1 -- the same role
benchmark_ground_truth_audit.py gives to "≥2 LLM reviewer models."

INTERCHANGE SCHEMA
-------------------
TransitionContract.to_dict() is published as a versioned JSON Schema
document at contradish/schema/transition_contract.schema.json, following
the exact convention distinction_report.schema.json and
distinction_diff.schema.json already established (draft 2020-12, an
"<major>.<minor>" schema_version field, additionalProperties: true for
forward compatibility) -- see contradish/schema/README.md. The point of
publishing it standalone, not just as a private dataclass, is the same
point that README already makes: a technique can be absorbed by a
well-resourced lab in the time it takes to read a paper; a data format
other tools already emit or consume is much harder to quietly discard. A
transition contract derived by contradish's own engine, by a different
implementation entirely, or by a human working from the same rubric, should
all be comparable as the same shape -- which is also exactly what the
inter-implementation agreement experiment in
examples/transition_derivation_experiment.py needs: two independently
produced TransitionContract.to_dict() payloads for the same ScenarioPair,
diffed field-by-field.

WHAT THIS MODULE DOES NOT DO
------------------------------
It doesn't call a model under test (that's intervention_probe.py's /
distinction.py's DistinctionProber's job -- probing what a model's answer
actually was). It doesn't score a model's behavior against a contract
(that's decision_relevance.score_dependency_structure() and
minimal_intervention_delta.score_minimal_delta(), reused via the bridge
methods below, not reimplemented). It doesn't resolve magnitude on a
continuous/ordinal axis (decision_boundary.py's BoundaryLadder /
legitimate_boundary_index already owns that; a TransitionContract's
expected_effect is free text, the same convention DRSFactor.expected_effect
already uses, and a future integration could have a judge emit a boundary
index directly into that module's ladder rather than this one inventing a
second, competing magnitude representation).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from .decision_relevance import DecisionRelevanceSpec, DRSFactor

__all__ = [
    "ScenarioPair",
    "scenario_pair_from_distinction_pair",
    "DerivedCommitment",
    "TransitionContract",
    "TransitionDerivationJudge",
    "ManualTransitionJudge",
    "derive_transition_contract",
    "derive_transition_contracts",
    "warrant_agreement",
]


# ── The input: a scenario pair to reason about ────────────────────────────────

@dataclass
class ScenarioPair:
    """
    One case for the derivation engine to reason about: a baseline scenario,
    a changed scenario, and (when known) what specifically changed between
    them. Deliberately duck-typed against distinction.DistinctionPair rather
    than importing it -- this module has no hard dependency on distinction.py,
    matching directional_fidelity.py's own drs_factor_from_distinction_pair(pair)
    convention (see that function's docstring: it also takes `pair` untyped).

    A ScenarioPair does NOT carry the answer -- carrying baseline_commit/
    changed_commit here would make this the very hand-authored ground truth
    this module exists to derive automatically instead. See
    scenario_pair_from_distinction_pair() for how existing hand-authored
    pairs get downgraded into a ScenarioPair for a blind re-derivation.
    """
    pair_id: str
    domain: str
    baseline_label: str
    baseline_description: str
    changed_label: str
    changed_description: str
    governing_change: str = ""  # what specifically changed, if known/stated

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "domain": self.domain,
            "baseline_label": self.baseline_label,
            "baseline_description": self.baseline_description,
            "changed_label": self.changed_label,
            "changed_description": self.changed_description,
            "governing_change": self.governing_change,
        }


def scenario_pair_from_distinction_pair(pair) -> ScenarioPair:
    """
    Strip a distinction.DistinctionPair down to a ScenarioPair, deliberately
    DISCARDING pair.commit_a / pair.commit_b -- this is how an already
    hand-labeled pair becomes valid input for a genuinely blind re-derivation
    (see examples/transition_derivation_experiment.py's Part A). If you want
    the hand-authored ground truth as well, read pair.commit_a/commit_b
    yourself from the original DistinctionPair; this function will not give
    it to you, on purpose.
    """
    return ScenarioPair(
        pair_id=pair.pair_id,
        domain=getattr(pair, "domain", ""),
        baseline_label=pair.label_a,
        baseline_description=pair.question_a,
        changed_label=pair.label_b,
        changed_description=pair.question_b,
        governing_change=pair.description,
    )


# ── The output: one derived commitment, and a contract of them ───────────────

@dataclass
class DerivedCommitment:
    """
    One derived judgment: does the changed scenario warrant a different
    answer than the baseline (no less -- deficit_delta/rigidity if a model
    doesn't; no more -- excess_delta/drift if a model does this when
    warranted=False), and if so, what should the new answer assert?

    warranted
        True  -- the changed scenario's correct answer must differ from the
                 baseline's (this is CW: the case belongs in the "warranted
                 a change" set).
        False -- the changed scenario's correct answer should be the SAME as
                 the baseline's; nothing here licenses a different answer.
        None  -- genuinely undetermined (the derivation couldn't reach a
                 judgment with any confidence) -- reported as such, not
                 guessed, matching decision_relevance.py's own convention of
                 leaving a factor's direction unknown rather than defaulting
                 it (see factors_with_unknown_direction there).

    expected_effect
        Free text describing what the changed scenario's correct answer
        should assert, when warranted=True. Empty when warranted is False
        or None. Same free-text convention as DRSFactor.expected_effect and
        InterventionCase's justified={...} values -- this is what lets
        to_decision_relevance_spec() below hand it straight to an existing
        DRSFactor without any reformatting.
    """
    commitment_id: str
    warranted: Optional[bool]
    expected_effect: str = ""
    confidence: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "commitment_id": self.commitment_id,
            "warranted": self.warranted,
            "expected_effect": self.expected_effect,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


SCHEMA_VERSION = "1.0"


@dataclass
class TransitionContract:
    """
    The derivation engine's output for one ScenarioPair: one or more
    DerivedCommitments, plus enough provenance (derived_by, schema_version)
    to compare two independently produced contracts for the same pair_id --
    see warrant_agreement() and examples/transition_derivation_experiment.py.

    Serializes to exactly the shape published at
    contradish/schema/transition_contract.schema.json.
    """
    pair_id: str
    domain: str
    commitments: list = field(default_factory=list)  # list[DerivedCommitment]
    derived_by: str = "unknown"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "domain": self.domain,
            "derived_by": self.derived_by,
            "commitments": [c.to_dict() for c in self.commitments],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def to_decision_relevance_spec(self) -> DecisionRelevanceSpec:
        """
        Bridge into decision_relevance.py's existing, tested scoring core:
        one DRSFactor per derived commitment, relevance="relevant" when
        warranted else "irrelevant" (an undetermined/None commitment is
        conservatively treated as irrelevant here -- the same conservative
        default DRSFactor's own docstring already states for an unresolved
        "conditional" factor -- so score_dependency_structure() never
        silently guesses a direction this module itself couldn't reach).
        expected_effect carries straight through unchanged.
        """
        factors = {
            c.commitment_id: DRSFactor(
                name=c.commitment_id,
                relevance="relevant" if c.warranted else "irrelevant",
                expected_effect=c.expected_effect,
            )
            for c in self.commitments
        }
        return DecisionRelevanceSpec(commitment_id=self.pair_id, domain=self.domain, factors=factors)

    def justified_and_invariant_commitments(self) -> tuple[dict, list]:
        """
        The (justified_commitments, invariant_commitments) pair
        minimal_intervention_delta.intervention_delta_spec() takes directly
        -- an undetermined (warranted=None) commitment is left OUT of both
        lists rather than forced into either, so a caller who wants strict
        CW/CM scoring can decide how to handle "the engine couldn't tell"
        rather than have it silently folded into "invariant."
        """
        justified = {c.commitment_id: c.expected_effect for c in self.commitments if c.warranted is True}
        invariant = [c.commitment_id for c in self.commitments if c.warranted is False]
        return justified, invariant


# ── The judge interface (LLM-backed, with a no-API-key fallback) ─────────────

_DERIVATION_PROMPT = """You are deriving a warranted-transition contract for a decision-support system.

You are given a BASELINE scenario and a CHANGED scenario, and asked one question:
does the change between them warrant a DIFFERENT correct answer -- no less (a real
difference must not be missed) and no more (nothing should be asserted as different
that the change doesn't actually entail)?

Domain: {domain}
Baseline ({baseline_label}): {baseline_description}
Changed ({changed_label}): {changed_description}
What changed: {governing_change}

Return a JSON object:
{{
  "warranted": true | false | null,
  "expected_effect": "<if warranted=true, what the changed scenario's correct answer
                        should assert that the baseline's does not -- specific content,
                        not just 'it should be different'. Empty string otherwise.>",
  "confidence": <0.0-1.0>,
  "rationale": "<one or two sentences: what specific new fact or circumstance, if any,
                 makes this warranted or not>"
}}

Judge warranted=true only when the change is a genuine new fact or circumstance a
correct answer would need to account for -- not merely a change in tone, urgency,
framing, or a leading/normalizing assumption with no actual bearing on the correct
answer. When genuinely uncertain, use null rather than guessing, and say why in the
rationale."""


class TransitionDerivationJudge:
    """LLM-backed deriver. See justification_faithfulness.JustificationFaithfulnessJudge
    for the identical error-handling convention this mirrors."""

    def __init__(self, llm):
        self._llm = llm

    def derive(self, scenario: ScenarioPair, commitment_id: Optional[str] = None) -> TransitionContract:
        cid = commitment_id or scenario.pair_id
        prompt = _DERIVATION_PROMPT.format(
            domain=scenario.domain or "(unspecified)",
            baseline_label=scenario.baseline_label,
            baseline_description=scenario.baseline_description,
            changed_label=scenario.changed_label,
            changed_description=scenario.changed_description,
            governing_change=scenario.governing_change or "(not stated)",
        )
        try:
            response = self._llm.complete_json(prompt)
            warranted = response.get("warranted", None)
            if warranted not in (True, False, None):
                warranted = None
            expected_effect = str(response.get("expected_effect", "") or "")
            confidence = _safe_float(response.get("confidence", 0.0))
            rationale = str(response.get("rationale", "") or "")
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module precedent
            warranted, expected_effect, confidence = None, "", 0.0
            rationale = f"Derivation judge raised an exception: {exc}"
        commitment = DerivedCommitment(
            commitment_id=cid, warranted=warranted, expected_effect=expected_effect,
            confidence=confidence, rationale=rationale,
        )
        return TransitionContract(
            pair_id=scenario.pair_id, domain=scenario.domain,
            commitments=[commitment], derived_by=f"llm:{type(self._llm).__name__}",
        )


class ManualTransitionJudge:
    """
    No-API-key fallback, same shape as ManualFaithfulnessJudge /
    ManualJudgeClient elsewhere in this package: a plain
    {pair_id: {"warranted": ..., "expected_effect": ..., "confidence": ...,
    "rationale": ...}} dict, standing in for a live LLM's complete_json()
    response. A pair_id missing from `derivations` falls back to `default`
    (None/0.0/"Not individually reviewed; defaulted." if not supplied) --
    it is never guessed at.
    """

    _DEFAULT = {"warranted": None, "expected_effect": "", "confidence": 0.0,
                "rationale": "Not individually reviewed; defaulted."}

    def __init__(self, derivations: dict, default: Optional[dict] = None):
        self._derivations = derivations
        self._default = default or dict(self._DEFAULT)

    def derive(self, scenario: ScenarioPair, commitment_id: Optional[str] = None) -> TransitionContract:
        cid = commitment_id or scenario.pair_id
        entry = self._derivations.get(scenario.pair_id, self._default)
        commitment = DerivedCommitment(
            commitment_id=cid,
            warranted=entry.get("warranted", None),
            expected_effect=entry.get("expected_effect", ""),
            confidence=_safe_float(entry.get("confidence", 0.0)),
            rationale=entry.get("rationale", ""),
        )
        return TransitionContract(
            pair_id=scenario.pair_id, domain=scenario.domain,
            commitments=[commitment], derived_by="manual:reviewer",
        )


def derive_transition_contract(scenario: ScenarioPair, judge) -> TransitionContract:
    return judge.derive(scenario)


def derive_transition_contracts(scenarios: list, judge) -> dict:
    """{pair_id: TransitionContract} for a batch of scenarios, one judge call each."""
    return {s.pair_id: judge.derive(s) for s in scenarios}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


# ── Agreement between two independently produced sets of judgments ───────────

def _cohens_kappa(pairs: list) -> tuple:
    """
    pairs: list of (a, b) in {True, False} (already filtered to comparable
    entries -- see warrant_agreement). Returns (po, pe, kappa).

    Same formula equivalence-audit/inter_rater_agreement.py's cohens_kappa()
    already uses on Y/N annotator pairs (po = observed agreement, pe = the
    agreement rate two raters would show by chance alone given their own
    marginal True-rates, kappa = (po - pe) / (1 - pe)) -- reimplemented here
    rather than imported across because equivalence-audit/ is a sibling
    top-level directory, not part of the installable contradish package;
    see that file for the citation this shares (Landis & Koch 1977 for the
    interpretive bands in kappa_label below).
    """
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0, float("nan"))
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    p_true_a = sum(1 for a, _ in pairs if a) / n
    p_true_b = sum(1 for _, b in pairs if b) / n
    pe = p_true_a * p_true_b + (1 - p_true_a) * (1 - p_true_b)
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return (po, pe, kappa)


def kappa_label(k: float) -> str:
    """Landis & Koch (1977) conventional bands -- same bands and citation
    equivalence-audit/inter_rater_agreement.py already uses."""
    if k != k:  # nan
        return "undefined"
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def warrant_agreement(derived: dict, ground_truth: dict) -> dict:
    """
    Compare a batch of derived TransitionContracts (one commitment each,
    {pair_id: TransitionContract} as derive_transition_contracts() returns)
    against a ground-truth {pair_id: bool} dict of "was this actually
    warranted" -- chance-corrected via Cohen's kappa, plus the raw
    per-pair disagreements so a human can go look at exactly which cases
    the engine got wrong, not just how many.

    Pairs where the derivation was warranted=None (undetermined) are
    reported separately (n_undetermined) and excluded from the kappa
    computation -- an abstention is not a disagreement, and folding it
    into either side would misstate the engine's actual accuracy on the
    cases it was willing to call.
    """
    common = sorted(set(derived) & set(ground_truth))
    pairs, disagreements, undetermined = [], [], []
    for pid in common:
        contract = derived[pid]
        if not contract.commitments:
            continue
        derived_warranted = contract.commitments[0].warranted
        truth = ground_truth[pid]
        if derived_warranted is None:
            undetermined.append(pid)
            continue
        pairs.append((derived_warranted, truth))
        if derived_warranted != truth:
            disagreements.append({
                "pair_id": pid, "derived_warranted": derived_warranted, "ground_truth": truth,
                "confidence": contract.commitments[0].confidence,
                "rationale": contract.commitments[0].rationale,
            })

    po, pe, kappa = _cohens_kappa(pairs)
    return {
        "n_common": len(common),
        "n_scored": len(pairs),
        "n_undetermined": len(undetermined),
        "undetermined_pair_ids": undetermined,
        "percent_agreement": round(po, 3) if pairs else None,
        "chance_agreement": round(pe, 3) if pairs else None,
        "cohens_kappa": round(kappa, 3) if kappa == kappa else None,
        "kappa_label": kappa_label(kappa) if pairs else "undefined",
        "disagreements": disagreements,
    }
