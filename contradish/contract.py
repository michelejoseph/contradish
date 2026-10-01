"""
contradish/contract.py -- the policy evaluation contract.

contradish's two measurements -- semantic invariance (CAI Strain, `Suite`)
and warranted behavioral change (`contradish update`) -- were built as
separate instruments with separate inputs. For a *policy-grounded
assistant* (a support bot, benefits navigator, claims assistant: anything
whose answers are supposed to follow from a written policy) they are two
halves of one requirement, and this module states that requirement as a
single, declarative, checkable contract.

─────────────────────────────────────────────────────────────────────────────
THE CONTRACT
─────────────────────────────────────────────────────────────────────────────
A policy P is a list of identified clauses. A decision case c names the
clauses it depends on, a canonical question, a set of meaning-preserving
variants of that question (its equivalence class [c]), and the outcome
label P warrants for it, f*(P, c), drawn from a declared outcome vocabulary.
An amendment A edits clauses, producing P' = A(P), and declares exactly
which cases' warranted outcomes change and to what -- f*(P', c) -- and, by
omission, that every other case keeps f*(P, c).

Let f(P, x) be the outcome label the assistant's answer to input x commits
to when P is its governing policy. The contract has four obligations:

  SI  semantic invariance   for every policy state P and case c,
                             f(P, x) is the same for every x in [c].
                             (Meaning held fixed -> behavior held fixed.)

  PG  policy grounding      f(P, x) = f*(P, c): the invariant answer is
                             the one the policy actually warrants.
                             (Consistent is not the same as correct.)

  WC  warranted change      for every amendment A: f(A(P), c) differs from
                             f(P, c) for exactly the cases whose warranted
                             outcome changed, and lands on f*(A(P), c).
                             Changing a case A did not license is DRIFT;
                             failing to change one it did is RIGIDITY;
                             changing it to the wrong outcome is
                             MISDIRECTION.

  FS  fact sensitivity      for every contrast of a case (the same situation
                             with one decisive fact changed), f(P, x') =
                             f*(P, x'), which differs from f*(P, c). Without
                             it, an assistant that ignores the decisive fact
                             passes SI perfectly.

SI and WC are the same principle seen from both sides: behavior must be a
function of the policy-relevant content of the situation -- insensitive to
everything else (surface form, framing, pressure, a meaning-preserving
rewording of the policy itself), and sensitive to exactly the changes in
that content the policy licenses. WC is scored with the existing
minimal_intervention_delta / decision_relevance core (no new scoring math);
this module supplies the policy-grounded spec that core needs and the probe
that measures against it.

Every obligation is reported with a 95% interval (Wilson over cases,
contrasts, or amendments; a case-clustered bootstrap for PG). With
samples=k each input is asked k times, its outcome is the modal label, and
the disagreement between repeats of the identical prompt is reported as the
noise floor, so an SI failure can be told apart from sampling noise. With a
small human-labelled calibration sample (calibration_sample() ->
ClassifierCalibration), PG is corrected for classifier error with the
Rogan-Gladen estimator, and the report says how many SI failures classifier
error alone would produce.

A "meaning-preserving" amendment (e.g. rewording a clause without changing
what it permits) is a control: its warranted change set is empty by
construction, so any behavioral change it causes is drift. That extends SI
from paraphrases of the *user's input* to paraphrases of the *governing
policy*.

─────────────────────────────────────────────────────────────────────────────
WHAT THE CONTRACT ENFORCES BEFORE ANY MODEL CALL
─────────────────────────────────────────────────────────────────────────────
PolicyContract.lint() checks the contract itself: every case cites real
clauses and a declared outcome, every declared change is a real change,
every expected change is traceable to a clause the amendment touched, every
case citing an amended clause has had its scope reviewed (declared either as
changing or as reviewed-invariant), and every clause is exercised by at
least one case. A warranted change that cannot be traced to a changed
clause is exactly what the contract exists to catch in a model; the linter
refuses to let the contract's own author get away with it either.

Usage::

    from contradish.contract import PolicyContract, run_contract, default_outcome_classifier
    from contradish.llm import LLMClient

    contract = PolicyContract.load("returns_contract.yaml")
    issues = contract.lint()                    # static, no API calls

    def app(system_prompt: str, question: str) -> str:
        ...                                     # your assistant, with this policy

    result = run_contract(contract, app, default_outcome_classifier(LLMClient()))
    print(result.report())
    assert result.passed

CLI::

    contradish contract lint returns_contract.yaml
    contradish contract run  returns_contract.yaml --app mymodule:app
    contradish contract run                                  # built-in demo contract
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from contradish.decision_relevance import score_dependency_structure
from contradish.minimal_intervention_delta import (
    MinimalDeltaVerdict,
    intervention_delta_spec,
    score_minimal_delta,
)

__all__ = [
    "SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "BASE_STATE",
    "UNCLEAR",
    "Clause",
    "Contrast",
    "DecisionCase",
    "Amendment",
    "Thresholds",
    "LintIssue",
    "PolicyContract",
    "Observation",
    "CaseStateResult",
    "StateResult",
    "TransitionResult",
    "Obligation",
    "ContractResult",
    "ClassifierCalibration",
    "calibration_sample",
    "wilson_interval",
    "default_outcome_classifier",
    "evaluate_contract",
    "run_contract",
    "builtin_contract_path",
    "load_builtin_contract",
    "list_builtin_contracts",
]

SCHEMA_VERSION = "1.1"
RESULT_SCHEMA_VERSION = "1.1"
BASE_STATE = "base"
UNCLEAR = "unclear"

ModelFn = Callable[[str, str], str]
Classifier = Callable[[str, str, dict], str]

_BUILTIN_DIR = Path(__file__).parent / "contracts"


# ─────────────────────────────────────────────────────────────────────────────
# Contract definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Clause:
    """One identified clause of the governing policy."""
    id: str
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text}


@dataclass
class Contrast:
    """
    A minimal edit to a case's situation that SHOULD change the outcome --
    the input-level counterpart of an amendment. A case's variants say "these
    differences must not matter"; its contrasts say "this difference must".
    Without contrasts, an assistant that ignores the decisive fact entirely
    (always "no refund", whatever the date) passes semantic invariance
    perfectly. Contrasts are probed under the base policy.

    question  the case's situation with one decisive fact changed
              (35 days -> 25 days).
    expected  the outcome the base policy warrants for it; must differ from
              the parent case's expected outcome.
    """
    id: str
    question: str
    expected: str

    def to_dict(self) -> dict:
        return {"id": self.id, "question": self.question, "expected": self.expected}


@dataclass
class DecisionCase:
    """
    One policy decision the assistant must get right.

    clauses   ids of the clauses this decision depends on.
    question  the canonical phrasing.
    variants  meaning-preserving rewrites of `question` -- different wording,
              framing, or pressure, but the same policy-relevant facts. Each
              variant must warrant the same outcome as `question` under
              every policy state; that is what makes it a variant.
    expected  the outcome label the base policy warrants.
    outcomes  optional subset of the contract's outcome vocabulary the
              classifier chooses between for this case (default: all).
    contrasts minimal fact edits whose warranted outcome differs (Contrast).
    """
    id: str
    clauses: list
    question: str
    expected: str
    variants: list = field(default_factory=list)
    outcomes: Optional[list] = None
    contrasts: list = field(default_factory=list)

    def inputs(self, include_variants: bool = True) -> list:
        items = [("canonical", self.question)]
        if include_variants:
            items += [(f"v{i + 1}", v) for i, v in enumerate(self.variants)]
        return items

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "clauses": list(self.clauses),
            "question": self.question,
            "expected": self.expected,
            "variants": list(self.variants),
        }
        if self.outcomes is not None:
            d["outcomes"] = list(self.outcomes)
        if self.contrasts:
            d["contrasts"] = [c.to_dict() for c in self.contrasts]
        return d


@dataclass
class Amendment:
    """
    One change to the governing policy, and its declared behavioral warrant.

    set_clauses          {clause_id: new_text}. An id not in the base policy
                         adds a new clause.
    remove_clauses       clause ids deleted by this amendment.
    expected_changes     {case_id: new_outcome_label} -- the cases whose
                         warranted outcome this amendment changes. Every case
                         not listed is warranted to keep its base outcome.
    reviewed_invariant   cases that cite an amended clause but whose outcome
                         the author has checked does NOT change (e.g. a
                         20-day return is still refundable when the window
                         goes from 30 to 45 days). Declaring them is how the
                         contract records that the scope was reviewed rather
                         than forgotten; lint warns about any case that cites
                         an amended clause and is in neither list.
    meaning_preserving   True for a control amendment (a rewording of the
                         policy that changes nothing it permits). Must have
                         no expected_changes; any behavioral change it causes
                         is drift.
    """
    id: str
    description: str = ""
    set_clauses: dict = field(default_factory=dict)
    remove_clauses: list = field(default_factory=list)
    expected_changes: dict = field(default_factory=dict)
    reviewed_invariant: list = field(default_factory=list)
    meaning_preserving: bool = False

    def touched_clauses(self) -> set:
        return set(self.set_clauses) | set(self.remove_clauses)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "set_clauses": dict(self.set_clauses),
            "remove_clauses": list(self.remove_clauses),
            "expected_changes": dict(self.expected_changes),
            "reviewed_invariant": list(self.reviewed_invariant),
            "meaning_preserving": self.meaning_preserving,
        }


@dataclass
class Thresholds:
    """
    Minimum rates each obligation must reach for the contract to pass.
    Defaults are 1.0 -- a contract, not a tendency. Lower them only with a
    stated reason (e.g. measured judge noise from `contradish judge-floor`).

    invariance        per policy state: fraction of cases whose every input
                      committed to one and the same outcome.
    grounding         per policy state: fraction of all inputs whose outcome
                      equals the warranted outcome.
    warranted_change  fraction of amendments whose behavioral delta was
                      exact (no drift, no rigidity, no misdirection).
    fact_sensitivity  fraction of contrasts answered with their (different)
                      warranted outcome.
    gate              "point" (default): an obligation passes when its point
                      estimate meets the threshold. "resolved": it fails only
                      when the whole 95% interval is below the threshold, i.e.
                      the shortfall is not explainable by sampling noise
                      alone. Use "resolved" with lowered thresholds on noisy
                      assistants; keep "point" for strict release gates.
    """
    invariance: float = 1.0
    grounding: float = 1.0
    warranted_change: float = 1.0
    fact_sensitivity: float = 1.0
    gate: str = "point"

    def to_dict(self) -> dict:
        return {
            "invariance": self.invariance,
            "grounding": self.grounding,
            "warranted_change": self.warranted_change,
            "fact_sensitivity": self.fact_sensitivity,
            "gate": self.gate,
        }


@dataclass
class LintIssue:
    level: str   # "error" | "warning" | "info"
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7s} {self.code}  {self.message}"

    def to_dict(self) -> dict:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass
class PolicyContract:
    """The full evaluation contract for one policy-grounded assistant."""
    contract_id: str
    domain: str
    clauses: list
    outcomes: dict
    cases: list
    amendments: list = field(default_factory=list)
    title: str = ""
    preamble: str = ""
    thresholds: Thresholds = field(default_factory=Thresholds)
    schema_version: str = SCHEMA_VERSION

    # ── lookups ─────────────────────────────────────────────────────────────

    @property
    def case_map(self) -> dict:
        return {c.id: c for c in self.cases}

    @property
    def amendment_map(self) -> dict:
        return {a.id: a for a in self.amendments}

    def states(self) -> list:
        """Policy states the contract is evaluated under: base, then each amendment."""
        return [BASE_STATE] + [a.id for a in self.amendments]

    def clauses_for(self, state: str = BASE_STATE) -> list:
        """The clause list in force under `state` (base or an amendment id)."""
        if state == BASE_STATE:
            return list(self.clauses)
        amendment = self.amendment_map[state]
        out = []
        seen = set()
        for c in self.clauses:
            seen.add(c.id)
            if c.id in amendment.remove_clauses:
                continue
            out.append(Clause(c.id, amendment.set_clauses.get(c.id, c.text)))
        for cid, text in amendment.set_clauses.items():
            if cid not in seen:
                out.append(Clause(cid, text))
        return out

    def policy_text(self, state: str = BASE_STATE) -> str:
        """The system prompt the assistant receives under `state`."""
        lines = []
        if self.preamble:
            lines += [self.preamble.strip(), ""]
        lines.append("Policy:")
        for c in self.clauses_for(state):
            lines.append(f"[{c.id}] {c.text}")
        return "\n".join(lines)

    def expected(self, case_id: str, state: str = BASE_STATE) -> str:
        """f*(state, case): the warranted outcome for a case under a state."""
        case = self.case_map[case_id]
        if state == BASE_STATE:
            return case.expected
        return self.amendment_map[state].expected_changes.get(case_id, case.expected)

    def expected_for_input(self, case_id: str, state: str, input_id: str) -> Optional[str]:
        """Warranted outcome for one input: the case's, or a contrast's (base policy only)."""
        if input_id.startswith("contrast:"):
            if state != BASE_STATE:
                return None
            cid = input_id[len("contrast:"):]
            x = next((x for x in self.case_map[case_id].contrasts if x.id == cid), None)
            return x.expected if x else None
        return self.expected(case_id, state)

    def outcomes_for(self, case: DecisionCase) -> dict:
        if case.outcomes is None:
            return dict(self.outcomes)
        return {k: self.outcomes[k] for k in case.outcomes if k in self.outcomes}

    # ── static validation ───────────────────────────────────────────────────

    def lint(self) -> list:
        """
        Check the contract itself, with no model calls. Errors make the
        contract unrunnable or self-contradictory; warnings mark places where
        the contract under-specifies what it claims to test.
        """
        issues: list = []

        def err(code, msg):
            issues.append(LintIssue("error", code, msg))

        def warn(code, msg):
            issues.append(LintIssue("warning", code, msg))

        def info(code, msg):
            issues.append(LintIssue("info", code, msg))

        for kind, ids in (
            ("clause", [c.id for c in self.clauses]),
            ("case", [c.id for c in self.cases]),
            ("amendment", [a.id for a in self.amendments]),
        ):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                err("E001", f"duplicate {kind} id(s): {', '.join(dupes)}")

        if BASE_STATE in {a.id for a in self.amendments}:
            err("E001", f"amendment id {BASE_STATE!r} is reserved for the unamended policy")
        if UNCLEAR in self.outcomes:
            err("E003", f"outcome label {UNCLEAR!r} is reserved for answers that commit to no outcome")
        if not self.cases:
            err("E009", "contract has no decision cases")
        if len(self.outcomes) < 2:
            err("E003", "outcome vocabulary needs at least two labels to distinguish anything")

        clause_ids = {c.id for c in self.clauses}
        for case in self.cases:
            if not case.clauses:
                err("E002", f"case {case.id!r} cites no clauses: its warrant is untraceable")
            for cid in case.clauses:
                if cid not in clause_ids and not any(cid in a.set_clauses for a in self.amendments):
                    err("E002", f"case {case.id!r} cites unknown clause {cid!r}")
            if case.expected not in self.outcomes:
                err("E003", f"case {case.id!r} expects undeclared outcome {case.expected!r}")
            if case.outcomes is not None:
                unknown = [o for o in case.outcomes if o not in self.outcomes]
                if unknown:
                    err("E008", f"case {case.id!r} restricts to undeclared outcome(s): {', '.join(unknown)}")
                if case.expected not in case.outcomes:
                    err("E008", f"case {case.id!r} expects {case.expected!r} but excludes it from its outcomes")
            if len(case.variants) < 2:
                warn("W102", f"case {case.id!r} has {len(case.variants)} variant(s): "
                              "semantic invariance is barely tested for it (aim for >= 3)")
            if len(set(case.variants) | {case.question}) < len(case.variants) + 1:
                warn("W106", f"case {case.id!r} repeats a phrasing among its variants")
            xids = [x.id for x in case.contrasts]
            xdupes = sorted({i for i in xids if xids.count(i) > 1})
            if xdupes:
                err("E001", f"case {case.id!r} has duplicate contrast id(s): {', '.join(xdupes)}")
            for x in case.contrasts:
                if x.expected not in self.outcomes:
                    err("E003", f"contrast {case.id}/{x.id} expects undeclared outcome {x.expected!r}")
                elif case.outcomes is not None and x.expected not in case.outcomes:
                    err("E008", f"contrast {case.id}/{x.id} expects {x.expected!r}, outside the case's outcomes")
                if x.expected == case.expected:
                    err("E014", f"contrast {case.id}/{x.id} warrants the same outcome as its case "
                                f"({x.expected!r}): that is a variant, not a contrast")
                if x.question.strip() == case.question.strip() or x.question in case.variants:
                    err("E014", f"contrast {case.id}/{x.id} repeats a phrasing of its own case")

        if self.cases and not any(c.contrasts for c in self.cases):
            warn("W108", "no case has contrasts: an assistant that ignores the decisive facts "
                         "entirely can pass semantic invariance; add at least one contrast per case")

        cited = {cid for case in self.cases for cid in case.clauses}
        for c in self.clauses:
            if c.id not in cited:
                warn("W101", f"clause {c.id!r} is cited by no case: that part of the policy is untested")

        case_ids = set(self.case_map)
        for a in self.amendments:
            touched = a.touched_clauses()
            if not touched:
                err("E011", f"amendment {a.id!r} changes no clause")
            for cid in a.remove_clauses:
                if cid not in clause_ids:
                    err("E004", f"amendment {a.id!r} removes unknown clause {cid!r}")
            for cid, text in a.set_clauses.items():
                base = next((c.text for c in self.clauses if c.id == cid), None)
                if base is not None and base.strip() == text.strip():
                    warn("W107", f"amendment {a.id!r} sets clause {cid!r} to its existing text")
            for cid, label in a.expected_changes.items():
                if cid not in case_ids:
                    err("E005", f"amendment {a.id!r} declares a change for unknown case {cid!r}")
                    continue
                if label not in self.outcomes:
                    err("E006", f"amendment {a.id!r} declares undeclared outcome {label!r} for case {cid!r}")
                case = self.case_map[cid]
                if label == case.expected:
                    err("E007", f"amendment {a.id!r} declares case {cid!r} changes to "
                                f"{label!r}, which is already its base outcome")
                if case.outcomes is not None and label not in case.outcomes:
                    err("E008", f"amendment {a.id!r} expects {label!r} for case {cid!r}, outside its outcomes")
                if not (set(case.clauses) & touched):
                    warn("W104", f"amendment {a.id!r} expects case {cid!r} to change, but the case "
                                 f"cites none of the clauses it touches ({', '.join(sorted(touched))}): "
                                 "the warrant is not traceable to the amendment")
            for cid in a.reviewed_invariant:
                if cid not in case_ids:
                    err("E005", f"amendment {a.id!r} lists unknown case {cid!r} as reviewed_invariant")
                if cid in a.expected_changes:
                    err("E010", f"amendment {a.id!r} lists case {cid!r} as both changing and invariant")
            if a.meaning_preserving:
                if a.expected_changes:
                    err("E013", f"amendment {a.id!r} is meaning_preserving but declares expected changes")
                info("I201", f"amendment {a.id!r} is a meaning-preserving control: "
                             "any behavioral change under it is drift")
                continue
            if not a.expected_changes:
                warn("W105", f"amendment {a.id!r} declares no expected changes; if it is a "
                             "rewording, mark it meaning_preserving: true")
            reviewed = set(a.expected_changes) | set(a.reviewed_invariant)
            for case in self.cases:
                if set(case.clauses) & touched and case.id not in reviewed:
                    warn("W103", f"case {case.id!r} cites a clause amendment {a.id!r} touches but is "
                                 "neither in expected_changes nor reviewed_invariant: its scope was not reviewed")
        return issues

    def lint_errors(self) -> list:
        return [i for i in self.lint() if i.level == "error"]

    # ── bridges ─────────────────────────────────────────────────────────────

    def to_intervention_cases(self) -> list:
        """
        One intervention_probe.InterventionCase per amendment, so any contract
        amendment can also be run through `contradish update`'s probe.
        """
        from contradish.intervention_probe import InterventionCase

        out = []
        for a in self.amendments:
            justified = {
                cid: {
                    "question": self.case_map[cid].question,
                    "expected_effect": f"{label}: {self.outcomes.get(label, '')}".rstrip(": "),
                }
                for cid, label in a.expected_changes.items() if cid in self.case_map
            }
            invariant = {
                c.id: c.question for c in self.cases if c.id not in a.expected_changes
            }
            out.append(InterventionCase(
                intervention_id=a.id,
                domain=self.domain,
                before=self.policy_text(BASE_STATE),
                after=self.policy_text(a.id),
                justified=justified,
                invariant=invariant,
            ))
        return out

    # ── (de)serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "domain": self.domain,
            "title": self.title,
            "preamble": self.preamble,
            "policy": [c.to_dict() for c in self.clauses],
            "outcomes": dict(self.outcomes),
            "cases": [c.to_dict() for c in self.cases],
            "amendments": [a.to_dict() for a in self.amendments],
            "thresholds": self.thresholds.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyContract":
        if not isinstance(data, dict):
            raise ValueError("a policy contract must be a mapping at the top level")
        missing = [k for k in ("contract_id", "policy", "outcomes", "cases") if k not in data]
        if missing:
            raise ValueError(f"policy contract is missing required field(s): {', '.join(missing)}")

        policy = data["policy"]
        if isinstance(policy, dict):          # {clause_id: text} shorthand
            clauses = [Clause(str(k), str(v)) for k, v in policy.items()]
        else:
            clauses = [Clause(str(c["id"]), str(c["text"])) for c in policy]

        outcomes = data["outcomes"]
        if isinstance(outcomes, list):        # [label, ...] shorthand
            outcomes = {str(o): "" for o in outcomes}
        outcomes = {str(k): str(v or "") for k, v in outcomes.items()}

        cases = []
        for c in data["cases"]:
            cl = c.get("clauses", [])
            cases.append(DecisionCase(
                id=str(c["id"]),
                clauses=[cl] if isinstance(cl, str) else [str(x) for x in cl],
                question=str(c["question"]),
                expected=str(c["expected"]),
                variants=[str(v) for v in c.get("variants", [])],
                outcomes=[str(o) for o in c["outcomes"]] if c.get("outcomes") is not None else None,
                contrasts=[
                    Contrast(id=str(x.get("id", f"x{i + 1}")), question=str(x["question"]),
                             expected=str(x["expected"]))
                    for i, x in enumerate(c.get("contrasts") or [])
                ],
            ))

        amendments = []
        for a in data.get("amendments", []) or []:
            amendments.append(Amendment(
                id=str(a["id"]),
                description=str(a.get("description", "")),
                set_clauses={str(k): str(v) for k, v in (a.get("set_clauses") or {}).items()},
                remove_clauses=[str(x) for x in a.get("remove_clauses", []) or []],
                expected_changes={str(k): str(v) for k, v in (a.get("expected_changes") or {}).items()},
                reviewed_invariant=[str(x) for x in a.get("reviewed_invariant", []) or []],
                meaning_preserving=bool(a.get("meaning_preserving", False)),
            ))

        t = data.get("thresholds") or {}
        thresholds = Thresholds(
            invariance=float(t.get("invariance", 1.0)),
            grounding=float(t.get("grounding", 1.0)),
            warranted_change=float(t.get("warranted_change", 1.0)),
            fact_sensitivity=float(t.get("fact_sensitivity", 1.0)),
            gate=str(t.get("gate", "point")),
        )
        if thresholds.gate not in ("point", "resolved"):
            raise ValueError(f"thresholds.gate must be 'point' or 'resolved', got {thresholds.gate!r}")
        return cls(
            contract_id=str(data["contract_id"]),
            domain=str(data.get("domain", data["contract_id"])),
            title=str(data.get("title", "")),
            preamble=str(data.get("preamble", "")),
            clauses=clauses,
            outcomes=outcomes,
            cases=cases,
            amendments=amendments,
            thresholds=thresholds,
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )

    @classmethod
    def load(cls, path) -> "PolicyContract":
        """Load a contract from a .json, .yaml, or .yml file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    "Reading a YAML contract requires PyYAML (pip install pyyaml), "
                    "or write the contract as JSON."
                ) from e
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in contracts
# ─────────────────────────────────────────────────────────────────────────────

def list_builtin_contracts() -> list:
    if not _BUILTIN_DIR.is_dir():
        return []
    return sorted(p.stem for p in _BUILTIN_DIR.glob("*.json"))


def builtin_contract_path(name: str) -> Path:
    path = _BUILTIN_DIR / f"{name}.json"
    if not path.exists():
        raise KeyError(
            f"no built-in contract named {name!r}. Available: {', '.join(list_builtin_contracts())}"
        )
    return path


def load_builtin_contract(name: str = "ecommerce_returns") -> PolicyContract:
    return PolicyContract.load(builtin_contract_path(name))


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────

def default_outcome_classifier(llm) -> Classifier:
    """
    Classify an answer into one label of the case's outcome vocabulary, or
    UNCLEAR if it commits to none. Classifying against declared labels --
    rather than asking a judge whether two freeform answers "agree" -- is
    what lets one judgment serve every obligation: SI compares labels
    across inputs, PG compares a label to the warranted one, WC compares
    labels across policy states.

    Like every default judge in contradish, this inherits the judge's own
    noise; measure it (`contradish judge-floor`) or pass your own classifier
    for anything you gate a release on.
    """
    def classify(question: str, answer: str, outcomes: dict) -> str:
        vocab = "\n".join(
            f"- {label}: {desc}" if desc else f"- {label}" for label, desc in outcomes.items()
        )
        prompt = (
            "A policy-grounded assistant answered a user. Classify which outcome "
            "the answer actually commits the user to. Judge the substance only -- "
            "ignore tone, apologies, hedging, and offers of further help. If the "
            f"answer commits to none of the outcomes, answer {UNCLEAR}.\n\n"
            f"User: {question}\n\n"
            f"Assistant: {answer}\n\n"
            f"Outcomes:\n{vocab}\n\n"
            f"Reply with exactly one label from the list, or {UNCLEAR}."
        )
        raw = llm.complete(prompt, model=getattr(llm, "fast_model", None), max_tokens=20)
        return parse_label(raw, outcomes)
    return classify


def parse_label(raw: str, outcomes: dict) -> str:
    """Map a classifier's raw reply onto a declared label (longest match wins)."""
    text = (raw or "").strip().lower().strip("`'\". ")
    labels = sorted(outcomes, key=len, reverse=True)
    for label in labels:
        if text == label.lower():
            return label
    for label in labels:
        if label.lower() in text:
            return label
    return UNCLEAR


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

_Z95 = 1.959963984540054
_BOOTSTRAP_DRAWS = 2000


def wilson_interval(k: int, n: int, z: float = _Z95) -> tuple:
    """Wilson score interval for k successes in n trials; (None, None) if n == 0."""
    if n <= 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _percentile(sorted_vals: list, q: float) -> float:
    idx = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _cluster_bootstrap(clusters: list, seed: int = 0, transform=None) -> tuple:
    """
    Percentile CI for a ratio sum(k)/sum(n) when items are clustered (inputs
    within a case are not independent). `clusters` is [(k, n), ...].
    `transform`, if given, maps (rate, rng) -> corrected rate, so the
    calibration-set uncertainty is resampled inside the same loop.
    """
    clusters = [c for c in clusters if c[1] > 0]
    if not clusters:
        return (None, None)
    rng = random.Random(seed)
    draws = []
    m = len(clusters)
    for _ in range(_BOOTSTRAP_DRAWS):
        k = n = 0
        for _ in range(m):
            ck, cn = clusters[rng.randrange(m)]
            k += ck
            n += cn
        rate = k / n
        if transform is not None:
            rate = transform(rate, rng)
            if rate is None:
                continue
        draws.append(rate)
    if not draws:
        return (None, None)
    draws.sort()
    return (_percentile(draws, 0.025), _percentile(draws, 0.975))


def _modal(labels: list) -> str:
    """Most frequent label; a tie for first place is UNCLEAR (use an odd sample count)."""
    if not labels:
        return UNCLEAR
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return UNCLEAR
    return counts[0][0]


def _kappa(pairs: list) -> Optional[float]:
    """Cohen's kappa over (a, b) label pairs."""
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


# ─────────────────────────────────────────────────────────────────────────────
# Observations and per-state results
# ─────────────────────────────────────────────────────────────────────────────

CONTRAST_PREFIX = "contrast:"


@dataclass
class Observation:
    state: str
    case_id: str
    input_id: str           # "canonical", "v1".., or "contrast:<id>"
    question: str
    answer: str
    label: str
    error: Optional[str] = None
    sample: int = 0
    human_label: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "state": self.state, "case_id": self.case_id, "input_id": self.input_id,
            "sample": self.sample, "question": self.question, "answer": self.answer,
            "label": self.label,
        }
        if self.error:
            d["error"] = self.error
        if self.human_label is not None:
            d["human_label"] = self.human_label
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        hl = d.get("human_label")
        return cls(
            state=str(d["state"]), case_id=str(d["case_id"]), input_id=str(d["input_id"]),
            question=str(d.get("question", "")), answer=str(d.get("answer", "")),
            label=str(d.get("label", UNCLEAR)), error=d.get("error"),
            sample=int(d.get("sample", 0)),
            human_label=str(hl).strip() if hl not in (None, "") else None,
        )


@dataclass
class CaseStateResult:
    """One case under one policy state. Every input may have been asked k times."""
    state: str
    case_id: str
    expected: str
    samples: dict = field(default_factory=dict)            # {input_id: [label, ...]}
    contrast_samples: dict = field(default_factory=dict)   # {contrast_id: [label, ...]}
    contrast_expected: dict = field(default_factory=dict)  # {contrast_id: label}

    @property
    def labels(self) -> dict:
        """{input_id: modal label across that input's samples}."""
        return {k: _modal(v) for k, v in self.samples.items()}

    @property
    def contrast_labels(self) -> dict:
        return {k: _modal(v) for k, v in self.contrast_samples.items()}

    @property
    def canonical_label(self) -> str:
        return self.labels.get("canonical", UNCLEAR)

    @property
    def invariant(self) -> bool:
        """SI for this case: every input's modal outcome is one and the same."""
        vals = set(self.labels.values())
        return len(vals) == 1 and UNCLEAR not in vals

    @property
    def strictly_invariant(self) -> bool:
        """pass^k-style: every sample of every input gave the same outcome."""
        vals = {l for v in self.samples.values() for l in v}
        return len(vals) == 1 and UNCLEAR not in vals

    @property
    def noise_explainable(self) -> bool:
        """
        Some outcome was produced at least once by every input, so a different
        draw could have been unanimous: a failure that sampling noise alone
        can explain. With one sample per input this equals `invariant`.
        """
        sets = [set(v) - {UNCLEAR} for v in self.samples.values()]
        return bool(sets) and bool(set.intersection(*sets))

    @property
    def grounded(self) -> bool:
        """PG for this case: every input's modal outcome is the warranted one."""
        labels = self.labels
        return bool(labels) and all(v == self.expected for v in labels.values())

    @property
    def n_grounded(self) -> int:
        return sum(1 for v in self.labels.values() if v == self.expected)

    @property
    def noisy_inputs(self) -> list:
        return sorted(k for k, v in self.samples.items() if len(v) > 1 and len(set(v)) > 1)

    @property
    def deviating_inputs(self) -> list:
        canon = self.canonical_label
        return sorted(k for k, v in self.labels.items() if v != canon or v == UNCLEAR)

    def to_dict(self) -> dict:
        d = {
            "case_id": self.case_id,
            "expected": self.expected,
            "labels": self.labels,
            "invariant": self.invariant,
            "grounded": self.grounded,
        }
        if any(len(v) > 1 for v in self.samples.values()):
            d["samples"] = {k: list(v) for k, v in self.samples.items()}
            d["strictly_invariant"] = self.strictly_invariant
            d["noise_explainable"] = self.noise_explainable
        if self.contrast_samples:
            d["contrasts"] = {
                cid: {"expected": self.contrast_expected.get(cid), "label": lab}
                for cid, lab in self.contrast_labels.items()
            }
        return d


@dataclass
class StateResult:
    """All cases under one policy state."""
    state: str
    cases: dict             # {case_id: CaseStateResult}

    @property
    def invariance_rate(self) -> Optional[float]:
        if not self.cases:
            return None
        return sum(1 for c in self.cases.values() if c.invariant) / len(self.cases)

    @property
    def strict_invariance_rate(self) -> Optional[float]:
        if not self.cases:
            return None
        return sum(1 for c in self.cases.values() if c.strictly_invariant) / len(self.cases)

    @property
    def grounding_rate(self) -> Optional[float]:
        n = sum(len(c.labels) for c in self.cases.values())
        if not n:
            return None
        return sum(c.n_grounded for c in self.cases.values()) / n

    @property
    def noise_rate(self) -> Optional[float]:
        """Share of repeatedly-sampled inputs whose identical prompt got different outcomes."""
        multi = [(c, k) for c in self.cases.values() for k, v in c.samples.items() if len(v) > 1]
        if not multi:
            return None
        return sum(1 for c, k in multi if len(set(c.samples[k])) > 1) / len(multi)

    @property
    def n_contrasts(self) -> int:
        return sum(len(c.contrast_samples) for c in self.cases.values())

    @property
    def fact_sensitivity_rate(self) -> Optional[float]:
        n = self.n_contrasts
        if not n:
            return None
        hits = sum(
            1 for c in self.cases.values() for cid, lab in c.contrast_labels.items()
            if lab == c.contrast_expected.get(cid)
        )
        return hits / n

    def contrast_failures(self) -> list:
        """[(case_id, contrast_id, label, expected, ignored_fact)]"""
        out = []
        for c in self.cases.values():
            for cid, lab in c.contrast_labels.items():
                exp = c.contrast_expected.get(cid)
                if lab != exp:
                    out.append((c.case_id, cid, lab, exp, lab == c.canonical_label and lab != UNCLEAR))
        return out

    def to_dict(self) -> dict:
        d = {
            "state": self.state,
            "invariance_rate": self.invariance_rate,
            "grounding_rate": self.grounding_rate,
            "cases": [c.to_dict() for c in self.cases.values()],
        }
        if self.noise_rate is not None:
            d["noise_rate"] = self.noise_rate
            d["strict_invariance_rate"] = self.strict_invariance_rate
        if self.n_contrasts:
            d["fact_sensitivity_rate"] = self.fact_sensitivity_rate
        return d


# Per-case outcome of one amendment.
WARRANTED = "warranted_change"
HELD = "held"
RIGIDITY = "rigidity"
DRIFT = "drift"
MISDIRECTED = "misdirected"
INDETERMINATE = "indeterminate"


@dataclass
class TransitionResult:
    """WC for one amendment: base -> amended, compared on canonical questions."""
    amendment_id: str
    meaning_preserving: bool
    verdict: Optional[MinimalDeltaVerdict]
    per_case: dict          # {case_id: status}
    before: dict            # {case_id: label}
    after: dict             # {case_id: label}
    unanchored: list        # cases whose base answer was not the warranted one

    @property
    def passed(self) -> bool:
        if any(s == INDETERMINATE for s in self.per_case.values()):
            return False
        return all(s in (WARRANTED, HELD) for s in self.per_case.values())

    def cases_with(self, status: str) -> list:
        return sorted(k for k, v in self.per_case.items() if v == status)

    def summary(self) -> str:
        tag = "control, " if self.meaning_preserving else ""
        s = f"{self.amendment_id} ({tag}{'EXACT' if self.passed else 'NOT EXACT'})"
        parts = []
        for status in (WARRANTED, RIGIDITY, DRIFT, MISDIRECTED, INDETERMINATE):
            ids = self.cases_with(status)
            if ids:
                parts.append(f"{status}={ids}")
        return s + ("  " + "  ".join(parts) if parts else "  no change warranted, none made")

    def to_dict(self) -> dict:
        return {
            "amendment_id": self.amendment_id,
            "meaning_preserving": self.meaning_preserving,
            "passed": self.passed,
            "per_case": dict(self.per_case),
            "before": dict(self.before),
            "after": dict(self.after),
            "unanchored": list(self.unanchored),
            "minimal_delta": self.verdict.to_dict() if self.verdict else None,
        }


@dataclass
class Obligation:
    """
    One obligation's measured value against its threshold, with a 95% interval.

    value      point estimate (judge-corrected for PG when a calibration is given;
               the uncorrected estimate is then in raw_value).
    ci_low/ci_high
               Wilson interval over the obligation's units (cases for SI,
               contrasts for FS, amendments for WC); a case-clustered bootstrap
               for PG, whose inputs are not independent within a case.
    gate       "point": passes iff value >= threshold. "resolved": also passes
               when the interval still reaches the threshold, i.e. it fails only
               on a shortfall that sampling noise cannot explain.
    """
    id: str                 # "SI[base]", "PG[window_45]", "FS[base]", "WC"
    kind: str               # semantic_invariance | policy_grounding | fact_sensitivity | warranted_change
    scope: str
    value: Optional[float]
    threshold: float
    n: int = 0
    unit: str = ""
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    gate: str = "point"
    raw_value: Optional[float] = None

    @property
    def meets_point(self) -> bool:
        return self.value is not None and self.value >= self.threshold - 1e-12

    @property
    def within_noise(self) -> bool:
        """Below a threshold under 1.0, but the 95% interval still reaches it."""
        # A true rate of exactly 1.0 can never produce a failure, so against a
        # threshold of 1.0 any observed shortfall is real, whatever the interval.
        return (not self.meets_point and self.threshold < 1 - 1e-12
                and self.ci_high is not None and self.ci_high >= self.threshold - 1e-12)

    @property
    def passed(self) -> bool:
        if self.meets_point:
            return True
        return self.gate == "resolved" and self.within_noise

    @property
    def verdict(self) -> str:
        if self.meets_point:
            return "ok"
        if self.passed:
            return "ok~"     # passed only because the shortfall is within noise
        return "FAIL~" if self.within_noise else "FAIL"

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "kind": self.kind, "scope": self.scope,
            "value": self.value, "threshold": self.threshold, "passed": self.passed,
            "n": self.n, "unit": self.unit, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "gate": self.gate, "within_noise": self.within_noise,
        }
        if self.raw_value is not None:
            d["raw_value"] = self.raw_value
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Classifier calibration against human labels
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassifierCalibration:
    """
    How well the outcome classifier agrees with human labels, from a small
    human-labelled sample of observations (see calibration_sample()).

    For policy grounding -- a binary "did the answer land on the warranted
    outcome" -- this gives the classifier's sensitivity q1 (it says grounded
    when a human says grounded) and specificity q0 (it says ungrounded when a
    human says ungrounded), and corrects an observed grounding rate p with the
    Rogan-Gladen estimator theta = (p + q0 - 1) / (q0 + q1 - 1). That
    correction stays valid when the grounding rate under test differs from
    the calibration sample's (prevalence shift), which is exactly the case
    for a new model or an amended policy. Needs at least one truly-grounded
    and one truly-ungrounded item, and a classifier better than chance
    (q0 + q1 > 1); otherwise `correctable` is False and nothing is corrected.
    """
    items: list             # [(expected, classifier_label, human_label)]

    @classmethod
    def from_observations(cls, contract: "PolicyContract", observations: list) -> "ClassifierCalibration":
        items = []
        for o in observations:
            if isinstance(o, dict):
                o = Observation.from_dict(o)
            if o.human_label is None or o.case_id not in contract.case_map:
                continue
            exp = contract.expected_for_input(o.case_id, o.state, o.input_id)
            if exp is None:
                continue
            items.append((exp, o.label, o.human_label))
        return cls(items)

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def accuracy(self) -> Optional[float]:
        if not self.items:
            return None
        return sum(1 for _, c, h in self.items if c == h) / len(self.items)

    @property
    def accuracy_ci(self) -> tuple:
        return wilson_interval(sum(1 for _, c, h in self.items if c == h), len(self.items))

    @property
    def kappa(self) -> Optional[float]:
        return _kappa([(c, h) for _, c, h in self.items])

    def _q(self, items) -> tuple:
        pos = [(e, c) for e, c, h in items if h == e]
        neg = [(e, c) for e, c, h in items if h != e]
        if not pos or not neg:
            return (None, None)
        q1 = sum(1 for e, c in pos if c == e) / len(pos)
        q0 = sum(1 for e, c in neg if c != e) / len(neg)
        return (q1, q0)

    @property
    def sensitivity(self) -> Optional[float]:
        return self._q(self.items)[0]

    @property
    def specificity(self) -> Optional[float]:
        return self._q(self.items)[1]

    @property
    def correctable(self) -> bool:
        q1, q0 = self._q(self.items)
        return q1 is not None and q0 + q1 - 1 > 1e-9

    @staticmethod
    def _rogan_gladen(p: float, q1: float, q0: float) -> Optional[float]:
        den = q0 + q1 - 1
        if den <= 1e-9:
            return None
        return min(1.0, max(0.0, (p + q0 - 1) / den))

    def correct(self, p: Optional[float]) -> Optional[float]:
        if p is None or not self.correctable:
            return None
        q1, q0 = self._q(self.items)
        return self._rogan_gladen(p, q1, q0)

    def resampled_correction(self):
        """transform(rate, rng) for _cluster_bootstrap: resamples the calibration set too."""
        items = self.items

        def transform(rate, rng):
            boot = [items[rng.randrange(len(items))] for _ in range(len(items))]
            q1, q0 = self._q(boot)
            if q1 is None:
                return None
            return self._rogan_gladen(rate, q1, q0)
        return transform

    def expected_false_si_failures(self, result: "ContractResult") -> Optional[float]:
        """
        If every case were truly invariant, roughly how many SI failures would
        classifier errors alone produce? (1 - accuracy^m per case-state with m
        inputs, assuming independent errors.) Compare with the observed count.
        """
        acc = self.accuracy
        if acc is None:
            return None
        return sum(
            1 - acc ** len(c.labels)
            for sr in result.states.values() for c in sr.cases.values() if len(c.labels) > 1
        )

    def summary(self) -> str:
        if not self.items:
            return "classifier calibration: no human-labelled observations"
        lo, hi = self.accuracy_ci
        s = (f"classifier vs humans: n={self.n}  agreement={self.accuracy * 100:.1f}% "
             f"(95% CI {lo * 100:.1f}-{hi * 100:.1f}%)")
        if self.kappa is not None:
            s += f"  kappa={self.kappa:.2f}"
        if self.correctable:
            s += f"  grounding sens={self.sensitivity:.2f} spec={self.specificity:.2f}"
        else:
            s += ("  (PG not corrected: the sample needs both grounded and ungrounded "
                  "answers, and a better-than-chance classifier)")
        return s

    def to_dict(self) -> dict:
        lo, hi = self.accuracy_ci
        return {
            "n": self.n, "accuracy": self.accuracy, "accuracy_ci": [lo, hi],
            "kappa": self.kappa, "sensitivity": self.sensitivity,
            "specificity": self.specificity, "correctable": self.correctable,
        }


def calibration_sample(observations: list, n: int = 40, seed: int = 0) -> list:
    """
    Pick up to n observations for humans to label, stratified across the
    classifier's labels (round-robin) so rare outcomes -- usually the failures
    -- are represented. Returns observation dicts with an empty "human_label"
    to fill in; feed the filled file back as a calibration.
    """
    obs = [Observation.from_dict(o) if isinstance(o, dict) else o for o in observations]
    obs = [o for o in obs if not o.error]
    rng = random.Random(seed)
    by_label: dict = {}
    for o in obs:
        by_label.setdefault(o.label, []).append(o)
    for v in by_label.values():
        rng.shuffle(v)
    picked = []
    keys = sorted(by_label)
    while len(picked) < n and any(by_label[k] for k in keys):
        for k in keys:
            if by_label[k] and len(picked) < n:
                picked.append(by_label[k].pop())
    out = []
    for o in picked:
        d = o.to_dict()
        d["human_label"] = ""
        out.append(d)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContractResult:
    contract_id: str
    domain: str
    states: dict            # {state: StateResult}
    transitions: dict       # {amendment_id: TransitionResult}
    obligations: list
    by_clause: dict
    observations: list = field(default_factory=list)
    lint: list = field(default_factory=list)
    samples: int = 1
    calibration: Optional[ClassifierCalibration] = None

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.obligations)

    @property
    def warranted_change_rate(self) -> Optional[float]:
        if not self.transitions:
            return None
        return sum(1 for t in self.transitions.values() if t.passed) / len(self.transitions)

    def failing_obligations(self) -> list:
        return [o for o in self.obligations if not o.passed]

    def summary(self) -> str:
        n_ok = sum(1 for o in self.obligations if o.passed)
        return (
            f"{self.contract_id}: {'PASS' if self.passed else 'FAIL'}  "
            f"({n_ok}/{len(self.obligations)} obligations met)"
        )

    def report(self) -> str:
        def pct(x):
            return "  n/a " if x is None else f"{x * 100:5.1f}%"

        def ci(o):
            if o.ci_low is None:
                return " " * 13
            return f"{o.ci_low * 100:5.1f}-{o.ci_high * 100:5.1f}%"

        lines = [
            "",
            f"  POLICY EVALUATION CONTRACT  *  {self.contract_id}  ({self.domain})"
            + (f"  *  {self.samples} samples/input" if self.samples > 1 else ""),
            "-" * 78,
            "",
            "  obligation                         value   95% CI          n   need   result",
        ]
        for o in self.obligations:
            lines.append(
                f"  {o.id:<32s} {pct(o.value)}  {ci(o)}  {o.n:>3d}  {pct(o.threshold)}   {o.verdict}"
            )
        if any(o.verdict in ("FAIL~", "ok~") for o in self.obligations):
            lines.append("  ~ = the shortfall is within the 95% interval (could be sampling noise)")
        corrected = [o for o in self.obligations if o.raw_value is not None]
        if corrected:
            lines.append("  PG values are corrected for classifier error (raw: "
                         + ", ".join(f"{o.id}={pct(o.raw_value).strip()}" for o in corrected) + ")")
        if self.calibration is not None:
            lines.append(f"  {self.calibration.summary()}")
            exp = self.calibration.expected_false_si_failures(self)
            if exp is not None:
                obs_n = sum(1 for sr in self.states.values() for c in sr.cases.values() if not c.invariant)
                lines.append(f"  SI failures observed: {obs_n}; expected from classifier error alone: {exp:.1f}")

        noise = [(s, sr.noise_rate) for s, sr in self.states.items() if sr.noise_rate is not None]
        if noise:
            lines += ["", "  sampling noise floor (same prompt re-asked, different outcome):"]
            lines.append("    " + "  ".join(f"[{s}] {pct(r).strip()}" for s, r in noise))

        si_fail = [
            (s, c) for s, sr in self.states.items() for c in sr.cases.values() if not c.invariant
        ]
        if si_fail:
            lines += ["", "  semantic invariance failures (same meaning, different outcome):"]
            for s, c in si_fail:
                labels = ", ".join(f"{k}={v}" for k, v in c.labels.items())
                tag = "  (within sampling noise)" if self.samples > 1 and c.noise_explainable else ""
                lines.append(f"    [{s}] {c.case_id}: {labels}{tag}")

        pg_fail = [
            (s, c) for s, sr in self.states.items() for c in sr.cases.values()
            if c.invariant and not c.grounded
        ]
        if pg_fail:
            lines += ["", "  consistent but ungrounded (stable answer the policy does not warrant):"]
            for s, c in pg_fail:
                lines.append(f"    [{s}] {c.case_id}: answered {c.canonical_label}, warranted {c.expected}")

        fs_fail = [(s, f) for s, sr in self.states.items() for f in sr.contrast_failures()]
        if fs_fail:
            lines += ["", "  fact sensitivity failures (decisive fact changed, outcome should have):"]
            for s, (case_id, cid, lab, exp, ignored) in fs_fail:
                why = "fact ignored" if ignored else "wrong outcome"
                lines.append(f"    [{s}] {case_id}/{cid}: answered {lab}, warranted {exp}  ({why})")

        if self.transitions:
            lines += ["", "  warranted change, per amendment:"]
            for t in self.transitions.values():
                lines.append(f"    {t.summary()}")
                if t.unanchored:
                    lines.append(f"      (base answer already off-policy for: {t.unanchored})")

        fragile = [
            (cid, d) for cid, d in self.by_clause.items()
            if d["invariance_failures"] or d["grounding_failures"] or d["transition_failures"]
            or d.get("fact_failures")
        ]
        if fragile:
            lines += ["", "  clauses implicated in failures:"]
            for cid, d in sorted(fragile, key=lambda kv: -(kv[1]["invariance_failures"]
                                                           + kv[1]["grounding_failures"]
                                                           + kv[1]["transition_failures"]
                                                           + kv[1].get("fact_failures", 0))):
                lines.append(
                    f"    [{cid}] invariance={d['invariance_failures']}  "
                    f"grounding={d['grounding_failures']}  facts={d.get('fact_failures', 0)}  "
                    f"change={d['transition_failures']}"
                )

        lines += ["", f"  {self.summary()}", ""]
        return "\n".join(lines)

    def to_dict(self, include_observations: bool = True) -> dict:
        d = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "domain": self.domain,
            "passed": self.passed,
            "samples": self.samples,
            "obligations": [o.to_dict() for o in self.obligations],
            "states": [s.to_dict() for s in self.states.values()],
            "transitions": [t.to_dict() for t in self.transitions.values()],
            "by_clause": self.by_clause,
            "lint": [i.to_dict() for i in self.lint],
        }
        if self.calibration is not None:
            d["calibration"] = self.calibration.to_dict()
        if include_observations:
            d["observations"] = [o.to_dict() for o in self.observations]
        return d


def _transition(contract: PolicyContract, amendment: Amendment,
                base: StateResult, after: StateResult, threshold: float = 0.5) -> TransitionResult:
    before_labels = {cid: r.canonical_label for cid, r in base.cases.items()}
    after_labels = {cid: r.canonical_label for cid, r in after.cases.items()}

    per_case: dict = {}
    sensitivity: dict = {}
    effect_matches: dict = {}
    for case in contract.cases:
        b = before_labels.get(case.id, UNCLEAR)
        a = after_labels.get(case.id, UNCLEAR)
        should = case.id in amendment.expected_changes
        if b == UNCLEAR or a == UNCLEAR:
            per_case[case.id] = INDETERMINATE
            continue           # left out of sensitivity -> reported as unmeasured, not guessed
        changed = a != b
        sensitivity[case.id] = 1.0 if changed else 0.0
        if should:
            landed = a == amendment.expected_changes[case.id]
            if changed:
                effect_matches[case.id] = landed
                per_case[case.id] = WARRANTED if landed else MISDIRECTED
            else:
                per_case[case.id] = RIGIDITY
        else:
            per_case[case.id] = DRIFT if changed else HELD

    verdict = None
    spec = intervention_delta_spec(
        intervention_id=amendment.id,
        domain=contract.domain,
        justified_commitments={
            cid: label for cid, label in amendment.expected_changes.items() if cid in sensitivity
        },
        invariant_commitments=[
            c.id for c in contract.cases if c.id not in amendment.expected_changes and c.id in sensitivity
        ],
    )
    if spec.factors:
        report = score_dependency_structure(
            spec, sensitivity, threshold=threshold,
            expected_effect_matches=effect_matches or None,
        )
        verdict = score_minimal_delta(report)

    unanchored = sorted(
        cid for cid, b in before_labels.items() if b != contract.expected(cid, BASE_STATE)
    )
    return TransitionResult(
        amendment_id=amendment.id,
        meaning_preserving=amendment.meaning_preserving,
        verdict=verdict,
        per_case=per_case,
        before=before_labels,
        after=after_labels,
        unanchored=unanchored,
    )


def evaluate_contract(contract: PolicyContract, observations: list,
                      lint: Optional[list] = None,
                      calibration: Optional[ClassifierCalibration] = None,
                      seed: int = 0) -> ContractResult:
    """
    Pure scoring step: turn labelled observations into obligations. No model
    or judge calls, so any source of observations -- run_contract(), a saved
    result file, replayed production logs, hand labels -- is scored
    identically. Observations may repeat an input (k samples); each input's
    outcome is then its modal label, and the disagreement between samples of
    the identical prompt is reported as the noise floor.

    calibration  a ClassifierCalibration from human labels; when it can
                 correct, PG values are Rogan-Gladen corrected (raw kept).
    seed         for the bootstrap intervals (results are deterministic).
    """
    observations = [Observation.from_dict(o) if isinstance(o, dict) else o for o in observations]
    states: dict = {s: StateResult(s, {}) for s in contract.states()}
    max_k = 1
    per_input: Counter = Counter()
    for obs in observations:
        sr = states.get(obs.state)
        case = contract.case_map.get(obs.case_id)
        if sr is None or case is None:
            continue
        csr = sr.cases.get(obs.case_id)
        if csr is None:
            csr = sr.cases[obs.case_id] = CaseStateResult(
                obs.state, obs.case_id, contract.expected(obs.case_id, obs.state)
            )
        if obs.input_id.startswith(CONTRAST_PREFIX):
            cid = obs.input_id[len(CONTRAST_PREFIX):]
            exp = contract.expected_for_input(obs.case_id, obs.state, obs.input_id)
            if exp is None:
                continue
            csr.contrast_samples.setdefault(cid, []).append(obs.label)
            csr.contrast_expected[cid] = exp
        else:
            csr.samples.setdefault(obs.input_id, []).append(obs.label)
        per_input[(obs.state, obs.case_id, obs.input_id)] += 1
    if per_input:
        max_k = max(per_input.values())

    t = contract.thresholds
    g = t.gate
    if calibration is not None and not calibration.items:
        calibration = None
    obligations = []
    for s, sr in states.items():
        if not sr.cases:
            continue
        n_cases = len(sr.cases)
        # SI is only measured where some case was asked more than one way;
        # a state probed with canonical questions only would pass it vacuously.
        if any(len(c.samples) > 1 for c in sr.cases.values()):
            k = sum(1 for c in sr.cases.values() if c.invariant)
            lo, hi = wilson_interval(k, n_cases)
            obligations.append(Obligation(
                f"SI[{s}]", "semantic_invariance", s, sr.invariance_rate, t.invariance,
                n=n_cases, unit="cases", ci_low=lo, ci_high=hi, gate=g,
            ))
        clusters = [(c.n_grounded, len(c.labels)) for c in sr.cases.values()]
        raw = sr.grounding_rate
        n_inputs = sum(n for _, n in clusters)
        if calibration is not None and calibration.correctable:
            value = calibration.correct(raw)
            lo, hi = _cluster_bootstrap(clusters, seed=seed, transform=calibration.resampled_correction())
            obligations.append(Obligation(
                f"PG[{s}]", "policy_grounding", s, value, t.grounding,
                n=n_inputs, unit="inputs", ci_low=lo, ci_high=hi, gate=g, raw_value=raw,
            ))
        else:
            if raw is not None and raw in (0.0, 1.0):
                # The bootstrap degenerates at 0% / 100%; fall back to a Wilson
                # interval over cases (clusters), which is conservative.
                lo, hi = wilson_interval(round(raw * n_cases), n_cases)
            else:
                lo, hi = _cluster_bootstrap(clusters, seed=seed)
            obligations.append(Obligation(
                f"PG[{s}]", "policy_grounding", s, raw, t.grounding,
                n=n_inputs, unit="inputs", ci_low=lo, ci_high=hi, gate=g,
            ))
        if sr.n_contrasts:
            n = sr.n_contrasts
            k = round(sr.fact_sensitivity_rate * n)
            lo, hi = wilson_interval(k, n)
            obligations.append(Obligation(
                f"FS[{s}]", "fact_sensitivity", s, sr.fact_sensitivity_rate, t.fact_sensitivity,
                n=n, unit="contrasts", ci_low=lo, ci_high=hi, gate=g,
            ))

    transitions: dict = {}
    base = states[BASE_STATE]
    for a in contract.amendments:
        if base.cases and states[a.id].cases:
            transitions[a.id] = _transition(contract, a, base, states[a.id])
    if transitions:
        k = sum(1 for tr in transitions.values() if tr.passed)
        n = len(transitions)
        lo, hi = wilson_interval(k, n)
        obligations.append(Obligation(
            "WC", "warranted_change", "all amendments", k / n, t.warranted_change,
            n=n, unit="amendments", ci_low=lo, ci_high=hi, gate=g,
        ))

    by_clause: dict = {}
    for c in contract.clauses + [
        Clause(cid, txt) for a in contract.amendments for cid, txt in a.set_clauses.items()
        if cid not in {x.id for x in contract.clauses}
    ]:
        by_clause.setdefault(c.id, {"cases": [], "invariance_failures": 0, "grounding_failures": 0,
                                    "fact_failures": 0, "transition_failures": 0})
    for case in contract.cases:
        for cid in case.clauses:
            if cid in by_clause:
                by_clause[cid]["cases"].append(case.id)
    for sr in states.values():
        fs_cases = Counter(case_id for case_id, *_ in sr.contrast_failures())
        for csr in sr.cases.values():
            for cid in contract.case_map[csr.case_id].clauses:
                if cid not in by_clause:
                    continue
                if not csr.invariant:
                    by_clause[cid]["invariance_failures"] += 1
                if not csr.grounded:
                    by_clause[cid]["grounding_failures"] += 1
                by_clause[cid]["fact_failures"] += fs_cases.get(csr.case_id, 0)
    for tr in transitions.values():
        touched = contract.amendment_map[tr.amendment_id].touched_clauses()
        for case_id, status in tr.per_case.items():
            if status in (WARRANTED, HELD):
                continue
            # Attribute to the amended clause(s) the case depends on -- the
            # clause whose change the model mishandled. Drift on a case that
            # cites no amended clause is attributed to the case's own clauses.
            cited = contract.case_map[case_id].clauses
            for cid in ([c for c in cited if c in touched] or cited):
                if cid in by_clause:
                    by_clause[cid]["transition_failures"] += 1

    return ContractResult(
        contract_id=contract.contract_id,
        domain=contract.domain,
        states=states,
        transitions=transitions,
        obligations=obligations,
        by_clause=by_clause,
        observations=list(observations),
        lint=list(lint or []),
        samples=max_k,
        calibration=calibration,
    )


def run_contract(
    contract: PolicyContract,
    model_fn: ModelFn,
    classifier: Classifier,
    amendment_variants: bool = True,
    workers: int = 4,
    strict_lint: bool = True,
    samples: int = 1,
    calibration: Optional[ClassifierCalibration] = None,
) -> ContractResult:
    """
    Probe an assistant against a contract and score it.

    model_fn(system_prompt, question) -> answer. The contract renders its own
    policy into system_prompt for every state, the same (system_prompt,
    question) calling convention `contradish update` uses, because the
    measurement varies the governing policy itself.

    amendment_variants  also ask every variant under each amended policy
                        (default). False asks only canonical questions under
                        amendments: WC is still fully measured, SI/PG only
                        under the base policy. Cuts calls by roughly
                        (#variants+1)x per amendment.
    samples             ask every input this many times (use an odd number).
                        Each input's outcome becomes its modal label, and
                        disagreement between repeats of the identical prompt
                        is reported as the noise floor, so a semantic
                        invariance failure can be told apart from sampling
                        noise. Multiplies calls by `samples`.
    calibration         human-label calibration of the classifier; corrects
                        PG for classifier error (see ClassifierCalibration).
    strict_lint         refuse to run a contract with lint errors.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    issues = contract.lint()
    errors = [i for i in issues if i.level == "error"]
    if strict_lint and errors:
        raise ValueError(
            "contract has lint errors; fix them or pass strict_lint=False:\n"
            + "\n".join(f"  {e}" for e in errors)
        )

    jobs = []
    for state in contract.states():
        system_prompt = contract.policy_text(state)
        include_variants = state == BASE_STATE or amendment_variants
        for case in contract.cases:
            outcomes = contract.outcomes_for(case)
            inputs = list(case.inputs(include_variants))
            if state == BASE_STATE:
                inputs += [(CONTRAST_PREFIX + x.id, x.question) for x in case.contrasts]
            for input_id, question in inputs:
                for k in range(samples):
                    jobs.append((state, system_prompt, case.id, input_id, question, outcomes, k))

    def run_one(job) -> Observation:
        state, system_prompt, case_id, input_id, question, outcomes, k = job
        try:
            answer = model_fn(system_prompt, question)
        except Exception as e:  # a failed call is an unmeasured input, not a crash
            return Observation(state, case_id, input_id, question, "", UNCLEAR,
                               error=f"model: {type(e).__name__}: {e}", sample=k)
        try:
            label = classifier(question, answer, outcomes)
        except Exception as e:
            return Observation(state, case_id, input_id, question, answer, UNCLEAR,
                               error=f"classifier: {type(e).__name__}: {e}", sample=k)
        if label not in outcomes:
            label = UNCLEAR
        return Observation(state, case_id, input_id, question, answer, label, sample=k)

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            observations = list(pool.map(run_one, jobs))
    else:
        observations = [run_one(j) for j in jobs]

    return evaluate_contract(contract, observations, lint=issues, calibration=calibration)
