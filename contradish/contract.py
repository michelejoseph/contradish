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
to when P is its governing policy. The contract has three obligations:

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

SI and WC are the same principle seen from both sides: behavior must be a
function of the policy-relevant content of the situation -- insensitive to
everything else (surface form, framing, pressure, a meaning-preserving
rewording of the policy itself), and sensitive to exactly the changes in
that content the policy licenses. WC is scored with the existing
minimal_intervention_delta / decision_relevance core (no new scoring math);
this module supplies the policy-grounded spec that core needs and the probe
that measures against it.

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
    "default_outcome_classifier",
    "evaluate_contract",
    "run_contract",
    "builtin_contract_path",
    "load_builtin_contract",
    "list_builtin_contracts",
]

SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "1.0"
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
    """
    id: str
    clauses: list
    question: str
    expected: str
    variants: list = field(default_factory=list)
    outcomes: Optional[list] = None

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
    """
    invariance: float = 1.0
    grounding: float = 1.0
    warranted_change: float = 1.0

    def to_dict(self) -> dict:
        return {
            "invariance": self.invariance,
            "grounding": self.grounding,
            "warranted_change": self.warranted_change,
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
        )
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
    what lets one judgment serve all three obligations: SI compares labels
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


@dataclass
class Observation:
    state: str
    case_id: str
    input_id: str
    question: str
    answer: str
    label: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "state": self.state, "case_id": self.case_id, "input_id": self.input_id,
            "question": self.question, "answer": self.answer, "label": self.label,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class CaseStateResult:
    """One case under one policy state."""
    state: str
    case_id: str
    expected: str
    labels: dict            # {input_id: label}

    @property
    def canonical_label(self) -> str:
        return self.labels.get("canonical", UNCLEAR)

    @property
    def invariant(self) -> bool:
        """SI for this case: every input committed to one and the same outcome."""
        vals = set(self.labels.values())
        return len(vals) == 1 and UNCLEAR not in vals

    @property
    def grounded(self) -> bool:
        """PG for this case: every input landed on the warranted outcome."""
        return bool(self.labels) and all(v == self.expected for v in self.labels.values())

    @property
    def n_grounded(self) -> int:
        return sum(1 for v in self.labels.values() if v == self.expected)

    @property
    def deviating_inputs(self) -> list:
        return sorted(k for k, v in self.labels.items() if v != self.canonical_label or v == UNCLEAR)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "expected": self.expected,
            "labels": dict(self.labels),
            "invariant": self.invariant,
            "grounded": self.grounded,
        }


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
    def grounding_rate(self) -> Optional[float]:
        n = sum(len(c.labels) for c in self.cases.values())
        if not n:
            return None
        return sum(c.n_grounded for c in self.cases.values()) / n

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "invariance_rate": self.invariance_rate,
            "grounding_rate": self.grounding_rate,
            "cases": [c.to_dict() for c in self.cases.values()],
        }


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
    id: str                 # "SI[base]", "PG[window_45]", "WC"
    kind: str               # semantic_invariance | policy_grounding | warranted_change
    scope: str
    value: Optional[float]
    threshold: float

    @property
    def passed(self) -> bool:
        return self.value is not None and self.value >= self.threshold - 1e-12

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "scope": self.scope,
            "value": self.value, "threshold": self.threshold, "passed": self.passed,
        }


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

        lines = [
            "",
            f"  POLICY EVALUATION CONTRACT  *  {self.contract_id}  ({self.domain})",
            "-" * 78,
            "",
            "  obligation                         value   need   result",
        ]
        for o in self.obligations:
            lines.append(
                f"  {o.id:<32s} {pct(o.value)}  {pct(o.threshold)}   {'ok' if o.passed else 'FAIL'}"
            )

        si_fail = [
            (s, c) for s, sr in self.states.items() for c in sr.cases.values() if not c.invariant
        ]
        if si_fail:
            lines += ["", "  semantic invariance failures (same meaning, different outcome):"]
            for s, c in si_fail:
                labels = ", ".join(f"{k}={v}" for k, v in c.labels.items())
                lines.append(f"    [{s}] {c.case_id}: {labels}")

        pg_fail = [
            (s, c) for s, sr in self.states.items() for c in sr.cases.values()
            if c.invariant and not c.grounded
        ]
        if pg_fail:
            lines += ["", "  consistent but ungrounded (stable answer the policy does not warrant):"]
            for s, c in pg_fail:
                lines.append(f"    [{s}] {c.case_id}: answered {c.canonical_label}, warranted {c.expected}")

        if self.transitions:
            lines += ["", "  warranted change, per amendment:"]
            for t in self.transitions.values():
                lines.append(f"    {t.summary()}")
                if t.unanchored:
                    lines.append(f"      (base answer already off-policy for: {t.unanchored})")

        fragile = [
            (cid, d) for cid, d in self.by_clause.items()
            if d["invariance_failures"] or d["grounding_failures"] or d["transition_failures"]
        ]
        if fragile:
            lines += ["", "  clauses implicated in failures:"]
            for cid, d in sorted(fragile, key=lambda kv: -(kv[1]["invariance_failures"]
                                                           + kv[1]["grounding_failures"]
                                                           + kv[1]["transition_failures"])):
                lines.append(
                    f"    [{cid}] invariance={d['invariance_failures']}  "
                    f"grounding={d['grounding_failures']}  change={d['transition_failures']}"
                )

        lines += ["", f"  {self.summary()}", ""]
        return "\n".join(lines)

    def to_dict(self, include_observations: bool = True) -> dict:
        d = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "domain": self.domain,
            "passed": self.passed,
            "obligations": [o.to_dict() for o in self.obligations],
            "states": [s.to_dict() for s in self.states.values()],
            "transitions": [t.to_dict() for t in self.transitions.values()],
            "by_clause": self.by_clause,
            "lint": [i.to_dict() for i in self.lint],
        }
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
                      lint: Optional[list] = None) -> ContractResult:
    """
    Pure scoring step: turn labelled observations into obligations. No model
    or judge calls, so any source of observations -- run_contract(), replayed
    production logs, hand labels -- is scored identically.
    """
    states: dict = {s: StateResult(s, {}) for s in contract.states()}
    for obs in observations:
        sr = states.get(obs.state)
        if sr is None or obs.case_id not in contract.case_map:
            continue
        csr = sr.cases.get(obs.case_id)
        if csr is None:
            csr = sr.cases[obs.case_id] = CaseStateResult(
                obs.state, obs.case_id, contract.expected(obs.case_id, obs.state), {}
            )
        csr.labels[obs.input_id] = obs.label

    t = contract.thresholds
    obligations = []
    for s, sr in states.items():
        if not sr.cases:
            continue
        # SI is only measured where some case was asked more than one way;
        # a state probed with canonical questions only would pass it vacuously.
        if any(len(c.labels) > 1 for c in sr.cases.values()):
            obligations.append(Obligation(f"SI[{s}]", "semantic_invariance", s, sr.invariance_rate, t.invariance))
        obligations.append(Obligation(f"PG[{s}]", "policy_grounding", s, sr.grounding_rate, t.grounding))

    transitions: dict = {}
    base = states[BASE_STATE]
    for a in contract.amendments:
        if base.cases and states[a.id].cases:
            transitions[a.id] = _transition(contract, a, base, states[a.id])
    if transitions:
        rate = sum(1 for tr in transitions.values() if tr.passed) / len(transitions)
        obligations.append(Obligation("WC", "warranted_change", "all amendments", rate, t.warranted_change))

    by_clause: dict = {}
    for c in contract.clauses + [
        Clause(cid, txt) for a in contract.amendments for cid, txt in a.set_clauses.items()
        if cid not in {x.id for x in contract.clauses}
    ]:
        by_clause.setdefault(c.id, {"cases": [], "invariance_failures": 0,
                                    "grounding_failures": 0, "transition_failures": 0})
    for case in contract.cases:
        for cid in case.clauses:
            if cid in by_clause:
                by_clause[cid]["cases"].append(case.id)
    for sr in states.values():
        for csr in sr.cases.values():
            for cid in contract.case_map[csr.case_id].clauses:
                if cid not in by_clause:
                    continue
                if not csr.invariant:
                    by_clause[cid]["invariance_failures"] += 1
                if not csr.grounded:
                    by_clause[cid]["grounding_failures"] += 1
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
    )


def run_contract(
    contract: PolicyContract,
    model_fn: ModelFn,
    classifier: Classifier,
    amendment_variants: bool = True,
    workers: int = 4,
    strict_lint: bool = True,
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
    strict_lint         refuse to run a contract with lint errors.
    """
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
            for input_id, question in case.inputs(include_variants):
                jobs.append((state, system_prompt, case.id, input_id, question, outcomes))

    def run_one(job) -> Observation:
        state, system_prompt, case_id, input_id, question, outcomes = job
        try:
            answer = model_fn(system_prompt, question)
        except Exception as e:  # a failed call is an unmeasured input, not a crash
            return Observation(state, case_id, input_id, question, "", UNCLEAR,
                               error=f"model: {type(e).__name__}: {e}")
        try:
            label = classifier(question, answer, outcomes)
        except Exception as e:
            return Observation(state, case_id, input_id, question, answer, UNCLEAR,
                               error=f"classifier: {type(e).__name__}: {e}")
        if label not in outcomes:
            label = UNCLEAR
        return Observation(state, case_id, input_id, question, answer, label)

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            observations = list(pool.map(run_one, jobs))
    else:
        observations = [run_one(j) for j in jobs]

    return evaluate_contract(contract, observations, lint=issues)
