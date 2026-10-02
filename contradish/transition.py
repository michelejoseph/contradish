"""
contradish/transition.py -- the transition contract: contradish's atomic object.

Intelligent systems need both persistence and revision. When governing
information changes, a correct system has to work out which of its
distinctions survive the new information and which must collapse. The unit
that states this is not a pair of responses. A pair of responses can only be
compared; it cannot say what should have happened. The unit is a
TRANSITION CONTRACT:

    before   the governing information the system had
    after    the governing information it has now
    cases    for each situation: the outcome warranted before, and the
             outcome warranted after

Everything else follows from those three things.

  persist / revise
      A case whose warranted outcome is the same before and after must
      PERSIST. A case whose warranted outcome differs must be REVISED, and to
      that outcome.

  distinctions
      Two cases are distinguished when they warrant different outcomes. Each
      pair of cases therefore has a fate under the transition:
        survive        distinguished before, distinguished after
        collapse       distinguished before, same outcome after
        emerge         same outcome before, distinguished after
        stay_merged    same outcome before and after
      (Only pairs that depend on the same part of the governing information
      count, when cases declare their grounds.)
      "A 20-day and a 35-day return are treated differently" is a
      distinction. Extending the window from 30 to 45 days collapses it, and
      makes a new one emerge between 35 and 50 days.

  the null transition
      When the new information means the same as the old (a rewording, or no
      change at all), every case persists and every distinction keeps its
      fate. Semantic invariance is this special case, not a separate idea.

SCORING A TRANSITION -- the core question
------------------------------------------
"How faithfully did the system move from its previous behavioral state
toward the behavioral state warranted by its new governing information?"

Given what the system actually did before (B0), what it did after (B1), and
the warranted target (B*, the `after` outcomes):

  fidelity = 1 - (cases off target after) / (cases that needed to move)

  1 means it landed on the target; 0 means no net progress; negative means
  it ended further from the target than it began. "Needed to move" is judged
  from what the system actually did before, not from what it should have
  done: a system that was wrong and is now right moved faithfully. When
  nothing needed to move, fidelity is undefined and `hold` (the share of
  cases still on target) is the answer.

  Every case off target after the transition is exactly one of:
    rigid        needed to move, did not
    misdirected  needed to move, moved somewhere else
    drift        was on target, left it

  persistence = of the distinctions that must keep their relation (survive,
                stay_merged), the share that did
  revision    = of the distinctions that must change relation (collapse,
                emerge), the share that did

Persistence and revision are scored on the partition the system draws, not
on its labels; fidelity is scored on the labels. A system can draw the right
distinctions and still attach the wrong outcomes, and the two scores say so
separately.

TWO LEVELS OF "THE SYSTEM"
---------------------------
run_transition(delivery="fresh") asks under `before` and under `after` in
independent runs. That is the deployed system (model + policy) before and
after a policy update. run_transition(delivery="in_conversation") gives one
agent the `before` information, lets it answer, then delivers `after` in the
same conversation and asks again. That is one agent revising a commitment it
actually made. Both are transitions; they are different levels and are
labelled as such in the result.

Usage::

    from contradish.transition import TransitionContract, TransitionCase, GoverningState, evaluate_transition

    t = TransitionContract(
        id="window_30_to_45", domain="ecommerce",
        before=GoverningState("v1", "Refunds within 30 days of delivery."),
        after=GoverningState("v2", "Refunds within 45 days of delivery."),
        outcomes={"refund": "refund allowed", "no_return": "no refund"},
        cases=[
            TransitionCase("day_20", "Delivered 20 days ago. Refund?", before="refund", after="refund"),
            TransitionCase("day_35", "Delivered 35 days ago. Refund?", before="no_return", after="refund"),
            TransitionCase("day_50", "Delivered 50 days ago. Refund?", before="no_return", after="no_return"),
        ],
    )
    out = evaluate_transition(t, previous={"day_20": "refund", "day_35": "no_return", "day_50": "no_return"},
                                 current={"day_20": "refund", "day_35": "no_return", "day_50": "no_return"})
    out.fidelity      # 0.0  -- one case needed to move and did not
    out.rigid         # ["day_35"]
    out.revision      # 0.0  -- neither the collapse nor the emergence happened
    out.persistence   # 1.0
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional

__all__ = [
    "SCHEMA_VERSION",
    "UNCLEAR",
    "GoverningState",
    "TransitionCase",
    "Distinction",
    "TransitionContract",
    "TransitionOutcome",
    "evaluate_transition",
    "run_transition",
    "SURVIVE", "COLLAPSE", "EMERGE", "STAY_MERGED",
]

SCHEMA_VERSION = "1.0"
UNCLEAR = "unclear"

SURVIVE = "survive"
COLLAPSE = "collapse"
EMERGE = "emerge"
STAY_MERGED = "stay_merged"

MOVED = "moved"
HELD = "held"
RIGID = "rigid"
MISDIRECTED = "misdirected"
DRIFT = "drift"


# ─────────────────────────────────────────────────────────────────────────────
# The contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoverningState:
    """One version of the governing information, as the system is given it."""
    id: str
    text: str

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text}


@dataclass
class TransitionCase:
    """
    One situation, and what is warranted for it on each side of the transition.

    before / after  outcome labels warranted by the previous / new governing
                    information.
    variants        meaning-preserving rewordings of `question`.
    grounds         what in the governing information this case depends on
                    (clause ids, fact names): the trace from change to effect.
    """
    id: str
    question: str
    before: str
    after: str
    variants: list = field(default_factory=list)
    outcomes: Optional[list] = None
    grounds: list = field(default_factory=list)

    @property
    def revises(self) -> bool:
        return self.before != self.after

    @property
    def persists(self) -> bool:
        return self.before == self.after

    def to_dict(self) -> dict:
        d = {"id": self.id, "question": self.question, "before": self.before, "after": self.after}
        if self.variants:
            d["variants"] = list(self.variants)
        if self.outcomes is not None:
            d["outcomes"] = list(self.outcomes)
        if self.grounds:
            d["grounds"] = list(self.grounds)
        return d


@dataclass
class Distinction:
    """A pair of cases and what the transition warrants for the line between them."""
    a: str
    b: str
    fate: str      # survive | collapse | emerge | stay_merged

    @property
    def distinct_after(self) -> bool:
        return self.fate in (SURVIVE, EMERGE)

    @property
    def must_persist(self) -> bool:
        return self.fate in (SURVIVE, STAY_MERGED)

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "fate": self.fate}


@dataclass
class TransitionContract:
    """The atomic object: previous information, new information, and what each warrants."""
    id: str
    before: GoverningState
    after: GoverningState
    outcomes: dict
    cases: list
    domain: str = ""
    description: str = ""
    preamble: str = ""
    meaning_preserving: bool = False     # declared null transition: nothing may be revised
    schema_version: str = SCHEMA_VERSION

    # ── what the contract warrants ──────────────────────────────────────────

    @property
    def case_map(self) -> dict:
        return {c.id: c for c in self.cases}

    def persist(self) -> list:
        """Cases whose warranted outcome survives the new information."""
        return [c for c in self.cases if c.persists]

    def revise(self) -> list:
        """Cases whose warranted outcome the new information changes."""
        return [c for c in self.cases if c.revises]

    @property
    def is_null(self) -> bool:
        return not self.revise()

    def target(self, side: str = "after") -> dict:
        return {c.id: (c.after if side == "after" else c.before) for c in self.cases}

    def distinctions(self, related_only: bool = True) -> list:
        """
        Pairs of cases, with the fate the transition warrants for each.

        related_only (default): when cases declare `grounds`, only pairs that
        share a ground are returned. A refund question and a shipping question
        are trivially "distinguished"; keeping that line is no evidence of
        persistence. Cases with no grounds are paired with everything.
        """
        out = []
        use_grounds = related_only and any(c.grounds for c in self.cases)
        for x, y in combinations(self.cases, 2):
            if use_grounds and x.grounds and y.grounds and not (set(x.grounds) & set(y.grounds)):
                continue
            d_before = x.before != y.before
            d_after = x.after != y.after
            fate = (SURVIVE if d_after else COLLAPSE) if d_before else (EMERGE if d_after else STAY_MERGED)
            out.append(Distinction(x.id, y.id, fate))
        return out

    def system_text(self, side: str) -> str:
        state = self.after if side == "after" else self.before
        return (self.preamble.strip() + "\n\n" + state.text).strip() if self.preamble else state.text

    def outcomes_for(self, case: TransitionCase) -> dict:
        if case.outcomes is None:
            return dict(self.outcomes)
        return {k: self.outcomes[k] for k in case.outcomes if k in self.outcomes}

    # ── static validation ───────────────────────────────────────────────────

    def lint(self) -> list:
        """[(level, code, message)] -- errors make the contract unusable."""
        issues = []
        ids = [c.id for c in self.cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            issues.append(("error", "T001", f"duplicate case id(s): {', '.join(dupes)}"))
        if not self.cases:
            issues.append(("error", "T002", "transition contract has no cases"))
        if UNCLEAR in self.outcomes:
            issues.append(("error", "T003", f"outcome label {UNCLEAR!r} is reserved"))
        for c in self.cases:
            for side, label in (("before", c.before), ("after", c.after)):
                if label not in self.outcomes:
                    issues.append(("error", "T003", f"case {c.id!r} {side} outcome {label!r} is not declared"))
                elif c.outcomes is not None and label not in c.outcomes:
                    issues.append(("error", "T003", f"case {c.id!r} {side} outcome {label!r} is outside its outcomes"))
        if self.meaning_preserving and self.revise():
            issues.append(("error", "T004", "declared meaning_preserving, but these cases are revised: "
                                            + ", ".join(c.id for c in self.revise())))
        if not self.meaning_preserving:
            if self.before.text.strip() == self.after.text.strip():
                issues.append(("warning", "T101", "before and after are the same text: mark it meaning_preserving"))
            if not self.revise():
                issues.append(("warning", "T102", "no case is revised: revision is untested "
                                                  "(mark it meaning_preserving if that is intended)"))
            if not self.persist():
                issues.append(("warning", "T103", "no case persists: drift is untested"))
        return issues

    def lint_errors(self) -> list:
        return [i for i in self.lint() if i[0] == "error"]

    # ── bridges to the existing scoring core ────────────────────────────────

    def justified_and_invariant(self) -> tuple:
        """(justified {case: after outcome}, invariant [case]) for minimal_intervention_delta."""
        return ({c.id: c.after for c in self.revise()}, [c.id for c in self.persist()])

    # ── (de)serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "domain": self.domain,
            "description": self.description,
            "preamble": self.preamble,
            "meaning_preserving": self.meaning_preserving,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "outcomes": dict(self.outcomes),
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionContract":
        missing = [k for k in ("id", "before", "after", "outcomes", "cases") if k not in d]
        if missing:
            raise ValueError(f"transition contract is missing: {', '.join(missing)}")

        def state(x, default_id):
            if isinstance(x, str):
                return GoverningState(default_id, x)
            return GoverningState(str(x.get("id", default_id)), str(x["text"]))

        outcomes = d["outcomes"]
        if isinstance(outcomes, list):
            outcomes = {str(o): "" for o in outcomes}
        cases = [
            TransitionCase(
                id=str(c["id"]), question=str(c["question"]), before=str(c["before"]), after=str(c["after"]),
                variants=[str(v) for v in c.get("variants", [])],
                outcomes=[str(o) for o in c["outcomes"]] if c.get("outcomes") is not None else None,
                grounds=[str(g) for g in c.get("grounds", [])],
            )
            for c in d["cases"]
        ]
        return cls(
            id=str(d["id"]), domain=str(d.get("domain", "")), description=str(d.get("description", "")),
            preamble=str(d.get("preamble", "")), meaning_preserving=bool(d.get("meaning_preserving", False)),
            before=state(d["before"], "before"), after=state(d["after"], "after"),
            outcomes={str(k): str(v or "") for k, v in outcomes.items()}, cases=cases,
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        )

    @classmethod
    def load(cls, path) -> "TransitionContract":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError("Reading a YAML transition contract requires PyYAML.") from e
            return cls.from_dict(yaml.safe_load(text))
        return cls.from_dict(json.loads(text))


# ─────────────────────────────────────────────────────────────────────────────
# Scoring a transition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitionOutcome:
    """How faithfully a system moved, for one transition contract."""
    contract_id: str
    delivery: str                       # "fresh" | "in_conversation" | "observed"
    previous: dict                      # B0  {case: label}
    current: dict                       # B1  {case: label}
    target: dict                        # B*  {case: label}
    status: dict                        # {case: moved | held | rigid | misdirected | drift}
    needed_to_move: int
    off_target_after: int
    fidelity: Optional[float]           # None when nothing needed to move
    hold: float                         # share of cases on target after
    persistence: Optional[float]
    revision: Optional[float]
    n_must_persist: int = 0
    n_must_revise: int = 0
    broken_distinctions: list = field(default_factory=list)   # [(a, b, fate)]

    def cases_with(self, status: str) -> list:
        return sorted(k for k, v in self.status.items() if v == status)

    @property
    def rigid(self) -> list:
        return self.cases_with(RIGID)

    @property
    def drift(self) -> list:
        return self.cases_with(DRIFT)

    @property
    def misdirected(self) -> list:
        return self.cases_with(MISDIRECTED)

    @property
    def exact(self) -> bool:
        """Landed exactly on the warranted state."""
        return self.off_target_after == 0

    def summary(self) -> str:
        def pct(x):
            return "n/a" if x is None else f"{x * 100:.0f}%"
        fid = "n/a (nothing needed to move)" if self.fidelity is None else f"{self.fidelity:.2f}"
        s = (f"{self.contract_id} [{self.delivery}]  fidelity {fid}  hold {pct(self.hold)}  "
             f"persistence {pct(self.persistence)}  revision {pct(self.revision)}")
        parts = [f"{name}={ids}" for name, ids in
                 (("rigid", self.rigid), ("misdirected", self.misdirected), ("drift", self.drift)) if ids]
        return s + ("  " + "  ".join(parts) if parts else "")

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "delivery": self.delivery,
            "fidelity": self.fidelity,
            "hold": self.hold,
            "persistence": self.persistence,
            "revision": self.revision,
            "needed_to_move": self.needed_to_move,
            "off_target_after": self.off_target_after,
            "exact": self.exact,
            "status": dict(self.status),
            "previous": dict(self.previous),
            "current": dict(self.current),
            "target": dict(self.target),
            "n_must_persist": self.n_must_persist,
            "n_must_revise": self.n_must_revise,
            "broken_distinctions": [list(x) for x in self.broken_distinctions],
        }


def evaluate_transition(contract: TransitionContract, previous: dict, current: dict,
                        delivery: str = "observed") -> TransitionOutcome:
    """
    Score one observed transition. Pure: no model or judge calls.

    previous  {case_id: outcome label} the system produced under `before`
    current   {case_id: outcome label} it produced under `after`
    A missing or "unclear" label is never on target.
    """
    target = contract.target("after")
    status: dict = {}
    needed = off = 0
    for c in contract.cases:
        b0 = previous.get(c.id, UNCLEAR)
        b1 = current.get(c.id, UNCLEAR)
        t = target[c.id]
        was_on = b0 == t
        is_on = b1 == t
        if not was_on:
            needed += 1
        if not is_on:
            off += 1
        if was_on:
            status[c.id] = HELD if is_on else DRIFT
        elif is_on:
            status[c.id] = MOVED
        else:
            status[c.id] = RIGID if b1 == b0 and b1 != UNCLEAR else MISDIRECTED
    n = len(contract.cases)
    fidelity = (1 - off / needed) if needed else None
    hold = (1 - off / n) if n else 0.0

    keep_ok = keep_n = rev_ok = rev_n = 0
    broken = []
    for d in contract.distinctions():
        la, lb = current.get(d.a, UNCLEAR), current.get(d.b, UNCLEAR)
        if UNCLEAR in (la, lb):
            ok = False
        else:
            ok = (la != lb) == d.distinct_after
        if d.must_persist:
            keep_n += 1
            keep_ok += ok
        else:
            rev_n += 1
            rev_ok += ok
        if not ok:
            broken.append((d.a, d.b, d.fate))
    return TransitionOutcome(
        contract_id=contract.id, delivery=delivery,
        previous={c.id: previous.get(c.id, UNCLEAR) for c in contract.cases},
        current={c.id: current.get(c.id, UNCLEAR) for c in contract.cases},
        target=target, status=status, needed_to_move=needed, off_target_after=off,
        fidelity=fidelity, hold=hold,
        persistence=(keep_ok / keep_n) if keep_n else None,
        revision=(rev_ok / rev_n) if rev_n else None,
        n_must_persist=keep_n, n_must_revise=rev_n, broken_distinctions=broken,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Probing a system
# ─────────────────────────────────────────────────────────────────────────────

ChatFn = Callable[[list], str]          # [{"role", "content"}, ...] -> assistant text
Classifier = Callable[[str, str, dict], str]

UPDATE_NOTICE = ("Update: the governing policy has changed. The policy below replaces the earlier one "
                 "in full, effective now.\n\n{after}\n\nWith that in mind: {question}")


def _modal(labels: list) -> str:
    if not labels:
        return UNCLEAR
    counts = Counter(labels).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return UNCLEAR
    return counts[0][0]


def run_transition(contract: TransitionContract, chat_fn: ChatFn, classifier: Classifier,
                   delivery: str = "fresh", samples: int = 1,
                   update_notice: str = UPDATE_NOTICE) -> TransitionOutcome:
    """
    Probe a system through one transition and score it.

    chat_fn(messages) -> assistant text, where messages is a list of
    {"role": "system" | "user" | "assistant", "content": str}.

    delivery
      "fresh"            the deployed system before and after the update:
                         one independent run under `before`, one under
                         `after`.
      "in_conversation"  one agent revising a commitment it made: it answers
                         under `before`; then, in the same conversation, it
                         is told the governing information has changed
                         (`update_notice`), given `after`, and asked again.
                         Its first answer is in its own context when it
                         gives the second.

    samples  repeat each case this many times and use the modal outcome
             (use an odd number).
    """
    if delivery not in ("fresh", "in_conversation"):
        raise ValueError("delivery must be 'fresh' or 'in_conversation'")
    errors = contract.lint_errors()
    if errors:
        raise ValueError("transition contract has errors:\n" + "\n".join(f"  {e[1]} {e[2]}" for e in errors))

    previous, current = {}, {}
    for case in contract.cases:
        outcomes = contract.outcomes_for(case)
        b0s, b1s = [], []
        for _ in range(max(1, samples)):
            first = [{"role": "system", "content": contract.system_text("before")},
                     {"role": "user", "content": case.question}]
            a0 = chat_fn(first)
            b0s.append(_label(classifier, case.question, a0, outcomes))
            if delivery == "fresh":
                second = [{"role": "system", "content": contract.system_text("after")},
                          {"role": "user", "content": case.question}]
            else:
                second = first + [
                    {"role": "assistant", "content": a0},
                    {"role": "user", "content": update_notice.format(after=contract.after.text,
                                                                      question=case.question)},
                ]
            a1 = chat_fn(second)
            b1s.append(_label(classifier, case.question, a1, outcomes))
        previous[case.id] = _modal(b0s)
        current[case.id] = _modal(b1s)
    return evaluate_transition(contract, previous, current, delivery=delivery)


def _label(classifier: Classifier, question: str, answer: str, outcomes: dict) -> str:
    try:
        label = classifier(question, answer, outcomes)
    except Exception:
        return UNCLEAR
    return label if label in outcomes else UNCLEAR
