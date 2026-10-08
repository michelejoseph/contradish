"""
transition_sequence.py -- Behavioral Update Fidelity over a HISTORY of updates.

A single transition asks whether a system moved correctly once. Real systems
live through sequences: the policy is amended, then amended back; a user
corrects themselves; a tool result contradicts an earlier document. The
warranted behavioral state is a FUNCTION OF THE INFORMATION HISTORY, and a
faithful system's behavior must be a function of that history too -- not of
the order it happened to arrive in, not of detours the history took, and not
of how the same information was packaged.

    SequenceContract   an initial state, the sources, and an ordered list of
                       Updates. Authority is decided per update, per case
                       (the same rule as derive_transition), so the warranted
                       trajectory beta*_0 ... beta*_k is derived, not authored.

    evaluate_sequence  scores an observed trajectory beta_0 ... beta_k:
        stages            one TransitionOutcome per update (did the system
                          move correctly at each step)
        cumulative        beta_0 -> beta_k against the net warranted change
        hysteresis        cases the system gets right when handed the NET
                          information afresh but wrong after living through
                          the history: the system's present behavior depends
                          on its past in a way the information does not
        path_dependent    cases where fresh and sequential delivery disagree
                          at all (right or wrong)
        round_trip        when the history returns the warranted state to
                          where it began, did the behavior come home too

    commuting_orders / order_independence
                       reorderings of the history that the warranted
                       semantics says reach the same final state, and whether
                       the system's behavior is invariant across them.

Hysteresis is defined relative to a baseline that delivers the same net
information at once (`compound_update`). That baseline is exact for oracle
agents and approximate for natural-language runs: the compound text is
authored from the contents of the updates that carried authority, so a real
measurement is only as clean as that text. See docs/BUF-SPEC.md section 7.
"""
from dataclasses import dataclass, field
from itertools import permutations
from typing import Callable, Optional

from .transition import (
    CHANNEL_TEMPLATES, GoverningState, Source, TransitionCase, TransitionContract,
    TransitionOutcome, Update, UNCLEAR, _label, _modal, derive_transition, evaluate_transition,
)

NET_SOURCE = "__net__"


@dataclass
class SequenceContract:
    id: str
    initial: GoverningState
    cases: list                      # TransitionCase; `before` is the warranted initial outcome
    sources: dict                    # {id: Source}
    updates: list                    # [Update], in order of arrival
    outcomes: dict = field(default_factory=dict)
    domain: str = ""
    description: str = ""

    # ── the warranted semantics ─────────────────────────────────────────────

    @property
    def grounds(self) -> dict:
        return {c.id: list(c.grounds) for c in self.cases}

    def state0(self) -> dict:
        return {c.id: c.before for c in self.cases}

    def footprint(self, j: int) -> dict:
        """{case_id: outcome} update j (1-based) is warranted to write."""
        u, src = self.updates[j - 1], self.sources[self.updates[j - 1].source]
        gr = self.grounds
        return {cid: lab for cid, lab in u.asserts.items()
                if cid in gr and src.has_authority_over(gr[cid])}

    def trajectory(self) -> list:
        """[beta*_0, ..., beta*_k]: the warranted state after each prefix of the history."""
        states = [self.state0()]
        for j in range(1, len(self.updates) + 1):
            nxt = dict(states[-1])
            nxt.update(self.footprint(j))
            states.append(nxt)
        return states

    @property
    def final(self) -> dict:
        return self.trajectory()[-1]

    def step_contract(self, j: int) -> TransitionContract:
        """The transition contract for update j (1-based), derived from the warranted state before it."""
        prev = self.trajectory()[j - 1]
        cases = [TransitionCase(c.id, c.question, prev[c.id], prev[c.id], list(c.variants), c.outcomes,
                                list(c.grounds)) for c in self.cases]
        before = self.initial if j == 1 else GoverningState(f"{self.id}@{j - 1}", self.initial.text)
        return derive_transition(f"{self.id}#{j}", before, cases, self.updates[j - 1], self.sources,
                                 self.outcomes, domain=self.domain, description=self.description)

    def compound_update(self) -> Update:
        """The whole history's NET warranted change, delivered as one update from an all-governing source."""
        net = {cid: lab for cid, lab in self.final.items() if lab != self.state0()[cid]}
        said = [self.updates[j - 1].content for j in range(1, len(self.updates) + 1) if self.footprint(j)]
        return Update(id=f"{self.id}:net", source=NET_SOURCE, channel="system",
                      content=" ".join(said), asserts=net)

    def final_contract(self) -> TransitionContract:
        """initial -> net: the single transition equivalent to the whole (authorized) history."""
        srcs = dict(self.sources)
        srcs[NET_SOURCE] = Source(NET_SOURCE, "net", ["*"])
        cases = [TransitionCase(c.id, c.question, c.before, c.before, list(c.variants), c.outcomes,
                                list(c.grounds)) for c in self.cases]
        return derive_transition(f"{self.id}#net", self.initial, cases, self.compound_update(), srcs,
                                 self.outcomes, domain=self.domain, description=self.description)

    def reordered(self, order: tuple) -> "SequenceContract":
        return SequenceContract(f"{self.id}~{''.join(map(str, order))}", self.initial, self.cases, self.sources,
                                [self.updates[i] for i in order], self.outcomes, self.domain, self.description)

    # ── (de)serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"id": self.id, "domain": self.domain, "description": self.description,
                "initial": self.initial.to_dict(), "outcomes": dict(self.outcomes),
                "cases": [c.to_dict() for c in self.cases],
                "sources": {k: v.to_dict() for k, v in self.sources.items()},
                "updates": [u.to_dict() for u in self.updates]}

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceContract":
        cases = [TransitionCase(c["id"], c["question"], c["before"], c.get("after", c["before"]),
                                list(c.get("variants", [])), c.get("outcomes"), list(c.get("grounds", [])))
                 for c in d["cases"]]
        sources = {k: Source(v.get("id", k), v.get("role", ""), list(v.get("governs", [])))
                   for k, v in d["sources"].items()}
        updates = [Update(u["id"], u["source"], u["channel"], u["content"], dict(u.get("asserts", {})))
                   for u in d["updates"]]
        return cls(d["id"], GoverningState(d["initial"]["id"], d["initial"]["text"]), cases, sources, updates,
                   dict(d.get("outcomes", {})), d.get("domain", ""), d.get("description", ""))


# ── evaluating an observed trajectory ───────────────────────────────────────

@dataclass
class SequenceOutcome:
    contract_id: str
    observed: list                       # [beta_0 ... beta_k]
    warranted: list                      # [beta*_0 ... beta*_k]
    stages: list                         # [TransitionOutcome] per update
    cumulative: Optional[TransitionOutcome]
    tracking: Optional[float]            # share of stages that landed exactly on the warranted state
    final_faithful: bool
    fresh_final: Optional[dict] = None
    hysteresis: Optional[list] = None
    path_dependent: Optional[list] = None
    round_trip_applicable: bool = False
    round_trip_returned: Optional[bool] = None

    @property
    def hysteresis_rate(self) -> Optional[float]:
        if self.hysteresis is None or not self.observed:
            return None
        return len(self.hysteresis) / len(self.observed[0]) if self.observed[0] else None

    def to_dict(self) -> dict:
        return {"contract_id": self.contract_id, "tracking": self.tracking, "final_faithful": self.final_faithful,
                "hysteresis": self.hysteresis, "hysteresis_rate": self.hysteresis_rate,
                "path_dependent": self.path_dependent, "round_trip_applicable": self.round_trip_applicable,
                "round_trip_returned": self.round_trip_returned,
                "cumulative": self.cumulative.to_dict() if self.cumulative else None,
                "stages": [s.to_dict() for s in self.stages],
                "observed": self.observed, "warranted": self.warranted}

    def summary(self) -> str:
        bits = [f"{self.contract_id}", f"stages faithful {self.tracking if self.tracking is None else f'{self.tracking:.0%}'}",
                f"final {'faithful' if self.final_faithful else 'OFF TARGET'}"]
        if self.hysteresis is not None:
            bits.append(f"hysteresis {self.hysteresis or 'none'}")
        if self.round_trip_applicable:
            bits.append("round trip " + ("home" if self.round_trip_returned else "NOT HOME"))
        return "  ".join(bits)


def evaluate_sequence(seq: SequenceContract, observed: list, fresh_final: Optional[dict] = None) -> SequenceOutcome:
    """
    Score an observed trajectory. Pure: no model or judge calls.

    observed     [beta_0, ..., beta_k]  the system's behavior {case: label} after each prefix of the history
    fresh_final  the system's behavior when handed the net information at once (the hysteresis baseline)
    """
    k = len(seq.updates)
    if len(observed) != k + 1:
        raise ValueError(f"observed must have {k + 1} states (initial + one per update), got {len(observed)}")
    warranted = seq.trajectory()
    stages = [evaluate_transition(seq.step_contract(j), observed[j - 1], observed[j]) for j in range(1, k + 1)]
    cumulative = evaluate_transition(seq.final_contract(), observed[0], observed[k]) if k else None
    final = warranted[-1]
    got = {c: observed[k].get(c, UNCLEAR) for c in final}
    hyst = path = None
    if fresh_final is not None:
        fresh = {c: fresh_final.get(c, UNCLEAR) for c in final}
        hyst = sorted(c for c in final if fresh[c] == final[c] and got[c] != final[c])
        path = sorted(c for c in final if fresh[c] != got[c])
    start = warranted[0]
    moved = any(w != start for w in warranted[1:])
    applicable = k >= 2 and moved and final == start
    return SequenceOutcome(
        contract_id=seq.id, observed=[dict(o) for o in observed], warranted=warranted, stages=stages,
        cumulative=cumulative, tracking=(sum(s.faithful for s in stages) / k) if k else None,
        final_faithful=got == final, fresh_final=fresh_final, hysteresis=hyst, path_dependent=path,
        round_trip_applicable=applicable,
        round_trip_returned=(all(got[c] == start[c] for c in start) if applicable else None),
    )


def commuting_orders(seq: SequenceContract, limit: int = 120) -> list:
    """Non-identity reorderings of the history that the warranted semantics says reach the SAME final state."""
    k = len(seq.updates)
    base, out = seq.final, []
    for order in permutations(range(k)):
        if order == tuple(range(k)):
            continue
        if seq.reordered(order).final == base:
            out.append(order)
            if len(out) >= limit:
                break
    return out


def order_independence(seq: SequenceContract, finals_by_order: dict) -> dict:
    """
    finals_by_order  {order tuple: the system's final {case: label} after receiving the history in that order}
    Returns {"orders": n, "faithful": n_on_target, "invariant": all orders agree with each other and the target}.
    """
    target = seq.final
    ok = sum(all(f.get(c, UNCLEAR) == target[c] for c in target) for f in finals_by_order.values())
    return {"orders": len(finals_by_order), "faithful": ok, "invariant": ok == len(finals_by_order)}


# ── probing a language-model system ─────────────────────────────────────────

ChatFn = Callable[[list], str]
Classifier = Callable[[str, str, dict], str]


def _delivery_message(update: Update) -> dict:
    text = CHANNEL_TEMPLATES[update.channel].format(source=update.source, content=update.content, question="").strip()
    role = "system" if update.channel in ("system", "memory") else "user"
    return {"role": role, "content": text}


def run_sequence(seq: SequenceContract, chat_fn: ChatFn, classifier: Classifier,
                 samples: int = 1, fresh_baseline: bool = True) -> SequenceOutcome:
    """
    Live the history with a system and score it. Updates are delivered in order, in one running
    conversation (each through its own channel); after each, every case's question is asked in a fork
    of that conversation. The hysteresis baseline asks the same questions after a single system-channel
    delivery of the net information.
    """
    sys_msg = {"role": "system", "content": seq.initial.text}

    def ask(history: list) -> dict:
        out = {}
        for c in seq.cases:
            outs = seq.outcomes if c.outcomes is None else {o: o for o in c.outcomes}
            labels = [_label(classifier, c.question, chat_fn([sys_msg] + history + [{"role": "user", "content": c.question}]),
                             outs) for _ in range(max(1, samples))]
            out[c.id] = _modal(labels)
        return out

    history, observed = [], [ask([])]
    for u in seq.updates:
        history = history + [_delivery_message(u)]
        if u.channel not in ("system", "memory"):
            history = history + [{"role": "assistant", "content": "Noted."}]
        observed.append(ask(history))
    fresh = ask([_delivery_message(seq.compound_update())]) if fresh_baseline else None
    return evaluate_sequence(seq, observed, fresh)
