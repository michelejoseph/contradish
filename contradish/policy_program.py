"""
contradish/policy_program.py -- a machine-readable policy that an evaluator
can execute, and the dependency model read off it.

A policy-grounded agent acts by calling tools. To say which of its actions a
policy change requires changing, the policy has to determine the actions.
A PolicyProgram does: it is the written policy split into clauses, each
clause owning the parameters and definitions it states, plus the procedure
(an ordered list of steps, each a tool call with a guard and arguments) that
the policy prescribes for a situation.

    facts     what describes a situation (days since delivery, category, ...)
    clauses   R1, R2, ...: prose text, and the names that clause defines:
                params   constants ("return_window_days": 30)
                define   derived quantities, as expressions
                steps    tool calls the clause prescribes
    steps     each:  tool, when (guard expression), args {name: expression}

Every name is owned by exactly one clause. That single rule is what makes
the dependency model exact: evaluating anything records which names it
read, and so which clauses it depended on, in THIS situation.

EXPRESSIONS (JSON)
    number / true / false / null      literal
    "name"                            a fact, param, or definition
    ["q", "text"]                     string literal
    [op, a, b, ...]                   op in: and or not if == != < <= > >=
                                      + - * / min max round in

`and`, `or` and `if` evaluate lazily, so what is recorded as read is what
the decision actually turned on: a dynamic slice, not every name mentioned.

AMENDMENTS
    An Edit changes one clause: new param values, new definitions of names
    that clause already owns, new step fields, new text. A clause cannot
    take over a name another clause owns. Edits are applied to a copy.

THE INDEPENDENCE LEMMA (what makes a "must not change" verdict a proof)
    Let P' differ from P only in the definitions of a set of names N.
    If evaluating step k in situation s under P reads no name whose
    definition differs in P' (and step k's own definition is unchanged),
    then evaluating step k in s under P' produces the identical result.
    (Stated per name, not per clause, so a clause whose fee rate changed
    does not make every reader of that clause's other definitions suspect.)
    Proof: evaluation is deterministic and reads only the definitions
    recorded in the trace. Each of those is unchanged in P', so P' takes
    the same branches, reads the same values, and returns the same result,
    by induction on the evaluation. (Checked by `tests/test_action_frontier.py`
    on every state of every built-in universe.)

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "PolicyProgram", "Edit", "ProgramUpdate", "Call", "Trace", "Norm", "norm_change",
    "OBLIGATORY", "PERMITTED", "FORBIDDEN",
    "load_program", "load_builtin_program", "list_builtin_programs",
    "canonical_json", "digest",
]

SCHEMA = "contradish.policy_program/1.0"
UPDATE_SCHEMA = "contradish.program_update/1.0"

_OPS = {"and", "or", "not", "if", "==", "!=", "<", "<=", ">", ">=",
        "+", "-", "*", "/", "min", "max", "round", "in", "q"}
_CMP = {"<", "<=", ">", ">="}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Calls
# ─────────────────────────────────────────────────────────────────────────────

AMOUNT_TOLERANCE = 0.005  # amounts are compared to the cent


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float, Fraction)) and not isinstance(x, bool)


def exact(x: Any) -> Any:
    """Numbers are evaluated exactly: a decimal literal becomes the rational it denotes."""
    if isinstance(x, bool) or not _is_num(x):
        return x
    if isinstance(x, Fraction):
        return x
    return Fraction(x) if isinstance(x, int) else Fraction(str(x))


def round_half_up(q: Any, places: int = 2) -> Fraction:
    """Exact rounding to `places` decimals, halves away from zero."""
    q = exact(q)
    scale = Fraction(10) ** places
    v = q * scale
    n = (abs(v) + Fraction(1, 2)).__floor__()
    return Fraction(n if v >= 0 else -n) / scale


def plain(q: Any) -> Any:
    """An exact number as JSON-friendly int or float."""
    if isinstance(q, Fraction):
        return int(q) if q.denominator == 1 else float(q)
    return q


def _val_eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if _is_num(a) and _is_num(b):
        return abs(exact(a) - exact(b)) <= Fraction(AMOUNT_TOLERANCE)
    return a == b


@dataclass(frozen=True)
class Call:
    """One tool call: the unit of agent behavior. None means 'not called'."""
    tool: str
    args: tuple = ()  # sorted (name, value) pairs

    @staticmethod
    def of(tool: str, args: Optional[dict] = None) -> "Call":
        return Call(tool, tuple(sorted((args or {}).items())))

    @staticmethod
    def from_dict(d: Optional[dict]) -> Optional["Call"]:
        if d is None:
            return None
        return Call.of(d["tool"], d.get("args") or {})

    def to_dict(self) -> dict:
        return {"tool": self.tool, "args": dict(self.args)}

    def same(self, other: Optional["Call"]) -> bool:
        return calls_equal(self, other)


def calls_equal(a: Optional[Call], b: Optional[Call]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a.tool != b.tool:
        return False
    da, db = dict(a.args), dict(b.args)
    if set(da) != set(db):
        return False
    return all(_val_eq(da[k], db[k]) for k in da)


def differing_args(a: Optional[Call], b: Optional[Call]) -> list:
    if a is None or b is None or a.tool != b.tool:
        return []
    da, db = dict(a.args), dict(b.args)
    return sorted(k for k in set(da) | set(db) if k not in da or k not in db or not _val_eq(da[k], db[k]))


# ─────────────────────────────────────────────────────────────────────────────
# Norms: obligation, permission, prohibition
# ─────────────────────────────────────────────────────────────────────────────

OBLIGATORY, PERMITTED, FORBIDDEN = "O", "P", "F"


@dataclass(frozen=True)
class Norm:
    """
    The deontic status of one step in one situation.

        O  obligatory  the agent must make exactly `call`
        P  permitted   the agent may make exactly `call`, or nothing
        F  forbidden   the agent must not call the step's tool

    A step is O where its `when` guard holds; otherwise P where its
    `allowed_when` guard holds; otherwise F. (`allowed_when` defaults to
    false, so a step without it is either required or forbidden.)
    """
    modality: str
    call: Optional[Call] = None

    def permits(self, observed: Optional[Call]) -> bool:
        if self.modality == OBLIGATORY:
            return calls_equal(observed, self.call)
        if self.modality == PERMITTED:
            return observed is None or calls_equal(observed, self.call)
        return observed is None

    @property
    def allowed(self) -> bool:
        return self.modality in (OBLIGATORY, PERMITTED)

    def same(self, other: "Norm") -> bool:
        if self.modality != other.modality:
            return False
        return self.modality == FORBIDDEN or calls_equal(self.call, other.call)

    def to_dict(self) -> dict:
        return {"modality": self.modality, "call": self.call.to_dict() if self.call else None}

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Norm":
        if d is None:
            return Norm(FORBIDDEN)
        return Norm(d["modality"], Call.from_dict(d.get("call")))

    @staticmethod
    def of(x: Any) -> "Norm":
        """Coerce: a Norm stays; a Call means obligatory; None means forbidden."""
        if isinstance(x, Norm):
            return x
        if x is None:
            return Norm(FORBIDDEN)
        return Norm(OBLIGATORY, x)


def norm_change(a: Norm, b: Norm) -> dict:
    """
    How a step's norm changed, on three separate channels:
        obligation  gained | lost | None
        permission  granted | revoked | None     (allowed = obligatory or permitted)
        content     changed | None               (allowed before and after, different call)
    """
    ob = (a.modality == OBLIGATORY, b.modality == OBLIGATORY)
    pe = (a.allowed, b.allowed)
    return {
        "obligation": "gained" if ob == (False, True) else "lost" if ob == (True, False) else None,
        "permission": "granted" if pe == (False, True) else "revoked" if pe == (True, False) else None,
        "content": "changed" if (a.allowed and b.allowed and not calls_equal(a.call, b.call)) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Edits and updates
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Edit:
    """A change to one clause."""
    clause: str
    params: dict = field(default_factory=dict)
    define: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)   # {step_id: {field: new value}}
    text: Optional[str] = None
    says: Optional[str] = None   # the sentence of the update's content that expresses this edit

    def to_dict(self) -> dict:
        d = {"clause": self.clause}
        if self.says is not None:
            d["says"] = self.says
        if self.params:
            d["params"] = dict(self.params)
        if self.define:
            d["define"] = dict(self.define)
        if self.steps:
            d["steps"] = copy.deepcopy(self.steps)
        if self.text is not None:
            d["text"] = self.text
        return d

    @staticmethod
    def from_dict(d: dict) -> "Edit":
        return Edit(d["clause"], dict(d.get("params") or {}), dict(d.get("define") or {}),
                    copy.deepcopy(d.get("steps") or {}), d.get("text"), d.get("says"))


@dataclass
class ProgramUpdate:
    """
    New information arriving: who it is from (`source`, a key of the
    program's sources), how it arrives (`channel`), what it says
    (`content`, the prose an agent actually reads) and what it would change
    if obeyed (`edits`). Which edits are WARRANTED is decided by authority.
    """
    id: str
    source: str
    channel: str
    content: str
    edits: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"schema": UPDATE_SCHEMA, "id": self.id, "source": self.source, "channel": self.channel,
                "content": self.content, "edits": [e.to_dict() for e in self.edits]}

    @staticmethod
    def from_dict(d: dict) -> "ProgramUpdate":
        return ProgramUpdate(d["id"], d["source"], d.get("channel", "system"), d.get("content", ""),
                             [Edit.from_dict(e) for e in d.get("edits", [])])


# ─────────────────────────────────────────────────────────────────────────────
# Trace
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    """What an evaluation read: names, and the clauses that own them."""
    names: set = field(default_factory=set)
    clauses: set = field(default_factory=set)
    facts: set = field(default_factory=set)


class ProgramError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# The program
# ─────────────────────────────────────────────────────────────────────────────

class PolicyProgram:
    def __init__(self, spec: dict):
        if spec.get("schema") not in (SCHEMA, None):
            raise ProgramError(f"unsupported schema {spec.get('schema')!r}")
        self.spec = copy.deepcopy(spec)
        self.spec["schema"] = SCHEMA
        self.id = spec["id"]
        self.facts: dict = self.spec.get("facts", {})
        self.clauses: dict = self.spec.get("clauses", {})
        self.sources: dict = self.spec.get("sources", {})
        self._index()

    # ----- structure ---------------------------------------------------------

    def _index(self) -> None:
        self.owner: dict = {}
        self.params: dict = {}
        self.defs: dict = {}
        self.steps: list = []          # ordered [(step_id, clause_id, step_dict)]
        self.tool_of: dict = {}
        for f in self.facts:
            self.owner[f] = None
        for cid, c in self.clauses.items():
            for kind in ("params", "define"):
                for name, v in (c.get(kind) or {}).items():
                    if name in self.owner:
                        raise ProgramError(f"name {name!r} is owned twice (clause {cid})")
                    self.owner[name] = cid
                    (self.params if kind == "params" else self.defs)[name] = v
        order = self.spec.get("procedure")
        all_steps = {}
        for cid, c in self.clauses.items():
            for sid, s in (c.get("steps") or {}).items():
                if sid in all_steps:
                    raise ProgramError(f"step {sid!r} defined twice")
                all_steps[sid] = (cid, s)
        order = order or sorted(all_steps)
        for sid in order:
            if sid not in all_steps:
                raise ProgramError(f"procedure names unknown step {sid!r}")
            cid, s = all_steps[sid]
            if s["tool"] in self.tool_of:
                raise ProgramError(f"tool {s['tool']!r} is used by two steps; each step needs its own tool")
            self.tool_of[s["tool"]] = sid
            self.steps.append((sid, cid, s))
        for name, expr in self.defs.items():
            self._check_expr(expr, f"definition {name}")
        for sid, cid, s in self.steps:
            self._check_expr(s.get("when", True), f"step {sid} guard")
            self._check_expr(s.get("allowed_when", False), f"step {sid} permission guard")
            for a, e in (s.get("args") or {}).items():
                self._check_expr(e, f"step {sid} arg {a}")
        self._check_acyclic()

    def _check_expr(self, e: Any, where: str) -> None:
        if isinstance(e, str):
            if e not in self.owner:
                raise ProgramError(f"{where}: unknown name {e!r}")
        elif isinstance(e, list):
            if not e or e[0] not in _OPS:
                raise ProgramError(f"{where}: bad expression {e!r}")
            if e[0] == "q":
                return
            for x in e[1:]:
                self._check_expr(x, where)

    def refs(self, e: Any) -> set:
        """Every name an expression mentions (all branches): the static model."""
        if isinstance(e, str):
            return {e}
        if isinstance(e, list) and e and e[0] != "q":
            out = set()
            for x in e[1:]:
                out |= self.refs(x)
            return out
        return set()

    def _check_acyclic(self) -> None:
        state: dict = {}

        def visit(n, path):
            if state.get(n) == 1:
                raise ProgramError("definitions are cyclic: " + " -> ".join(path + [n]))
            if state.get(n) == 2 or n not in self.defs:
                return
            state[n] = 1
            for m in self.refs(self.defs[n]):
                visit(m, path + [n])
            state[n] = 2

        for n in self.defs:
            visit(n, [])

    def step_ids(self) -> list:
        return [sid for sid, _, _ in self.steps]

    def step(self, sid: str) -> tuple:
        for s in self.steps:
            if s[0] == sid:
                return s
        raise KeyError(sid)

    # ----- the dependency model ------------------------------------------------

    def dependency_graph(self) -> dict:
        """
        Static dependency model. Nodes are clauses, names and steps:
            clause:R1 -> name:return_window_days       (owns)
            name:x    -> name:y                        (y's definition mentions x)
            name:x    -> step:refund                   (guard or args mention x)
            clause:P1 -> step:refund                   (the step is defined by P1)
        Edges point downstream, in the direction a change propagates.
        """
        edges = set()
        for n, cid in self.owner.items():
            if cid is not None:
                edges.add((f"clause:{cid}", f"name:{n}"))
        for n, e in self.defs.items():
            for m in self.refs(e):
                edges.add((f"name:{m}", f"name:{n}"))
        for sid, cid, s in self.steps:
            edges.add((f"clause:{cid}", f"step:{sid}"))
            mentioned = self.refs(s.get("when", True)) | self.refs(s.get("allowed_when", False))
            for e in (s.get("args") or {}).values():
                mentioned |= self.refs(e)
            for m in mentioned:
                edges.add((f"name:{m}", f"step:{sid}"))
        nodes = sorted({a for a, _ in edges} | {b for _, b in edges})
        return {"nodes": nodes, "edges": sorted(edges)}

    def changed_names(self, other: "PolicyProgram") -> set:
        """Names (and `step:<id>`) whose definition differs between two versions of a program."""
        out = set()
        for n in set(self.params) | set(other.params):
            if canonical_json(self.params.get(n)) != canonical_json(other.params.get(n)):
                out.add(n)
        for n in set(self.defs) | set(other.defs):
            if canonical_json(self.defs.get(n)) != canonical_json(other.defs.get(n)):
                out.add(n)
        mine = {sid: s for sid, _, s in self.steps}
        theirs = {sid: s for sid, _, s in other.steps}
        for sid in set(mine) | set(theirs):
            if canonical_json(mine.get(sid)) != canonical_json(theirs.get(sid)):
                out.add("step:" + sid)
        return out

    def downstream_of_names(self, names: set) -> dict:
        """Steps statically reachable from changed names, each with one shortest path."""
        starts = [(f"step:{n[5:]}" if n.startswith("step:") else f"name:{n}") for n in sorted(names)]
        return self._reach(starts)

    def downstream(self, clauses: set) -> dict:
        """
        Steps statically reachable from the given clauses, each with one
        shortest dependency path. Reachable means MAY change; whether it
        MUST change in a given situation is decided by evaluation.
        """
        return self._reach([f"clause:{c}" for c in sorted(clauses)])

    def _reach(self, starts: list) -> dict:
        g = self.dependency_graph()
        adj: dict = {}
        for a, b in g["edges"]:
            adj.setdefault(a, []).append(b)
        out = {}
        for n in starts:
            if n.startswith("step:"):
                out[n[5:]] = [n]
        frontier = [(n, [n]) for n in starts]
        seen = {n for n, _ in frontier}
        while frontier:
            nxt = []
            for node, path in frontier:
                for m in sorted(adj.get(node, [])):
                    if m in seen:
                        continue
                    seen.add(m)
                    p = path + [m]
                    if m.startswith("step:"):
                        out[m[5:]] = p
                    nxt.append((m, p))
            frontier = nxt
        return out

    # ----- evaluation ----------------------------------------------------------

    def complete_state(self, state: dict) -> dict:
        s = {}
        for f, spec in self.facts.items():
            if f in state:
                s[f] = state[f]
            elif "default" in spec:
                s[f] = spec["default"]
            else:
                raise ProgramError(f"state is missing fact {f!r} and it has no default")
        extra = set(state) - set(self.facts)
        if extra:
            raise ProgramError(f"unknown facts in state: {sorted(extra)}")
        return s

    def eval(self, e: Any, state: dict, trace: Optional[Trace] = None, _memo: Optional[dict] = None) -> Any:
        memo = {} if _memo is None else _memo
        t = trace if trace is not None else Trace()

        def ev(x):
            if isinstance(x, bool) or x is None:
                return x
            if _is_num(x):
                return exact(x)
            if isinstance(x, str):
                t.names.add(x)
                cid = self.owner.get(x)
                if cid is None:
                    t.facts.add(x)
                    return exact(state[x])
                t.clauses.add(cid)
                if x in self.params:
                    return exact(self.params[x])
                if x not in memo:
                    memo[x] = ev(self.defs[x])
                else:
                    # re-read: record what that definition read, again
                    self._reread(x, state, t)
                return memo[x]
            op, args = x[0], x[1:]
            if op == "q":
                return args[0]
            if op == "and":
                for a in args:
                    if not ev(a):
                        return False
                return True
            if op == "or":
                for a in args:
                    if ev(a):
                        return True
                return False
            if op == "not":
                return not ev(args[0])
            if op == "if":
                return ev(args[1]) if ev(args[0]) else ev(args[2])
            vals = [ev(a) for a in args]
            if op == "==":
                return _val_eq(vals[0], vals[1])
            if op == "!=":
                return not _val_eq(vals[0], vals[1])
            if op == "<":
                return vals[0] < vals[1]
            if op == "<=":
                return vals[0] <= vals[1]
            if op == ">":
                return vals[0] > vals[1]
            if op == ">=":
                return vals[0] >= vals[1]
            if op == "+":
                return sum(vals)
            if op == "-":
                return vals[0] - sum(vals[1:]) if len(vals) >= 2 else -vals[0]
            if op == "*":
                out = 1
                for v in vals:
                    out *= v
                return out
            if op == "/":
                return vals[0] / vals[1]
            if op == "min":
                return min(vals)
            if op == "max":
                return max(vals)
            if op == "round":
                return round_half_up(vals[0], int(vals[1]) if len(vals) > 1 else 0)
            if op == "in":
                return vals[0] in vals[1:]
            raise ProgramError(f"unknown op {op!r}")

        return ev(e)

    def _reread(self, name: str, state: dict, t: Trace) -> None:
        sub = Trace()
        self.eval(self.defs[name], state, sub)
        t.names |= sub.names
        t.clauses |= sub.clauses
        t.facts |= sub.facts

    def run_step(self, sid: str, state: dict) -> tuple:
        """(Call or None, Trace) for one step in one situation."""
        _, cid, s = self.step(sid)
        t = Trace(names={"step:" + sid}, clauses={cid})
        memo: dict = {}
        if not self.eval(s.get("when", True), state, t, memo):
            return None, t
        args = {a: self.eval(e, state, t, memo) for a, e in sorted((s.get("args") or {}).items())}
        args = {a: (plain(round_half_up(v, 2)) if _is_num(v) else v) for a, v in args.items()}
        return Call.of(s["tool"], args), t

    def run_norm(self, sid: str, state: dict) -> tuple:
        """(Norm, Trace) for one step in one situation: obligatory, permitted or forbidden."""
        _, cid, s = self.step(sid)
        t = Trace(names={"step:" + sid}, clauses={cid})
        memo: dict = {}
        if self.eval(s.get("when", True), state, t, memo):
            modality = OBLIGATORY
        elif "allowed_when" in s and self.eval(s["allowed_when"], state, t, memo):
            modality = PERMITTED
        else:
            return Norm(FORBIDDEN), t
        args = {a: self.eval(e, state, t, memo) for a, e in sorted((s.get("args") or {}).items())}
        args = {a: (plain(round_half_up(v, 2)) if _is_num(v) else v) for a, v in args.items()}
        return Norm(modality, Call.of(s["tool"], args)), t

    def norms(self, state: dict) -> dict:
        st = self.complete_state(state)
        return {sid: self.run_norm(sid, st)[0] for sid in self.step_ids()}

    def run(self, state: dict) -> dict:
        """The obligated trajectory: {step_id: Call or None} (permitted steps are not taken)."""
        st = self.complete_state(state)
        return {sid: self.run_step(sid, st)[0] for sid in self.step_ids()}

    # ----- amendment -----------------------------------------------------------

    def apply(self, edits: list) -> "PolicyProgram":
        spec = copy.deepcopy(self.spec)
        for e in edits:
            if e.clause not in spec["clauses"]:
                raise ProgramError(f"edit targets unknown clause {e.clause!r}")
            c = spec["clauses"][e.clause]
            for name, v in e.params.items():
                if name not in (c.get("params") or {}):
                    raise ProgramError(f"clause {e.clause} does not own param {name!r}")
                c["params"][name] = v
            for name, v in e.define.items():
                if name not in (c.get("define") or {}):
                    raise ProgramError(f"clause {e.clause} does not own definition {name!r}")
                c["define"][name] = v
            for sid, fields in e.steps.items():
                if sid not in (c.get("steps") or {}):
                    raise ProgramError(f"clause {e.clause} does not own step {sid!r}")
                c["steps"][sid].update(copy.deepcopy(fields))
            if e.text is not None:
                c["text"] = e.text
        return PolicyProgram(spec)

    def has_authority(self, source: str, clause: str) -> bool:
        if source not in self.sources:
            raise ProgramError(f"unknown source {source!r}")
        gov = self.sources[source].get("governs", [])
        return "*" in gov or clause in gov

    def split_edits(self, update: ProgramUpdate) -> tuple:
        """(authorized edits, unauthorized edits) for an update."""
        ok, bad = [], []
        for e in update.edits:
            (ok if self.has_authority(update.source, e.clause) else bad).append(e)
        return ok, bad

    # ----- prose ---------------------------------------------------------------

    def render(self) -> str:
        """The policy as an agent reads it: each clause's text, params filled in."""
        lines = [self.spec.get("title", self.id)]
        for cid, c in self.clauses.items():
            text = c.get("text", "")
            for k, v in (c.get("params") or {}).items():
                text = text.replace("{" + k + "}", _fmt(v))
            lines.append(f"{cid}. {text}")
        return "\n".join(lines)

    def tools(self) -> list:
        out = []
        for sid, cid, s in self.steps:
            out.append({"tool": s["tool"], "args": sorted((s.get("args") or {}).keys()),
                        "description": s.get("description", "")})
        return out

    def to_dict(self) -> dict:
        return copy.deepcopy(self.spec)

    def digest(self) -> str:
        return digest(self.spec)


def _fmt(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(__file__)
_BUILTIN = os.path.join(_HERE, "contracts", "programs")


def load_program(path: str) -> PolicyProgram:
    with open(path) as f:
        return PolicyProgram(json.load(f))


def list_builtin_programs() -> list:
    if not os.path.isdir(_BUILTIN):
        return []
    return sorted(n[:-5] for n in os.listdir(_BUILTIN) if n.endswith(".json"))


def load_builtin_program(name: str = "returns") -> tuple:
    """(PolicyProgram, {update_id: ProgramUpdate}) for a built-in program."""
    with open(os.path.join(_BUILTIN, name + ".json")) as f:
        spec = json.load(f)
    updates = {u["id"]: ProgramUpdate.from_dict(u) for u in spec.pop("updates", [])}
    return PolicyProgram(spec), updates
