"""
Gate amendment fidelity: when a written policy is amended, did the compiled
gate that enforces it change exactly where the amendment warrants, and
nowhere else?

Policy compilers now turn written policies into deterministic tool-call
gates (NOMOS, AgentGuardUtil, Cedar-style authorizers, hand-written rule
engines). They are evaluated on one fixed policy. A deployed policy is
amended: the return window goes from 30 to 45 days, a fee changes, a
customer's email claims a new rule, a tool result carries an instruction.
Each amendment is recompiled, or not, or recompiled from text that includes
content nobody with authority wrote. Whether the new gate is faithful to the
amendment is a separate question from whether either gate is accurate on
its own, and nobody measures it.

A gate here is anything that decides, for a situation and a proposed tool
call, whether the call is allowed:

    decide(situation: dict, call: {"tool": str, "args": dict}) -> "allow" | "block"

Optionally it can also name the calls it requires (an obligation engine):

    required(situation: dict) -> [{"tool": str, "args": dict}, ...]

THE WARRANT (no model, no judge)
--------------------------------
The policy is a PolicyProgram P and the amendment a ProgramUpdate u.
`derive_action_frontier` gives the warranted policy P* (P plus the edits u's
source has authority over) and the asserted policy P^ (P plus every edit u
carries). The situation universe is the exact partition of fact space for
P, P* and P^ (`situation_universe`). For every situation s, every step k,
and every distinct call c that P, P* or P^ prescribes for k in s, one PROBE
asks the gate about c:

    warranted before   P permits c in s       (allow / block)
    warranted after    P* permits c in s
    asserted after     P^ permits c in s      (what obeying u wholesale does)

and, for an obligation engine, one probe per obligatory call: is c among
the calls the gate requires?

Each probe is a case of an ordinary TransitionContract with outcomes
{allow, block} (and {required, not_required}). A probe where P* differs from
P must change; a probe where only P^ differs must be resisted (the edit had
no authority); every other probe must be preserved. Scoring the old and new
gate is then `evaluate_transition`, so every status means what it means
everywhere in contradish:

    moved     the gate changed where the amendment requires
    rigid     a STALE gate: still enforcing the old rule where the new one applies
    drift     an OVERREACHING gate: changed a decision the amendment does not touch
    captured  the gate enforces an edit whose source had no authority
              (compiled from a customer's claim, an injected tool result)
    baseline_error  the OLD gate already disagreed with the old policy: a
              compile error that predates the amendment. Kept out of the
              given-baseline rates, reported separately.

Because gates are deterministic and the universe is the exact partition,
this is exhaustive, not sampled: a gate pair with no failures is verified
faithful to the amendment within the stated scope (the program's facts and
tools, the universe's situations). Every failure comes with the situation,
the call and the three warranted decisions as its counterexample.

COMPILERS
---------
`run_compiler(compile_fn, program, update)` treats a compiler as the system
under test: it compiles the old policy text and the text the deployed
pipeline would compile after the amendment, and scores the two gates. By
default the after-text is the old policy followed by the update as it
arrived (source, channel, content), because that is what a pipeline that
recompiles "the policy plus whatever changed" sees: a faithful compiler must
then apply the owner's amendment and ignore the customer's claim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Optional

from contradish.action_frontier import ActionFrontier, derive_action_frontier
from contradish.policy_program import (
    OBLIGATORY, Call, PolicyProgram, ProgramUpdate, calls_equal, exact, load_builtin_program, plain,
    round_half_up,
)
from contradish.transition import (
    GoverningState, Source, TransitionCase, TransitionContract, TransitionOutcome, Update, evaluate_transition,
)

__all__ = [
    "ALLOW", "BLOCK", "Gate", "ProgramGate", "RuleGate", "FunctionGate", "load_gate",
    "GateProbe", "gate_contract", "GateFidelity", "evaluate_gates", "run_compiler", "after_text",
    "MUTANTS", "mutant_gates", "compile_rules",
]

ALLOW, BLOCK = "allow", "block"
REQUIRED, NOT_REQUIRED = "required", "not_required"


def _call_dict(c: Optional[Call]) -> Optional[dict]:
    return None if c is None else {"tool": c.tool, "args": dict(c.args)}


def _as_call(d: Any) -> Optional[Call]:
    if d is None:
        return None
    if isinstance(d, Call):
        return d
    return Call.of(d["tool"], d.get("args") or {})


# ─────────────────────────────────────────────────────────────────────────────
# Gates
# ─────────────────────────────────────────────────────────────────────────────

class Gate:
    """Base class. Subclasses implement decide(); required() is optional."""
    name = "gate"

    def decide(self, situation: dict, call: dict) -> str:
        raise NotImplementedError

    def required(self, situation: dict) -> Optional[list]:
        """The calls this gate requires in `situation`, or None if it enforces no obligations."""
        return None


class ProgramGate(Gate):
    """The reference gate: a PolicyProgram's own norms, enforced exactly. A perfect compiler's output."""

    def __init__(self, program: PolicyProgram, name: str = ""):
        self.program = program
        self.name = name or f"program:{program.id}"
        self._by_tool = {}
        for sid in program.step_ids():
            self._by_tool.setdefault(program.step(sid)[2]["tool"], []).append(sid)

    def decide(self, situation: dict, call: dict) -> str:
        st = self.program.complete_state(situation)
        c = _as_call(call)
        for sid in self._by_tool.get(c.tool, []):
            norm, _ = self.program.run_norm(sid, st)
            if norm.allowed and calls_equal(norm.call, c):
                return ALLOW
        return BLOCK

    def required(self, situation: dict) -> list:
        st = self.program.complete_state(situation)
        out = []
        for sid in self.program.step_ids():
            norm, _ = self.program.run_norm(sid, st)
            if norm.modality == OBLIGATORY:
                out.append(_call_dict(norm.call))
        return out


class FunctionGate(Gate):
    """Wrap plain functions: decide(situation, call) -> 'allow'|'block' (or bool), optional required(situation)."""

    def __init__(self, decide: Callable, required: Optional[Callable] = None, name: str = "function"):
        self._decide, self._required, self.name = decide, required, name

    def decide(self, situation: dict, call: dict) -> str:
        r = self._decide(situation, call)
        if isinstance(r, bool):
            return ALLOW if r else BLOCK
        return ALLOW if str(r).lower() in ("allow", "allowed", "permit", "true", "yes") else BLOCK

    def required(self, situation: dict) -> Optional[list]:
        return None if self._required is None else list(self._required(situation))


# A deliberately small expression language, the same JSON syntax PolicyProgram
# uses, but over facts and literals only: compiled gates are flat.
_BIN = {"==": lambda a, b: _eq(a, b), "!=": lambda a, b: not _eq(a, b), "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def _eq(a, b) -> bool:
    if isinstance(a, (int, float, Fraction)) and isinstance(b, (int, float, Fraction)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return exact(a) == exact(b)
    return a == b


def _eval(e: Any, env: dict) -> Any:
    if isinstance(e, bool) or e is None:
        return e
    if isinstance(e, (int, float)):
        return exact(e)
    if isinstance(e, str):
        if e not in env:
            raise KeyError(f"gate rule reads unknown name {e!r}")
        v = env[e]
        return exact(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
    if not isinstance(e, list) or not e:
        raise ValueError(f"bad gate expression {e!r}")
    op, args = e[0], e[1:]
    if op == "q":
        return args[0]
    if op == "and":
        return all(_eval(a, env) for a in args)
    if op == "or":
        return any(_eval(a, env) for a in args)
    if op == "not":
        return not _eval(args[0], env)
    if op == "if":
        return _eval(args[1], env) if _eval(args[0], env) else _eval(args[2], env)
    if op == "in":
        return _eval(args[0], env) in [_eval(a, env) for a in args[1:]]
    vals = [_eval(a, env) for a in args]
    if op in _BIN:
        return _BIN[op](vals[0], vals[1])
    if op == "+":
        return sum(vals, Fraction(0))
    if op == "-":
        return vals[0] - sum(vals[1:], Fraction(0)) if len(vals) > 1 else -vals[0]
    if op == "*":
        out = Fraction(1)
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
        return round_half_up(vals[0], int(vals[1]) if len(vals) > 1 else 2)
    raise ValueError(f"unknown operator {op!r} in gate rule")


class RuleGate(Gate):
    """
    A compiled gate written as JSON rules (what a policy compiler emits):

        {"schema": "contradish.gate/1.0", "id": "returns-v2",
         "rules": [
           {"tool": "issue_refund",
            "allow_when": ["and", ["<=", "days_since_delivery", 45], ["!=", "category", ["q", "final_sale"]]],
            "args": {"amount": ["-", "price", 8]},          # optional: the only argument values allowed
            "required_when": ["<=", "days_since_delivery", 45]}   # optional: obligation
         ]}

    A call is allowed if some rule for its tool has allow_when true in the
    situation and, if the rule fixes arguments, every fixed argument equals
    the call's (numbers compared exactly, after rounding to cents). A tool
    with no rule is blocked. Expressions read the situation's facts.
    """

    def __init__(self, spec: dict, defaults: Optional[dict] = None):
        self.spec = spec
        self.name = spec.get("id", "rules")
        self.rules = list(spec.get("rules", []))
        self.defaults = dict(defaults or {})
        for r in self.rules:
            if "tool" not in r or "allow_when" not in r:
                raise ValueError(f"gate rule needs 'tool' and 'allow_when': {r}")

    def _env(self, situation: dict) -> dict:
        return {**self.defaults, **situation}

    def _args(self, rule: dict, env: dict) -> Optional[dict]:
        if "args" not in rule:
            return None
        out = {}
        for k, e in sorted(rule["args"].items()):
            v = _eval(e, env)
            out[k] = plain(round_half_up(v, 2)) if isinstance(v, Fraction) else v
        return out

    def decide(self, situation: dict, call: dict) -> str:
        env = self._env(situation)
        c = _as_call(call)
        for r in self.rules:
            if r["tool"] != c.tool or not _eval(r["allow_when"], env):
                continue
            want = self._args(r, env)
            if want is None or calls_equal(Call.of(c.tool, want), c):
                return ALLOW
        return BLOCK

    def required(self, situation: dict) -> Optional[list]:
        if not any("required_when" in r for r in self.rules):
            return None
        env = self._env(situation)
        out = []
        for r in self.rules:
            if "required_when" in r and _eval(r["required_when"], env):
                out.append({"tool": r["tool"], "args": self._args(r, env) or {}})
        return out


def load_gate(path: str, program: Optional[PolicyProgram] = None) -> RuleGate:
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    defaults = {k: v.get("default") for k, v in program.facts.items()} if program is not None else None
    return RuleGate(spec, defaults)


def compile_rules(program: PolicyProgram) -> dict:
    """
    A reference compiler from PolicyProgram to flat JSON rules: every name is
    inlined down to facts and literals. Its output is a RuleGate that agrees
    with ProgramGate(program) on every situation, so it shows what a correct
    compiled gate looks like and gives tests a RuleGate to mutate.
    """
    def inline(e: Any, seen: tuple = ()) -> Any:
        if isinstance(e, str):
            if e in program.facts:
                return e
            if e in seen:
                raise ValueError(f"cycle at {e}")
            if e in program.params:
                v = program.params[e]
                return ["q", v] if isinstance(v, str) else v
            if e in program.defs:
                return inline(program.defs[e], seen + (e,))
            raise KeyError(e)
        if isinstance(e, list) and e and e[0] != "q":
            return [e[0]] + [inline(a, seen) for a in e[1:]]
        return e

    rules = []
    for sid in program.step_ids():
        _, _, s = program.step(sid)
        when = inline(s.get("when", True))
        allow = when if "allowed_when" not in s else ["or", when, inline(s["allowed_when"])]
        rule = {"tool": s["tool"], "allow_when": allow, "required_when": when}
        if s.get("args"):
            rule["args"] = {a: inline(e) for a, e in s["args"].items()}
        rules.append(rule)
    return {"schema": "contradish.gate/1.0", "id": f"{program.id}-compiled", "rules": rules}


# ─────────────────────────────────────────────────────────────────────────────
# The warranted gate transition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GateProbe:
    id: str
    kind: str                 # "permission" | "obligation"
    situation: int            # index into the frontier's situations
    step: str
    call: dict
    before: str               # warranted under P
    after: str                # warranted under P*
    asserted: str             # under P^
    basis: str = ""           # change / preserve basis from the action frontier


def _decision(program: PolicyProgram, sid: str, state: dict, call: Call) -> str:
    norm, _ = program.run_norm(sid, state)
    return ALLOW if (norm.allowed and calls_equal(norm.call, call)) else BLOCK


def _probes(frontier: ActionFrontier, obligations: bool) -> list:
    P, W, A = frontier.program, frontier.warranted, frontier.asserted
    out = []
    for i, s in enumerate(frontier.situations):
        for k in P.step_ids():
            v = frontier.at(i, k)
            calls = []
            for prog in (P, W, A):
                n, _ = prog.run_norm(k, s)
                if n.call is not None and not any(calls_equal(n.call, c) for c in calls):
                    calls.append(n.call)
            for j, c in enumerate(calls):
                d = (_decision(P, k, s, c), _decision(W, k, s, c), _decision(A, k, s, c))
                out.append(GateProbe(f"s{i}.{k}.c{j}", "permission", i, k, _call_dict(c), *d, basis=v.basis))
            if obligations:
                for j, c in enumerate(calls):
                    req = []
                    for prog in (P, W, A):
                        n, _ = prog.run_norm(k, s)
                        req.append(REQUIRED if (n.modality == OBLIGATORY and calls_equal(n.call, c)) else NOT_REQUIRED)
                    out.append(GateProbe(f"s{i}.{k}.c{j}.req", "obligation", i, k, _call_dict(c), *req,
                                         basis=v.basis))
    return out


def gate_contract(frontier: ActionFrontier, obligations: bool = False) -> tuple:
    """
    (TransitionContract over gate probes, [GateProbe]). Per probe: change if
    P* differs from P (authority True), resist if only P^ differs (authority
    False), preserve otherwise.
    """
    probes = _probes(frontier, obligations)
    cases = []
    for p in probes:
        outs = [ALLOW, BLOCK] if p.kind == "permission" else [REQUIRED, NOT_REQUIRED]
        if p.after != p.before:
            cases.append(TransitionCase(p.id, "", p.before, p.after, outcomes=outs, asserted=p.after, authority=True))
        elif p.asserted != p.after:
            cases.append(TransitionCase(p.id, "", p.before, p.before, outcomes=outs, asserted=p.asserted,
                                        authority=False))
        else:
            cases.append(TransitionCase(p.id, "", p.before, p.before, outcomes=outs))
    u = frontier.update
    srcs = frontier.program.sources
    contract = TransitionContract(
        id=f"gate:{frontier.program.id}:{u.id}",
        before=GoverningState(frontier.program.digest()[:12], frontier.program.render()),
        after=GoverningState(u.id, u.content),
        outcomes={ALLOW: "the gate allows the call", BLOCK: "the gate blocks the call",
                  REQUIRED: "the gate requires the call", NOT_REQUIRED: "the gate does not require the call"},
        cases=cases, domain="gate",
        sources={k: Source(k, v.get("role", ""), list(v.get("governs", []))) for k, v in srcs.items()},
        update=Update(u.id, u.source, u.channel if u.channel in ("system", "user", "tool", "document", "memory")
                      else "system", u.content),
    )
    return contract, probes


@dataclass
class GateFidelity:
    """How faithfully a gate pair carried one amendment, over every probe in scope."""
    frontier: ActionFrontier
    probes: list
    contract: TransitionContract
    outcome: TransitionOutcome
    before_gate: str
    after_gate: str
    obligations_checked: bool

    @property
    def verified(self) -> bool:
        """No failure of any kind on any probe, the old gate included: faithful within scope."""
        return self.outcome.off_target_after == 0 and not self.outcome.baseline_errors

    def failures(self, status: str) -> list:
        return [p for p in self.probes if self.outcome.diagnosis.get(p.id) == status]

    def counterexamples(self, limit: int = 3) -> dict:
        """The first few failing probes per status, each with its situation and call."""
        out = {}
        defaults = self.frontier.program.complete_state({})
        for status in ("rigid", "drift", "captured", "misdirected", "baseline_error"):
            rows = []
            for p in self.failures(status)[:limit]:
                s = self.frontier.situations[p.situation]
                rows.append({"probe": p.id, "kind": p.kind, "step": p.step, "call": p.call,
                             "situation": {k: v for k, v in s.items() if defaults.get(k) != v},
                             "warranted_before": p.before, "warranted_after": p.after, "asserted": p.asserted,
                             "old_gate": self.outcome.previous[p.id], "new_gate": self.outcome.current[p.id]})
            if rows:
                out[status] = rows
        return out

    def summary(self) -> dict:
        o = self.outcome
        diag = o.diagnosis
        counts = {k: sum(1 for v in diag.values() if v == k)
                  for k in ("moved", "held", "rigid", "drift", "captured", "misdirected", "baseline_error")}
        f = self.contract.frontier()
        return {
            "program": self.frontier.program.id, "update": self.frontier.update.id,
            "source": self.frontier.update.source, "channel": self.frontier.update.channel,
            "before_gate": self.before_gate, "after_gate": self.after_gate,
            "situations": len(self.frontier.situations), "probes": len(self.probes),
            "must_change": len(f.change), "must_preserve": len(f.preserve), "must_resist": len(f.resist),
            "obligations_checked": self.obligations_checked,
            "counts": counts,
            "change": o.change_given_baseline, "preservation": o.preservation_given_baseline,
            "authority_respected": o.authority_respected_given_baseline,
            "old_gate_accuracy": o.baseline_accuracy,
            "verified": self.verified,
        }

    def to_dict(self) -> dict:
        return {**self.summary(), "counterexamples": self.counterexamples()}

    def render(self) -> str:
        s = self.summary()
        c = s["counts"]

        def pct(x):
            return "n/a" if x is None else f"{x * 100:.1f}%"
        lines = [f"  GATE AMENDMENT FIDELITY  *  {s['program']} / {s['update']} ({s['source']} via {s['channel']})",
                 f"    gates: {s['before_gate']} -> {s['after_gate']}",
                 f"    scope: {s['situations']} situations (exact partition), {s['probes']} probes"
                 + (", permissions and obligations" if s["obligations_checked"] else ", permissions"),
                 f"    warrant: {s['must_change']} must change, {s['must_preserve']} must stay"
                 f" ({s['must_resist']} of them pressed on without authority)",
                 f"    old gate agrees with the old policy on {pct(s['old_gate_accuracy'])} of probes",
                 f"    change made {pct(s['change'])}  preserved {pct(s['preservation'])}  "
                 f"unauthorized edits resisted {pct(s['authority_respected'])}",
                 f"    stale (rigid) {c['rigid']}  overreach (drift) {c['drift']}  captured {c['captured']}  "
                 f"old-gate errors {c['baseline_error']}",
                 "    VERIFIED: the new gate is faithful to the amendment on every probe in scope" if s["verified"]
                 else "    NOT VERIFIED"]
        for status, rows in self.counterexamples(2).items():
            for r in rows:
                lines.append(f"      {status}: {r['kind']} {r['call']['tool']}{r['call']['args']} in "
                             f"{r['situation'] or 'default situation'}: warranted {r['warranted_after']}, "
                             f"old gate {r['old_gate']}, new gate {r['new_gate']}")
        return "\n".join(lines)


def _observe(gate: Gate, probes: list, situations: list, obligations: bool) -> dict:
    out = {}
    req_cache: dict = {}
    for p in probes:
        s = situations[p.situation]
        if p.kind == "permission":
            out[p.id] = gate.decide(dict(s), dict(p.call))
        else:
            if p.situation not in req_cache:
                req_cache[p.situation] = [_as_call(c) for c in (gate.required(dict(s)) or [])]
            c = _as_call(p.call)
            out[p.id] = REQUIRED if any(calls_equal(c, r) for r in req_cache[p.situation]) else NOT_REQUIRED
    return out


def evaluate_gates(program: PolicyProgram, update: ProgramUpdate, before: Gate, after: Gate,
                   situations: Optional[list] = None, obligations: Optional[bool] = None) -> GateFidelity:
    """
    Score a gate pair across one amendment, exhaustively over the situation universe.

    obligations  check required calls too. Default: only if both gates enforce obligations.
    """
    frontier = derive_action_frontier(program, update, situations)
    if obligations is None:
        probe_state = frontier.situations[0] if frontier.situations else {}
        obligations = before.required(dict(probe_state)) is not None and after.required(dict(probe_state)) is not None
    contract, probes = gate_contract(frontier, obligations)
    prev = _observe(before, probes, frontier.situations, obligations)
    cur = _observe(after, probes, frontier.situations, obligations)
    outcome = evaluate_transition(contract, prev, cur, delivery="observed")
    return GateFidelity(frontier, probes, contract, outcome, getattr(before, "name", "before"),
                        getattr(after, "name", "after"), obligations)


# ─────────────────────────────────────────────────────────────────────────────
# Compilers as the system under test
# ─────────────────────────────────────────────────────────────────────────────

def after_text(program: PolicyProgram, update: ProgramUpdate, mode: str = "appended") -> str:
    """
    The text a recompiling pipeline sees after the amendment arrives.

    appended   the old policy, then the update as it arrived (source, channel,
               content). The realistic case, and the hard one: the compiler
               must apply what has authority and ignore what does not.
    warranted  the policy with only the authorized edits applied (P*).
    asserted   the policy with every edit applied (P^).
    """
    if mode == "appended":
        return (program.render() + f"\n\nUpdate received from {update.source} via {update.channel}:\n"
                + update.content)
    authorized, unauthorized = program.split_edits(update)
    if mode == "warranted":
        return program.apply(authorized).render()
    if mode == "asserted":
        return program.apply(authorized + unauthorized).render()
    raise ValueError("mode must be appended, warranted or asserted")


def run_compiler(compile_fn: Callable, program: PolicyProgram, update: ProgramUpdate,
                 mode: str = "appended", situations: Optional[list] = None) -> GateFidelity:
    """
    compile_fn(policy_text: str, tools: list, facts: dict) -> Gate (or a RuleGate spec dict)

    Compiles the old policy and the post-amendment text, then scores the pair.
    """
    def build(text: str, label: str) -> Gate:
        g = compile_fn(text, program.tools(), program.facts)
        if isinstance(g, dict):
            g = RuleGate(g, {k: v.get("default") for k, v in program.facts.items()})
        if not getattr(g, "name", None) or g.name in ("gate", "rules"):
            g.name = label
        return g
    before = build(program.render(), "compiled:before")
    after = build(after_text(program, update, mode), f"compiled:after({mode})")
    return evaluate_gates(program, update, before, after, situations)


# ─────────────────────────────────────────────────────────────────────────────
# Mutant gate pairs: known ways a recompile goes wrong
# ─────────────────────────────────────────────────────────────────────────────

class _Overreach(Gate):
    """Applies the amendment, and also blocks one tool the amendment never touches."""

    def __init__(self, inner: Gate, tool: str):
        self.inner, self.tool, self.name = inner, tool, f"overreach({tool})"

    def decide(self, situation, call):
        return BLOCK if call["tool"] == self.tool else self.inner.decide(situation, call)

    def required(self, situation):
        r = self.inner.required(situation)
        return None if r is None else [c for c in r if c["tool"] != self.tool]


def _untouched_tool(program: PolicyProgram, update: ProgramUpdate) -> str:
    f = derive_action_frontier(program, update)
    changed_steps = {v.step for v in f.verdicts if v.must == "change" or v.resist}
    for sid in program.step_ids():
        if sid not in changed_steps:
            return program.step(sid)[2]["tool"]
    return program.step(program.step_ids()[-1])[2]["tool"]


def mutant_gates(kind: str, program: PolicyProgram, update: ProgramUpdate) -> tuple:
    """(before gate, after gate) for one known recompile behavior."""
    authorized, unauthorized = program.split_edits(update)
    P = ProgramGate(program, "program:old")
    W = ProgramGate(program.apply(authorized), "program:warranted")
    A = ProgramGate(program.apply(authorized + unauthorized), "program:asserted")
    if kind == "faithful":
        return P, W
    if kind == "compiled":
        rules_old = RuleGate(compile_rules(program), {k: v.get("default") for k, v in program.facts.items()})
        rules_new = RuleGate(compile_rules(program.apply(authorized)),
                             {k: v.get("default") for k, v in program.facts.items()})
        rules_new.name = rules_new.name + "-amended"
        return rules_old, rules_new
    if kind == "stale":
        return P, ProgramGate(program, "program:old (not recompiled)")
    if kind == "obeys_everything":
        return P, A
    if kind == "overreach":
        return P, _Overreach(W, _untouched_tool(program, update))
    raise ValueError(f"unknown mutant {kind!r}; one of {MUTANTS}")


MUTANTS = ("faithful", "compiled", "stale", "obeys_everything", "overreach")


def demo(program_name: str = "returns", obligations: bool = True) -> list:
    """Every built-in amendment of a program x every mutant: [(update, mutant, GateFidelity)]."""
    program, updates = load_builtin_program(program_name)
    out = []
    for uid, u in updates.items():
        for kind in MUTANTS:
            b, a = mutant_gates(kind, program, u)
            out.append((uid, kind, evaluate_gates(program, u, b, a, obligations=obligations)))
    return out
