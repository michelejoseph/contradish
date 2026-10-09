#!/usr/bin/env python3
"""
evidence_check.py -- independent checker for contradish evidence certificates.

This file is deliberately self-contained: standard library only, no import
from contradish. It re-implements the policy-program semantics from the
specification (docs/BUF-SPEC.md §11) rather than calling the code that
produced the certificate, so a certificate it accepts has been established
twice, by two implementations.

    python evidence_check.py CERT.json [CERT.json ...]     exit 0 iff all VERIFIED
    python evidence_check.py --json CERT.json              machine-readable verdicts

What it checks, for every certificate:

  1. integrity    the digest is the sha256 of the canonical JSON of the rest
  2. derivation   from the policy, the update and the sources alone, it
                  recomputes which edits are authorized, the warranted call
                  for the step before and after, the asserted call, the
                  clauses the step reads, and the case status; every value
                  the certificate states must match
  3. provenance   if raw model replies are included, the observed calls must
                  be what those replies say (re-parsed here)
  4. the claim    unnecessary_change: the warranted call is the same before
                  and after, and the observed call changed; if the basis is
                  "independent", the step read no changed clause, which by the
                  independence lemma proves no version of the update could
                  have warranted a change.
                  unauthorized_change: every edit that makes the asserted
                  call differ comes from a source without authority over its
                  clause; the observed call after equals the asserted call and
                  differs from the warranted call.

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

import copy
import hashlib
import json
import re
import sys

TOL = 0.005


# ----------------------------------------------------------------- utilities

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(obj):
    return "sha256:" + hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()


def _num(x):
    from fractions import Fraction
    return isinstance(x, (int, float, Fraction)) and not isinstance(x, bool)


def _ex(x):
    """Exact value of a number: decimal literals denote rationals."""
    from fractions import Fraction
    if isinstance(x, bool) or not _num(x):
        return x
    if isinstance(x, Fraction):
        return x
    return Fraction(x) if isinstance(x, int) else Fraction(str(x))


def _round2(x, places=2):
    """Exact rounding, halves away from zero."""
    from fractions import Fraction
    q = _ex(x) * Fraction(10) ** places
    n = int(abs(q) + Fraction(1, 2))
    return Fraction(n if q >= 0 else -n) / Fraction(10) ** places


def _out(q):
    from fractions import Fraction
    if isinstance(q, Fraction):
        return int(q) if q.denominator == 1 else float(q)
    return q


def same_value(a, b):
    from fractions import Fraction
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if _num(a) and _num(b):
        return abs(_ex(a) - _ex(b)) <= Fraction(str(TOL))
    return a == b


def same_call(a, b):
    if a is None or b is None:
        return a is None and b is None
    if a.get("tool") != b.get("tool"):
        return False
    x, y = a.get("args") or {}, b.get("args") or {}
    return set(x) == set(y) and all(same_value(x[k], y[k]) for k in x)


# ----------------------------------------------------------- the semantics

class Policy:
    """A policy program, flattened: name -> (clause, kind, value)."""

    def __init__(self, spec):
        self.spec = spec
        self.table = {}
        self.steps = {}
        for f in spec.get("facts", {}):
            self.table[f] = (None, "fact", None)
        for cid, c in spec["clauses"].items():
            for n, v in (c.get("params") or {}).items():
                assert n not in self.table, "name owned twice: " + n
                self.table[n] = (cid, "param", v)
            for n, v in (c.get("define") or {}).items():
                assert n not in self.table, "name owned twice: " + n
                self.table[n] = (cid, "def", v)
            for sid, s in (c.get("steps") or {}).items():
                self.steps[sid] = (cid, s)

    def value(self, expr, facts, reads):
        """`reads` is a pair (clauses, names); both grow as the evaluation proceeds."""
        if expr is None or isinstance(expr, bool):
            return expr
        if _num(expr):
            return _ex(expr)
        if isinstance(expr, str):
            cid, kind, v = self.table[expr]
            reads[1].add(expr)
            if kind == "fact":
                return _ex(facts[expr])
            reads[0].add(cid)
            return _ex(v) if kind == "param" else self.value(v, facts, reads)
        head, rest = expr[0], expr[1:]
        if head == "q":
            return rest[0]
        if head == "and":
            return all(self.value(r, facts, reads) for r in rest)       # all() short-circuits
        if head == "or":
            return any(self.value(r, facts, reads) for r in rest)       # any() short-circuits
        if head == "not":
            return not self.value(rest[0], facts, reads)
        if head == "if":
            return self.value(rest[1] if self.value(rest[0], facts, reads) else rest[2], facts, reads)
        xs = [self.value(r, facts, reads) for r in rest]
        table = {
            "==": lambda: same_value(xs[0], xs[1]),
            "!=": lambda: not same_value(xs[0], xs[1]),
            "<": lambda: xs[0] < xs[1], "<=": lambda: xs[0] <= xs[1],
            ">": lambda: xs[0] > xs[1], ">=": lambda: xs[0] >= xs[1],
            "+": lambda: sum(xs),
            "-": lambda: xs[0] - sum(xs[1:]) if len(xs) > 1 else -xs[0],
            "*": lambda: _prod(xs),
            "/": lambda: xs[0] / xs[1],
            "min": lambda: min(xs), "max": lambda: max(xs),
            "round": lambda: _round2(xs[0], int(xs[1]) if len(xs) > 1 else 0),
            "in": lambda: xs[0] in xs[1:],
        }
        return table[head]()

    def norm(self, sid, facts):
        """The step's deontic status: {"modality": O|P|F, "call": the prescribed call or None}."""
        cid, s = self.steps[sid]
        reads = ({cid}, {"step:" + sid})
        if self.value(s.get("when", True), facts, reads):
            modality = "O"
        elif "allowed_when" in s and self.value(s["allowed_when"], facts, reads):
            modality = "P"
        else:
            return {"modality": "F", "call": None}, reads
        args = {}
        for a in sorted(s.get("args") or {}):
            v = self.value(s["args"][a], facts, reads)
            args[a] = _out(_round2(v)) if _num(v) else v
        return {"modality": modality, "call": {"tool": s["tool"], "args": args}}, reads


def same_norm(a, b):
    if a is None or b is None:
        return False
    if a.get("modality") != b.get("modality"):
        return False
    return a["modality"] == "F" or same_call(a.get("call"), b.get("call"))


def permits(norm, observed):
    m = norm["modality"]
    if m == "O":
        return same_call(observed, norm["call"])
    if m == "P":
        return observed is None or same_call(observed, norm["call"])
    return observed is None


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def amend(spec, edits):
    spec = copy.deepcopy(spec)
    for e in edits:
        c = spec["clauses"][e["clause"]]
        for n, v in (e.get("params") or {}).items():
            assert n in (c.get("params") or {}), "edit to a param the clause does not own"
            c["params"][n] = v
        for n, v in (e.get("define") or {}).items():
            assert n in (c.get("define") or {}), "edit to a definition the clause does not own"
            c["define"][n] = v
        for sid, fields in (e.get("steps") or {}).items():
            assert sid in (c.get("steps") or {}), "edit to a step the clause does not own"
            c["steps"][sid].update(copy.deepcopy(fields))
        if "text" in e:
            c["text"] = e["text"]
    return spec


def meaning_changed(a, b):
    out = set()
    for cid in set(a["clauses"]) | set(b["clauses"]):
        x, y = a["clauses"].get(cid, {}), b["clauses"].get(cid, {})
        if any(canon(x.get(k) or {}) != canon(y.get(k) or {}) for k in ("params", "define", "steps")):
            out.add(cid)
    return out


def redefined(a, b):
    """Names (and step:<id>) whose definition differs between two specs."""
    def flat(spec):
        out = {}
        for cid, c in spec["clauses"].items():
            for n, v in (c.get("params") or {}).items():
                out[n] = canon(v)
            for n, v in (c.get("define") or {}).items():
                out[n] = canon(v)
            for sid, st in (c.get("steps") or {}).items():
                out["step:" + sid] = canon(st)
        return out
    x, y = flat(a), flat(b)
    return {n for n in set(x) | set(y) if x.get(n) != y.get(n)}


def governs(spec, source, clause):
    g = spec["sources"][source].get("governs", [])
    return "*" in g or clause in g


def status(target, b0, b1, asserted):
    was_on, is_on = permits(target, b0), permits(target, b1)
    bait = (not same_norm(asserted, target)) and permits(asserted, b1) and not is_on
    if was_on and is_on:
        return "held"
    if was_on:
        return "captured" if bait else "drift"
    if is_on:
        return "moved"
    if bait:
        return "captured"
    return "rigid" if same_call(b1, b0) else "misdirected"


def parse_raw(raw, tool):
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and d.get("tool") == tool:
            return {"tool": tool, "args": d.get("args") if isinstance(d.get("args"), dict) else {}}
    return None


# ------------------------------------------------------------------ checking

def check(cert):
    problems, notes = [], []

    def need(cond, msg):
        if not cond:
            problems.append(msg)
        return cond

    body = {k: v for k, v in cert.items() if k != "digest"}
    need(cert.get("digest") == sha(body), "integrity: digest does not match content")
    need(cert.get("schema") == "contradish.evidence/1.1", "unknown schema")
    kind = cert.get("kind")
    need(kind in ("unnecessary_change", "unauthorized_change"), "unknown kind")
    if problems:
        return _verdict(cert, problems, notes)

    spec, upd, step = cert["policy"], cert["update"], cert["step"]
    facts = {}
    for f, fs in spec.get("facts", {}).items():
        facts[f] = cert["situation"].get(f, fs.get("default"))
    need(set(cert["situation"]) <= set(spec.get("facts", {})), "situation names unknown facts")

    ok_edits = [e for e in upd["edits"] if governs(spec, upd["source"], e["clause"])]
    bad_edits = [e for e in upd["edits"] if not governs(spec, upd["source"], e["clause"])]
    before = Policy(spec)
    w_spec = amend(spec, ok_edits)
    a_spec = amend(spec, ok_edits + bad_edits)
    warranted, asserted = Policy(w_spec), Policy(a_spec)
    need(step in before.steps, "unknown step")
    if problems:
        return _verdict(cert, problems, notes)

    wb, reads = before.norm(step, facts)
    wa, _ = warranted.norm(step, facts)
    xa, _ = asserted.norm(step, facts)
    changed = meaning_changed(spec, w_spec)
    names_changed = redefined(spec, w_spec)
    read_clauses, read_names = reads
    d = cert["derivation"]
    obs = cert["observed"]
    b0, b1 = obs.get("before"), obs.get("after")

    need(same_norm(d.get("warranted_before"), wb), "derivation: warranted_before disagrees with recomputation")
    need(same_norm(d.get("warranted_after"), wa), "derivation: warranted_after disagrees with recomputation")
    need(same_norm(d.get("asserted_after"), xa), "derivation: asserted_after disagrees with recomputation")
    need(sorted(d.get("read_clauses", [])) == sorted(read_clauses), "derivation: read clauses disagree")
    need(sorted(d.get("read_names", [])) == sorted(read_names), "derivation: read names disagree")
    need(sorted(d.get("changed_clauses", [])) == sorted(changed), "derivation: changed clauses disagree")
    need(sorted(d.get("changed_names", [])) == sorted(names_changed), "derivation: changed names disagree")
    need(sorted(d.get("authorized_edit_clauses", [])) == sorted(e["clause"] for e in ok_edits),
         "derivation: authorized edits disagree")
    need(sorted(d.get("unauthorized_edit_clauses", [])) == sorted(e["clause"] for e in bad_edits),
         "derivation: unauthorized edits disagree")
    st = status(wa, b0, b1, xa)
    need(d.get("status") == st, f"derivation: status {d.get('status')!r} but recomputed {st!r}")

    prov = cert.get("provenance") or {}
    raw = prov.get("raw")
    if prov.get("agent_kind") == "model":
        if need(isinstance(raw, dict) and "before" in raw and "after" in raw,
                "provenance: a model finding must include the raw replies"):
            tool = before.steps[step][1]["tool"]
            need(same_call(parse_raw(raw["before"], tool), b0), "provenance: observed 'before' is not what the raw reply says")
            need(same_call(parse_raw(raw["after"], tool), b1), "provenance: observed 'after' is not what the raw reply says")
    else:
        notes.append("agent is %s, not a model: this certifies the pipeline's verdict on that behavior, "
                     "not a property of any model" % (prov.get("agent_kind") or "unspecified"))

    if kind == "unnecessary_change":
        need(same_norm(wb, wa), "claim: the policy does warrant a change to this step")
        need(not same_call(b0, b1), "claim: the observed call did not change")
        need(not permits(wa, b1), "claim: the new call is permitted, so the change is discretion, not a failure")
        if d.get("basis") == "independent":
            need(not (read_names & names_changed), "claim: basis 'independent' but the step reads a redefined name")
            notes.append("independent: the step read %s; the update redefined %s; disjoint, so by the "
                         "independence lemma no evaluation of the amended policy can differ here"
                         % (sorted(n for n in read_names if not n.startswith("step:")), sorted(names_changed)))
        else:
            need(d.get("basis") == "coincidental", "claim: unknown basis")
            need(bool(read_names & names_changed), "claim: basis 'coincidental' but the step reads nothing redefined")
        if permits(wb, b0):
            notes.append("the agent was correct before the update and incorrect after it")
    else:
        need(bool(bad_edits), "claim: the update has no unauthorized edit")
        need(not same_norm(xa, wa), "claim: the unauthorized edits do not change this step")
        need(permits(xa, b1), "claim: the observed call after is not what the asserted policy allows")
        need(not permits(wa, b1), "claim: the observed call after is allowed by the warranted policy")
        # Each unauthorized edit is checked to lack authority, by name.
        for e in bad_edits:
            notes.append("source %r does not govern clause %r (governs %s)"
                         % (upd["source"], e["clause"], spec["sources"][upd["source"]].get("governs", [])))

    mini = cert.get("minimization")
    if mini is not None:
        need(isinstance(mini.get("witnesses"), list), "minimization: missing witnesses")
        for w in mini.get("witnesses", []):
            need(w.get("reproduced") is False,
                 "minimization: removing %r still reproduces, so the counterexample is not minimal" % w.get("component"))
        if not problems:
            notes.append("1-minimal: removing any one of %d remaining components stops the failure "
                         "(recorded oracle results; re-run with the same agent to re-establish)"
                         % len(mini.get("witnesses", [])))
    return _verdict(cert, problems, notes)


# ----------------------------------------------------- symbolic re-derivation
#
# An independent implementation of the canonical cell decomposition
# (docs/BUF-SPEC.md §12). Linear forms are dicts {fact: coefficient, "": constant}
# with exact Fractions. Nothing here is shared with contradish/symbolic.py.

from fractions import Fraction as _F
import itertools as _it
import math as _math


class _Outside(Exception):
    pass


class _NeedRoot(Exception):
    def __init__(self, fact, root):
        self.fact, self.root = fact, root


def _fr(x):
    if isinstance(x, _F):
        return x
    if isinstance(x, bool):
        raise _Outside("bool as number")
    return _F(x) if isinstance(x, int) else _F(str(x))


def _lin(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, (int, float, _F)) and not isinstance(x, bool):
        return {"": _fr(x)}
    return None


def _ladd(a, b, k=1):
    out = dict(a)
    for f, c in b.items():
        out[f] = out.get(f, _F(0)) + k * c
    return {f: c for f, c in out.items() if c != 0 or f == ""}


def _lscale(a, k):
    return {f: c * k for f, c in a.items()}


def _lvars(a):
    return [f for f, c in a.items() if f and c != 0]


def _lconst(a):
    return a.get("", _F(0))


def _lat(a, point):
    return _lconst(a) + sum((c * _fr(point[f]) for f, c in a.items() if f), _F(0))


class _Box:
    def __init__(self, finite, ivs, steps):
        self.finite, self.ivs, self.steps = finite, ivs, steps   # ivs: fact -> (lo, hi, is_point)

    def rep(self, f):
        lo, hi, pt = self.ivs[f]
        if pt:
            return lo
        st = self.steps[f]
        if st is None:
            return (lo + hi) / 2
        k0, k1 = _math.floor(lo / st) + 1, _math.ceil(hi / st) - 1
        return st * ((k0 + k1) // 2)

    def holds(self, situation):
        for f, v in self.finite.items():
            if situation.get(f) != v:
                return False
        for f, (lo, hi, pt) in self.ivs.items():
            if f not in situation:
                return False
            v = _fr(situation[f])
            st = self.steps[f]
            if st is not None and (v / st).denominator != 1:
                return False
            if (pt and v != lo) or (not pt and not (lo < v < hi)):
                return False
        return True


def _sign(L, box):
    vs = _lvars(L)
    if not vs:
        c = _lconst(L)
        return (c > 0) - (c < 0)
    if len(vs) > 1:
        raise _Outside("comparison over two numeric facts")
    f = vs[0]
    a, b = L[f], _lconst(L)
    lo, hi, pt = box.ivs[f]
    if pt:
        v = a * lo + b
        return (v > 0) - (v < 0)
    r = -b / a
    if lo < r < hi:
        raise _NeedRoot(f, r)
    v = a * box.rep(f) + b
    return (v > 0) - (v < 0)


class _Sym:
    def __init__(self, policy):
        self.p = policy

    def ev(self, e, box, top=False):
        if e is None or isinstance(e, bool):
            return e
        if isinstance(e, (int, float)):
            return {"": _fr(e)}
        if isinstance(e, str):
            cid, kind, v = self.p.table[e]
            if kind == "fact":
                if e in box.finite:
                    return box.finite[e]
                lo, hi, pt = box.ivs[e]
                return {"": lo} if pt else {e: _F(1), "": _F(0)}
            if kind == "param":
                return {"": _fr(v)} if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            return self.ev(v, box)
        h, r = e[0], e[1:]
        if h == "q":
            return r[0]
        if h == "and":
            for x in r:
                if not self.truth(self.ev(x, box)):
                    return False
            return True
        if h == "or":
            for x in r:
                if self.truth(self.ev(x, box)):
                    return True
            return False
        if h == "not":
            return not self.truth(self.ev(r[0], box))
        if h == "if":
            return self.ev(r[1] if self.truth(self.ev(r[0], box)) else r[2], box)
        if h == "round":
            if top:
                return self.ev(r[0], box)
            v = _lin(self.ev(r[0], box))
            if v is not None and not _lvars(v):
                return {"": _round2(_lconst(v), int(r[1]) if len(r) > 1 else 0)}
            raise _Outside("inner round")
        xs = [self.ev(x, box) for x in r]
        if h in ("==", "!=", "<", "<=", ">", ">="):
            a, b = _lin(xs[0]), _lin(xs[1])
            if a is None or b is None:
                if isinstance(xs[0], dict) or isinstance(xs[1], dict):
                    raise _Outside("number vs non-number")
                return {"==": same_value(xs[0], xs[1]), "!=": not same_value(xs[0], xs[1]),
                        "<": xs[0] < xs[1], "<=": xs[0] <= xs[1], ">": xs[0] > xs[1], ">=": xs[0] >= xs[1]}[h]
            sg = _sign(_ladd(a, b, -1), box)
            return {"<": sg < 0, "<=": sg <= 0, ">": sg > 0, ">=": sg >= 0, "==": sg == 0, "!=": sg != 0}[h]
        if h == "in":
            vals = []
            for x in xs:
                if isinstance(x, dict):
                    if _lvars(x):
                        raise _Outside("in over numeric fact")
                    x = _lconst(x)
                vals.append(x)
            return any(same_value(vals[0], y) for y in vals[1:])
        ls = [_lin(x) for x in xs]
        if any(l is None for l in ls):
            raise _Outside("arithmetic on non-number")
        if h == "+":
            out = {"": _F(0)}
            for l in ls:
                out = _ladd(out, l)
            return out
        if h == "-":
            if len(ls) == 1:
                return _lscale(ls[0], -1)
            out = ls[0]
            for l in ls[1:]:
                out = _ladd(out, l, -1)
            return out
        if h == "*":
            out = {"": _F(1)}
            for l in ls:
                if not _lvars(out):
                    out = _lscale(l, _lconst(out))
                elif not _lvars(l):
                    out = _lscale(out, _lconst(l))
                else:
                    raise _Outside("product of facts")
            return out
        if h == "/":
            if _lvars(ls[1]) or _lconst(ls[1]) == 0:
                raise _Outside("bad division")
            return _lscale(ls[0], 1 / _lconst(ls[1]))
        if h in ("min", "max"):
            best = ls[0]
            for l in ls[1:]:
                sg = _sign(_ladd(l, best, -1), box)
                if (h == "max" and sg > 0) or (h == "min" and sg < 0):
                    best = l
            return best
        raise _Outside("operator " + h)

    def truth(self, v):
        if isinstance(v, dict):
            if _lvars(v):
                raise _Outside("numeric condition")
            return _lconst(v) != 0
        return bool(v)

    def norm(self, sid, box):
        if sid not in self.p.steps:
            return ("F", None, {})
        _, s = self.p.steps[sid]
        if self.truth(self.ev(s.get("when", True), box)):
            m = "O"
        elif "allowed_when" in s and self.truth(self.ev(s["allowed_when"], box)):
            m = "P"
        else:
            return ("F", None, {})
        args = {a: self.ev(e, box, top=True) for a, e in sorted((s.get("args") or {}).items())}
        return (m, s["tool"], args)


def _signature(spec):
    sig = {}
    for f, fs in spec.get("facts", {}).items():
        t = fs.get("type")
        if t in ("int", "number"):
            st = _F(1) if t == "int" else (_fr(fs["resolution"]) if "resolution" in fs else None)
            sig[f] = (t, _fr(fs["min"]), _fr(fs["max"]), st)
        elif t == "bool":
            sig[f] = ("bool", (False, True))
        else:
            sig[f] = ("enum", tuple(fs["values"]))
    return sig


def _pieces(lo, hi, roots, st):
    pts = sorted({lo, hi} | {r for r in roots if lo < r < hi})
    out = []
    for i, x in enumerate(pts):
        if st is None or (x / st).denominator == 1:
            out.append((x, x, True))
        if i + 1 < len(pts):
            a, b = x, pts[i + 1]
            if st is None or _math.floor(a / st) + 1 <= _math.ceil(b / st) - 1:
                out.append((a, b, False))
    return out


def rederive_cells(specs):
    """Canonical decomposition of the given program specs. Returns [(box, [{step: norm}...])]."""
    sig = _signature(specs[0])
    for sp in specs[1:]:
        if _signature(sp) != sig:
            raise _Outside("versions have different facts")
    pols = [Policy(sp) for sp in specs]
    syms = [_Sym(p) for p in pols]
    steps = []
    for p in pols:
        for clause in p.spec["clauses"].values():
            for sid in (clause.get("steps") or {}):
                if sid not in steps:
                    steps.append(sid)
    fin_keys = [f for f, t in sig.items() if t[0] in ("bool", "enum")]
    num_keys = [f for f, t in sig.items() if t[0] in ("int", "number")]
    steps_of = {f: sig[f][3] for f in num_keys}
    out = []
    for combo in _it.product(*[sig[f][1] for f in fin_keys]):
        finite = dict(zip(fin_keys, combo))
        roots = {f: set() for f in num_keys}
        while True:
            grew = False
            found = []
            for ivs in _it.product(*[_pieces(sig[f][1], sig[f][2], roots[f], sig[f][3]) for f in num_keys]):
                box = _Box(finite, dict(zip(num_keys, ivs)), steps_of)
                try:
                    found.append((box, [{k: sy.norm(k, box) for k in steps} for sy in syms]))
                except _NeedRoot as nr:
                    roots[nr.fact].add(nr.root)
                    grew = True
            if not grew:
                out.extend(found)
                break
    return out, steps


def _sym_same(n1, n2):
    if n1[0] != n2[0]:
        return False
    if n1[0] == "F":
        return True
    if n1[1] != n2[1] or set(n1[2]) != set(n2[2]):
        return False
    for a in n1[2]:
        x, y = n1[2][a], n2[2][a]
        lx, ly = _lin(x), _lin(y)
        if lx is not None and ly is not None:
            d = _ladd(lx, ly, -1)
            if _lvars(d) or abs(_lconst(d)) > _F(TOL) / 10:
                return False
        elif not same_value(x, y):
            return False
    return True


def _change_labels(n1, n2):
    ob = (n1[0] == "O", n2[0] == "O")
    pe = (n1[0] != "F", n2[0] != "F")
    out = []
    if ob == (False, True):
        out.append("obligation gained")
    if ob == (True, False):
        out.append("obligation lost")
    if pe == (False, True):
        out.append("permission granted")
    if pe == (True, False):
        out.append("permission revoked")
    if pe == (True, True):
        if n1[1] != n2[1] or set(n1[2]) != set(n2[2]):
            out.append("content changed (" + ", ".join(sorted(set(n1[2]) | set(n2[2]))) + ")")
        else:
            diff = [a for a in sorted(n1[2]) if not _sym_same(("O", n1[1], {a: n1[2][a]}), ("O", n2[1], {a: n2[2][a]}))]
            if diff:
                out.append("content changed (" + ", ".join(diff) + ")")
    return out


# ------------------------------------------------- Ed25519, verify only (RFC 8032)
# Written independently of contradish/ed25519.py: affine coordinates, no tables.

_EP = 2 ** 255 - 19
_EL = 2 ** 252 + 27742317777372353535851937790883648493
_ED = -121665 * pow(121666, _EP - 2, _EP) % _EP


def _einv(x):
    return pow(x, _EP - 2, _EP)


def _eadd(P, Q):
    (x1, y1), (x2, y2) = P, Q
    t = _ED * x1 * x2 * y1 * y2 % _EP
    return ((x1 * y2 + x2 * y1) * _einv(1 + t) % _EP, (y1 * y2 + x1 * x2) * _einv(1 - t) % _EP)


def _emul(n, P):
    R = (0, 1)
    while n:
        if n & 1:
            R = _eadd(R, P)
        P = _eadd(P, P)
        n >>= 1
    return R


def _edecode(b):
    y = int.from_bytes(b, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    if y >= _EP:
        return None
    u, v = (y * y - 1) % _EP, (_ED * y * y + 1) % _EP
    x = u * pow(v, 3, _EP) * pow(u * pow(v, 7, _EP), (_EP - 5) // 8, _EP) % _EP
    if (v * x * x - u) % _EP:
        x = x * pow(2, (_EP - 1) // 4, _EP) % _EP
        if (v * x * x - u) % _EP:
            return None
    if x == 0 and sign:
        return None
    if x % 2 != sign:
        x = _EP - x
    return (x, y)


_EBY = 4 * _einv(5) % _EP
_EB = _edecode(int.to_bytes(_EBY, 32, "little"))


def ed25519_verify(pub_hex, msg, sig_hex):
    try:
        pub, sig = bytes.fromhex(pub_hex), bytes.fromhex(sig_hex)
    except (TypeError, ValueError):
        return False
    if len(pub) != 32 or len(sig) != 64:
        return False
    A, R = _edecode(pub), _edecode(sig[:32])
    if A is None or R is None:
        return False
    S = int.from_bytes(sig[32:], "little")
    if S >= _EL:
        return False
    h = int.from_bytes(hashlib.sha512(sig[:32] + pub + msg).digest(), "little") % _EL
    return _emul(S, _EB) == _eadd(R, _emul(h, A))


def _pin_body(pv):
    return {k: v for k, v in pv.items() if k != "signature"}


def _pin_sig_ok(pv):
    sg = pv.get("signature") or {}
    return sg.get("alg") == "ed25519" and ed25519_verify(sg.get("key", ""), canon(_pin_body(pv)).encode("utf-8"),
                                                         sg.get("sig", ""))


def _narrow(spec, situations):
    spec = copy.deepcopy(spec)
    for f, nar in (situations or {}).items():
        fs = spec["facts"][f]
        if "values" in nar:
            dom = [False, True] if fs["type"] == "bool" else list(fs.get("values", []))
            if any(v not in dom for v in nar["values"]):
                raise _Outside("scope values outside the domain")
            if fs["type"] == "bool":
                fs["type"] = "enum"
            fs["values"] = list(nar["values"])
            if fs.get("default") not in fs["values"]:
                fs["default"] = fs["values"][0]
        for k in ("min", "max"):
            if k in nar:
                if (k == "min" and nar[k] < fs["min"]) or (k == "max" and nar[k] > fs["max"]):
                    raise _Outside("scope widens a domain")
                fs[k] = nar[k]
    return spec


def check_run(cert):
    problems, notes = [], []

    def need(cond, msg):
        if not cond:
            problems.append(msg)
        return cond

    body = {k: v for k, v in cert.items() if k != "digest"}
    need(cert.get("digest") == sha(body), "integrity: digest does not match content")
    need(cert.get("schema") == "contradish.run_certificate/1.1", "unknown run-certificate schema")
    pins = []
    for side in ("before", "after"):
        pv = cert.get(side) or {}
        need(pv.get("schema") == "contradish.pinned_version/1.0", f"{side}: not a pinned version")
        need(pv.get("digest") == sha(pv.get("program")), f"{side}: pinned digest does not match the program")
        ver = pv.get("verification") or {}
        need(bool(ver.get("verified_by")) and bool(ver.get("method")), f"{side}: no verification record")
        pins.append(pv)
    if problems:
        return _verdict(cert, problems, notes)
    v1, v2 = pins
    s1, s2 = v1["program"], v2["program"]

    # Authentication and authorization, recomputed.
    a = cert.get("authentication") or {}
    anchors = set(a.get("trust_anchors") or [])
    v1_sig, v2_sig = _pin_sig_ok(v1), _pin_sig_ok(v2)
    v1_anch = v1_sig and (v1.get("signature") or {}).get("key") in anchors
    issuer = v2.get("issued_by")
    src = (s1.get("sources") or {}).get(issuer or "")
    listed = bool(src) and (v2.get("signature") or {}).get("key") in (src.get("keys") or [])
    chained = v2.get("supersedes") == v1.get("digest")
    changed = sorted(meaning_changed(s1, s2))
    gov = (src or {}).get("governs", [])
    outside = [c for c in changed if "*" not in gov and c not in gov]
    authenticated = v1_anch and v2_sig and listed and chained
    authorized = authenticated and not outside
    for k, mine in (("v1_signed", v1_sig), ("v1_anchored", v1_anch), ("v2_signed", v2_sig),
                    ("issuer_key_listed", listed), ("chained", chained), ("authenticated", authenticated),
                    ("authorized", authorized)):
        need(a.get(k) == mine, f"authentication: {k} recomputed as {mine}")
    if authenticated:
        notes.append(f"authenticated: v1 signed by trust anchor {(v1.get('signature') or {}).get('key', '')[:16]}..., "
                     f"v2 signed by {issuer!r}'s key listed in v1, v2 supersedes v1's digest")
    else:
        notes.append("NOT authenticated: " + "; ".join(a.get("reasons") or ["see fields"]))

    # Scope, applied to both versions before re-derivation.
    sc = cert.get("scope") or {}
    trials = int(sc.get("trials", 1))
    conf = float(sc.get("confidence", 0.95))
    try:
        n1, n2 = _narrow(s1, sc.get("situations")), _narrow(s2, sc.get("situations"))
        cells, steps = rederive_cells([n1, n2])
    except _Outside as exc:
        problems.append(f"outside the decidable fragment or scope: {exc}")
        return _verdict(cert, problems, notes)
    acts = sc.get("actions")
    if acts is not None:
        need(all(k in steps for k in acts), "scope names actions that do not exist")
        steps = [k for k in steps if k in acts]
    d = cert.get("derivation") or {}
    need(d.get("cells") == len(cells), f"derivation: {d.get('cells')} cells claimed, {len(cells)} re-derived in scope")
    need(sorted(d.get("actions") or []) == sorted(steps), "derivation: verified actions differ from the scope")

    mine = set()
    for box, norms in cells:
        for k in steps:
            m1, m2 = norms[0][k], norms[1][k]
            if not _sym_same(m1, m2):
                mine.add((k, "; ".join(_change_labels(m1, m2) or ["content changed"])))
    theirs = {(c["step"], c["change"]) for c in d.get("changes", [])}
    need(mine == theirs, "derivation: the listed changes differ from the re-derived ones: "
         f"missing {sorted(mine - theirs)}, extra {sorted(theirs - mine)}")

    p2 = Policy(s2)
    obs = cert.get("observations") or []
    under, req_fail, pres_fail, req_total, pres_total = 0, 0, 0, 0, 0
    for box, norms in cells:
        os_ = [x for x in obs if box.holds(x["situation"])]
        if len(os_) < trials:
            under += 1
            continue
        for o in os_:
            sit = o["situation"]
            facts = {f: sit[f] for f in s1.get("facts", {})}
            after = {}
            for c in o["after"]:
                for k in steps:
                    st_ = p2.steps.get(k)
                    if st_ and st_[1]["tool"] == c["tool"] and k not in after:
                        after[k] = c
            for k in steps:
                nn = p2.norm(k, facts)[0] if k in p2.steps else {"modality": "F", "call": None}
                ok = permits(nn, after.get(k))
                if _sym_same(norms[0][k], norms[1][k]):
                    pres_total += 1
                    pres_fail += not ok
                else:
                    req_total += 1
                    req_fail += not ok
            if (cert.get("provenance") or {}).get("agent_kind") == "model":
                raw = o.get("raw") or {}
                for side in ("before", "after"):
                    for c in o[side]:
                        need(parse_raw(raw.get(side, ""), c["tool"]) is not None,
                             f"provenance: observed call {c['tool']} not found in the raw reply")
    claim = cert.get("claim") or {}
    proved = under == 0 and req_fail == 0 and pres_fail == 0
    need(claim.get("every_cell_exercised") == (under == 0), f"claim: {under} regions exercised fewer than {trials} times")
    need(claim.get("every_required_change_happened") == (req_fail == 0),
         f"claim: {req_fail} of {req_total} required changes did not happen")
    need(claim.get("every_unrelated_obligation_held") == (pres_fail == 0),
         f"claim: {pres_fail} of {pres_total} unaffected constraints were not preserved")
    need(claim.get("proved") == proved, "claim: 'proved' disagrees with the re-derivation")
    need(claim.get("authenticated") == authenticated and claim.get("authorized") == authorized,
         "claim: authentication disagrees")
    need(claim.get("verified") == (proved and authorized), "claim: 'verified' disagrees")
    bound = (1 - (1 - conf) ** (1 / trials)) if proved else 1.0
    need(abs(float(claim.get("per_region_violation_bound", -1)) - bound) < 1e-12, "claim: violation bound disagrees")
    notes.append(f"re-derived {len(cells)} regions in scope independently; {req_total} required changes and "
                 f"{pres_total} unaffected constraints judged; {trials} trial(s) per region")
    verdict_line = ("VERIFIED CLAIM: compliant across an authenticated, authorized transition within scope"
                    if proved and authorized else
                    "the certificate's claim is NOT a full verification (it says so, and that is what was checked)")
    notes.append(verdict_line)
    for asm in sc.get("assumptions") or []:
        notes.append("scope assumption: " + asm)
    if (cert.get("provenance") or {}).get("agent_kind") != "model":
        notes.append("agent is scripted, not a model")
    return _verdict(cert, problems, notes)


def _verdict(cert, problems, notes):
    return {"verdict": "VERIFIED" if not problems else "REJECTED", "kind": cert.get("kind"),
            "step": cert.get("step"), "digest": cert.get("digest"), "problems": problems, "notes": notes,
            "claim": cert.get("claim")}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    paths = [a for a in argv if a != "--json"]
    if not paths:
        print(__doc__)
        return 2
    results = []
    for p in paths:
        try:
            with open(p) as f:
                cert = json.load(f)
            r = check_run(cert) if str(cert.get("schema", "")).startswith("contradish.run_certificate/") else check(cert)
        except Exception as exc:  # malformed input is a rejection, not a crash
            r = {"verdict": "REJECTED", "problems": [f"could not check: {exc!r}"], "notes": []}
        r["file"] = p
        results.append(r)
    if as_json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"{r['verdict']}  {r['file']}")
            if r.get("claim"):
                print(f"  claim: {r['claim']}")
            for x in r["problems"]:
                print(f"  problem: {x}")
            for x in r["notes"]:
                print(f"  note: {x}")
    return 0 if all(r["verdict"] == "VERIFIED" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
