"""
contradish/symbolic.py -- the full set of warranted changes between versions
of a policy, stated as conditions and proved complete.

Sampling situations can miss a region where the policy decides differently.
This module does not sample. It partitions the whole situation space into
CELLS, and within each cell every decision every version makes is constant.
It then classifies each cell. Two things together make that a proof of
completeness: the cells cover the space, and the classification really is
constant inside each one.

THE DECIDABLE FRAGMENT
    finite facts    bool and enum: enumerated exhaustively
    numeric facts   int and number, with min and max: symbolic
    expressions     linear in the numeric facts once the finite facts and
                    params are fixed. `*` needs at least one constant factor
                    and `/` a constant divisor. `round` may appear only as
                    the outermost operator of a step argument (arguments are
                    rounded to cents on output anyway). Every comparison may
                    involve at most ONE numeric fact.
    A program outside the fragment raises OutsideFragment, naming the
    expression. Nothing is approximated silently.

HOW THE CELLS ARE FOUND (canonical, so an independent implementation
derives the same ones)
    For each assignment of the finite facts:
      roots[x] = {} for every numeric fact x
      repeat:
        cells = every product of ELEMENTARY intervals of each numeric fact:
                the points in roots[x] ∪ {min, max}, and the open intervals
                between consecutive points (an int interval with no integer
                in it is dropped)
        evaluate every step of every version in every cell; whenever a
        comparison a·x + b ⋚ 0 has its root r = −b/a inside a cell's
        interval for x (and the interval is not the single point r),
        add r to roots[x]
      until a pass adds no root.
    Inside a cell, every comparison has a constant truth value. That holds
    by construction: a comparison that could flip inside the cell would have
    added a root. So every guard, and therefore every modality, is constant,
    and every argument is a fixed linear function of the numeric facts.

CLASSIFYING A CELL, per step, between version A and version B
    modality: compared directly (constant in the cell).
    arguments: two linear functions are identical iff their coefficients
    are equal. Identical means unchanged everywhere in the cell. That is a
    proof. Different means they differ on a dense subset of the cell. A
    witness situation where the rounded values differ is searched for and
    recorded; if none is found the argument is marked "undetermined" rather
    than guessed.

The output is the full set of warranted changes as conditions:

    obligation gained / lost
    permission granted / revoked      (tracked separately from obligations)
    content changed                   (allowed before and after, different call)

each with the cells (conditions on the facts) where it holds, plus the
cells where the step is proved unchanged, and whether that proof is
independent (the step reads nothing the new version redefined) or
coincidental.

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

from contradish.policy_program import (
    AMOUNT_TOLERANCE, Call, Norm, PolicyProgram, FORBIDDEN, OBLIGATORY, PERMITTED,
    _val_eq, canonical_json, calls_equal, norm_change, plain, round_half_up,
)

__all__ = [
    "OutsideFragment", "Lin", "Interval", "Cell", "decompose", "diff_versions", "VersionDiff",
    "norm_in_cell", "fact_signature", "regions",
]

NUM = (int, float, Fraction)


class OutsideFragment(ValueError):
    pass


class _Split(Exception):
    def __init__(self, fact: str, root: Fraction):
        self.fact, self.root = fact, root


# ─────────────────────────────────────────────────────────────────────────────
# Linear forms over numeric facts, with exact rational coefficients
# ─────────────────────────────────────────────────────────────────────────────

def _q(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise OutsideFragment("boolean used as a number")
    if isinstance(x, int):
        return Fraction(x)
    return Fraction(str(x))  # decimal literal, exactly


@dataclass(frozen=True)
class Lin:
    coef: tuple   # sorted ((fact, Fraction), ...), nonzero only
    const: Fraction

    @staticmethod
    def var(f: str) -> "Lin":
        return Lin(((f, Fraction(1)),), Fraction(0))

    @staticmethod
    def c(v) -> "Lin":
        return Lin((), _q(v))

    def is_const(self) -> bool:
        return not self.coef

    def facts(self) -> list:
        return [f for f, _ in self.coef]

    def _combine(self, other: "Lin", sign: int) -> "Lin":
        d = dict(self.coef)
        for f, a in other.coef:
            d[f] = d.get(f, Fraction(0)) + sign * a
        return Lin(tuple(sorted((f, a) for f, a in d.items() if a != 0)), self.const + sign * other.const)

    def __add__(self, o):
        return self._combine(o, 1)

    def __sub__(self, o):
        return self._combine(o, -1)

    def scale(self, k: Fraction) -> "Lin":
        if k == 0:
            return Lin((), Fraction(0))
        return Lin(tuple((f, a * k) for f, a in self.coef), self.const * k)

    def at(self, point: dict) -> Fraction:
        return self.const + sum((a * _q(point[f]) for f, a in self.coef), Fraction(0))

    def to_json(self) -> dict:
        return {"coef": {f: str(a) for f, a in self.coef}, "const": str(self.const)}

    def __str__(self) -> str:
        parts = [f"{_fmtq(a)}·{f}" for f, a in self.coef]
        if self.const != 0 or not parts:
            parts.append(_fmtq(self.const))
        return " + ".join(parts)


def _fmtq(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    f = float(q)
    return f"{f:.6g}"


def _as_lin(v) -> Optional[Lin]:
    if isinstance(v, Lin):
        return v
    if isinstance(v, NUM) and not isinstance(v, bool):
        return Lin.c(v)
    return None


def _norm_val(v):
    """Collapse a constant linear form to a plain number."""
    if isinstance(v, Lin) and v.is_const():
        q = v.const
        return int(q) if q.denominator == 1 else float(q)
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Intervals and cells
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Interval:
    """
    A point [lo] or an open interval (lo, hi) of one numeric fact. `step` is
    the fact's resolution (1 for int, e.g. 0.01 for an amount in dollars):
    only multiples of it are situations. None means continuous.
    """
    lo: Fraction
    hi: Fraction
    point: bool
    step: Optional[Fraction] = None

    @property
    def integer(self) -> bool:
        return self.step == 1

    def _on_grid(self, v: Fraction) -> bool:
        return self.step is None or (v / self.step).denominator == 1

    def contains(self, v) -> bool:
        v = _q(v)
        if not self._on_grid(v):
            return False
        return v == self.lo if self.point else (self.lo < v < self.hi)

    def strictly_inside(self, r: Fraction) -> bool:
        return (not self.point) and self.lo < r < self.hi

    def _grid_points(self):
        """First lattice point strictly above lo, and last strictly below hi."""
        k0 = math.floor(self.lo / self.step) + 1
        k1 = math.ceil(self.hi / self.step) - 1
        return k0, k1

    def rep_q(self) -> Fraction:
        if self.point:
            return self.lo
        if self.step is None:
            return (self.lo + self.hi) / 2
        k0, k1 = self._grid_points()
        return self.step * ((k0 + k1) // 2)

    def rep(self):
        v = self.rep_q()
        if v.denominator == 1:
            return int(v)
        return float(v)

    def empty(self) -> bool:
        if self.point:
            return not self._on_grid(self.lo)
        if self.lo >= self.hi:
            return True
        if self.step is not None:
            k0, k1 = self._grid_points()
            return k0 > k1
        return False

    def to_json(self) -> dict:
        if self.point:
            return {"eq": str(self.lo)}
        return {"gt": str(self.lo), "lt": str(self.hi)}

    def describe(self, name: str) -> str:
        if self.point:
            return f"{name} = {_fmtq(self.lo)}"
        return f"{_fmtq(self.lo)} < {name} < {_fmtq(self.hi)}"


def elementary(lo: Fraction, hi: Fraction, roots: set, step: Optional[Fraction]) -> list:
    pts = sorted({lo, hi} | {r for r in roots if lo < r < hi})
    out = []
    for i, p in enumerate(pts):
        out.append(Interval(p, p, True, step))
        if i + 1 < len(pts):
            out.append(Interval(p, pts[i + 1], False, step))
    return [iv for iv in out if not iv.empty()]


@dataclass
class Cell:
    finite: dict
    box: dict            # numeric fact -> Interval

    def rep(self) -> dict:
        s = dict(self.finite)
        for f, iv in self.box.items():
            s[f] = iv.rep()
        return s

    def contains(self, situation: dict) -> bool:
        for f, v in self.finite.items():
            if situation.get(f) != v:
                return False
        for f, iv in self.box.items():
            if f not in situation or not iv.contains(situation[f]):
                return False
        return True

    def describe(self) -> str:
        parts = [f"{f} = {v!r}" if not isinstance(v, bool) else (f if v else f"not {f}")
                 for f, v in sorted(self.finite.items())]
        parts += [iv.describe(f) for f, iv in sorted(self.box.items())]
        return " ∧ ".join(parts)

    def to_json(self) -> dict:
        return {"finite": dict(self.finite), "box": {f: iv.to_json() for f, iv in sorted(self.box.items())}}


def fact_signature(p: PolicyProgram) -> dict:
    out = {}
    for f, spec in p.facts.items():
        t = spec.get("type")
        if t in ("int", "number"):
            if "min" not in spec or "max" not in spec:
                raise OutsideFragment(f"numeric fact {f!r} needs min and max for symbolic analysis")
            step = Fraction(1) if t == "int" else (_q(spec["resolution"]) if "resolution" in spec else None)
            out[f] = (t, spec["min"], spec["max"], step)
        elif t == "bool":
            out[f] = ("bool",)
        elif t == "enum":
            out[f] = ("enum", tuple(spec["values"]))
        else:
            raise OutsideFragment(f"fact {f!r} has unsupported type {t!r}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Symbolic evaluation inside a cell
# ─────────────────────────────────────────────────────────────────────────────

def _sign_in_cell(L: Lin, cell: Cell) -> int:
    """Sign of L on the cell, or raise _Split if it is not constant there."""
    if L.is_const():
        return (L.const > 0) - (L.const < 0)
    if len(L.coef) > 1:
        raise OutsideFragment(f"comparison involves more than one numeric fact: {L} ⋚ 0")
    (f, a), = L.coef
    iv = cell.box[f]
    r = -L.const / a
    if iv.point:
        v = a * iv.lo + L.const
        return (v > 0) - (v < 0)
    if iv.strictly_inside(r):
        raise _Split(f, r)
    v = a * iv.rep_q() + L.const
    return (v > 0) - (v < 0)


def _cmp(op: str, x, y, cell: Cell):
    lx, ly = _as_lin(x), _as_lin(y)
    if lx is None or ly is None:
        if isinstance(x, Lin) or isinstance(y, Lin):
            raise OutsideFragment(f"comparing a number with a non-number: {op}")
        if op == "==":
            return _val_eq(x, y)
        if op == "!=":
            return not _val_eq(x, y)
        return {"<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]
    sg = _sign_in_cell(lx - ly, cell)
    return {"<": sg < 0, "<=": sg <= 0, ">": sg > 0, ">=": sg >= 0, "==": sg == 0, "!=": sg != 0}[op]


def seval(p: PolicyProgram, e: Any, cell: Cell, reads: tuple, top_arg: bool = False):
    names, clauses = reads
    if isinstance(e, bool) or e is None:
        return e
    if isinstance(e, NUM):
        return Lin.c(e)
    if isinstance(e, str):
        names.add(e)
        cid = p.owner.get(e)
        if cid is None:
            if e in cell.finite:
                return cell.finite[e]
            iv = cell.box[e]
            return Lin.c(iv.lo) if iv.point else Lin.var(e)
        clauses.add(cid)
        if e in p.params:
            v = p.params[e]
            return Lin.c(v) if isinstance(v, NUM) and not isinstance(v, bool) else v
        return seval(p, p.defs[e], cell, reads)
    op, args = e[0], e[1:]
    if op == "q":
        return args[0]
    if op == "and":
        for a in args:
            if not _truth(seval(p, a, cell, reads)):
                return False
        return True
    if op == "or":
        for a in args:
            if _truth(seval(p, a, cell, reads)):
                return True
        return False
    if op == "not":
        return not _truth(seval(p, args[0], cell, reads))
    if op == "if":
        return seval(p, args[1] if _truth(seval(p, args[0], cell, reads)) else args[2], cell, reads)
    if op == "round":
        if not top_arg:
            v = _as_lin(seval(p, args[0], cell, reads))
            if v is not None and v.is_const():
                return Lin.c(round_half_up(v.const, int(args[1]) if len(args) > 1 else 0))
            raise OutsideFragment("round() is allowed only as the outermost operator of a step argument")
        return seval(p, args[0], cell, reads)
    vals = [seval(p, a, cell, reads) for a in args]
    if op in ("==", "!=", "<", "<=", ">", ">="):
        return _cmp(op, vals[0], vals[1], cell)
    if op == "in":
        if any(isinstance(v, Lin) and not v.is_const() for v in vals):
            raise OutsideFragment("`in` over a numeric fact")
        vv = [_norm_val(v) for v in vals]
        return any(_val_eq(vv[0], x) for x in vv[1:])
    lins = [_as_lin(v) for v in vals]
    if any(l is None for l in lins):
        raise OutsideFragment(f"arithmetic on a non-number: {op}")
    if op == "+":
        out = Lin.c(0)
        for l in lins:
            out = out + l
        return out
    if op == "-":
        if len(lins) == 1:
            return lins[0].scale(Fraction(-1))
        out = lins[0]
        for l in lins[1:]:
            out = out - l
        return out
    if op == "*":
        out = Lin.c(1)
        for l in lins:
            if out.is_const():
                out = l.scale(out.const)
            elif l.is_const():
                out = out.scale(l.const)
            else:
                raise OutsideFragment("product of two numeric facts")
        return out
    if op == "/":
        if not lins[1].is_const() or lins[1].const == 0:
            raise OutsideFragment("division by a non-constant")
        return lins[0].scale(1 / lins[1].const)
    if op in ("min", "max"):
        best = lins[0]
        for l in lins[1:]:
            sg = _sign_in_cell(l - best, cell)
            if (op == "max" and sg > 0) or (op == "min" and sg < 0):
                best = l
        return best
    raise OutsideFragment(f"operator {op!r}")


def _truth(v) -> bool:
    if isinstance(v, Lin):
        if not v.is_const():
            raise OutsideFragment("a numeric expression used as a condition")
        return v.const != 0
    return bool(v)


@dataclass
class SymNorm:
    modality: str
    tool: Optional[str]
    args: dict                       # arg -> Lin | str | bool | None
    names: frozenset

    def call_at(self, point: dict) -> Optional[Call]:
        if self.modality == FORBIDDEN:
            return None
        out = {}
        for a, v in self.args.items():
            if isinstance(v, Lin):
                out[a] = plain(round_half_up(v.at(point), 2))
            elif isinstance(v, (int, float, Fraction)) and not isinstance(v, bool):
                out[a] = plain(round_half_up(v, 2))
            else:
                out[a] = v
        return Call.of(self.tool, out)

    def norm_at(self, point: dict) -> Norm:
        return Norm(self.modality, self.call_at(point))

    def to_json(self) -> dict:
        return {"modality": self.modality, "tool": self.tool,
                "args": {a: (v.to_json() if isinstance(v, Lin) else v) for a, v in sorted(self.args.items())}}


def norm_in_cell(p: Optional[PolicyProgram], sid: str, cell: Cell) -> SymNorm:
    """The step's norm in the cell (constant there); raises _Split if the cell must be refined."""
    if p is None or sid not in p.step_ids():
        return SymNorm(FORBIDDEN, None, {}, frozenset())
    _, cid, s = p.step(sid)
    names, clauses = {"step:" + sid}, {cid}
    reads = (names, clauses)
    if _truth(seval(p, s.get("when", True), cell, reads)):
        m = OBLIGATORY
    elif "allowed_when" in s and _truth(seval(p, s["allowed_when"], cell, reads)):
        m = PERMITTED
    else:
        return SymNorm(FORBIDDEN, None, {}, frozenset(names))
    args = {}
    for a, e in sorted((s.get("args") or {}).items()):
        v = seval(p, e, cell, reads, top_arg=True)
        args[a] = v if isinstance(v, Lin) and not v.is_const() else _norm_val(v)
    return SymNorm(m, s["tool"], args, frozenset(names))


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition
# ─────────────────────────────────────────────────────────────────────────────

def _finite_assignments(sig: dict) -> list:
    keys = [f for f, t in sig.items() if t[0] in ("bool", "enum")]
    axes = [[False, True] if sig[f][0] == "bool" else list(sig[f][1]) for f in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]


def decompose(programs: list, steps: Optional[list] = None, max_cells: int = 200000) -> list:
    """
    Canonical cells over which every step of every program has a constant
    norm. Returns [(Cell, {program_index: {step: SymNorm}})].
    """
    sig = fact_signature(programs[0])
    for q in programs[1:]:
        if fact_signature(q) != sig:
            raise OutsideFragment("the versions describe situations with different facts")
    if steps is None:
        steps = []
        for q in programs:
            for k in q.step_ids():
                if k not in steps:
                    steps.append(k)
    numeric = [f for f, t in sig.items() if t[0] in ("int", "number")]
    out = []
    for fin in _finite_assignments(sig):
        roots = {f: set() for f in numeric}
        while True:
            boxes = itertools.product(*[
                elementary(_q(sig[f][1]), _q(sig[f][2]), roots[f], sig[f][3]) for f in numeric])
            added = False
            results = []
            for ivs in boxes:
                cell = Cell(dict(fin), dict(zip(numeric, ivs)))
                try:
                    norms = {i: {k: norm_in_cell(q, k, cell) for k in steps} for i, q in enumerate(programs)}
                except _Split as sp:
                    roots[sp.fact].add(sp.root)
                    added = True
                    continue
                results.append((cell, norms))
            if not added:
                out.extend(results)
                break
        if len(out) > max_cells:
            raise OutsideFragment(f"more than {max_cells} cells")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Comparing two versions
# ─────────────────────────────────────────────────────────────────────────────

def _witness(a: SymNorm, b: SymNorm, cell: Cell, arg: str):
    """A situation in the cell where the rounded argument values differ, or None."""
    candidates = [cell.rep()]
    numeric = list(cell.box)
    for f in numeric:
        iv = cell.box[f]
        if iv.point:
            continue
        span = iv.hi - iv.lo
        for t in (Fraction(1, 4), Fraction(3, 4), Fraction(1, 10), Fraction(9, 10)):
            pt = dict(cell.rep())
            v = iv.lo + span * t
            if iv.step is not None:
                v = iv.step * math.floor(v / iv.step)
                if not iv.contains(v):
                    continue
            pt[f] = int(v) if v.denominator == 1 else float(v)
            candidates.append(pt)
    for pt in candidates:
        ca, cb = a.call_at(pt), b.call_at(pt)
        if ca is not None and cb is not None:
            va, vb = dict(ca.args).get(arg), dict(cb.args).get(arg)
            if not _val_eq(va, vb):
                return pt
    return None


def _same_value_sym(x, y) -> bool:
    if isinstance(x, Lin) or isinstance(y, Lin):
        lx, ly = _as_lin(x), _as_lin(y)
        if lx is None or ly is None:
            return False
        d = lx - ly
        return d.is_const() and abs(d.const) <= Fraction(AMOUNT_TOLERANCE) / 10
    return _val_eq(x, y)


def classify(a: SymNorm, b: SymNorm, cell: Cell, redefined: set) -> dict:
    """Per-step classification of one cell between two versions."""
    na = Norm(a.modality, Call.of(a.tool, {}) if a.tool else None)
    nb = Norm(b.modality, Call.of(b.tool, {}) if b.tool else None)
    ch = norm_change(na, nb)
    ch["content"] = None
    differing, undetermined, witness = [], [], None
    if a.modality != FORBIDDEN and b.modality != FORBIDDEN:
        if a.tool != b.tool or set(a.args) != set(b.args):
            differing = sorted(set(a.args) | set(b.args))
        else:
            for arg in sorted(a.args):
                if not _same_value_sym(a.args[arg], b.args[arg]):
                    # Different linear functions: a change on a dense part of the cell.
                    differing.append(arg)
                    w = _witness(a, b, cell, arg)
                    if w is None:
                        undetermined.append(arg)   # no grid point found where the cents differ
                    else:
                        witness = witness or w
        if differing:
            ch["content"] = "changed"
    changed = bool(ch["obligation"] or ch["permission"] or ch["content"])
    if changed:
        kind = "change"
    else:
        kind = "unchanged:" + ("coincidental" if (a.names & redefined) else "independent")
    return {"kind": kind, "obligation": ch["obligation"], "permission": ch["permission"],
            "content": ch["content"], "differing_args": differing, "undetermined_args": undetermined,
            "modality": [a.modality, b.modality], "witness": witness}


@dataclass
class VersionDiff:
    before: PolicyProgram
    after: PolicyProgram
    steps: list
    cells: list               # [(Cell, {0: {step: SymNorm}, 1: {...}}, {step: classification})]
    redefined: set

    def summary(self) -> dict:
        per = {k: {"obligation_gained": 0, "obligation_lost": 0, "permission_granted": 0,
                   "permission_revoked": 0, "content_changed": 0, "unchanged_independent": 0,
                   "unchanged_coincidental": 0, "undetermined": 0} for k in self.steps}
        for _, _, cls in self.cells:
            for k, c in cls.items():
                d = per[k]
                if c["obligation"]:
                    d["obligation_" + c["obligation"]] += 1
                if c["permission"]:
                    d["permission_" + c["permission"]] += 1
                if c["content"]:
                    d["content_changed"] += 1
                if c["kind"] == "unchanged:independent":
                    d["unchanged_independent"] += 1
                elif c["kind"] == "unchanged:coincidental":
                    d["unchanged_coincidental"] += 1
                elif c["kind"] == "undetermined":
                    d["undetermined"] += 1
        return {
            "cells": len(self.cells),
            "redefined": sorted(self.redefined),
            "complete": True,
            "undetermined_cases": sum(v["undetermined"] for v in per.values()),
            "per_step": per,
            "steps_that_change": sorted(k for k, v in per.items()
                                        if any(v[x] for x in ("obligation_gained", "obligation_lost",
                                                              "permission_granted", "permission_revoked",
                                                              "content_changed"))),
        }

    def changes(self) -> list:
        """The full set of warranted changes, as conditions: one row per (cell, step) that changes."""
        rows = []
        for cell, norms, cls in self.cells:
            for k, c in cls.items():
                if c["kind"] == "change":
                    rows.append({"step": k, "condition": cell.describe(), "cell": cell.to_json(),
                                 "obligation": c["obligation"], "permission": c["permission"],
                                 "content": c["content"], "differing_args": c["differing_args"],
                                 "before": norms[0][k].to_json(), "after": norms[1][k].to_json(),
                                 "witness": c["witness"] or cell.rep()})
        return rows

    def to_json(self) -> dict:
        return {
            "schema": "contradish.version_diff/1.0",
            "before_digest": self.before.digest(),
            "after_digest": self.after.digest(),
            "summary": self.summary(),
            "changes": self.changes(),
            "cells": [{"cell": c.to_json(), "condition": c.describe(), "rep": c.rep(),
                       "steps": cls} for c, _, cls in self.cells],
        }


def redefined_between(a: PolicyProgram, b: PolicyProgram) -> set:
    out = a.changed_names(b)
    return out


def diff_versions(before: PolicyProgram, after: PolicyProgram) -> VersionDiff:
    """The complete, condition-level difference in norms between two versions."""
    steps = []
    for q in (before, after):
        for k in q.step_ids():
            if k not in steps:
                steps.append(k)
    red = redefined_between(before, after)
    cells = []
    for cell, norms in decompose([before, after], steps):
        cls = {k: classify(norms[0][k], norms[1][k], cell, red) for k in steps}
        cells.append((cell, norms, cls))
    return VersionDiff(before, after, steps, cells, red)


# ─────────────────────────────────────────────────────────────────────────────
# Readable regions: merge adjacent cells with the same classification
# ─────────────────────────────────────────────────────────────────────────────

def _span(ivs: list):
    """Merge elementary intervals of one fact into maximal contiguous spans (lo, lo_closed, hi, hi_closed)."""
    ivs = sorted(set((iv.lo, iv.hi, iv.point) for iv in ivs), key=lambda t: (t[0], 0 if t[2] else 1))
    spans = []
    for lo, hi, point in ivs:
        if spans:
            slo, slc, shi, shc = spans[-1]
            if point and lo == shi and not shc:          # (a, p) then [p]
                spans[-1] = (slo, slc, lo, True)
                continue
            if not point and lo == shi and shc:          # [.., p] then (p, b)
                spans[-1] = (slo, slc, hi, False)
                continue
        spans.append((lo, point, hi, point))
    return spans


def _desc_span(f: str, sp, full) -> str:
    lo, lc, hi, hc = sp
    if lo == hi:
        return f"{f} = {_fmtq(lo)}"
    left = "" if (lo == full[0] and lc) else f"{_fmtq(lo)} {'≤' if lc else '<'} "
    right = "" if (hi == full[1] and hc) else f" {'≤' if hc else '<'} {_fmtq(hi)}"
    if not left and not right:
        return ""
    return f"{left}{f}{right}"


def regions(diff: "VersionDiff", step: str) -> list:
    """
    The changes to one step as a short list of readable conditions. Cells
    with the same classification and the same finite facts are merged along
    each numeric fact where they are contiguous.
    """
    sig = fact_signature(diff.before)
    numeric = [f for f, t in sig.items() if t[0] in ("int", "number")]
    groups: dict = {}
    for cell, _, cls in diff.cells:
        c = cls[step]
        if c["kind"] != "change":
            continue
        key = (c["obligation"], c["permission"], c["content"], tuple(c["differing_args"]))
        groups.setdefault(key, []).append(cell)
    out = []
    for key, cells in groups.items():
        # merge numeric intervals per finite assignment, one fact at a time
        rows = [(tuple(sorted(c.finite.items())), {f: [c.box[f]] for f in numeric}) for c in cells]
        for f in numeric:
            merged: dict = {}
            for fin, box in rows:
                k = (fin, tuple((g, tuple(sorted((iv.lo, iv.hi, iv.point) for iv in box[g])))
                                for g in numeric if g != f))
                merged.setdefault(k, (fin, {g: list(box[g]) for g in numeric if g != f}, []))[2].extend(box[f])
            rows = []
            for (fin, rest, ivs) in merged.values():
                b = dict(rest)
                b[f] = ivs
                rows.append((fin, b))
        # describe numeric spans, then drop finite facts that do not matter
        items = []
        for fin, box in rows:
            num = []
            for f in numeric:
                full = (_q(sig[f][1]), _q(sig[f][2]))
                for sp in _span(box[f]):
                    d = _desc_span(f, sp, full)
                    if d:
                        num.append(d)
            items.append((dict(fin), tuple(num)))
        finite_keys = [f for f, t in sig.items() if t[0] in ("bool", "enum")]
        changed = True
        while changed:
            changed = False
            for f in finite_keys:
                domain = [False, True] if sig[f][0] == "bool" else list(sig[f][1])
                groups2: dict = {}
                for fin, num in items:
                    if f not in fin:
                        continue
                    k = (tuple(sorted((a, b) for a, b in fin.items() if a != f)), num)
                    groups2.setdefault(k, set()).add(fin[f])
                for (rest, num), vals in groups2.items():
                    if set(domain) <= vals:
                        items = [(fin, nm) for fin, nm in items
                                 if not (nm == num and f in fin and
                                         tuple(sorted((a, b) for a, b in fin.items() if a != f)) == rest)]
                        items.append((dict(rest), num))
                        changed = True
                if changed:
                    break
        texts = []
        for fin, num in items:
            parts = [(f"{k} = {v!r}" if not isinstance(v, bool) else (k if v else f"not {k}"))
                     for k, v in sorted(fin.items())]
            parts += list(num)
            texts.append(" ∧ ".join(parts) if parts else "always")
        label = []
        if key[0]:
            label.append(f"obligation {key[0]}")
        if key[1]:
            label.append(f"permission {key[1]}")
        if key[2]:
            label.append(f"content changed ({', '.join(key[3])})")
        out.append({"change": "; ".join(label), "conditions": sorted(set(texts)), "cells": len(cells)})
    return out
