"""
contradish/policy_source.py -- readable conditions for policy programs.

A policy program's definitions, guards and arguments are expressions. The canonical form is JSON lists
(["and", "eligible", [">=", "days_since_fill", "refill_day"]]); this module lets a policy file write them as text:

    "identity_verified and days_since_fill >= refill_day and not controlled_ii"
    "supply_days * early_refill_pct / 100"
    "'schedule_ii' if controlled_ii else 'no_refills'"
    "max(0, price - fee - shipping)"
    "category in ('apparel', 'electronics')"

Grammar (Python-like):  A if C else B | or | and | not | == != < <= > >= in | + - | * / | unary - | f(x, ...)
with f in min, max, round; names; numbers; 'quoted' or "quoted" strings; true / false; parentheses.

A string with nothing but a name in it ("eligible") is a name, as in the canonical form, so existing programs are
unchanged: `compile_spec` only rewrites strings that contain an operator, a space, a quote or a parenthesis, and the
program stores the compiled form, so a program's digest does not depend on how its conditions were written.
"""

from __future__ import annotations

import copy
import re
from typing import Any

__all__ = ["parse", "compile_spec", "SourceError"]


class SourceError(ValueError):
    pass


_TOKEN = re.compile(r"""\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<str>'[^']*'|"[^"]*")|(?P<op>==|!=|<=|>=|[<>+\-*/(),])|"""
                    r"""(?P<name>[A-Za-z_][A-Za-z0-9_:.]*))""")
_KEYWORDS = {"and", "or", "not", "if", "else", "in", "true", "false", "True", "False"}
_FUNCS = {"min", "max", "round"}


def _tokens(text: str) -> list:
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise SourceError(f"cannot read {text!r} at position {pos}: {text[pos:pos + 12]!r}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        if kind == "num":
            out.append(("num", float(val) if "." in val else int(val)))
        elif kind == "str":
            out.append(("str", val[1:-1]))
        elif kind == "name" and val in _KEYWORDS:
            out.append(("kw", val))
        else:
            out.append((kind, val))
    out.append(("end", None))
    return out


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = _tokens(text)
        self.i = 0

    def peek(self, kind=None, val=None):
        k, v = self.toks[self.i]
        return (kind is None or k == kind) and (val is None or v == val)

    def take(self, kind=None, val=None):
        if not self.peek(kind, val):
            k, v = self.toks[self.i]
            raise SourceError(f"in {self.text!r}: expected {val or kind}, found {v!r}")
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse(self):
        e = self.ternary()
        self.take("end")
        return e

    def ternary(self):
        a = self.or_()
        if self.peek("kw", "if"):
            self.take()
            cond = self.or_()
            self.take("kw", "else")
            b = self.ternary()
            return ["if", cond, a, b]
        return a

    def _chain(self, op, sub):
        items = [sub()]
        while self.peek("kw", op):
            self.take()
            items.append(sub())
        return items[0] if len(items) == 1 else [op, *items]

    def or_(self):
        return self._chain("or", self.and_)

    def and_(self):
        return self._chain("and", self.not_)

    def not_(self):
        if self.peek("kw", "not"):
            self.take()
            return ["not", self.not_()]
        return self.cmp()

    def cmp(self):
        a = self.sum()
        for op in ("==", "!=", "<=", ">=", "<", ">"):
            if self.peek("op", op):
                self.take()
                return [op, a, self.sum()]
        if self.peek("kw", "in"):
            self.take()
            self.take("op", "(")
            items = [self.sum()]
            while self.peek("op", ","):
                self.take()
                items.append(self.sum())
            self.take("op", ")")
            return ["in", a, *items]
        return a

    def sum(self):
        a = self.term()
        while self.peek("op", "+") or self.peek("op", "-"):
            op = self.take()[1]
            a = [op, a, self.term()]
        return a

    def term(self):
        a = self.unary()
        while self.peek("op", "*") or self.peek("op", "/"):
            op = self.take()[1]
            a = [op, a, self.unary()]
        return a

    def unary(self):
        if self.peek("op", "-"):
            self.take()
            x = self.unary()
            return -x if isinstance(x, (int, float)) and not isinstance(x, bool) else ["-", 0, x]
        return self.atom()

    def atom(self):
        k, v = self.toks[self.i]
        if k == "num":
            self.i += 1
            return v
        if k == "str":
            self.i += 1
            return ["q", v]
        if k == "kw" and v in ("true", "True"):
            self.i += 1
            return True
        if k == "kw" and v in ("false", "False"):
            self.i += 1
            return False
        if k == "op" and v == "(":
            self.i += 1
            e = self.ternary()
            self.take("op", ")")
            return e
        if k == "name":
            self.i += 1
            if self.peek("op", "("):
                if v not in _FUNCS:
                    raise SourceError(f"in {self.text!r}: unknown function {v!r} (allowed: {sorted(_FUNCS)})")
                self.take()
                args = [self.ternary()]
                while self.peek("op", ","):
                    self.take()
                    args.append(self.ternary())
                self.take("op", ")")
                return [v, *args]
            return v
        raise SourceError(f"in {self.text!r}: unexpected {v!r}")


_PLAIN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_:.]*$")


def parse(text: str) -> Any:
    """One readable expression -> the canonical JSON expression."""
    return _Parser(text).parse()


def _expr(x: Any, where: str) -> Any:
    if isinstance(x, str) and not _PLAIN_NAME.match(x):
        try:
            return parse(x)
        except SourceError as e:
            raise SourceError(f"{where}: {e}") from None
    return x


def compile_spec(spec: dict) -> dict:
    """A copy of the program spec with every readable expression compiled; canonical expressions pass through."""
    out = copy.deepcopy(spec)
    for cid, c in (out.get("clauses") or {}).items():
        for name, v in list((c.get("define") or {}).items()):
            c["define"][name] = _expr(v, f"clause {cid} definition {name}")
        for sid, s in (c.get("steps") or {}).items():
            for key in ("when", "allowed_when"):
                if key in s:
                    s[key] = _expr(s[key], f"step {sid} {key}")
            for a, v in list((s.get("args") or {}).items()):
                s["args"][a] = _expr(v, f"step {sid} arg {a}")
    return out
