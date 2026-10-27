"""
contradish/transform.py -- a verified interface for changing what an agent remembers without changing what it is
authorized to do.

Agent frameworks transform memory all the time: summarize a conversation, consolidate notes, forget old entries,
merge duplicates, rewrite a case file, roll authority into "may act until October 20". Each transformation is safe or
not by one criterion this module makes exact:

    A transformation of memory M into M' is AUTHORIZATION-PRESERVING, relative to a policy, if on every future the
    policy permits (every day of the horizon, under the version in force that day; clock facts advancing; any kept
    grant revoked by its issuer or the root), the actions the agent is required, permitted and forbidden to take
    under M' are exactly those under M.

A memory is the format contradish.memory_compiler produces (contradish.memory/1): facts kept exactly, as a range, a
set of values, or dropped; delegated authority kept as chains of (grant id, issuer) with their interval in force (a
chain with no links is authority nobody can revoke). Under a memory, the agent's behaviour on a day, under a future,
is the set of call sequences the policy requires over every situation the memory is consistent with. One element:
the memory decides. More: it cannot.

    check_transform(policy, before, after, as_of)  ->  verdict (contradish.transform_verdict/1)

        SAFE        identical on every future
        UNSAFE      with every violation, each one of:
          authority_expansion   after the change, an irreversible action is taken that the memory before forbids
          withholding           a call the memory before requires is no longer made
          changed_action        a different decision otherwise (arguments changed, a new reversible call)
          indeterminacy         the memory before decides; after the change it cannot
          fabrication           the memory before cannot decide; after the change it decides (certainty the
                                history never supported)
        and a counterexample: the day, the future (which grant was revoked, if any), the step, what the agent does
        before and after.

    repair(policy, before, after, as_of)  ->  the largest part of the requested change that is safe (each fact and
        each delegated fact either changed as requested or kept as before), with its verdict.

    MemoryGuard  -- the adapter a framework wraps around its memory store: propose a transformation, get the verdict,
        commit only safe ones (enforce), or the repaired one, or log only (shadow); every decision is a receipt that
        contradish/memory_check.py re-derives and that the signed audit ledger can hold.

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import itertools
import json
from typing import Any, Callable, Optional

from contradish.memory_compiler import _World, _axis, _canon, _d, _digest, _iso, _version_on, load_memory_policy

__all__ = ["check_transform", "repair", "compress", "MemoryGuard", "serve", "behaviour_sets", "VERDICT_SCHEMA", "MEMORY_SCHEMA"]

VERDICT_SCHEMA = "contradish.transform_verdict/1"
MEMORY_SCHEMA = "contradish.memory/1"
KINDS = ("authority_expansion", "withholding", "changed_action", "indeterminacy", "fabrication")


def _allowed(v, c) -> bool:
    if c is None or c.get("drop"):
        return True
    if "values" in c:
        return v in c["values"]
    if "value" in c:
        return v == c["value"]
    return (c.get("lo") is None or v >= c["lo"]) and (c.get("hi") is None or v <= c["hi"])


def _chains_ok(chains: list, day: str, revoked: dict) -> bool:
    """In force on `day` given the chains and the future's revocations {grant id: effective day}."""
    for c in chains:
        if not (c["from"] <= day and (c["until"] is None or day < c["until"])):
            continue
        if any(l["id"] in revoked and revoked[l["id"]] <= day for l in c["links"]):
            continue
        return True
    return False


def _futures(before: dict, after: dict, w: _World) -> list:
    """No revocation; each kept grant revoked by its issuer on the first day and on each version change; and every
    pair of kept grants revoked on the first day."""
    links = {}
    for m in (before, after):
        for a in (m.get("authority") or {}).values():
            for c in a.get("chains") or []:
                for l in c["links"]:
                    links[l["id"]] = l["issuer"]
    days = [w.as_of] + [t[0] for t in w.tl[1:]]
    out = [{"label": "no revocation", "revoked": {}}]
    for gid, issuer in sorted(links.items()):
        for d in days:
            out.append({"label": f"{issuer} revokes {gid} effective {d}", "revoked": {gid: d}})
    for (g1, i1), (g2, i2) in itertools.combinations(sorted(links.items()), 2):
        out.append({"label": f"{i1} revokes {g1} and {i2} revokes {g2}, effective {w.as_of}",
                    "revoked": {g1: w.as_of, g2: w.as_of}})
    return out


def _days(w: _World, worlds_clock: list) -> list:
    offs = {0, w.H} | set(w.eff) | {e - 1 for e in w.eff if e > 0}
    for v in worlds_clock:
        offs |= set(w.offsets(v))
    return sorted(o for o in offs if 0 <= o <= w.H)


def critical_days(w: _World, worlds: list) -> list:
    """Day offsets on which some behaviour can change for some situation in `worlds`: the first and last day,
    every version change (and the day before), and every day a clock fact crosses a threshold."""
    offs = {0, w.H} | set(w.eff) | {e - 1 for e in w.eff if e > 0}
    for s in worlds:
        offs |= set(w.offsets(s))
    return sorted(o for o in offs if 0 <= o <= w.H)


def _worlds(memory: dict, w: _World, ax: dict) -> list:
    """Every situation (on as_of) the memory is consistent with, over the policy's decisive points."""
    facts = memory.get("facts") or {}
    axes = []
    for f in w.p0.facts:
        if f in w.deleg:
            continue
        c = facts.get(f)
        vals = list(ax[f])
        if c and not c.get("drop"):
            pts = [c[k] for k in ("lo", "hi", "value") if c.get(k) is not None] + list(c.get("values") or [])
            shift = (_d(w.as_of) - _d(c["on"])).days if (f in w.clock and c.get("on")) else 0
            pts = [p + shift if isinstance(p, int) and not isinstance(p, bool) else p for p in pts]
            vals = sorted(set(vals) | {p for p in pts if p in ax[f] or not isinstance(p, int)}, key=lambda v: (str(type(v)), v))
            cc = dict(c)
            if shift:
                cc = dict(c, **{k: c[k] + shift for k in ("lo", "hi", "value") if c.get(k) is not None})
            vals = [v for v in vals if _allowed(v, cc)]
        axes.append((f, vals))
    names = [a for a, _ in axes]
    return [dict(zip(names, c)) for c in itertools.product(*[v for _, v in axes])]


def behaviour_sets(policy: Any, memory: dict, as_of: str, horizon: Optional[int] = None, futures=None, _w=None,
                   days: Optional[list] = None):
    """{(day, future label): set of behaviours} for one memory, on `days` (offsets; default: the memory's own
    critical days). To compare two memories, pass the union of both memories' critical days."""
    spec, updates, cfg = load_memory_policy(policy)
    w = _w or _World(spec, updates, cfg, as_of, int(horizon or cfg.get("horizon_days", 30)))
    ax = _axis(w)
    worlds = _worlds(memory, w, ax)
    futures = futures or [{"label": "no revocation", "revoked": {}}]
    if days is None:
        days = critical_days(w, worlds)
    out = {}
    cache = {}
    for F in futures:
        for o in days:
            day = _iso(_d(as_of) + _dt.timedelta(days=o))
            auth = {f: _chains_ok((memory.get("authority") or {}).get(f, {}).get("chains") or [], day, F["revoked"])
                    for f in w.deleg}
            ver = _version_on(w.tl, day)
            bs = set()
            for s in worlds:
                st = dict(w.advance(s, o), **auth)
                k = (ver[1], _canon(st))
                if k not in cache:
                    cache[k] = _canon({sid: (c.to_dict() if c else None) for sid, c in ver[2].run(st).items()})
                bs.add(cache[k])
            out[(day, F["label"])] = bs
    return out, w


def _classify(b: set, a: set, irreversible: set) -> Optional[str]:
    """How the behaviour under the memory after (a) differs from before (b), on one day under one future."""
    if a == b:
        return None
    if len(b) == 1:
        if len(a) > 1:
            return "indeterminacy"
        bb, aa = json.loads(next(iter(b))), json.loads(next(iter(a)))
        for sid, call in aa.items():
            if call is not None and call["tool"] in irreversible and bb.get(sid) != call:
                return "authority_expansion"
        for sid, call in bb.items():
            if call is not None and (aa.get(sid) is None or aa[sid]["tool"] != call["tool"]):
                return "withholding"                 # a call the memory before requires is no longer made
        return "changed_action"                      # the same calls, with different arguments, or new reversible ones
    # the memory before cannot decide this day: after the change it must not narrow what is possible either
    return "fabrication" if a < b else "changed_action"


def _diff(b: set, a: set) -> dict:
    if len(b) == 1 and len(a) == 1:
        bb, aa = json.loads(next(iter(b))), json.loads(next(iter(a)))
        steps = sorted(k for k in set(bb) | set(aa) if bb.get(k) != aa.get(k))
        return {"steps": steps, "before_does": {k: bb.get(k) for k in steps}, "after_does": {k: aa.get(k) for k in steps}}
    return {"before_outcomes": len(b), "after_outcomes": len(a)}


def check_transform(policy: Any, before: dict, after: dict, as_of: str, horizon: Optional[int] = None,
                    reason: str = "") -> dict:
    spec, updates, cfg = load_memory_policy(policy)
    H = int(horizon or cfg.get("horizon_days", 30))
    w = _World(spec, updates, cfg, as_of, H)
    futures = _futures(before, after, w)
    ax = _axis(w)
    days = critical_days(w, _worlds(before, w, ax) + _worlds(after, w, ax))
    sb, _ = behaviour_sets(policy, before, as_of, H, futures, w, days)
    sa, _ = behaviour_sets(policy, after, as_of, H, futures, w, days)
    assert set(sb) == set(sa)
    irreversible = set(spec.get("irreversible_tools") or [])
    violations = []
    order = {f["label"]: i for i, f in enumerate(futures)}        # simplest future first, then the earliest day
    for key in sorted(sb, key=lambda k: (order[k[1]], k[0])):
        b, a = sb[key], sa[key]
        kind = _classify(b, a, irreversible)
        if kind:
            violations.append({"kind": kind, "day": key[0], "future": key[1], **_diff(b, a)})
    first = {}
    for v in violations:
        first.setdefault(v["kind"], v)
    from contradish.policy_program import PolicyProgram
    body = {"schema": VERDICT_SCHEMA, "as_of": as_of, "horizon_days": H, "reason": reason,
            "policy_digest": _digest({"spec": spec, "updates": updates}),
            "policy": {"base": PolicyProgram(spec).spec, "updates": updates,
                       "irreversible_tools": sorted(irreversible)},
            "memory_config": cfg,
            "timeline": [{"from": t[0], "label": t[1], "applied": t[3], "spec": t[2].spec} for t in w.tl],
            "futures_checked": [f["label"] for f in futures],
            "before": before, "after": after,
            "before_digest": _digest(before), "after_digest": _digest(after),
            "verdict": "SAFE" if not violations else "UNSAFE",
            "violations_by_kind": {k: sum(v["kind"] == k for v in violations) for k in KINDS if any(v["kind"] == k for v in violations)},
            "counterexamples": [first[k] for k in KINDS if k in first],
            "checked": {"days_x_futures": len(sb), "violations": len(violations)}}
    body["digest"] = _digest(body)
    return body


def _components(m: dict) -> list:
    return [("facts", f) for f in (m.get("facts") or {})] + [("authority", f) for f in (m.get("authority") or {})]


def repair(policy: Any, before: dict, after: dict, as_of: str, horizon: Optional[int] = None) -> dict:
    """The largest part of the requested change that is safe: try each changed component (a fact, a delegated
    fact) alone; keep those that are safe alone; then check them together, reverting the last kept until safe."""
    changed = []
    for part, f in sorted(set(_components(before)) | set(_components(after))):
        if (before.get(part) or {}).get(f) != (after.get(part) or {}).get(f):
            changed.append((part, f))
    accepted, refused = [], []
    for part, f in changed:
        trial = copy.deepcopy(before)
        trial.setdefault(part, {})
        if f in (after.get(part) or {}):
            trial[part][f] = copy.deepcopy(after[part][f])
        else:
            trial[part].pop(f, None)
        v = check_transform(policy, before, trial, as_of, horizon)
        (accepted if v["verdict"] == "SAFE" else refused).append(((part, f), v))
    result = copy.deepcopy(before)
    for (part, f), _ in accepted:
        result.setdefault(part, {})
        if f in (after.get(part) or {}):
            result[part][f] = copy.deepcopy(after[part][f])
        else:
            result[part].pop(f, None)
    v = check_transform(policy, before, result, as_of, horizon, reason="repair")
    while v["verdict"] != "SAFE" and accepted:
        (part, f), _ = accepted.pop()
        refused.append(((part, f), v))
        result.setdefault(part, {})
        if f in (before.get(part) or {}):
            result[part][f] = copy.deepcopy(before[part][f])
        else:
            result[part].pop(f, None)
        v = check_transform(policy, before, result, as_of, horizon, reason="repair")
    return {"memory": result, "verdict": v,
            "applied": [f"{p}.{f}" for (p, f), _ in accepted],
            "refused": [{"component": f"{p}.{f}", "kinds": list(x["violations_by_kind"]),
                         "counterexample": (x["counterexamples"] or [None])[0]} for (p, f), x in refused]}


def compress(policy: Any, memory: dict, as_of: str, horizon: Optional[int] = None) -> dict:
    """Push compression past the compiler's per-fact memory: drop each fact, in turn, whenever the result is still
    authorization-preserving given everything else kept (a joint check, so a fact that matters only in situations
    the rest of the memory rules out is dropped). Returns the compressed memory, what was dropped, and the verdict
    of the whole compression against the original."""
    cur, dropped = copy.deepcopy(memory), []
    for f in sorted((memory.get("facts") or {})):
        if (cur["facts"][f] or {}).get("drop"):
            continue
        trial = copy.deepcopy(cur)
        trial["facts"][f] = {"drop": True}
        if check_transform(policy, cur, trial, as_of, horizon)["verdict"] == "SAFE":
            cur, dropped = trial, dropped + [f]
    return {"memory": cur, "dropped": dropped,
            "verdict": check_transform(policy, memory, cur, as_of, horizon, reason="compress")}


class MemoryGuard:
    """
    The adapter. Wrap whatever your framework uses as memory (a dict, a store, a summary) by giving the guard the
    memory in contradish.memory/1 form and two functions it can use to read and write your store.

        guard = MemoryGuard("policy.json", memory, as_of=lambda: today(), mode="enforce", ledger=IntegrityLedger(...))
        receipt = guard.propose(new_memory, reason="nightly summarization")
        guard.memory          # what the agent now remembers

    mode="enforce"   commit a SAFE change; for an UNSAFE one commit the repaired change (on_unsafe="repair", the
                     default) or nothing (on_unsafe="reject")
    mode="shadow"    commit what was proposed, and record what enforce would have done
    Every proposal returns a receipt (the verdict, what was committed, the memory digests before and after) and, with
    a ledger, appends it there.
    """

    def __init__(self, policy: Any, memory: dict, as_of: Callable[[], str], mode: str = "enforce",
                 on_unsafe: str = "repair", ledger=None, horizon: Optional[int] = None):
        if mode not in ("enforce", "shadow") or on_unsafe not in ("repair", "reject"):
            raise ValueError("mode is enforce|shadow; on_unsafe is repair|reject")
        self.policy, self.memory, self.as_of = policy, copy.deepcopy(memory), as_of
        self.mode, self.on_unsafe, self.ledger, self.horizon = mode, on_unsafe, ledger, horizon
        self.receipts = []

    def propose(self, new_memory: dict, reason: str = "") -> dict:
        day = self.as_of()
        v = check_transform(self.policy, self.memory, new_memory, day, self.horizon, reason)
        committed, action = new_memory, "committed as proposed"
        rep = None
        if v["verdict"] != "SAFE":
            if self.mode == "shadow":
                action = "committed as proposed (shadow: enforce would have " + (
                    "repaired it)" if self.on_unsafe == "repair" else "rejected it)")
            elif self.on_unsafe == "reject":
                committed, action = self.memory, "rejected"
            else:
                rep = repair(self.policy, self.memory, new_memory, day, self.horizon)
                committed, action = rep["memory"], "repaired: " + (", ".join(rep["applied"]) or "nothing") + \
                    " applied; " + ", ".join(r["component"] for r in rep["refused"]) + " kept as before"
        receipt = {"schema": "contradish.transform_receipt/1", "mode": self.mode, "action": action, "verdict": v,
                   "repair": ({"applied": rep["applied"], "refused": rep["refused"],
                               "verdict_digest": rep["verdict"]["digest"]} if rep else None),
                   "memory_before_digest": _digest(self.memory), "memory_after_digest": _digest(committed),
                   "committed": committed}
        receipt["digest"] = _digest(receipt)
        self.memory = copy.deepcopy(committed)
        self.receipts.append(receipt)
        if self.ledger is not None:
            self.ledger.append("transform", receipt)
        return receipt


def serve(policy: Any, host: str = "127.0.0.1", port: int = 8788, ledger=None):
    """JSON over HTTP, for frameworks in any language. Standard library only.

        POST /v1/memory/transform   {"before", "after", "as_of", "reason"?, "repair"?: true}
            -> {"verdict": contradish.transform_verdict/1, "repair"?: {"memory", "applied", "refused", "verdict"}}
        POST /v1/memory/compress    {"memory", "as_of"} -> {"memory", "dropped", "verdict"}
        GET  /healthz
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def _send(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200 if self.path.rstrip("/") == "/healthz" else 404, {"ok": True, "schema": VERDICT_SCHEMA})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
                if self.path.rstrip("/") == "/v1/memory/transform":
                    v = check_transform(policy, req["before"], req["after"], req["as_of"], req.get("horizon"),
                                        req.get("reason", ""))
                    out = {"verdict": v}
                    if v["verdict"] != "SAFE" and req.get("repair"):
                        r = repair(policy, req["before"], req["after"], req["as_of"], req.get("horizon"))
                        out["repair"] = {"memory": r["memory"], "applied": r["applied"], "refused": r["refused"],
                                         "verdict": r["verdict"]}
                    if ledger is not None:
                        ledger.append("transform", {"schema": "contradish.transform_receipt/1", "mode": "service",
                                                    "action": "checked", "verdict": v})
                    self._send(200, out)
                elif self.path.rstrip("/") == "/v1/memory/compress":
                    self._send(200, compress(policy, req["memory"], req["as_of"], req.get("horizon")))
                else:
                    self._send(404, {"error": "POST /v1/memory/transform or /v1/memory/compress"})
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer((host, port), H)
