"""
contradish/memory_compiler.py -- a policy-aware compiler for AI agent memory.

    From an agent's full history and its policy, derive the SMALLEST memory that decides every future action the
    policy permits exactly as the full history would; certify that compression; revalidate the certificate as
    authorization evolves.

The futures are not open-ended. They are what the policy file and the history make possible within a horizon:
  - time passing: clock facts advance (days since a fill), and decisions are taken on every future day;
  - scheduled policy changes: updates already issued, by a source with authority, effective later;
  - delegation and revocation: on any future day, anyone with standing (the root of a delegation, or the grant's
    issuer) may revoke any grant; scheduled revocations and expiries take effect on their dates; a revoked grant
    ends every sub-delegation made under it (cascade), and a later grant does not restore them.

Two histories need the same memory exactly when every such future treats them the same (the Myhill-Nerode
quotient, relative to the policy). The compiler computes it:

  FACTS     For each fact, the partition of its values into classes no decision distinguishes on any day of the
            horizon, under the version in force that day, whatever the other facts are and whether or not the
            agent holds its delegation that day. The memory keeps the class of the established value (the latest
            from a source with authority over the fact); claims from anyone else are discarded. A clock fact is
            kept as a class of values on an anchor date, so the memory stays exact as days pass.
  AUTHORITY For a delegated fact, the minimal chains of grants from the root to the agent that are in force at
            some point in the horizon, each kept as [(grant id, issuer)] and the interval it is in force. Holders
            of intermediate grants, sibling delegations, expired and revoked grants, and revocations by people with
            no standing are discarded: no permitted future can depend on them. Issuers are kept, because whether a
            future revocation counts depends on who issued the grant.

The certificate (contradish.memory_certificate/1) carries the memory, every partition with a witness for every
boundary (two neighbouring classes, the other facts, the day and the step on which they are decided differently:
proof that no coarser memory is sufficient), the version timeline, what was discarded and why, and a digest.
`contradish/memory_check.py` re-derives all of it with a separate, standard-library implementation (it evaluates the
policy with evidence_check.py's evaluator, not this package's).

`revalidate(cert, ...)` takes a certificate and what changed (new policy updates, new events, a later date):
  - partitions of facts no changed definition can reach are reused, with the reason recorded;
  - the kept memory is re-checked against the new partitions. If a fact's kept class still lies inside one new
    class, the memory stays valid (and may be coarsened). If it now spans classes a newly announced change tells
    apart, the certificate is INVALIDATED: the memory dropped a distinction that has become load-bearing, and the
    agent must recover it from the record. With the full history supplied, it is recovered automatically.
  - a grant under a grant the memory did not keep cannot be evaluated from memory: INVALIDATED unless the history
    is supplied.
Each revalidated certificate names the one it supersedes.

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import itertools
import json
import math
from typing import Any, Optional

from contradish.action_frontier import _thresholds, situation_universe
from contradish.policy_program import PolicyProgram, ProgramUpdate

__all__ = ["compile_memory", "revalidate", "load_memory_policy", "authority_chains", "established_facts",
           "render_memory", "CERT_SCHEMA"]

CERT_SCHEMA = "contradish.memory_certificate/1"


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _digest(o) -> str:
    return "sha256:" + hashlib.sha256(_canon(o).encode("utf-8")).hexdigest()


def _d(s: str) -> _dt.date:
    return _dt.date.fromisoformat(s)


def _iso(d: _dt.date) -> str:
    return d.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Policy, versions over time
# ─────────────────────────────────────────────────────────────────────────────

def load_memory_policy(policy: Any) -> tuple:
    """(spec without memory/approval/updates, updates, memory config)."""
    from contradish.drafting import APPROVAL_FIELDS
    if isinstance(policy, dict):
        spec = copy.deepcopy(policy)
    else:
        with open(policy) as f:
            spec = json.load(f)
    raw = spec.pop("updates", [])
    cfg = spec.pop("memory", {}) or {}
    for k in list(spec):
        if k in APPROVAL_FIELDS:
            spec.pop(k)
    spec.pop("examples", None)
    return spec, [dict(u) for u in raw], cfg


def timeline(spec: dict, updates: list, as_of: str, horizon: int) -> list:
    """[(start date ISO, label, PolicyProgram, applied update ids)] for the policy in force from `as_of` to
    `as_of + horizon`: one entry per change. Updates from a source without authority over a clause change nothing."""
    p = PolicyProgram(spec)
    end = _iso(_d(as_of) + _dt.timedelta(days=horizon))
    applied = []
    for u in sorted(updates, key=lambda u: (u.get("effective") or "", u["id"])):
        if (u.get("effective") or "") <= as_of:
            ok, _ = p.split_edits(ProgramUpdate.from_dict(u))
            if ok:
                p = p.apply(ok)
                applied.append(u["id"])
    out = [(as_of, "now", p, list(applied))]
    for u in sorted(updates, key=lambda u: (u.get("effective") or "", u["id"])):
        e = u.get("effective") or ""
        if as_of < e <= end:
            ok, _ = p.split_edits(ProgramUpdate.from_dict(u))
            if not ok:
                continue
            p = p.apply(ok)
            applied = applied + [u["id"]]
            out.append((e, u["id"], p, list(applied)))
    return out


def _version_on(tl: list, day: str):
    cur = tl[0]
    for t in tl:
        if t[0] <= day:
            cur = t
    return cur


# ─────────────────────────────────────────────────────────────────────────────
# History: established facts and delegation chains
# ─────────────────────────────────────────────────────────────────────────────

def established_facts(spec: dict, history: list, cfg: dict) -> tuple:
    """({fact: {"value", "at", "source", "event"}}, discarded). The latest observation from a source with authority
    over the fact wins; observations from anyone else are claims and settle nothing."""
    auth = spec.get("fact_authority") or {}
    out, discarded = {}, []
    for i, e in enumerate(history):
        if e.get("kind") != "fact":
            continue
        f = e["fact"]
        if f not in spec["facts"] or f in (cfg.get("delegation") or {}):
            discarded.append({"event": i, "reason": f"not a fact the policy decides on: {f}"})
            continue
        if e.get("source") not in auth.get(f, []):
            discarded.append({"event": i, "reason": f"claim: {e.get('source')} does not establish {f}"})
            continue
        prev = out.get(f)
        if prev is not None:
            discarded.append({"event": prev["event"], "reason": f"superseded: a later record of {f}"})
        out[f] = {"value": e["value"], "at": e.get("at"), "source": e["source"], "event": i}
    return out, discarded


def _grants(history: list) -> tuple:
    grants, revokes = {}, []
    for i, e in enumerate(history):
        if e.get("kind") == "grant":
            grants[e["id"]] = dict(e, event=i)
        elif e.get("kind") == "revoke":
            revokes.append(dict(e, event=i))
    return grants, revokes


def authority_chains(history: list, root: str, scope: str, agent: str, as_of: str, horizon: int) -> tuple:
    """(minimal chains in force at some point in [as_of, as_of + horizon], discarded events).

    A grant {id, by, to, scope, under?, from, until?} has standing if it is issued by the root (with no `under`),
    or by the holder of the grant it is made `under`. A revocation {target, by, effective} has standing if `by` is
    the root or the grant's issuer. A grant is in force on day d if d is in [from, until), no revocation with standing
    took effect on or before d, and the grant it is made under is in force on d. Chains are kept as
    [(grant id, issuer)] with the interval they are in force (the intersection over their links), clipped to the
    horizon."""
    grants, revokes = _grants(history)
    end = _iso(_d(as_of) + _dt.timedelta(days=horizon))
    discarded, ok_rev = [], {}
    for r in revokes:
        g = grants.get(r.get("target"))
        if g is None:
            discarded.append({"event": r["event"], "reason": "revocation of an unknown grant"})
        elif r.get("by") not in (root, g["by"]):
            discarded.append({"event": r["event"], "reason": f"no standing: {r.get('by')} neither issued {g['id']} nor is the root"})
        else:
            eff = r.get("effective") or r.get("at")
            ok_rev[g["id"]] = min(ok_rev.get(g["id"], eff), eff)
    scoped = {k: g for k, g in grants.items() if g.get("scope") == scope}
    for k, g in grants.items():
        if g.get("scope") != scope:
            discarded.append({"event": g["event"], "reason": f"grant {k} is for a different scope"})

    def standing(g):
        if g.get("under") is None:
            return g["by"] == root
        p = scoped.get(g["under"])
        return p is not None and p["to"] == g["by"] and standing(p)

    def interval(k):
        g = scoped[k]
        lo = g.get("from") or g.get("at")
        hi = min([x for x in (g.get("until"), ok_rev.get(k)) if x] or [None]) if (g.get("until") or ok_rev.get(k)) else None
        if g.get("under"):
            plo, phi = interval(g["under"])
            lo = max(lo, plo)
            hi = phi if hi is None else (hi if phi is None else min(hi, phi))
        return lo, hi

    chains = []
    for k, g in scoped.items():
        if not standing(g):
            discarded.append({"event": g["event"], "reason": f"no standing: {g['by']} could not grant {k}"})
            continue
        if g["to"] != agent:
            continue
        path, cur = [], g
        while cur is not None:
            path.append({"id": cur["id"], "issuer": cur["by"]})
            cur = scoped.get(cur.get("under")) if cur.get("under") else None
        lo, hi = interval(k)
        lo2, hi2 = max(lo, as_of), (end if hi is None else min(hi, end))
        if lo2 < hi2 or (hi is None and lo2 <= end):
            chains.append({"links": list(reversed(path)), "from": lo2, "until": None if hi is None or hi > end else hi})
        else:
            discarded.append({"event": g["event"], "reason": f"chain through {k} is not in force within the horizon"})
    used = {l["id"] for c in chains for l in c["links"]}
    for k, g in scoped.items():
        if k not in used and standing(g) and g["to"] != agent and not any(d["event"] == g["event"] for d in discarded):
            discarded.append({"event": g["event"], "reason": f"grant {k} is not on any chain to the agent"})
    for r in revokes:
        if r.get("target") in scoped and r["target"] not in used and not any(d["event"] == r["event"] for d in discarded):
            discarded.append({"event": r["event"], "reason": "revocation of a grant no kept chain depends on (applied)"})
        elif r.get("target") in used and not any(d["event"] == r["event"] for d in discarded):
            discarded.append({"event": r["event"], "reason": "folded into the kept chain's interval"})
    chains.sort(key=lambda c: [l["id"] for l in c["links"]])
    return chains, discarded


def chains_in_force(chains: list, day: str) -> bool:
    return any(c["from"] <= day and (c["until"] is None or day < c["until"]) for c in chains)


# ─────────────────────────────────────────────────────────────────────────────
# Partitions: which values no future decision tells apart
# ─────────────────────────────────────────────────────────────────────────────

def _behaviour(p: PolicyProgram, s: dict) -> str:
    return _canon({k: (c.to_dict() if c is not None else None) for k, c in p.run(s).items()})


class _World:
    """Evaluates every future decision for a situation given on `as_of`: on each critical day of the horizon, under
    the version in force, with clock facts advanced."""

    def __init__(self, spec, updates, cfg, as_of, horizon):
        self.spec, self.cfg, self.as_of, self.H = spec, cfg, as_of, horizon
        self.tl = timeline(spec, updates, as_of, horizon)
        self.progs = [t[2] for t in self.tl]
        self.p0 = self.progs[0]
        self.clock = [f for f in (cfg.get("clock") or {}) if f in self.p0.facts]
        if len(self.clock) > 1:
            raise ValueError("one clock fact is supported (the partition is exact for one)")
        self.deleg = [f for f in (cfg.get("delegation") or {}) if f in self.p0.facts]
        self.thr = {f: sorted({float(t) for p in self.progs for t in _thresholds(p, f)}) for f in self.clock}
        self.eff = sorted({(_d(t[0]) - _d(as_of)).days for t in self.tl[1:]})
        self.cache = {}

    def offsets(self, s: dict) -> list:
        pts = {0, self.H} | set(self.eff) | {e - 1 for e in self.eff if e > 0}
        for f in self.clock:
            mx = self.p0.facts[f].get("max")
            for t in self.thr[f]:
                for k in (-1, 0, 1):
                    pts.add(math.ceil(t) - s[f] + k)
            if mx is not None:
                pts.add(mx - s[f])
        return sorted(x for x in pts if 0 <= x <= self.H)

    def advance(self, s: dict, dt: int) -> dict:
        x = dict(s)
        for f in self.clock:
            mx = self.p0.facts[f].get("max")
            x[f] = s[f] + dt if mx is None else min(mx, s[f] + dt)
        return x

    def trajectory(self, s: dict) -> tuple:
        """The decisions on every future day, as a canonical piecewise-constant function of the day offset."""
        key = _canon(s)
        if key in self.cache:
            return self.cache[key]
        out, last = [], None
        for dt in self.offsets(s):
            day = _iso(_d(self.as_of) + _dt.timedelta(days=dt))
            b = _behaviour(_version_on(self.tl, day)[2], self.advance(s, dt))
            if b != last:
                out.append((dt, b))
                last = b
        self.cache[key] = tuple(out)
        return self.cache[key]

    def first_difference(self, s1: dict, s2: dict):
        days = sorted(set(self.offsets(s1)) | set(self.offsets(s2)))
        for dt in days:
            day = _iso(_d(self.as_of) + _dt.timedelta(days=dt))
            lab, p = _version_on(self.tl, day)[1], _version_on(self.tl, day)[2]
            r1, r2 = p.run(self.advance(s1, dt)), p.run(self.advance(s2, dt))
            for k in p.step_ids():
                a, b = r1[k], r2[k]
                if (a.to_dict() if a else None) != (b.to_dict() if b else None):
                    return {"day": day, "version": lab, "step": k, "a_requires": a.to_dict() if a else None,
                            "b_requires": b.to_dict() if b else None}
        return None


def _axis(w: _World) -> dict:
    uni = situation_universe(w.progs, limit=200000)
    ax = {f: sorted({s[f] for s in uni}, key=lambda v: (str(type(v)), v)) for f in w.p0.facts}
    for f in w.deleg:
        ax[f] = [False, True]
    for f in w.clock:
        sp = w.p0.facts[f]
        ax[f] = list(range(sp.get("min", 0), sp.get("max", 0) + 1))
    return ax


def _partition(w: _World, f: str, ax: dict) -> tuple:
    """(classes, witnesses). Classes are lists of values of f; witnesses separate each pair of neighbouring classes."""
    others = [g for g in w.p0.facts if g != f]
    other_axes = []
    for g in others:
        if g in w.clock:                                  # another clock fact: its own threshold representatives
            other_axes.append(sorted({v for v in ax[g] if any(abs(v - t) <= 1 for t in w.thr.get(g, []))}
                                     | {w.p0.facts[g].get("min", 0)}))
        else:
            other_axes.append(ax[g])
    combos = [dict(zip(others, c)) for c in itertools.product(*other_axes)]

    def sig(v):
        return tuple(w.trajectory(dict(c, **{f: v})) for c in combos)
    vals = ax[f]
    sigs = {v: sig(v) for v in vals}
    is_int = all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
    groups = []
    if is_int:
        for v in vals:
            if groups and sigs[groups[-1][-1]] == sigs[v]:
                groups[-1].append(v)
            else:
                groups.append([v])
    else:
        by = {}
        for v in vals:
            by.setdefault(sigs[v], []).append(v)
        groups = list(by.values())
    wit = []
    pairs = list(zip(groups, groups[1:])) if is_int else list(itertools.combinations(groups, 2))
    for g1, g2 in pairs:
        a, b = (g1[-1], g2[0]) if is_int else (g1[0], g2[0])
        for c in combos:
            d = w.first_difference(dict(c, **{f: a}), dict(c, **{f: b}))
            if d:
                wit.append({"fact": f, "a": a, "b": b, "others": c, **d})
                break
    return groups, wit


def _describe(f: str, groups: list, spec: dict) -> list:
    if spec.get("type") == "int":
        out = []
        for i, g in enumerate(groups):
            lo = spec.get("min", g[0]) if i == 0 else g[0]
            hi = spec.get("max", g[-1]) if i == len(groups) - 1 else groups[i + 1][0] - 1
            out.append({"lo": lo, "hi": hi})
        return out
    return [{"values": list(g)} for g in groups]


def _member(cls: dict, v) -> bool:
    if "values" in cls:
        return v in cls["values"]
    return cls["lo"] <= v <= cls["hi"]


def _depends(p: PolicyProgram, name: str) -> set:
    """Every name `name`'s definition reads, transitively (including facts)."""
    seen, stack = set(), [name]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in p.defs:
            stack.extend(p.refs(p.defs[n]))
    return seen


def _readers(p: PolicyProgram, f: str) -> set:
    """Definitions and steps whose value can depend on fact f."""
    out = set()
    for n in p.defs:
        if f in _depends(p, n):
            out.add(n)
    for sid, cid, s in p.steps:
        e = [s.get("when", True)] + list((s.get("args") or {}).values())
        if any(f in _depends_expr(p, x) for x in e):
            out.add("step:" + sid)
    return out


def _depends_expr(p: PolicyProgram, e) -> set:
    out = set()
    for n in p.refs(e):
        out |= _depends(p, n)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Compile
# ─────────────────────────────────────────────────────────────────────────────

def _covered(cfg, w) -> list:
    c = [f"every day from {w.as_of} to {_iso(_d(w.as_of) + _dt.timedelta(days=w.H))}",
         "the policy version in force on each of those days (updates already issued by a source with authority)"]
    if w.clock:
        c.append("clock facts advancing: " + ", ".join(w.clock))
    if w.deleg:
        c.append("revocation of any kept grant, on any day, by its issuer or the root; scheduled revocations and "
                 "expiries; cascade to sub-delegations; no restoration by a later grant")
    c.append("decisions in every combination of the facts not kept exactly")
    return c


def compile_memory(policy: Any, history: list, agent: str, as_of: str, horizon: Optional[int] = None,
                   _reuse: Optional[dict] = None) -> dict:
    spec, updates, cfg = load_memory_policy(policy)
    H = int(horizon if horizon is not None else cfg.get("horizon_days", 30))
    w = _World(spec, updates, cfg, as_of, H)
    ax = _axis(w)
    facts_est, discarded = established_facts(spec, history, cfg)
    partitions, witnesses, reused = {}, [], {}
    for f in w.p0.facts:
        if f in w.deleg:
            continue
        r = (_reuse or {}).get(f)
        if r is not None:
            partitions[f], wit = r["classes"], r["witnesses"]
            reused[f] = r["reason"]
        else:
            groups, wit = _partition(w, f, ax)
            partitions[f] = _describe(f, groups, w.p0.facts[f])
        witnesses += wit
    memory = {"facts": {}, "authority": {}, "unknown": []}
    for f, cls in partitions.items():
        e = facts_est.get(f)
        if e is None:
            memory["unknown"].append(f)
            continue
        v = e["value"]
        if f in w.clock and e.get("at"):
            v = v + (_d(as_of) - _d(e["at"])).days
        k = next((c for c in cls if _member(c, v)), None)
        if k is None:
            raise ValueError(f"{f} = {v} is outside the fact's domain")
        if len(cls) == 1:
            memory["facts"][f] = {"drop": True}
            continue
        entry = dict(k)
        if f in w.clock:
            entry["on"] = as_of
        memory["facts"][f] = entry
    for f in w.deleg:
        d = cfg["delegation"][f]
        chains, disc = authority_chains(history, d["root"], d["scope"], agent, as_of, H)
        # is the delegated fact read by any decision in the horizon?
        read = any(f in _depends_expr(p, x) for p in w.progs for _, _, s in p.steps
                   for x in [s.get("when", True)] + list((s.get("args") or {}).values()))
        if not read:
            memory["authority"][f] = {"drop": True}
            discarded += [dict(x, reason=x["reason"] + " (the delegated fact is read by no decision)") for x in disc]
            continue
        memory["authority"][f] = {"root": d["root"], "scope": d["scope"], "chains": chains}
        discarded += disc
    body = {"schema": CERT_SCHEMA, "agent": agent, "as_of": as_of, "horizon_days": H,
            "policy": {"base": PolicyProgram(spec).spec,
                       "base_digest": PolicyProgram(spec).digest(), "updates": updates,
                       "fact_authority": spec.get("fact_authority") or {}, "sources": spec.get("sources") or {}},
            "memory_config": cfg,
            "timeline": [{"from": t[0], "label": t[1], "applied": t[3], "digest": t[2].digest(), "spec": t[2].spec}
                         for t in w.tl],
            "history_digest": _digest(history), "history_events": len(history),
            "partitions": partitions, "witnesses": witnesses, "memory": memory,
            "discarded": sorted(discarded, key=lambda x: x["event"]),
            "covered_futures": _covered(cfg, w), "reused_partitions": reused}
    body["size"] = {"history_events": len(history), "kept_facts": sum(1 for v in memory["facts"].values() if not v.get("drop")),
                    "kept_grants": len({l["id"] for a in memory["authority"].values() for c in a.get("chains", [])
                                        for l in c["links"]}),
                    "discarded_events": len({x["event"] for x in discarded})}
    body["digest"] = _digest(body)
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Revalidate as authorization evolves
# ─────────────────────────────────────────────────────────────────────────────

def revalidate(cert: dict, policy: Any, new_updates: Optional[list] = None, new_events: Optional[list] = None,
               as_of: Optional[str] = None, history: Optional[list] = None) -> dict:
    """A new certificate for the same agent after `new_updates` (policy changes), `new_events` (facts, grants,
    revocations) and/or a later `as_of`. Uses the full `history` when given; otherwise works from the kept memory
    alone and says when that is no longer possible."""
    spec, updates, cfg = load_memory_policy(policy)
    updates = updates + list(new_updates or [])
    pol = dict(spec, updates=updates, memory=cfg)
    as_of = as_of or cert["as_of"]
    H = cert["horizon_days"]
    old_tl = timeline(spec, cert["policy"]["updates"], cert["as_of"], H)
    new_w = _World(spec, updates, cfg, as_of, H)
    issues, reuse = [], {}
    # which partitions can be reused: same days, and no changed definition can reach the fact
    same_window = as_of == cert["as_of"]
    changed = set()
    for a, b in itertools.zip_longest([t[2] for t in old_tl], new_w.progs):
        if a is not None and b is not None:
            changed |= a.changed_names(b)
        else:
            changed.add("*")
    if [t[0] for t in old_tl] != [t[0] for t in new_w.tl]:
        changed.add("*")
    for f, cls in cert["partitions"].items():
        if not same_window:
            continue
        if "*" in changed:
            continue
        reach = set()
        for p in new_w.progs:
            reach |= _readers(p, f)
        hit = set()
        for p in new_w.progs:
            for n in changed:
                for r in reach:
                    if r.startswith("step:"):
                        continue
                    if n in _depends(p, r):
                        hit.add(n)
        if not hit and not any(f in _depends(p, n) for p in new_w.progs for n in changed):
            reuse[f] = {"classes": cls, "witnesses": [x for x in cert["witnesses"] if x["fact"] == f],
                        "reason": "no changed definition reads it or shares a reader with it"}
    if history is not None:
        new = compile_memory(pol, list(history) + list(new_events or []), cert["agent"], as_of, H, _reuse=reuse)
        new.update(status="valid", supersedes=cert["digest"], issues=issues, recovered_from_history=True)
        new.pop("digest")
        new["digest"] = _digest(new)
        return new
    # from memory alone
    mem = copy.deepcopy(cert["memory"])
    synthetic = []
    for f, k in mem["facts"].items():
        if k.get("drop"):
            continue
        v_lo = k.get("lo")
        if "value" in k or "values" in k:
            synthetic.append({"kind": "fact", "fact": f, "value": (k.get("values") or [k.get("value")])[0],
                              "source": (spec.get("fact_authority") or {}).get(f, ["record"])[0], "at": k.get("on")})
        else:
            synthetic.append({"kind": "fact", "fact": f, "value": v_lo,
                              "source": (spec.get("fact_authority") or {}).get(f, ["record"])[0], "at": k.get("on")})
    fresh = [e for e in (new_events or []) if e.get("kind") == "fact"]
    new = compile_memory(pol, synthetic + fresh, cert["agent"], as_of, H, _reuse=reuse)
    refreshed = {e["fact"] for e in fresh if e.get("source") in (spec.get("fact_authority") or {}).get(e["fact"], [])}
    for f, k in cert["memory"]["facts"].items():
        if f in refreshed:
            continue
        if k.get("drop") or f not in new["partitions"]:
            if k.get("drop") and f in new["partitions"] and len(new["partitions"][f]) > 1:
                issues.append({"fact": f, "problem": "dropped, but a decision in the new horizon reads it",
                               "recover": f"re-establish {f} from {', '.join((spec.get('fact_authority') or {}).get(f, []))}"})
            continue
        lo = k.get("lo")
        hi = k.get("hi")
        if lo is None:
            continue
        shift = (_d(as_of) - _d(k["on"])).days if k.get("on") else 0
        lo, hi = lo + shift, hi + shift
        spans = [c for c in new["partitions"][f] if not (c["hi"] < lo or c["lo"] > hi)]
        if len(spans) > 1:
            why = [x for x in new["witnesses"] if x["fact"] == f and lo <= x["a"] <= hi and lo <= x["b"] <= hi]
            issues.append({"fact": f, "problem": f"the memory kept {f} only as {lo}-{hi}, and the new futures tell "
                                                 f"{', '.join(str(c['lo']) + '-' + str(c['hi']) for c in spans)} apart",
                           "load_bearing_under": why[0]["version"] if why else None, "day": why[0]["day"] if why else None,
                           "recover": f"re-establish {f} from {', '.join((spec.get('fact_authority') or {}).get(f, []))}"})
            new["memory"]["facts"][f] = {"lo": lo, "hi": hi, **({"on": as_of} if f in new_w.clock else {}),
                                         "insufficient": True}
        elif spans:
            new["memory"]["facts"][f] = dict(spans[0], **({"on": as_of} if f in new_w.clock else {}))
    # authority from kept chains plus new events
    for f, a in cert["memory"]["authority"].items():
        if a.get("drop"):
            continue
        kept = {l["id"]: l for c in a["chains"] for l in c["links"]}
        hist = []
        for c in a["chains"]:
            prev = None
            for l in c["links"]:
                hist.append({"kind": "grant", "id": l["id"], "by": l["issuer"], "to": None, "scope": a["scope"],
                             "under": prev, "from": c["from"], "until": c["until"]})
                prev = l["id"]
        holders = {}
        for c in a["chains"]:
            ls = c["links"]
            for i, l in enumerate(ls):
                holders[l["id"]] = ls[i + 1]["issuer"] if i + 1 < len(ls) else cert["agent"]
        for g in hist:
            g["to"] = holders[g["id"]]
        seen = {}
        for g in hist:
            seen.setdefault(g["id"], g)
        hist = list(seen.values())
        for e in new_events or []:
            if e.get("kind") == "grant" and e.get("under") and e["under"] not in kept:
                issues.append({"authority": f, "problem": f"grant {e['id']} is made under {e['under']}, which the "
                                                         "memory did not keep", "recover": "supply the history"})
            hist.append(e)
        chains, disc = authority_chains(hist, a["root"], a["scope"], cert["agent"], as_of, H)
        new["memory"]["authority"][f] = {"root": a["root"], "scope": a["scope"], "chains": chains}
    new["memory"]["facts"] = {**new["memory"]["facts"], **{f: k for f, k in mem["facts"].items() if k.get("drop")
                                                          and f not in new["memory"]["facts"]}}
    new.update(status="invalidated" if issues else "valid", supersedes=cert["digest"], issues=issues,
               recovered_from_history=False, history_digest=cert["history_digest"],
               note="revalidated from the kept memory; the history was not consulted")
    new["memory"]["unknown"] = [u for u in new["memory"]["unknown"] if u not in cert["memory"]["facts"]] + \
        [u for u in cert["memory"]["unknown"]]
    new.pop("digest")
    new["digest"] = _digest(new)
    return new


def render_memory(cert: dict) -> str:
    """The memory as an agent would carry it in its context."""
    m = cert["memory"]
    L = [f"MEMORY for {cert['agent']} (certified {cert['digest'][:19]}, valid {cert['as_of']} + {cert['horizon_days']} days)"]
    for f, k in m["facts"].items():
        if k.get("drop"):
            continue
        if "values" in k:
            L.append(f"- {f}: one of {k['values']}")
        elif k.get("insufficient"):
            L.append(f"- {f}: INSUFFICIENT (kept only as {k['lo']}-{k['hi']}); re-establish it from the record")
        elif k["lo"] == k["hi"]:
            L.append(f"- {f}: {k['lo']}" + (f" on {k['on']} (advances daily)" if k.get("on") else ""))
        else:
            L.append(f"- {f}: between {k['lo']} and {k['hi']}" + (f" on {k['on']} (advances daily)" if k.get("on") else ""))
    for f, a in m["authority"].items():
        if a.get("drop"):
            continue
        if not a["chains"]:
            L.append(f"- {f}: no delegation in force at any point in the horizon")
        for c in a["chains"]:
            path = " -> ".join(f"{l['id']} (issued by {l['issuer']})" for l in c["links"])
            L.append(f"- {f}: via {path}, in force from {c['from']}" + (f" until {c['until']}" if c["until"] else ""))
    if m["unknown"]:
        L.append("- not established: " + ", ".join(m["unknown"]))
    return "\n".join(L)
