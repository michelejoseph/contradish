#!/usr/bin/env python3
"""
memory_check.py -- independent checker for contradish memory certificates (contradish.memory_certificate/1).

Standard library only. It imports nothing from the contradish package: policies are evaluated with the evaluator in
evidence_check.py (loaded by path; itself standard-library only and written separately from the compiler), and the
situation points, future days, delegation chains and established facts are all re-derived here.

    python memory_check.py CERT.json --history HISTORY.json [--previous OLD_CERT.json] [--json]
    python memory_check.py VERDICT_OR_RECEIPT.json         (a contradish.transform_verdict/1 or transform_receipt/1)

For a transformation verdict it re-derives, from the verdict alone: the version timeline; every situation each
memory is consistent with; every future (no revocation, each kept grant revoked by its issuer on the first day and
on each version change, every pair revoked on the first day); the behaviour sets on every critical day; and so the
verdict, the count of every kind of violation, and every counterexample.

It checks:
  1 integrity      the digest is the sha256 of the canonical JSON of the rest
  2 versions       each version in the timeline is the previous one amended by exactly the edits its update makes
                   to clauses its source governs, in effective-date order; nothing else changes
  3 sufficiency    every class of every partition: all its values are decided identically on every critical day of
                   the horizon, under the version in force that day, in every combination of the other facts
                   (both values of every delegated fact; clock facts advanced)
  4 minimality     every witness: the two values it names are in neighbouring classes and are decided differently on
                   the day, under the version, on the step it names. So no coarser memory is sufficient.
  5 the memory     with the history: each kept fact is the class of the value established by a source with
                   authority over it (the latest such record, clock facts advanced to the anchor date); each delegated
                   fact's chains are exactly the chains this checker derives; the open obligations, with every
                   fulfilment parameter, are exactly those this checker derives; nothing kept is a claim
  6 lineage        with --previous: the certificate supersedes the previous one's digest

Exit 0 iff VERIFIED.
"""
import copy
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("contradish_memory_check_evaluator", os.path.join(_here, "evidence_check.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha(o):
    return "sha256:" + hashlib.sha256(canon(o).encode("utf-8")).hexdigest()


def day(s, n=0):
    return (dt.date.fromisoformat(s) + dt.timedelta(days=n)).isoformat()


def ndays(a, b):
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


# ----------------------------------------------------------------- evaluation
class World:
    def __init__(self, cert):
        self.c = cert
        self.as_of, self.H = cert["as_of"], cert["horizon_days"]
        self.tl = cert["timeline"]
        self.pol = {t["label"]: E.Policy(t["spec"]) for t in self.tl}
        self.facts = self.tl[0]["spec"]["facts"]
        cfg = cert.get("memory_config") or {}
        self.clock = [f for f in (cfg.get("clock") or {}) if f in self.facts]
        self.deleg = [f for f in (cfg.get("delegation") or {}) if f in self.facts]
        self.thr = {f: self.thresholds(f) for f in self.facts}
        self.eff = sorted({ndays(self.as_of, t["from"]) for t in self.tl[1:]})
        self._memo = {}

    def thresholds(self, f):
        """Constants an int fact is compared with, in any version (fact-free sides evaluated)."""
        out = set()
        for t in self.tl:
            P = self.pol[t["label"]]

            def fact_free_value(e):
                try:
                    return P.value(e, {}, (set(), set()))
                except (KeyError, TypeError):
                    return None

            def walk(e):
                if isinstance(e, list) and e and e[0] != "q":
                    if e[0] in ("<", "<=", ">", ">=", "==", "!=") and len(e) == 3:
                        for x, y in ((e[1], e[2]), (e[2], e[1])):
                            if x == f:
                                v = fact_free_value(y)
                                if v is not None and not isinstance(v, bool) and E._num(v):
                                    out.add(float(v))
                            elif isinstance(x, str) and x in P.table and P.table[x][1] == "def":
                                pass
                    for x in e[1:]:
                        walk(x)
            # comparisons may compare a definition that is just the fact; inline one level of definitions
            for n, (cid, kind, v) in P.table.items():
                if kind == "def":
                    walk(v)
            for sid, (cid, s) in P.steps.items():
                walk(s.get("when", True))
                for a in (s.get("args") or {}).values():
                    walk(a)
        return sorted(out)

    def axis(self, f, own=False):
        """Values to try for fact f: all of them for the fact being checked (up to 400), otherwise the points
        around every constant it is compared with (which decide every comparison)."""
        sp = self.facts[f]
        t = sp.get("type")
        if t == "bool" or f in self.deleg:
            return [False, True]
        if t == "enum":
            return list(sp["values"])
        if t == "int":
            lo, hi = sp.get("min", 0), sp.get("max", 0)
            if own and hi - lo <= 400:
                return list(range(lo, hi + 1))
            pts = {lo, hi, sp.get("default", lo)}
            for th in self.thr[f]:
                for k in (-1, 0, 1):
                    pts |= {math.floor(th) + k, math.ceil(th) + k}
            return sorted(v for v in pts if lo <= v <= hi)
        return list(sp.get("values") or [sp.get("default")])

    def version(self, d):
        cur = self.tl[0]
        for t in self.tl:
            if t["from"] <= d:
                cur = t
        return cur

    def advance(self, s, n):
        x = dict(s)
        for f in self.clock:
            mx = self.facts[f].get("max")
            x[f] = s[f] + n if mx is None else min(mx, s[f] + n)
        return x

    def behaviour(self, label, s):
        key = (label, canon(s))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        P = self.pol[label]
        out = {}
        for sid in P.steps:
            norm, _ = P.norm(sid, s)
            out[sid] = norm["call"] if norm["modality"] == "O" else None
        self._memo[key] = canon(out)
        return self._memo[key]

    def offsets(self, s):
        pts = {0, self.H} | set(self.eff) | {e - 1 for e in self.eff if e > 0}
        for f in self.clock:
            for t in self.thr[f]:
                for k in (-1, 0, 1):
                    pts.add(math.ceil(t) - s[f] + k)
            if self.facts[f].get("max") is not None:
                pts.add(self.facts[f]["max"] - s[f])
        return sorted(x for x in pts if 0 <= x <= self.H)

    def trajectory(self, s):
        out, last = [], None
        for n in self.offsets(s):
            b = self.behaviour(self.version(day(self.as_of, n))["label"], self.advance(s, n))
            if b != last:
                out.append((n, b))
                last = b
        return tuple(out)


# ----------------------------------------------------------------- re-derivations
def versions_ok(cert, problems):
    pol = cert["policy"]
    base = pol["base"]
    p = copy.deepcopy(base)
    as_of, H = cert["as_of"], cert["horizon_days"]
    end = day(as_of, H)
    want = []
    applied = []
    ups = sorted(pol["updates"], key=lambda u: (u.get("effective") or "", u["id"]))
    for u in ups:
        if (u.get("effective") or "") <= as_of:
            ok = [e for e in u.get("edits", []) if E.governs(base, u["source"], e["clause"])]
            if ok:
                p = E.amend(p, ok) if not any(e.get("define") for e in ok) else None
                applied.append(u["id"])
    want.append((as_of, "now", p, list(applied)))
    for u in ups:
        e = u.get("effective") or ""
        if as_of < e <= end:
            ok = [x for x in u.get("edits", []) if E.governs(base, u["source"], x["clause"])]
            if not ok:
                continue
            p = E.amend(p, ok) if (p is not None and not any(x.get("define") for x in ok)) else None
            applied = applied + [u["id"]]
            want.append((e, u["id"], p, list(applied)))
    got = cert["timeline"]
    if [(w[0], w[1], w[3]) for w in want] != [(g["from"], g["label"], g["applied"]) for g in got]:
        problems.append("versions: the timeline is not the authorized updates in effective-date order")
        return
    for w, g in zip(want, got):
        if w[2] is None:
            continue                                   # definition edits: compared by the evaluator below only
        a = {k: v for k, v in w[2].items() if k in ("clauses", "facts")}
        b = {k: v for k, v in g["spec"].items() if k in ("clauses", "facts")}
        if canon(a) != canon(b):
            problems.append(f"versions: {g['label']} is not the previous version amended by its authorized edits")


def established(cert, history):
    auth = cert["policy"]["fact_authority"]
    deleg = set((cert.get("memory_config") or {}).get("delegation") or {})
    out = {}
    for e in history:
        if e.get("kind") == "fact" and e["fact"] not in deleg and e.get("source") in auth.get(e["fact"], []):
            out[e["fact"]] = e
    return out


def chains(history, root, scope, agent, as_of, H):
    end = day(as_of, H)
    G = {e["id"]: e for e in history if e.get("kind") == "grant" and e.get("scope") == scope}
    allg = {e["id"]: e for e in history if e.get("kind") == "grant"}
    rev = {}
    for r in history:
        if r.get("kind") != "revoke":
            continue
        g = allg.get(r.get("target"))
        if g is not None and r.get("by") in (root, g["by"]):
            eff = r.get("effective") or r.get("at")
            rev[g["id"]] = min(rev.get(g["id"], eff), eff)

    def good(g):
        if not g.get("under"):
            return g["by"] == root
        p = G.get(g["under"])
        return p is not None and p["to"] == g["by"] and good(p)

    def span(k):
        g = G[k]
        lo = g.get("from") or g.get("at")
        ends = [x for x in (g.get("until"), rev.get(k)) if x]
        hi = min(ends) if ends else None
        if g.get("under"):
            a, b = span(g["under"])
            lo = max(lo, a)
            hi = b if hi is None else (hi if b is None else min(hi, b))
        return lo, hi

    out = []
    for k, g in G.items():
        if g["to"] != agent or not good(g):
            continue
        path, cur = [], g
        while cur is not None:
            path.append({"id": cur["id"], "issuer": cur["by"]})
            cur = G.get(cur.get("under")) if cur.get("under") else None
        lo, hi = span(k)
        lo2, hi2 = max(lo, as_of), end if hi is None else min(hi, end)
        if lo2 < hi2 or (hi is None and lo2 <= end):
            out.append({"links": list(reversed(path)), "from": lo2, "until": None if hi is None or hi > end else hi})
    return sorted(out, key=lambda c: [l["id"] for l in c["links"]])


def member(c, v):
    return v in c["values"] if "values" in c else c["lo"] <= v <= c["hi"]


# ----------------------------------------------------------------- the check
def check(cert, history=None, previous=None):
    problems, notes = [], []
    body = {k: v for k, v in cert.items() if k != "digest"}
    if cert.get("digest") != sha(body):
        problems.append("integrity: digest does not match content")
    if cert.get("schema") != "contradish.memory_certificate/1":
        problems.append("unknown schema")
    if problems:
        return verdict(cert, problems, notes)
    versions_ok(cert, problems)
    W = World(cert)
    facts = [f for f in W.facts if f not in W.deleg]
    if set(cert["partitions"]) != set(facts):
        problems.append("sufficiency: partitions do not cover exactly the non-delegated facts")
        return verdict(cert, problems, notes)
    # 3 sufficiency
    for f in facts:
        cls = cert["partitions"][f]
        others = [g for g in W.facts if g != f]
        axes = [W.axis(g) for g in others]
        combos = [dict(zip(others, c)) for c in itertools.product(*axes)]
        vals = W.axis(f, own=True)
        homes = [[c for c in cls if member(c, v)] for v in vals]
        if any(len(h) != 1 for h in homes):
            problems.append(f"sufficiency: {f}: classes do not partition its values")
            continue
        for c in cls:
            mem = [v for v in vals if member(c, v)]
            if not mem:
                continue
            rep = mem[0]
            sig_rep = [W.trajectory(dict(x, **{f: rep})) for x in combos]
            for v in mem[1:]:
                if [W.trajectory(dict(x, **{f: v})) for x in combos] != sig_rep:
                    problems.append(f"sufficiency: {f} = {v} and {rep} are in one class but decided differently")
                    break
    # 4 minimality
    need = {}
    for f in facts:
        cls = cert["partitions"][f]
        need[f] = len(cls) - 1 if W.facts[f].get("type") == "int" else len(cls) * (len(cls) - 1) // 2
    have = {}
    for w in cert["witnesses"]:
        f = w["fact"]
        cls = cert["partitions"].get(f, [])
        ia = next((i for i, c in enumerate(cls) if member(c, w["a"])), None)
        ib = next((i for i, c in enumerate(cls) if member(c, w["b"])), None)
        if ia is None or ib is None or ia == ib:
            problems.append(f"minimality: witness for {f} does not separate two classes")
            continue
        s1, s2 = dict(w["others"], **{f: w["a"]}), dict(w["others"], **{f: w["b"]})
        n = ndays(W.as_of, w["day"])
        lab = W.version(w["day"])["label"]
        if lab != w["version"]:
            problems.append(f"minimality: witness for {f} names the wrong version for {w['day']}")
            continue
        P = W.pol[lab]
        a, _ = P.norm(w["step"], W.advance(s1, n))
        b, _ = P.norm(w["step"], W.advance(s2, n))
        ca = a["call"] if a["modality"] == "O" else None
        cb = b["call"] if b["modality"] == "O" else None
        if canon(ca) == canon(cb):
            problems.append(f"minimality: witness for {f} ({w['a']} vs {w['b']}) is not decided differently")
            continue
        have[f] = have.get(f, 0) + 1
    for f, k in need.items():
        if have.get(f, 0) < k:
            problems.append(f"minimality: {f} has {have.get(f, 0)} verified witnesses for {k} class boundaries")
    # 5 the memory
    mem = cert["memory"]
    if history is not None:
        if sha(history) != cert["history_digest"] and not cert.get("supersedes"):
            problems.append("memory: the history supplied is not the one certified")
        est = established(cert, history)
        for f in facts:
            k = mem["facts"].get(f)
            if f not in est:
                if f not in mem["unknown"]:
                    problems.append(f"memory: {f} kept, but nothing in the history establishes it")
                continue
            v = est[f]["value"]
            if f in W.clock and est[f].get("at"):
                v = v + ndays(est[f]["at"], W.as_of)
            home = next(c for c in cert["partitions"][f] if member(c, v))
            if len(cert["partitions"][f]) == 1:
                if not (k or {}).get("drop"):
                    problems.append(f"memory: {f} is read by no decision and should be dropped")
            elif k is not None and k.get("insufficient"):
                notes.append(f"memory: {f} is marked insufficient (the certificate is invalidated for it)")
            elif k is None or {x: y for x, y in k.items() if x != "on"} != home:
                problems.append(f"memory: {f} kept as {k}, the established value {v} is in {home}")
        cfg = cert.get("memory_config") or {}
        if cfg.get("obligations") and canon(obligations_from(history, cfg, W.as_of)) != canon(mem.get("obligations") or []):
            problems.append("memory: open obligations differ from those derived from the history "
                            "(an obligation dropped, a fulfilment parameter missing, or a claim kept)")
        for f in W.deleg:
            d = cfg["delegation"][f]
            want = chains(history, d["root"], d["scope"], cert["agent"], W.as_of, W.H)
            got = (mem["authority"].get(f) or {}).get("chains")
            if not (mem["authority"].get(f) or {}).get("drop") and canon(got) != canon(want):
                problems.append(f"memory: {f} chains differ from the chains derived from the history")
    else:
        notes.append("no history supplied: checks 1-4 and 6 only")
    # 6 lineage
    if previous is not None and cert.get("supersedes") != previous.get("digest"):
        problems.append("lineage: does not supersede the previous certificate")
    if cert.get("status") == "invalidated":
        notes.append("the certificate itself reports the memory INVALIDATED: " +
                     "; ".join(i.get("problem", "") for i in cert.get("issues", [])))
    return verdict(cert, problems, notes)


# ----------------------------------------------------------------- obligations
def obligations_from(history, cfg, as_of):
    types = cfg.get("obligations") or {}
    done = {e["id"]: (e.get("at") or as_of) for e in history if e.get("kind") == "discharged"}
    out = {}
    for e in history:
        if e.get("kind") != "obligation" or e.get("type") not in types:
            continue
        t = types[e["type"]]
        if t.get("sources") and e.get("source") not in t["sources"]:
            continue
        if e["id"] in done and done[e["id"]] <= as_of:
            continue
        out[e["id"]] = {"id": e["id"], "type": e["type"], "owed_to": e.get("owed_to"), "incurred": e.get("incurred"),
                        "due": e["due"], "params": {k: e["params"][k] for k in t.get("params", [])
                                                    if k in (e.get("params") or {})}}
    return [out[k] for k in sorted(out)]


def _ob_capable(memory, cfg, ob, d, revoked):
    t = (cfg.get("obligations") or {}).get(ob["type"]) or {}
    if any((ob.get("params") or {}).get(p) is None for p in t.get("params", [])):
        return False

    def open_(auth):
        if auth is None:
            return True
        return _auth(((memory.get("authority") or {}).get(auth) or {}).get("chains") or [], d, revoked)
    if open_(t.get("authority")):
        return True
    fb = t.get("fallback")
    return bool(fb) and open_(fb.get("authority"))


def _ob_days(memory, ob, as_of, futures):
    pts = {as_of, ob["due"]}
    for a in (memory.get("authority") or {}).values():
        for c in a.get("chains") or []:
            pts.add(c["from"])
            if c["until"]:
                pts |= {c["until"], day(c["until"], -1)}
    for _, rev in futures:
        for x in rev.values():
            pts |= {x, day(x, -1)}
    return sorted(p for p in pts if as_of <= p <= ob["due"])


def _ob_violations(cfg, before, after, as_of, futures):
    kinds = {}
    aft = {o["id"]: o for o in after.get("obligations") or []}
    for ob in before.get("obligations") or []:
        if ob["id"] not in aft:
            kinds["obligation_forgotten"] = kinds.get("obligation_forgotten", 0) + 1
            continue
        ob2 = aft[ob["id"]]
        for _, rev in futures:
            ds = _ob_days(before, ob, as_of, futures)
            if not any(_ob_capable(before, cfg, ob, d, rev) for d in ds):
                continue
            ds2 = sorted(set(ds) | set(_ob_days(after, ob2, as_of, futures)))
            if not any(_ob_capable(after, cfg, ob2, d, rev) for d in ds2):
                kinds["fulfillment_loss"] = kinds.get("fulfillment_loss", 0) + 1
    return kinds


# ----------------------------------------------------------------- transformation verdicts
KINDS = ("authority_expansion", "obligation_forgotten", "fulfillment_loss", "withholding", "changed_action",
         "indeterminacy", "fabrication")


def _allowed(v, c):
    if c is None or c.get("drop"):
        return True
    if "values" in c:
        return v in c["values"]
    if "value" in c:
        return v == c["value"]
    return (c.get("lo") is None or v >= c["lo"]) and (c.get("hi") is None or v <= c["hi"])


def _auth(chains, d, revoked):
    for c in chains:
        if c["from"] <= d and (c["until"] is None or d < c["until"]) and \
                not any(l["id"] in revoked and revoked[l["id"]] <= d for l in c["links"]):
            return True
    return False


def _worlds(W, memory):
    facts = memory.get("facts") or {}
    axes = []
    for f in W.facts:
        if f in W.deleg:
            continue
        c = facts.get(f)
        vals = W.axis(f, own=f in W.clock)
        if c and not c.get("drop"):
            shift = ndays(c["on"], W.as_of) if (f in W.clock and c.get("on")) else 0
            cc = dict(c, **{k: c[k] + shift for k in ("lo", "hi", "value") if c.get(k) is not None}) if shift else c
            pts = [cc[k] for k in ("lo", "hi", "value") if cc.get(k) is not None] + list(cc.get("values") or [])
            vals = sorted(set(vals) | {p for p in pts if p in vals or not isinstance(p, int)},
                          key=lambda v: (str(type(v)), v))
            vals = [v for v in vals if _allowed(v, cc)]
        axes.append((f, vals))
    names = [a for a, _ in axes]
    return [dict(zip(names, c)) for c in itertools.product(*[v for _, v in axes])]


def _futures(W, before, after):
    links = {}
    for m in (before, after):
        for a in (m.get("authority") or {}).values():
            for c in a.get("chains") or []:
                for l in c["links"]:
                    links[l["id"]] = l["issuer"]
    days = [W.as_of] + [t["from"] for t in W.tl[1:]]
    out = [("no revocation", {})]
    for g, i in sorted(links.items()):
        for d in days:
            out.append((f"{i} revokes {g} effective {d}", {g: d}))
    for (g1, i1), (g2, i2) in itertools.combinations(sorted(links.items()), 2):
        out.append((f"{i1} revokes {g1} and {i2} revokes {g2}, effective {W.as_of}", {g1: W.as_of, g2: W.as_of}))
    return out


def _days(W, worlds):
    offs = {0, W.H} | set(W.eff) | {e - 1 for e in W.eff if e > 0}
    for s in worlds:
        offs |= set(W.offsets(s))
    return offs


def _sets(W, memory, futures, offs):
    worlds = _worlds(W, memory)
    out = {}
    for label, rev in futures:
        for n in sorted(o for o in offs if 0 <= o <= W.H):
            d = day(W.as_of, n)
            auth = {f: _auth(((memory.get("authority") or {}).get(f) or {}).get("chains") or [], d, rev) for f in W.deleg}
            lab = W.version(d)["label"]
            out[(d, label)] = {W.behaviour(lab, dict(W.advance(s, n), **auth)) for s in worlds}
    return out


def _kind(b, a, irreversible):
    if a == b:
        return None
    if len(b) == 1:
        if len(a) > 1:
            return "indeterminacy"
        bb, aa = json.loads(next(iter(b))), json.loads(next(iter(a)))
        if any(c is not None and c["tool"] in irreversible and bb.get(k) != c for k, c in aa.items()):
            return "authority_expansion"
        if any(c is not None and (aa.get(k) is None or aa[k]["tool"] != c["tool"]) for k, c in bb.items()):
            return "withholding"
        return "changed_action"
    return "fabrication" if a < b else "changed_action"


def check_verdict(v):
    """Re-derive a contradish.transform_verdict/1 from its own contents."""
    problems, notes = [], []
    body = {k: x for k, x in v.items() if k != "digest"}
    if v.get("digest") != sha(body):
        problems.append("integrity: digest does not match content")
    if v.get("schema") != "contradish.transform_verdict/1":
        problems.append("unknown schema")
    if v.get("before_digest") != sha(v.get("before")) or v.get("after_digest") != sha(v.get("after")):
        problems.append("integrity: memory digests do not match the memories")
    if problems:
        return {"verdict": "REJECTED", "digest": v.get("digest"), "problems": problems, "notes": notes}
    versions_ok(v, problems)
    W = World(v)
    fut = _futures(W, v["before"], v["after"])
    if [f[0] for f in fut] != v["futures_checked"]:
        problems.append("futures: the futures checked are not every revocation the kept grants allow")
    offs = _days(W, _worlds(W, v["before"]) + _worlds(W, v["after"]))   # one set of days for both memories
    sb, sa = _sets(W, v["before"], fut, offs), _sets(W, v["after"], fut, offs)
    irr = set(v["policy"]["irreversible_tools"])
    kinds = {}
    for key in sorted(set(sb) | set(sa)):
        k = _kind(sb[key], sa[key], irr)
        if k:
            kinds[k] = kinds.get(k, 0) + 1
    for k, n in _ob_violations(v.get("memory_config") or {}, v["before"], v["after"], v["as_of"], fut).items():
        kinds[k] = kinds.get(k, 0) + n
    want = "SAFE" if not kinds else "UNSAFE"
    if want != v["verdict"]:
        problems.append(f"verdict: re-derived {want}, the receipt says {v['verdict']}")
    if kinds != v["violations_by_kind"]:
        problems.append(f"violations: re-derived {kinds}, the receipt says {v['violations_by_kind']}")
    for c in v["counterexamples"]:
        if c["kind"] in ("obligation_forgotten", "fulfillment_loss"):
            continue                                   # counted above, from the obligations themselves
        key = (c["day"], c["future"])
        if key not in sb or _kind(sb[key], sa[key], irr) != c["kind"]:
            problems.append(f"counterexample: {c['kind']} on {c['day']} under '{c['future']}' does not hold")
    return {"verdict": "VERIFIED" if not problems else "REJECTED", "digest": v.get("digest"),
            "claims": v["verdict"], "problems": problems, "notes": notes}


def verdict(cert, problems, notes):
    return {"verdict": "VERIFIED" if not problems else "REJECTED", "digest": cert.get("digest"),
            "agent": cert.get("agent"), "as_of": cert.get("as_of"), "problems": problems, "notes": notes}


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in a
    a = [x for x in a if x != "--json"]
    hist = prev = None
    if "--history" in a:
        i = a.index("--history"); hist = json.load(open(a[i + 1])); del a[i:i + 2]
    if "--previous" in a:
        i = a.index("--previous"); prev = json.load(open(a[i + 1])); del a[i:i + 2]
    if not a:
        print(__doc__)
        return 2
    rs = []
    for p in a:
        d = json.load(open(p))
        if d.get("schema") == "contradish.transform_receipt/1":
            d = d["verdict"]
        if d.get("schema") == "contradish.transform_verdict/1":
            r = check_verdict(d)
            r.update(agent=f"transform ({r.get('claims')})", as_of=d.get("as_of"))
            rs.append(r)
        else:
            rs.append(check(d, hist, prev))
    if as_json:
        print(json.dumps(rs, indent=2))
    else:
        for r in rs:
            print(f"{r['verdict']}  {r['agent']} as of {r['as_of']}  {str(r['digest'])[:23]}")
            for p in r["problems"]:
                print("  - " + p)
            for n in r["notes"]:
                print("  . " + n)
    return 0 if all(r["verdict"] == "VERIFIED" for r in rs) else 1


if __name__ == "__main__":
    sys.exit(main())
