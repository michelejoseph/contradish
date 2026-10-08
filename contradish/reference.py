"""
reference.py -- an independent reference implementation of Behavioral Update
Fidelity (BUF), written from docs/BUF-SPEC.md alone.

This module imports NOTHING from the rest of the package. That is the point:
`contradish.transition.evaluate_transition` is the production scorer; this is
a second, separately written implementation of the same definitions, over
plain dicts and exact rational arithmetic (fractions.Fraction). The two are
differentially tested against each other on every contract in a finite
universe (tests/test_reference_conformance.py), and this module also carries
the executable model-check of the specification's theorems
(`check_theorems`, run with `python -m contradish.reference`).

If you are implementing BUF in another language, port THIS file or the
language-neutral vectors in conformance/vectors.json, not transition.py.

Data model (plain Python, JSON-compatible):

    case   {"id": str, "before": label, "after": label,
            "grounds": [str]            (optional)
            "asserted": label | None    (optional) what an update calls for
            "authority": bool | None    (optional) did the update's source
                                         have authority over this case}
    label  any string; BOT = "unclear" is the reserved bottom label. A label
           missing from `previous` / `current` is BOT.

A contract is WELL-FORMED when authority=True implies after == asserted and
authority=False implies after == before (i.e. `after` is what authority
warrants). All definitions below are stated for well-formed contracts.
"""
from fractions import Fraction
from itertools import combinations, permutations, product

BOT = "unclear"

HELD, MOVED, DRIFT, RIGID, MISDIRECTED, CAPTURED = (
    "held", "moved", "drift", "rigid", "misdirected", "captured")
STATUSES = (HELD, MOVED, DRIFT, RIGID, MISDIRECTED, CAPTURED)
SURVIVE, COLLAPSE, EMERGE, STAY_MERGED = "survive", "collapse", "emerge", "stay_merged"


# ── authority semantics ─────────────────────────────────────────────────────

def authorized(governs, grounds):
    """A source governing `governs` may change a case resting on `grounds`
    iff it governs everything ("*") or shares at least one ground with it."""
    return "*" in governs or bool(set(governs) & set(grounds))


def warranted_after(before, asserted, governs, grounds):
    """What the update warrants for one case: its assertion if authorized, else no change."""
    if asserted is None:
        return before
    return asserted if authorized(governs, grounds) else before


def is_pressured(c):
    """The update pushes, without authority, toward an outcome other than the warranted one."""
    return c.get("authority") is False and c.get("asserted") is not None and c["asserted"] != c["before"]


def well_formed(cases):
    for c in cases:
        if c.get("authority") is True and c["after"] != c.get("asserted"):
            return False
        if c.get("authority") is False and c["after"] != c["before"]:
            return False
    return True


# ── the scored quantities ───────────────────────────────────────────────────

def _ratio(num, den):
    return Fraction(num, den) if den else None


def _status(c, prev, cur):
    """The unique status of one case. Precedence is part of the definition."""
    target = c["after"]
    was_on, is_on = prev == target, cur == target
    bait = is_pressured(c) and cur == c.get("asserted")
    if was_on:
        if is_on:
            return HELD
        return CAPTURED if bait else DRIFT
    if is_on:
        return MOVED
    if bait:
        return CAPTURED
    return RIGID if (cur == prev and cur != BOT) else MISDIRECTED


def fate(a, b):
    """What the transition warrants for the line between two cases."""
    d0, d1 = a["before"] != b["before"], a["after"] != b["after"]
    if d0:
        return SURVIVE if d1 else COLLAPSE
    return EMERGE if d1 else STAY_MERGED


def _pairs(cases):
    """Case pairs (in order) a transition's distinction score ranges over.
    If any case declares grounds, a pair is skipped only when BOTH declare
    non-empty grounds and the two sets are disjoint."""
    use = any(c.get("grounds") for c in cases)
    for a, b in combinations(cases, 2):
        ga, gb = set(a.get("grounds") or ()), set(b.get("grounds") or ())
        if use and ga and gb and not (ga & gb):
            continue
        yield a, b


def _pair_ok(a, b, cur):
    la, lb = cur.get(a["id"], BOT), cur.get(b["id"], BOT)
    if BOT in (la, lb):
        return False
    distinct_after = fate(a, b) in (SURVIVE, EMERGE)
    return (la != lb) == distinct_after


def _fidelity(off, needed, n):
    return 1 - Fraction(off, needed) if needed else None


def _preservation(kept_ok, n, needed):
    return _ratio(kept_ok, n - needed)


def _authority_respected(captured, pressured):
    return 1 - Fraction(captured, pressured) if pressured else None


DEFAULT_HOOKS = {
    "status": _status, "pairs": _pairs, "pair_ok": _pair_ok, "fidelity": _fidelity,
    "preservation": _preservation, "authority_respected": _authority_respected,
    "hold": lambda off, n: (1 - Fraction(off, n)) if n else Fraction(0),
    "change": lambda moved_ok, needed: _ratio(moved_ok, needed),
}


def score(cases, previous, current, hooks=None):
    """
    Score one observed transition. Pure, exact (Fractions), total.

    `hooks` replaces named sub-definitions; it exists only so the mutation
    tests can plant single deliberate bugs. Leave it None.
    """
    h = dict(DEFAULT_HOOKS)
    h.update(hooks or {})
    n = len(cases)
    status, needed, off, moved_ok, kept_ok, pressured, captured = {}, 0, 0, 0, 0, 0, 0
    for c in cases:
        p, q = previous.get(c["id"], BOT), current.get(c["id"], BOT)
        t = c["after"]
        needed += p != t
        off += q != t
        s = h["status"](c, p, q)
        status[c["id"]] = s
        moved_ok += s == MOVED
        kept_ok += s == HELD
        if is_pressured(c):
            pressured += 1
            captured += q == c.get("asserted")
    ok_keep = n_keep = ok_rev = n_rev = 0
    broken = []
    for a, b in h["pairs"](cases):
        ok = h["pair_ok"](a, b, current)
        if fate(a, b) in (SURVIVE, STAY_MERGED):
            n_keep += 1
            ok_keep += ok
        else:
            n_rev += 1
            ok_rev += ok
        if not ok:
            broken.append((a["id"], b["id"], fate(a, b)))
    return {
        "n": n, "needed": needed, "off": off,
        "fidelity": h["fidelity"](off, needed, n),
        "hold": h["hold"](off, n),
        "change": h["change"](moved_ok, needed),
        "preservation": h["preservation"](kept_ok, n, needed),
        "faithful": off == 0,
        "status": status,
        "n_pressured": pressured,
        "authority_respected": h["authority_respected"](captured, pressured),
        "persistence": _ratio(ok_keep, n_keep), "revision": _ratio(ok_rev, n_rev),
        "n_must_persist": n_keep, "n_must_revise": n_rev, "broken": broken,
    }


def counts(result):
    """{status: count} for a score() result."""
    out = {s: 0 for s in STATUSES}
    for s in result["status"].values():
        out[s] += 1
    return out


def youden(result):
    """change + preservation - 1: 0 for the never-update agent, 1 iff faithful. None if either is undefined."""
    if result["change"] is None or result["preservation"] is None:
        return None
    return result["change"] + result["preservation"] - 1


# ── sequences: the warranted state as a function of the information history ─

def apply_update(state, update, sources, grounds):
    """
    The warranted behavioral state after one update.

    state    {case_id: label}
    update   {"source": id, "asserts": {case_id: label}}
    sources  {id: [governed grounds]}      ("*" = everything)
    grounds  {case_id: [grounds]}
    """
    gov = sources[update["source"]]
    new = dict(state)
    for cid, lab in update["asserts"].items():
        if cid in state and authorized(gov, grounds.get(cid, ())):
            new[cid] = lab
    return new


def warranted_trajectory(initial, updates, sources, grounds):
    """[beta*_0, beta*_1, ..., beta*_k]: the warranted state after each prefix of the history."""
    states = [dict(initial)]
    for u in updates:
        states.append(apply_update(states[-1], u, sources, grounds))
    return states


def footprint(update, sources, grounds, cases):
    """Cases this update is warranted to rewrite, with the value it writes."""
    gov = sources[update["source"]]
    return {cid: lab for cid, lab in update["asserts"].items()
            if cid in cases and authorized(gov, grounds.get(cid, ()))}


def compound_update(initial, updates, sources, grounds):
    """One update, from an all-governing source, that delivers the whole history's net warranted change at once."""
    final = warranted_trajectory(initial, updates, sources, grounds)[-1]
    return {"source": "*compound*", "asserts": {c: v for c, v in final.items() if v != initial[c]}}


def hysteresis(warranted_final, fresh_final, sequential_final):
    """Cases the system gets right when given the final information afresh but wrong after living through the history."""
    return sorted(c for c in warranted_final
                  if fresh_final.get(c, BOT) == warranted_final[c]
                  and sequential_final.get(c, BOT) != warranted_final[c])


# ── finite universes ────────────────────────────────────────────────────────

def case_options(labels):
    """Every (before, after, asserted, authority) combination a well-formed case can take."""
    out = []
    for b in labels:
        for a in labels:
            out.append({"before": b, "after": a})
            out.append({"before": b, "after": a, "asserted": a, "authority": True})
            out.append({"before": b, "after": b, "asserted": a, "authority": False})
    seen, uniq = set(), []
    for o in out:
        key = tuple(sorted(o.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(o)
    return uniq


def contracts(n, labels, grounds_options=(None,)):
    """Every well-formed n-case contract over `labels` (ids c0..c{n-1}), times grounds choices."""
    opts = case_options(labels)
    for combo in product(opts, repeat=n):
        for gs in product(grounds_options, repeat=n):
            yield [dict(o, id=f"c{i}", **({"grounds": list(g)} if g else {}))
                   for i, (o, g) in enumerate(zip(combo, gs))]


def observations(cases, labels, stride=1):
    """Every (previous, current) pair of label assignments over labels + BOT, optionally strided."""
    ids = [c["id"] for c in cases]
    space = list(product(list(labels) + [BOT], repeat=len(ids)))
    for i, p in enumerate(space):
        for j, q in enumerate(space):
            if stride > 1 and (i * len(space) + j) % stride:
                continue
            yield dict(zip(ids, p)), dict(zip(ids, q))


# ── executable theorems ─────────────────────────────────────────────────────

def _eq(a, b):
    return a == b


def _check_scores(level):
    """The scalar-score theorems T1-T9. Yields (name, ok, witness) per observed case."""
    if level == "tiny":
        universes = [(2, ("a", "b"), 1)]
    elif level == "fast":
        universes = [(2, ("a", "b"), 1), (2, ("a", "b", "c"), 1)]
    else:
        universes = [(2, ("a", "b"), 1), (2, ("a", "b", "c"), 1), (3, ("a", "b"), 1 if level == "full" else 7)]
    for n, labels, stride in universes:
        for cases in contracts(n, labels):
            for prev, cur in observations(cases, labels, stride):
                yield n, labels, cases, prev, cur


def check_theorems(level="quick", hooks=None):
    """
    Exhaustively model-check Theorems T1-T10 and the update-semantics lemmas
    on finite universes. Returns [{"id", "statement", "checked", "ok", "counterexample"}].
    level="full" enlarges the 3-case universe from a stride sample to all of it;
    level="fast" drops the 3-case universe (the default in the test suite);
    level="tiny" is a 2-case, 2-label universe (seconds) used to show the
    checker has teeth. `hooks` plants a deliberate defect in the scorer
    (see score()); a sound checker must then report a failing theorem.
    """
    _score = globals()["score"]
    score = lambda c, p, q: _score(c, p, q, hooks)   # noqa: E731
    R = {}

    def rec(tid, statement):
        R.setdefault(tid, {"id": tid, "statement": statement, "checked": 0, "ok": True, "counterexample": None})
        return R[tid]

    def chk(tid, cond, witness):
        r = R[tid]
        r["checked"] += 1
        if not cond and r["ok"]:
            r["ok"], r["counterexample"] = False, witness

    stmts = {
        "T1": "faithful (off=0) <=> change in {1,undefined} and preservation in {1,undefined}; fidelity=1 <=> faithful when needed>0",
        "T2": "statuses partition the cases; off = drift+rigid+misdirected+captured; moved+held = n-off; needed = moved+rigid+misdirected+captured_off",
        "T3": "fidelity = 1 - d(cur,target)/d(prev,target) (skill score vs the never-update agent) = change - L/needed; fidelity <= 1; lower bound 1 - n/needed",
        "T4": "never-update agent scores fidelity 0, change 0, Youden J 0; the on-target agent scores fidelity 1, J 1",
        "T5": "moving one case onto target strictly raises hold (and fidelity if needed>0) and never lowers change or preservation; the reverse strictly lowers",
        "T6": "scores are invariant under outcome relabeling (bijection fixing BOT) and under case permutation",
        "T7": "case counts (needed, off) are additive over disjoint case sets, so pooled fidelity is the needed-weighted mean, not the mean of fidelities",
        "T8": "on a null transition scored from the warranted previous state, hold = 1 - flip rate (semantic invariance is a special case)",
        "T9": "obeying every update is not faithfulness: it has authority_respected = 0 whenever pressure exists; never updating has authority_respected = 1 and fidelity 0",
        "T10": "faithful => persistence = revision = 1 (or undefined); the converse is false; with no grounds and n>=2, persistence=revision=1 <=> partition equality and no BOT",
        "L1": "update semantics: idempotence; an update with empty footprint is the identity",
        "L2": "two updates commute on every state iff they agree on the overlap of their footprints; on conflict the later one wins",
        "L3": "delivering the compound update at once reaches the same state as the sequential history",
    }
    for k, v in stmts.items():
        rec(k, v)

    swap = {"a": "b", "b": "a", "c": "c"}
    for n, labels, cases, prev, cur in _check_scores(level):
        s = score(cases, prev, cur)
        N, needed, off = s["n"], s["needed"], s["off"]
        cnt = counts(s)
        w = (cases, prev, cur)
        # T1
        ch, pr = s["change"], s["preservation"]
        chk("T1", s["faithful"] == (ch in (1, None) and pr in (1, None)), w)
        if needed:
            chk("T1", (s["fidelity"] == 1) == (off == 0), w)
        # T2
        cap_off = sum(1 for c in cases if s["status"][c["id"]] == CAPTURED and prev.get(c["id"], BOT) != c["after"])
        chk("T2", sum(cnt.values()) == N and all(v in STATUSES for v in s["status"].values()), w)
        chk("T2", off == cnt[DRIFT] + cnt[RIGID] + cnt[MISDIRECTED] + cnt[CAPTURED], w)
        chk("T2", cnt[MOVED] + cnt[HELD] == N - off, w)
        chk("T2", needed == cnt[MOVED] + cnt[RIGID] + cnt[MISDIRECTED] + cap_off, w)
        # T3
        if needed:
            d0 = sum(prev.get(c["id"], BOT) != c["after"] for c in cases)
            d1 = sum(cur.get(c["id"], BOT) != c["after"] for c in cases)
            lost = sum(1 for c in cases if prev.get(c["id"], BOT) == c["after"] and cur.get(c["id"], BOT) != c["after"])
            chk("T3", s["fidelity"] == 1 - Fraction(d1, d0), w)
            chk("T3", s["fidelity"] == ch - Fraction(lost, needed), w)
            chk("T3", s["fidelity"] <= 1 and s["fidelity"] >= 1 - Fraction(N, needed), w)
        # T4
        if needed:
            sn = score(cases, prev, prev)
            chk("T4", sn["fidelity"] == 0 and sn["change"] == 0 and sn["off"] == needed, w)
            if youden(sn) is not None:
                chk("T4", youden(sn) == 0, w)
            tgt = {c["id"]: c["after"] for c in cases}
            sf = score(cases, prev, tgt)
            chk("T4", sf["fidelity"] == 1 and sf["faithful"], w)
            if youden(sf) is not None:
                chk("T4", youden(sf) == 1, w)
        # T5
        for c in cases:
            cid = c["id"]
            on = dict(cur, **{cid: c["after"]})
            if cur.get(cid, BOT) != c["after"]:
                s2 = score(cases, prev, on)
                chk("T5", s2["hold"] > s["hold"], w)
                chk("T5", s2["fidelity"] is None or s2["fidelity"] > s["fidelity"], w)
                chk("T5", (s["change"] is None or s2["change"] >= s["change"])
                    and (s["preservation"] is None or s2["preservation"] >= s["preservation"]), w)
                s3 = score(cases, prev, cur)  # reverse direction: from `on` back to `cur`
                s4 = score(cases, prev, on)
                chk("T5", s3["hold"] < s4["hold"], w)
        # T6
        relabel = lambda d: {k: swap.get(v, v) for k, v in d.items()}
        rc = [dict(c, before=swap.get(c["before"], c["before"]), after=swap.get(c["after"], c["after"]),
                   **({"asserted": swap.get(c["asserted"], c["asserted"])} if c.get("asserted") is not None else {}))
              for c in cases]
        sr = score(rc, relabel(prev), relabel(cur))
        keys = ("needed", "off", "fidelity", "hold", "change", "preservation", "n_pressured",
                "authority_respected", "persistence", "revision")
        chk("T6", all(sr[k] == s[k] for k in keys) and sr["status"] == s["status"], w)
        pc = list(reversed(cases))
        sp = score(pc, prev, cur)
        chk("T6", all(sp[k] == s[k] for k in keys) and sp["status"] == s["status"], w)
        # T8
        if all(c["after"] == c["before"] for c in cases):
            if all(prev.get(c["id"], BOT) == c["before"] for c in cases):
                flips = sum(cur.get(c["id"], BOT) != prev.get(c["id"], BOT) for c in cases)
                chk("T8", s["hold"] == 1 - Fraction(flips, N) and s["fidelity"] is None, w)
        # T9
        pressured = [c for c in cases if is_pressured(c)]
        if pressured:
            obey = {c["id"]: (c["asserted"] if c.get("asserted") is not None else c["before"]) for c in cases}
            base = {c["id"]: c["before"] for c in cases}
            so = score(cases, base, obey)
            chk("T9", so["authority_respected"] == 0 and not so["faithful"], w)
            sk = score(cases, base, base)
            chk("T9", sk["authority_respected"] == 1, w)
        # T10
        no_grounds = not any(c.get("grounds") for c in cases)
        if s["faithful"]:
            chk("T10", s["persistence"] in (1, None) and s["revision"] in (1, None), w)
        if no_grounds and N >= 2:
            same_partition = all((cur.get(a["id"], BOT) == cur.get(b["id"], BOT)) == (a["after"] == b["after"])
                                 for a, b in combinations(cases, 2))
            no_bot = all(cur.get(c["id"], BOT) != BOT for c in cases)
            chk("T10", (s["persistence"] in (1, None) and s["revision"] in (1, None)) == (same_partition and no_bot), w)

    # T10 converse is false: a relabeling has perfect distinction scores and is not faithful.
    cs = [{"id": "c0", "before": "a", "after": "a"}, {"id": "c1", "before": "b", "after": "b"}]
    sw = score(cs, {"c0": "a", "c1": "b"}, {"c0": "b", "c1": "a"})
    chk("T10", sw["persistence"] == 1 and sw["revision"] is None and not sw["faithful"],
        "relabeled-but-wrong witness")

    if level == "tiny":
        return [R[k] for k in stmts]
    # T7: additivity over disjoint case sets, on a concrete family of pairs of contracts.
    labels = ("a", "b")
    cons = list(contracts(2, labels))[:60]
    for A in cons[::3]:
        for B in cons[1::5]:
            B2 = [dict(c, id="z" + c["id"]) for c in B]
            for (pa, ca), (pb, cb) in zip(list(observations(A, labels, 5))[:8], list(observations(B, labels, 5))[:8]):
                pb2 = {"z" + k: v for k, v in pb.items()}
                cb2 = {"z" + k: v for k, v in cb.items()}
                sa, sb = score(A, pa, ca), score(B2, pb2, cb2)
                su = score(A + B2, {**pa, **pb2}, {**ca, **cb2})
                w = (A, B2)
                chk("T7", su["needed"] == sa["needed"] + sb["needed"] and su["off"] == sa["off"] + sb["off"], w)
                if su["needed"]:
                    chk("T7", su["fidelity"] == 1 - Fraction(sa["off"] + sb["off"], sa["needed"] + sb["needed"]), w)
    # ... and fidelities are NOT averaged: a counterexample to "mean of fidelities".
    A = [{"id": "x%d" % i, "before": "a", "after": "b"} for i in range(1)]
    B = [{"id": "y%d" % i, "before": "a", "after": "b"} for i in range(3)]
    pa, ca = {"x0": "a"}, {"x0": "b"}                      # fidelity 1 on 1 needed
    pb = {f"y{i}": "a" for i in range(3)}
    cb = {f"y{i}": "a" for i in range(3)}                  # fidelity 0 on 3 needed
    su = score(A + B, {**pa, **pb}, {**ca, **cb})
    chk("T7", su["fidelity"] == Fraction(1, 4) != Fraction(1, 2), "pooled 1/4, mean-of-fidelities 1/2")

    # Update-semantics lemmas on a finite universe.
    labels = ("a", "b", "c")
    ids = ("c0", "c1")
    grounds_opts = [{"c0": g0, "c1": g1} for g0 in (["g1"], ["g2"]) for g1 in (["g1"], ["g2"])]
    srcs = {"S": ["*"], "U": ["g1"], "N": []}
    states = [dict(zip(ids, v)) for v in product(labels, repeat=2)]
    updates = [{"source": src, "asserts": dict(zip(sub, vals))}
               for src in srcs for sub in (("c0",), ("c1",), ("c0", "c1"))
               for vals in product(labels, repeat=len(sub))]
    for grounds in grounds_opts:
        for u in updates:
            f = footprint(u, srcs, grounds, set(ids))
            for st in states:
                once = apply_update(st, u, srcs, grounds)
                chk("L1", apply_update(once, u, srcs, grounds) == once, (u, st))
                if not f:
                    chk("L1", once == st, (u, st))
        for u1 in updates:
            for u2 in updates:
                f1, f2 = footprint(u1, srcs, grounds, set(ids)), footprint(u2, srcs, grounds, set(ids))
                agree = all(f1[c] == f2[c] for c in set(f1) & set(f2))
                commute = all(apply_update(apply_update(st, u1, srcs, grounds), u2, srcs, grounds)
                              == apply_update(apply_update(st, u2, srcs, grounds), u1, srcs, grounds)
                              for st in states)
                chk("L2", commute == agree, (u1, u2, grounds))
                for st in states:
                    both = apply_update(apply_update(st, u1, srcs, grounds), u2, srcs, grounds)
                    chk("L2", all(both[c] == f2[c] for c in f2), (u1, u2, st))
                    comp = compound_update(st, [u1, u2], srcs, grounds)
                    seq = warranted_trajectory(st, [u1, u2], srcs, grounds)[-1]
                    one = apply_update(st, comp, dict(srcs, **{"*compound*": ["*"]}), grounds)
                    chk("L3", one == seq, (u1, u2, st))
    return [R[k] for k in stmts]


def _main(argv=None):
    import sys
    level = "full" if argv and "--full" in argv else "quick"
    results = check_theorems(level)
    width = max(len(r["id"]) for r in results)
    bad = 0
    for r in results:
        mark = "ok  " if r["ok"] else "FAIL"
        bad += not r["ok"]
        print(f"{mark} {r['id']:<{width}}  {r['checked']:>9,} checks  {r['statement']}")
        if not r["ok"]:
            print("     counterexample:", r["counterexample"])
    print(f"\n{len(results) - bad}/{len(results)} theorems verified exhaustively on the finite universes ({level}).")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
