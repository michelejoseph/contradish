"""
conformance.py -- language-neutral conformance for Behavioral Update Fidelity.

The specification (docs/BUF-SPEC.md) defines the scores; the reference
implementation (contradish.reference) computes them exactly; this module
turns the reference into a TEST SUITE any implementation, in any language,
can be held to:

    conformance/vectors.json   JSON test vectors. Rationals are "p/q" strings,
                               the reserved bottom label is "unclear", a
                               missing key means "unclear", `null` is
                               "undefined" (e.g. fidelity when no case needed
                               to move).

    python -m contradish.conformance --check     run production against vectors
    python -m contradish.conformance --generate  regenerate conformance/vectors.json
    python -m contradish.conformance --mutants   show which planted bugs the vectors kill

A suite is only as strong as the bugs it can catch, so MUTANTS is a library
of single-defect scorers (each a plausible implementation mistake) and the
vector set is built so that every mutant fails at least one vector.
"""
import json
import sys
from fractions import Fraction
from functools import partial
from pathlib import Path

from . import reference as ref

VECTORS_PATH = Path(__file__).resolve().parent.parent / "conformance" / "vectors.json"
SPEC_VERSION = "1.0"
TOL = 1e-9
SCALARS = ("fidelity", "hold", "change", "preservation", "authority_respected", "persistence", "revision")
INTS = ("n", "needed", "off", "n_pressured", "n_must_persist", "n_must_revise")


# ── encoding ────────────────────────────────────────────────────────────────

def _enc(x):
    if x is None:
        return None
    f = Fraction(x)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def encode_result(r):
    out = {k: r[k] for k in INTS}
    for k in SCALARS:
        out[k] = _enc(r[k])
    out["faithful"] = bool(r["faithful"])
    out["status"] = dict(r["status"])
    out["broken"] = [list(b) for b in r["broken"]]
    return out


def _same(expected, got):
    if expected is None or got is None:
        return expected is None and got is None
    return abs(float(Fraction(expected)) - float(got)) <= TOL


def diff(expect, got):
    """Fields where `got` (any scorer's output dict) disagrees with a vector's expectation."""
    bad = []
    for k in INTS:
        if got.get(k) != expect[k]:
            bad.append(k)
    for k in SCALARS:
        if not _same(expect[k], got.get(k)):
            bad.append(k)
    if bool(got.get("faithful")) != expect["faithful"]:
        bad.append("faithful")
    if dict(got.get("status", {})) != expect["status"]:
        bad.append("status")
    if [list(b) for b in got.get("broken", [])] != expect["broken"]:
        bad.append("broken")
    return bad


# ── the production adapter ──────────────────────────────────────────────────

def production_scorer(cases, previous, current):
    """Run contradish.transition.evaluate_transition on a plain-dict case list."""
    from .transition import GoverningState, TransitionCase, TransitionContract, evaluate_transition
    tc = [TransitionCase(id=c["id"], question=c["id"], before=c["before"], after=c["after"],
                         grounds=list(c.get("grounds") or []), asserted=c.get("asserted"),
                         authority=c.get("authority")) for c in cases]
    con = TransitionContract(id="conformance", before=GoverningState("before", ""),
                             after=GoverningState("after", ""), outcomes={}, cases=tc)
    o = evaluate_transition(con, previous, current)
    return {"n": len(cases), "needed": o.needed_to_move, "off": o.off_target_after,
            "fidelity": o.fidelity, "hold": o.hold, "change": o.change, "preservation": o.preservation,
            "faithful": o.faithful, "status": o.status, "n_pressured": o.n_pressured,
            "authority_respected": o.authority_respected, "persistence": o.persistence,
            "revision": o.revision, "n_must_persist": o.n_must_persist, "n_must_revise": o.n_must_revise,
            "broken": o.broken_distinctions}


# ── running vectors ─────────────────────────────────────────────────────────

def load_vectors(path=None):
    return json.loads(Path(path or VECTORS_PATH).read_text())


def check_scorer(scorer, vectors):
    """Return [(vector_id, [fields that disagree])] for every failing score vector."""
    fails = []
    for v in vectors["score_vectors"]:
        bad = diff(v["expect"], scorer(v["cases"], v["previous"], v["current"]))
        if bad:
            fails.append((v["id"], bad))
    return fails


def check_sequences(vectors, trajectory_fn=None, hysteresis_fn=None):
    """Optional second conformance level: update semantics and hysteresis."""
    traj = trajectory_fn or ref.warranted_trajectory
    hyst = hysteresis_fn or ref.hysteresis
    fails = []
    for v in vectors["sequence_vectors"]:
        got = traj(v["initial"], v["updates"], v["sources"], v["grounds"])
        if got != v["expect_trajectory"]:
            fails.append((v["id"], "trajectory"))
    for v in vectors["hysteresis_vectors"]:
        if sorted(hyst(v["warranted_final"], v["fresh_final"], v["sequential_final"])) != v["expect"]:
            fails.append((v["id"], "hysteresis"))
    return fails


# ── mutants: single plausible implementation bugs ───────────────────────────

def _status_no_capture_when_was_on(c, p, q):
    s = ref._status(c, p, q)
    return ref.DRIFT if (p == c["after"] and s == ref.CAPTURED) else s


def _status_rigid_ignores_bot(c, p, q):
    s = ref._status(c, p, q)
    return ref.RIGID if s == ref.MISDIRECTED and q == p else s


def _status_moved_beats_held(c, p, q):
    s = ref._status(c, p, q)
    return ref.MOVED if s == ref.HELD else s


def _status_capture_from_off_only(c, p, q):
    s = ref._status(c, p, q)
    return ref.MISDIRECTED if (s == ref.CAPTURED and p != c["after"]) else s


def _pairs_no_grounds_filter(cases):
    return ref.combinations(cases, 2)


def _pairs_empty_grounds_disjoint(cases):
    use = any(c.get("grounds") for c in cases)
    for a, b in ref.combinations(cases, 2):
        if use and not (set(a.get("grounds") or ()) & set(b.get("grounds") or ())):
            continue
        yield a, b


def _pair_ok_bot_is_ok(a, b, cur):
    la, lb = cur.get(a["id"], ref.BOT), cur.get(b["id"], ref.BOT)
    return (la != lb) == (ref.fate(a, b) in (ref.SURVIVE, ref.EMERGE))


def _pair_ok_emerge_not_distinct(a, b, cur):
    la, lb = cur.get(a["id"], ref.BOT), cur.get(b["id"], ref.BOT)
    if ref.BOT in (la, lb):
        return False
    return (la != lb) == (ref.fate(a, b) == ref.SURVIVE)


def _pair_ok_requires_same_labels_as_target(a, b, cur):
    la, lb = cur.get(a["id"], ref.BOT), cur.get(b["id"], ref.BOT)
    return la == a["after"] and lb == b["after"]


def _missing_as_previous(cases, previous, current):
    cur = {c["id"]: current.get(c["id"], previous.get(c["id"], ref.BOT)) for c in cases}
    return ref.score(cases, previous, cur)


def _float_truncation(cases, previous, current):
    r = ref.score(cases, previous, current)
    for k in ("fidelity", "change", "preservation", "hold"):
        if r[k] is not None:
            r[k] = Fraction(int(r[k] * 10) , 10)
    return r


MUTANTS = {
    "fidelity_denominator_is_n": partial(ref.score, hooks={"fidelity": lambda off, needed, n: 1 - Fraction(off, n) if n else None}),
    "fidelity_clipped_at_zero": partial(ref.score, hooks={"fidelity": lambda off, needed, n: max(Fraction(0), 1 - Fraction(off, needed)) if needed else None}),
    "fidelity_zero_when_nothing_needed": partial(ref.score, hooks={"fidelity": lambda off, needed, n: (1 - Fraction(off, needed)) if needed else Fraction(1 if not off else 0)}),
    "preservation_denominator_is_n": partial(ref.score, hooks={"preservation": lambda k, n, needed: Fraction(k, n) if n else None}),
    "change_denominator_off_by_one": partial(ref.score, hooks={"change": lambda moved, needed: Fraction(moved, needed + 1) if needed else None}),
    "hold_always_one": partial(ref.score, hooks={"hold": lambda off, n: Fraction(1)}),
    "authority_never_violated": partial(ref.score, hooks={"authority_respected": lambda c, p: Fraction(1) if p else None}),
    "authority_denominator_is_total_cases": partial(ref.score, hooks={"authority_respected": lambda c, p: 1 - Fraction(c, p + 1) if p else None}),
    "capture_ignored_when_held_before": partial(ref.score, hooks={"status": _status_no_capture_when_was_on}),
    "rigid_includes_unclear": partial(ref.score, hooks={"status": _status_rigid_ignores_bot}),
    "moved_shadows_held": partial(ref.score, hooks={"status": _status_moved_beats_held}),
    "capture_only_when_was_on": partial(ref.score, hooks={"status": _status_capture_from_off_only}),
    "distinctions_ignore_grounds": partial(ref.score, hooks={"pairs": _pairs_no_grounds_filter}),
    "empty_grounds_count_as_disjoint": partial(ref.score, hooks={"pairs": _pairs_empty_grounds_disjoint}),
    "unclear_passes_distinction": partial(ref.score, hooks={"pair_ok": _pair_ok_bot_is_ok}),
    "emerge_not_distinct": partial(ref.score, hooks={"pair_ok": _pair_ok_emerge_not_distinct}),
    "distinction_requires_exact_labels": partial(ref.score, hooks={"pair_ok": _pair_ok_requires_same_labels_as_target}),
    "missing_label_inherits_previous": _missing_as_previous,
    "scores_rounded_down_to_tenths": _float_truncation,
}


# ── vector generation ───────────────────────────────────────────────────────

def _drop_bots(obs, flip):
    """Spell some BOT labels as missing keys, so vectors exercise both spellings."""
    return {k: v for i, (k, v) in enumerate(obs.items()) if not (v == ref.BOT and (i + flip) % 2)}


def _pool():
    grounds = (None, ("g1",), ("g2",))
    for cases in ref.contracts(2, ("a", "b"), grounds):
        yield from ((cases, p, q) for p, q in ref.observations(cases, ("a", "b"), 3))
    for cases in ref.contracts(2, ("a", "b", "c")):
        yield from ((cases, p, q) for p, q in ref.observations(cases, ("a", "b", "c"), 11))
    # 3-case contracts with mixed grounds (sampled): the pair filter's interesting regime
    import itertools
    opts = ref.case_options(("a", "b"))
    for i, combo in enumerate(itertools.product(opts, repeat=3)):
        if i % 37:
            continue
        for gs in ((("g1",), ("g2",), None), (("g1",), ("g1",), ("g2",)), (None, None, ("g1",))):
            cases = [dict(o, id=f"c{k}", **({"grounds": list(g)} if g else {})) for k, (o, g) in enumerate(zip(combo, gs))]
            yield from ((cases, p, q) for p, q in ref.observations(cases, ("a", "b"), 53))


def _vec(i, cases, p, q):
    p2, q2 = _drop_bots(p, 0), _drop_bots(q, 1)
    return {"id": f"s{i:05d}", "cases": cases, "previous": p2, "current": q2,
            "expect": encode_result(ref.score(cases, p2, q2))}


def generate_vectors(base_stride=40):
    """
    Build the vector set: a strided slice of the finite universe plus, for each
    mutant, the earliest pool vector that kills it. Raises if a mutant survives
    the whole pool (the pool would then be too weak to define the spec).
    """
    chosen, killers, i = [], {}, 0
    survivors = dict(MUTANTS)
    seen_sig = set()
    for cases, p, q in _pool():
        v = _vec(i, cases, p, q)
        i += 1
        sig = json.dumps([v["cases"], v["previous"], v["current"]], sort_keys=True)
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        killed_now = [n for n, m in survivors.items() if diff(v["expect"], m(cases, v["previous"], v["current"]))]
        if killed_now or (i % base_stride == 0):
            chosen.append(v)
            for n in killed_now:
                killers[n] = v["id"]
                del survivors[n]
    if survivors:
        raise RuntimeError(f"pool too weak; surviving mutants: {sorted(survivors)}")
    return chosen, killers


def _sequence_vectors():
    labels = ("a", "b")
    ids = ("c0", "c1")
    srcs = {"S": ["*"], "U": ["g1"], "N": []}
    grounds_opts = [{"c0": ["g1"], "c1": ["g2"]}, {"c0": ["g1"], "c1": ["g1"]}, {"c0": ["g2"], "c1": ["g2"]}]
    states = [dict(zip(ids, v)) for v in ref.product(labels, repeat=2)]
    ups = [{"source": s, "asserts": dict(zip(sub, vals))}
           for s in srcs for sub in (("c0",), ("c1",), ("c0", "c1")) for vals in ref.product(labels, repeat=len(sub))]
    out, k = [], 0
    for g in grounds_opts:
        for u1 in ups[::3]:
            for u2 in ups[1::4]:
                for st in states[::2]:
                    upd = [u1, u2]
                    out.append({"id": f"q{k:05d}", "initial": st, "updates": upd, "sources": srcs, "grounds": g,
                                "expect_trajectory": ref.warranted_trajectory(st, upd, srcs, g)})
                    k += 1
    return out[::7]


def _hysteresis_vectors():
    out, k = [], 0
    for wf in ref.product(("a", "b"), repeat=2):
        for ff in ref.product(("a", "b", ref.BOT), repeat=2):
            for sf in ref.product(("a", "b", ref.BOT), repeat=2):
                w, f, s = (dict(zip(("c0", "c1"), x)) for x in (wf, ff, sf))
                out.append({"id": f"h{k:04d}", "warranted_final": w, "fresh_final": f, "sequential_final": s,
                            "expect": ref.hysteresis(w, f, s)})
                k += 1
    return out


def build(path=None):
    chosen, killers = generate_vectors()
    doc = {
        "spec": "Behavioral Update Fidelity", "spec_version": SPEC_VERSION,
        "introduced_by": "Michele Joseph, 2026",
        "conventions": {"unclear": "the reserved bottom label; a missing key means unclear",
                        "rationals": "\"p/q\" strings; null means undefined",
                        "tolerance": "float implementations may differ from the exact rational by <= 1e-9",
                        "case": "{id, before, after, grounds?, asserted?, authority?}"},
        "score_vectors": chosen, "sequence_vectors": _sequence_vectors(),
        "hysteresis_vectors": _hysteresis_vectors(),
        "mutant_killers": killers,
    }
    p = Path(path or VECTORS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def _main(argv):
    if "--generate" in argv:
        d = build()
        print(f"wrote {VECTORS_PATH}: {len(d['score_vectors'])} score, {len(d['sequence_vectors'])} sequence, "
              f"{len(d['hysteresis_vectors'])} hysteresis vectors; {len(d['mutant_killers'])} mutants killed")
        return 0
    vectors = load_vectors()
    if "--mutants" in argv:
        alive = [n for n, m in MUTANTS.items() if not check_scorer(m, vectors)]
        for n, m in MUTANTS.items():
            print(f"{'SURVIVED' if n in alive else 'killed  '} {n}  ({len(check_scorer(m, vectors))} vectors fail)")
        return 1 if alive else 0
    fails = check_scorer(production_scorer, vectors) + check_sequences(vectors)
    print(f"production vs {len(vectors['score_vectors'])} score / {len(vectors['sequence_vectors'])} sequence / "
          f"{len(vectors['hysteresis_vectors'])} hysteresis vectors: {'PASS' if not fails else 'FAIL'}")
    for f in fails[:10]:
        print("  ", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
