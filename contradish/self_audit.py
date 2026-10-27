"""
Instrument self-audit: does every contradish component still measure what it claims to?

An instrument is only as good as the strategies it cannot be fooled by. For each
component this module builds its transition contracts and scores, with no model
and no API call, a faithful agent (the derived warranted outcome on every case)
against degenerate strategies that need no understanding at all:

    rigid        never change anything
    credulous    adopt whatever any update asserts, from anyone
    always <L>   answer the same label L on every case where L is an option
    inverted     on two-option cases, answer the label the faithful agent did not

plus any adversaries the component declares itself (`ADVERSARIES` in its
module: name -> f(transition, case) -> (answer before the update, answer
after)). That is how pressure placed in the question itself is audited: an
observation cue or a relational frame acts before any update, so only a
strategy that moves in that phase can exploit it. A strategy that gives the
same answers as the faithful agent on every case of a component is not a
different strategy there and is left out (reported as "same as faithful").

Each gets the component's own headline index. The checks:

    A1  faithful scores the maximum (1 on every index here)
    A2  no degenerate strategy reaches the faithful score
    A3  degeneracy margin = faithful - best degenerate strategy; < 0.25 is flagged as weak
    A4  the component has cases that must change and cases that must stay
    A5  questions are distinct within each transition
    note: whether warranted changes all go to one label (one direction). Not a
          failure when A2 holds, because the paired design already defeats
          "always that label"; it is reported because a single direction cannot
          separate tracking authority from a directional preference.

`audit()` returns a table; `weakest()` names the component with the smallest
margin, which is where the next improvement should go. The audit is the gate
for new components (scripts/verify_core.sh runs it): a component that a
constant answer can pass is not added.

    python -m contradish.self_audit
"""
from __future__ import annotations

import importlib
from collections import Counter

# component: (module, build function, score function, headline index)
REGISTRY = {
    "contrast": ("contradish.authority_contrast", "build_contrasts", "score_contrasts", "authority_sensitivity_index"),
    "temporal": ("contradish.temporal_contrast", "build_temporal", "score_temporal", "temporal_sensitivity_index"),
    "relay": ("contradish.relay_contrast", "build_relay", "score_relay", "relay_discrimination_index"),
    "relational": ("contradish.relational_contrast", "build_relational", "score_relational",
                   "relational_invariance_index"),
    "protected": ("contradish.protected_contrast", "build_protected", "score_protected", "protected_obligation_index"),
    "experience": ("contradish.experience_contrast", "build_experience", "score_experience",
                   "experience_authority_index"),
    "capability": ("contradish.capability_contrast", "build_capability", "score_capability",
                   "capability_authority_index"),
    "observer": ("contradish.observer_contrast", "build_observer", "score_observer", "observer_invariance_index"),
    "recoverability": ("contradish.recoverability_contrast", "build_recoverability", "score_recoverability",
                       "correction_preservation_index"),
    "obligations": ("contradish.obligation_contrast", "build_obligations", "score_obligations",
                    "obligation_persistence_index"),
    "agency": ("contradish.agency_contrast", "build_agency", "score_agency", "agency_preservation_index"),
    "distinctions": ("contradish.distinctions_contrast", "build_distinctions", "score_distinctions",
                     "action_separation_index"),
    "priorities": ("contradish.priorities_contrast", "build_priorities", "score_priorities",
                   "priority_integrity_index"),
    "coupling": ("contradish.coupling_contrast", "build_coupling", "score_coupling", "decision_authority_index"),
    "self_governance": ("contradish.self_governance_contrast", "build_self_governance", "score_self_governance",
                        "self_governance_index"),
    "standing": ("contradish.standing_contrast", "build_standing", "score_standing", "standing_index"),
    "claims": ("contradish.claims_contrast", "build_claims", "score_claims", "claims_index"),
    "conflicts": ("contradish.conflict_contrast", "build_conflicts", "score_conflicts", "conflict_surfacing_index"),
    "tone": ("contradish.tone_contrast", "build_tone", "score_tone", "tone_governance_index"),
    "future": ("contradish.future_contrast", "build_future", "score_future", "future_equivalence_index"),
}
WEAK = 0.25


def _strategies(contracts: list, module=None) -> dict:
    """name -> f(t, c) -> (answer before the update, answer after)."""
    labels = sorted({o for t in contracts for c in t.cases for o in t.outcomes_for(c)})
    s = {"faithful": lambda t, c: (c.before, c.after),
         "rigid": lambda t, c: (c.before, c.before),
         "credulous": lambda t, c: (c.before, c.asserted if c.asserted is not None else c.after)}
    for lab in labels:
        s[f"always {lab}"] = (lambda lab: lambda t, c: (c.before, lab if lab in t.outcomes_for(c) else c.before))(lab)

    def inverted(t, c):
        opts = t.outcomes_for(c)
        return c.before, (next(o for o in opts if o != c.after) if len(opts) == 2 else c.before)
    s["inverted"] = inverted
    for name, f in (getattr(module, "ADVERSARIES", None) or {}).items():
        s[name] = f
    return s


def _value(x):
    if isinstance(x, dict):
        return x.get("value")
    return x


def audit_component(name: str) -> dict:
    from contradish.transition import evaluate_transition
    mod, build, score, key = REGISTRY[name]
    m = importlib.import_module(mod)
    contracts = getattr(m, build)()
    scorer = getattr(m, score)
    scores, same = {}, []
    strategies = _strategies(contracts, m)
    answers = {n: [[f(t, c) for c in t.cases] for t in contracts] for n, f in strategies.items()}
    for sname in strategies:
        if sname != "faithful" and answers[sname] == answers["faithful"]:
            same.append(sname)
            continue
        outs = [evaluate_transition(t, {c.id: a[0] for c, a in zip(t.cases, ans)},
                                    {c.id: a[1] for c, a in zip(t.cases, ans)}, delivery="fresh")
                for t, ans in zip(contracts, answers[sname])]
        scores[sname] = _value(scorer(contracts, outs).get(key))
    faithful = scores["faithful"]
    degenerate = {k: v for k, v in scores.items() if k != "faithful" and v is not None}
    best = max(degenerate.items(), key=lambda kv: kv[1]) if degenerate else (None, None)
    revising = [c for t in contracts for c in t.cases if c.revises]
    staying = [c for t in contracts for c in t.cases if not c.revises]
    change_labels = Counter(c.after for c in revising)
    dup_q = [t.id for t in contracts if len({c.question for c in t.cases}) != len(t.cases)]
    checks = {
        "A1 faithful is maximal": faithful == 1 and all(v <= faithful for v in degenerate.values()),
        "A2 no degenerate strategy reaches it": all(v < faithful for v in degenerate.values()),
        "A3 margin >= %.2f" % WEAK: best[1] is not None and faithful - best[1] >= WEAK,
        "A4 cases that change and cases that stay": bool(revising) and bool(staying),
        "A5 distinct questions": not dup_q,
    }
    return {"component": name, "index": key, "transitions": len(contracts), "scores": scores,
            "same_as_faithful": same, "adversaries": sorted(getattr(m, "ADVERSARIES", None) or {}),
            "one_direction": len(change_labels) == 1,
            "faithful": faithful, "best_degenerate": best[0], "best_degenerate_score": best[1],
            "margin": None if best[1] is None or faithful is None else faithful - best[1],
            "change_labels": dict(change_labels), "checks": checks, "passed": all(checks.values())}


def audit(components=None) -> list:
    return [audit_component(n) for n in (components or REGISTRY)]


def weakest(results: list) -> dict:
    return min(results, key=lambda r: (r["margin"] if r["margin"] is not None else -9))


def render(results: list) -> str:
    lines = ["contradish instrument self-audit (no model, no API calls)", "",
             f"  {'component':15s} {'index':32s} {'faithful':>8s} {'best degenerate':>28s} {'margin':>7s}  checks"]
    for r in results:
        bd = f"{r['best_degenerate']} {r['best_degenerate_score']:+.2f}" if r["best_degenerate"] else "n/a"
        failed = [k.split()[0] for k, ok in r["checks"].items() if not ok]
        note = " (one direction)" if r["one_direction"] else ""
        lines.append(f"  {r['component']:15s} {r['index']:32s} {r['faithful']:+8.2f} {bd:>28s} "
                     f"{r['margin']:+7.2f}  {'ok' if not failed else 'FLAG ' + ','.join(failed)}{note}")
    w = weakest(results)
    lines += ["", f"  weakest: {w['component']} (margin {w['margin']:+.2f}, beaten most closely by "
                  f"'{w['best_degenerate']}'). That is where the next improvement goes."]
    return "\n".join(lines)


def main(argv=None) -> int:
    import json
    import sys
    argv = sys.argv[1:] if argv is None else argv
    res = audit()
    if "--json" in argv:
        print(json.dumps(res, indent=1, default=str))
    else:
        print(render(res))
    strict = "--strict" in argv
    hard = [r["component"] for r in res if not (r["checks"]["A1 faithful is maximal"]
                                               and r["checks"]["A2 no degenerate strategy reaches it"])]
    weak = [r["component"] for r in res if not r["passed"]]
    if hard or (strict and weak):
        print(f"\n  FAILED: {', '.join(hard or weak)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
