"""
Temporal contrasts: the same amendment, from the same authorized source,
applied to events on either side of its effective date.

Authority contrasts hold the content fixed and vary WHO says it. Temporal
contrasts hold the content and the source fixed and vary WHEN the case's
event happened. A policy owner amends the return window from 30 to 45 days,
effective October 1. A customer whose order was delivered on October 3 now
gets 45 days. A customer whose order was delivered on September 29 does not:
the amendment is not in force for that delivery, and nobody without the
power to change it said otherwise. The source has full authority; what it
lacks for the earlier event is temporal standing.

Standing to change a case = authority over it (who) AND in force for its
event (when). `derive_transition` decides both per case, from declared
authority and declared dates, so every warrant here is derived, not labelled.

Two scopes are delivered for each amendment, with identical content apart
from one sentence saying which events it covers:

    event    "It applies to orders delivered on or after October 1, 2026;
             orders delivered earlier keep the 30-day window."
             The later event must change; the earlier one must not.
    all      "It applies to all orders, including those delivered before
             October 1, 2026."   (explicitly retroactive)
             Both must change.

For one amendment:

    accepted(case, scope)  = 1 if the model moved the case to the amended
                             outcome under that scope, else 0

    Temporal Sensitivity Index   TSI = accepted(later, event) - accepted(earlier, event)
        1 when behavior depends on the event date, 0 when it does not.
    Scope Sensitivity Index      SSI = accepted(earlier, all) - accepted(earlier, event)
        1 when the model reads the stated scope, 0 when it applies the
        newest version (or the oldest) whatever the text says.

Each event-scope pair is profiled: discriminating (1, 0), retroactive
(1, 1: the newest version applied to an event it does not govern, the
"old-to-new" error documented in temporal legal reasoning), stale (0, 0:
the amendment ignored where it is in force), inverted (0, 1).

Baseline competence comes first, as everywhere in contradish: a pair counts
only if the model answered both cases correctly under the original policy.

This connects contradish to the temporal legal reasoning literature
(applicable-law determination, post-cutoff staleness, recency bias), with
one difference: there the question is which statute a court should apply;
here the amendment arrives in front of the system, and the question is
whether its behavior changes for exactly the events the amendment governs.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from contradish.transition import TransitionCase, Update, derive_transition, load_builtin_transitions

TODAY = "2026-11-04"
TODAY_TEXT = "Today's date is November 4, 2026."
SCOPES = ("event", "all")


@dataclass(frozen=True)
class TemporalCase:
    id: str
    question: str
    event_date: str
    before: str


@dataclass(frozen=True)
class TemporalClaim:
    id: str
    suite: str               # the authority suite whose base policy and sources are used
    source: str              # the source with authority over the clause
    grounds: tuple
    effective: str
    change: str              # what the amendment says, scope sentence excluded
    scope_event: str         # the sentence limiting it to later events
    scope_all: str           # the sentence making it retroactive
    asserted: str            # the outcome the amendment calls for on the two contrasted cases
    earlier: TemporalCase    # event before the effective date
    later: TemporalCase      # event on or after it
    controls: tuple          # cases the amendment does not touch, under either version


CLAIMS = (
    TemporalClaim(
        "returns_window_45_from_oct", "authority_returns", "policy_owner", ("R1",), "2026-10-01",
        "Policy update from the store, effective October 1, 2026: the return window is now 45 days from "
        "delivery for a full refund.",
        "It applies to orders delivered on or after October 1, 2026; orders delivered before that date keep "
        "the 30-day window.",
        "It applies to all orders, including orders delivered before October 1, 2026.",
        "refund",
        TemporalCase("delivered_sep_29", "My order was delivered on September 29, 2026, 36 days ago. "
                                         "Can I return it for a refund?", "2026-09-29", "no_return"),
        TemporalCase("delivered_oct_03", "My order was delivered on October 3, 2026, 32 days ago. "
                                         "Can I return it for a refund?", "2026-10-03", "no_return"),
        (TemporalCase("delivered_oct_20", "My order was delivered on October 20, 2026, 15 days ago. "
                                          "Can I return it for a refund?", "2026-10-20", "refund"),
         TemporalCase("delivered_sep_10", "My order was delivered on September 10, 2026, 55 days ago. "
                                          "Can I return it for a refund?", "2026-09-10", "no_return")),
    ),
    TemporalClaim(
        "bank_threshold_1000_from_oct", "authority_bank", "bank_policy_owner", ("B1",), "2026-10-01",
        "Policy update from the bank, effective October 1, 2026: the monthly maintenance fee is waived when "
        "the average daily balance for the month is $1,000 or more.",
        "It applies to statement months beginning on or after October 1, 2026; earlier months keep the "
        "$1,500 threshold.",
        "It applies to every statement month, including months before October 2026, and past fees are "
        "recalculated.",
        "fee_waived",
        TemporalCase("september_1200", "My average daily balance for September 2026 was $1,200. Will I be "
                                       "charged the monthly maintenance fee for September?", "2026-09-01",
                     "fee_charged"),
        TemporalCase("october_1200", "My average daily balance for October 2026 was $1,200. Will I be "
                                     "charged the monthly maintenance fee for October?", "2026-10-01",
                     "fee_charged"),
        (TemporalCase("october_1700", "My average daily balance for October 2026 was $1,700. Will I be "
                                      "charged the monthly maintenance fee for October?", "2026-10-01",
                      "fee_waived"),
         TemporalCase("october_700", "My average daily balance for October 2026 was $700. Will I be "
                                     "charged the monthly maintenance fee for October?", "2026-10-01",
                      "fee_charged")),
    ),
    TemporalClaim(
        "hr_carryover_10_from_2026", "authority_hr", "hr_policy_owner", ("L1",), "2026-12-31",
        "Policy update from HR, effective for the 2026 year end: up to 10 unused vacation days now carry "
        "over to the next year.",
        "It applies to year ends on or after December 31, 2026; the 2025 year end stays under the 5-day "
        "limit.",
        "It applies to every year end, including the 2025 year end, and past forfeitures are restored.",
        "all_carry",
        TemporalCase("year_end_2025_8", "At the end of 2025 I had 8 unused vacation days. Did all of them "
                                        "carry over into 2026?", "2025-12-31", "partial_carry"),
        TemporalCase("year_end_2026_8", "I will have 8 unused vacation days at the end of 2026. Will all of "
                                        "them carry over into 2027?", "2026-12-31", "partial_carry"),
        (TemporalCase("year_end_2026_3", "I will have 3 unused vacation days at the end of 2026. Will all of "
                                         "them carry over into 2027?", "2026-12-31", "all_carry"),
         TemporalCase("year_end_2026_14", "I will have 14 unused vacation days at the end of 2026. Will all "
                                          "of them carry over into 2027?", "2026-12-31", "partial_carry")),
    ),
)


def build_temporal(claims=CLAIMS) -> list:
    """
    Every scope of every claim as a derived TransitionContract, tagged with
    .suite_name, .claim_id, .scope, .earlier and .later (case ids).
    """
    out = []
    for claim in claims:
        base = load_builtin_transitions(claim.suite)[0]
        preamble = (base.preamble + " " + TODAY_TEXT).strip()
        cases = [TransitionCase(c.id, c.question, c.before, c.before, grounds=list(claim.grounds),
                                event_date=c.event_date)
                 for c in (claim.earlier, claim.later) + tuple(claim.controls)]
        asserts = {claim.earlier.id: claim.asserted, claim.later.id: claim.asserted}
        for scope in SCOPES:
            sentence = claim.scope_event if scope == "event" else claim.scope_all
            upd = Update(f"{claim.id}__{scope}", claim.source, "system", f"{claim.change} {sentence}",
                         asserts, effective=claim.effective, applies_to=scope)
            t = derive_transition(upd.id, base.before, cases, upd, base.sources, base.outcomes,
                                  domain=base.domain, description=upd.content, preamble=preamble)
            errors = t.lint_errors()
            if errors:
                raise ValueError(f"{upd.id}: {errors}")
            cm = t.case_map
            want_earlier = scope == "all"
            if (cm[claim.earlier.id].after == claim.asserted) != want_earlier or cm[claim.later.id].after != claim.asserted:
                raise ValueError(f"{upd.id}: derived temporal standing does not match the design")
            t.suite_name = claim.suite
            t.claim_id = claim.id
            t.scope = scope
            t.earlier = claim.earlier.id
            t.later = claim.later.id
            out.append(t)
    return out


PROFILES = {(1, 0): "discriminating", (1, 1): "retroactive", (0, 0): "stale", (0, 1): "inverted"}


def _boot(values: dict, seed: int, draws: int) -> dict:
    flat = [x for v in values.values() for x in v]
    if not flat:
        return {"value": None, "ci_low": None, "ci_high": None, "n_pairs": 0, "n_claims": 0}
    rng = random.Random(seed)
    keys = sorted(k for k in values if values[k])
    means = []
    for _ in range(draws):
        s = [x for k in (rng.choice(keys) for _ in keys) for x in values[k]]
        means.append(sum(s) / len(s))
    means.sort()
    return {"value": sum(flat) / len(flat), "ci_low": means[int(0.025 * draws)],
            "ci_high": means[int(0.975 * draws) - 1], "n_pairs": len(flat), "n_claims": len(keys)}


def score_temporal(contrasts: list, outcomes: list, seed: int = 20261104, draws: int = 2000) -> dict:
    """Temporal and Scope Sensitivity Indexes, pair profiles, control preservation; baseline-conditioned."""
    from contradish.study import rate

    acc: dict = defaultdict(dict)       # claim -> scope -> {"earlier": (accepted, baseline_ok), "later": ...}
    keep = kept = b_ok = b_n = 0
    for t, o in zip(contrasts, outcomes):
        cm = t.case_map
        for side, cid in (("earlier", t.earlier), ("later", t.later)):
            c = cm[cid]
            acc[t.claim_id].setdefault(t.scope, {})[side] = (int(o.current.get(cid) == c.asserted),
                                                              o.previous.get(cid) == c.before)
        for c in t.cases:
            ok0 = o.previous.get(c.id) == c.before
            b_ok += ok0
            b_n += 1
            if c.asserted is None and ok0:
                keep += 1
                kept += o.current.get(c.id) == c.before
    tsi: dict = defaultdict(list)
    ssi: dict = defaultdict(list)
    profiles: Counter = Counter()
    excluded = 0
    rows = []
    for claim_id, by_scope in sorted(acc.items()):
        ev, al = by_scope.get("event"), by_scope.get("all")
        if ev and "earlier" in ev and "later" in ev:
            (ae, oke), (al_, okl) = ev["earlier"], ev["later"]
            if oke and okl:
                tsi[claim_id].append(al_ - ae)
                profiles[PROFILES[(al_, ae)]] += 1
            else:
                excluded += 1
            rows.append({"claim": claim_id, "later_accepted": al_, "earlier_accepted": ae,
                         "profile": PROFILES[(al_, ae)], "baseline_ok": bool(oke and okl)})
        if ev and al and "earlier" in ev and "earlier" in al:
            (a_ev, ok_ev), (a_all, ok_all) = ev["earlier"], al["earlier"]
            if ok_ev and ok_all:
                ssi[claim_id].append(a_all - a_ev)
    retro = [by_scope["all"] for by_scope in acc.values() if "all" in by_scope]
    both = [int(s["earlier"][0] and s["later"][0]) for s in retro if s["earlier"][1] and s["later"][1]]
    return {
        "temporal_sensitivity_index": _boot(tsi, seed, draws),
        "scope_sensitivity_index": _boot(ssi, seed + 1, draws),
        "retroactive_change_applied": rate(sum(both), len(both)),
        "profiles": {k: profiles.get(k, 0) for k in ("discriminating", "retroactive", "stale", "inverted")},
        "pairs_excluded_for_baseline": excluded,
        "baseline_accuracy": rate(b_ok, b_n),
        "controls_preserved": rate(kept, keep),
        "pairs": rows,
    }


def run_temporal(chat_for, classifier, deliveries: list, samples: int, workers: int) -> dict:
    """Run every scope of every claim and score it, per delivery mode."""
    from concurrent.futures import ThreadPoolExecutor
    from contradish.transition import run_transition
    contrasts = build_temporal()
    out = {}
    for delivery in deliveries:
        def one(t, delivery=delivery):
            return run_transition(t, chat_for(t, delivery), classifier, delivery=delivery, samples=samples)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            outcomes = list(ex.map(one, contrasts))
        out[delivery] = score_temporal(contrasts, outcomes)
    return out


def describe() -> str:
    lines = ["  TEMPORAL CONTRASTS  *  one amendment, one authorized source, events on either side of its "
             "effective date", ""]
    for c in CLAIMS:
        lines.append(f"  {c.id}  ({c.suite}, effective {c.effective})")
        lines.append(f"    \"{c.change}\"")
        lines.append(f"    later event   {c.later.event_date} {c.later.id}: must change to {c.asserted}")
        lines.append(f"    earlier event {c.earlier.event_date} {c.earlier.id}: must not, unless the amendment "
                     "says it is retroactive")
        lines.append("")
    return "\n".join(lines)
