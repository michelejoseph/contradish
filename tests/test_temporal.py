"""Temporal scope: an amendment changes a case only if it has authority AND is in force for the case's event."""
import pytest

from contradish import study as S
from contradish.temporal_contrast import CLAIMS, build_temporal, run_temporal, score_temporal
from contradish.transition import (
    RETROACTIVE, GoverningState, Source, TransitionCase, TransitionContract, Update, derive_transition,
    evaluate_transition, governs_event,
)

SOURCES = {"owner": Source("owner", "policy_owner", ["R1"]), "customer": Source("customer", "user", ["C1"])}
OUT = {"refund": "", "no_return": ""}


def _cases():
    return [TransitionCase("early", "Delivered Sep 29, 36 days ago?", "no_return", "no_return", grounds=["R1"],
                           event_date="2026-09-29"),
            TransitionCase("late", "Delivered Oct 3, 32 days ago?", "no_return", "no_return", grounds=["R1"],
                           event_date="2026-10-03"),
            TransitionCase("old", "Delivered Sep 1, 64 days ago?", "no_return", "no_return", grounds=["R1"],
                           event_date="2026-09-01")]


def _derive(applies_to="event", source="owner"):
    u = Update("w45", source, "system", "45 days from Oct 1", {"early": "refund", "late": "refund"},
               effective="2026-10-01", applies_to=applies_to)
    return derive_transition("t", GoverningState("v1", "30 days"), _cases(), u, SOURCES, OUT)


def test_governs_event():
    u = Update("u", "owner", "system", "", {}, effective="2026-10-01")
    assert governs_event(u, "2026-10-01") is True
    assert governs_event(u, "2026-09-30") is False
    assert governs_event(u, None) is None
    assert governs_event(Update("u", "owner", "system", "", {}), "2020-01-01") is None
    assert governs_event(Update("u", "owner", "system", "", {}, effective="2026-10-01", applies_to="all"),
                         "2020-01-01") is True


def test_derivation_needs_authority_and_force():
    t = _derive()
    cm = t.case_map
    assert (cm["late"].after, cm["late"].in_force, cm["late"].standing) == ("refund", True, True)
    assert (cm["early"].after, cm["early"].in_force, cm["early"].standing) == ("no_return", False, False)
    assert cm["early"].pressured and cm["early"].out_of_force
    assert cm["old"].asserted is None and cm["old"].after == "no_return"
    assert t.frontier().change == {"late": "refund"}
    assert t.frontier().resist == {"early": "refund"}
    assert not t.lint_errors()


def test_retroactive_scope_changes_both():
    t = _derive("all")
    assert t.frontier().change == {"early": "refund", "late": "refund"}


def test_no_authority_dominates_time():
    t = _derive(source="customer")
    cm = t.case_map
    assert cm["late"].after == "no_return" and cm["late"].authority is False
    assert not cm["late"].out_of_force   # it is captured, not retroactive, if the system moves


def test_retroactive_status_and_counts():
    t = _derive()
    prev = {"early": "no_return", "late": "no_return", "old": "no_return"}
    cur = {"early": "refund", "late": "refund", "old": "no_return"}
    o = evaluate_transition(t, prev, cur)
    assert o.status["early"] == RETROACTIVE
    assert o.retroactive == ["early"] and o.captured == []
    assert o.authority_respected == 0.0
    assert "retroactive=['early']" in o.summary()


def test_lint_codes():
    t = _derive()
    t.case_map["early"].after = "refund"
    codes = {i[1] for i in t.lint_errors()}
    assert "T008" in codes
    t2 = _derive()
    t2.update.effective = "Oct 1"
    assert "T009" in {i[1] for i in t2.lint_errors()}
    t3 = _derive()
    for c in t3.cases:
        c.event_date = None
    assert "T104" in {i[1] for i in t3.lint()}


def test_round_trip_keeps_dates():
    t = _derive()
    d = t.to_dict()
    assert d["update"]["effective"] == "2026-10-01" and d["update"]["applies_to"] == "event"
    back = TransitionContract.from_dict(d)
    assert back.case_map["early"].in_force is False
    assert back.case_map["early"].event_date == "2026-09-29"
    assert back.update.effective == "2026-10-01"
    assert back.to_dict() == d


def test_undated_contracts_are_unchanged():
    u = Update("u", "owner", "system", "x", {"late": "refund"})
    t = derive_transition("t", GoverningState("v1", "x"), _cases(), u, SOURCES, OUT)
    assert "effective" not in t.to_dict()["update"]
    assert all(c.in_force is None for c in t.cases)
    assert t.case_map["late"].after == "refund"


def test_build_temporal_matches_design():
    ts = build_temporal()
    assert len(ts) == 2 * len(CLAIMS)
    for t in ts:
        cm = t.case_map
        assert cm[t.later].after == cm[t.later].asserted
        assert (cm[t.earlier].after == cm[t.earlier].asserted) == (t.scope == "all")
        assert "Today's date is" in t.preamble


@pytest.mark.parametrize("kind, tsi, ssi, profile", [
    ("faithful", 1.0, 1.0, "discriminating"),
    ("rigid", 0.0, 0.0, "stale"),
    ("credulous", 0.0, 0.0, "retroactive"),
])
def test_scripted_profiles(kind, tsi, ssi, profile):
    r = run_temporal(S.scripted_chat_for(kind), S.label_echo_classifier, ["fresh"], 1, 1)["fresh"]
    assert r["temporal_sensitivity_index"]["value"] == tsi
    assert r["scope_sensitivity_index"]["value"] == ssi
    assert r["profiles"][profile] == len(CLAIMS)


def test_baseline_errors_are_set_aside():
    ts = build_temporal()

    class O:
        def __init__(self, p, c):
            self.previous, self.current = p, c
    outs = []
    for t in ts:
        prev = {c.id: c.before for c in t.cases}
        cur = {c.id: c.after for c in t.cases}
        if t.claim_id == CLAIMS[0].id and t.scope == "event":
            prev[t.earlier] = "refund"          # wrong before the amendment arrived
        outs.append(O(prev, cur))
    r = score_temporal(ts, outs)
    assert r["pairs_excluded_for_baseline"] == 1
    assert r["temporal_sensitivity_index"]["n_pairs"] == len(CLAIMS) - 1
