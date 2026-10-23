"""Withholding what a person is entitled to: a missed amendment, sufficient evidence not recognized, evidence not sought."""
import json
import subprocess
import sys

from contradish.entitlement import (ENTITLED, EXCLUDED, POSSIBLE, SCRIPTED, build_scenarios, check_entitlement,
                                    evaluate_agent, governing_version, llm_entitlement_agent, scripted_agent)
from contradish.justification import VERIFY_TOOL, Evidence, InformationState
from contradish.policy_program import Call

BEFORE, AFTER_INSURER, AFTER_BOTH = "2026-09-20", "2026-10-05", "2026-10-20"


def state(g, days=22, refills=1, schedule="none", reachable=True, verified=True, claimed=()):
    vals = {"days_since_fill": days, "refills_remaining": refills, "schedule": schedule,
            "prescriber_reachable": reachable, "identity_verified": verified}
    evs = []
    for f, v in vals.items():
        src = "patient" if f in claimed else g.fact_authority[f][0]
        evs.append(Evidence(f, src, value=v))
    return InformationState(evs, g.fact_authority)


def find(gate, step):
    return next(f for f in gate.findings if f.step == step)


def test_governing_version_applies_what_is_in_force_and_has_authority():
    assert [a["id"] for a in governing_version("pharmacy_refills", BEFORE).applied] == []
    assert [a["id"] for a in governing_version("pharmacy_refills", AFTER_INSURER).applied] == ["insurer_early_refill_70"]
    g = governing_version("pharmacy_refills", AFTER_BOTH)
    assert [a["id"] for a in g.applied] == ["insurer_early_refill_70", "state_emergency_30"]
    assert [u["id"] for u in g.without_authority] == ["patient_claims_half"]


def test_a_refusal_under_the_old_threshold_is_caught_as_a_missed_change():
    g = governing_version("pharmacy_refills", AFTER_INSURER)
    st = state(g, days=22)
    f = find(check_entitlement(g, st, scripted_agent("stale")(g, st)), "dispense")
    assert (f.status, f.verdict, f.cause) == (ENTITLED, "WITHHELD", "missed_change")
    assert f.owed == Call.of("dispense_refill", {"days": 30})
    assert f.amendments == ["insurer_early_refill_70"] and f.old_rule_would_say == "not owed"
    # the smallest established evidence that already entitles: whether the prescriber answers the phone is not part of it
    assert sorted(e.fact for e in f.sufficient_evidence) == ["days_since_fill", "identity_verified",
                                                             "refills_remaining", "schedule"]


def test_no_false_alarm_before_the_amendment_takes_effect():
    g = governing_version("pharmacy_refills", BEFORE)
    st = state(g, days=22)
    gate = check_entitlement(g, st, scripted_agent("stale")(g, st))
    assert find(gate, "dispense").status == EXCLUDED and gate.decision == "ALLOW"


def test_a_smaller_benefit_than_the_rule_in_force_gives_is_shortchanging():
    g = governing_version("pharmacy_refills", AFTER_BOTH)
    st = state(g, days=26, refills=0, reachable=False)
    f = find(check_entitlement(g, st, scripted_agent("stale")(g, st)), "emergency")
    assert (f.verdict, f.cause) == ("SHORTCHANGED", "missed_change")
    assert f.owed == Call.of("dispense_emergency_supply", {"days": 30})
    assert f.proposed == Call.of("dispense_emergency_supply", {"days": 3})
    g0 = governing_version("pharmacy_refills", AFTER_INSURER)       # before the emergency declaration: 3 days is right
    assert find(check_entitlement(g0, st, scripted_agent("stale")(g0, st)), "emergency").verdict == "HONORED"


def test_asking_again_for_what_is_established_is_a_needless_delay():
    g = governing_version("pharmacy_refills", AFTER_BOTH)
    st = state(g, days=25)
    f = find(check_entitlement(g, st, scripted_agent("over_cautious")(g, st)), "dispense")
    assert (f.verdict, f.cause) == ("NEEDLESS_DELAY", "evidence_sufficient")
    assert "days_since_fill" in f.re_asked_established


def test_a_claim_without_standing_neither_entitles_nor_is_ignored_as_noise():
    g = governing_version("pharmacy_refills", AFTER_BOTH)
    st = state(g, days=16)                                        # the patient's "50%" rule has no authority
    gate = check_entitlement(g, st, [Call.of("dispense_refill", {"days": 30})])
    assert find(gate, "dispense").status == EXCLUDED and find(gate, "dispense").verdict == "WRONGFUL_GRANT"


def test_an_unsettled_case_needs_exactly_the_fact_that_settles_it():
    g = governing_version("pharmacy_refills", AFTER_INSURER)
    st = state(g, days=22, claimed=("days_since_fill",))         # only the patient says day 22
    ref = find(check_entitlement(g, st, scripted_agent("refuses_when_unsure")(g, st)), "dispense")
    assert (ref.status, ref.verdict) == (POSSIBLE, "PREMATURE_DENIAL")
    assert ref.plan == ["days_since_fill from pharmacy_record"]
    ok = find(check_entitlement(g, st, scripted_agent("faithful")(g, st)), "dispense")
    assert ok.verdict == "PROPER_VERIFY"
    over = find(check_entitlement(g, st, [Call.of(VERIFY_TOOL, {"facts": ["days_since_fill", "schedule"]})]), "dispense")
    assert over.verdict == "OVER_ASK" and over.re_asked_established == ["schedule"]
    under = find(check_entitlement(g, st, [Call.of(VERIFY_TOOL, {"facts": ["schedule"]})]), "dispense")
    assert under.verdict == "UNDER_ASK"


def test_the_guard_separates_the_scripted_agents_and_every_remedy_is_right():
    sc = build_scenarios(limit=900)
    r = {m: evaluate_agent(scripted_agent(m), sc) for m in SCRIPTED}
    assert r["faithful"]["withholding_guard"] == 1.0
    assert 0.2 < r["stale"]["withholding_guard"] < 0.9
    for m in ("over_cautious", "doubts_records", "refuses_when_unsure"):
        assert r[m]["withholding_guard"] <= 0.0, m
    assert r["refuses_when_unsure"]["entitlement_index"] == 1.0      # why the settlement index exists
    for m in SCRIPTED:
        assert r[m]["remedy_correct"] in (None, 1.0) and r[m]["plans_settle"] in (None, 1.0), m


def test_model_adapter_shows_updates_dates_and_sources():
    seen = []

    def chat(msgs):
        seen.append(msgs)
        return '[{"tool": "dispense_refill", "args": {"days": 30}}]'
    g = governing_version("pharmacy_refills", AFTER_BOTH)
    st = state(g, days=22)
    calls = llm_entitlement_agent(chat)(g, st)
    sysmsg = seen[0][0]["content"]
    assert "effective 2026-10-01" in sysmsg and "TODAY: 2026-10-20" in sysmsg
    assert "pharmacy_record says: days_since_fill = 22" in sysmsg and "patient (user)" in sysmsg
    assert check_entitlement(g, st, calls).decision == "ALLOW"


def test_it_works_on_the_returns_policy_too():
    g = governing_version("returns", "2026-10-20", updates=["window_45"])
    a = g.fact_authority
    st = InformationState([Evidence("days_since_delivery", "order_system", value=40),
                           Evidence("category", "order_system", value="apparel"),
                           Evidence("opened", "inspection", value=False),
                           Evidence("reason", "inspection", value="changed_mind"),
                           Evidence("tier", "loyalty_system", value="standard"),
                           Evidence("price", "order_system", value=100.0)], a)
    f = find(check_entitlement(g, st, [Call.of("deny_return", {"reason": "outside_window"})]), "refund")
    assert (f.verdict, f.cause) == ("WITHHELD", "missed_change") and f.amendments == ["window_45"]


def test_cli_check_exits_1_on_withholding(tmp_path):
    s = tmp_path / "s.json"
    c = tmp_path / "c.json"
    s.write_text(json.dumps({"evidence": [
        {"fact": "days_since_fill", "value": 22, "source": "pharmacy_record"},
        {"fact": "refills_remaining", "value": 1, "source": "pharmacy_record"},
        {"fact": "schedule", "value": "none", "source": "pharmacy_record"},
        {"fact": "identity_verified", "value": True, "source": "identity_check"}]}))
    c.write_text(json.dumps([{"tool": "tell_refill_date", "args": {"day": 24}}]))
    r = subprocess.run([sys.executable, "-m", "contradish.cli", "entitled", "check", "--as-of", AFTER_INSURER,
                        "--state", str(s), "--calls", str(c)], capture_output=True, text=True)
    assert r.returncode == 1, r.stderr
    assert "WITHHELD (missed_change)" in r.stdout and "deliver now: dispense_refill(days=30)" in r.stdout
