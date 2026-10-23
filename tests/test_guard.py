"""The guard: one call before tool calls run, certificates the stdlib checker re-derives, and authoring checks."""
import copy
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

from contradish import entitlement as E
from contradish.guard import Guard, calls_from, check_policy, evidence_from, run_examples
from contradish.policy_program import Call, PolicyProgram, load_builtin_program
from contradish.policy_source import parse

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("evidence_check", ROOT / "contradish" / "evidence_check.py")
CHECK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CHECK)

FULL = {"days_since_fill": (22, "pharmacy_record"), "refills_remaining": (1, "pharmacy_record"),
        "schedule": ("none", "pharmacy_record"), "prescriber_reachable": (True, "prescriber_office"),
        "identity_verified": (True, "identity_check")}
STALE = [{"tool": "tell_refill_date", "args": {"day": 24}}, {"tool": "notify_patient", "args": {"outcome": "not_yet_due"}}]
RIGHT = [{"tool": "dispense_refill", "args": {"days": 30}}, {"tool": "notify_patient", "args": {"outcome": "refill_ready"}}]


def test_adapters_read_openai_and_anthropic_tool_calls():
    openai_msg = {"role": "assistant", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "dispense_refill", "arguments": '{"days": 30}'}}]}
    openai_obj = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(name="dispense_refill",
                                                                                         arguments='{"days": 30}'))])
    anthropic = [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t", "name": "dispense_refill",
                                                  "input": {"days": 30}}]
    anthropic_msg = SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="dispense_refill", input={"days": 30})])
    want = [Call.of("dispense_refill", {"days": 30})]
    for x in (openai_msg, openai_obj, anthropic, anthropic_msg, [{"tool": "dispense_refill", "args": {"days": 30}}], want):
        assert calls_from(x) == want


def test_evidence_forms():
    a = evidence_from({"days_since_fill": (22, "pharmacy_record")})
    b = evidence_from([{"fact": "days_since_fill", "value": 22, "source": "pharmacy_record"}])
    c = evidence_from({"days_since_fill": 22}, default_source="pharmacy_record")
    assert a[0].to_dict() == b[0].to_dict() == c[0].to_dict()


def test_a_stale_refusal_is_delivered_and_enforce_runs_what_the_policy_requires():
    d = Guard("pharmacy_refills", mode="enforce").check(FULL, STALE, as_of="2026-10-20")
    assert d.outcome == "deliver" and d.owed == [Call.of("dispense_refill", {"days": 30})]
    assert any("insurer_early_refill_70" in r for r in d.reasons)
    assert sorted(c.tool for c in d.calls_to_run(deliver_owed=True)) == ["dispense_refill", "notify_patient"]
    assert Call.of("notify_patient", {"outcome": "refill_ready"}) in d.calls_to_run(deliver_owed=True)


def test_acting_on_a_claim_is_blocked_with_the_one_fact_to_ask_for():
    ev = dict(FULL, days_since_fill=(22, "patient"))
    d = Guard("pharmacy_refills", mode="enforce").check(ev, RIGHT, as_of="2026-10-20")
    assert d.outcome == "block" and d.ask == [{"fact": "days_since_fill", "from": ["pharmacy_record"]}]
    assert d.calls_to_run() == []


def test_shadow_mode_never_stops_anything():
    d = Guard("pharmacy_refills", mode="shadow").check(FULL, STALE, as_of="2026-10-20")
    assert d.outcome == "deliver" and d.calls_to_run() == calls_from(STALE)


def test_the_right_action_is_allowed():
    assert Guard("pharmacy_refills").check(FULL, RIGHT, as_of="2026-10-20").outcome == "allow"


def test_every_logged_decision_is_reverified_independently(tmp_path):
    log = tmp_path / "d.jsonl"
    g = Guard("pharmacy_refills", log=str(log))
    rng = random.Random(3)
    sc = E.build_scenarios(limit=240)
    for s in sc:
        gv = E.governing_version("pharmacy_refills", s["as_of"])
        calls = E.scripted_agent(rng.choice(E.SCRIPTED))(gv, s["state"])
        if rng.random() < 0.3:
            calls = [c for c in gv.current.run(s["truth"]).values() if c is not None]
        g.check(s["state"].evidence, calls, as_of=s["as_of"], request_id=s["id"])
    certs = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(certs) == len(sc)
    results = [CHECK.check_decision(c) for c in certs]
    assert all(r["verdict"] == "VERIFIED" for r in results), [r for r in results if r["verdict"] != "VERIFIED"][:2]
    assert {c["derivation"]["outcome"] for c in certs} >= {"allow", "deliver", "verify"}


def _redigest(cert):
    body = {k: v for k, v in cert.items() if k != "digest"}
    cert["digest"] = CHECK.sha(body)
    return cert


def test_the_checker_rejects_tampering(tmp_path):
    log = tmp_path / "d.jsonl"
    Guard("pharmacy_refills", log=str(log)).check(FULL, STALE, as_of="2026-10-20")
    cert = json.loads(log.read_text())
    assert CHECK.check_decision(cert)["verdict"] == "VERIFIED"
    t1 = copy.deepcopy(cert)
    t1["derivation"]["outcome"] = "allow"                          # content changed, digest not updated
    assert any("integrity" in p for p in CHECK.check_decision(t1)["problems"])
    t2 = _redigest(copy.deepcopy(t1))                              # digest updated: the derivation must still match
    r2 = CHECK.check_decision(t2)
    assert r2["verdict"] == "REJECTED" and any("outcome" in p for p in r2["problems"])
    t3 = copy.deepcopy(cert)
    t3["derivation"]["owed"] = []
    assert CHECK.check_decision(_redigest(t3))["verdict"] == "REJECTED"
    t4 = copy.deepcopy(cert)                                       # claim the insurer's update was never in force
    t4["derivation"]["applied_updates"] = ["state_emergency_30"]
    assert any("governing version" in p for p in CHECK.check_decision(_redigest(t4))["problems"])
    t5 = copy.deepcopy(cert)
    t5["derivation"]["ask"] = [{"fact": "schedule", "from": ["pharmacy_record"]}]
    assert CHECK.check_decision(_redigest(t5))["verdict"] == "REJECTED"


def test_readable_conditions_compile_to_the_same_program():
    spec = json.loads((ROOT / "contradish/contracts/programs/pharmacy_refills.json").read_text())
    spec.pop("updates")
    readable = copy.deepcopy(spec)
    c = readable["clauses"]
    c["R1"]["define"] = {"refill_day": "supply_days * (early_refill_pct / 100)",
                         "refill_due": "days_since_fill >= refill_day"}
    c["R2"]["define"] = {"has_refill": "refills_remaining >= 1"}
    c["R3"]["define"] = {"controlled_ii": "schedule == 'c2'"}
    c["R5"]["define"] = {"emergency_eligible": "identity_verified and refill_due and not has_refill "
                                               "and not prescriber_reachable and not controlled_ii"}
    c["R6"]["define"] = {"eligible": "identity_verified and refill_due and has_refill and not controlled_ii"}
    c["R7"]["steps"]["prescriber"]["args"]["reason"] = "'schedule_ii' if controlled_ii else 'no_refills'"
    assert PolicyProgram(readable).digest() == PolicyProgram(spec).digest()


def test_parse_errors_say_where():
    import pytest
    from contradish.policy_program import ProgramError
    spec = json.loads((ROOT / "contradish/contracts/programs/pharmacy_refills.json").read_text())
    spec.pop("updates")
    spec["clauses"]["R2"]["define"]["has_refill"] = "refills_remaining >= "
    with pytest.raises(ProgramError, match="clause R2 definition has_refill"):
        PolicyProgram(spec)
    assert parse("a if b else c") == ["if", "b", "a", "c"]


def test_examples_check_the_encoding_in_the_teams_terms():
    examples = [
        {"name": "day 25, one refill left", "facts": {"days_since_fill": 25, "refills_remaining": 1, "schedule": "none",
                                                      "prescriber_reachable": True, "identity_verified": True},
         "includes": ["dispense_refill"], "excludes": ["contact_prescriber"]},
        {"name": "wrong on purpose", "facts": {"days_since_fill": 10, "refills_remaining": 1, "schedule": "none",
                                               "prescriber_reachable": True, "identity_verified": True},
         "includes": ["dispense_refill"]},
    ]
    problems = run_examples("pharmacy_refills", examples)
    assert [n for n, _ in problems] == ["wrong on purpose"]


def test_policy_check_reports_what_would_make_the_guard_wrong():
    spec = json.loads((ROOT / "contradish/contracts/programs/pharmacy_refills.json").read_text())
    levels = {lvl for lvl, _ in check_policy(spec)}
    assert "error" not in levels
    del spec["fact_authority"]["schedule"]
    spec["irreversible_tools"].append("nonexistent_tool")
    msgs = [m for lvl, m in check_policy(spec) if lvl == "error"]
    assert any("schedule" in m for m in msgs) and any("nonexistent_tool" in m for m in msgs)


def test_wrap_returns_calls_and_decision():
    g = Guard("pharmacy_refills", mode="enforce")
    run = g.wrap(lambda req: STALE, lambda req: FULL, as_of=lambda req: "2026-10-20")
    calls, d = run("req")
    assert d.outcome == "deliver" and calls == [c for c in calls_from(STALE) if c not in d.blocked]


def _cli(*args, cwd=None):
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, "-m", "contradish.cli", "guard", *args], capture_output=True, text=True,
                          cwd=cwd, env=env)


def test_day_one_flow_from_the_command_line(tmp_path):
    import subprocess
    import sys
    r = _cli("init", str(tmp_path / "team"))
    assert r.returncode == 0, r.stderr
    team = tmp_path / "team"
    assert _cli("lint", str(team / "policy.json")).returncode == 0
    t = _cli("test", str(team / "policy.json"))
    assert t.returncode == 0 and "5 of 5 examples pass" in t.stdout
    ex = subprocess.run([sys.executable, str(team / "guard_example.py")], capture_output=True, text=True,
                        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert ex.returncode == 0, ex.stderr
    assert "DELIVER" in ex.stdout and "in enforce mode it would execute: [('dispense_refill'" in ex.stdout
    log = team / "decisions.jsonl"
    v = _cli("verify", str(log))
    assert v.returncode == 0 and v.stdout.count("VERIFIED") == 2
    rep = _cli("report", str(log))
    assert rep.returncode == 0 and "2 verified, 0 rejected" in rep.stdout
    assert "Amendments missed: insurer_early_refill_70 (1)" in rep.stdout


def test_check_exit_codes(tmp_path):
    ev = tmp_path / "e.json"
    calls = tmp_path / "c.json"
    ev.write_text(json.dumps({k: {"value": v, "source": s} for k, (v, s) in FULL.items()}))
    calls.write_text(json.dumps(STALE))
    r = _cli("check", "pharmacy_refills", "--evidence", str(ev), "--calls", str(calls), "--as-of", "2026-10-20")
    assert r.returncode == 1 and r.stdout.startswith("DELIVER")
    calls.write_text(json.dumps(RIGHT))
    assert _cli("check", "pharmacy_refills", "--evidence", str(ev), "--calls", str(calls),
                "--as-of", "2026-10-20").returncode == 0


def test_a_tampered_log_fails_verification(tmp_path):
    log = tmp_path / "d.jsonl"
    Guard("pharmacy_refills", log=str(log)).check(FULL, STALE, as_of="2026-10-20")
    cert = json.loads(log.read_text())
    cert["derivation"]["outcome"] = "allow"
    log.write_text(json.dumps(_redigest(cert)) + "\n")
    v = _cli("verify", str(log))
    assert v.returncode == 1 and "REJECTED" in v.stdout
    rep = _cli("report", str(log))
    assert "0 verified, 1 rejected" in rep.stdout
