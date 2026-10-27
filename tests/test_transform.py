"""Authorization-preserving memory transformations: verdicts, repair, compression, the guard, independent re-derivation."""
import copy
import importlib.util
import json
import os
import threading
import urllib.request

import pytest

from contradish.integrity import IntegrityLedger, verify_ledger
from contradish.memory_compiler import compile_memory
from contradish.transform import MemoryGuard, check_transform, compress, repair, serve
from contradish.versions import fixture_key

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(HERE, "contradish", "templates")
POLICY = os.path.join(T, "memory_policy.json")
DAY = "2026-09-30"
_s = importlib.util.spec_from_file_location("mc_t", os.path.join(HERE, "contradish", "memory_check.py"))
MC = importlib.util.module_from_spec(_s)
_s.loader.exec_module(MC)


@pytest.fixture(scope="module")
def M():
    with open(os.path.join(T, "memory_history.json")) as f:
        return compile_memory(POLICY, json.load(f), "agent", DAY)["memory"]


def _change(M, **kw):
    a = copy.deepcopy(M)
    if "chains" in kw:
        a["authority"]["agent_may_dispense"]["chains"] = kw.pop("chains")
    for k, v in kw.items():
        a["facts"][k] = v
    return a


def _redigest(v):
    v = copy.deepcopy(v)
    v.pop("digest")
    v["digest"] = MC.sha(v)
    return v


def test_identity_and_same_class_rewrites_are_safe(M):
    assert check_transform(POLICY, M, copy.deepcopy(M), DAY)["verdict"] == "SAFE"
    assert check_transform(POLICY, M, _change(M, prescriber_reachable={"drop": True}), DAY)["verdict"] == "SAFE"


def test_authority_summary_expands_authority(M):
    v = check_transform(POLICY, M, _change(M, chains=[{"links": [], "from": DAY, "until": "2026-10-20"}]), DAY)
    assert v["verdict"] == "UNSAFE" and list(v["violations_by_kind"]) == ["authority_expansion"]
    c = v["counterexamples"][0]
    assert "revokes" in c["future"] and c["after_does"]["dispense"]["tool"] == "dispense_refill"
    assert c["before_does"]["dispense"] is None


def test_each_kind(M):
    assert list(check_transform(POLICY, M, _change(M, days_since_fill={"lo": 18, "hi": 23, "on": DAY}), DAY)
                ["violations_by_kind"]) == ["indeterminacy"]
    g = M["authority"]["agent_may_dispense"]["chains"][:1]
    assert list(check_transform(POLICY, M, _change(M, chains=g), DAY)["violations_by_kind"]) == ["withholding"]
    vague = _change(M, identity_verified={"drop": True})
    assert list(check_transform(POLICY, vague, M, DAY)["violations_by_kind"]) == ["fabrication"]
    # misremembering "no refills left": the patient is sent a different notice, an irreversible call the memory
    # before forbids, so it ranks as authority expansion (above the withheld dispense)
    assert list(check_transform(POLICY, M, _change(M, refills_remaining={"lo": 0, "hi": 0}), DAY)
                ["violations_by_kind"]) == ["authority_expansion"]


def test_argument_change_is_a_changed_action_not_withholding():
    from contradish.transform import _classify
    b = {json.dumps({"tell": {"tool": "tell_refill_date", "args": {"day": 24}}})}
    a = {json.dumps({"tell": {"tool": "tell_refill_date", "args": {"day": 21}}})}
    assert _classify(b, a, set()) == "changed_action"
    assert _classify(b, a, {"tell_refill_date"}) == "authority_expansion"
    assert _classify(b, {json.dumps({"tell": None})}, set()) == "withholding"


def test_simplest_counterexample_first(M):
    v = check_transform(POLICY, M, _change(M, chains=[{"links": [], "from": DAY, "until": "2026-10-20"}]), DAY)
    assert " and " not in v["counterexamples"][0]["future"]


def test_independent_rederivation(M):
    for after in (copy.deepcopy(M), _change(M, chains=[{"links": [], "from": DAY, "until": "2026-10-20"}]),
                  _change(M, days_since_fill={"lo": 18, "hi": 23, "on": DAY})):
        v = check_transform(POLICY, M, after, DAY)
        assert MC.check_verdict(v)["verdict"] == "VERIFIED"
    v = check_transform(POLICY, M, _change(M, chains=[{"links": [], "from": DAY, "until": "2026-10-20"}]), DAY)
    lie = copy.deepcopy(v)
    lie.update(verdict="SAFE", violations_by_kind={}, counterexamples=[])
    assert MC.check_verdict(_redigest(lie))["verdict"] == "REJECTED"
    hide = copy.deepcopy(v)
    hide["futures_checked"] = ["no revocation"]
    assert MC.check_verdict(_redigest(hide))["verdict"] == "REJECTED"
    swap = copy.deepcopy(v)
    swap["after"] = copy.deepcopy(M)
    assert MC.check_verdict(_redigest(swap))["verdict"] == "REJECTED"     # after_digest no longer matches


def test_repair_keeps_the_safe_part(M):
    a = _change(M, chains=[{"links": [], "from": DAY, "until": "2026-10-20"}],
                days_since_fill={"lo": 18, "hi": 23, "on": DAY}, prescriber_reachable={"drop": True})
    r = repair(POLICY, M, a, DAY)
    assert r["applied"] == ["facts.prescriber_reachable"] and r["verdict"]["verdict"] == "SAFE"
    assert {x["component"] for x in r["refused"]} == {"authority.agent_may_dispense", "facts.days_since_fill"}


def test_compress_goes_past_the_per_fact_memory(M):
    c = compress(POLICY, M, DAY)
    assert "prescriber_reachable" in c["dropped"] and c["verdict"]["verdict"] == "SAFE"


@pytest.mark.parametrize("mode,on_unsafe,expect", [("enforce", "repair", "repaired"), ("enforce", "reject", "rejected"),
                                                   ("shadow", "repair", "committed as proposed (shadow")])
def test_guard_modes(M, mode, on_unsafe, expect, tmp_path):
    led = IntegrityLedger(str(tmp_path / "a.jsonl"), key=fixture_key("auditor"))
    g = MemoryGuard(POLICY, M, as_of=lambda: DAY, mode=mode, on_unsafe=on_unsafe, ledger=led)
    unsafe = _change(M, chains=[{"links": [], "from": DAY, "until": "2026-10-20"}])
    r = g.propose(unsafe, reason="consolidate")
    assert r["action"].startswith(expect)
    if mode == "enforce":
        assert g.memory["authority"] == M["authority"]
    else:
        assert g.memory == unsafe
    g.propose(_change(g.memory, prescriber_reachable={"drop": True}))
    led.checkpoint()
    out = verify_ledger(str(tmp_path / "a.jsonl"), trust=[fixture_key("auditor")["public"]])
    assert out["verdict"] == "VERIFIED" and out["kinds"]["transform"] == 2


def test_ledger_flags_an_enforcing_guard_that_committed_unsafe(M, tmp_path):
    L = str(tmp_path / "b.jsonl")
    led = IntegrityLedger(L)
    v = check_transform(POLICY, M, _change(M, chains=[{"links": [], "from": DAY, "until": "2026-10-20"}]), DAY)
    led.append("transform", {"schema": "contradish.transform_receipt/1", "mode": "enforce",
                             "action": "committed as proposed", "verdict": v})
    assert verify_ledger(L)["verdict"] == "REJECTED"


def test_http_service(M):
    srv = serve(POLICY, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/memory/transform"
    body = {"before": M, "after": _change(M, chains=[{"links": [], "from": DAY, "until": "2026-10-20"}]),
            "as_of": DAY, "repair": True}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    out = json.loads(urllib.request.urlopen(req).read())
    srv.shutdown()
    assert out["verdict"]["verdict"] == "UNSAFE" and out["repair"]["verdict"]["verdict"] == "SAFE"
