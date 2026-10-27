"""Remembering what you owe is not the same as being able to pay it."""
import copy
import importlib.util
import json
import os

import pytest

from contradish.fulfillment import assess, open_obligations
from contradish.memory_compiler import compile_memory, load_memory_policy, revalidate
from contradish.transform import check_transform, repair

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(HERE, "contradish", "templates")
POLICY = os.path.join(T, "memory_policy.json")
DAY = "2026-09-30"
with open(os.path.join(T, "memory_history.json")) as _f:
    HIST = json.load(_f)
_s = importlib.util.spec_from_file_location("mc_f", os.path.join(HERE, "contradish", "memory_check.py"))
MC = importlib.util.module_from_spec(_s)
_s.loader.exec_module(MC)


@pytest.fixture(scope="module")
def cert():
    return compile_memory(POLICY, HIST, "agent", DAY)


def _ob(m, i):
    return next(o for o in m["obligations"] if o["id"] == i)


def test_open_obligations_keep_every_parameter_and_drop_claims(cert):
    m = cert["memory"]
    assert [o["id"] for o in m["obligations"]] == ["O1", "O2"]
    assert _ob(m, "O1")["params"] == {"order_id": "A-1043", "card_token": "tok_4242", "amount": 40}
    reasons = " ".join(d["reason"] for d in cert["discarded"])
    assert "patient does not establish refund obligations" in reasons and "O0 discharged" in reasons
    assert MC.check(cert, HIST)["verdict"] == "VERIFIED"


def test_assess_act_by_and_fragile(cert):
    r = {o["id"]: o for o in assess(POLICY, cert["memory"], DAY)["obligations"]}
    assert r["O1"]["status"] == "act_by" and r["O1"]["act_by"] == "2026-10-19"
    assert r["O1"]["stranding_futures"]                  # depends only on revocable grants
    assert r["O2"]["status"] == "fulfillable" and not r["O2"]["stranding_futures"]


def test_params_missing_and_stranded_by_schedule(cert):
    m = copy.deepcopy(cert["memory"])
    _ob(m, "O1")["params"].pop("card_token")
    assert {o["id"]: o["status"] for o in assess(POLICY, m, DAY)["obligations"]}["O1"] == "params_missing"
    m = copy.deepcopy(cert["memory"])
    m["authority"]["agent_may_dispense"]["chains"] = []
    assert {o["id"]: o["status"] for o in assess(POLICY, m, DAY)["obligations"]}["O1"] == "stranded_by_schedule"


def test_a_fallback_that_needs_no_grant_removes_the_fragility(cert):
    spec, ups, cfg = load_memory_policy(POLICY)
    pol = dict(spec, updates=ups, memory=copy.deepcopy(cfg))
    pol["memory"]["obligations"]["refund"]["fallback"] = {"tool": "hand_to_pharmacist", "authority": None}
    r = {o["id"]: o for o in assess(pol, cert["memory"], DAY)["obligations"]}
    assert r["O1"]["status"] == "fulfillable" and not r["O1"]["stranding_futures"]


@pytest.mark.parametrize("change,kind", [("card", "fulfillment_loss"), ("forget", "obligation_forgotten"),
                                         ("route", "fulfillment_loss")])
def test_transform_kinds_and_independent_rederivation(cert, change, kind):
    M = cert["memory"]
    a = copy.deepcopy(M)
    if change == "card":
        _ob(a, "O1")["params"].pop("card_token")
    elif change == "forget":
        a["obligations"] = [o for o in a["obligations"] if o["id"] != "O1"]
    else:
        a["authority"]["agent_may_dispense"]["chains"] = []
    v = check_transform(POLICY, M, a, DAY)
    assert v["verdict"] == "UNSAFE" and kind in v["violations_by_kind"]
    assert MC.check_verdict(v)["verdict"] == "VERIFIED"
    lie = copy.deepcopy(v)
    lie["violations_by_kind"].pop(kind)
    lie.pop("digest")
    lie["digest"] = MC.sha(lie)
    assert MC.check_verdict(lie)["verdict"] == "REJECTED"


def test_forged_certificate_without_the_card_is_rejected(cert):
    x = copy.deepcopy(cert)
    _ob(x["memory"], "O1")["params"].pop("card_token")
    x.pop("digest")
    x["digest"] = MC.sha(x)
    assert MC.check(x, HIST)["verdict"] == "REJECTED"


def test_repair_keeps_the_obligation_intact(cert):
    M = cert["memory"]
    a = copy.deepcopy(M)
    _ob(a, "O1")["params"].pop("card_token")
    a["facts"]["prescriber_reachable"] = {"drop": True}
    r = repair(POLICY, M, a, DAY)
    assert r["applied"] == ["facts.prescriber_reachable"] and "obligations.O1" in [x["component"] for x in r["refused"]]
    assert _ob(r["memory"], "O1")["params"]["card_token"] == "tok_4242"


def test_revalidate_applies_discharge_and_new_obligations(cert):
    n = revalidate(cert, POLICY, new_events=[
        {"kind": "discharged", "id": "O1", "at": DAY},
        {"kind": "obligation", "id": "O5", "type": "recall_notice", "owed_to": "patient", "incurred": DAY,
         "due": "2026-10-12", "params": {"contact": "+1-555-0100"}, "source": "pharmacy_record"}])
    assert [o["id"] for o in n["memory"]["obligations"]] == ["O2", "O5"]
    assert open_obligations([{"kind": "obligation", "id": "X", "type": "nope", "due": DAY}], {}, DAY)[0] == []
