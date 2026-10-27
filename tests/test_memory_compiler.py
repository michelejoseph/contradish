"""The policy-aware memory compiler: minimal sufficient memory, independent certificate check, revalidation."""
import copy
import importlib.util
import json
import os

import pytest

from contradish.integrity import IntegrityLedger, verify_ledger
from contradish.memory_compiler import authority_chains, compile_memory, render_memory, revalidate
from contradish.versions import fixture_key

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(HERE, "contradish", "templates")
POLICY = os.path.join(T, "memory_policy.json")
with open(os.path.join(T, "memory_history.json")) as _f:
    HIST = json.load(_f)
_s = importlib.util.spec_from_file_location("mc_standalone", os.path.join(HERE, "contradish", "memory_check.py"))
MC = importlib.util.module_from_spec(_s)
_s.loader.exec_module(MC)
REVERSAL = {"id": "insurer_back_to_90", "source": "insurer", "channel": "system", "effective": "2026-10-03",
            "content": "From 2026-10-03 refills need 90% elapsed.",
            "edits": [{"clause": "R1", "params": {"early_refill_pct": 90}}]}


@pytest.fixture(scope="module")
def cert():
    return compile_memory(POLICY, HIST, "agent", "2026-09-30")


def _redigest(x):
    x = copy.deepcopy(x)
    x.pop("digest")
    x["digest"] = MC.sha(x)
    return x


def test_minimal_memory(cert):
    m = cert["memory"]
    assert m["facts"]["days_since_fill"] == {"lo": 20, "hi": 23, "on": "2026-09-30"}
    assert m["facts"]["refills_remaining"] == {"lo": 1, "hi": 11}
    assert sorted(m["facts"]["schedule"]["values"]) == ["c3_c5", "none"]
    chains = [([l["id"] for l in c["links"]], c["until"]) for c in m["authority"]["agent_may_dispense"]["chains"]]
    assert chains == [(["G1", "G2"], "2026-10-20"), (["G3"], "2026-10-10")]
    reasons = {d["event"]: d["reason"] for d in cert["discarded"]}
    assert "claim" in reasons[12] and "no standing" in reasons[11] and "superseded" in reasons[0]
    assert "different scope" in reasons[3] and "not on any chain" in reasons[2]
    assert "holder" not in json.dumps(m["authority"])           # holders are not needed, issuers are
    assert compile_memory(POLICY, HIST, "agent", "2026-09-30")["digest"] == cert["digest"]


def test_independent_checker_accepts(cert):
    r = MC.check(cert, HIST)
    assert r["verdict"] == "VERIFIED", r["problems"]


@pytest.mark.parametrize("name", ["coarser", "witness", "chain", "claim", "version", "digest"])
def test_independent_checker_rejects_forgeries(cert, name):
    x = copy.deepcopy(cert)
    if name == "coarser":
        p = x["partitions"]["days_since_fill"]
        i = p.index({"lo": 20, "hi": 23})
        p[i - 1:i + 1] = [{"lo": p[i - 1]["lo"], "hi": 23}]
        x["memory"]["facts"]["days_since_fill"] = {"lo": p[i - 1]["lo"], "hi": 23, "on": "2026-09-30"}
    elif name == "witness":
        x["witnesses"].pop()
    elif name == "chain":
        x["memory"]["authority"]["agent_may_dispense"]["chains"][0]["until"] = None
    elif name == "claim":
        x["memory"]["facts"]["days_since_fill"] = {"lo": 24, "hi": 90, "on": "2026-09-30"}
    elif name == "version":
        x["timeline"][1]["spec"]["clauses"]["R1"]["params"]["early_refill_pct"] = 50
    if name != "digest":
        x = _redigest(x)
    else:
        x["memory"]["unknown"] = ["x"]
    assert MC.check(x, HIST)["verdict"] == "REJECTED"


def test_authority_semantics():
    h = [{"kind": "grant", "id": "A", "by": "root", "to": "lead", "scope": "s", "from": "2026-01-01"},
         {"kind": "grant", "id": "B", "by": "lead", "to": "agent", "scope": "s", "under": "A", "from": "2026-01-02"},
         {"kind": "revoke", "target": "A", "by": "root", "effective": "2026-01-10"},
         {"kind": "grant", "id": "A2", "by": "root", "to": "lead", "scope": "s", "from": "2026-01-12"}]
    ch, _ = authority_chains(h, "root", "s", "agent", "2026-01-05", 30)
    assert [(c["from"], c["until"]) for c in ch] == [("2026-01-05", "2026-01-10")]     # cascade, no restoration
    h2 = h + [{"kind": "grant", "id": "X", "by": "lead", "to": "agent", "scope": "s", "from": "2026-01-02"}]
    ch2, d2 = authority_chains(h2, "root", "s", "agent", "2026-01-05", 30)
    assert len(ch2) == 1 and any("no standing" in d["reason"] for d in d2)              # no `under`: no standing


def test_revalidate_events_reuse_every_partition(cert):
    n = revalidate(cert, POLICY, new_events=[{"kind": "revoke", "id": "R3", "target": "G3", "by": "pharmacy_owner",
                                              "effective": "2026-10-05"}])
    assert n["status"] == "valid" and len(n["reused_partitions"]) == 5 and n["supersedes"] == cert["digest"]
    assert [c["until"] for c in n["memory"]["authority"]["agent_may_dispense"]["chains"]] == ["2026-10-20", "2026-10-05"]
    assert MC.check(n, None, cert)["verdict"] == "VERIFIED"


def test_revalidate_invalidates_when_memory_cannot_decide(cert):
    g7 = {"kind": "grant", "id": "G7", "by": "marcus", "to": "agent", "scope": "dispense", "under": "G4",
          "from": "2026-10-02"}
    n = revalidate(cert, POLICY, new_events=[g7])
    assert n["status"] == "invalidated" and "G4" in n["issues"][0]["problem"]
    h = revalidate(cert, POLICY, new_events=[g7], history=HIST)
    assert h["status"] == "valid"
    assert any([l["id"] for l in c["links"]] == ["G4", "G7"] for c in h["memory"]["authority"]["agent_may_dispense"]["chains"])


def test_announced_change_makes_a_dropped_distinction_load_bearing(cert):
    n = revalidate(cert, POLICY, new_updates=[REVERSAL])
    assert n["status"] == "invalidated"
    i = n["issues"][0]
    assert i["fact"] == "days_since_fill" and i["load_bearing_under"] == "insurer_back_to_90"
    assert n["memory"]["facts"]["days_since_fill"]["insufficient"]
    assert "INSUFFICIENT" in render_memory(n)
    h = revalidate(cert, POLICY, new_updates=[REVERSAL], history=HIST)
    assert h["status"] == "valid" and h["memory"]["facts"]["days_since_fill"] == {"lo": 22, "hi": 22, "on": "2026-09-30"}
    assert MC.check(h, None, cert)["verdict"] == "VERIFIED"


def test_memory_in_the_signed_ledger(tmp_path, cert):
    key = fixture_key("auditor")
    L = str(tmp_path / "audit.jsonl")
    led = IntegrityLedger(L, key=key)
    led.record_memory(cert)
    led.record_memory(revalidate(cert, POLICY, new_events=[{"kind": "revoke", "id": "R3", "target": "G3",
                                                            "by": "pharmacy_owner", "effective": "2026-10-05"}]))
    led.checkpoint()
    r = verify_ledger(L, trust=[key["public"]])
    assert r["verdict"] == "VERIFIED" and r["kinds"]["memory"] == 2
