"""The decision-integrity layer: compiler, differential verification, gateway, ledger."""
import json
import os
import warnings

import pytest

from contradish.gateway import Gateway, apply_decision
from contradish.guard import Guard
from contradish.integrity import (IntegrityLedger, compile_change, model_agent, resolve_agent, verify_change,
                                  verify_ledger)
from contradish.versions import fixture_key

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = os.path.join(HERE, "contradish", "templates", "starter_policy.json")
REC = {"days_since_fill": {"value": 27, "source": "pharmacy_record"},
       "refills_remaining": {"value": 2, "source": "pharmacy_record"},
       "schedule": {"value": "c3_c5", "source": "pharmacy_record"},
       "identity_verified": {"value": True, "source": "identity_check"},
       "prescriber_reachable": {"value": True, "source": "prescriber_office"}}
EARLY = dict(REC, days_since_fill={"value": 10, "source": "pharmacy_record"})
CLAIM = dict(REC, days_since_fill={"value": 27, "source": "patient"})
DISPENSE = [("dispense_refill", {"days": 30}), ("notify_patient", {"outcome": "refill_ready"})]


@pytest.fixture(scope="module")
def plan():
    return compile_change(POLICY, as_of="2026-10-10", since="2026-09-30", max_situations=40)


def _guard(mode="shadow", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = Guard(POLICY, mode="shadow", **kw)
    g.mode = mode
    return g


# ── 1. compiler ──────────────────────────────────────────────────────────────

def test_plan_classifies_each_update(plan):
    st = {it.update["id"]: it.status for it in plan.items}
    assert st == {"insurer_early_refill_70": "in_force", "patient_claims_half": "in_force",
                  "state_emergency_30": "pending"}
    s = {it.update["id"]: it.summary for it in plan.items}
    assert s["insurer_early_refill_70"]["must_change"] > 0 and s["insurer_early_refill_70"]["must_resist"] == 0
    assert s["patient_claims_half"]["must_change"] == 0 and s["patient_claims_half"]["must_resist"] > 0
    assert s["state_emergency_30"]["must_change"] == 0 and s["state_emergency_30"]["must_resist"] > 0


def test_plan_is_deterministic_and_covers_every_update(plan):
    again = compile_change(POLICY, as_of="2026-10-10", since="2026-09-30", max_situations=40)
    assert again.digest == plan.digest
    assert all(v > 0 for v in plan.sampling["affected_selected_per_update"].values())
    assert len(plan.situations) == 40


def test_after_effective_date_the_pending_update_must_change():
    later = compile_change(POLICY, as_of="2026-10-20", since="2026-09-30", max_situations=40)
    it = next(i for i in later.items if i.update["id"] == "state_emergency_30")
    assert it.status == "in_force" and it.summary["must_change"] > 0


def test_since_makes_earlier_updates_the_baseline():
    p = compile_change(POLICY, as_of="2026-10-20", since="2026-10-05", max_situations=30)
    assert [i.update["id"] for i in p.items] == ["state_emergency_30"] or \
        "insurer_early_refill_70" not in [i.update["id"] for i in p.items]


# ── 2. differential verification ─────────────────────────────────────────────

def test_each_witness_is_held_on_the_property_it_violates(plan):
    r = verify_change(plan, {m: f"witness:{m}" for m in ("faithful", "rigid", "credulous")}).models
    assert r["faithful"]["verdict"] == "ship" and r["faithful"]["failed"] == []
    assert r["rigid"]["failed"] == ["complete"]
    assert {"authorized", "timely"} <= set(r["credulous"]["failed"]) and "complete" not in r["credulous"]["failed"]
    assert r["rigid"]["failures"] and "missed change" in r["rigid"]["failures"][0]


def test_release_digest_binds_plan(plan):
    rel = verify_change(plan, {"f": "witness:faithful"})
    assert rel.plan_digest == plan.digest and rel.digest.startswith("sha256:")


def test_model_agent_runs_any_chat_function(plan):
    """A fake model that answers with the policy's own baseline calls: it never applies updates (rigid)."""
    seen = []

    def chat(messages):
        seen.append(messages)
        return 'Sure. [{"tool": "verify_identity", "args": {}}]'
    agent = resolve_agent(chat, title="pharmacy", today="2026-10-10")
    rel = verify_change(plan, {"fake": chat})
    assert rel.models["fake"]["scores"]["complete"] is not None
    sysmsg = seen[0][0]["content"]
    assert "Today is 2026-10-10" in sysmsg and "JSON array" in sysmsg and "returns" not in sysmsg
    assert any("Update to your instructions" in m["content"] or "Customer message" in m["content"]
               for msgs in seen for m in msgs[1:])


def test_resolve_agent_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_agent("nonsense")


# ── 3. gateway ───────────────────────────────────────────────────────────────

def _oai(calls):
    return {"id": "x", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": t, "arguments": json.dumps(a)}}
                       for i, (t, a) in enumerate(calls)]}}]}


def _ant(calls):
    return {"id": "m", "type": "message", "role": "assistant", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "ok"}] + [{"type": "tool_use", "id": f"t{i}", "name": t, "input": a}
                                                           for i, (t, a) in enumerate(calls)]}


def _run(mode, fmt, evidence, header=False, extra=None):
    seen = {}
    mk = _oai if fmt == "openai" else _ant

    def transport(url, h, body):
        seen["url"], seen["h"], seen["body"] = url, h, json.loads(body)
        return 200, json.dumps(mk(DISPENSE)).encode()
    gw = Gateway(_guard(mode), "https://up.example", transport)
    req = {"model": "m", "messages": [], "tools": [{}], **(extra or {})}
    hdr = {"Authorization": "Bearer sk-x", "anthropic-version": "2023-06-01", "Cookie": "secret"}
    ctx = {"evidence": evidence, "as_of": "2026-10-10"} if evidence is not None else None
    if ctx and header:
        hdr["X-Contradish-Context"] = json.dumps(ctx)
    elif ctx:
        req["contradish"] = ctx
    path = "/v1/chat/completions" if fmt == "openai" else "/v1/messages"
    st, h, b = gw.handle(path, hdr, json.dumps(req).encode())
    out = json.loads(b) if st == 200 else b
    return st, h, out, seen


def _left(fmt, out):
    if fmt == "openai":
        return [t["function"]["name"] for t in (out["choices"][0]["message"].get("tool_calls") or [])]
    return [c["name"] for c in out["content"] if c["type"] == "tool_use"]


@pytest.mark.parametrize("fmt", ["openai", "anthropic"])
def test_gateway_enforce(fmt):
    for ev, outcome, left in ((REC, "allow", ["dispense_refill", "notify_patient"]), (EARLY, "block", []),
                              (CLAIM, "block", []), (None, "block", [])):
        st, h, out, seen = _run("enforce", fmt, ev)
        assert st == 200 and h["X-Contradish-Outcome"] == outcome and _left(fmt, out) == left
        assert out["contradish"]["digest"] == h["X-Contradish-Digest"]
        assert "contradish" not in seen["body"]
        assert seen["h"]["authorization"] == "Bearer sk-x" and "cookie" not in seen["h"]
        if not left:
            text = out["choices"][0]["message"]["content"] if fmt == "openai" else out["content"][-1]["text"]
            assert text.startswith("[contradish] Not executed")


@pytest.mark.parametrize("fmt", ["openai", "anthropic"])
def test_gateway_shadow_changes_nothing(fmt):
    st, h, out, _ = _run("shadow", fmt, EARLY, header=True)
    assert h["X-Contradish-Outcome"] == "block" and _left(fmt, out) == ["dispense_refill", "notify_patient"]
    assert out["contradish"]["enforced"] is False


def test_gateway_refuses_streamed_tools_when_enforcing():
    st, h, out, seen = _run("enforce", "openai", REC, extra={"stream": True})
    assert st == 400 and not seen


def test_gateway_passes_through_responses_without_tool_calls():
    gw = Gateway(_guard("enforce"), "u", lambda u, h, b: (200, json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}).encode()))
    st, h, b = gw.handle("/v1/chat/completions", {}, b'{"messages": []}')
    assert st == 200 and h["X-Contradish-Outcome"] == "no-tool-calls" and json.loads(b)["choices"][0]["message"]["content"] == "hi"


def test_gateway_unknown_path_and_upstream_error():
    gw = Gateway(_guard(), "u", lambda u, h, b: (429, b'{"error": "rate"}'))
    assert gw.handle("/v1/embeddings", {}, b"{}")[0] == 404
    assert gw.handle("/v1/chat/completions", {}, b"{}")[:3:2] == (429, b'{"error": "rate"}')


# ── 4. ledger ────────────────────────────────────────────────────────────────

def test_ledger_end_to_end(tmp_path, plan):
    key = fixture_key("auditor")
    L = str(tmp_path / "audit.jsonl")
    led = IntegrityLedger(L, key=key, checkpoint_every=3)
    led.record_plan(plan)
    led.record_release(plan, verify_change(plan, {"f": "witness:faithful"}))
    g = _guard("enforce", ledger=led)
    for ev in (REC, EARLY, CLAIM):
        g.check(ev, [{"tool": t, "args": a} for t, a in DISPENSE], as_of="2026-10-10")
    led.checkpoint()
    r = verify_ledger(L, trust=[key["public"]])
    assert r["verdict"] == "VERIFIED", r["problems"]
    assert r["kinds"]["decision"] == 3 and r["decisions_rederived"]["VERIFIED"] == 3 and r["unsigned_tail"] == 0
    # reopening continues the chain
    IntegrityLedger(L).append("note", {"text": "reopened"})
    assert verify_ledger(L)["verdict"] == "VERIFIED"


def test_ledger_detects_edit_removal_and_untrusted_key(tmp_path, plan):
    key = fixture_key("auditor")
    L = str(tmp_path / "a.jsonl")
    led = IntegrityLedger(L, key=key)
    for i in range(4):
        led.append("note", {"i": i})
    led.checkpoint()
    lines = open(L).read().splitlines()
    e = json.loads(lines[1]); e["record"]["i"] = 99
    edited = lines[:1] + [json.dumps(e)] + lines[2:]
    (tmp_path / "e.jsonl").write_text("\n".join(edited) + "\n")
    assert verify_ledger(str(tmp_path / "e.jsonl"))["verdict"] == "REJECTED"
    (tmp_path / "r.jsonl").write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    assert verify_ledger(str(tmp_path / "r.jsonl"))["verdict"] == "REJECTED"
    other = fixture_key("someone_else")
    assert verify_ledger(L, trust=[other["public"]])["verdict"] == "REJECTED"
    assert verify_ledger(L, trust=[key["public"]])["verdict"] == "VERIFIED"


def test_checkpoint_needs_key(tmp_path):
    with pytest.raises(ValueError):
        IntegrityLedger(str(tmp_path / "x.jsonl")).checkpoint()
