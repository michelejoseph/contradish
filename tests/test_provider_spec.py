"""Provider-spec authority: a model judged by its own provider's published precedence rules."""
from contradish import study as S
from contradish.provider_spec import (
    ANTHROPIC, CLAUSE_CLAIMS, DELIVERIES, build_provider_contrasts, describe, run_provider,
)
from contradish.transition import _second_turn


def test_only_the_system_turn_has_standing():
    ts = build_provider_contrasts()
    assert len(ts) == len(CLAUSE_CLAIMS) * len(DELIVERIES)
    for t in ts:
        must_change = bool(t.frontier().change)
        assert must_change == (t.delivery_role == "system_turn"), t.id
        if t.delivery_role != "system_turn":
            assert t.frontier().resist, t.id


def test_system_turn_is_a_real_system_message_after_the_user_turn():
    t = next(x for x in build_provider_contrasts() if x.delivery_role == "system_turn")
    case = next(c for c in t.cases if c.asserted)
    first = [{"role": "system", "content": t.system_text("before")}, {"role": "user", "content": case.question}]
    fresh = _second_turn(t, case, first, "a0", "fresh", "")
    assert [m["role"] for m in fresh] == ["system", "user", "system"]
    assert fresh[-1]["content"] == t.update.content
    conv = _second_turn(t, case, first, "a0", "in_conversation", "")
    assert [m["role"] for m in conv] == ["system", "user", "assistant", "user", "system"]


def test_anthropic_payload_native_mid_conversation_system():
    msgs = [{"role": "system", "content": "policy"}, {"role": "user", "content": "q"},
            {"role": "system", "content": "45 days now"}]
    body = S._anthropic_payload(msgs, "m", 100, None, native_system=True)
    assert body["system"] == "policy"
    assert body["messages"] == [{"role": "user", "content": "q"}, {"role": "system", "content": "45 days now"}]
    # A system message after an assistant turn cannot be sent mid-conversation: it is joined.
    msgs2 = [{"role": "system", "content": "policy"}, {"role": "user", "content": "q"},
             {"role": "assistant", "content": "a"}, {"role": "system", "content": "note"},
             {"role": "user", "content": "q2"}]
    body2 = S._anthropic_payload(msgs2, "m", 100, None, native_system=True)
    assert body2["system"] == "policy\n\nnote"
    assert [m["role"] for m in body2["messages"]] == ["user", "assistant", "user"]


def test_anthropic_falls_back_to_joined_system_when_rejected(monkeypatch):
    sent = []

    def fake_post(url, headers, body, retries=0):
        sent.append(body)
        if any(m["role"] == "system" for m in body["messages"]):
            raise S.ProviderError("bad request", status=400, body="system role not supported for this model")
        return {"content": [{"type": "text", "text": "ok"}]}
    monkeypatch.setattr(S, "_post", fake_post)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    spec = S.parse_model("anthropic:claude-test-model")
    S._FALLBACKS.pop(spec.id, None)
    msgs = [{"role": "system", "content": "p"}, {"role": "user", "content": "q"}, {"role": "system", "content": "u"}]
    assert S.call_model(spec, msgs) == "ok"
    assert len(sent) == 2 and sent[1]["system"] == "p\n\nu"
    assert "rejected" in spec.message_transform
    S.call_model(spec, msgs)
    assert len(sent) == 3                       # learned: no second rejection
    S._FALLBACKS.pop(spec.id, None)


def test_scripted_agents():
    def pci(kind):
        r = run_provider(S.scripted_chat_for(kind), S.label_echo_classifier, ["fresh"], 1, 1)["fresh"]
        return r["provider_conformance_index"]["value"]
    assert pci("faithful") == 1.0
    assert pci("credulous") == 0.0
    assert pci("rigid") == 0.0
    assert pci("channel_trusting") == 0.5      # takes the tool content's word, not the user's


def test_spec_is_cited_and_dated():
    d = ANTHROPIC.to_dict()
    assert d["url"].startswith("https://") and d["retrieved"] == "2026-10-14"
    assert d["excluded"]
    assert "Not scored" in describe()
