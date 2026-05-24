"""
Tests for contradish.memory: commitment extraction, relevance retrieval,
contradiction detection, repair, the session-scoped stores, and the
memory-aware Firewall path.
Run with: pytest tests/test_memory.py
No API key required: every LLM call is served by an injected fake.
"""
import contradish.firewall as fw_mod
from contradish.memory import (
    Commitment,
    ContradictionFinding,
    CommitmentStore,
    InMemoryCommitmentStore,
    RedisCommitmentStore,
    ConversationMemory,
    _overlap_score,
)


# ── Fakes ────────────────────────────────────────────────────────────────

class FakeLLM:
    """
    Scriptable stand-in for LLMClient. Routes by prompt content:
      - extraction prompt  -> returns a JSON list of commitments
      - detection prompt   -> returns a contradiction verdict
      - repair prompt      -> returns rewritten text
    Override the canned values per-instance as needed.
    """
    fast_model = "fake-fast"

    def __init__(self, extract_map=None, detect=None, repair_text="REPAIRED"):
        # extract_map: substring in (query+response) -> JSON string to return
        self.extract_map = extract_map or {}
        self.detect = detect if detect is not None else {"contradiction": False}
        self.repair_text = repair_text
        self.calls = {"extract": 0, "detect": 0, "repair": 0}

    def complete(self, prompt, model=None, **kw):
        if "extract the durable" in prompt:
            self.calls["extract"] += 1
            for needle, payload in self.extract_map.items():
                if needle in prompt:
                    return payload
            return "[]"
        if "Rewrite the reply" in prompt or "rewritten reply text" in prompt:
            self.calls["repair"] += 1
            return self.repair_text
        return "[]"

    def complete_json(self, prompt, model=None, **kw):
        if "decide whether a new assistant statement contradicts" in prompt:
            self.calls["detect"] += 1
            return self.detect
        return {"contradiction": False}


class FakeRedis:
    """Minimal in-process Redis double supporting the ops RedisCommitmentStore uses."""
    def __init__(self):
        self.lists = {}
        self.sets = {}

    def pipeline(self):
        outer = self
        class P:
            def __init__(self): self.ops = []
            def rpush(self, k, v): self.ops.append(("rpush", k, v)); return self
            def ltrim(self, k, a, b): self.ops.append(("ltrim", k, a, b)); return self
            def sadd(self, k, v): self.ops.append(("sadd", k, v)); return self
            def execute(self):
                for op in self.ops:
                    getattr(outer, op[0])(*op[1:])
        return P()

    def rpush(self, k, v): self.lists.setdefault(k, []).append(v)
    def ltrim(self, k, a, b):
        cur = self.lists.get(k, [])
        self.lists[k] = cur[a:] if b == -1 else cur[a:b + 1]
    def sadd(self, k, v): self.sets.setdefault(k, set()).add(v)
    def srem(self, k, v): self.sets.get(k, set()).discard(v)
    def smembers(self, k): return set(self.sets.get(k, set()))
    def lrange(self, k, a, b): return list(self.lists.get(k, []))
    def llen(self, k): return len(self.lists.get(k, []))
    def delete(self, k): self.lists.pop(k, None); self.sets.pop(k, None)


# ── Commitment model ────────────────────────────────────────────────────

def test_commitment_roundtrip():
    c = Commitment(claim="Refund window is 30 days", topic="refund window",
                   session="u1", source_query="q", source_response="r", turn=3)
    d = c.to_dict()
    c2 = Commitment.from_dict(d)
    assert c2.claim == c.claim and c2.topic == c.topic and c2.turn == 3
    assert c2.session == "u1"


# ── Relevance ───────────────────────────────────────────────────────────

def test_overlap_same_topic_beats_unrelated():
    a = Commitment(claim="Refund window is 30 days", topic="refund window")
    b = Commitment(claim="Refunds allowed up to 30 days after purchase", topic="refund window")
    c = Commitment(claim="Store hours are 9 to 5", topic="store hours")
    assert _overlap_score(a, b) > _overlap_score(a, c)
    assert _overlap_score(a, c) < 0.3   # unrelated falls below threshold


def test_relevant_retrieves_on_topic_only():
    mem = ConversationMemory(store=InMemoryCommitmentStore())  # no LLM needed
    mem.store.add(Commitment(claim="Refund window is 30 days", topic="refund window", session="u1"))
    mem.store.add(Commitment(claim="Store hours are 9 to 5", topic="store hours", session="u1"))
    new = [Commitment(claim="Refunds at 45 days?", topic="refund window", session="u1")]
    rel = mem.relevant("u1", new)
    assert len(rel) == 1
    assert "Refund window" in rel[0].claim


def test_relevant_is_session_scoped():
    mem = ConversationMemory(store=InMemoryCommitmentStore())
    mem.store.add(Commitment(claim="Refund window is 30 days", topic="refund window", session="u1"))
    new = [Commitment(claim="Refunds at 45 days?", topic="refund window", session="u2")]
    # u2 has no prior commitments -> nothing relevant
    assert mem.relevant("u2", new) == []


def test_relevant_respects_top_k():
    mem = ConversationMemory(store=InMemoryCommitmentStore(), top_k=2)
    for i in range(5):
        mem.store.add(Commitment(claim=f"Refund rule {i} window days", topic="refund window", session="u1"))
    new = [Commitment(claim="Refund window question", topic="refund window", session="u1")]
    assert len(mem.relevant("u1", new)) == 2


# ── Extraction ──────────────────────────────────────────────────────────

def test_extract_parses_list():
    llm = FakeLLM(extract_map={"30 days": '[{"claim":"Refund window is 30 days","topic":"refund window"}]'})
    mem = ConversationMemory(llm=llm, store=InMemoryCommitmentStore())
    out = mem.extract("policy?", "We refund within 30 days.", session="u1")
    assert len(out) == 1
    assert out[0].claim == "Refund window is 30 days"
    assert out[0].topic == "refund window"
    assert out[0].session == "u1"
    assert out[0].turn == 0


def test_extract_tolerates_fences_and_objects():
    llm = FakeLLM(extract_map={"x": '```json\n[{"claim":"A","topic":"t"}]\n```'})
    mem = ConversationMemory(llm=llm, store=InMemoryCommitmentStore())
    out = mem.extract("x", "x", session="s")
    assert len(out) == 1 and out[0].claim == "A"


def test_extract_empty_on_no_commitment():
    llm = FakeLLM(extract_map={})   # returns "[]"
    mem = ConversationMemory(llm=llm, store=InMemoryCommitmentStore())
    assert mem.extract("hi", "hello there", session="s") == []


def test_extract_turn_increments_with_store():
    llm = FakeLLM(extract_map={"a": '[{"claim":"first","topic":"t"}]',
                               "b": '[{"claim":"second","topic":"t"}]'})
    mem = ConversationMemory(llm=llm, store=InMemoryCommitmentStore())
    first = mem.ingest("u1", "a", "a")     # extract + store
    second = mem.extract("b", "b", session="u1")
    assert first[0].turn == 0
    assert second[0].turn == 1             # store now has 1 commitment


# ── Detection ───────────────────────────────────────────────────────────

def test_detect_flags_contradiction_with_provenance():
    detect = {"contradiction": True, "new_claim": "Refunds at 45 days",
              "prior_claim": "Refund window is 30 days", "explanation": "conflict", "confidence": 0.8}
    llm = FakeLLM(detect=detect)
    mem = ConversationMemory(llm=llm, store=InMemoryCommitmentStore())
    prior = Commitment(claim="Refund window is 30 days", topic="refund window",
                       session="u1", source_query="orig q", source_response="orig r")
    new = [Commitment(claim="Refunds at 45 days", topic="refund window", session="u1")]
    finding = mem.detect(new, [prior])
    assert finding.contradiction is True
    assert finding.prior_claim == "Refund window is 30 days"
    assert finding.prior_query == "orig q"
    assert finding.prior_response == "orig r"
    assert finding.confidence == 0.8


def test_detect_false_without_priors():
    mem = ConversationMemory(llm=FakeLLM(), store=InMemoryCommitmentStore())
    new = [Commitment(claim="anything", topic="t", session="u1")]
    assert mem.detect(new, []).contradiction is False


# ── Repair ──────────────────────────────────────────────────────────────

def test_repair_returns_text_on_contradiction():
    llm = FakeLLM(repair_text="Our policy is 30 days, so 45 days is outside the window.")
    mem = ConversationMemory(llm=llm, store=InMemoryCommitmentStore())
    finding = ContradictionFinding(contradiction=True, prior_claim="Refund window is 30 days")
    fixed = mem.repair("Refund at 45 days?", "Sure, 45 days is fine.", finding)
    assert fixed and "30 days" in fixed


def test_repair_none_without_contradiction():
    mem = ConversationMemory(llm=FakeLLM(), store=InMemoryCommitmentStore())
    assert mem.repair("q", "r", ContradictionFinding(contradiction=False)) is None


# ── Stores ──────────────────────────────────────────────────────────────

def test_inmemory_store_session_isolation():
    s = InMemoryCommitmentStore()
    s.add(Commitment(claim="x", session="u1"))
    s.add(Commitment(claim="y", session="u2"))
    assert s.size("u1") == 1 and s.size("u2") == 1 and s.size() == 2
    s.clear("u1")
    assert s.size("u1") == 0 and s.size("u2") == 1
    s.clear()
    assert s.size() == 0


def test_inmemory_store_per_session_cap():
    s = InMemoryCommitmentStore(per_session=2)
    for i in range(5):
        s.add(Commitment(claim=f"c{i}", session="u1"))
    got = [c.claim for c in s.by_session("u1")]
    assert got == ["c3", "c4"]   # oldest dropped


def test_inmemory_store_is_protocol():
    assert isinstance(InMemoryCommitmentStore(), CommitmentStore)


def test_redis_store_via_fake_client():
    rs = RedisCommitmentStore(client=FakeRedis())
    rs.add(Commitment(claim="Refund 30d", topic="refund", session="u1"))
    rs.add(Commitment(claim="Hours 9-5", topic="hours", session="u1"))
    rs.add(Commitment(claim="Other", session="u2"))
    assert rs.size("u1") == 2 and rs.size("u2") == 1 and rs.size() == 3
    assert [c.claim for c in rs.by_session("u1")] == ["Refund 30d", "Hours 9-5"]
    rs.clear("u1")
    assert rs.size("u1") == 0 and rs.size("u2") == 1
    rs.clear()
    assert rs.size() == 0


def test_redis_store_per_session_cap():
    rs = RedisCommitmentStore(client=FakeRedis(), per_session=2)
    for i in range(4):
        rs.add(Commitment(claim=f"c{i}", session="u1"))
    assert [c.claim for c in rs.by_session("u1")] == ["c2", "c3"]


# ── Dedup on ingest ──────────────────────────────────────────────────────

def test_ingest_dedups_identical_claim():
    mem = ConversationMemory(store=InMemoryCommitmentStore())   # no LLM needed
    same = lambda: Commitment(claim="Refund window is 30 days", topic="refund window", session="u1")
    mem.ingest_commitments([same(), same()])
    assert mem.store.size("u1") == 1
    mem.ingest_commitments([Commitment(claim="Store hours are 9 to 5", topic="store hours", session="u1")])
    assert mem.store.size("u1") == 2


def test_ingest_dedup_normalizes_case_and_punctuation():
    mem = ConversationMemory(store=InMemoryCommitmentStore())
    mem.ingest_commitments([Commitment(claim="Refund window is 30 days", session="u1")])
    mem.ingest_commitments([Commitment(claim="  refund window is 30 days.  ", session="u1")])
    assert mem.store.size("u1") == 1


def test_ingest_dedup_is_session_scoped():
    mem = ConversationMemory(store=InMemoryCommitmentStore())
    mem.ingest_commitments([Commitment(claim="Refund window is 30 days", session="u1")])
    mem.ingest_commitments([Commitment(claim="Refund window is 30 days", session="u2")])
    assert mem.store.size("u1") == 1 and mem.store.size("u2") == 1


def test_ingest_dedup_keeps_conflicting_claim():
    # Safety: a claim that conflicts with a prior one must NOT be merged away.
    mem = ConversationMemory(store=InMemoryCommitmentStore())
    mem.ingest_commitments([Commitment(claim="Refund window is 30 days", topic="refund window", session="u1")])
    mem.ingest_commitments([Commitment(claim="Refund window is 14 days", topic="refund window", session="u1")])
    claims = {c.claim for c in mem.store.by_session("u1")}
    assert claims == {"Refund window is 30 days", "Refund window is 14 days"}


def test_ingest_dedup_off_keeps_duplicates():
    mem = ConversationMemory(store=InMemoryCommitmentStore(), dedup=False)
    mem.ingest_commitments([Commitment(claim="Refund window is 30 days", session="u1"),
                            Commitment(claim="Refund window is 30 days", session="u1")])
    assert mem.store.size("u1") == 2


# ── Redundancy-aware eviction ─────────────────────────────────────────────

def test_eviction_keeps_unique_drops_redundant():
    # cap 2. A is unique; B and B2 are the same fact (shared topic). Plain FIFO
    # would drop A (oldest) on the third add; redundancy eviction drops B
    # instead, keeping the unique, load-bearing commitment A.
    s = InMemoryCommitmentStore(per_session=2)
    A  = Commitment(claim="Refund window is 30 days", topic="refund window", session="u1")
    B  = Commitment(claim="Store hours are 9 to 5", topic="store hours", session="u1")
    B2 = Commitment(claim="We are open 9 to 5 daily", topic="store hours", session="u1")
    s.add(A); s.add(B); s.add(B2)
    claims = [c.claim for c in s.by_session("u1")]
    assert len(claims) == 2
    assert "Refund window is 30 days" in claims          # unique fact survived
    assert "Store hours are 9 to 5" not in claims        # redundant restatement dropped


def test_eviction_fifo_when_nothing_redundant():
    # Unrelated claims -> redundancy is zero everywhere -> degrade to oldest-first.
    s = InMemoryCommitmentStore(per_session=2)
    for i in range(5):
        s.add(Commitment(claim=f"c{i}", session="u1"))
    assert [c.claim for c in s.by_session("u1")] == ["c3", "c4"]


def test_eviction_fifo_mode_explicit():
    s = InMemoryCommitmentStore(per_session=2, eviction="fifo")
    A  = Commitment(claim="Refund window is 30 days", topic="refund window", session="u1")
    B  = Commitment(claim="Store hours are 9 to 5", topic="store hours", session="u1")
    B2 = Commitment(claim="We are open 9 to 5 daily", topic="store hours", session="u1")
    s.add(A); s.add(B); s.add(B2)
    assert [c.claim for c in s.by_session("u1")] == ["Store hours are 9 to 5", "We are open 9 to 5 daily"]


# ── Firewall: memory-aware path ──────────────────────────────────────────

def _refund_llm():
    return FakeLLM(
        extract_map={
            "30 days": '[{"claim":"Refund window is 30 days, no exceptions","topic":"refund window"}]',
            "45 days": '[{"claim":"Refunds are allowed at 45 days","topic":"refund window"}]',
        },
        detect={"contradiction": True, "new_claim": "Refunds are allowed at 45 days",
                "prior_claim": "Refund window is 30 days, no exceptions",
                "explanation": "45 days contradicts the 30-day window.", "confidence": 0.9},
        repair_text="To confirm: refunds are within 30 days only, so 45 days is outside the window.",
    )


def _refund_app(q):
    return "Refunds within 30 days, no exceptions." if "policy" in q else "Sure, refund at 45 days is fine."


def test_firewall_monitor_flags_and_offers_repair():
    orig = fw_mod.LLMClient
    fw_mod.LLMClient = lambda *a, **kw: _refund_llm()
    try:
        fw = fw_mod.Firewall(app=_refund_app, mode="monitor")
        r1 = fw.check("What is the refund policy?", session="u1")
        assert not r1.contradiction_detected
        r2 = fw.check("Can I refund at 45 days?", session="u1")
        assert r2.contradiction_detected
        assert r2.response.startswith("Sure")            # monitor passes original
        assert r2.repaired_response and "30 days" in r2.repaired_response
        assert r2.grounded_on == "Refund window is 30 days, no exceptions"
        assert r2.confidence == 0.9
        assert r2.session == "u1"
    finally:
        fw_mod.LLMClient = orig


def test_firewall_session_isolation():
    orig = fw_mod.LLMClient
    fw_mod.LLMClient = lambda *a, **kw: _refund_llm()
    try:
        fw = fw_mod.Firewall(app=_refund_app, mode="monitor")
        fw.check("What is the refund policy?", session="u1")
        # Fresh session has no prior commitment -> no contradiction.
        r = fw.check("Can I refund at 45 days?", session="u2")
        assert not r.contradiction_detected
    finally:
        fw_mod.LLMClient = orig


def test_firewall_block_returns_repaired_reply():
    orig = fw_mod.LLMClient
    fw_mod.LLMClient = lambda *a, **kw: _refund_llm()
    try:
        fw = fw_mod.Firewall(app=_refund_app, mode="block")
        fw.check("What is the refund policy?", session="b1")
        r = fw.check("Can I refund at 45 days?", session="b1")
        assert r.blocked and r.contradiction_detected
        assert "30 days" in r.response                   # corrected, not the fallback
        s = fw.summary()
        assert s["contradictions_detected"] == 1 and s["responses_repaired"] == 1
    finally:
        fw_mod.LLMClient = orig


def test_firewall_block_falls_back_when_repair_off():
    orig = fw_mod.LLMClient
    fw_mod.LLMClient = lambda *a, **kw: _refund_llm()
    try:
        fw = fw_mod.Firewall(app=_refund_app, mode="block", repair=False)
        fw.check("What is the refund policy?", session="b1")
        r = fw.check("Can I refund at 45 days?", session="b1")
        assert r.blocked and r.contradiction_detected
        assert r.repaired_response is None
        assert "connect you with someone" in r.response  # default fallback
    finally:
        fw_mod.LLMClient = orig


def test_firewall_legacy_path_still_works():
    # memory_aware=False -> the original recency-window single-call behavior.
    orig = fw_mod.LLMClient

    class LegacyLLM:
        fast_model = "fake"
        def __init__(self, *a, **kw): pass
        def complete_json(self, prompt, model=None, **kw):
            return {"contradiction": False}

    fw_mod.LLMClient = LegacyLLM
    try:
        fw = fw_mod.Firewall(app=lambda q: "x", mode="monitor", memory_aware=False, window=3)
        fw.check("a?")
        fw.check("b?")
        assert fw.cache.size() == 2
        assert fw.memory is None
        fw.reset()
        assert fw.cache.size() == 0
    finally:
        fw_mod.LLMClient = orig


# ── Firewall: confidence gate ─────────────────────────────────────────────

def _refund_llm_conf(conf):
    return FakeLLM(
        extract_map={
            "30 days": '[{"claim":"Refund window is 30 days, no exceptions","topic":"refund window"}]',
            "45 days": '[{"claim":"Refunds are allowed at 45 days","topic":"refund window"}]',
        },
        detect={"contradiction": True, "new_claim": "Refunds are allowed at 45 days",
                "prior_claim": "Refund window is 30 days, no exceptions",
                "explanation": "45 days contradicts the 30-day window.", "confidence": conf},
        repair_text="To confirm: refunds are within 30 days only, so 45 days is outside the window.",
    )


def test_firewall_min_confidence_suppresses_low_conf_block():
    orig = fw_mod.LLMClient
    fw_mod.LLMClient = lambda *a, **kw: _refund_llm_conf(0.4)
    try:
        fw = fw_mod.Firewall(app=_refund_app, mode="block", min_confidence=0.6)
        fw.check("What is the refund policy?", session="b1")
        r = fw.check("Can I refund at 45 days?", session="b1")
        assert r.contradiction_detected         # still detected and monitored
        assert not r.blocked                     # but not acted on below threshold
        assert r.response.startswith("Sure")     # original reply passed through
        assert r.repaired_response is None       # no repair generated below threshold
    finally:
        fw_mod.LLMClient = orig


def test_firewall_min_confidence_allows_high_conf_block():
    orig = fw_mod.LLMClient
    fw_mod.LLMClient = lambda *a, **kw: _refund_llm_conf(0.9)
    try:
        fw = fw_mod.Firewall(app=_refund_app, mode="block", min_confidence=0.6)
        fw.check("What is the refund policy?", session="b1")
        r = fw.check("Can I refund at 45 days?", session="b1")
        assert r.blocked and r.contradiction_detected
        assert "30 days" in r.response
    finally:
        fw_mod.LLMClient = orig


def test_firewall_default_min_confidence_acts_on_any():
    # Default min_confidence=0.0 preserves prior behavior: even low confidence acts.
    orig = fw_mod.LLMClient
    fw_mod.LLMClient = lambda *a, **kw: _refund_llm_conf(0.4)
    try:
        fw = fw_mod.Firewall(app=_refund_app, mode="block")
        fw.check("What is the refund policy?", session="b1")
        r = fw.check("Can I refund at 45 days?", session="b1")
        assert r.blocked and r.contradiction_detected
    finally:
        fw_mod.LLMClient = orig


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
