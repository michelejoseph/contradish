"""
Tests for contradish.replay: transcript loading (chat / paired / multi-session /
multi-conversation, JSON and JSONL) and the offline replay engine.
Run with: pytest tests/test_replay.py
No API key required: the memory layer is replaced by a scriptable fake.
"""
import json
import os
import tempfile

from contradish.replay import (
    load_transcript,
    replay_transcript,
    replay,
    ReplayTurn,
    ReplayReport,
)
from contradish.memory import Commitment, ContradictionFinding


# ── Loader: format tolerance ─────────────────────────────────────────────

def test_load_openai_chat_messages_pairs_turns():
    chat = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "Refund policy?"},
        {"role": "assistant", "content": "30 days, no exceptions."},
        {"role": "user", "content": "After 45 days?"},
        {"role": "assistant", "content": "Sure, 45 is fine."},
    ]
    turns = load_transcript(chat)
    assert len(turns) == 2
    assert turns[0].query == "Refund policy?" and turns[0].response == "30 days, no exceptions."
    assert turns[0].index == 0 and turns[1].index == 1
    assert all(t.session == "default" for t in turns)


def test_load_paired_with_sessions_indexes_per_session():
    paired = [
        {"session": "A", "query": "hours?", "response": "9 to 5"},
        {"session": "B", "input": "refund?", "output": "30 days"},
        {"session": "A", "prompt": "weekends?", "completion": "no"},
    ]
    turns = load_transcript(paired)
    a = [t for t in turns if t.session == "A"]
    b = [t for t in turns if t.session == "B"]
    assert [t.index for t in a] == [0, 1]
    assert [t.index for t in b] == [0]
    assert a[0].query == "hours?" and a[0].response == "9 to 5"


def test_load_list_of_conversations_with_nested_messages():
    convs = [
        {"id": "c1", "messages": [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
        {"conversation_id": "c2", "turns": [
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
    ]
    turns = load_transcript(convs)
    assert {t.session for t in turns} == {"c1", "c2"}
    assert len(turns) == 2


def test_load_top_level_conversations_container():
    container = {"conversations": [
        {"session": "s1", "messages": [
            {"role": "user", "content": "u"}, {"role": "assistant", "content": "r"}]}]}
    turns = load_transcript(container)
    assert len(turns) == 1 and turns[0].session == "s1"


def test_load_jsonl_file(tmp_path=None):
    rows = [
        {"role": "user", "content": "Refund policy?"},
        {"role": "assistant", "content": "30 days."},
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        turns = load_transcript(path)
        assert len(turns) == 1 and turns[0].response == "30 days."
    finally:
        os.remove(path)


def test_load_json_array_file():
    data = [{"query": "a", "response": "b"}]
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        turns = load_transcript(path)
        assert len(turns) == 1 and turns[0].query == "a"
    finally:
        os.remove(path)


def test_load_skips_unpaired_and_nondict():
    msgs = ["junk", {"role": "assistant", "content": "answer with no question"},
            {"role": "system", "content": "ignored"}]
    turns = load_transcript(msgs)
    # The assistant message with no prior user still becomes a turn (empty query);
    # the string and system message contribute nothing.
    assert len(turns) == 1 and turns[0].query == "" and turns[0].response.startswith("answer")


def test_load_empty_is_empty():
    assert load_transcript([]) == []
    assert load_transcript({}) == []


# ── Engine ────────────────────────────────────────────────────────────────

class FakeMemory:
    """
    Stand-in for ConversationMemory. A response commits to its own text; a
    response containing '45 days' contradicts any prior on the same session
    containing '30 days'. Tracks ingest so cross-turn accumulation is exercised.
    """
    def __init__(self):
        self.by_session = {}

    def check(self, session, query, response):
        new = [Commitment(claim=response, topic="refund", session=session, source_query=query)]
        finding = ContradictionFinding(contradiction=False)
        if "45 days" in response:
            prior = next((c for c in self.by_session.get(session, []) if "30 days" in c.claim), None)
            if prior:
                finding = ContradictionFinding(
                    contradiction=True, new_claim=response, prior_claim=prior.claim,
                    prior_query=prior.source_query, explanation="45 contradicts 30",
                    confidence=0.9)
        finding._new_commitments = new
        return finding

    def extract(self, query, response, session="default"):
        return [Commitment(claim=response, session=session, source_query=query)]

    def repair(self, query, response, finding):
        return "Corrected: 30 days only."

    def ingest_commitments(self, commitments):
        for c in commitments:
            self.by_session.setdefault(c.session, []).append(c)


def _refund_chat():
    return [
        {"role": "user", "content": "Refund policy?"},
        {"role": "assistant", "content": "Refunds within 30 days, no exceptions."},
        {"role": "user", "content": "What are your hours?"},
        {"role": "assistant", "content": "9 to 5."},
        {"role": "user", "content": "Can I refund after 45 days?"},
        {"role": "assistant", "content": "Sure, 45 days is fine."},
    ]


def test_engine_detects_cross_turn_contradiction_and_grounds_it():
    turns = load_transcript(_refund_chat())
    report = replay_transcript(turns, memory=FakeMemory())
    assert len(report.contradictions) == 1
    c = report.contradictions[0]
    assert c.turn_index == 2            # the 45-day turn (0-based, 3rd assistant reply)
    assert c.prior_turn_index == 0      # grounded to the 30-day turn
    assert c.prior_claim == "Refunds within 30 days, no exceptions."
    assert c.confidence == 0.9
    assert report.n_turns == 3 and report.n_sessions == 1


def test_engine_turn_zero_never_flagged():
    # A single turn cannot contradict anything.
    turns = load_transcript([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "Sure, 45 days is fine."}])
    report = replay_transcript(turns, memory=FakeMemory())
    assert report.contradictions == []


def test_engine_session_isolation():
    # The 30-day commitment is in session A; the 45-day reply is in session B.
    transcript = [
        {"session": "A", "query": "policy?", "response": "Refunds within 30 days."},
        {"session": "B", "query": "after 45?", "response": "Sure, 45 days is fine."},
    ]
    turns = load_transcript(transcript)
    report = replay_transcript(turns, memory=FakeMemory())
    assert report.contradictions == []   # different sessions never compared


def test_engine_repair_populated_when_requested():
    turns = load_transcript(_refund_chat())
    report = replay_transcript(turns, memory=FakeMemory(), repair=True)
    assert report.contradictions[0].repaired == "Corrected: 30 days only."


def test_engine_no_repair_by_default():
    turns = load_transcript(_refund_chat())
    report = replay_transcript(turns, memory=FakeMemory())
    assert report.contradictions[0].repaired is None


# ── Report ──────────────────────────────────────────────────────────────

def test_report_summary_and_to_dict():
    turns = load_transcript(_refund_chat())
    report = replay_transcript(turns, memory=FakeMemory(), repair=True)
    s = report.summary()
    assert "contradish replay" in s
    assert "turn 2 contradicts turn 0" in s
    assert "fix:" in s
    d = report.to_dict()
    assert d["n_contradictions"] == 1
    assert d["n_turns"] == 3
    assert d["contradictions"][0]["prior_turn_index"] == 0


def test_report_clean_transcript_summary():
    turns = load_transcript([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}])
    report = replay_transcript(turns, memory=FakeMemory())
    assert "no self-contradictions found" in report.summary()
    assert report.contradiction_rate == 0.0


def test_replay_convenience_loads_then_runs():
    report = replay(_refund_chat(), memory=FakeMemory())
    assert isinstance(report, ReplayReport)
    assert len(report.contradictions) == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
