"""
Tests for contradish.ledger: the append-only, hash-chained commitment ledger
that turns contradiction detection into a tamper-evident audit over time.
Run with: pytest tests/test_ledger.py   (no API key, pure stdlib)
"""
from contradish.ledger import CommitmentLedger, _GENESIS
from contradish.memory import Commitment, ContradictionFinding


def _commit(claim, session="u1", topic="t"):
    return Commitment(claim=claim, topic=topic, session=session)


def _contra(prior, session="u1", conf=0.9):
    return ContradictionFinding(contradiction=True, prior_claim=prior,
                                new_claim="changed", confidence=conf)


def test_record_and_verify():
    led = CommitmentLedger()
    assert led.head() == _GENESIS and len(led) == 0
    led.record_commitment(_commit("Refund window is 30 days"))
    led.record_commitment(_commit("Shipping is 5 days"))
    led.record_contradiction(_contra("Refund window is 30 days"), session="u1")
    assert len(led) == 3
    assert led.verify() is True
    assert led.head() != _GENESIS


def test_tamper_breaks_verify():
    led = CommitmentLedger()
    led.record_commitment(_commit("Refund window is 30 days"))
    led.record_commitment(_commit("Shipping is 5 days"))
    assert led.verify() is True
    led._entries[0].payload["claim"] = "Refund window is 14 days"   # silent edit
    assert led.verify() is False


def test_delete_breaks_verify():
    led = CommitmentLedger()
    for i in range(3):
        led.record_commitment(_commit(f"claim {i}"))
    assert led.verify() is True
    del led._entries[1]                                             # drop a past entry
    assert led.verify() is False


def test_timeline_filters_by_session_and_type():
    led = CommitmentLedger()
    led.record_commitment(_commit("a", session="u1"))
    led.record_commitment(_commit("b", session="u2"))
    led.record_contradiction(_contra("a"), session="u1")
    assert len(led.timeline(session="u1")) == 2
    assert len(led.timeline(session="u2")) == 1
    assert len(led.timeline(type="contradiction")) == 1
    assert len(led.timeline(type="commitment")) == 2


def test_export_roundtrip_reverifies():
    led = CommitmentLedger()
    led.record_commitment(_commit("Refund window is 30 days"))
    led.record_commitment(_commit("Shipping is 5 days"))
    head = led.head()
    led2 = CommitmentLedger.from_dict(led.to_dict())
    assert led2.verify() is True
    assert led2.head() == head and len(led2) == 2


def test_audit_summary():
    led = CommitmentLedger()
    led.record_commitment(_commit("a"))
    led.record_commitment(_commit("b"))
    led.record_contradiction(_contra("a"), session="u1")
    s = led.audit_summary()
    assert s["commitments"] == 2 and s["contradictions"] == 1
    assert s["contradiction_rate"] == 0.5
    assert s["verified"] is True and s["entries"] == 3
    assert s["first_at"] is not None and s["last_at"] is not None


def test_record_event_is_generic():
    """record_event covers anything outside the commitment/contradiction
    shape, e.g. the run summary `contradish monitor` appends."""
    led = CommitmentLedger()
    led.record_event("monitor_run", "session-1", {"drift_rate": 0.3, "total": 10})
    assert len(led) == 1
    assert led.verify() is True
    s = led.audit_summary()
    assert s["other_event_types"] == {"monitor_run": 1}
    assert s["commitments"] == 0 and s["contradictions"] == 0


def test_save_and_load_roundtrip():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "ledger.json")
        led = CommitmentLedger()
        led.record_commitment(_commit("Refund window is 30 days"))
        led.record_contradiction(_contra("Refund window is 30 days"), session="u1")
        head = led.head()
        saved = led.save(path)
        assert saved.exists()

        reloaded = CommitmentLedger.load(path)
        assert reloaded.verify() is True
        assert reloaded.head() == head
        assert len(reloaded) == 2


def test_load_missing_file_returns_empty_ledger():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        led = CommitmentLedger.load(os.path.join(d, "does-not-exist.json"))
        assert len(led) == 0
        assert led.verify() is True


def test_save_then_tamper_on_disk_breaks_verify():
    import tempfile, os, json
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        led = CommitmentLedger()
        led.record_commitment(_commit("Refund window is 30 days"))
        led.save(path)

        raw = json.loads(open(path).read())
        raw["entries"][0]["payload"]["claim"] = "Refund window is 14 days"
        open(path, "w").write(json.dumps(raw))

        reloaded = CommitmentLedger.load(path)
        assert reloaded.verify() is False


def test_anchor_text_contains_head_and_count():
    led = CommitmentLedger()
    led.record_commitment(_commit("a"))
    led.record_commitment(_commit("b"))
    text = led.anchor_text(label="weekly-audit")
    assert "weekly-audit" in text
    assert led.head() in text
    assert "2 entries" in text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
