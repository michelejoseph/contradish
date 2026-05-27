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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
