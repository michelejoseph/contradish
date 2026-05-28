"""
Tests for contradish.report: rendering a CommitmentLedger into the verifiable
HTML audit (the seal). Run with: pytest tests/test_report.py  (pure stdlib).
"""
from contradish.ledger import CommitmentLedger
from contradish.report import audit_report_html
from contradish.memory import Commitment, ContradictionFinding


def _ledger():
    led = CommitmentLedger()
    led.record_commitment(Commitment(claim="Refund window is 30 days", topic="refund window", session="u1"))
    led.record_commitment(Commitment(claim="Standard shipping is 5 days", topic="shipping", session="u1"))
    led.record_contradiction(
        ContradictionFinding(contradiction=True, new_claim="Refund window is 60 days",
                             prior_claim="Refund window is 30 days", confidence=0.9),
        session="u1")
    return led


def test_report_renders_core_fields():
    led = _ledger()
    htmlout = audit_report_html(led, model="gpt-4o")
    assert "contradish audit" in htmlout
    assert "gpt-4o" in htmlout
    assert led.head() in htmlout                       # the fingerprint is shown
    assert "Refund window is 30 days" in htmlout        # a commitment is in the timeline
    assert "Refund window is 60 days" in htmlout        # the contradiction is in the timeline
    assert htmlout.strip().startswith("<!doctype html>")


def test_report_shows_intact_when_verified():
    led = _ledger()
    out = audit_report_html(led)
    assert "RECORD INTACT" in out
    assert "VERIFICATION FAILED" not in out


def test_report_shows_altered_when_tampered():
    led = _ledger()
    led._entries[0].payload["claim"] = "Refund window is 14 days"   # silent edit
    out = audit_report_html(led)
    assert "VERIFICATION FAILED" in out
    assert "RECORD INTACT" not in out


def test_report_handles_empty_ledger():
    out = audit_report_html(CommitmentLedger(), model="empty")
    assert "No entries recorded yet." in out
    assert "RECORD INTACT" in out                       # an empty ledger is trivially intact


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
