"""
Tests for the system-level loop: production gaps -> benchmark cases -> repair.
Run with: pytest tests/test_improve_from_production.py

No API key required. cases_from_reconciliation is pure, and the
improve_from_production test mocks the module-level improve() so no model is
called.
"""
import importlib

# `contradish.improve` the attribute is the function (re-exported in __init__),
# so grab the actual submodule object to monkeypatch its module-level improve().
improve_mod = importlib.import_module("contradish.improve")
from contradish.improve import improve_from_production
from contradish.reconcile import (
    reconcile, cases_from_reconciliation, ReconciliationReport, CommitmentMatch,
)
from contradish.models import TestCase, TestResult, Report, RiskLevel
from contradish.replay import ReplayReport, ReplayContradiction


# ── helpers (mirror test_reconcile.py) ──────────────────────────────────────

def _bench_result(name, canonical, consistency, contradiction=0.0):
    tc = TestCase(input=name, name=name, canonical_answer=canonical)
    return TestResult(test_case=tc, paraphrases=[], outputs=[],
                      consistency_score=consistency, contradiction_score=contradiction,
                      risk=RiskLevel.LOW)


def _contradiction(prior_claim, query="q", session="u1", turn=1):
    return ReplayContradiction(
        session=session, turn_index=turn + 3, query=query, response="r",
        prior_turn_index=turn, prior_query="p", new_claim="changed",
        prior_claim=prior_claim, explanation="conflict", confidence=0.9)


def _match(verdict, prod_claim, prod_query=None):
    return CommitmentMatch(verdict=verdict, prod_claim=prod_claim, prod_query=prod_query)


# ── cases_from_reconciliation (pure) ────────────────────────────────────────

def test_cases_from_reconciliation_converts_gaps():
    rec = ReconciliationReport(matches=[
        _match("validity_gap", "Refund window is 30 days", prod_query="can i refund after 45 days?"),
        _match("coverage_gap", "Warranty is lifetime", prod_query="is the warranty forever?"),
    ], n_bench=1, n_prod=2)
    cases = cases_from_reconciliation(rec)
    assert len(cases) == 2
    inputs = {c.input for c in cases}
    assert "can i refund after 45 days?" in inputs and "is the warranty forever?" in inputs
    canon = {c.canonical_answer for c in cases}
    assert "Refund window is 30 days" in canon and "Warranty is lifetime" in canon
    assert all(c.contradiction_type == "adversarial" for c in cases)
    assert all(c.name.startswith("prod regression") for c in cases)


def test_cases_from_reconciliation_falls_back_to_claim_when_no_query():
    rec = ReconciliationReport(matches=[_match("validity_gap", "Refund window is 30 days")])
    cases = cases_from_reconciliation(rec)
    assert len(cases) == 1
    assert cases[0].input == "Refund window is 30 days"      # query absent -> claim
    assert cases[0].canonical_answer == "Refund window is 30 days"


def test_cases_from_reconciliation_excludes_confirmed_by_default():
    rec = ReconciliationReport(matches=[
        _match("confirmed", "Refund window is 30 days", prod_query="q1"),
        _match("validity_gap", "Warranty is lifetime", prod_query="q2"),
    ])
    cases = cases_from_reconciliation(rec)
    assert [c.canonical_answer for c in cases] == ["Warranty is lifetime"]


def test_cases_from_reconciliation_kinds_filter():
    rec = ReconciliationReport(matches=[
        _match("validity_gap", "A", prod_query="qa"),
        _match("coverage_gap", "B", prod_query="qb"),
    ])
    only_validity = cases_from_reconciliation(rec, kinds=("validity_gap",))
    assert [c.canonical_answer for c in only_validity] == ["A"]


def test_cases_from_reconciliation_dedupes():
    rec = ReconciliationReport(matches=[
        _match("validity_gap", "Refund window is 30 days", prod_query="same q"),
        _match("coverage_gap", "Refund window is 30 days", prod_query="same q"),
    ])
    assert len(cases_from_reconciliation(rec)) == 1


def test_cases_from_reconciliation_empty():
    assert cases_from_reconciliation(ReconciliationReport()) == []


def test_cases_from_reconciliation_threads_query_end_to_end():
    # Build via reconcile() so we confirm ReplayContradiction.query reaches the case.
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days, no exceptions", 0.95)])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days, no exceptions",
                       query="can I refund after 45 days?")],
        n_turns=10, sessions=["u1"])
    rec = reconcile(report, replay)
    assert len(rec.validity_gaps) == 1
    assert rec.validity_gaps[0].prod_query == "can I refund after 45 days?"
    cases = cases_from_reconciliation(rec)
    assert cases[0].input == "can I refund after 45 days?"
    assert cases[0].canonical_answer == "Refund window is 30 days, no exceptions"


# ── improve_from_production (mocked improve) ─────────────────────────────────

class _Recorder:
    def __init__(self, ret="RESULT"):
        self.ret = ret
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.ret


def _with_mocked_improve(body):
    """Run body(recorder) with improve_mod.improve replaced; always restore."""
    original = improve_mod.improve
    rec = _Recorder()
    improve_mod.improve = rec
    try:
        body(rec)
    finally:
        improve_mod.improve = original


def test_improve_from_production_calls_improve_with_derived_cases():
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days, no exceptions", 0.95)])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days, no exceptions", query="refund after 45?")],
        n_turns=10, sessions=["u1"])

    def body(recorder):
        out = improve_from_production(report, replay, system_prompt="SP", model="m",
                                      verbose=False, provider="openai")
        assert out == "RESULT"
        assert len(recorder.calls) == 1
        call = recorder.calls[0]
        assert call["system_prompt"] == "SP" and call["model"] == "m"
        assert call["provider"] == "openai"           # **improve_kwargs forwarded
        cases = call["cases"]
        assert len(cases) == 1 and cases[0].input == "refund after 45?"

    _with_mocked_improve(body)


def test_improve_from_production_returns_none_when_no_gaps():
    # Bench FAILED the case -> confirmed, not a gap. Nothing new to repair.
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days", 0.40, contradiction=0.7)])
    replay = ReplayReport(contradictions=[_contradiction("Refund window is 30 days")],
                          n_turns=10, sessions=["u1"])

    def body(recorder):
        out = improve_from_production(report, replay, verbose=False)
        assert out is None
        assert recorder.calls == []                    # improve never called

    _with_mocked_improve(body)


def test_improve_from_production_merges_and_dedupes_base_cases():
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days, no exceptions", 0.95)])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days, no exceptions", query="refund after 45?")],
        n_turns=10, sessions=["u1"])
    base = [
        TestCase(input="base one", canonical_answer="ans1"),
        # exact dup of the production-derived case:
        TestCase(input="refund after 45?", canonical_answer="Refund window is 30 days, no exceptions"),
    ]

    def body(recorder):
        improve_from_production(report, replay, base_cases=base, verbose=False)
        cases = recorder.calls[0]["cases"]
        assert len(cases) == 2                          # base(2) + derived(1) - dup(1)
        inputs = [c.input for c in cases]
        assert inputs.count("refund after 45?") == 1
        assert "base one" in inputs

    _with_mocked_improve(body)


def test_improve_from_production_forwards_kinds():
    # Only coverage gaps requested; a validity gap should be ignored.
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days, no exceptions", 0.95)])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days, no exceptions", query="qv", session="u1"),
        _contradiction("Warranty is lifetime", query="qc", session="u2"),
    ], n_turns=20, sessions=["u1", "u2"])

    def body(recorder):
        improve_from_production(report, replay, kinds=("coverage_gap",), verbose=False)
        cases = recorder.calls[0]["cases"]
        assert [c.canonical_answer for c in cases] == ["Warranty is lifetime"]

    _with_mocked_improve(body)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
