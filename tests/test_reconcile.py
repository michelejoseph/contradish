"""
Tests for the keystone integration: the shared Commitment unit and the
benchmark-vs-production reconciler.
Run with: pytest tests/test_reconcile.py
No API key required: reconciliation is pure (matches already-extracted claims).
"""
from contradish.models import TestCase, TestResult, Report, RiskLevel
from contradish.replay import ReplayReport, ReplayContradiction
from contradish.reconcile import reconcile, ReconciliationReport, CommitmentMatch
from contradish.memory import Commitment
from contradish.prompt_analyzer import (
    commitments_from_analysis, PromptAnalysis, PromptTension,
)


# ── Shared unit ─────────────────────────────────────────────────────────

def test_commitment_origin_roundtrip():
    c = Commitment(claim="Refund window is 30 days", topic="refund window", origin="benchmark")
    assert Commitment.from_dict(c.to_dict()).origin == "benchmark"
    # Default origin is "response", and missing key decodes to it.
    assert Commitment(claim="x").origin == "response"
    assert Commitment.from_dict({"claim": "x"}).origin == "response"


def test_commitments_from_analysis_emits_prompt_origin():
    analysis = PromptAnalysis(prompt="p", tensions=[
        PromptTension(clauses=["Be empathetic.", "Refunds within 30 days only."],
                      description="sympathy tips policy",
                      exploiting_techniques=["sympathy"], severity="high"),
    ])
    cs = commitments_from_analysis(analysis)
    assert len(cs) == 2
    assert all(c.origin == "prompt" for c in cs)
    claims = {c.claim for c in cs}
    assert "Be empathetic." in claims and "Refunds within 30 days only." in claims
    assert all(c.topic for c in cs)            # a topic was derived


def test_commitments_from_analysis_dedupes_repeated_clauses():
    shared = "Refunds within 30 days only."
    analysis = PromptAnalysis(prompt="p", tensions=[
        PromptTension(clauses=["Be empathetic.", shared], description="a", severity="high"),
        PromptTension(clauses=["Be concise.", shared], description="b", severity="critical"),
    ])
    cs = commitments_from_analysis(analysis)
    claims = [c.claim for c in cs]
    assert claims.count(shared) == 1          # the recurring clause appears once
    assert len(cs) == 3


def test_commitments_from_analysis_empty():
    assert commitments_from_analysis(PromptAnalysis(prompt="p", tensions=[])) == []


# ── Reconciler ────────────────────────────────────────────────────────────

def _bench_result(name, canonical, consistency, contradiction=0.0):
    tc = TestCase(input=name, name=name, canonical_answer=canonical)
    return TestResult(test_case=tc, paraphrases=[], outputs=[],
                      consistency_score=consistency, contradiction_score=contradiction,
                      risk=RiskLevel.LOW)


def _contradiction(prior_claim, session="u1", turn=1):
    return ReplayContradiction(
        session=session, turn_index=turn + 3, query="q", response="r",
        prior_turn_index=turn, prior_query="p", new_claim="changed",
        prior_claim=prior_claim, explanation="conflict", confidence=0.9)


def test_reconcile_flags_validity_gap():
    # Bench PASSED the refund-window case (high consistency), but it broke in prod.
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days, no exceptions", 0.95),
    ])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days, no exceptions")],
        n_turns=10, sessions=["u1"])
    rec = reconcile(report, replay)
    assert len(rec.validity_gaps) == 1
    assert rec.validity_gaps[0].verdict == "validity_gap"
    assert rec.validity_gaps[0].bench_passed is True
    assert rec.confirmed == [] and rec.coverage_gaps == []


def test_reconcile_confirmed_when_bench_failed():
    # Bench FAILED the case (low consistency, high contradiction) AND it broke in prod.
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days", 0.40, contradiction=0.7),
    ])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days")], n_turns=10, sessions=["u1"])
    rec = reconcile(report, replay)
    assert len(rec.confirmed) == 1
    assert rec.confirmed[0].bench_passed is False
    assert rec.validity_gaps == []


def test_reconcile_coverage_gap_when_untested():
    # Production broke on "warranty"; the benchmark only tested "refund".
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days", 0.95),
    ])
    replay = ReplayReport(contradictions=[
        _contradiction("Warranty is lifetime")], n_turns=10, sessions=["u1"])
    rec = reconcile(report, replay)
    assert len(rec.coverage_gaps) == 1
    assert rec.coverage_gaps[0].verdict == "coverage_gap"
    assert rec.coverage_gaps[0].bench_claim is None


def test_reconcile_metrics():
    # 2 prod breaks: one covered-but-passed (validity gap), one uncovered (coverage gap).
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days, no exceptions", 0.95),
    ])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days, no exceptions", session="u1"),
        _contradiction("Warranty is lifetime", session="u2"),
    ], n_turns=20, sessions=["u1", "u2"])
    rec = reconcile(report, replay)
    assert rec.n_prod == 2 and rec.n_bench == 1
    assert rec.coverage == 0.5             # 1 of 2 prod breaks matched a bench commitment
    assert rec.predictive_validity == 0.0  # the covered one passed the bench (not caught)


def test_reconcile_dedupes_repeated_prod_breaks():
    report = Report(results=[_bench_result("refund", "Refund window is 30 days", 0.95)])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days", turn=1),
        _contradiction("Refund window is 30 days", turn=5),   # same commitment, again
    ], n_turns=10, sessions=["u1"])
    rec = reconcile(report, replay)
    assert rec.n_prod == 1                 # deduped by claim


def test_reconcile_empty_replay():
    report = Report(results=[_bench_result("refund", "Refund window is 30 days", 0.95)])
    rec = reconcile(report, ReplayReport())
    assert rec.matches == [] and rec.n_prod == 0
    assert rec.coverage is None and rec.predictive_validity is None
    assert "no production contradictions" in rec.summary()


def test_reconcile_to_dict_and_summary():
    report = Report(results=[_bench_result("refund", "Refund window is 30 days", 0.95)])
    replay = ReplayReport(contradictions=[_contradiction("Refund window is 30 days")],
                          n_turns=10, sessions=["u1"])
    rec = reconcile(report, replay)
    d = rec.to_dict()
    assert d["n_validity_gaps"] == 1 and d["coverage"] == 1.0
    s = rec.summary()
    assert "contradish reconcile" in s and "validity gap" in s


def test_reconcile_custom_relevance_fn():
    # An always-1 scorer matches everything; an always-0 matches nothing.
    report = Report(results=[_bench_result("anything", "totally unrelated", 0.95)])
    replay = ReplayReport(contradictions=[_contradiction("Refund window is 30 days")],
                          n_turns=10, sessions=["u1"])
    matched = reconcile(report, replay, relevance_fn=lambda a, b: 1.0)
    assert len(matched.validity_gaps) == 1
    unmatched = reconcile(report, replay, relevance_fn=lambda a, b: 0.0)
    assert len(unmatched.coverage_gaps) == 1


# ── ReplayReport (de)serialization for the CLI path ──────────────────────

def test_replay_report_from_dict_roundtrip():
    original = ReplayReport(contradictions=[_contradiction("Refund window is 30 days")],
                            n_turns=10, n_commitments=20, sessions=["u1"])
    restored = ReplayReport.from_dict(original.to_dict())
    assert restored.n_turns == 10 and restored.n_commitments == 20
    assert restored.sessions == ["u1"]
    assert len(restored.contradictions) == 1
    assert restored.contradictions[0].prior_claim == "Refund window is 30 days"
    assert restored.contradictions[0].confidence == 0.9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
