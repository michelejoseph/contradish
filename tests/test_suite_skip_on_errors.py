"""
Tests for Suite._run_one's error-dominated skip path.

When too many of a case's app(...) calls fail (the original call, or more
than half of [original] + paraphrases), the case must NOT be sent to the
consistency judge -- judging "[APP ERROR: ...]" text produces a plausible
but fabricated strain number. Instead the case comes back with
skipped=True, consistency_score=None, and the real error preserved.

The judge is a MagicMock here so a call to it (which should never happen
on the skip path) is easy to assert against, without needing a real API key.
"""
from unittest.mock import MagicMock

from contradish import Suite, TestCase
from contradish.models import RiskLevel


def _make_suite(app) -> Suite:
    suite = Suite(app=app, api_key="sk-fake-not-a-real-key", provider="openai")
    suite._judge = MagicMock()
    return suite


class TestSkipOnErrorDominatedCase:
    def test_original_call_errors_skips_even_if_paraphrases_succeed(self, monkeypatch):
        monkeypatch.setattr("contradish.runner.time.sleep", lambda s: None)
        def app(q):
            if q == "original question":
                raise RuntimeError("500 internal server error")
            return "a fine answer"
        suite = _make_suite(app)
        suite._runner.generate_paraphrases = MagicMock(return_value=["p1", "p2"])
        tc = TestCase(input="original question", name="case")
        result = suite._run_one(tc, paraphrases=2, verbose=False)

        assert result.skipped is True
        assert result.skip_reason == "api_errors"
        assert result.consistency_score is None
        assert result.contradiction_score is None
        assert result.n_errors == 1
        assert result.risk == RiskLevel.LOW
        suite._judge.evaluate_consistency.assert_not_called()

    def test_majority_of_variants_error_skips(self, monkeypatch):
        monkeypatch.setattr("contradish.runner.time.sleep", lambda s: None)
        calls = {"n": 0}
        def app(q):
            calls["n"] += 1
            # original + 3 paraphrases = 4 calls; fail 3 of them (>half).
            # Non-transient message so each failure is exactly one attempt
            # (no retries consumed), keeping the call count easy to reason about.
            if calls["n"] <= 3:
                raise RuntimeError("invalid request")
            return "a fine answer"
        suite = _make_suite(app)
        suite._runner.generate_paraphrases = MagicMock(return_value=["p1", "p2", "p3"])
        tc = TestCase(input="q", name="case")
        result = suite._run_one(tc, paraphrases=3, verbose=False)

        assert result.skipped is True
        assert result.n_errors == 3
        assert result.consistency_score is None
        suite._judge.evaluate_consistency.assert_not_called()

    def test_minority_of_variants_error_still_judged_normally(self, monkeypatch):
        monkeypatch.setattr("contradish.runner.time.sleep", lambda s: None)
        calls = {"n": 0}
        def app(q):
            calls["n"] += 1
            # original + 3 paraphrases = 4 calls; fail only 1 (minority)
            if calls["n"] == 4:
                raise RuntimeError("429 rate limited")
            return "a fine answer"
        suite = _make_suite(app)
        suite._runner.generate_paraphrases = MagicMock(return_value=["p1", "p2", "p3"])
        suite._judge.evaluate_consistency.return_value = {"consistency_score": 0.9}
        suite._judge.find_contradictions.return_value = []
        tc = TestCase(input="q", name="case")
        result = suite._run_one(tc, paraphrases=3, verbose=False)

        assert result.skipped is False
        assert result.consistency_score == 0.9
        suite._judge.evaluate_consistency.assert_called_once()

    def test_report_excludes_skipped_from_passed_and_failed(self, monkeypatch):
        monkeypatch.setattr("contradish.runner.time.sleep", lambda s: None)
        def always_errors(q):
            raise RuntimeError("503 unavailable")
        suite = _make_suite(always_errors)
        suite._runner.generate_paraphrases = MagicMock(return_value=["p1"])
        suite.add(TestCase(input="q", name="only case"))
        report = suite.run(paraphrases=1, verbose=False, concurrency=1)

        assert len(report.skipped) == 1
        assert report.passed == []
        assert report.failed == []
        assert report.cai_strain is None
        assert report.headline_strain is None
