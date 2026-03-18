"""
Tests for contradish.
Run with: pytest tests/
"""

import pytest
from unittest.mock import MagicMock, patch
from contradish import Suite, TestCase, RegressionSuite
from contradish.models import RiskLevel, Report, RegressionResult


# --- Fixtures ---

def make_stable_app(answer: str = "Refunds are only allowed within 30 days."):
    """An app that always returns the same answer."""
    def app(question: str) -> str:
        return answer
    return app


def make_unstable_app():
    """An app that sometimes contradicts itself."""
    call_count = [0]
    def app(question: str) -> str:
        call_count[0] += 1
        if call_count[0] % 3 == 0:
            return "Yes, refunds are allowed after 60 days."
        return "No, refunds are only allowed within 30 days."
    return app


# --- Unit tests ---

class TestTestCase:
    def test_auto_name_from_input(self):
        tc = TestCase(input="What is the refund policy?")
        assert tc.name == "What is the refund policy?"

    def test_long_input_truncated_name(self):
        long_input = "A" * 100
        tc = TestCase(input=long_input)
        assert len(tc.name) <= 53  # 50 + "..."

    def test_explicit_name(self):
        tc = TestCase(input="foo", name="my test")
        assert tc.name == "my test"

    def test_with_context(self):
        tc = TestCase(input="foo", context="some context")
        assert tc.context == "some context"


class TestReport:
    def _make_report(self, consistency=0.9, contradiction=0.1, grounding=0.9):
        from contradish.models import TestResult
        tc = TestCase(input="test", context="ctx")
        result = TestResult(
            test_case=tc,
            outputs=["answer"],
            paraphrases=["paraphrase"],
            consistency_score=consistency,
            contradiction_score=contradiction,
            grounding_score=grounding,
            risk=RiskLevel.LOW,
        )
        return Report(results=[result])

    def test_avg_consistency(self):
        report = self._make_report(consistency=0.85)
        assert abs(report.avg_consistency - 0.85) < 0.001

    def test_passed_above_threshold(self):
        report = self._make_report(consistency=0.95, contradiction=0.05, grounding=0.95)
        report.thresholds = {"consistency": 0.80, "contradiction": 0.25, "grounding": 0.80}
        assert len(report.passed) == 1
        assert len(report.failed) == 0

    def test_failed_below_threshold(self):
        report = self._make_report(consistency=0.60, contradiction=0.40, grounding=0.60)
        report.thresholds = {"consistency": 0.80, "contradiction": 0.25, "grounding": 0.80}
        assert len(report.failed) == 1

    def test_summary_string(self):
        report = self._make_report()
        s = report.summary()
        assert "CONTRADISH REPORT" in s
        assert "Tests run" in s


class TestRegressionResult:
    def _make_regression(self, baseline_consistency=0.90, candidate_consistency=0.75):
        from contradish.models import TestResult
        def make_report(consistency):
            tc = TestCase(input="test")
            result = TestResult(
                test_case=tc,
                outputs=["answer"],
                paraphrases=[],
                consistency_score=consistency,
                contradiction_score=0.1,
                risk=RiskLevel.LOW,
            )
            return Report(results=[result])

        return RegressionResult(
            baseline_label="v1",
            candidate_label="v2",
            baseline_report=make_report(baseline_consistency),
            candidate_report=make_report(candidate_consistency),
        )

    def test_consistency_delta(self):
        r = self._make_regression(0.90, 0.75)
        assert abs(r.consistency_delta - (-0.15)) < 0.001

    def test_fail_if_below_raises_on_regression(self):
        r = self._make_regression(0.90, 0.70)
        with pytest.raises(AssertionError) as exc_info:
            r.fail_if_below(consistency=0.85)
        assert "REGRESSION DETECTED" in str(exc_info.value)

    def test_fail_if_below_passes_when_ok(self):
        r = self._make_regression(0.90, 0.88)
        # Should not raise
        r.fail_if_below(consistency=0.85)

    def test_str_output(self):
        r = self._make_regression()
        s = str(r)
        assert "REGRESSION" in s
        assert "v1" in s
        assert "v2" in s


# --- Integration tests (require ANTHROPIC_API_KEY) ---
# These are skipped unless you set ANTHROPIC_API_KEY in your environment.

@pytest.mark.integration
class TestSuiteIntegration:
    """Integration tests — run with: pytest -m integration"""

    def test_basic_run(self):
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        app = make_stable_app()
        suite = Suite(api_key=api_key, app=app)
        suite.add_test(TestCase(
            name="refund policy",
            input="Can I get a refund after 45 days?",
            context="Refunds are only allowed within 30 days of purchase.",
        ))

        report = suite.run(paraphrases=3, verbose=False)
        assert len(report.results) == 1
        assert report.results[0].consistency_score is not None
        assert 0.0 <= report.results[0].consistency_score <= 1.0

    def test_unstable_app_flagged(self):
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        app = make_unstable_app()
        suite = Suite(api_key=api_key, app=app)
        suite.add_test(TestCase(
            name="refund policy",
            input="Can I get a refund after 45 days?",
        ))

        report = suite.run(paraphrases=5, verbose=False)
        result = report.results[0]
        # Unstable app should produce contradictions
        assert result.contradiction_score is not None
