"""
contradish test suite.

Unit tests run without any API key.
Integration tests require ANTHROPIC_API_KEY or OPENAI_API_KEY.

    pytest tests/                        # unit tests only
    pytest tests/ -m integration         # requires API key
"""

import pytest
from unittest.mock import MagicMock, patch

from contradish.models import (
    TestCase, TestResult, Report, RegressionResult,
    RiskLevel, ContradictionPair
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def stable_app(q: str) -> str:
    return "Refunds are only allowed within 30 days of purchase."

def unstable_app():
    calls = [0]
    def app(q: str) -> str:
        calls[0] += 1
        if calls[0] % 3 == 0:
            return "Yes, you can get a refund up to 60 days after purchase."
        return "No, refunds are only allowed within 30 days."
    return app

def _make_result(consistency=0.90, contradiction=0.05, risk=RiskLevel.LOW):
    tc = TestCase(input="Can I get a refund after 45 days?", name="refund policy")
    return TestResult(
        test_case=tc,
        paraphrases=["Is a refund possible after 45 days?"],
        outputs=["No refunds after 30 days.", "Refunds not available after 30 days."],
        consistency_score=consistency,
        contradiction_score=contradiction,
        risk=risk,
    )

def _make_report(consistency=0.90, contradiction=0.05):
    r = _make_result(consistency=consistency, contradiction=contradiction)
    return Report(results=[r], thresholds={"consistency": 0.75, "contradiction": 0.30})


# ─────────────────────────────────────────────────────────────
# TestCase
# ─────────────────────────────────────────────────────────────

class TestTestCase:
    def test_auto_name_short(self):
        tc = TestCase(input="What is the policy?")
        assert tc.name == "What is the policy?"

    def test_auto_name_truncated(self):
        tc = TestCase(input="A" * 100)
        assert tc.name.endswith("...")
        assert len(tc.name) <= 53

    def test_explicit_name_preserved(self):
        tc = TestCase(input="foo", name="my label")
        assert tc.name == "my label"


# ─────────────────────────────────────────────────────────────
# TestResult
# ─────────────────────────────────────────────────────────────

class TestTestResult:
    def test_passes_above_threshold(self):
        r = _make_result(consistency=0.90, contradiction=0.05)
        assert r.passed({"consistency": 0.75, "contradiction": 0.30}) is True

    def test_fails_low_consistency(self):
        r = _make_result(consistency=0.60, contradiction=0.05)
        assert r.passed({"consistency": 0.75, "contradiction": 0.30}) is False

    def test_fails_high_contradiction(self):
        r = _make_result(consistency=0.90, contradiction=0.50)
        assert r.passed({"consistency": 0.75, "contradiction": 0.30}) is False


# ─────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────

class TestReport:
    def test_avg_consistency(self):
        report = _make_report(consistency=0.85)
        assert abs(report.avg_consistency - 0.85) < 0.01

    def test_passed_failed_split(self):
        r_pass = _make_result(consistency=0.90, contradiction=0.05)
        r_fail = _make_result(consistency=0.50, contradiction=0.60, risk=RiskLevel.HIGH)
        report = Report(
            results=[r_pass, r_fail],
            thresholds={"consistency": 0.75, "contradiction": 0.30},
        )
        assert len(report.passed) == 1
        assert len(report.failed) == 1

    def test_empty_report(self):
        report = Report(results=[])
        assert report.avg_consistency is None
        assert report.avg_contradiction is None
        assert report.passed == []
        assert report.failed == []


# ─────────────────────────────────────────────────────────────
# LLMClient
# ─────────────────────────────────────────────────────────────

class TestLLMClient:
    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY",    raising=False)
        from contradish.llm import LLMClient
        with pytest.raises(EnvironmentError, match="API key"):
            LLMClient()

    def test_detects_anthropic_from_key_prefix(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY",    raising=False)
        from contradish.llm import LLMClient
        with patch("anthropic.Anthropic"):
            client = LLMClient(api_key="sk-ant-fakekey")
            assert client.provider == "anthropic"

    def test_detects_openai_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fakekey")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from contradish.llm import LLMClient
        with patch("openai.OpenAI"):
            client = LLMClient()
            assert client.provider == "openai"

    def test_prefers_anthropic_over_openai(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fakekey")
        monkeypatch.setenv("OPENAI_API_KEY",    "sk-fakekey")
        from contradish.llm import LLMClient
        with patch("anthropic.Anthropic"):
            client = LLMClient()
            assert client.provider == "anthropic"

    def test_json_parsing_strips_fences(self):
        from contradish.llm import LLMClient
        raw = '```json\n{"score": 0.9}\n```'
        result = LLMClient._parse_json(raw)
        assert result == {"score": 0.9}

    def test_json_parsing_plain(self):
        from contradish.llm import LLMClient
        raw = '{"score": 0.75, "ok": true}'
        result = LLMClient._parse_json(raw)
        assert result["score"] == 0.75

    def test_json_parsing_fallback(self):
        from contradish.llm import LLMClient
        raw = 'Some text {"key": "value"} trailing'
        result = LLMClient._parse_json(raw)
        assert result["key"] == "value"


# ─────────────────────────────────────────────────────────────
# Printer (smoke test — just ensure it doesn't crash)
# ─────────────────────────────────────────────────────────────

class TestPrinter:
    def test_print_report_no_crash(self, capsys):
        from contradish.printer import print_report
        report = _make_report()
        print_report(report)
        out = capsys.readouterr().out
        assert "contradish" in out.lower()

    def test_print_report_shows_test_name(self, capsys):
        from contradish.printer import print_report
        report = _make_report()
        print_report(report)
        out = capsys.readouterr().out
        assert "refund policy" in out

    def test_print_report_with_contradictions(self, capsys):
        from contradish.printer import print_report
        r = _make_result(consistency=0.55, contradiction=0.45, risk=RiskLevel.HIGH)
        r.contradictions = [
            ContradictionPair(
                input_a="Can I get a refund after 45 days?",
                input_b="Is a refund possible 45 days later?",
                output_a="No, 30 days only.",
                output_b="Yes, up to 60 days.",
                explanation="A says 30 days; B says 60 days",
                severity="factual",
            )
        ]
        report = Report(results=[r])
        print_report(report)
        out = capsys.readouterr().out
        assert "Contradiction" in out


# ─────────────────────────────────────────────────────────────
# Integration tests  (skipped unless API key present)
# ─────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestSuiteIntegration:
    """Run with:  pytest tests/ -m integration"""

    def test_stable_app_passes(self):
        import os
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            pytest.skip("No API key set")
        from contradish import Suite, TestCase
        suite = Suite(app=stable_app)
        suite.add(TestCase(name="refund policy", input="Can I get a refund after 45 days?"))
        report = suite.run(paraphrases=3, verbose=False)
        assert report.results[0].consistency_score is not None
        assert report.results[0].consistency_score > 0.70

    def test_unstable_app_flagged(self):
        import os
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            pytest.skip("No API key set")
        from contradish import Suite, TestCase
        suite = Suite(app=unstable_app())
        suite.add(TestCase(name="refund policy", input="Can I get a refund after 45 days?"))
        report = suite.run(paraphrases=5, verbose=False)
        result = report.results[0]
        # Unstable app should show contradictions or low consistency
        assert result.contradiction_score is not None
        assert result.contradiction_score > 0 or result.consistency_score < 0.85
