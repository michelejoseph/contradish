"""
Tests for Runner.run_matrix's handling of app(...) failures.

Before this fix, a failed app call was silently turned into the string
"[APP ERROR: ...]" and fed to the consistency judge as if it were a real
answer -- an outage or a bad API key would get scored as either "perfectly
consistent" (if every call failed the same way) or "wildly inconsistent"
(if some calls failed and some didn't), neither of which says anything
about the model under test.

These tests cover the fix at the Runner level (no LLM/API key needed --
run_matrix never calls the judge). See test_suite_skip_on_errors.py for
the Suite-level behavior (skip scoring entirely on an error-dominated case).
"""
import pytest
from contradish.llm import LLMClient
from contradish.runner import Runner


def _make_runner():
    # Runner only needs an LLMClient for generate_paraphrases, which these
    # tests don't exercise -- a throwaway client is fine.
    return Runner(LLMClient(api_key="sk-fake-not-a-real-key", provider="openai"))


class TestRunMatrixSuccess:
    def test_all_calls_succeed(self):
        runner = _make_runner()
        app = lambda q: f"answer to: {q}"
        inputs, outputs, errors = runner.run_matrix(app=app, original="q1", paraphrases=["q2", "q3"])
        assert inputs == ["q1", "q2", "q3"]
        assert outputs == ["answer to: q1", "answer to: q2", "answer to: q3"]
        assert errors == [None, None, None]


class TestRunMatrixFailures:
    def test_persistent_non_transient_error_fails_fast_no_retry(self):
        calls = []
        def app(q):
            calls.append(q)
            raise ValueError("invalid api key")
        runner = _make_runner()
        inputs, outputs, errors = runner.run_matrix(app=app, original="q1", paraphrases=[])
        # Not a transient-looking error -> exactly one attempt, no retry sleep.
        assert len(calls) == 1
        assert errors == ["invalid api key"]
        assert outputs == ["[APP ERROR: invalid api key]"]

    def test_transient_error_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("contradish.runner.time.sleep", lambda s: None)
        attempts = {"n": 0}
        def app(q):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("429 rate limit exceeded")
            return "ok"
        runner = _make_runner()
        inputs, outputs, errors = runner.run_matrix(app=app, original="q1", paraphrases=[])
        assert attempts["n"] == 3
        assert errors == [None]
        assert outputs == ["ok"]

    def test_transient_error_exhausts_retries_and_reports_last_error(self, monkeypatch):
        monkeypatch.setattr("contradish.runner.time.sleep", lambda s: None)
        attempts = {"n": 0}
        def app(q):
            attempts["n"] += 1
            raise RuntimeError(f"503 service unavailable (attempt {attempts['n']})")
        runner = _make_runner()
        inputs, outputs, errors = runner.run_matrix(app=app, original="q1", paraphrases=[], retries=3)
        assert attempts["n"] == 3
        assert errors == ["503 service unavailable (attempt 3)"]

    def test_mixed_success_and_failure_parallel_lists(self, monkeypatch):
        monkeypatch.setattr("contradish.runner.time.sleep", lambda s: None)
        def app(q):
            if q == "bad":
                raise ValueError("invalid model id")
            return f"ok:{q}"
        runner = _make_runner()
        inputs, outputs, errors = runner.run_matrix(app=app, original="good1", paraphrases=["bad", "good2"])
        assert inputs == ["good1", "bad", "good2"]
        assert outputs == ["ok:good1", "[APP ERROR: invalid model id]", "ok:good2"]
        assert errors == [None, "invalid model id", None]


class TestCallWithRetry:
    def test_non_transient_error_not_retried(self):
        calls = {"n": 0}
        def app(q):
            calls["n"] += 1
            raise ValueError("bad api key")
        result, err = Runner._call_with_retry(app, "q", retries=5)
        assert result is None
        assert err == "bad api key"
        assert calls["n"] == 1

    def test_success_returns_no_error(self):
        result, err = Runner._call_with_retry(lambda q: "fine", "q", retries=3)
        assert result == "fine"
        assert err is None
