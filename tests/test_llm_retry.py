"""
Tests for LLMClient retry/backoff on transient failures.
Run with: pytest tests/test_llm_retry.py
No API key required: we construct LLMClient via __new__ to skip provider setup
and drive _call_with_retry directly with a stubbed sleep.
"""
from contradish.llm import LLMClient, _is_transient


def _bare_client():
    # Skip __init__ (which would resolve a provider/key); we only exercise the
    # retry helper, which has no provider dependency.
    c = LLMClient.__new__(LLMClient)
    c._sleep = lambda d: None  # never actually sleep in tests
    return c


def test_is_transient_status_code():
    class E429(Exception):
        status_code = 429
    class E503(Exception):
        status = 503
    class E400(Exception):
        status_code = 400
    assert _is_transient(E429())
    assert _is_transient(E503())
    assert not _is_transient(E400())


def test_is_transient_by_name_and_message():
    assert _is_transient(Exception("Rate limit exceeded, try again"))
    assert _is_transient(type("APITimeoutError", (Exception,), {})())
    assert _is_transient(type("APIConnectionError", (Exception,), {})())
    assert _is_transient(Exception("503 Service Unavailable"))
    # auth / validation are not transient
    assert not _is_transient(Exception("invalid api key"))
    assert not _is_transient(Exception("400 bad request: malformed"))


def test_retry_then_succeed():
    c = _bare_client()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("rate limit (429)")
        return "ok"

    assert c._call_with_retry(flaky) == "ok"
    assert calls["n"] == 3   # failed twice, succeeded on the third


def test_fail_fast_on_non_transient():
    c = _bare_client()
    calls = {"n": 0}

    def auth_error():
        calls["n"] += 1
        raise Exception("invalid api key")

    try:
        c._call_with_retry(auth_error)
        assert False, "expected the auth error to propagate"
    except Exception as e:
        assert "invalid api key" in str(e)
    assert calls["n"] == 1   # never retried


def test_gives_up_after_max_retries():
    c = _bare_client()
    calls = {"n": 0}

    def always_429():
        calls["n"] += 1
        raise Exception("429 too many requests")

    try:
        c._call_with_retry(always_429)
        assert False, "expected to give up and re-raise"
    except Exception:
        pass
    # 1 initial attempt + _MAX_RETRIES retries
    from contradish.llm import _MAX_RETRIES
    assert calls["n"] == 1 + _MAX_RETRIES


def test_success_first_try_no_sleep():
    c = _bare_client()
    slept = {"n": 0}
    c._sleep = lambda d: slept.__setitem__("n", slept["n"] + 1)
    assert c._call_with_retry(lambda: "fine") == "fine"
    assert slept["n"] == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
