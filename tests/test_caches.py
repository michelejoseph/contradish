"""
Tests for contradish.caches: the pluggable Firewall cache backends.
Run with: pytest tests/test_caches.py
No API key required.

RedisCache tests below use a small in-memory fake redis client (injected
via the `client=` constructor param, which the real class supports
specifically to make this kind of testing possible) rather than a real
Redis server or a mock library, so they exercise the real
pipeline/fallback/JSON-decode logic in caches.py itself.
"""
import json
import sys

import pytest

from contradish.caches import InMemoryCache, FirewallCache, RedisCache


# ── Fake redis client (list-backed, mimics the handful of commands used) ───

class _FakePipeline:
    def __init__(self, client):
        self.client = client
        self.ops = []

    def rpush(self, key, val):
        self.ops.append(("rpush", key, val))
        return self

    def ltrim(self, key, start, end):
        self.ops.append(("ltrim", key, start, end))
        return self

    def execute(self):
        for op in self.ops:
            if op[0] == "rpush":
                self.client.rpush(op[1], op[2])
            else:
                self.client.ltrim(op[1], op[2], op[3])
        self.ops = []


def _slice_redis_range(lst, start, end):
    n = len(lst)
    s = start if start >= 0 else max(n + start, 0)
    e = end if end >= 0 else n + end
    if n == 0:
        return []
    return lst[s:e + 1]


class _FakeRedisClient:
    """Supports .pipeline() -- exercises RedisCache's pipelined append path."""

    def __init__(self):
        self.store: dict = {}

    def rpush(self, key, val):
        self.store.setdefault(key, []).append(val)

    def ltrim(self, key, start, end):
        self.store[key] = _slice_redis_range(self.store.get(key, []), start, end)

    def lrange(self, key, start, end):
        return _slice_redis_range(self.store.get(key, []), start, end)

    def delete(self, key):
        self.store.pop(key, None)

    def llen(self, key):
        return len(self.store.get(key, []))

    def pipeline(self):
        return _FakePipeline(self)


class _FakeRedisClientNoPipeline:
    """Deliberately has no .pipeline attribute -- exercises the
    AttributeError fallback path in RedisCache.append()."""

    def __init__(self):
        self.store: dict = {}
        self.direct_calls = []

    def rpush(self, key, val):
        self.direct_calls.append(("rpush", key, val))
        self.store.setdefault(key, []).append(val)

    def ltrim(self, key, start, end):
        self.direct_calls.append(("ltrim", key, start, end))
        self.store[key] = _slice_redis_range(self.store.get(key, []), start, end)


def test_inmemory_basic_append_and_recent():
    c = InMemoryCache(window=5)
    c.append("q1", "a1")
    c.append("q2", "a2")
    assert c.size() == 2
    recent = c.recent(2)
    assert recent[-1] == {"query": "q2", "response": "a2"}
    assert recent[0] == {"query": "q1", "response": "a1"}


def test_inmemory_window_trim():
    c = InMemoryCache(window=3)
    for i in range(10):
        c.append(f"q{i}", f"a{i}")
    assert c.size() == 3
    # Only the most recent three survive
    recent = c.recent(10)
    assert [r["query"] for r in recent] == ["q7", "q8", "q9"]


def test_inmemory_recent_caps_at_size():
    c = InMemoryCache(window=50)
    c.append("only", "one")
    assert len(c.recent(15)) == 1


def test_inmemory_recent_zero_or_empty():
    c = InMemoryCache(window=5)
    assert c.recent(5) == []
    c.append("q", "a")
    assert c.recent(0) == []


def test_inmemory_clear():
    c = InMemoryCache(window=5)
    c.append("q", "a")
    c.clear()
    assert c.size() == 0


def test_inmemory_window_must_be_positive():
    try:
        InMemoryCache(window=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_protocol_conformance_of_custom_cache():
    class DictCache:
        def __init__(self):
            self.items = []
        def append(self, q, r):
            self.items.append({"query": q, "response": r})
        def recent(self, n):
            return self.items[-n:]
        def clear(self):
            self.items.clear()
        def size(self):
            return len(self.items)

    # runtime_checkable Protocol: a structurally-matching object passes isinstance
    assert isinstance(DictCache(), FirewallCache)


def test_firewall_uses_inmemory_by_default(monkeypatch=None):
    # Build a Firewall with a stubbed LLM so no API key / network is needed.
    import contradish.firewall as fw_mod

    class FakeLLM:
        def __init__(self, *a, **kw):
            pass
        fast_model = "fake"
        def complete_json(self, prompt, model=None):
            return {"contradiction": False}

    orig = fw_mod.LLMClient
    fw_mod.LLMClient = FakeLLM
    try:
        fw = fw_mod.Firewall(app=lambda q: "x", window=3)
        assert isinstance(fw.cache, InMemoryCache)
        assert fw.cache.window == 3
        fw.check("a?")
        fw.check("b?")
        assert fw.cache.size() == 2
        fw.reset()
        assert fw.cache.size() == 0
    finally:
        fw_mod.LLMClient = orig


# ── RedisCache: construction ────────────────────────────────────────────────

def test_redis_cache_rejects_non_positive_window():
    with pytest.raises(ValueError, match="window must be > 0"):
        RedisCache(window=0, client=_FakeRedisClient())


def test_redis_cache_uses_injected_client_without_importing_redis(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)  # even if "redis" is unimportable...
    cache = RedisCache(client=_FakeRedisClient(), window=5)  # ...client= bypasses the import
    cache.append("q", "r")
    assert cache.size() == 1


def test_redis_cache_raises_import_error_when_redis_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)
    with pytest.raises(ImportError, match="redis is not installed"):
        RedisCache(window=5)


def test_redis_cache_builds_client_via_redis_from_url(monkeypatch):
    captured = {}

    class _FakeRedisModule:
        @staticmethod
        def from_url(url, decode_responses=True):
            captured["url"] = url
            captured["decode_responses"] = decode_responses
            return _FakeRedisClient()

    monkeypatch.setitem(sys.modules, "redis", _FakeRedisModule)
    cache = RedisCache(url="redis://myhost:6379/2", window=5, decode_responses=False)
    assert captured["url"] == "redis://myhost:6379/2"
    assert captured["decode_responses"] is False
    assert cache.size() == 0


# ── RedisCache: append / recent / clear / size ──────────────────────────────

def test_redis_cache_append_uses_pipeline_and_trims_to_window():
    client = _FakeRedisClient()
    cache = RedisCache(client=client, window=2, key="k")
    cache.append("q1", "r1")
    cache.append("q2", "r2")
    cache.append("q3", "r3")
    assert cache.size() == 2
    assert cache.recent(10) == [
        {"query": "q2", "response": "r2"},
        {"query": "q3", "response": "r3"},
    ]


def test_redis_cache_append_falls_back_when_client_has_no_pipeline():
    client = _FakeRedisClientNoPipeline()
    cache = RedisCache(client=client, window=5, key="k")
    cache.append("q1", "r1")
    assert client.direct_calls[0][0] == "rpush"
    assert client.direct_calls[1][0] == "ltrim"
    assert client.store["k"] == [json.dumps({"query": "q1", "response": "r1"})]


def test_redis_cache_recent_zero_or_negative_returns_empty():
    client = _FakeRedisClient()
    cache = RedisCache(client=client, window=5, key="k")
    cache.append("q", "r")
    assert cache.recent(0) == []
    assert cache.recent(-3) == []


def test_redis_cache_recent_caps_at_window_even_if_n_is_larger():
    client = _FakeRedisClient()
    cache = RedisCache(client=client, window=2, key="k")
    cache.append("q1", "r1")
    cache.append("q2", "r2")
    assert cache.recent(100) == [
        {"query": "q1", "response": "r1"},
        {"query": "q2", "response": "r2"},
    ]


def test_redis_cache_recent_skips_corrupted_entries():
    client = _FakeRedisClient()
    cache = RedisCache(client=client, window=10, key="k")
    client.store["k"] = [
        json.dumps({"query": "good1", "response": "r1"}),
        "not valid json {{{",     # ValueError (json.JSONDecodeError) -> skipped
        None,                      # TypeError from json.loads(None) -> skipped
        json.dumps({"query": "good2", "response": "r2"}),
    ]
    out = cache.recent(10)
    assert out == [
        {"query": "good1", "response": "r1"},
        {"query": "good2", "response": "r2"},
    ]


def test_redis_cache_clear_deletes_the_key():
    client = _FakeRedisClient()
    cache = RedisCache(client=client, window=5, key="k")
    cache.append("q", "r")
    cache.clear()
    assert "k" not in client.store
    assert cache.size() == 0


def test_redis_cache_size_casts_to_int():
    class _StringLenClient(_FakeRedisClient):
        def llen(self, key):
            return str(len(self.store.get(key, "")))  # simulate a client returning a str

    client = _StringLenClient()
    cache = RedisCache(client=client, window=5, key="k")
    cache.append("q", "r")
    size = cache.size()
    assert size == 1
    assert isinstance(size, int)


def test_redis_cache_satisfies_firewall_cache_protocol():
    assert isinstance(RedisCache(client=_FakeRedisClient(), window=5), FirewallCache)


if __name__ == "__main__":
    import inspect

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = skipped = 0
    for fn in fns:
        # This manual runner predates pytest fixtures in this file; skip any
        # test that now takes a fixture (e.g. monkeypatch) since it can't
        # supply one -- those still run fine under `pytest tests/`.
        if len(inspect.signature(fn).parameters) > 0:
            skipped += 1
            continue
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed ({skipped} fixture-based tests skipped -- run under pytest for full coverage)")
