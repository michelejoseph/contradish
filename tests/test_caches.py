"""
Tests for contradish.caches: the pluggable Firewall cache backends.
Run with: pytest tests/test_caches.py
No API key required.
"""
from contradish.caches import InMemoryCache, FirewallCache


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


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
