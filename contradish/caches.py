"""
Firewall cache backends.

The production Firewall keeps a rolling window of recent (query, response)
pairs and asks an LLM judge whether each new response contradicts any of
them. In a single-process toy that lives fine in a Python list. In a real
deployment with multiple workers — gunicorn, uvicorn workers, ECS tasks,
Lambda invocations — each process gets its own list and contradictions
across workers are invisible. That defeats the entire point of the
Firewall, which is to catch the contradiction before it reaches the user.

This module abstracts the storage. Default is `InMemoryCache` (matches the
historical behavior, no new deps). `RedisCache` ships in the box for any
shared-state deployment. The interface is small enough that a developer
can plug in their own (Memcached, DynamoDB, a Postgres table, etc.) in
fewer than fifty lines.

    from contradish import Firewall
    from contradish.caches import RedisCache

    firewall = Firewall(
        app   = my_llm_app,
        cache = RedisCache(url="redis://prod-cache:6379/0", window=200),
    )
"""

from __future__ import annotations

import json as _json
from typing import Protocol, runtime_checkable


@runtime_checkable
class FirewallCache(Protocol):
    """
    The contract every Firewall cache must satisfy.

    Methods:
        append(query, response)  store one interaction
        recent(n)                return up to n most-recent items, oldest-first,
                                 as a list of {"query": str, "response": str}
        clear()                  drop everything
        size()                   current item count
    """
    def append(self, query: str, response: str) -> None: ...
    def recent(self, n: int) -> list[dict]: ...
    def clear(self) -> None: ...
    def size(self) -> int: ...


class InMemoryCache:
    """
    Per-process rolling list. Matches the original Firewall behavior exactly.

    Good for: single-worker apps, testing, demos.
    Not good for: any deployment with more than one worker — cross-worker
    contradictions will be silently missed.
    """

    def __init__(self, window: int = 50):
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        self.window = int(window)
        self._items: list[dict] = []

    def append(self, query: str, response: str) -> None:
        self._items.append({"query": query, "response": response})
        if len(self._items) > self.window:
            # Drop oldest to maintain the window
            del self._items[0:len(self._items) - self.window]

    def recent(self, n: int) -> list[dict]:
        if n <= 0 or not self._items:
            return []
        return list(self._items[-min(n, len(self._items)):])

    def clear(self) -> None:
        self._items.clear()

    def size(self) -> int:
        return len(self._items)


class RedisCache:
    """
    Shared rolling window backed by a Redis list. Survives worker restarts.
    Multiple workers, multiple processes, multiple hosts can all share the
    same Firewall state — which is the only configuration in which
    production contradiction detection actually works.

    Args:
        url:        Redis connection URL. Default `redis://localhost:6379/0`.
        key:        Redis key under which the cache list is stored. Use a
                    different key per Firewall instance if you want isolated
                    state per app / per environment.
        window:     Maximum number of entries to retain. The list is trimmed
                    to this length on every append.
        client:     Optional pre-built redis.Redis instance. Useful when you
                    want to inject a connection pool, TLS config, or a
                    fakeredis instance for tests. When provided, `url` is
                    ignored.
        decode_responses: Passed to redis.from_url when building a client.
                    Default True so we get strings back, not bytes.

    Raises:
        ImportError if the `redis` package isn't installed. Install with:
            pip install "contradish[redis]"

    Example:
        from contradish import Firewall
        from contradish.caches import RedisCache

        fw = Firewall(
            app   = my_app,
            mode  = "monitor",
            cache = RedisCache(
                url    = "redis://cache.internal:6379/0",
                key    = "support-bot:firewall",
                window = 200,
            ),
        )
    """

    def __init__(
        self,
        url:              str  = "redis://localhost:6379/0",
        key:              str  = "contradish:firewall",
        window:           int  = 50,
        client:           object = None,
        decode_responses: bool = True,
    ):
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        self.window = int(window)
        self.key    = key

        if client is not None:
            self._r = client
        else:
            try:
                import redis  # noqa
            except ImportError as e:
                raise ImportError(
                    "redis is not installed. Install with:\n"
                    "    pip install \"contradish[redis]\""
                ) from e
            self._r = redis.from_url(url, decode_responses=decode_responses)

    def append(self, query: str, response: str) -> None:
        payload = _json.dumps({"query": query, "response": response})
        # Pipeline so the trim happens atomically with the push.
        try:
            pipe = self._r.pipeline()
            pipe.rpush(self.key, payload)
            pipe.ltrim(self.key, -self.window, -1)
            pipe.execute()
        except AttributeError:
            # Fallback for clients without pipeline support (rare).
            self._r.rpush(self.key, payload)
            self._r.ltrim(self.key, -self.window, -1)

    def recent(self, n: int) -> list[dict]:
        if n <= 0:
            return []
        take = min(n, self.window)
        raw  = self._r.lrange(self.key, -take, -1)
        out  = []
        for item in raw:
            try:
                out.append(_json.loads(item))
            except (TypeError, ValueError):
                # Corrupted entry — skip. Don't let one bad cache row poison
                # the whole contradiction check.
                continue
        return out

    def clear(self) -> None:
        self._r.delete(self.key)

    def size(self) -> int:
        return int(self._r.llen(self.key))


__all__ = ["FirewallCache", "InMemoryCache", "RedisCache"]
