"""
Conversation memory: commitment-level, relevance-retrieved contradiction
detection for multi-turn LLM apps.

The production Firewall used to keep a rolling window of the last N raw
(query, response) pairs and ask a judge "does the new answer contradict any
of the most recent ones?". That has two structural limits:

  1. Recency, not relevance. The contradiction that matters is usually with a
     commitment made arbitrarily far back ("refund window is 30 days" at
     turn 3, contradicted at turn 60). By then that turn has fallen out of
     the window or is buried among unrelated pairs. The right retrieval key
     is topic relevance, not recency.

  2. Raw prose, not commitments. Re-reading whole answers every call is noisy,
     can't be retrieved over, and gives a repair step nothing concrete to
     anchor to.

This module fixes both. Each response is distilled into atomic *commitments*
(normalized, checkable assertions with a topic key and provenance). Commitments
are stored per session. For a new response we retrieve only the *relevant*
prior commitments, judge entailment claim-vs-claim, and on a contradiction can
*repair* the new answer so it honors the established commitment and cites it.

    from contradish.memory import ConversationMemory

    mem = ConversationMemory()                       # uses env API key
    mem.ingest("user-42", "Refund window?", "30 days, no exceptions.")
    finding = mem.check("user-42", "Refunds after 45 days?",
                        "Sure, we can refund at 45 days.")
    if finding.contradiction:
        fixed = mem.repair("Refunds after 45 days?",
                           "Sure, we can refund at 45 days.", finding)

Relevance retrieval is dependency-free (lexical topic/claim overlap) by
default. Pass `relevance_fn=` to plug in an embedding scorer for higher recall;
the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json as _json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Protocol, runtime_checkable

# Optional at construction time; only needed for extract/detect/repair. Imported
# lazily so the data structures and the store are usable with no provider/key.
from .llm import LLMClient


# ── Commitment ──────────────────────────────────────────────────────────────

@dataclass
class Commitment:
    """
    One durable, checkable assertion an app made in a reply.

    Fields:
        claim:           normalized standalone statement ("Refund window is 30 days").
        topic:           short retrieval key ("refund window"), 2-4 lowercase words.
        session:         scope key (user id / conversation id) this belongs to.
        source_query:    the user query that produced the reply (provenance).
        source_response: the full reply text the claim was extracted from.
        turn:            monotonic index within the session (0-based).
        created_at:      unix timestamp of ingestion.
        embedding:       cached embedding vector for the claim (set by an
                         embedding-based relevance scorer at ingest time and
                         persisted with the commitment, so it is computed once
                         globally rather than re-embedded per worker). None for
                         lexical relevance.
        origin:          which layer produced this commitment: "prompt" (a
                         clause the system prompt promises), "benchmark" (a
                         commitment a test case stresses), or "response" (one
                         extracted from a model reply at runtime). The shared
                         field that lets the same unit flow across every layer
                         and be reconciled (see contradish.reconcile).
    """
    claim:           str
    topic:           str = ""
    session:         str = "default"
    source_query:    str = ""
    source_response: str = ""
    turn:            int = 0
    created_at:      float = field(default_factory=time.time)
    embedding:       Optional[list] = None
    origin:          str = "response"
    # "durable" = a policy, rule, definition, or stable fact that should not
    # change (a refund window, a dosage ceiling). "volatile" = state that
    # legitimately changes over time (a balance, an order status, a count, a
    # date, a scheduled time). A change to a durable commitment is a
    # contradiction; a change to a volatile one is a normal state update. This
    # is what makes the memory layer safe for stateful agents.
    kind:            str = "durable"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Commitment":
        emb = d.get("embedding")
        return cls(
            claim=str(d.get("claim", "")),
            topic=str(d.get("topic", "")),
            session=str(d.get("session", "default")),
            source_query=str(d.get("source_query", "")),
            source_response=str(d.get("source_response", "")),
            turn=int(d.get("turn", 0)),
            created_at=float(d.get("created_at", time.time())),
            embedding=list(emb) if isinstance(emb, (list, tuple)) else None,
            origin=str(d.get("origin", "response")),
            kind=str(d.get("kind", "durable") or "durable"),
        )


@dataclass
class ContradictionFinding:
    """Result of checking new commitments against the relevant prior ones."""
    contradiction: bool = False
    new_claim:     Optional[str] = None
    prior_claim:   Optional[str] = None
    prior_query:   Optional[str] = None
    prior_response: Optional[str] = None
    explanation:   Optional[str] = None
    confidence:    Optional[float] = None


# ── Stores ────────────────────────────────────────────────────────────────

@runtime_checkable
class CommitmentStore(Protocol):
    """
    Storage contract for commitments. Scoped by session so one user's history
    never pollutes another's contradiction check.

    Methods:
        add(commitment)            store one commitment under its session
        by_session(session)        all commitments for a session, oldest-first
        clear(session=None)        drop one session, or everything if None
        size(session=None)         count for a session, or total if None
    """
    def add(self, commitment: Commitment) -> None: ...
    def by_session(self, session: str) -> list: ...
    def clear(self, session: Optional[str] = None) -> None: ...
    def size(self, session: Optional[str] = None) -> int: ...


class InMemoryCommitmentStore:
    """
    Per-process, session-keyed commitment store.

    Good for single-worker apps, testing, demos. For multi-worker production
    (gunicorn/uvicorn workers, ECS tasks) use RedisCommitmentStore so workers
    share one history.

    Args:
        per_session: optional cap on commitments retained per session. None
                     means unbounded.
        eviction: which commitment to drop when a session is over its cap.
                  "redundancy" (default) drops the most redundant commitment, so
                  a near-duplicate restatement goes before a unique fact, and
                  falls back to oldest-first when nothing is redundant. "fifo"
                  always drops the oldest, which is cheaper but can evict a
                  load-bearing early commitment.
    """

    def __init__(self, per_session: Optional[int] = None, eviction: str = "redundancy"):
        if per_session is not None and per_session <= 0:
            raise ValueError(f"per_session must be > 0 or None, got {per_session}")
        if eviction not in ("redundancy", "fifo"):
            raise ValueError(f"eviction must be 'redundancy' or 'fifo', got {eviction!r}")
        self.per_session = per_session
        self.eviction = eviction
        self._by_session: dict[str, list[Commitment]] = {}

    def add(self, commitment: Commitment) -> None:
        bucket = self._by_session.setdefault(commitment.session, [])
        bucket.append(commitment)
        if self.per_session is None:
            return
        if self.eviction == "fifo":
            if len(bucket) > self.per_session:
                del bucket[0:len(bucket) - self.per_session]
            return
        while len(bucket) > self.per_session:
            del bucket[_redundancy_evict_index(bucket)]

    def by_session(self, session: str) -> list:
        return list(self._by_session.get(session, []))

    def clear(self, session: Optional[str] = None) -> None:
        if session is None:
            self._by_session.clear()
        else:
            self._by_session.pop(session, None)

    def size(self, session: Optional[str] = None) -> int:
        if session is None:
            return sum(len(v) for v in self._by_session.values())
        return len(self._by_session.get(session, []))

    def next_turn(self, session: str) -> int:
        return len(self._by_session.get(session, []))


class RedisCommitmentStore:
    """
    Shared, session-keyed commitment store backed by Redis lists. Survives
    worker restarts and lets every worker see the same conversation history,
    which is the only configuration in which cross-worker contradiction
    detection actually works.

    One Redis list per session under `{key}:{session}`. A companion set
    `{key}:__sessions__` tracks known sessions so clear()/size() with no
    argument can operate across all of them without SCAN.

    Eviction here is oldest-first (atomic LTRIM), not the redundancy-aware
    policy InMemoryCommitmentStore uses, because smart eviction would need a
    read-modify-write that is not safe across concurrent workers. The primary
    way to keep a shared store small is dedup at ingest (ConversationMemory),
    which applies equally to both stores.

    Args:
        url:      Redis connection URL. Default redis://localhost:6379/0.
        key:      Base key namespace. Default "contradish:memory".
        per_session: optional cap retained per session (list trimmed on add).
        client:   Optional pre-built redis client (inject a pool, TLS config,
                  or a fake for tests). When provided, `url` is ignored.
        decode_responses: passed to redis.from_url. Default True (strings back).

    Raises:
        ImportError if `redis` is not installed: pip install "contradish[redis]"
    """

    def __init__(
        self,
        url:              str = "redis://localhost:6379/0",
        key:              str = "contradish:memory",
        per_session:      Optional[int] = None,
        client:           object = None,
        decode_responses: bool = True,
    ):
        if per_session is not None and per_session <= 0:
            raise ValueError(f"per_session must be > 0 or None, got {per_session}")
        self.per_session = per_session
        self.key = key

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

    def _skey(self, session: str) -> str:
        return f"{self.key}:{session}"

    def _index(self) -> str:
        return f"{self.key}:__sessions__"

    def add(self, commitment: Commitment) -> None:
        payload = _json.dumps(commitment.to_dict())
        skey = self._skey(commitment.session)
        try:
            pipe = self._r.pipeline()
            pipe.rpush(skey, payload)
            if self.per_session is not None:
                pipe.ltrim(skey, -self.per_session, -1)
            pipe.sadd(self._index(), commitment.session)
            pipe.execute()
        except AttributeError:
            self._r.rpush(skey, payload)
            if self.per_session is not None:
                self._r.ltrim(skey, -self.per_session, -1)
            self._r.sadd(self._index(), commitment.session)

    def by_session(self, session: str) -> list:
        raw = self._r.lrange(self._skey(session), 0, -1)
        out: list[Commitment] = []
        for item in raw:
            try:
                out.append(Commitment.from_dict(_json.loads(item)))
            except (TypeError, ValueError):
                continue
        return out

    def _sessions(self) -> list:
        members = self._r.smembers(self._index()) or []
        return [m.decode() if isinstance(m, bytes) else m for m in members]

    def clear(self, session: Optional[str] = None) -> None:
        if session is None:
            for s in self._sessions():
                self._r.delete(self._skey(s))
            self._r.delete(self._index())
        else:
            self._r.delete(self._skey(session))
            self._r.srem(self._index(), session)

    def size(self, session: Optional[str] = None) -> int:
        if session is None:
            return sum(int(self._r.llen(self._skey(s))) for s in self._sessions())
        return int(self._r.llen(self._skey(session)))

    def next_turn(self, session: str) -> int:
        return int(self._r.llen(self._skey(session)))


# ── Relevance (dependency-free default) ───────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "but", "if", "then", "this",
    "that", "these", "those", "it", "its", "as", "at", "by", "with", "from",
    "you", "your", "we", "our", "i", "they", "them", "can", "will", "may",
    "not", "no", "do", "does", "did", "has", "have", "had", "about",
}


def topic_of(text: str, n: int = 4) -> str:
    """
    Derive a short topic key from free text: the first `n` significant words
    (stopwords dropped) in order. Used to give commitments a retrieval/matching
    key when one was not supplied. Shared by the prompt analyzer and the
    reconciler so every layer derives topics the same way.
    """
    out = []
    word = []
    for ch in (text or "").lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                w = "".join(word)
                if w not in _STOPWORDS and len(w) > 1:
                    out.append(w)
                word = []
            if len(out) >= n:
                break
    if word and len(out) < n:
        w = "".join(word)
        if w not in _STOPWORDS and len(w) > 1:
            out.append(w)
    return " ".join(out[:n])


def _tokens(s: str) -> set:
    out = set()
    word = []
    for ch in (s or "").lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                out.add("".join(word))
                word = []
    if word:
        out.add("".join(word))
    return {t for t in out if t not in _STOPWORDS and len(t) > 1}


def _overlap_score(a: Commitment, b: Commitment) -> float:
    """
    Lexical relevance in [0, 1]. Topic tokens are weighted double because a
    shared topic is a much stronger signal of "same matter" than a shared
    content word. Uses the overlap coefficient (intersection over the smaller
    set) so a short claim still matches a longer related one.
    """
    a_topic, b_topic = _tokens(a.topic), _tokens(b.topic)
    a_all = _tokens(a.claim) | a_topic
    b_all = _tokens(b.claim) | b_topic
    if not a_all or not b_all:
        return 0.0
    inter = a_all & b_all
    if not inter:
        return 0.0
    base = len(inter) / min(len(a_all), len(b_all))
    topic_bonus = 0.25 if (a_topic & b_topic) else 0.0
    return min(1.0, base + topic_bonus)


def _redundancy_evict_index(commitments: list) -> int:
    """
    Index of the commitment to drop when a session is over its cap. Picks the
    most redundant commitment (the one with the highest lexical overlap to any
    other in the list), so a near-duplicate restatement is dropped before a
    unique, load-bearing fact. Ties resolve to the lowest index (the oldest),
    which makes the policy degrade to plain oldest-first when nothing in the
    session is redundant.
    """
    n = len(commitments)
    if n <= 1:
        return 0
    best_i = 0
    best_score = -1.0
    for i in range(n):
        ci = commitments[i]
        nn = 0.0
        for j in range(n):
            if i == j:
                continue
            s = _overlap_score(ci, commitments[j])
            if s > nn:
                nn = s
        if nn > best_score:
            best_score = nn
            best_i = i
    return best_i


# Signature for a pluggable scorer: (new_commitment, prior_commitment) -> [0,1]
RelevanceFn = Callable[[Commitment, Commitment], float]


# ── Semantic relevance (embeddings) ───────────────────────────────────────

def _cosine(u: list, v: list) -> float:
    import math
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


# A batch embedder: many texts in, one vector per text out, in input order.
EmbedFn = Callable[[list], list]


class EmbeddingRelevance:
    """
    Semantic relevance scorer for ConversationMemory.

    Lexical overlap (the default) misses paraphrased topics: "refund window"
    and "return timeframe" share no tokens but mean the same thing, so the
    contradiction between them is never even retrieved for the judge to see.
    This scorer embeds each commitment and scores pairs by cosine similarity,
    closing that recall gap.

    It is a drop-in `relevance_fn`: pass it to ConversationMemory(relevance_fn=...)
    or use the ConversationMemory.with_embeddings(...) factory, which also sets a
    sensible threshold.

    Args:
        embed_fn: a batch embedder, Callable[[list[str]], list[list[float]]].
                  Use openai_embedder(), or bring your own (Voyage, Cohere, a
                  local sentence-transformer, etc.). Only the call shape matters.
        cache:    when True (default) memoize embeddings by text, so each prior
                  commitment is embedded once total rather than once per turn.
    """

    def __init__(self, embed_fn: EmbedFn, cache: bool = True):
        self.embed_fn = embed_fn
        self._cache: Optional[dict] = {} if cache else None

    @staticmethod
    def _text(c: Commitment) -> str:
        topic = (c.topic or "").strip()
        claim = (c.claim or "").strip()
        combined = f"{topic}. {claim}".strip(". ").strip()
        return combined or claim

    def embed_many(self, texts: list) -> list:
        """Embed a list of texts, using and filling the cache. Batches misses."""
        if self._cache is None:
            return list(self.embed_fn(list(texts)))
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self.embed_fn(missing)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return [self._cache[t] for t in texts]

    def _vec(self, text: str) -> list:
        return self.embed_many([text])[0]

    def _vector(self, c: Commitment) -> list:
        """A commitment's embedding: its persisted vector if present, else embed
        its text (memoized). Persisted vectors come from the shared store, so a
        prior is never re-embedded once any worker has embedded it."""
        stored = getattr(c, "embedding", None)
        if isinstance(stored, (list, tuple)) and stored:
            return list(stored)
        return self._vec(self._text(c))

    def precompute(self, commitments: list) -> None:
        """Optional: warm the cache for a batch of commitments in one call."""
        self.embed_many([self._text(c) for c in commitments])

    def attach(self, commitments: list) -> None:
        """
        Fill and persist .embedding on any commitment that lacks one, batching
        the embed call. Called by ConversationMemory at ingest so the vector is
        stored alongside the commitment and shared across workers. Idempotent:
        commitments that already carry an embedding are skipped.
        """
        todo = [c for c in commitments if not isinstance(getattr(c, "embedding", None), (list, tuple)) or not getattr(c, "embedding")]
        if not todo:
            return
        vecs = self.embed_many([self._text(c) for c in todo])
        for c, v in zip(todo, vecs):
            c.embedding = list(v)

    def __call__(self, a: Commitment, b: Commitment) -> float:
        va = self._vector(a)
        vb = self._vector(b)
        return max(0.0, min(1.0, _cosine(va, vb)))


def openai_embedder(
    model:      str = "text-embedding-3-small",
    api_key:    Optional[str] = None,
    client:     object = None,
    dimensions: Optional[int] = None,
) -> EmbedFn:
    """
    Return a batch embed function backed by OpenAI embeddings, suitable for
    EmbeddingRelevance / ConversationMemory.with_embeddings.

    Args:
        model:      embedding model. Default "text-embedding-3-small".
        api_key:    OpenAI key. Falls back to OPENAI_API_KEY if omitted.
        client:     pre-built OpenAI client (inject for tests / custom config).
                    When provided, api_key is ignored.
        dimensions: optional output dimensionality (text-embedding-3-* support
                    truncation for cheaper storage).

    Returns:
        Callable[[list[str]], list[list[float]]] preserving input order.

    Raises:
        ImportError if the OpenAI SDK is not installed.
    """
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The OpenAI SDK is required for openai_embedder. Install with:\n"
                "    pip install \"contradish[openai]\""
            ) from e
        import os
        key = (api_key or os.environ.get("OPENAI_API_KEY", "").strip()) or None
        client = OpenAI(api_key=key) if key else OpenAI()

    def _embed(texts: list) -> list:
        kwargs = {"model": model, "input": list(texts)}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        resp = client.embeddings.create(**kwargs)
        # resp.data is returned in the same order as the input.
        return [d.embedding for d in resp.data]

    return _embed


# ── Prompts ───────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You extract the durable, checkable commitments an assistant made in a reply, so they can be checked for consistency later in the conversation.

User asked: {query}
Assistant replied: {response}

A commitment is a specific assertion the reply commits to: a policy, number, rule, eligibility, recommendation, or factual claim that could later be contradicted. Ignore pleasantries, hedges, clarifying questions, and pure restatement of the user's words. Normalize each into a short standalone statement that makes sense on its own.

For each commitment give a short topic key of 2 to 4 lowercase words, used to find related statements later (for example "refund window", "melatonin long term use").

Also tag each commitment's kind. Use "durable" if it is a policy, rule, definition, or stable fact that should not change during the conversation (a refund window, a dosage ceiling, an eligibility rule). Use "volatile" if it is a value that legitimately changes over time (a current balance, an order status, a count, today's date, a scheduled time). A later change to a durable commitment is a contradiction; a change to a volatile one is just an update.

Return JSON only, no markdown, as a list:
[{{"claim": "<normalized standalone assertion>", "topic": "<short topic key>", "kind": "durable" or "volatile"}}]

If the reply makes no checkable commitment, return []."""

_DETECT_PROMPT = """You decide whether a new assistant statement contradicts something the same assistant already committed to earlier in this conversation.

New statement(s):
{new_claims}

The assistant's earlier commitments on related topics, oldest first:
{prior_claims}

A contradiction means the assistant now asserts something that cannot both be true with an earlier assertion: a different policy, number, rule, eligibility, or conclusion on the same matter. Ignore differences in wording, added detail, or tone. A more specific or more cautious restatement is not a contradiction. Check the new statement against the earliest established commitment above, not only the most recent one: a value that has crept across several turns still contradicts the original position.

Return JSON only, no markdown, as an object with a "contradictions" list, one entry per earlier commitment the new statement contradicts (usually empty or one entry; include several only when the new statement conflicts with multiple distinct earlier commitments):
{{"contradictions": [{{"new_claim": "<the new claim>", "prior_claim": "<the exact earlier claim it contradicts>", "explanation": "<one sentence>", "confidence": <number 0.0 to 1.0>}}]}}
If there is no contradiction, return {{"contradictions": []}}."""

_REPAIR_PROMPT = """The assistant is about to send a reply that contradicts something it already told this same user earlier in the conversation. Rewrite the reply so it is consistent with the earlier commitment, keeps whatever is still helpful, and briefly acknowledges the established fact rather than silently reversing it.

User asked: {query}
Draft reply (contradictory): {response}
Earlier commitment that must be honored: {prior_claim}

Return only the rewritten reply text. No preamble, no explanation, no JSON."""


# ── ConversationMemory ────────────────────────────────────────────────────

_MAX_COMMITMENTS_PER_REPLY = 8


class ConversationMemory:
    """
    Commitment-level memory with relevance retrieval, contradiction detection,
    and repair. Backs the memory-aware Firewall, but is usable directly.

    Args:
        llm:        an LLMClient. Built from env / args if omitted.
        store:      a CommitmentStore. Defaults to InMemoryCommitmentStore().
        api_key:    forwarded to LLMClient when `llm` is omitted.
        provider:   forwarded to LLMClient when `llm` is omitted.
        model:      model used for extract/detect/repair. Defaults to the
                    client's fast_model (these are cheap, high-volume calls).
        relevance_fn:   scorer (new, prior) -> [0,1]. Default lexical overlap.
        relevance_threshold: minimum score to treat a prior as relevant.
        top_k:      max prior commitments retrieved per check.
        dedup:      when True (default), an identical restatement is not stored
                    twice in a session. Only exact normalized-claim matches are
                    dropped, so a conflicting claim is still stored and surfaced
                    by the check path rather than merged away.
        flag_state_changes: when False (default), a changed value on a volatile
                    commitment (a balance, a status, a count) is treated as a
                    normal state update and not reported as a contradiction, so
                    a stateful agent that legitimately mutates state is not
                    flagged on every change. Set True to treat volatile changes
                    as contradictions too.

    For a persistent agent, pass a stable id as the `session` so commitments
    carry across conversations and a contradiction made weeks apart is still
    caught.
    """

    def __init__(
        self,
        llm:                 Optional[LLMClient] = None,
        store:               Optional[CommitmentStore] = None,
        api_key:             Optional[str] = None,
        provider:            Optional[str] = None,
        model:               Optional[str] = None,
        relevance_fn:        Optional[RelevanceFn] = None,
        relevance_threshold: float = 0.3,
        top_k:               int = 5,
        dedup:               bool = True,
        flag_state_changes:  bool = False,
    ):
        self._llm_arg     = llm
        self._api_key     = api_key
        self._provider    = provider
        self.model        = model
        self.store        = store if store is not None else InMemoryCommitmentStore()
        self.relevance_fn = relevance_fn or _overlap_score
        self.relevance_threshold = relevance_threshold
        self.top_k        = top_k
        self.dedup        = dedup
        self.flag_state_changes = flag_state_changes
        self._llm_cached  = llm  # may be None; built lazily on first LLM use

    @classmethod
    def with_embeddings(
        cls,
        embed_fn:  EmbedFn,
        threshold: float = 0.55,
        cache:     bool = True,
        **kwargs,
    ) -> "ConversationMemory":
        """
        Build a ConversationMemory whose relevance step uses semantic embeddings
        instead of lexical overlap, so paraphrased topics ("refund window" vs
        "return timeframe") are still retrieved for the contradiction judge.

        Args:
            embed_fn:  batch embedder (see openai_embedder, or bring your own).
            threshold: minimum cosine to treat a prior as relevant. 0.55 is a
                       reasonable cut for normalized text-embedding cosine; tune
                       for your embedder. Note this is a different scale than the
                       lexical default (0.3), so it is set for you here.
            cache:     memoize embeddings by text (default True).
            **kwargs:  forwarded to __init__ (llm, store, api_key, top_k, ...).

        Example:
            from contradish import ConversationMemory, openai_embedder
            mem = ConversationMemory.with_embeddings(openai_embedder())
        """
        scorer = EmbeddingRelevance(embed_fn, cache=cache)
        return cls(relevance_fn=scorer, relevance_threshold=threshold, **kwargs)

    # -- lazy LLM so the store/relevance are usable with no key --
    @property
    def llm(self) -> LLMClient:
        if self._llm_cached is None:
            self._llm_cached = LLMClient(api_key=self._api_key, provider=self._provider)
        return self._llm_cached

    def _model(self) -> str:
        return self.model or self.llm.fast_model

    # -- extraction --
    def extract(self, query: str, response: str, session: str = "default") -> list:
        """Distill a reply into a list of Commitment objects (one LLM call)."""
        prompt = _EXTRACT_PROMPT.format(query=query, response=response)
        try:
            raw = self.llm.complete(prompt, model=self._model())
            data = self._parse_list(raw)
        except Exception:
            return []
        turn = self._next_turn(session)
        out: list[Commitment] = []
        for item in data[:_MAX_COMMITMENTS_PER_REPLY]:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            out.append(Commitment(
                claim=claim,
                topic=str(item.get("topic", "")).strip().lower(),
                session=session,
                source_query=query,
                source_response=response,
                turn=turn,
                kind=("volatile" if str(item.get("kind", "")).strip().lower() == "volatile" else "durable"),
            ))
        return out

    def _next_turn(self, session: str) -> int:
        nt = getattr(self.store, "next_turn", None)
        if callable(nt):
            return nt(session)
        return self.store.size(session)

    # -- retrieval --
    def relevant(self, session: str, commitments: list) -> list:
        """
        Return prior commitments in `session` relevant to any of `commitments`,
        capped at top_k. No LLM call.

        Selection is anchor-aware. For each relevant topic the earliest-turn
        commitment (the model's original, established position) is retrieved
        first, then the remaining slots are filled by relevance score. This is
        what lets multi-turn drift surface: under accumulating pressure a model
        tends to move one small step per turn, so the newest reply agrees with
        the most recent (already-drifted) restatement and only conflicts with the
        position it started from. Ranking purely by lexical score favors the
        closer recent restatements and can push that original commitment out of
        the top_k window, hiding the drift. Anchoring keeps it in, so the new
        claim is judged against where the model began, not only where it drifted
        to.
        """
        priors = self.store.by_session(session)
        if not priors or not commitments:
            return []
        scored: dict[int, tuple] = {}
        for prior in priors:
            best = 0.0
            for new in commitments:
                s = self.relevance_fn(new, prior)
                if s > best:
                    best = s
            if best >= self.relevance_threshold:
                scored[id(prior)] = (best, prior)
        if not scored:
            return []
        ranked = [p for _, p in sorted(scored.values(), key=lambda t: t[0], reverse=True)]
        # Anchor = earliest-turn commitment per topic (the original position).
        anchors: dict = {}
        for _, p in scored.values():
            topic = (p.topic or "").strip().lower()
            if not topic:
                continue
            if topic not in anchors or p.turn < anchors[topic].turn:
                anchors[topic] = p
        result: list = []
        seen: set = set()
        for p in sorted(anchors.values(), key=lambda c: c.turn):   # oldest anchors first
            if id(p) not in seen:
                result.append(p)
                seen.add(id(p))
            if len(result) >= self.top_k:
                break
        if len(result) < self.top_k:                                # fill remaining by score
            for p in ranked:
                if id(p) not in seen:
                    result.append(p)
                    seen.add(id(p))
                if len(result) >= self.top_k:
                    break
        return result

    # -- detection --
    def detect_all(self, new_commitments: list, prior_commitments: list) -> list:
        """
        Judge which earlier commitments the new ones contradict (one LLM call).
        Returns a list of ContradictionFinding, strongest (highest confidence)
        first; empty if none. A reply under accumulating multi-turn pressure can
        break more than one established commitment at once, so every conflict is
        surfaced, not just the first the judge happens to name.
        """
        if not new_commitments or not prior_commitments:
            return []
        # Oldest first, so the model's original position leads and the judge can
        # anchor a drift verdict on it rather than on a later restatement.
        ordered_priors = sorted(prior_commitments, key=lambda c: getattr(c, "turn", 0))
        new_blob = "\n".join(f"- {c.claim}" for c in new_commitments)
        prior_blob = "\n".join(f"- {c.claim}" for c in ordered_priors)
        prompt = _DETECT_PROMPT.format(new_claims=new_blob, prior_claims=prior_blob)
        try:
            res = self.llm.complete_json(prompt, model=self._model())
        except Exception:
            return []
        findings: list = []
        for item in self._contradiction_items(res):
            if not isinstance(item, dict):
                continue
            prior_claim = item.get("prior_claim")
            matched = next(
                (c for c in ordered_priors if c.claim == prior_claim),
                None,
            ) or ordered_priors[0]
            # A changed volatile fact (balance, status, count, date) is a normal
            # state update for a stateful agent, not a contradiction. Suppress it
            # unless the caller opted into flagging state changes.
            if getattr(matched, "kind", "durable") == "volatile" and not self.flag_state_changes:
                continue
            conf = item.get("confidence")
            try:
                conf = float(conf) if conf is not None else None
            except (TypeError, ValueError):
                conf = None
            findings.append(ContradictionFinding(
                contradiction=True,
                new_claim=item.get("new_claim"),
                prior_claim=prior_claim or matched.claim,
                prior_query=matched.source_query,
                prior_response=matched.source_response,
                explanation=item.get("explanation"),
                confidence=conf,
            ))
        findings.sort(key=lambda f: f.confidence if f.confidence is not None else -1.0, reverse=True)
        return findings

    @staticmethod
    def _contradiction_items(res) -> list:
        """
        Normalize the judge reply into a list of contradiction dicts. Accepts the
        current {"contradictions": [...]} shape, a bare list, or the legacy
        single {"contradiction": bool, ...} verdict, so older judges and mocks
        keep working.
        """
        if isinstance(res, list):
            return res
        if isinstance(res, dict):
            if isinstance(res.get("contradictions"), list):
                return res["contradictions"]
            if "contradiction" in res:                       # legacy single verdict
                return [res] if res.get("contradiction") else []
            if res.get("prior_claim"):                       # a single bare item
                return [res]
        return []

    def detect(self, new_commitments: list, prior_commitments: list) -> ContradictionFinding:
        """
        Judge whether any new commitment contradicts a relevant prior one (one
        LLM call). Returns the strongest contradiction, or a negative finding.
        Backward-compatible single-finding view of detect_all.
        """
        found = self.detect_all(new_commitments, prior_commitments)
        return found[0] if found else ContradictionFinding(contradiction=False)

    # -- orchestration --
    def check(self, session: str, query: str, response: str) -> ContradictionFinding:
        """
        Full read path for one new reply: extract its commitments, retrieve the
        relevant priors, and judge contradiction. Does NOT store the new
        commitments (call ingest_commitments to persist). Costs 1 LLM call when
        there is nothing relevant to compare against, 2 otherwise.
        """
        new = self.extract(query, response, session=session)
        priors = self.relevant(session, new)
        finding = self.detect(new, priors) if priors else ContradictionFinding(contradiction=False)
        finding._new_commitments = new  # type: ignore[attr-defined]
        return finding

    def check_all(self, session: str, query: str, response: str) -> list:
        """
        Like check(), but returns every contradiction the new reply raises
        against the conversation, strongest first (empty if none). Use it when a
        single reply can break more than one established commitment, which gets
        common once a conversation has accumulated several. Does NOT store the
        new commitments (call ingest_commitments to persist).
        """
        new = self.extract(query, response, session=session)
        priors = self.relevant(session, new)
        findings = self.detect_all(new, priors) if priors else []
        for f in findings:
            f._new_commitments = new  # type: ignore[attr-defined]
        return findings

    def ingest_commitments(self, commitments: list) -> None:
        # Dedup first so an identical restatement does not accumulate in the
        # store, where it would bloat every later retrieval and cost more to
        # scan. Only exact normalized-claim matches are dropped, scoped to the
        # session, so a claim that conflicts with a prior one is never silently
        # merged away; that conflict is what the check path is meant to surface.
        keep = self._dedup(commitments) if self.dedup else list(commitments)
        # If the relevance scorer can persist embeddings (EmbeddingRelevance),
        # attach them before storing so the vector is computed once and shared
        # via the store across workers. Embedding is an optimization, so a
        # provider hiccup must never block storing the commitment itself.
        attach = getattr(self.relevance_fn, "attach", None)
        if callable(attach):
            try:
                attach(keep)
            except Exception:
                pass
        for c in keep:
            self.store.add(c)

    def _dedup(self, commitments: list) -> list:
        """Drop commitments whose normalized claim already exists in their
        session, or repeats earlier in this same batch. Returns the survivors
        in input order."""
        keep: list = []
        seen_by_session: dict = {}
        for c in commitments:
            sess = getattr(c, "session", "default")
            if sess not in seen_by_session:
                seen_by_session[sess] = {
                    self._norm_claim(p.claim) for p in self.store.by_session(sess)
                }
            key = self._norm_claim(getattr(c, "claim", ""))
            if not key or key in seen_by_session[sess]:
                continue
            seen_by_session[sess].add(key)
            keep.append(c)
        return keep

    @staticmethod
    def _norm_claim(claim: str) -> str:
        """Normalize a claim for duplicate detection: lowercase, collapse inner
        whitespace, strip surrounding quotes and trailing punctuation. Two
        commitments with the same normalized claim are treated as one."""
        import re
        t = re.sub(r"\s+", " ", (claim or "").strip().lower())
        return t.strip(" .;,:!?\"'")

    def ingest(self, session: str, query: str, response: str) -> list:
        """Extract commitments from a reply and store them. Returns them."""
        new = self.extract(query, response, session=session)
        self.ingest_commitments(new)
        return new

    # -- repair --
    def repair(self, query: str, response: str, finding: ContradictionFinding) -> Optional[str]:
        """
        Rewrite a contradictory reply so it honors the prior commitment (one LLM
        call). Returns the corrected text, or None if repair failed.
        """
        if not finding.contradiction or not finding.prior_claim:
            return None
        prompt = _REPAIR_PROMPT.format(
            query=query, response=response, prior_claim=finding.prior_claim,
        )
        try:
            fixed = self.llm.complete(prompt, model=self._model())
        except Exception:
            return None
        fixed = (fixed or "").strip()
        return fixed or None

    def clear(self, session: Optional[str] = None) -> None:
        self.store.clear(session)

    # -- parsing --
    @staticmethod
    def _parse_list(raw: str) -> list:
        """Coerce a model reply into a list of dicts. Tolerant of fences/objects."""
        import re
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return [raw]
        text = str(raw).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = _json.loads(text)
        except (ValueError, TypeError):
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                return []
            try:
                data = _json.loads(m.group())
            except (ValueError, TypeError):
                return []
        if isinstance(data, dict):
            # A single object, or {"commitments": [...]}
            for k in ("commitments", "claims", "items"):
                if isinstance(data.get(k), list):
                    return data[k]
            return [data]
        return data if isinstance(data, list) else []


__all__ = [
    "Commitment",
    "ContradictionFinding",
    "CommitmentStore",
    "InMemoryCommitmentStore",
    "RedisCommitmentStore",
    "ConversationMemory",
    "RelevanceFn",
    "EmbeddingRelevance",
    "openai_embedder",
    "EmbedFn",
]
