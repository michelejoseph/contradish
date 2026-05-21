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
    """
    claim:           str
    topic:           str = ""
    session:         str = "default"
    source_query:    str = ""
    source_response: str = ""
    turn:            int = 0
    created_at:      float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Commitment":
        return cls(
            claim=str(d.get("claim", "")),
            topic=str(d.get("topic", "")),
            session=str(d.get("session", "default")),
            source_query=str(d.get("source_query", "")),
            source_response=str(d.get("source_response", "")),
            turn=int(d.get("turn", 0)),
            created_at=float(d.get("created_at", time.time())),
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
        per_session: optional cap on commitments retained per session
                     (oldest dropped first). None means unbounded.
    """

    def __init__(self, per_session: Optional[int] = None):
        if per_session is not None and per_session <= 0:
            raise ValueError(f"per_session must be > 0 or None, got {per_session}")
        self.per_session = per_session
        self._by_session: dict[str, list[Commitment]] = {}

    def add(self, commitment: Commitment) -> None:
        bucket = self._by_session.setdefault(commitment.session, [])
        bucket.append(commitment)
        if self.per_session is not None and len(bucket) > self.per_session:
            del bucket[0:len(bucket) - self.per_session]

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


# Signature for a pluggable scorer: (new_commitment, prior_commitment) -> [0,1]
RelevanceFn = Callable[[Commitment, Commitment], float]


# ── Prompts ───────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You extract the durable, checkable commitments an assistant made in a reply, so they can be checked for consistency later in the conversation.

User asked: {query}
Assistant replied: {response}

A commitment is a specific assertion the reply commits to: a policy, number, rule, eligibility, recommendation, or factual claim that could later be contradicted. Ignore pleasantries, hedges, clarifying questions, and pure restatement of the user's words. Normalize each into a short standalone statement that makes sense on its own.

For each commitment also give a short topic key of 2 to 4 lowercase words, used to find related statements later (for example "refund window", "melatonin long term use").

Return JSON only, no markdown, as a list:
[{{"claim": "<normalized standalone assertion>", "topic": "<short topic key>"}}]

If the reply makes no checkable commitment, return []."""

_DETECT_PROMPT = """You decide whether a new assistant statement contradicts something the same assistant already committed to earlier in this conversation.

New statement(s):
{new_claims}

Earlier statement(s) on related topics:
{prior_claims}

A contradiction means the assistant now asserts something that cannot both be true with an earlier assertion: a different policy, number, rule, eligibility, or conclusion on the same matter. Ignore differences in wording, added detail, or tone. A more specific or more cautious restatement is not a contradiction.

Return JSON only, no markdown:
{{"contradiction": true or false, "new_claim": "<the new claim involved, or null>", "prior_claim": "<the exact earlier claim it contradicts, or null>", "explanation": "<one sentence, or null>", "confidence": <number 0.0 to 1.0>}}"""

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
    ):
        self._llm_arg     = llm
        self._api_key     = api_key
        self._provider    = provider
        self.model        = model
        self.store        = store if store is not None else InMemoryCommitmentStore()
        self.relevance_fn = relevance_fn or _overlap_score
        self.relevance_threshold = relevance_threshold
        self.top_k        = top_k
        self._llm_cached  = llm  # may be None; built lazily on first LLM use

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
        scored by relevance_fn, deduped, highest score first, capped at top_k.
        No LLM call.
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
        ranked = sorted(scored.values(), key=lambda t: t[0], reverse=True)
        return [p for _, p in ranked[:self.top_k]]

    # -- detection --
    def detect(self, new_commitments: list, prior_commitments: list) -> ContradictionFinding:
        """Judge whether any new commitment contradicts a relevant prior one (one LLM call)."""
        if not new_commitments or not prior_commitments:
            return ContradictionFinding(contradiction=False)
        new_blob = "\n".join(f"- {c.claim}" for c in new_commitments)
        prior_blob = "\n".join(f"- {c.claim}" for c in prior_commitments)
        prompt = _DETECT_PROMPT.format(new_claims=new_blob, prior_claims=prior_blob)
        try:
            res = self.llm.complete_json(prompt, model=self._model())
        except Exception:
            return ContradictionFinding(contradiction=False)
        if not isinstance(res, dict) or not res.get("contradiction"):
            return ContradictionFinding(contradiction=False)
        prior_claim = res.get("prior_claim")
        matched = next(
            (c for c in prior_commitments if c.claim == prior_claim),
            None,
        ) or prior_commitments[0]
        conf = res.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        return ContradictionFinding(
            contradiction=True,
            new_claim=res.get("new_claim"),
            prior_claim=prior_claim or matched.claim,
            prior_query=matched.source_query,
            prior_response=matched.source_response,
            explanation=res.get("explanation"),
            confidence=conf,
        )

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

    def ingest_commitments(self, commitments: list) -> None:
        for c in commitments:
            self.store.add(c)

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
]
