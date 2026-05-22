"""
Replay: run the memory-aware contradiction check offline over logged
conversation transcripts.

The production Firewall (contradish.firewall) catches self-contradictions as a
live app generates them. Replay does the same thing after the fact: point it at
your recorded conversation logs and it reports every place the assistant
contradicted a commitment it made earlier in the same session. No app callable
is invoked, the responses already exist; replay only runs extraction,
relevance retrieval, and detection over the recorded text (plus optional
repair).

    from contradish import replay

    report = replay("conversations.jsonl")
    print(report.summary())
    for c in report.contradictions:
        print(c.session, c.turn_index, "contradicts", c.prior_turn_index)

Transcript formats are auto-detected and tolerant:

  - OpenAI chat-message logs: a list (or JSONL) of {"role", "content"}.
    Consecutive user/assistant messages are paired into turns.
  - Paired turns: {"query"/"input"/"prompt", "response"/"output"/"completion"}.
  - Multiple conversations in one file: a list of objects each carrying a
    nested "messages"/"turns" list, or a top-level {"conversations": [...]}.
  - Sessions: a "session"/"session_id"/"conversation_id" field on a turn or
    conversation scopes it. Without one, everything is a single session.

Because each turn runs the same extraction/detection as the Firewall, replay
costs LLM calls (roughly one extraction per turn, one detection per turn that
has a relevant prior). It is an audit you run over real logs, not a hot path.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Optional

from .memory import ConversationMemory


# ── Field name vocabularies (tolerant matching) ─────────────────────────────

_QUERY_KEYS    = ("query", "input", "prompt", "user", "question",
                  "user_message", "user_input", "request")
_RESPONSE_KEYS = ("response", "output", "completion", "assistant", "answer",
                  "assistant_message", "reply", "model_output", "text_out")
_SESSION_KEYS  = ("session", "session_id", "conversation_id", "convo_id",
                  "thread_id", "user_id")
_NESTED_KEYS   = ("messages", "turns", "conversation", "history", "dialog",
                  "dialogue")
_CONV_LIST_KEYS = ("conversations", "sessions", "logs", "threads")


def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _session_of(d: dict, default: str, extra=()) -> str:
    for k in tuple(_SESSION_KEYS) + tuple(extra):
        v = d.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v)
    return default


def _has_nested_list(d: dict) -> bool:
    return any(isinstance(d.get(k), list) for k in _NESTED_KEYS)


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class ReplayTurn:
    """One (query, response) turn from a transcript, scoped to a session."""
    session:  str
    index:    int          # 0-based position within its session
    query:    str
    response: str

    def to_dict(self) -> dict:
        return {"session": self.session, "index": self.index,
                "query": self.query, "response": self.response}


@dataclass
class ReplayContradiction:
    """A self-contradiction found between a turn and an earlier turn."""
    session:          str
    turn_index:       int               # the offending (later) turn
    query:            str
    response:         str
    prior_turn_index: Optional[int]      # the earlier turn it contradicts
    prior_query:      Optional[str]
    new_claim:        Optional[str]
    prior_claim:      Optional[str]
    explanation:      Optional[str]
    confidence:       Optional[float]
    repaired:         Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session":          self.session,
            "turn_index":       self.turn_index,
            "query":            self.query,
            "response":         self.response,
            "prior_turn_index": self.prior_turn_index,
            "prior_query":      self.prior_query,
            "new_claim":        self.new_claim,
            "prior_claim":      self.prior_claim,
            "explanation":      self.explanation,
            "confidence":       self.confidence,
            "repaired":         self.repaired,
        }


@dataclass
class ReplayReport:
    """Result of replaying a transcript through the memory contradiction check."""
    contradictions: list = field(default_factory=list)
    n_turns:        int = 0
    n_commitments:  int = 0
    sessions:       list = field(default_factory=list)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def contradiction_rate(self) -> float:
        return round(len(self.contradictions) / self.n_turns, 4) if self.n_turns else 0.0

    def by_session(self, session: str) -> list:
        return [c for c in self.contradictions if c.session == session]

    def to_dict(self) -> dict:
        return {
            "n_turns":            self.n_turns,
            "n_sessions":         self.n_sessions,
            "n_commitments":      self.n_commitments,
            "n_contradictions":   len(self.contradictions),
            "contradiction_rate": self.contradiction_rate,
            "sessions":           list(self.sessions),
            "contradictions":     [c.to_dict() for c in self.contradictions],
        }

    def summary(self) -> str:
        lines = []
        lines.append("")
        lines.append("  contradish replay")
        lines.append(f"  {self.n_sessions} sessions · {self.n_turns} turns · "
                     f"{self.n_commitments} commitments")
        n = len(self.contradictions)
        if n == 0:
            lines.append(f"  no self-contradictions found across {self.n_turns} turns.")
            lines.append("")
            return "\n".join(lines)
        lines.append(f"  contradictions: {n}  (rate {self.contradiction_rate})")
        lines.append("")
        for session in self.sessions:
            hits = self.by_session(session)
            if not hits:
                continue
            lines.append(f"  [session {session}]")
            for c in hits:
                prior = f"turn {c.prior_turn_index}" if c.prior_turn_index is not None else "an earlier turn"
                lines.append(f"    turn {c.turn_index} contradicts {prior}")
                if c.new_claim:
                    lines.append(f"      now:     {c.new_claim}")
                if c.prior_claim:
                    lines.append(f"      earlier: {c.prior_claim}")
                if c.explanation:
                    lines.append(f"      why:     {c.explanation}")
                if c.repaired:
                    lines.append(f"      fix:     {c.repaired}")
                lines.append("")
        return "\n".join(lines)


# ── Loading / normalization ─────────────────────────────────────────────────

def _read(source) -> object:
    """Path or in-memory object -> parsed JSON value (auto-detects JSON vs JSONL)."""
    if isinstance(source, (list, dict)):
        return source
    with open(source, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        return _json.loads(text)            # whole-file JSON (array or object)
    except (ValueError, TypeError):
        pass
    out = []                                # fall back to JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(_json.loads(line))
        except (ValueError, TypeError):
            continue
    return out


def _walk_flat(items, default_session: str) -> list:
    """A flat list of chat messages and/or paired turns -> ReplayTurns."""
    turns: list = []
    pending: dict = {}     # session -> last unpaired user content
    idx: dict = {}         # session -> next per-session turn index

    def _emit(session, query, response):
        i = idx.get(session, 0)
        turns.append(ReplayTurn(session=session, index=i, query=query, response=response))
        idx[session] = i + 1

    for it in items:
        if not isinstance(it, dict):
            continue
        session = _session_of(it, default_session)
        if "role" in it:
            role = str(it.get("role", "")).strip().lower()
            content = it.get("content")
            if content is None:
                content = it.get("text", "")
            content = content if isinstance(content, str) else _json.dumps(content)
            if role in ("user", "human", "customer"):
                pending[session] = content
            elif role in ("assistant", "ai", "bot", "model", "agent"):
                query = pending.pop(session, "")
                _emit(session, query, content)
            # system / tool / developer messages carry no commitment to check
        else:
            response = _first(it, _RESPONSE_KEYS, None)
            if response is None:
                continue
            query = _first(it, _QUERY_KEYS, "") or ""
            _emit(session, str(query), str(response))
    return turns


def _normalize(data, default_session: str) -> list:
    if isinstance(data, dict):
        # A container of multiple conversations.
        for k in _CONV_LIST_KEYS:
            if isinstance(data.get(k), list):
                turns = []
                for conv in data[k]:
                    s = _session_of(conv, default_session, extra=("id", "name")) if isinstance(conv, dict) else default_session
                    turns += _normalize(conv, s)
                return turns
        # A single conversation wrapping a message/turn list.
        for k in _NESTED_KEYS:
            if isinstance(data.get(k), list):
                s = _session_of(data, default_session, extra=("id", "name"))
                return _walk_flat(data[k], s)
        # A lone paired turn.
        response = _first(data, _RESPONSE_KEYS, None)
        if response is not None:
            s = _session_of(data, default_session)
            return [ReplayTurn(session=s, index=0,
                               query=str(_first(data, _QUERY_KEYS, "") or ""),
                               response=str(response))]
        return []

    if isinstance(data, list):
        # A list of conversation objects (each has a nested message list).
        if data and all(isinstance(x, dict) and _has_nested_list(x) for x in data):
            turns = []
            for conv in data:
                s = _session_of(conv, default_session, extra=("id", "name"))
                turns += _normalize(conv, s)
            return turns
        # Otherwise a flat list of messages and/or paired turns.
        return _walk_flat(data, default_session)

    return []


def load_transcript(source) -> list:
    """
    Load a transcript from a path (JSON or JSONL) or an in-memory list/dict and
    normalize it into an ordered list of ReplayTurn. Tolerant of the common
    chat-message, paired-turn, and multi-conversation shapes.
    """
    return _normalize(_read(source), "default")


# ── Engine ──────────────────────────────────────────────────────────────────

def _find_prior_index(turns, offending: ReplayTurn, prior_query: Optional[str]) -> Optional[int]:
    """Best-effort map the matched commitment's source query back to a turn index."""
    if not prior_query:
        return None
    best = None
    for t in turns:
        if t.session != offending.session:
            continue
        if t.index >= offending.index:
            continue
        if t.query == prior_query:
            best = t.index if best is None else max(best, t.index)
    return best


def replay_transcript(
    turns,
    memory:     Optional[ConversationMemory] = None,
    repair:     bool = False,
    api_key:    Optional[str] = None,
    provider:   Optional[str] = None,
    model:      Optional[str] = None,
    embed_fn=None,
) -> ReplayReport:
    """
    Replay an ordered list of ReplayTurn through a ConversationMemory and report
    cross-turn self-contradictions.

    Args:
        turns:    list of ReplayTurn (from load_transcript).
        memory:   a ConversationMemory. Built lexical by default; pass one from
                  ConversationMemory.with_embeddings(...) for semantic relevance,
                  or pass embed_fn to have replay build it.
        repair:   when True, also compute a corrected reply for each contradiction.
        embed_fn: optional batch embedder; if given (and memory is None), the
                  memory is built with embedding-based relevance.
    """
    if memory is None:
        if embed_fn is not None:
            memory = ConversationMemory.with_embeddings(
                embed_fn, api_key=api_key, provider=provider, model=model)
        else:
            memory = ConversationMemory(api_key=api_key, provider=provider, model=model)

    contradictions: list = []
    sessions: list = []
    n_commitments = 0

    for t in turns:
        if t.session not in sessions:
            sessions.append(t.session)
        finding = memory.check(t.session, t.query, t.response)
        new = getattr(finding, "_new_commitments", None)
        if new is None:
            new = memory.extract(t.query, t.response, session=t.session)
        if finding.contradiction:
            rep = memory.repair(t.query, t.response, finding) if repair else None
            contradictions.append(ReplayContradiction(
                session=t.session,
                turn_index=t.index,
                query=t.query,
                response=t.response,
                prior_turn_index=_find_prior_index(turns, t, finding.prior_query),
                prior_query=finding.prior_query,
                new_claim=finding.new_claim,
                prior_claim=finding.prior_claim,
                explanation=finding.explanation,
                confidence=finding.confidence,
                repaired=rep,
            ))
        memory.ingest_commitments(new)
        n_commitments += len(new)

    return ReplayReport(
        contradictions=contradictions,
        n_turns=len(turns),
        n_commitments=n_commitments,
        sessions=sessions,
    )


def replay(source, **kwargs) -> ReplayReport:
    """Convenience: load a transcript from `source` and replay it. Accepts the
    same keyword arguments as replay_transcript()."""
    return replay_transcript(load_transcript(source), **kwargs)


__all__ = [
    "ReplayTurn",
    "ReplayContradiction",
    "ReplayReport",
    "load_transcript",
    "replay_transcript",
    "replay",
]
