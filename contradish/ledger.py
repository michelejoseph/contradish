"""
contradish.ledger: an append-only, hash-chained record of what a model
committed to and where it contradicted itself, over time.

Contradiction detection (contradish.memory) is point-in-time: it answers "does
this reply conflict with what was said before". Auditing a model is a different
job. It is about observing behavior across time and being able to show a record
an outside party can trust. A CommitmentLedger is that record. Every commitment
and every contradiction is appended with a timestamp and a SHA-256 hash that
chains to the previous entry, so altering or dropping any past entry breaks the
chain and verify() returns False. Publish head() somewhere you do not control (a
git commit, a timestamped post) and the record becomes independently
tamper-evident: you can show later that the log was not quietly rewritten.

The chain proves the integrity of the log, not the truthfulness of the model.
Each commitment entry keeps the originating query and response as provenance.

    from contradish import CommitmentLedger, ConversationMemory

    ledger = CommitmentLedger()
    mem = ConversationMemory(ledger=ledger)
    # ... run the agent through the Firewall or call check()/ingest over time ...
    ledger.verify()           # True while untouched
    ledger.audit_summary()    # counts, time span, contradiction rate, head hash
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

_GENESIS = "0" * 64


@dataclass
class LedgerEntry:
    """One appended event: a commitment made or a contradiction observed."""
    seq:       int
    at:        float
    type:      str          # "commitment" or "contradiction"
    session:   str
    payload:   dict
    prev_hash: str
    hash:      str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerEntry":
        return cls(
            seq=int(d["seq"]),
            at=float(d["at"]),
            type=str(d["type"]),
            session=str(d.get("session", "default")),
            payload=dict(d.get("payload", {})),
            prev_hash=str(d.get("prev_hash", _GENESIS)),
            hash=str(d.get("hash", "")),
        )


def _entry_hash(prev_hash: str, seq: int, at: float, type_: str, session: str, payload: dict) -> str:
    """SHA-256 over the previous hash plus this entry's canonical body. Sorting
    keys makes the hash independent of dict ordering, so a re-serialized export
    verifies identically."""
    body = json.dumps(
        {"seq": seq, "at": at, "type": type_, "session": session, "payload": payload},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()


class CommitmentLedger:
    """
    Append-only, hash-chained record of commitments and contradictions over time.

    Tamper-evidence: each entry's hash covers the previous hash, so any edit,
    deletion, or reordering of a past entry makes verify() return False unless
    the whole chain is recomputed. Publishing head() externally closes that gap,
    because recomputing the chain would change the head you already committed to.
    """

    def __init__(self):
        self._entries: List[LedgerEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def head(self) -> str:
        """Hash of the latest entry (the value to publish for external anchoring)."""
        return self._entries[-1].hash if self._entries else _GENESIS

    def _append(self, type_: str, session: str, payload: dict, at: Optional[float] = None) -> LedgerEntry:
        seq = len(self._entries)
        at = time.time() if at is None else float(at)
        prev = self.head()
        h = _entry_hash(prev, seq, at, type_, session, payload)
        entry = LedgerEntry(seq=seq, at=at, type=type_, session=session,
                            payload=payload, prev_hash=prev, hash=h)
        self._entries.append(entry)
        return entry

    def record_commitment(self, commitment) -> LedgerEntry:
        """Append a commitment. Accepts a Commitment (or any object with
        to_dict) or a plain dict."""
        payload = commitment.to_dict() if hasattr(commitment, "to_dict") else dict(commitment)
        session = str(payload.get("session", "default"))
        at = payload.get("created_at")
        return self._append("commitment", session, payload, at=at)

    def record_contradiction(self, finding, session: str = "default") -> LedgerEntry:
        """Append a contradiction observation (a ContradictionFinding or dict)."""
        get = (lambda k: getattr(finding, k, None)) if not isinstance(finding, dict) else finding.get
        payload = {
            "new_claim":   get("new_claim"),
            "prior_claim": get("prior_claim"),
            "explanation": get("explanation"),
            "confidence":  get("confidence"),
        }
        return self._append("contradiction", session, payload)

    def timeline(self, session: Optional[str] = None, type: Optional[str] = None) -> list:
        """Entries in append (time) order, optionally filtered by session and/or
        type ("commitment" | "contradiction")."""
        out = self._entries
        if session is not None:
            out = [e for e in out if e.session == session]
        if type is not None:
            out = [e for e in out if e.type == type]
        return list(out)

    def verify(self) -> bool:
        """True only if every entry is in sequence, links to the prior hash, and
        re-hashes to its stored hash. Any tampering returns False."""
        prev = _GENESIS
        for i, e in enumerate(self._entries):
            if e.seq != i or e.prev_hash != prev:
                return False
            if _entry_hash(e.prev_hash, e.seq, e.at, e.type, e.session, e.payload) != e.hash:
                return False
            prev = e.hash
        return True

    def audit_summary(self, session: Optional[str] = None) -> dict:
        """At-a-glance audit: how much was observed, over what span, how often it
        contradicted itself, and whether the record is still intact."""
        entries = self.timeline(session)
        commits = [e for e in entries if e.type == "commitment"]
        contras = [e for e in entries if e.type == "contradiction"]
        times = [e.at for e in entries]
        return {
            "entries":            len(entries),
            "commitments":        len(commits),
            "contradictions":     len(contras),
            "contradiction_rate": round(len(contras) / len(commits), 3) if commits else 0.0,
            "first_at":           min(times) if times else None,
            "last_at":            max(times) if times else None,
            "verified":           self.verify(),
            "head":               self.head(),
        }

    def to_dict(self) -> dict:
        """Export the full chain for storage or external re-verification."""
        return {"entries": [e.to_dict() for e in self._entries], "head": self.head()}

    @classmethod
    def from_dict(cls, d: dict) -> "CommitmentLedger":
        led = cls()
        led._entries = [LedgerEntry.from_dict(x) for x in d.get("entries", [])]
        return led


__all__ = ["CommitmentLedger", "LedgerEntry"]
