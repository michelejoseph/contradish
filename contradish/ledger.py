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

To persist across runs instead of a single Python process:

    ledger = CommitmentLedger.load(".contradish/ledger.json")
    # ... record_commitment() / record_contradiction() / record_event() ...
    ledger.save(".contradish/ledger.json")

`contradish monitor` does exactly this by default (see --ledger / --no-ledger),
so every run appends to one growing, tamper-evident record instead of a
one-off snapshot. `contradish ledger show|verify|anchor` reads the same file
from the command line without writing any Python.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

_GENESIS = "0" * 64

#: Default on-disk location for `contradish ledger` and `contradish monitor
#: --ledger`. A dotfile next to `.contradish.yaml`, so a project accumulates
#: one shared, growing audit trail unless a path is given explicitly.
DEFAULT_LEDGER_PATH = ".contradish/ledger.json"


@dataclass
class LedgerEntry:
    """One appended event: a commitment made, a contradiction observed, or any
    other event a caller chooses to record (see record_event)."""
    seq:       int
    at:        float
    type:      str          # "commitment" | "contradiction" | caller-defined
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

    def record_event(self, type_: str, session: str, payload: dict,
                      at: Optional[float] = None) -> LedgerEntry:
        """Append an arbitrary event under a caller-defined type, e.g. a
        `contradish monitor` run summary, a benchmark run, a deploy marker.
        Use record_commitment / record_contradiction for those two specific,
        well-known shapes; use this for anything else you want on the record."""
        return self._append(str(type_), session, dict(payload), at=at)

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
        other_types: dict = {}
        for e in entries:
            if e.type not in ("commitment", "contradiction"):
                other_types[e.type] = other_types.get(e.type, 0) + 1
        times = [e.at for e in entries]
        return {
            "entries":            len(entries),
            "commitments":        len(commits),
            "contradictions":     len(contras),
            "contradiction_rate": round(len(contras) / len(commits), 3) if commits else 0.0,
            "other_event_types":  other_types,
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

    def save(self, path: Union[str, Path] = DEFAULT_LEDGER_PATH) -> Path:
        """Write the full chain to disk as JSON, creating parent directories
        as needed. Loading it back with load() and calling verify() confirms
        nothing between save and load was altered."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p

    @classmethod
    def load(cls, path: Union[str, Path] = DEFAULT_LEDGER_PATH) -> "CommitmentLedger":
        """Read a ledger back from disk. A missing file is not an error: it
        means tracking hasn't started yet, so this returns a fresh, empty
        ledger that save() will create the file for."""
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text()))

    def anchor_text(self, label: Optional[str] = None) -> str:
        """A short, copy-pasteable line that ties the current state of this
        ledger to a point in time. Paste it somewhere you don't control and
        can't quietly edit later: a git commit message, a support ticket
        reply, a public post. If the chain is later found not to verify, or
        the head hash on record doesn't match, the log was altered after
        that anchor point. This needs no external service, only that
        wherever you paste it is somewhere you don't unilaterally control."""
        s = self.audit_summary()
        when = datetime.fromtimestamp(s["last_at"], tz=timezone.utc).isoformat() if s["last_at"] else "n/a"
        tag = f"[{label}] " if label else ""
        return (f"{tag}contradish ledger anchor · {s['entries']} entries · "
                f"head {s['head']} · as of {when}")


__all__ = ["CommitmentLedger", "LedgerEntry", "DEFAULT_LEDGER_PATH"]
