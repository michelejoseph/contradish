"""
contradish/evidence.py -- machine-verifiable evidence of a behavioral change
that the governing information did not warrant.

A finding from verify_trajectories() is turned into a CERTIFICATE: one JSON
document that carries everything needed to re-establish the finding from
scratch, and nothing that has to be trusted:

    policy        the full machine-readable policy before the update
    update        who it came from, through which channel, what it said,
                  and the edits it would make
    situation     the facts of the case
    step          which action of the procedure
    observed      the agent's tool call for that step before and after the
                  update, and (for a model) its raw replies, so the calls can
                  be re-parsed rather than taken on trust
    derivation    what contradish computed: warranted calls before and
                  after, the asserted call, which clauses were read, which
                  edits were authorized. A checker recomputes all of this and
                  rejects the certificate if any of it disagrees.
    digest        sha256 of the canonical JSON of everything else

Two kinds:

    unnecessary_change   the policy required the action to stay the same
                         (warranted call identical before and after) and
                         the agent changed it.
    unauthorized_change  the agent's new call is exactly what an edit from a
                         source WITHOUT authority over that clause would
                         produce, and differs from the warranted call.

The checker (contradish/evidence_check.py) is a separate implementation with
its own policy evaluator and imports nothing from contradish, so a
certificate can be verified without trusting the code that produced it:

    python -m contradish.evidence_check certificate.json
    python contradish/evidence_check.py certificate.json   # standalone file

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Optional

from contradish.action_frontier import ActionCaseResult, ActionFrontier, ActionTransitionResult
from contradish.policy_program import canonical_json, digest

__all__ = ["SCHEMA", "certificate", "certificates", "seal", "save", "load", "ATTRIBUTION"]

SCHEMA = "contradish.evidence/1.1"
ATTRIBUTION = "Behavioral Update Fidelity was introduced by Michele Joseph in 2026."


def _c(call) -> Optional[dict]:
    return call.to_dict() if call is not None else None


def seal(doc: dict) -> dict:
    body = {k: v for k, v in doc.items() if k != "digest"}
    out = dict(body)
    out["digest"] = digest(body)
    return out


def certificate(frontier: ActionFrontier, case: ActionCaseResult, agent: str, agent_kind: str = "scripted",
                model: Optional[str] = None, raw: Optional[dict] = None, kind: Optional[str] = None,
                note: str = "") -> dict:
    """Certificate for one finding. `kind` defaults to unauthorized_change when it applies."""
    v = frontier.at(case.situation, case.step)
    if kind is None:
        kind = "unauthorized_change" if case.unauthorized_change else "unnecessary_change"
    if kind == "unauthorized_change" and not case.unauthorized_change:
        raise ValueError("case is not an unauthorized change")
    if kind == "unnecessary_change" and not case.unnecessary_change:
        raise ValueError("case is not an unnecessary change")
    s = frontier.situations[case.situation]
    if kind == "unnecessary_change":
        claim = (f"Under update '{frontier.update.id}', the policy requires step '{case.step}' to stay "
                 f"{_show(v.after)} in this situation ({v.basis}: "
                 + ("the step reads nothing the update redefined" if v.basis == "independent"
                    else "the step reads something the update redefined, but its value is unchanged")
                 + f"). The agent changed it from {_show(case.observed_before)} to {_show(case.observed_after)}.")
    else:
        bad = [e.clause for e in frontier.unauthorized]
        claim = (f"Source '{frontier.update.source}' has no authority over clause(s) {bad}, so step "
                 f"'{case.step}' must be {_show(v.after)}. The agent's call after the update, "
                 f"{_show(case.observed_after)}, is exactly what obeying the unauthorized edit produces.")
    doc = {
        "schema": SCHEMA,
        "kind": kind,
        "claim": claim,
        "policy": frontier.program.to_dict(),
        "update": frontier.update.to_dict(),
        "situation": s,
        "step": case.step,
        "observed": {"before": _c(case.observed_before), "after": _c(case.observed_after)},
        "derivation": {
            "warranted_before": v.before.to_dict(),
            "warranted_after": v.after.to_dict(),
            "asserted_after": v.asserted.to_dict(),
            "must": v.must,
            "basis": v.basis,
            "resist": v.resist,
            "read_clauses": list(v.read_clauses),
            "read_names": list(v.read_names),
            "changed_clauses": sorted(frontier.changed),
            "changed_names": sorted(frontier.changed_names),
            "authorized_edit_clauses": [e.clause for e in frontier.authorized],
            "unauthorized_edit_clauses": [e.clause for e in frontier.unauthorized],
            "status": case.status,
        },
        "provenance": {
            "agent": agent,
            "agent_kind": agent_kind,         # "scripted" (a witness, not a model) or "model"
            "model": model,
            "raw": raw,                       # {"before": str, "after": str} for model agents
            "note": note,
            "produced_by": _producer(),
            "created": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        },
        "attribution": ATTRIBUTION,
    }
    return seal(doc)


def certificates(result: ActionTransitionResult, agent_kind: str = "scripted", model: Optional[str] = None,
                 raws: Optional[dict] = None, limit: Optional[int] = None) -> list:
    out = []
    for c in result.findings():
        raw = (raws or {}).get(c.situation)
        out.append(certificate(result.frontier, c, result.agent, agent_kind, model, raw))
        if limit is not None and len(out) >= limit:
            break
    return out


def _show(call) -> str:
    if hasattr(call, "modality"):
        if call.modality == "F":
            return "forbidden (not called)"
        return ("required: " if call.modality == "O" else "permitted: ") + _show(call.call)
    if call is None:
        return "not called"
    d = call.to_dict() if hasattr(call, "to_dict") else call
    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in sorted(d["args"].items()))
    return f"{d['tool']}({args})"


def _producer() -> str:
    try:
        from contradish import __version__
        return f"contradish {__version__}"
    except Exception:  # pragma: no cover
        return "contradish"


def save(doc: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
