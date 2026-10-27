"""
contradish/fulfillment.py -- remembering what you owe is not the same as being able to pay it.

An agent can remember what it is allowed to do, remember what it owes, and still lose the ability to fulfil what it
owes. Three ways:

  1  the memory drops what fulfilment needs: "owes a $40 refund" kept, the card and the order dropped. No DECISION
     depends on which card, so a decision-preserving compression throws it away;
  2  the capacity ends before the obligation does: a grant expires, or a scheduled revocation takes effect, before
     the deadline;
  3  the capacity can be revoked: the only route to fulfilment runs through grants someone may legitimately revoke.

The policy declares obligation types (`memory.obligations`):

    "refund": {"tool": "issue_refund", "params": ["order_id", "card_token", "amount"],
               "authority": "agent_may_dispense",          # a delegated fact, or null if none is needed
               "fallback": {"tool": "hand_to_pharmacist", "authority": null}}   # optional

History events: {"kind": "obligation", "id", "type", "owed_to", "incurred", "due", "params": {...}, "source"} and
{"kind": "discharged", "id", "at"}. An obligation is open from `incurred` until discharged.

On a day, under a future (the same futures as contradish.transform: no revocation; each kept grant revoked by its
issuer on the first day and on each version change; every pair revoked on the first day), an obligation can be
fulfilled if every parameter is known exactly and the agent holds the authority its type needs that day, or its
fallback route is open that day.

    assess(policy, memory, as_of)  ->  per open obligation:
        fulfillable       some capable day before the deadline on every future
        act_by D          fulfillable, but with no revocation the capacity ends on D, before the deadline: the
                          last safe day to act
        stranded_by_schedule   with no revocation at all, there is no capable day before the deadline
        params_missing    the memory does not hold what fulfilment needs (names them)
        and, for every status, `stranding_futures`: the legitimate revocations under which no capable day remains
        (with who would have to revoke what), and the remedy: act earlier, secure authority that does not run
        through a revocable grant, or add a fallback route.

The transformation verdict (contradish.transform) gains two kinds: `obligation_forgotten` (an obligation open in
the memory before is not in the memory after) and `fulfillment_loss` (an obligation the memory before could fulfil
on some future cannot be fulfilled on it after the change).

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Optional

__all__ = ["open_obligations", "capable", "assess", "fulfillment_violations", "OB_KINDS"]

OB_KINDS = ("obligation_forgotten", "fulfillment_loss")


def _d(s: str) -> _dt.date:
    return _dt.date.fromisoformat(s)


def _iso(d: _dt.date) -> str:
    return d.isoformat()


def open_obligations(history: list, cfg: dict, as_of: str) -> tuple:
    """(open obligations as kept in memory, discarded). Obligations from a source the policy does not accept for the
    type (cfg obligations[type].sources, if given) are claims and discarded; discharged ones are dropped."""
    types = (cfg.get("obligations") or {})
    out, discarded, done = {}, [], {}
    for i, e in enumerate(history):
        if e.get("kind") == "discharged":
            done[e["id"]] = (i, e.get("at") or as_of)
    for i, e in enumerate(history):
        if e.get("kind") != "obligation":
            continue
        t = types.get(e.get("type"))
        if t is None:
            discarded.append({"event": i, "reason": f"obligation of an undeclared type {e.get('type')}"})
            continue
        if t.get("sources") and e.get("source") not in t["sources"]:
            discarded.append({"event": i, "reason": f"claim: {e.get('source')} does not establish {e['type']} obligations"})
            continue
        if e["id"] in done and done[e["id"]][1] <= as_of:
            discarded.append({"event": i, "reason": f"obligation {e['id']} discharged"})
            discarded.append({"event": done[e["id"]][0], "reason": f"discharge of {e['id']} (applied)"})
            continue
        if (e.get("incurred") or as_of) > as_of:
            pass
        out[e["id"]] = {"id": e["id"], "type": e["type"], "owed_to": e.get("owed_to"),
                        "incurred": e.get("incurred"), "due": e["due"],
                        "params": {k: e.get("params", {}).get(k) for k in t.get("params", []) if k in (e.get("params") or {})}}
    return [out[k] for k in sorted(out)], discarded


def _chains_ok(chains: list, day: str, revoked: dict) -> bool:
    for c in chains:
        if not (c["from"] <= day and (c["until"] is None or day < c["until"])):
            continue
        if any(l["id"] in revoked and revoked[l["id"]] <= day for l in c["links"]):
            continue
        return True
    return False


def _route_open(memory: dict, authority: Optional[str], day: str, revoked: dict) -> bool:
    if authority is None:
        return True
    a = (memory.get("authority") or {}).get(authority) or {}
    return _chains_ok(a.get("chains") or [], day, revoked)


def capable(memory: dict, cfg: dict, ob: dict, day: str, revoked: dict) -> Optional[str]:
    """The route by which `ob` can be fulfilled on `day` under the revocations `revoked`, or None."""
    t = (cfg.get("obligations") or {}).get(ob["type"]) or {}
    if any(p not in (ob.get("params") or {}) or ob["params"][p] is None for p in t.get("params", [])):
        return None
    if _route_open(memory, t.get("authority"), day, revoked):
        return t.get("tool")
    fb = t.get("fallback")
    if fb and _route_open(memory, fb.get("authority"), day, revoked):
        return fb.get("tool")
    return None


def _days(memory: dict, ob: dict, as_of: str, futures: list) -> list:
    """Days on which capacity can change, from as_of to the deadline (exclusive of days after it)."""
    due = ob["due"]
    pts = {as_of, due}
    for a in (memory.get("authority") or {}).values():
        for c in a.get("chains") or []:
            pts.add(c["from"])
            if c["until"]:
                pts.add(c["until"])
                pts.add(_iso(_d(c["until"]) - _dt.timedelta(days=1)))
    for F in futures:
        for d in F["revoked"].values():
            pts.add(d)
            pts.add(_iso(_d(d) - _dt.timedelta(days=1)))
    return sorted(p for p in pts if as_of <= p <= due)


def _status_one(memory: dict, cfg: dict, ob: dict, as_of: str, futures: list) -> dict:
    t = (cfg.get("obligations") or {}).get(ob["type"]) or {}
    missing = [p for p in t.get("params", []) if (ob.get("params") or {}).get(p) is None]
    per = {}
    for F in futures:
        days = _days(memory, ob, as_of, futures)
        ok = [(d, capable(memory, cfg, ob, d, F["revoked"])) for d in days]
        ok = [(d, r) for d, r in ok if r]
        per[F["label"]] = {"capable_days": [d for d, _ in ok], "routes": sorted({r for _, r in ok})}
    base = per[futures[0]["label"]]
    if missing:
        status = "params_missing"
    elif ob["due"] < as_of:
        status = "overdue"
    elif not base["capable_days"]:
        status = "stranded_by_schedule"
    else:
        last = max(base["capable_days"])
        # capacity ends before the deadline (the last capable day is not the deadline)
        status = "act_by" if last < ob["due"] else "fulfillable"
    stranding = [{"future": k} for k, v in per.items() if not v["capable_days"]]
    out = {"id": ob["id"], "type": ob["type"], "due": ob["due"], "status": status, "missing_params": missing,
           "stranding_futures": stranding, "routes": base["routes"]}
    if status == "act_by":
        out["act_by"] = max(base["capable_days"])
    remedy = []
    if missing:
        remedy.append("re-establish " + ", ".join(missing) + " from the record before the obligation can be paid")
    if status == "act_by":
        remedy.append(f"fulfil by {out['act_by']}: after that no route is open before the deadline {ob['due']}")
    if status == "stranded_by_schedule":
        remedy.append("no route is open on any day before the deadline: escalate now")
    if stranding and not missing:
        remedy.append("a legitimate revocation would leave no route: fulfil now, or add a fallback that does not run "
                      "through a revocable grant")
    out["remedy"] = remedy
    return out


def assess(policy: Any, memory: dict, as_of: str, horizon: Optional[int] = None) -> dict:
    from contradish.memory_compiler import _World, load_memory_policy
    from contradish.transform import _futures
    spec, updates, cfg = load_memory_policy(policy)
    w = _World(spec, updates, cfg, as_of, int(horizon or cfg.get("horizon_days", 30)))
    futures = _futures(memory, memory, w)
    obs = [_status_one(memory, cfg, ob, as_of, futures) for ob in memory.get("obligations") or []]
    return {"as_of": as_of, "obligations": obs, "futures_checked": [f["label"] for f in futures],
            "summary": {s: sum(o["status"] == s for o in obs) for s in
                        ("fulfillable", "act_by", "stranded_by_schedule", "params_missing", "overdue")},
            "fragile": [o["id"] for o in obs if o["stranding_futures"]]}


def fulfillment_violations(cfg: dict, before: dict, after: dict, as_of: str, futures: list) -> list:
    """Violations a memory change causes for obligations open in the memory before."""
    out = []
    after_ids = {o["id"]: o for o in after.get("obligations") or []}
    for ob in before.get("obligations") or []:
        if ob["id"] not in after_ids:
            out.append({"kind": "obligation_forgotten", "day": as_of, "future": futures[0]["label"],
                        "obligation": ob["id"], "owed_to": ob.get("owed_to"), "due": ob["due"]})
            continue
        ob2 = after_ids[ob["id"]]
        for F in futures:
            days = _days(before, ob, as_of, futures)
            b = [d for d in days if capable(before, cfg, ob, d, F["revoked"])]
            if not b:
                continue
            days2 = sorted(set(days) | set(_days(after, ob2, as_of, futures)))
            a = [d for d in days2 if capable(after, cfg, ob2, d, F["revoked"])]
            if not a:
                t = (cfg.get("obligations") or {}).get(ob["type"]) or {}
                miss = [p for p in t.get("params", []) if (ob2.get("params") or {}).get(p) is None]
                out.append({"kind": "fulfillment_loss", "day": b[0], "future": F["label"], "obligation": ob["id"],
                            "due": ob["due"], "missing_params": miss,
                            "detail": ("the memory after no longer holds " + ", ".join(miss)) if miss else
                                      "no route to fulfilment remains open before the deadline"})
    return out
