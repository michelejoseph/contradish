"""
contradish/charter.py -- the legitimacy of a change to an AI agent's
governing specification.

Verifying that an agent follows its specification is worth nothing if the
specification itself was changed illegitimately. This module verifies the
change. The rules about how a specification may change (Hart's "rules of
change") are written down as a machine-readable CHARTER inside the
specification itself, and every transition v1 -> v2 is checked against
v1's charter on four counts:

  1. PROVENANCE   who issued v2, authenticated (versions.authenticate_transition):
                  signatures, a trust anchor, the issuer's key listed in v1,
                  and v2 bound to v1's digest.

  2. PROCEDURE    the process the charter requires for what the change
                  touches: approvals from k of n named roles (each an
                  Ed25519 signature over v2's body by a key v1 lists for
                  that role), a minimum review period between proposal and
                  effect, and no retroactive effect.

  3. AUTHORITY BY EFFECT
                  The issuer must be granted every behavioral EFFECT the
                  change has, not merely the clauses it edits. Effects are
                  derived exactly: the complete symbolic difference
                  (symbolic.py) gives, per region of the situation space and
                  per action, whether an obligation was gained or lost, a
                  permission granted or revoked, or the content of a call
                  changed (and which arguments). The charter grants DOMAINS
                  of effect to roles: actions x channels x arguments x a
                  condition on the situation, evaluated under v1. An effect
                  outside every domain granted to the issuer makes the
                  change illegitimate, even when the issuer owns every
                  clause it edited. Editing one clause can change behavior
                  someone else governs; clause-level authority cannot see
                  that, effect-level authority can.

  4. INVARIANTS   constraints every version must satisfy, proved over the
                  ENTIRE situation space of v2, not tested at sample points:
                    modality   "under condition C, action A is forbidden /
                               required / allowed"
                    bound      "under condition C, argument X of action A is
                               <= / >= / == an expression"
                  Modality is constant within each region; a bound on a
                  linear argument is checked exactly at the extreme
                  situations of each region (a linear function attains its
                  extremes there). A violation comes with a witness
                  situation. ENTRENCHED invariants must also survive every
                  amendment of the charter unchanged.

  Charter amendments (v2's charter differs from v1's) additionally need
  the charter's own amendment procedure.

The result is a LEGITIMACY CERTIFICATE. The independent checker
(evidence_check.check_legitimacy) recomputes every part with separate code:
signatures, approvals, the effect decomposition, domain coverage, the
procedure, and every invariant proof.

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import copy
import datetime as _dt
import itertools
import json
import math
from fractions import Fraction
from typing import Any, Optional

from contradish.policy_program import FORBIDDEN, OBLIGATORY, PERMITTED, PolicyProgram, ProgramError, canonical_json
from contradish.symbolic import (
    Cell, Lin, OutsideFragment, _as_lin, _q, classify, decompose, regions, seval,
)

__all__ = ["CHARTER_KEY", "validate_charter", "derive_effects", "check_invariants", "legitimacy_certificate",
           "approve_pin", "LEGIT_SCHEMA"]

CHARTER_KEY = "charter"
LEGIT_SCHEMA = "contradish.legitimacy_certificate/1.0"
HIDDEN = "__charter__"
ATTRIBUTION = "Behavioral Update Fidelity was introduced by Michele Joseph in 2026."
CHANNELS = ("obligation", "permission", "content")


# ─────────────────────────────────────────────────────────────────────────────
# The charter
# ─────────────────────────────────────────────────────────────────────────────

def charter_of(p: PolicyProgram) -> dict:
    return copy.deepcopy(p.spec.get(CHARTER_KEY) or {})


def validate_charter(p: PolicyProgram) -> list:
    """Structural problems with a program's charter (empty list = well formed)."""
    ch = charter_of(p)
    probs = []
    if not ch:
        return ["the specification has no charter"]
    steps = set(p.step_ids())
    doms = ch.get("domains") or {}
    for d, spec in doms.items():
        for a in spec.get("actions") or []:
            if a not in steps:
                probs.append(f"domain {d!r} names unknown action {a!r}")
        for c in spec.get("channels") or []:
            if c not in CHANNELS:
                probs.append(f"domain {d!r} has unknown channel {c!r}")
    for role, ds in (ch.get("grants") or {}).items():
        if role not in p.sources:
            probs.append(f"grant to unknown role {role!r}")
        for d in ds:
            if d not in doms:
                probs.append(f"grant of unknown domain {d!r} to {role!r}")
    for r in ch.get("procedures") or []:
        for role in r.get("approvers") or []:
            if role not in p.sources:
                probs.append(f"procedure {r.get('id')!r} names unknown role {role!r}")
        ds = r.get("domains")
        if ds != "*":
            for d in ds or []:
                if d not in doms:
                    probs.append(f"procedure {r.get('id')!r} names unknown domain {d!r}")
    ids = set()
    for inv in ch.get("invariants") or []:
        if inv.get("id") in ids:
            probs.append(f"duplicate invariant {inv.get('id')!r}")
        ids.add(inv.get("id"))
        if inv.get("step") not in steps:
            probs.append(f"invariant {inv.get('id')!r} names unknown action {inv.get('step')!r}")
    for e in ch.get("entrenched") or []:
        if e not in ids:
            probs.append(f"entrenched invariant {e!r} is not defined")
    return probs


def _augment(p: PolicyProgram, charter: dict) -> PolicyProgram:
    """The program plus hidden guard steps for every domain and invariant condition, so the
    decomposition splits on them too. Hidden steps are never shown to an agent."""
    spec = p.to_dict()
    hidden = {}
    for d, ds in (charter.get("domains") or {}).items():
        hidden[f"__w_{d}"] = {"tool": f"__w_{d}", "when": ds.get("where", True), "args": {}}
    for inv in charter.get("invariants") or []:
        hidden[f"__i_{inv['id']}"] = {"tool": f"__i_{inv['id']}", "when": inv.get("when", True), "args": {}}
        if inv.get("kind") == "bound":
            hidden[f"__r_{inv['id']}"] = {"tool": f"__r_{inv['id']}", "when": True,
                                          "args": {"rhs": inv["rhs"]}}
    if HIDDEN in spec["clauses"]:
        raise ProgramError(f"clause id {HIDDEN!r} is reserved")
    spec["clauses"][HIDDEN] = {"text": "", "steps": hidden}
    if spec.get("procedure"):
        spec["procedure"] = list(spec["procedure"]) + list(hidden)
    spec.pop(CHARTER_KEY, None)
    return PolicyProgram(spec)


# ─────────────────────────────────────────────────────────────────────────────
# Effects and authority by effect
# ─────────────────────────────────────────────────────────────────────────────

def _effects_in_cell(cls: dict) -> list:
    out = []
    if cls["obligation"]:
        out.append(("obligation", cls["obligation"], ()))
    if cls["permission"]:
        out.append(("permission", cls["permission"], ()))
    if cls["content"]:
        for a in cls["differing_args"]:            # each changed argument is its own effect
            out.append(("content", "changed", (a,)))
    return out


def _covers(dspec: dict, step: str, channel: str, args: tuple, where_true: bool) -> bool:
    if step not in (dspec.get("actions") or []):
        return False
    if channel not in (dspec.get("channels") or list(CHANNELS)):
        return False
    if channel == "content" and dspec.get("args") is not None and not set(args) <= set(dspec["args"]):
        return False
    return where_true


def derive_effects(v1: PolicyProgram, v2: PolicyProgram, charter: dict) -> dict:
    """
    Every behavioral effect of v1 -> v2, region by region, and which charter
    domains cover it. Domain conditions are evaluated under v1.
    """
    a1, a2 = _augment(v1, charter), _augment(v2, charter)
    real = [k for k in v1.step_ids()] + [k for k in v2.step_ids() if k not in v1.step_ids()]
    hidden = [k for k in a1.step_ids() if k.startswith("__")]
    cells = decompose([a1, a2], real + hidden)
    redefined = v1.changed_names(v2)
    doms = charter.get("domains") or {}
    effects = []
    for cell, norms in cells:
        where = {d: norms[0][f"__w_{d}"].modality == OBLIGATORY for d in doms}
        for k in real:
            cls = classify(norms[0][k], norms[1][k], cell, redefined)
            for channel, change, args in _effects_in_cell(cls):
                covering = sorted(d for d, ds in doms.items() if _covers(ds, k, channel, args, where[d]))
                effects.append({"cell": cell, "step": k, "channel": channel, "change": change,
                                "args": list(args), "domains": covering})
    return {"cells": cells, "effects": effects, "real_steps": real, "augmented": (a1, a2)}


def _summarize(effects: list) -> list:
    """Group effects by (step, channel, change, args, domains) with readable conditions."""
    groups: dict = {}
    for e in effects:
        key = (e["step"], e["channel"], e["change"], tuple(e["args"]), tuple(e["domains"]))
        groups.setdefault(key, []).append(e["cell"])
    out = []
    for (step, ch, change, args, ds), cells in sorted(groups.items(), key=lambda t: t[0][:4]):
        out.append({"step": step, "channel": ch, "change": change, "args": list(args), "domains": list(ds),
                    "regions": len(cells), "example": cells[0].describe(), "example_situation": cells[0].rep()})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Invariants, proved over every region
# ─────────────────────────────────────────────────────────────────────────────

def _extremes(cell: Cell) -> list:
    """Situations at which a linear function on the cell attains its extremes (on the resolution grid)."""
    axes = []
    names = list(cell.box)
    for f in names:
        iv = cell.box[f]
        if iv.point:
            vals = [iv.lo]
        elif iv.step is None:
            vals = [iv.lo, iv.hi]          # closure; the supremum is approached
        else:
            k0, k1 = iv._grid_points()
            vals = sorted({iv.step * k0, iv.step * k1})
        axes.append(vals)
    out = []
    for combo in itertools.product(*axes) if axes else [()]:
        s = dict(cell.finite)
        for f, v in zip(names, combo):
            s[f] = int(v) if v.denominator == 1 else float(v)
        out.append((s, dict(zip(names, combo))))
    return out


_TOL = Fraction(1, 200)  # half a cent: amounts are compared at cent resolution


def check_invariants(v2: PolicyProgram, charter: dict, cells: list, index: int = 1) -> list:
    """Each invariant: proved over every region of v2, or violated with a witness situation."""
    out = []
    for inv in charter.get("invariants") or []:
        iid, step = inv["id"], inv["step"]
        violations, regions_checked = [], 0
        for cell, norms in cells:
            if norms[index][f"__i_{iid}"].modality != OBLIGATORY:
                continue
            regions_checked += 1
            n = norms[index][step]
            if inv.get("kind", "modality") == "modality":
                if n.modality not in inv["modality_in"]:
                    violations.append({"situation": cell.rep(), "condition": cell.describe(),
                                       "found": n.modality, "required": inv["modality_in"]})
                continue
            # bound
            if n.modality == FORBIDDEN:
                continue                     # no call, nothing to bound
            arg = n.args.get(inv["arg"])
            rhs = norms[index][f"__r_{iid}"].args.get("rhs")
            la, lr = _as_lin(arg), _as_lin(rhs)
            if la is None or lr is None:
                violations.append({"situation": cell.rep(), "condition": cell.describe(),
                                   "found": "non-numeric argument"})
                continue
            diff = la - lr
            for sit, point in _extremes(cell):
                v = diff.at(point)
                bad = ((inv["op"] == "<=" and v > _TOL) or (inv["op"] == ">=" and v < -_TOL)
                       or (inv["op"] == "==" and abs(v) > _TOL))
                if bad:
                    violations.append({"situation": sit, "condition": cell.describe(),
                                       "found": f"{inv['arg']} - ({inv.get('rhs_text', 'rhs')}) = {float(v):.4g}"})
                    break
        out.append({"id": iid, "text": inv.get("text", ""), "entrenched": iid in (charter.get("entrenched") or []),
                    "regions_checked": regions_checked, "holds": not violations,
                    "violations": violations[:5], "violation_count": len(violations)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Approvals and procedure
# ─────────────────────────────────────────────────────────────────────────────

def approve_pin(pv, key: dict, role: str):
    """Add an approval: the role's Ed25519 signature over the version's body."""
    from contradish import ed25519 as _ed
    msg = canonical_json(pv.body()).encode("utf-8")
    sig = _ed.sign(bytes.fromhex(key["seed"]), msg)
    pv.approvals = list(getattr(pv, "approvals", None) or []) + [
        {"role": role, "key": key["public"], "sig": sig.hex()}]
    return pv


def _valid_approvals(v1: PolicyProgram, v2pin) -> list:
    from contradish import ed25519 as _ed
    msg = canonical_json(v2pin.body()).encode("utf-8")
    roles = []
    signers = list(getattr(v2pin, "approvals", None) or [])
    if v2pin.signature and v2pin.issued_by:
        signers.append({"role": v2pin.issued_by, "key": v2pin.signature.get("key"), "sig": v2pin.signature.get("sig")})
    for a in signers:
        src = v1.sources.get(a.get("role"), {})
        if a.get("key") not in (src.get("keys") or []):
            continue
        try:
            if _ed.verify(bytes.fromhex(a["key"]), msg, bytes.fromhex(a["sig"])):
                roles.append(a["role"])
        except (KeyError, ValueError):
            continue
    return sorted(set(roles))


def _date(s: Optional[str]):
    return _dt.date.fromisoformat(s) if s else None


def _procedure(charter: dict, touched: set, approved: list, v1pin, v2pin, rules_key: str = "procedures") -> list:
    """`touched`: the set of licensing-domain tuples, one per distinct effect."""
    results = []
    rules = charter.get(rules_key) or []
    if rules_key == "amendment":
        rules = [dict(charter.get("amendment") or {}, id="charter-amendment", domains="*")]
    for r in rules:
        ds = r.get("domains")
        # A rule governs an effect when every domain that could license the effect is under the rule,
        # so an effect cannot route around a procedure through a laxer overlapping domain.
        applies = ds == "*" or any(lic and set(lic) <= set(ds or []) for lic in touched)
        if not applies:
            continue
        have = sorted(set(approved) & set(r.get("approvers") or []))
        need = int(r.get("approvals_min", 1))
        days_needed = int(r.get("review_days", 0))
        p0, e0 = _date(getattr(v2pin, "proposed_at", None)), _date(getattr(v2pin, "effective_at", None))
        review_ok = days_needed == 0 or (p0 is not None and e0 is not None and (e0 - p0).days >= days_needed)
        results.append({"id": r.get("id"), "approvals_required": need, "approvers": r.get("approvers"),
                        "approved_by": have, "approvals_ok": len(have) >= need,
                        "review_days_required": days_needed, "review_ok": review_ok,
                        "ok": len(have) >= need and review_ok})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# The legitimacy certificate
# ─────────────────────────────────────────────────────────────────────────────

def legitimacy_certificate(v1pin, v2pin, trust_anchors: Optional[list] = None) -> dict:
    """
    Is the change v1 -> v2 to the governing specification legitimate under
    v1's charter? Provenance, procedure, authority by effect, invariants
    (proved over every region of v2), and charter amendment rules.
    """
    from contradish.evidence import seal
    from contradish.versions import authenticate_transition, transition_authority
    p1, p2 = v1pin.program, v2pin.program
    charter = charter_of(p1)
    probs = validate_charter(p1)
    if probs:
        raise ProgramError("v1's charter is malformed: " + "; ".join(probs))
    auth = authenticate_transition(v1pin, v2pin, trust_anchors)
    eff = derive_effects(p1, p2, charter)
    issuer = v2pin.issued_by
    granted = set((charter.get("grants") or {}).get(issuer or "", []))
    unauth = [e for e in eff["effects"] if not (set(e["domains"]) & granted)]
    touched = {tuple(e["domains"]) for e in eff["effects"]}
    approved = _valid_approvals(p1, v2pin) if auth["v2_signed"] else []
    proc = _procedure(charter, touched, approved, v1pin, v2pin)
    e1, e2 = _date(getattr(v1pin, "effective_at", None)), _date(getattr(v2pin, "effective_at", None))
    retro_ok = not (charter.get("no_retroactivity") and e1 and e2 and e2 < e1)
    if charter.get("no_retroactivity") and (e2 is None):
        retro_ok = False
    # Invariants: v1's charter governs this change; check them over v2. Then
    # v2's own charter, if it changed, must keep the entrenched ones and pass amendment.
    inv = check_invariants(p2, charter, eff["cells"], index=1)
    baseline = check_invariants(p1, charter, eff["cells"], index=0)
    ch2 = charter_of(p2)
    amended = canonical_json(ch2) != canonical_json(charter)
    amend = None
    if amended:
        ent = set(charter.get("entrenched") or [])
        old = {i["id"]: i for i in charter.get("invariants") or []}
        new = {i["id"]: i for i in ch2.get("invariants") or []}
        removed = sorted(e for e in ent if e not in new or canonical_json(new[e]) != canonical_json(old[e])
                         or e not in (ch2.get("entrenched") or []))
        amend = {"amended": True, "entrenched_altered_or_removed": removed,
                 "procedure": _procedure(charter, set(), approved, v1pin, v2pin, rules_key="amendment")}
        amend["ok"] = not removed and all(r["ok"] for r in amend["procedure"])
    checks = {
        "provenance": auth["authenticated"],
        "procedure": all(r["ok"] for r in proc) and retro_ok,
        "authority_by_effect": not unauth,
        "invariants": all(i["holds"] for i in inv),
        "charter_amendment": True if amend is None else amend["ok"],
    }
    legitimate = all(checks.values())
    clause_level = transition_authority(v1pin, v2pin)
    doc = {
        "schema": LEGIT_SCHEMA,
        "before": v1pin.to_dict(),
        "after": v2pin.to_dict(),
        "charter_digest": "sha256:" + __import__("hashlib").sha256(canonical_json(charter).encode()).hexdigest(),
        "authentication": auth,
        "effects": {
            "regions": len(eff["cells"]),
            "summary": _summarize(eff["effects"]),
            "issuer": issuer,
            "granted_domains": sorted(granted),
            "unauthorized": _summarize(unauth),
            "touched_domains": sorted({d for t in touched for d in t}),
        },
        "procedure": {"approved_by": approved, "rules": proc, "no_retroactivity_ok": retro_ok,
                      "proposed_at": getattr(v2pin, "proposed_at", None),
                      "effective_at": getattr(v2pin, "effective_at", None)},
        "invariants": inv,
        "baseline_invariants_hold": {i["id"]: i["holds"] for i in baseline},
        "charter_amendment": amend,
        "clause_level_authority": clause_level,
        "checks": checks,
        "legitimate": legitimate,
        "statement": _statement(v1pin, v2pin, checks, unauth, inv, clause_level),
        "scope_note": ("Effects and invariants are derived over the entire declared situation space of both "
                       "versions (exact decomposition). Dates are the signers' statements, not trusted "
                       "timestamps. Trust anchors are the verifier's choice."),
        "attribution": ATTRIBUTION,
    }
    return seal(doc)


def _statement(v1pin, v2pin, checks, unauth, inv, clause_level) -> str:
    head = "LEGITIMATE" if all(checks.values()) else "NOT LEGITIMATE"
    bad = [k.replace("_", " ") for k, v in checks.items() if not v]
    s = f"{head}: change {v1pin.id} -> {v2pin.id} issued by {v2pin.issued_by!r}"
    if bad:
        s += " fails " + ", ".join(bad)
    if unauth and clause_level.get("status") == "authorized":
        s += ("; NOTE: the issuer owns every clause it edited (clause-level authority would pass), "
              f"but {len(unauth)} of the change's behavioral effects lie outside its granted domains")
    broken = [i["id"] for i in inv if not i["holds"]]
    if broken:
        s += "; violated invariants: " + ", ".join(broken)
    return s + "."


# ─────────────────────────────────────────────────────────────────────────────
# Built-in demonstrations (public fixture keys: they demonstrate, they secure nothing)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CASES = {
    "restocking_fee_20": ("restocking_fee_20", (), "2026-10-01", "2026-10-02",
                          "Policy owner raises the restocking fee: a pricing effect it is granted."),
    "pricing_prepaid_cut": ("pricing_prepaid_cut", (), "2026-10-01", "2026-10-02",
                            "Pricing team edits only a clause it owns (R5), but the effects reach return labels "
                            "and defective-goods refunds, and break an entrenched invariant."),
    "window_45_unapproved": ("window_45", (), "2026-10-01", "2026-10-02",
                             "Policy owner extends the window alone: an eligibility change needs two approvals "
                             "and a 7-day review."),
    "window_45": ("window_45", ("legal",), "2026-10-01", "2026-10-09",
                  "The same change with legal's approval and a 8-day review."),
    "defective_restocking": ("defective_restocking", ("legal", "consumer_protection"), "2026-10-01", "2026-10-09",
                             "Fully approved, but it charges a fee on defective goods: an entrenched invariant fails."),
    "charter_amendment": (None, ("legal",), "2026-10-01", "2026-10-31",
                          "Policy owner amends the charter to drop the final-sale invariant: entrenched, "
                          "and the amendment procedure needs all three roles."),
}


def demo_transition(case: str):
    """(v1, v2, trust_anchors, description) for a built-in case."""
    from contradish.policy_program import load_builtin_program
    from contradish.versions import fixture_key, pin, sign_pin
    p, U = load_builtin_program("returns")
    uid, approvals, proposed, effective, desc = DEMO_CASES[case]
    keys = {r: fixture_key(r) for r in ("policy_owner", "legal", "consumer_protection", "pricing_team")}
    v1 = sign_pin(pin(p, "returns-v1", "contradish fixture", "built-in example", effective_at="2026-10-01"),
                  keys["policy_owner"])
    if uid is None:
        spec = p.to_dict()
        ch = spec["charter"]
        ch["invariants"] = [i for i in ch["invariants"] if i["id"] != "INV-final-sale"]
        ch["entrenched"] = [e for e in ch["entrenched"] if e != "INV-final-sale"]
        q, issuer = PolicyProgram(spec), "policy_owner"
    else:
        u = U[uid]
        ok, _ = p.split_edits(u)
        q, issuer = p.apply(ok), u.source
    v2 = pin(q, f"returns-v2-{case}", "contradish fixture", "built-in example", issued_by=issuer,
             supersedes=v1.digest, proposed_at=proposed, effective_at=effective)
    sign_pin(v2, keys[issuer])
    for r in approvals:
        approve_pin(v2, keys[r], r)
    return v1, v2, [keys["policy_owner"]["public"]], desc
