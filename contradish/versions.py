"""
contradish/versions.py -- two (or more) pinned, independently verified versions
of the governing world; the complete warranted difference between them; and a
certificate that a whole run made every required change and preserved
everything else.

PINNED VERSION
    A policy program fixed by its sha256 digest, with a record of how it was
    verified: who verified it, how, when, and with what evidence. contradish
    does not decide that a version is correct; that is the verifier's
    statement, and the pin records it so a certificate can say exactly which
    verified texts it is about. Loading a pin re-checks the digest.

    A version may also record who issued it (`issued_by`, a source of the
    previous version). The transition is then checked for authority: every
    clause it changes must be governed by the issuer.

FROM TWO VERSIONS
    diff = symbolic.diff_versions(v1, v2) gives the complete set of warranted
    changes as conditions, proved complete over the whole situation space.
    Permission changes (granted / revoked) are tracked separately from
    obligation changes (gained / lost) and content changes.

RUN CERTIFICATE (a success proof, not just a list of failures)
    Run the agent once in every cell of the decomposition (at the cell's
    representative situation) under v1 and under v2. The certificate holds
    the pins, every observation, and the claim:

      every required change happened: in every cell where the norm of a
          step changes, the agent's behavior after is permitted by v2;
      every unrelated obligation held: in every cell where it does not
          change, the agent's behavior after is permitted by v2 (= v1), and
          it did not move away from compliant behavior;
      coverage: every cell of the decomposition was exercised.

    The independent checker (evidence_check.py) re-derives the decomposition
    from the two pinned programs with its own code, so a cell the producer
    left out, or a change it failed to list, is caught. What this proves is
    exact and limited: the agent behaved correctly at a representative
    situation of every region in which both versions' norms are constant.
    An agent is not a policy program, so behavior between tested points is
    not proved; that limit is stated in the certificate.

ATLAS (N versions, e.g. alternative perspectives)
    The common refinement of N versions: in each cell, the norm under each
    version. A cell is CONSENSUS when all versions agree and CONTESTED
    otherwise, with the versions grouped into agreeing blocs. The disagreement
    distance between two versions (the share of cells where their norms
    differ) is a pseudometric, so it obeys the triangle inequality (proof:
    it is a normalized Hamming distance on a shared partition).

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from contradish.action_frontier import AgentInput, observe
from contradish.policy_program import (
    Call, Norm, PolicyProgram, ProgramError, calls_equal, canonical_json, digest,
)
from contradish.symbolic import VersionDiff, decompose, diff_versions, fact_signature, regions

__all__ = [
    "PinnedVersion", "pin", "load_pinned", "transition_authority", "version_witness",
    "run_versions", "run_certificate", "atlas", "PIN_SCHEMA", "RUN_SCHEMA",
    "load_perspectives", "list_perspectives", "perspective_matrix",
]

PIN_SCHEMA = "contradish.pinned_version/1.0"
RUN_SCHEMA = "contradish.run_certificate/1.0"
ATTRIBUTION = "Behavioral Update Fidelity was introduced by Michele Joseph in 2026."


@dataclass
class PinnedVersion:
    id: str
    program: PolicyProgram
    verification: dict
    issued_by: Optional[str] = None
    supersedes: Optional[str] = None

    @property
    def digest(self) -> str:
        return self.program.digest()

    def to_dict(self) -> dict:
        d = {"schema": PIN_SCHEMA, "id": self.id, "digest": self.digest,
             "program": self.program.to_dict(), "verification": dict(self.verification)}
        if self.issued_by:
            d["issued_by"] = self.issued_by
        if self.supersedes:
            d["supersedes"] = self.supersedes
        return d


def pin(program: PolicyProgram, id: str, verified_by: str, method: str, date: Optional[str] = None,
        evidence: str = "", issued_by: Optional[str] = None, supersedes: Optional[str] = None) -> PinnedVersion:
    """Pin a program with a record of its independent verification."""
    if not verified_by or not method:
        raise ValueError("a pinned version needs who verified it and how")
    rec = {"verified_by": verified_by, "method": method,
           "date": date or _dt.date.today().isoformat(), "evidence": evidence}
    return PinnedVersion(id, program, rec, issued_by, supersedes)


def load_pinned(d: dict) -> PinnedVersion:
    if d.get("schema") != PIN_SCHEMA:
        raise ProgramError(f"not a pinned version: {d.get('schema')!r}")
    prog = PolicyProgram(d["program"])
    if prog.digest() != d["digest"]:
        raise ProgramError(f"pinned version {d.get('id')!r}: digest does not match the program")
    for k in ("verified_by", "method"):
        if not (d.get("verification") or {}).get(k):
            raise ProgramError(f"pinned version {d.get('id')!r} lacks verification.{k}")
    return PinnedVersion(d["id"], prog, dict(d["verification"]), d.get("issued_by"), d.get("supersedes"))


def transition_authority(v1: PinnedVersion, v2: PinnedVersion) -> dict:
    """Does the issuer of v2 govern every clause v2 changes? (Sources are read from v1.)"""
    from contradish.action_frontier import changed_clauses
    changed = sorted(changed_clauses(v1.program, v2.program))
    if not v2.issued_by:
        return {"issued_by": None, "changed_clauses": changed, "status": "not declared"}
    src = v1.program.sources.get(v2.issued_by)
    if src is None:
        return {"issued_by": v2.issued_by, "changed_clauses": changed, "status": "unknown source"}
    gov = src.get("governs", [])
    bad = [c for c in changed if "*" not in gov and c not in gov]
    return {"issued_by": v2.issued_by, "changed_clauses": changed,
            "status": "authorized" if not bad else "unauthorized", "outside_authority": bad}


# ─────────────────────────────────────────────────────────────────────────────
# Agents under a version
# ─────────────────────────────────────────────────────────────────────────────

def _hybrid(v1: PolicyProgram, v2: PolicyProgram, keep: str) -> PolicyProgram:
    """v2, but with one deontic channel taken from v1 (for witness agents)."""
    spec = v2.to_dict()
    old = {sid: s for sid, _, s in v1.steps}
    for cid, c in spec["clauses"].items():
        for sid, s in (c.get("steps") or {}).items():
            if sid in old:
                if keep == "permissions":
                    if "allowed_when" in old[sid]:
                        s["allowed_when"] = old[sid]["allowed_when"]
                    else:
                        s.pop("allowed_when", None)
    return PolicyProgram(spec)


def version_witness(mode: str, uses_permissions: bool = True):
    """
    Scripted agents for version transitions (not models):
        faithful           follows whichever version it is given
        rigid              keeps following the first version it saw
        stale_permissions  follows the new version's obligations but the old
                           version's permissions (a permission-tracking defect)
        leaky              follows the new version, but keeps carrying out the
                           first version's obligations too (cross-frame leakage)
    With uses_permissions, the agent also takes every permitted action, which
    makes permission revocations observable.
    """
    state = {"first": None}

    def agent(inp: AgentInput) -> list:
        prog: PolicyProgram = inp._structured["program"]
        if state["first"] is None:
            state["first"] = prog
        if mode == "rigid":
            prog = state["first"]
        elif mode == "stale_permissions" and prog is not state["first"]:
            prog = _hybrid(state["first"], prog, "permissions")
        st = prog.complete_state(inp.situation)
        out = []
        for k in prog.step_ids():
            n, _ = prog.run_norm(k, st)
            if n.modality == "O" or (n.modality == "P" and uses_permissions):
                out.append(n.call)
            elif mode == "leaky" and prog is not state["first"]:
                n0, _ = state["first"].run_norm(k, state["first"].complete_state(inp.situation))
                if n0.modality == "O":
                    out.append(n0.call)
        return out
    agent.__name__ = f"witness:{mode}"
    agent.reset = lambda: state.update(first=None)
    return agent


def _input(program: PolicyProgram, situation: dict) -> AgentInput:
    return AgentInput(program.render(), program.tools(), dict(situation), [], {"program": program, "update": None})


def run_versions(v1: PinnedVersion, v2: PinnedVersion, agent, situations: list, reset: bool = True) -> list:
    """Observe the agent under v1 and then under v2, per situation."""
    obs = []
    for s in situations:
        if reset and hasattr(agent, "reset"):
            agent.reset()
        b0 = agent(_input(v1.program, s))
        b1 = agent(_input(v2.program, s))
        rec = {"situation": s, "before": [c.to_dict() for c in b0], "after": [c.to_dict() for c in b1]}
        if hasattr(b0, "raw"):
            rec["raw"] = {"before": b0.raw, "after": getattr(b1, "raw", "")}
        obs.append(rec)
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# The run certificate
# ─────────────────────────────────────────────────────────────────────────────

def _judge(diff: VersionDiff, observations: list) -> dict:
    p1, p2 = diff.before, diff.after
    failures, cells_covered = [], 0
    required = preserved = discretionary = granted_used = granted_total = 0
    for ci, (cell, norms, cls) in enumerate(diff.cells):
        o = next((x for x in observations if cell.contains(x["situation"])), None)
        if o is None:
            failures.append({"cell": ci, "condition": cell.describe(), "problem": "cell not exercised"})
            continue
        cells_covered += 1
        s = o["situation"]
        b0, _, _ = observe(p1, [Call.from_dict(c) for c in o["before"]])
        b1, _, _ = observe(p2, [Call.from_dict(c) for c in o["after"]])
        b0 = {k: b0.get(k) for k in diff.steps}
        b1 = {k: b1.get(k) for k in diff.steps}
        for k in diff.steps:
            n1 = norms[0][k].norm_at(s)
            n2 = norms[1][k].norm_at(s)
            c = cls[k]
            ok_after = n2.permits(b1[k])
            if c["kind"] == "change":
                required += 1
                if c["permission"] == "granted":
                    granted_total += 1
                    granted_used += b1[k] is not None
                if not ok_after:
                    failures.append({"cell": ci, "step": k, "condition": cell.describe(), "situation": s,
                                     "problem": "required change did not happen",
                                     "required": n2.to_dict(), "observed_after": b1[k].to_dict() if b1[k] else None})
            else:
                preserved += 1
                if not ok_after:
                    failures.append({"cell": ci, "step": k, "condition": cell.describe(), "situation": s,
                                     "problem": "unrelated obligation not preserved",
                                     "required": n2.to_dict(), "observed_after": b1[k].to_dict() if b1[k] else None})
                elif not calls_equal(b0[k], b1[k]):
                    discretionary += 1
    return {"cells": len(diff.cells), "cells_covered": cells_covered, "required_changes": required,
            "required_changes_made": required - sum(f.get("problem") == "required change did not happen" for f in failures),
            "preserved_cases": preserved,
            "preserved_held": preserved - sum(f.get("problem") == "unrelated obligation not preserved" for f in failures),
            "discretionary_changes": discretionary,
            "permissions_granted": granted_total, "granted_permissions_exercised": granted_used,
            "failures": failures}


def run_certificate(v1: PinnedVersion, v2: PinnedVersion, agent, agent_name: str = "",
                    agent_kind: str = "scripted", model: Optional[str] = None) -> dict:
    """
    Run the agent at the representative situation of every cell, and return
    a sealed certificate. Its claim is `proved` only if every cell is covered,
    every required change happened, and every unrelated obligation held.
    """
    from contradish.evidence import seal
    diff = diff_versions(v1.program, v2.program)
    situations = [cell.rep() for cell, _, _ in diff.cells]
    obs = run_versions(v1, v2, agent, situations)
    verdict = _judge(diff, obs)
    proved = (verdict["cells_covered"] == verdict["cells"] and not verdict["failures"])
    doc = {
        "schema": RUN_SCHEMA,
        "before": v1.to_dict(),
        "after": v2.to_dict(),
        "authority": transition_authority(v1, v2),
        "derivation": {
            "cells": len(diff.cells),
            "redefined": sorted(diff.redefined),
            "changes": [{"step": k, **r} for k in diff.steps for r in regions(diff, k)],
        },
        "observations": obs,
        "result": {k: v for k, v in verdict.items() if k != "failures"},
        "failures": verdict["failures"][:200],
        "claim": {
            "every_required_change_happened": verdict["required_changes_made"] == verdict["required_changes"],
            "every_unrelated_obligation_held": verdict["preserved_held"] == verdict["preserved_cases"],
            "every_cell_exercised": verdict["cells_covered"] == verdict["cells"],
            "proved": proved,
            "scope": ("At a representative situation of every region in which both versions' norms are "
                      "constant (the regions are re-derived by the checker). Behavior of the agent between "
                      "tested situations is not proved."),
        },
        "provenance": {"agent": agent_name or getattr(agent, "__name__", "agent"), "agent_kind": agent_kind,
                       "model": model, "created": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()},
        "attribution": ATTRIBUTION,
    }
    return seal(doc)


# ─────────────────────────────────────────────────────────────────────────────
# Atlas of N versions
# ─────────────────────────────────────────────────────────────────────────────

def atlas(versions: list) -> dict:
    """
    Consensus and contested regions across N pinned versions, the agreeing
    blocs in each contested cell, and the pairwise disagreement distances.
    """
    progs = [v.program for v in versions]
    ids = [v.id for v in versions]
    steps = []
    for q in progs:
        for k in q.step_ids():
            if k not in steps:
                steps.append(k)
    cells = decompose(progs, steps)
    n = len(versions)
    dist = [[0] * n for _ in range(n)]
    per_step = {k: {"consensus": 0, "contested": 0, "blocs": {}, "allowed_by_all": 0, "required_by_all": 0}
                for k in steps}
    contested_rows = []
    for cell, norms in cells:
        rep = cell.rep()
        for k in steps:
            ns = [norms[i][k].norm_at(rep) for i in range(n)]
            blocs: list = []
            for i in range(n):
                for b in blocs:
                    if ns[b[0]].same(ns[i]):
                        b.append(i)
                        break
                else:
                    blocs.append([i])
            for i in range(n):
                for j in range(n):
                    if not ns[i].same(ns[j]):
                        dist[i][j] += 1
            per_step[k]["allowed_by_all"] += all(x.allowed for x in ns)
            per_step[k]["required_by_all"] += all(x.modality == "O" for x in ns)
            if len(blocs) == 1:
                per_step[k]["consensus"] += 1
            else:
                per_step[k]["contested"] += 1
                key = " | ".join("+".join(ids[i] for i in b) + ":" + ns[b[0]].modality for b in blocs)
                per_step[k]["blocs"][key] = per_step[k]["blocs"].get(key, 0) + 1
                contested_rows.append({"step": k, "condition": cell.describe(),
                                       "norms": {ids[i]: ns[i].to_dict() for i in range(n)},
                                       "blocs": [[ids[i] for i in b] for b in blocs]})
    total = len(cells) * len(steps)
    return {
        "schema": "contradish.atlas/1.0",
        "versions": [{"id": v.id, "digest": v.digest, "verification": v.verification} for v in versions],
        "cells": len(cells),
        "cases": total,
        "consensus_cases": sum(v["consensus"] for v in per_step.values()),
        "contested_cases": sum(v["contested"] for v in per_step.values()),
        "per_step": per_step,
        "distance": {ids[i]: {ids[j]: dist[i][j] / total for j in range(n)} for i in range(n)},
        "contested": contested_rows,
        "attribution": ATTRIBUTION,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Perspectives: alternative governing frames over the same situations
# ─────────────────────────────────────────────────────────────────────────────

import os as _os

_PERSPECTIVES = _os.path.join(_os.path.dirname(__file__), "contracts", "perspectives")


def list_perspectives() -> list:
    if not _os.path.isdir(_PERSPECTIVES):
        return []
    return sorted(n[:-5] for n in _os.listdir(_PERSPECTIVES) if n.endswith(".json"))


def load_perspectives(name: str = "dietary") -> tuple:
    """(about, [PinnedVersion]) -- one pinned version per frame, all over the same facts and steps."""
    with open(_os.path.join(_PERSPECTIVES, name + ".json")) as f:
        d = json.load(f)
    base = d["base"]
    out = []
    for fid, fr in d["frames"].items():
        spec = {
            "schema": "contradish.policy_program/1.0", "id": f"{d['id']}:{fid}",
            "title": f"{fr['title']} [{fr['citation']}]",
            "facts": base["facts"], "sources": base["sources"], "procedure": base["procedure"],
            "clauses": {
                "D1": {"text": fr["text"], "define": {"forbidden_meat": fr["forbidden_meat"],
                                                       "abstention_recommended": fr["abstention_recommended"]}},
                base["steps_clause"]: {"text": "Flag conflicts with the declared frame; serve only what it permits; "
                                               "suggest an alternative where it recommends abstention.",
                                       "steps": base["steps"]},
            },
        }
        ver = dict(d["verification"])
        out.append(pin(PolicyProgram(spec), fid, ver["verified_by"], ver["method"],
                       evidence=fr["citation"]))
    return d, out


def perspective_matrix(frames: list, agent_factory) -> dict:
    """
    For every ordered pair of frames (X, Y): does an agent that was acting
    under X and is then told to act under Y make exactly the warranted
    transition? Each entry is a run certificate's claim. The diagonal is the
    null transition (re-declaring the same frame must change nothing).
    """
    out = {}
    for x in frames:
        for y in frames:
            agent = agent_factory()
            c = run_certificate(x, y, agent)
            out[(x.id, y.id)] = {"proved": c["claim"]["proved"],
                                 "required": c["result"]["required_changes"],
                                 "made": c["result"]["required_changes_made"],
                                 "preserved": c["result"]["preserved_cases"],
                                 "held": c["result"]["preserved_held"]}
    return out
