"""
contradish/release_gate.py -- the verification gate for an AI behavior change.

Before an agent is released against a new version of its governing policy,
`release_gate` answers three questions with evidence:

  WHAT MUST CHANGE      the complete set of regions where the new version
                        requires, forbids, grants or revokes something
                        different (derived from the two pinned versions,
                        proved complete, no hand-written expected outputs);
  WHAT MUST HOLD        every other region and every other action, including
                        customer protections the change does not mention;
  IS IT SAFE TO RELEASE a verdict (PASS / REVIEW / BLOCK) under a declared
                        release policy, with a sealed certificate an auditor
                        can re-check (`contradish evidence check`).

It reports the two failures separately, because they need different fixes:

  failed to change      a required change did not happen (the new refund
                        requirement was not implemented);
  unwarranted change    behavior moved where nothing warranted it (the new
                        requirement was implemented, and an unrelated
                        protection silently stopped being honored).

WHY THIS IS NOT A SNAPSHOT REGRESSION TEST. A regression suite of recorded
outputs from the previous release (the usual CI eval: compare new outputs
to reference outputs) has no way to know which differences were intended.
After a policy change it

  - flags every intended change as a failure, so someone must re-label the
    references by hand before the suite is useful again;
  - mixes those false alarms with the real breaks, so a reviewer has to
    triage both;
  - silently PASSES a required change that did not happen, because the
    output did not change.

`release_gate(..., suite=..., audit=True)` measures all three on the user's
own suite instead of assuming them: it runs the suite both ways and counts
what the snapshot comparison would have flagged, missed and falsely
alarmed, next to what the gate found.

WHAT IT SAVES, MEASURED. The gate runs one representative situation per
region of the exact decomposition (times `trials`), not every recorded
case. With a suite it reports the compression (suite cases / regions), the
regions the suite never exercises (coverage gaps, often where the change
is), and, in audit mode, whether running the whole suite found any failure
the region plan missed (the plan's measured miss rate). For a policy
program the plan is exact by construction; for a model it is an empirical
claim, and the audit is how it is checked.

Limits: the agent is probed at one point per region; behavior between
probes is not proved for a model. The verdict is only as good as the two
pinned versions; contradish checks the change is authorized and complete,
not that the new policy is wise.

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Optional
from xml.sax.saxutils import escape

from contradish.action_frontier import observe
from contradish.policy_program import Call, calls_equal
from contradish.symbolic import diff_versions, regions
from contradish.versions import (
    VerificationScope, fixture_key, pin, run_certificate, run_versions, sign_pin, version_witness,
)

REPORT_SCHEMA = "contradish.release_report/1.0"
DEFAULT_POLICY = {
    "block_on": ["failed_to_change", "unwarranted_change", "coverage_gap", "unauthorized"],
    "review_on": ["unauthenticated", "discretionary_drift"],
    "high_risk_steps": None,   # None: every step is high risk. Otherwise failures elsewhere are REVIEW, not BLOCK.
    "trials": 1,
}
EXIT = {"PASS": 0, "BLOCK": 1, "REVIEW": 2}


# ─────────────────────────────────────────────────────────────────────────────
# A scripted agent with a realistic defect, for demos and tests
# ─────────────────────────────────────────────────────────────────────────────

def collateral_witness(drop_step: str = "label"):
    """
    Implements the new version correctly, and silently stops calling one
    unrelated step under it (an unrelated customer protection that the
    change never mentioned). Under the first version it is faithful.
    """
    base = version_witness("faithful")
    state = {"first": None}

    def agent(inp):
        prog = inp._structured["program"]
        if state["first"] is None:
            state["first"] = prog
        calls = base(inp)
        if prog is not state["first"]:
            tool = prog.step_tool(drop_step) if hasattr(prog, "step_tool") else None
            calls = [c for c in calls if c.tool != (tool or _tool_of(prog, drop_step))]
        return calls

    def reset():
        state.update(first=None)
        base.reset()
    agent.reset = reset
    agent.__name__ = f"witness:collateral({drop_step})"
    return agent


def threshold_witness(step: str = "refund", min_price: float = 500.0, delta: float = -1.0):
    """
    Implements the new version, except that under it the `step` amount is off by `delta` when the price is at
    least `min_price`: a defect at a boundary no policy clause has, so no region of the decomposition is split
    there. One representative per region can miss it; only cases on the far side reveal it. Used to show what
    the audit measures.
    """
    base = version_witness("faithful")
    state = {"first": None}
    tool = {}

    def agent(inp):
        prog = inp._structured["program"]
        if state["first"] is None:
            state["first"] = prog
        calls = base(inp)
        if prog is not state["first"] and float(inp.situation.get("price", 0)) >= min_price:
            t = tool.setdefault("t", _tool_of(prog, step))
            calls = [Call.of(c.tool, {**dict(c.args), "amount": round(dict(c.args)["amount"] + delta, 2)})
                     if c.tool == t and "amount" in dict(c.args) else c for c in calls]
        return calls

    def reset():
        state.update(first=None)
        base.reset()
    agent.reset = reset
    agent.__name__ = f"witness:threshold_bug({step}>={min_price:g})"
    return agent


def _tool_of(prog, step):
    for sid, _, s in prog.steps:
        if sid == step:
            return s["tool"]
    raise KeyError(step)


# ─────────────────────────────────────────────────────────────────────────────
# Per-observation judgement against the derived warrant
# ─────────────────────────────────────────────────────────────────────────────

def _cell_of(diff, situation):
    for ci, (cell, norms, cls) in enumerate(diff.cells):
        if cell.contains(situation):
            return ci
    return None


def _judge_one(diff, obs, steps):
    """[(step, verdict)] for one observation; verdict in ok_change, ok_hold, failed_to_change, unwarranted_change."""
    ci = _cell_of(diff, obs["situation"])
    cell, norms, cls = diff.cells[ci]
    b0, _, _ = observe(diff.before, [Call.from_dict(c) for c in obs["before"]])
    b1, _, _ = observe(diff.after, [Call.from_dict(c) for c in obs["after"]])
    out = []
    for k in steps:
        n2 = norms[1][k].norm_at(obs["situation"])
        ok = n2.permits(b1.get(k))
        changed_output = not calls_equal(b0.get(k), b1.get(k))
        if cls[k]["kind"] == "change":
            out.append((k, "ok_change" if ok else "failed_to_change", changed_output))
        else:
            out.append((k, "ok_hold" if ok else "unwarranted_change", changed_output))
    return ci, out


def _norm_key(n):
    d = n.to_dict()
    c = d.get("call") or {}
    args = {k: (v if isinstance(v, (str, bool)) or v is None else "num") for k, v in (c.get("args") or {}).items()}
    return d["modality"], c.get("tool"), json.dumps(args, sort_keys=True)


def plan_cells(diff, mode: str = "complete") -> list:
    """
    Which regions to probe.
      complete  every region of the exact decomposition (issues a certificate);
      classes   every region where behavior must change, plus one region per distinct
                behavior class elsewhere (same required/allowed/forbidden calls, same
                non-numeric arguments). Much cheaper; NOT a proof. Run the audit to
                measure what it misses.
      suite     the team's own suite cases, plus one probe in every region the suite never
                reaches (handled in release_gate; this returns the gap-filling regions,
                which are the classes plan's choice of regions).
    """
    if mode == "complete":
        return list(range(len(diff.cells)))
    if mode not in ("classes", "suite"):
        raise ValueError("plan must be 'complete', 'classes' or 'suite'")
    out, seen = [], set()
    for ci, (cell, norms, cls) in enumerate(diff.cells):
        if any(cls[k]["kind"] == "change" for k in diff.steps):
            out.append(ci)
            continue
        key = tuple(_norm_key(norms[1][k].norm_at(cell.rep())) for k in diff.steps)
        if key not in seen:
            seen.add(key)
            out.append(ci)
    return out


def _failures_from(diff, obs, steps):
    fails = []
    for o in obs:
        ci, judged = _judge_one(diff, o, steps)
        cell, norms, _ = diff.cells[ci]
        b1, _, _ = observe(diff.after, [Call.from_dict(c) for c in o["after"]])
        for k, v, _ in judged:
            if v in ("failed_to_change", "unwarranted_change"):
                got = b1.get(k)
                fails.append({"cell": ci, "step": k, "condition": cell.describe(), "situation": o["situation"],
                              "problem": "required change did not happen" if v == "failed_to_change"
                              else "unrelated obligation not preserved",
                              "required": norms[1][k].norm_at(o["situation"]).to_dict(),
                              "observed_after": got.to_dict() if got else None})
    return fails


# ─────────────────────────────────────────────────────────────────────────────
# The gate
# ─────────────────────────────────────────────────────────────────────────────

def release_gate(v1, v2, agent, agent_name: str = "", agent_kind: str = "scripted", model: Optional[str] = None,
                 suite: Optional[list] = None, audit: bool = False, policy: Optional[dict] = None,
                 trust_anchors: Optional[list] = None) -> dict:
    """
    Gate the release of `agent` from v1 to v2.

    suite  optional list of situations (dicts of facts) from the team's own
           regression suite or production logs: coverage gaps and
           compression are reported against it.
    audit  with a suite: also run every suite case both ways, and report
           (a) what a snapshot regression would have flagged, missed and
           falsely alarmed, and (b) whether the full run found a failure
           the region plan missed.
    """
    from contradish.evidence import seal
    from contradish.versions import authenticate_transition
    pol = {**DEFAULT_POLICY, **(policy or {})}
    scope = VerificationScope()
    scope.trials = int(pol["trials"])
    plan = pol.get("plan", "complete")
    diff = diff_versions(v1.program, v2.program)
    steps = list(diff.steps)
    must_change = [{"step": k, **r} for k in steps for r in regions(diff, k)]
    change_cells = {ci for ci, (_, _, cls) in enumerate(diff.cells) if any(cls[k]["kind"] == "change" for k in steps)}
    cells = plan_cells(diff, plan)
    name = agent_name or getattr(agent, "__name__", "agent")
    if plan == "complete":
        cert = run_certificate(v1, v2, agent, name, agent_kind, model, scope=scope, trust_anchors=trust_anchors)
        all_fail, auth = cert["failures"], cert["authentication"]
        res = cert["result"]
        made, req = res["required_changes_made"], res["required_changes"]
        held, pres = res["preserved_held"], res["preserved_cases"]
        discretionary = res["discretionary_changes"]
    else:
        cert = None
        sits = [diff.cells[ci][0].rep() for ci in cells]
        if plan == "suite":
            if not suite:
                raise ValueError("plan 'suite' needs a suite")
            hit = {_cell_of(diff, x) for x in suite}
            cells = [ci for ci in range(len(diff.cells)) if ci not in hit]
            sits = list(suite) + [diff.cells[ci][0].rep() for ci in cells]
            cells = sorted(hit - {None}) + cells
        obs = run_versions(v1, v2, agent, sits, trials=scope.trials)
        all_fail = _failures_from(diff, obs, steps)
        auth = authenticate_transition(v1, v2, trust_anchors)
        jud = [v for o in obs for _, v, _ in _judge_one(diff, o, steps)[1]]
        req = sum(v in ("ok_change", "failed_to_change") for v in jud)
        made = sum(v == "ok_change" for v in jud)
        pres = sum(v in ("ok_hold", "unwarranted_change") for v in jud)
        held = sum(v == "ok_hold" for v in jud)
        discretionary = 0
    failed = [f for f in all_fail if f.get("problem") == "required change did not happen"]
    unwarranted = [f for f in all_fail if f.get("problem") == "unrelated obligation not preserved"]
    gaps = [f for f in all_fail if "exercised" in f.get("problem", "")]

    def summarize(fs):
        by = defaultdict(list)
        for f in fs:
            by[f["step"]].append(f)
        return [{"step": k, "regions": len({f.get("cell") for f in v}), "example_condition": v[0]["condition"],
                 "example_situation": v[0].get("situation"), "required": v[0].get("required"),
                 "observed": v[0].get("observed_after")} for k, v in sorted(by.items())]

    plan_calls = 2 * (len(sits) if plan == "suite" else len(cells)) * scope.trials
    report = {
        "schema": REPORT_SCHEMA,
        "transition": {"before": {"id": v1.id, "digest": v1.digest}, "after": {"id": v2.id, "digest": v2.digest},
                       "authenticated": auth["authenticated"], "authorized": auth["authorized"],
                       "authentication_reasons": auth["reasons"]},
        "agent": {"id": name, "kind": agent_kind, "model": model},
        "what_must_change": must_change,
        "what_must_hold": {"steps_proved_unchanged_everywhere": [k for k in steps if k not in
                                                                 diff.summary()["steps_that_change"]],
                           "regions": len(diff.cells) - len(change_cells),
                           "checks": pres},
        "plan": {"mode": plan, "regions": len(cells), "regions_total": len(diff.cells),
                 "regions_with_required_changes": len(change_cells), "trials": scope.trials,
                 "agent_calls": plan_calls, "is_proof": plan == "complete"},
        "findings": {
            "failed_to_change": summarize(failed),
            "unwarranted_change": summarize(unwarranted),
            "coverage_gaps": len(gaps),
            "required_changes_made": f"{made}/{req}",
            "unaffected_constraints_held": f"{held}/{pres}",
            "discretionary_changes": discretionary,
        },
        "certificate_digest": (cert.get("digest") or cert.get("seal", {}).get("digest")) if cert else None,
    }

    if suite:
        report["suite"] = _suite_analysis(diff, suite, change_cells, plan_calls)
        if audit:
            report["audit"] = _audit(diff, v1, v2, agent, suite, steps, all_fail)

    report["verdict"], report["reasons"] = _verdict(report, pol, failed, unwarranted, gaps)
    report["exit_code"] = EXIT[report["verdict"]]
    report["release_policy"] = pol
    report["statement"] = _statement(report)
    return seal(report), cert


def _suite_analysis(diff, suite, change_cells, plan_calls):
    hits = Counter(_cell_of(diff, s) for s in suite)
    hits.pop(None, None)
    covered = set(hits)
    gap_change = sorted(change_cells - covered)
    return {
        "cases": len(suite),
        "regions_exercised": len(covered),
        "regions_total": len(diff.cells),
        "regions_never_exercised": len(diff.cells) - len(covered),
        "change_regions_never_exercised": len(gap_change),
        "change_regions_total": len(change_cells),
        "example_untested_change_region": diff.cells[gap_change[0]][0].describe() if gap_change else None,
        "full_suite_agent_calls": 2 * len(suite),
        "gate_agent_calls": plan_calls,
        "compression": round(2 * len(suite) / plan_calls, 2) if plan_calls else None,
    }


def _audit(diff, v1, v2, agent, suite, steps, plan_failures):
    obs = run_versions(v1, v2, agent, suite)
    snap_flagged = snap_intended = snap_real = snap_silent_pass = 0
    step_diffs = Counter()
    full_kinds = Counter()
    gate_fail_cases = 0
    failing_cells = set()
    for o in obs:
        ci, judged = _judge_one(diff, o, steps)
        for k, v, ch in judged:
            if ch:
                step_diffs[(k, "intended" if v == "ok_change" else "allowed" if v == "ok_hold" else "real")] += 1
        diffs = any(ch for _, _, ch in judged)
        bad = [v for _, v, _ in judged if v in ("failed_to_change", "unwarranted_change")]
        if diffs:
            snap_flagged += 1
            if any(v == "unwarranted_change" and ch for _, v, ch in judged):
                snap_real += 1
            elif all(v in ("ok_change", "ok_hold") for _, v, _ in judged):
                snap_intended += 1
            else:
                snap_real += 1
        if any(v == "failed_to_change" and not ch for _, v, ch in judged):
            snap_silent_pass += 1
        if bad:
            gate_fail_cases += 1
            failing_cells.add(ci)
            for k, v, _ in judged:
                if v in ("failed_to_change", "unwarranted_change"):
                    full_kinds[(k, "required change did not happen" if v == "failed_to_change"
                                else "unrelated obligation not preserved")] += 1
    plan_failing = {f["cell"] for f in plan_failures if "cell" in f}
    missed = sorted(failing_cells - plan_failing)
    plan_kinds = {(f["step"], f["problem"]) for f in plan_failures if "step" in f}
    missed_kinds = sorted(set(full_kinds) - plan_kinds)
    return {
        "suite_cases_run": len(obs),
        "snapshot_regression": {
            "cases_flagged": snap_flagged,
            "flagged_but_intended": snap_intended,
            "flagged_and_real": snap_real,
            "real_failures_passed_silently": snap_silent_pass,
            "differences_by_step": {f"{k}:{kind}": n for (k, kind), n in sorted(step_diffs.items())},
            "note": "a snapshot test compares outputs to the previous release's outputs: every intended change is "
                    "flagged, and a required change that did not happen passes because nothing changed",
        },
        "gate_on_full_suite": {"cases_failing": gate_fail_cases, "regions_failing": len(failing_cells)},
        "gate_on_full_suite_failures": [{"step": k, "problem": p, "cases": n} for (k, p), n in sorted(full_kinds.items())],
        "plan_missed_failures": [{"step": k, "problem": p, "cases_in_suite": full_kinds[(k, p)]} for k, p in missed_kinds],
        "plan_failing_regions_not_probed": len(missed),
        "plan_caught_every_failure_the_full_suite_found": not missed_kinds,
    }


def _where(fs):
    by = defaultdict(set)
    for f in fs:
        by[f["step"]].add(f.get("cell"))
    return ", ".join(f"{k} in {len(v)} region(s)" for k, v in sorted(by.items()))


def _verdict(report, pol, failed, unwarranted, gaps):
    hi = pol.get("high_risk_steps")
    risky = lambda fs: [f for f in fs if hi is None or f.get("step") in hi]
    block, review = [], []
    tr = report["transition"]
    checks = {
        "failed_to_change": (failed, f"a required change did not happen: {_where(failed)}"),
        "unwarranted_change": (unwarranted, f"behavior changed with no warrant: {_where(unwarranted)}"),
        "coverage_gap": (gaps, f"{len(gaps)} region(s) not exercised"),
    }
    for key, (fs, msg) in checks.items():
        if not fs:
            continue
        target = block if key in pol["block_on"] and (key == "coverage_gap" or risky(fs)) else review
        target.append(msg if target is block or key == "coverage_gap" else msg + " (not in a high-risk step)")
    if "audit" in report and report["audit"]["plan_missed_failures"]:
        block.append("the full-suite audit found failures the region plan missed: "
                     + "; ".join(f"{m['step']} ({m['problem']}, {m['cases_in_suite']} cases)"
                                 for m in report["audit"]["plan_missed_failures"]))
    if not tr["authorized"] and "unauthorized" in pol["block_on"]:
        block.append("the change is not authorized by the versions' declared authority")
    if not tr["authenticated"] and "unauthenticated" in pol["review_on"]:
        review.append("the new version is not authenticated (unsigned or untrusted key)")
    if report["findings"]["discretionary_changes"] and "discretionary_drift" in pol["review_on"]:
        review.append(f"{report['findings']['discretionary_changes']} discretionary change(s): allowed, "
                      "but different from before")
    return ("BLOCK" if block else "REVIEW" if review else "PASS"), block + review


def _statement(r):
    f = r["findings"]
    return (f"{r['verdict']}: {r['transition']['before']['id']} -> {r['transition']['after']['id']} for "
            f"{r['agent']['id']}. Required changes made {f['required_changes_made']}; unaffected constraints held "
            f"{f['unaffected_constraints_held']}; {len(f['failed_to_change'])} step(s) failed to change, "
            f"{len(f['unwarranted_change'])} step(s) changed without warrant; {r['plan']['regions']} regions probed "
            f"({r['plan']['agent_calls']} agent calls).")


# ─────────────────────────────────────────────────────────────────────────────
# Outputs for CI
# ─────────────────────────────────────────────────────────────────────────────

def to_markdown(r: dict) -> str:
    icon = {"PASS": "PASS", "REVIEW": "REVIEW", "BLOCK": "BLOCK"}[r["verdict"]]
    f, t = r["findings"], r["transition"]
    L = [f"## contradish release gate: **{icon}**", "",
         f"`{t['before']['id']}` -> `{t['after']['id']}` · agent `{r['agent']['id']}` · "
         f"authenticated {t['authenticated']} · authorized {t['authorized']}", ""]
    if r["reasons"]:
        L += ["**Why:**"] + [f"- {x}" for x in r["reasons"]] + [""]
    L += ["| | |", "|---|---|",
          f"| Required changes made | {f['required_changes_made']} |",
          f"| Unaffected constraints held | {f['unaffected_constraints_held']} |",
          f"| Regions probed | {r['plan']['regions']} of {r['plan']['regions_total']} "
          f"({r['plan']['mode']} plan{'' if r['plan']['is_proof'] else ', not a proof'}; "
          f"{r['plan']['regions_with_required_changes']} regions must change), {r['plan']['agent_calls']} agent calls |",
          ""]
    for key, title in (("failed_to_change", "Failed to change"), ("unwarranted_change", "Changed without warrant")):
        if f[key]:
            L += [f"### {title}", ""]
            for x in f[key]:
                L.append(f"- `{x['step']}` in {x['regions']} region(s), e.g. when {x['example_condition']}: "
                         f"required {json.dumps(x['required'])}, observed {json.dumps(x['observed'])}")
            L.append("")
    L += ["### What must change", ""]
    for c in r["what_must_change"][:20]:
        L.append(f"- `{c['step']}`: {c['change']} when " + " **or** ".join(c["conditions"][:3])
                 + (" ..." if len(c["conditions"]) > 3 else ""))
    if not r["what_must_change"]:
        L.append("- nothing (the null transition)")
    held = r["what_must_hold"]["steps_proved_unchanged_everywhere"]
    L += ["", f"Proved unchanged in every situation: {', '.join(f'`{k}`' for k in held) or 'none'}.", ""]
    if "suite" in r:
        s = r["suite"]
        L += ["### Against your regression suite", "",
              f"- {s['cases']} cases exercise {s['regions_exercised']} of {s['regions_total']} regions; "
              f"{s['change_regions_never_exercised']} of the {s['change_regions_total']} regions where behavior must "
              f"change are never exercised"
              + (f" (e.g. {s['example_untested_change_region']})" if s["example_untested_change_region"] else ""),
              f"- gate: {s['gate_agent_calls']} agent calls vs {s['full_suite_agent_calls']} for the full suite "
              f"({s['compression']}x)", ""]
    if "audit" in r:
        a, sn = r["audit"], r["audit"]["snapshot_regression"]
        L += ["### Audit: the full suite, run both ways", "",
              f"- a snapshot regression (compare to the previous release's outputs) would flag {sn['cases_flagged']} "
              f"cases: {sn['flagged_but_intended']} intended changes (false alarms to re-label) and "
              f"{sn['flagged_and_real']} real failures, mixed together",
              f"- it would pass {sn['real_failures_passed_silently']} case(s) where a required change did not happen",
              "- differences by step: " + (", ".join(f"`{k}` {n}" for k, n in sn["differences_by_step"].items())
                                           or "none"),
              ("- running the whole suite found no failure the gate's plan missed" if
               a["plan_caught_every_failure_the_full_suite_found"] else
               "- **the whole suite found failures the gate's plan missed:** " +
               "; ".join(f"`{m['step']}` {m['problem']} ({m['cases_in_suite']} cases)" for m in a["plan_missed_failures"])
               + ". Probing one situation per region did not reach them: add the suite to the plan, "
                 "or more trials, for this agent."), ""]
    L.append(f"_{r['statement']}_")
    return "\n".join(L) + "\n"


def to_junit(r: dict) -> str:
    cases = []
    f = r["findings"]
    for key in ("failed_to_change", "unwarranted_change"):
        for x in f[key]:
            msg = f"{key.replace('_', ' ')} in {x['regions']} region(s), e.g. when {x['example_condition']}"
            cases.append(f'  <testcase classname="contradish.{key}" name="{escape(x["step"])}">'
                         f'<failure message="{escape(msg, {chr(34): "&quot;"})}"/></testcase>')
    for k in r["what_must_hold"]["steps_proved_unchanged_everywhere"]:
        if not any(x["step"] == k for x in f["unwarranted_change"]):
            cases.append(f'  <testcase classname="contradish.must_hold" name="{escape(k)}"/>')
    for c in {c["step"] for c in r["what_must_change"]}:
        if not any(x["step"] == c for x in f["failed_to_change"]):
            cases.append(f'  <testcase classname="contradish.must_change" name="{escape(c)}"/>')
    nfail = sum(1 for c in cases if "<failure" in c)
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name="contradish release gate" '
            f'tests="{len(cases)}" failures="{nfail}">\n' + "\n".join(cases) + "\n</testsuite>\n")


def export_examples(v2, diff=None, situations: Optional[list] = None) -> list:
    """
    The gate's expectations as eval examples ({inputs, outputs, metadata}), one per region
    representative (or per given situation), so another eval runner can execute them.
    """
    diff = diff or diff_versions(v2.program if hasattr(v2, "program") else v2,
                                 v2.program if hasattr(v2, "program") else v2)
    prog = diff.after
    sits = situations or [cell.rep() for cell, _, _ in diff.cells]
    out = []
    for s in sits:
        ci = _cell_of(diff, s)
        cell, norms, cls = diff.cells[ci]
        st = prog.complete_state(s)
        expected = {}
        for k in prog.step_ids():
            n, _ = prog.run_norm(k, st)
            expected[k] = n.to_dict()
        out.append({"inputs": {"situation": s, "policy": prog.render()},
                    "outputs": {"expected_norms": expected},
                    "metadata": {"region": cell.describe(),
                                 "must_change": sorted(k for k in prog.step_ids() if cls[k]["kind"] == "change"),
                                 "source": "contradish release gate"}})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Built-in demonstration
# ─────────────────────────────────────────────────────────────────────────────

def demo_suite(program, n: int = 600, seed: int = 20261016) -> list:
    """A synthetic 'production log' regression suite, skewed the way real traffic is."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append({"days_since_delivery": min(120, int(rng.expovariate(1 / 14))),
                    "category": rng.choices(["apparel", "electronics", "final_sale"], [0.7, 0.25, 0.05])[0],
                    "opened": rng.random() < 0.3,
                    "reason": rng.choices(["changed_mind", "defective"], [0.92, 0.08])[0],
                    "tier": rng.choices(["standard", "gold"], [0.85, 0.15])[0],
                    "price": float(rng.choice([19.99, 49.0, 100.0, 249.0, 899.0]))})
    return out


def demo(update: str = "restocking_fee_20", agent: str = "collateral", audit: bool = True, n: int = 600,
         plan: str = "complete"):
    from contradish.policy_program import load_builtin_program
    p, U = load_builtin_program("returns")
    u = U[update]
    ok, _ = p.split_edits(u)
    k = fixture_key("policy_owner")
    v1 = sign_pin(pin(p, "returns-v1", "contradish fixture", "built-in example"), k)
    v2 = pin(p.apply(ok), "returns-" + u.id, "contradish fixture", "built-in example",
             issued_by=u.source, supersedes=v1.digest)
    if u.source == "policy_owner":
        sign_pin(v2, k)
    a = (collateral_witness("label") if agent == "collateral" else threshold_witness() if agent == "threshold_bug"
         else version_witness(agent))
    return release_gate(v1, v2, a, a.__name__, "scripted", suite=demo_suite(p, n), audit=audit,
                        policy={"plan": plan}, trust_anchors=[k["public"]])
