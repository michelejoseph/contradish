#!/usr/bin/env python3
"""Measure the release gate on real models: what it catches, what a snapshot regression would have done,
and what its cheap plan misses, on the built-in returns policy.

    python3 scripts/release_gate_experiment.py --models qwen2.5:7b,llama3.1:8b --out studies/release-gate.json
    python3 scripts/release_gate_experiment.py --scripted          # the same pipeline on scripted agents, offline

For each model and update: the `classes` plan (every region that must change, one region per behavior
class elsewhere) plus a full audit on a synthetic production-like suite of --suite-size cases, run both ways.
Calls per model and update: 2 x (plan regions + suite size). Local models through Ollama (examples/ollama_agent.py).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contradish.action_frontier import llm_tool_agent  # noqa: E402
from contradish.policy_program import load_builtin_program  # noqa: E402
from contradish.release_gate import collateral_witness, demo_suite, release_gate, threshold_witness  # noqa: E402
from contradish.versions import fixture_key, pin, sign_pin, version_witness  # noqa: E402

UPDATES = ("restocking_fee_20", "window_45")


def versions(update):
    p, U = load_builtin_program("returns")
    u = U[update]
    ok, _ = p.split_edits(u)
    k = fixture_key("policy_owner")
    v1 = sign_pin(pin(p, "returns-v1", "contradish fixture", "built-in example"), k)
    v2 = sign_pin(pin(p.apply(ok), "returns-" + u.id, "contradish fixture", "built-in example",
                      issued_by=u.source, supersedes=v1.digest), k)
    return p, v1, v2, k["public"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="")
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--updates", default=",".join(UPDATES))
    ap.add_argument("--suite-size", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    agents = []
    if a.scripted:
        agents = [("witness:faithful", lambda: version_witness("faithful"), "scripted"),
                  ("witness:rigid", lambda: version_witness("rigid"), "scripted"),
                  ("witness:collateral", lambda: collateral_witness("label"), "scripted"),
                  ("witness:threshold_bug", lambda: threshold_witness(), "scripted")]
    for m in [x for x in a.models.split(",") if x]:
        os.environ["OLLAMA_MODEL"] = m
        import importlib
        import examples.ollama_agent as OA
        importlib.reload(OA)
        agents.append((m, (lambda chat=OA.chat, m=m: llm_tool_agent(chat, m)), "model"))
    if not agents:
        sys.exit("pass --models or --scripted")
    rows = []
    for name, make, kind in agents:
        for upd in a.updates.split(","):
            p, v1, v2, anchor = versions(upd)
            t0 = time.time()
            r, _ = release_gate(v1, v2, make(), name, kind, name if kind == "model" else None,
                                suite=demo_suite(p, a.suite_size), audit=True, policy={"plan": "classes"},
                                trust_anchors=[anchor])
            au, sn = r["audit"], r["audit"]["snapshot_regression"]
            row = {"agent": name, "update": upd, "verdict": r["verdict"], "reasons": r["reasons"],
                   "plan_calls": r["plan"]["agent_calls"], "suite_calls": r["suite"]["full_suite_agent_calls"],
                   "required_changes_made": r["findings"]["required_changes_made"],
                   "unaffected_held": r["findings"]["unaffected_constraints_held"],
                   "failed_to_change_steps": [x["step"] for x in r["findings"]["failed_to_change"]],
                   "unwarranted_change_steps": [x["step"] for x in r["findings"]["unwarranted_change"]],
                   "suite_change_regions_untested": f"{r['suite']['change_regions_never_exercised']}/"
                                                   f"{r['suite']['change_regions_total']}",
                   "snapshot_flagged": sn["cases_flagged"], "snapshot_false_alarms": sn["flagged_but_intended"],
                   "snapshot_silent_passes": sn["real_failures_passed_silently"],
                   "plan_missed": au["plan_missed_failures"], "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"{name:24} {upd:18} {row['verdict']:6} plan {row['plan_calls']:4} calls | changes "
                  f"{row['required_changes_made']:>7} held {row['unaffected_held']:>9} | snapshot flagged "
                  f"{row['snapshot_flagged']:3} (false alarms {row['snapshot_false_alarms']:3}, silent passes "
                  f"{row['snapshot_silent_passes']:3}) | plan missed {[m['step'] for m in row['plan_missed']]}",
                  flush=True)
            if a.out:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                with open(a.out, "w") as f:
                    json.dump({"suite_size": a.suite_size, "rows": rows}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
