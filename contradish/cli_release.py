"""`contradish release`: the verification gate for an AI behavior change (CI entry point)."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

COMMANDS = ("release",)


def register(sub) -> None:
    r = sub.add_parser(
        "release",
        help="Gate an agent release across a policy change: what must change, what must hold, PASS/REVIEW/BLOCK "
             "with a sealed certificate. Exit code 0 pass, 1 block, 2 review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  contradish release check v1.pin.json v2.pin.json --app mymodule:chat \\\n"
            "      [--suite cases.jsonl --audit] [--plan complete|classes] [--policy release_policy.json] \\\n"
            "      [--markdown gate.md] [--junit gate.xml] [--json gate.json] [--certificate run.json]\n"
            "  contradish release demo [--agent collateral|faithful|rigid|threshold_bug] [--plan classes]\n"
            "  contradish release export v1.pin.json v2.pin.json --out examples.jsonl   (for another eval runner)\n\n"
            "It derives the complete set of regions where behavior must change from the two pinned versions,\n"
            "probes the agent in each region and in every region that must hold, and reports a required change\n"
            "that did not happen separately from a change that should never have happened. With --suite, the\n"
            "team's own regression cases are mapped to regions (coverage gaps, cost); with --audit, the whole\n"
            "suite is also run both ways, to measure what a snapshot regression would have flagged, missed and\n"
            "falsely alarmed, and whether the gate's plan missed anything the full run found."
        ),
    )
    r.add_argument("release_action", choices=["check", "demo", "export"])
    r.add_argument("files", nargs="*")
    r.add_argument("--app", default=None, metavar="MODULE:FUNCTION", help="the agent's chat function")
    r.add_argument("--agent", default=None,
                   help="a scripted witness instead of --app: faithful, rigid, stale_permissions, leaky, "
                        "collateral, threshold_bug")
    r.add_argument("--model-name", dest="model_name", default=None)
    r.add_argument("--suite", default=None, metavar="CASES.jsonl",
                   help="one situation (a JSON object of facts) per line, from your regression suite or logs")
    r.add_argument("--audit", action="store_true", default=False,
                   help="with --suite: also run every suite case both ways and measure what the plan missed")
    r.add_argument("--plan", default=None, choices=["complete", "classes", "suite"],
                   help="complete: every region, issues a certificate (default). classes: every region that must "
                        "change plus one per behavior class elsewhere; cheaper, not a proof. suite: your --suite "
                        "cases plus one probe in every region they never reach")
    r.add_argument("--policy", default=None, metavar="POLICY.json",
                   help="release policy: block_on, review_on, high_risk_steps, trials, plan")
    r.add_argument("--update", default=None, help="demo: built-in update id (default restocking_fee_20)")
    r.add_argument("--trust", action="append", default=[], metavar="PUBKEY_HEX")
    r.add_argument("--json", default=None, metavar="FILE")
    r.add_argument("--markdown", default=None, metavar="FILE", help="a PR-comment summary")
    r.add_argument("--junit", default=None, metavar="FILE", help="JUnit XML for CI dashboards")
    r.add_argument("--certificate", default=None, metavar="FILE",
                   help="the run certificate (complete plan only); re-check with `contradish evidence check`")
    r.add_argument("--out", default=None, metavar="FILE", help="export: the examples file")
    r.add_argument("--no-exit-code", dest="no_exit", action="store_true", default=False,
                   help="always exit 0 (report only)")


def _agent(args):
    from contradish.action_frontier import llm_tool_agent
    from contradish.release_gate import collateral_witness, threshold_witness
    from contradish.versions import version_witness
    if args.app:
        mod, fn = args.app.split(":")
        sys.path.insert(0, os.getcwd())
        chat = getattr(importlib.import_module(mod), fn)
        return llm_tool_agent(chat, args.model_name or args.app), "model"
    name = (args.agent or "faithful").replace("witness:", "")
    if name == "collateral":
        return collateral_witness("label"), "scripted"
    if name == "threshold_bug":
        return threshold_witness(), "scripted"
    if name not in ("faithful", "rigid", "stale_permissions", "leaky"):
        sys.exit("--agent must be faithful, rigid, stale_permissions, leaky, collateral or threshold_bug "
                 "(or pass --app)")
    return version_witness(name), "scripted"


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def dispatch(args) -> bool:
    if args.command not in COMMANDS:
        return False
    from contradish.evidence import save
    from contradish.release_gate import demo, export_examples, release_gate, to_junit, to_markdown
    from contradish.symbolic import diff_versions
    from contradish.versions import load_pinned

    def read_pin(path):
        with open(path) as f:
            return load_pinned(json.load(f))

    policy = {}
    if args.policy:
        with open(args.policy) as f:
            policy = json.load(f)
    if args.plan:
        policy["plan"] = args.plan

    if args.release_action == "export":
        if len(args.files) != 2 or not args.out:
            sys.exit("usage: contradish release export V1.pin.json V2.pin.json --out examples.jsonl")
        v1, v2 = read_pin(args.files[0]), read_pin(args.files[1])
        ex = export_examples(v2, diff_versions(v1.program, v2.program))
        _write(args.out, "".join(json.dumps(e, default=str) + "\n" for e in ex))
        print(f"{len(ex)} examples (one per region, with the norms version {v2.id} requires) -> {args.out}")
        return True

    if args.release_action == "demo":
        r, cert = demo(update=args.update or "restocking_fee_20", agent=(args.agent or "collateral").replace(
            "witness:", ""), plan=policy.get("plan", "complete"))
    else:
        if len(args.files) != 2:
            sys.exit("usage: contradish release check V1.pin.json V2.pin.json (--app MODULE:FN | --agent NAME)")
        v1, v2 = read_pin(args.files[0]), read_pin(args.files[1])
        suite = None
        if args.suite:
            with open(args.suite) as f:
                suite = [json.loads(line) for line in f if line.strip()]
        agent, kind = _agent(args)
        r, cert = release_gate(v1, v2, agent, getattr(agent, "__name__", "agent"), kind, args.model_name,
                               suite=suite, audit=args.audit, policy=policy, trust_anchors=list(args.trust) or None)

    md = to_markdown(r)
    print(md)
    if args.markdown:
        _write(args.markdown, md)
    if args.junit:
        _write(args.junit, to_junit(r))
    if args.json:
        _write(args.json, json.dumps(r, indent=2, default=str) + "\n")
    if args.certificate:
        if cert is None:
            print("no certificate: the classes plan is not a proof (use --plan complete)", file=sys.stderr)
        else:
            save(cert, args.certificate)
    if not args.no_exit and r["exit_code"]:
        sys.exit(r["exit_code"])
    return True
