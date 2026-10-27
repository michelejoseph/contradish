"""
CLI for action-level transitions, evidence, counterexamples, exhibits and
attribution. Registered into `contradish` by cli.py.

    contradish about                      what contradish is, who made it, how to cite it
    contradish actions derive             which downstream actions an update requires changing
    contradish actions verify             score an agent's actions; write certificates for findings
    contradish counterexample             shrink an unauthorized change to a minimal certified case
    contradish evidence check FILE...     verify certificates with the independent checker
    contradish exhibits [list|show|verify] the shipped, verifiable failure exhibits
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

VERIFIES = ("Contradish verifies that consequential AI actions and obligations remain compliant across "
            "authenticated governing-state transitions, proving required changes and preservation of unaffected "
            "constraints within an explicitly defined verification scope.")

LEGITIMACY = ("Contradish verifies that every change to an AI agent's governing specification is legitimate "
              "(authentic, procedurally valid, authorized by its effects, and within entrenched constraints) and "
              "that the agent's behavior then follows it exactly, within an explicit verification scope.")

ABOUT = {
    "name": "contradish",
    "legitimacy": LEGITIMACY,
    "verifies": VERIFIES,
    "top_line": "Contradish measures whether AI transitions remain faithful to their governing information.",
    "principle": "Change what truth requires. Preserve what truth does not require changing.",
    "definition": ("Behavioral Update Fidelity measures whether an AI changes its behavior exactly when, "
                   "and only as far as, changes in governing information warrant."),
    "author": "Michele Joseph",
    "attribution": "Behavioral Update Fidelity was introduced by Michele Joseph in 2026.",
    "also_introduced_by_the_author": ["CAI Strain", "CAI-Bench", "the policy evaluation contract",
                                      "the transition contract", "action-level transition verification",
                                      "whole-run warranted-transition proofs", "the perspective atlas",
                                      "authority by behavioral effect"],
    "repository": "https://github.com/michelejoseph/contradish",
    "website": "https://contradish.com",
    "license": "MIT",
    "cite": "CITATION.cff / CITATION.bib in the repository (key: joseph2026buf)",
    "spec": "docs/BUF-SPEC.md",
    "entry_points": {
        "contradish guard demo": "start here: the product, one call before tool calls run (docs/INTEGRATE.md)",
        "gate a release": "contradish release check v1.pin.json v2.pin.json --app mymodule:chat --suite cases.jsonl --audit",
        "derive what must change": "contradish actions derive --update restocking_fee_20",
        "verify an agent": "contradish actions verify --update customer_claims --app mymodule:chat",
        "complete difference of two versions": "contradish versions demo --update exchange_closed_late",
        "whole-run proof": "contradish versions certify v1.pin.json v2.pin.json --app mymodule:chat",
        "alternative frames": "contradish perspectives atlas",
        "is a change legitimate": "contradish legitimacy demo",
        "inspect a verified failure": "contradish exhibits show EX-0001",
        "re-verify independently": "contradish exhibits verify",
        "run CAI-Bench": "contradish benchmark --model <model> --provider anthropic|openai",
    },
}

_HERE = os.path.dirname(__file__)
EXHIBITS_DIR = os.path.join(_HERE, "exhibits")


def register(sub) -> None:
    from contradish import cli_integrity as _ci
    _ci.register(sub)
    ab = sub.add_parser("about", help="What contradish is, who introduced it, and how to cite it.")
    ab.add_argument("--json", action="store_true", default=False)

    ac = sub.add_parser(
        "actions",
        help="Which downstream agent actions a policy update requires changing; verify an agent against it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Derive the warranted change frontier over an agent's tool calls from a machine-readable\n"
            "policy (contradish/contracts/programs/*.json) and its dependency model, then verify\n"
            "observed actions against it. No model or judge is involved in the derivation.\n\n"
            "  contradish actions derive --update restocking_fee_20\n"
            "  contradish actions derive --program my_policy.json --update my_update.json --json\n"
            "  contradish actions verify --update customer_claims --agent witness:credulous --evidence-dir out/\n"
            "  contradish actions verify --update customer_claims --app mymodule:chat --evidence-dir out/\n\n"
            "--app takes OpenAI-style messages and returns the reply text; it must answer with a JSON\n"
            "array of tool calls. witness:{faithful,rigid,credulous,overreach} are scripted agents\n"
            "with known behavior, for checking the pipeline; they are not models."
        ),
    )
    ac.add_argument("actions_cmd", choices=["derive", "verify", "list"])
    _program_args(ac)
    ac.add_argument("--agent", default=None, help="witness:faithful|rigid|credulous|overreach")
    ac.add_argument("--app", default=None, metavar="MODULE:FUNCTION", help="A chat function to verify.")
    ac.add_argument("--model-name", dest="model_name", default=None, help="Recorded in certificates for --app.")
    ac.add_argument("--max-situations", dest="max_situations", type=int, default=None,
                    help="verify: only the first N situations (to bound model calls).")
    ac.add_argument("--evidence-dir", dest="evidence_dir", default=None, help="verify: write certificates here.")
    ac.add_argument("--json", action="store_true", default=False)

    ce = sub.add_parser("counterexample",
                        help="Shrink an unauthorized change to a 1-minimal scenario and certify it.")
    _program_args(ce)
    ce.add_argument("--situation", required=True, help="JSON object of facts, or @file.json")
    ce.add_argument("--step", required=True, help="The step (action) that was captured.")
    ce.add_argument("--agent", default=None, help="witness:credulous, ...")
    ce.add_argument("--app", default=None, metavar="MODULE:FUNCTION")
    ce.add_argument("--model-name", dest="model_name", default=None)
    ce.add_argument("--trials", type=int, default=1)
    ce.add_argument("--k", type=int, default=1, help="Reproduces if captured in at least k of --trials runs.")
    ce.add_argument("--out", default=None, help="Write the certificate here (default: stdout).")

    ev = sub.add_parser("evidence", help="Verify evidence certificates with the independent checker.")
    ev.add_argument("evidence_cmd", choices=["check"])
    ev.add_argument("files", nargs="+")
    ev.add_argument("--rerun", action="store_true", default=False,
                    help="For scripted witness agents, also re-run the agent to re-establish the observed "
                         "calls and every minimality witness.")
    ev.add_argument("--json", action="store_true", default=False)

    vs = sub.add_parser(
        "versions",
        help="Two pinned, verified versions of a policy: the complete difference (as conditions) and a "
             "whole-run certificate that every required change happened and everything else held.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  contradish versions pin my_policy.json --id v2 --verified-by 'Legal' --method 'review' --out v2.pin.json\n"
            "  contradish versions diff v1.pin.json v2.pin.json\n"
            "  contradish versions certify v1.pin.json v2.pin.json --agent witness:faithful --out run.json\n"
            "  contradish versions certify v1.pin.json v2.pin.json --app mymodule:chat --out run.json\n"
            "  contradish versions demo --update exchange_closed_late      (built-in returns policy)\n\n"
            "The difference is proved complete over the whole situation space by an exact cell\n"
            "decomposition (docs/BUF-SPEC.md section 12); permission changes are reported separately from\n"
            "obligation changes. `contradish evidence check run.json` re-derives everything independently."
        ),
    )
    vs.add_argument("versions_cmd", choices=["pin", "diff", "certify", "demo", "keygen", "approve"])
    vs.add_argument("--role", default=None, help="approve: the approving role")
    vs.add_argument("--key", default=None, metavar="KEY.json", help="approve: the role's key")
    vs.add_argument("--proposed-at", dest="proposed_at", default=None, help="pin: ISO date proposed")
    vs.add_argument("--effective-at", dest="effective_at", default=None, help="pin: ISO date effective")
    vs.add_argument("files", nargs="*")
    vs.add_argument("--id", default=None)
    vs.add_argument("--verified-by", dest="verified_by", default=None)
    vs.add_argument("--method", default=None)
    vs.add_argument("--evidence", default="")
    vs.add_argument("--issued-by", dest="issued_by", default=None)
    vs.add_argument("--update", default=None, help="demo: built-in update id (default exchange_closed_late)")
    vs.add_argument("--agent", default=None, help="witness:faithful|rigid|stale_permissions|leaky")
    vs.add_argument("--app", default=None, metavar="MODULE:FUNCTION")
    vs.add_argument("--model-name", dest="model_name", default=None)
    vs.add_argument("--supersedes", default=None, help="pin: the previous pinned version file (binds the chain)")
    vs.add_argument("--sign", default=None, metavar="KEY.json", help="pin: sign with this key (contradish versions keygen)")
    vs.add_argument("--source", default=None, help="keygen: the source the key belongs to")
    vs.add_argument("--trust", action="append", default=[], metavar="PUBKEY_HEX",
                    help="certify: a trust anchor for the first version's signature (repeatable)")
    vs.add_argument("--scope", default=None, metavar="SCOPE.json",
                    help="certify: the verification scope (situations, actions, trials, confidence, assumptions)")
    vs.add_argument("--trials", type=int, default=None, help="certify: runs per region (overrides the scope)")
    vs.add_argument("--out", default=None)
    vs.add_argument("--json", action="store_true", default=False)

    lg = sub.add_parser(
        "legitimacy",
        help="Is a change to the governing specification legitimate? Provenance, procedure, authority by "
             "effect, and invariants proved over every situation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  contradish legitimacy demo                         (all built-in cases)\n"
            "  contradish legitimacy demo --case pricing_prepaid_cut\n"
            "  contradish legitimacy check v1.pin.json v2.pin.json --trust <pubkey> --out legit.json\n"
            "  contradish versions approve v2.pin.json --key legal.key.json --role legal --out v2.pin.json\n\n"
            "The rules of change live in the specification's `charter`: domains of behavioral effect granted\n"
            "to roles, approval procedures, entrenched invariants, and how the charter itself may be amended.\n"
            "docs/BUF-SPEC.md section 15."
        ),
    )
    lg.add_argument("legitimacy_cmd", choices=["demo", "check"])
    lg.add_argument("files", nargs="*")
    lg.add_argument("--case", default=None)
    lg.add_argument("--trust", action="append", default=[], metavar="PUBKEY_HEX")
    lg.add_argument("--out", default=None)
    lg.add_argument("--json", action="store_true", default=False)

    pp = sub.add_parser(
        "perspectives",
        help="Alternative governing frames over the same situations: where they agree, where they "
             "contest (and which frames side together), and whether an agent switches frames faithfully.",
    )
    pp.add_argument("perspectives_cmd", choices=["list", "atlas", "switch"])
    pp.add_argument("name", nargs="?", default="dietary")
    pp.add_argument("--agent", default="witness:faithful")
    pp.add_argument("--app", default=None, metavar="MODULE:FUNCTION")
    pp.add_argument("--json", action="store_true", default=False)

    cb = sub.add_parser("cai-bench", help="CAI-Bench offline: what is in the frozen files, with content hashes.")
    cb.add_argument("cai_cmd", nargs="?", default="manifest", choices=["manifest"])
    cb.add_argument("--version", dest="bench_version", default="v2")
    cb.add_argument("--json", action="store_true", default=False)

    ex = sub.add_parser("exhibits", help="Shipped, independently verifiable failure exhibits.")
    ex.add_argument("exhibits_cmd", nargs="?", default="list", choices=["list", "show", "verify", "path"])
    ex.add_argument("id", nargs="?", default=None)
    ex.add_argument("--json", action="store_true", default=False)

    js = sub.add_parser(
        "justify",
        help="Before an irreversible action: is it justified by what has been established? If not, the fewest "
             "verifications that would make it so.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Detects, before execution, an agent acting as if it had enough information when it does not, and\n"
            "derives the verification plan that restores justified action. No model or judge is involved in\n"
            "the check or the plan.\n\n"
            "  contradish justify demo\n"
            "  contradish justify check --state state.json --calls calls.json\n"
            "  contradish justify plan --state state.json --steps refund,notify\n"
            "  contradish justify evaluate --agent scripted:assumes_defaults\n"
            "  contradish justify evaluate --app mymodule:chat --model-name M --limit 100 --out run.json\n\n"
            "state.json: {\"evidence\": [{\"fact\": \"days_since_delivery\", \"value\": 12, \"source\": \"customer\"}, ...]}\n"
            "Evidence counts only from a source with standing over the fact (built-in for returns; otherwise\n"
            "give \"fact_authority\" in the state file). calls.json: a JSON array of tool calls."
        ),
    )
    js.add_argument("justify_cmd", choices=["demo", "check", "plan", "evaluate"])
    js.add_argument("--program", default="returns", help="Built-in program name or a .json path (default: returns).")
    js.add_argument("--state", default=None, help="Information state JSON.")
    js.add_argument("--calls", default=None, help="Proposed tool calls JSON (check).")
    js.add_argument("--steps", default=None, help="Comma-separated steps to settle (plan; default: every step).")
    js.add_argument("--agent", default=None, help="scripted:justified|assumes_defaults|trusts_claims|always_verifies")
    js.add_argument("--app", default=None, metavar="MODULE:FUNCTION", help="A chat function to evaluate.")
    js.add_argument("--model-name", default=None)
    js.add_argument("--limit", type=int, default=None, help="Evaluate on an evenly spaced subset of pairs.")
    js.add_argument("--out", default=None, help="Write the full evaluation (every pair, every gate) to this file.")
    js.add_argument("--json", action="store_true", default=False)

    en = sub.add_parser(
        "entitled",
        help="Before a refusal goes out: does what is established already entitle the person to more? Catches "
             "withholding caused by a missed amendment, sufficient evidence not recognized, or evidence not sought.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "The other half of `justify`: stops an AI from withholding what a person is entitled to. No model or\n"
            "judge is involved in the check.\n\n"
            "  contradish entitled demo                                   pharmacy refills, scripted agents\n"
            "  contradish entitled check --as-of 2026-10-20 --state s.json --calls c.json\n"
            "  contradish entitled evaluate --agent scripted:stale\n"
            "  contradish entitled evaluate --app mymodule:chat --limit 200 --out run.json\n\n"
            "Exit codes for check: 0 allow, 1 withholding (deliver or verify), 2 review (over-asking or a grant)."
        ),
    )
    gd = sub.add_parser(
        "guard",
        help="The product: one call before an agent's tool calls run. Blocks unjustified irreversible actions, catches "
             "withheld entitlements, says what to verify; every decision is a certificate anyone can re-check.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  contradish guard demo                          the whole loop offline, in under a minute\n"
            "  contradish guard init DIR                      a starter policy, examples and an integration file\n"
            "  contradish guard lint POLICY                   what would make the guard wrong or unusable\n"
            "  contradish guard test POLICY                   the policy's examples, in your own terms\n"
            "  contradish guard check POLICY --evidence E.json --calls C.json [--as-of DATE] [--log L.jsonl]\n"
            "  contradish guard verify L.jsonl                re-derive every decision independently (stdlib only)\n"
            "  contradish guard report L.jsonl [--out R.md]   what a period of decisions shows\n\n"
            "check exits 0 allow, 1 deliver or block, 2 verify or review. See docs/INTEGRATE.md."
        ),
    )
    gd.add_argument("guard_cmd", choices=["demo", "init", "draft", "review", "approve", "lint", "test", "check",
                                          "verify", "report"])
    gd.add_argument("target", nargs="?", default=None, help="POLICY, DIR or LOG depending on the command")
    gd.add_argument("--evidence", default=None)
    gd.add_argument("--calls", default=None)
    gd.add_argument("--as-of", default=None)
    gd.add_argument("--log", default=None)
    gd.add_argument("--mode", default="shadow", choices=["shadow", "enforce"])
    gd.add_argument("--examples", default=None, help="Examples JSON (default: the policy file's own).")
    gd.add_argument("--out", default=None)
    gd.add_argument("--no-verify", action="store_true", default=False)
    gd.add_argument("--json", action="store_true", default=False)
    gd.add_argument("--model", default=None, help="draft: the encoder model, provider:model (e.g. anthropic:claude-sonnet-4-5)")
    gd.add_argument("--examiner", default=None, help="draft: the independent examiner model (default: --model)")
    gd.add_argument("--tools", default=None, help="draft: comma-separated tool names your agent already has")
    gd.add_argument("--demo", action="store_true", default=False, help="draft: offline, with scripted stand-in replies")
    gd.add_argument("--text", default=None, help="review: the policy text (default: the copy stored in the draft)")
    gd.add_argument("--by", default=None, help="approve: the name of the person approving")
    gd.add_argument("--role", default="", help="approve: their role")
    gd.add_argument("--note", default="", help="approve: a note kept with the approval")
    gd.add_argument("--despite-disagreements", action="store_true", default=False,
                    help="approve: you checked each disagreement with the independent reading and the encoding is right")

    en.add_argument("entitled_cmd", choices=["demo", "check", "evaluate"])
    en.add_argument("--policy", default="pharmacy_refills", help="Built-in program name or a .json path.")
    en.add_argument("--as-of", default="2026-10-20", help="Date of the request (YYYY-MM-DD): decides what is in force.")
    en.add_argument("--state", default=None)
    en.add_argument("--calls", default=None)
    en.add_argument("--agent", default=None)
    en.add_argument("--app", default=None, metavar="MODULE:FUNCTION")
    en.add_argument("--model-name", default=None)
    en.add_argument("--limit", type=int, default=None)
    en.add_argument("--out", default=None)
    en.add_argument("--json", action="store_true", default=False)


def _program_args(p) -> None:
    p.add_argument("--program", default="returns", help="Built-in program name or a .json path (default: returns).")
    p.add_argument("--update", default=None, help="Built-in update id or an update .json path.")


COMMANDS = ("about", "actions", "counterexample", "evidence", "exhibits", "cai-bench", "versions", "perspectives",
            "legitimacy", "justify", "entitled", "guard")


def dispatch(args) -> bool:
    from contradish import cli_integrity as _ci
    if _ci.dispatch(args):
        return True
    if args.command not in COMMANDS:
        return False
    {"about": cmd_about, "actions": cmd_actions, "counterexample": cmd_counterexample,
     "evidence": cmd_evidence, "exhibits": cmd_exhibits, "cai-bench": cmd_cai_bench,
     "versions": cmd_versions, "perspectives": cmd_perspectives, "legitimacy": cmd_legitimacy,
     "justify": cmd_justify, "entitled": cmd_entitled, "guard": cmd_guard}[args.command](args)
    return True


# ── helpers ──────────────────────────────────────────────────────────────────

def _load(args):
    from contradish.policy_program import PolicyProgram, ProgramUpdate, load_builtin_program
    if args.program.endswith(".json") or os.path.sep in args.program:
        with open(args.program) as f:
            spec = json.load(f)
        updates = {u["id"]: ProgramUpdate.from_dict(u) for u in spec.pop("updates", [])}
        program = PolicyProgram(spec)
    else:
        program, updates = load_builtin_program(args.program)
    update = None
    if args.update:
        if args.update in updates:
            update = updates[args.update]
        else:
            with open(args.update) as f:
                update = ProgramUpdate.from_dict(json.load(f))
    return program, updates, update


def _agent(args):
    from contradish.action_frontier import WITNESSES, llm_tool_agent, witness_agent
    if args.app:
        mod, fn = args.app.split(":")
        sys.path.insert(0, os.getcwd())
        chat = getattr(importlib.import_module(mod), fn)
        return llm_tool_agent(chat, args.model_name or args.app), "model"
    name = (args.agent or "").replace("witness:", "")
    if name not in WITNESSES:
        sys.exit(f"--agent must be one of {['witness:' + w for w in WITNESSES]} (or pass --app)")
    return witness_agent(name), "scripted"


def _show(c) -> str:
    if hasattr(c, "modality") or (isinstance(c, dict) and "modality" in c):
        d = c.to_dict() if hasattr(c, "to_dict") else c
        if d["modality"] == "F":
            return "forbidden"
        return ("required " if d["modality"] == "O" else "permitted ") + _show(d["call"])
    if c is None:
        return "(no call)"
    d = c.to_dict() if hasattr(c, "to_dict") else c
    return d["tool"] + "(" + ", ".join(f"{k}={v}" for k, v in sorted(d["args"].items())) + ")"


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_about(args) -> None:
    from contradish import __version__
    info = dict(ABOUT, version=__version__)
    if args.json:
        print(json.dumps(info, indent=2))
        return
    print(info["top_line"])
    print(info["legitimacy"])
    print(info["verifies"])
    print(info["principle"])
    print()
    print(info["definition"])
    print(info["attribution"])
    print()
    print(f"version     {__version__}")
    print(f"author      {info['author']}")
    print(f"repository  {info['repository']}")
    print(f"spec        {info['spec']}")
    print(f"cite        {info['cite']}")
    print()
    print("Integrating it into an agent: contradish guard demo, then docs/INTEGRATE.md.")
    print()
    for k, v in info["entry_points"].items():
        print(f"  {k:<28} {v}")


def cmd_actions(args) -> None:
    from contradish.action_frontier import derive_action_frontier, run_agent, verify_trajectories
    program, updates, update = _load(args)
    if args.actions_cmd == "list":
        print(f"program {program.id}: steps {program.step_ids()}")
        for uid, u in updates.items():
            print(f"  {uid:<22} source={u.source:<13} channel={u.channel:<7} {u.content[:70]}")
        return
    if update is None:
        sys.exit("--update is required (see `contradish actions list`)")
    fr = derive_action_frontier(program, update)
    if args.actions_cmd == "derive":
        if args.json:
            print(json.dumps(fr.to_dict(), indent=2))
            return
        s = fr.summary()
        print(f"update {update.id}: from '{update.source}' via {update.channel}")
        print(f"  authorized edits:   {[e.clause for e in fr.authorized] or 'none'}")
        print(f"  unauthorized edits: {[e.clause for e in fr.unauthorized] or 'none'} (must be resisted)")
        print(f"  clauses whose meaning changed: {sorted(fr.changed) or 'none'}")
        print(f"  {s['situations']} situations x {len(program.step_ids())} steps = {s['cases']} action cases:"
              f" {s['must_change']} must change, {s['must_preserve']} must be preserved, {s['must_resist']} must resist")
        print()
        print(f"  {'step':<9} {'change':>6} {'indep.':>7} {'coinc.':>7} {'resist':>7}  dependency path")
        for k, ps in s["per_step"].items():
            path = " -> ".join(s["downstream_paths"].get(k, [])) or "(not downstream of any change)"
            print(f"  {k:<9} {ps['change']:>6} {ps['preserve_independent']:>7} {ps['preserve_coincidental']:>7}"
                  f" {ps['resist']:>7}  {path}")
        ex = [v for v in fr.verdicts if v.must == "change"][:3]
        if ex:
            print("\n  examples of required changes:")
            for v in ex:
                print(f"    situation {fr.situations[v.situation]}\n      {v.step}: {_show(v.before)}  ->  {_show(v.after)}")
        return
    agent, kind = _agent(args)
    idx = None if args.max_situations is None else list(range(min(args.max_situations, len(fr.situations))))
    if idx is not None:
        fr.situations = fr.situations[:len(idx)]
        fr.verdicts = [v for v in fr.verdicts if v.situation < len(idx)]
        fr.__post_init__()
    observed = run_agent(fr, agent)
    name = getattr(agent, "__name__", "agent")
    r = verify_trajectories(fr, observed, name)
    sc = r.scores()
    if args.evidence_dir:
        from contradish.evidence import certificate, save
        os.makedirs(args.evidence_dir, exist_ok=True)
        n = 0
        for c in r.findings():
            raw = None
            if kind == "model":
                o = observed[c.situation]
                raw = {"before": getattr(o["before"], "raw", ""), "after": getattr(o["after"], "raw", "")}
            cert = certificate(fr, c, name, kind, args.model_name, raw)
            save(cert, os.path.join(args.evidence_dir, f"{update.id}-s{c.situation}-{c.step}-{cert['kind']}.json"))
            n += 1
        sc["certificates_written"] = n
    if args.json:
        print(json.dumps(r.to_dict() if not args.evidence_dir else {"scores": sc}, indent=2))
        return
    print(f"agent {name} on update {update.id}: {sc['cases']} action cases")
    for k in ("fidelity", "change", "preservation", "authority_respected", "unnecessary_change_rate"):
        v = sc[k]
        print(f"  {k:<24} {'n/a' if v is None else round(v, 3)}")
    print(f"  statuses                 {sc['status_counts']}")
    print(f"  unnecessary changes      {sc['unnecessary_changes']}")
    print(f"  unauthorized changes     {sc['unauthorized_changes']}")
    if args.evidence_dir:
        print(f"  certificates written     {sc['certificates_written']} -> {args.evidence_dir}")
        print(f"  verify them:             contradish evidence check {args.evidence_dir}/*.json")


def cmd_counterexample(args) -> None:
    from contradish.counterexample import minimize_unauthorized_change
    from contradish.evidence import save
    program, _, update = _load(args)
    if update is None:
        sys.exit("--update is required")
    sit = args.situation
    if sit.startswith("@"):
        with open(sit[1:]) as f:
            situation = json.load(f)
    else:
        situation = json.loads(sit)
    agent, kind = _agent(args)
    cert = minimize_unauthorized_change(program, update, situation, args.step, agent,
                                        getattr(agent, "__name__", "agent"), kind, args.model_name,
                                        args.trials, args.k)
    if args.out:
        save(cert, args.out)
        m = cert["minimization"]
        print(f"minimal counterexample written to {args.out}")
        print(f"  kept:    {[k['component'] for k in m['kept']]}")
        print(f"  removed: {m['removed']}")
        print(f"  claim:   {cert['claim']}")
    else:
        print(json.dumps(cert, indent=2, sort_keys=True))


def _rerun(cert: dict) -> list:
    """Re-establish a scripted-witness certificate by running the witness again."""
    from contradish.action_frontier import build_agent_input, observe, witness_agent
    from contradish.counterexample import _build, components_of
    from contradish.policy_program import Call, PolicyProgram, ProgramUpdate, calls_equal
    prov = cert.get("provenance", {})
    if str(cert.get("schema", "")).startswith("contradish.run_certificate/"):
        return _rerun_run(cert)
    if str(cert.get("schema", "")).startswith("contradish.legitimacy_certificate/"):
        return []          # fully determined by its inputs; the checker re-derives it
    if prov.get("agent_kind") != "scripted" or not str(prov.get("agent", "")).startswith("witness:"):
        return ["--rerun applies to scripted witness agents only; a model's replies are checked from the raw text"]
    agent = witness_agent(prov["agent"].split(":", 1)[1])
    p = PolicyProgram(cert["policy"])
    u = ProgramUpdate.from_dict(cert["update"])
    probs = []
    b0, _, _ = observe(p, agent(build_agent_input(p, cert["situation"])))
    b1, _, _ = observe(p, agent(build_agent_input(p, cert["situation"], u)))
    if not calls_equal(b0[cert["step"]], Call.from_dict(cert["observed"]["before"])):
        probs.append("rerun: observed 'before' not reproduced")
    if not calls_equal(b1[cert["step"]], Call.from_dict(cert["observed"]["after"])):
        probs.append("rerun: observed 'after' not reproduced")
    mini = cert.get("minimization")
    if mini:
        from contradish.counterexample import minimize_unauthorized_change
        o = mini["original"]
        ou = ProgramUpdate.from_dict(o["update"])
        try:
            again = minimize_unauthorized_change(p, ou, o["situation"], cert["step"], agent, prov["agent"])
            if [k["component"] for k in again["minimization"]["kept"]] != [k["component"] for k in mini["kept"]]:
                probs.append("rerun: minimization did not reproduce the same minimal scenario")
            if any(w["reproduced"] for w in again["minimization"]["witnesses"]):
                probs.append("rerun: a single removal still reproduces")
        except ValueError as exc:
            probs.append(f"rerun: {exc}")
    return probs


def cmd_evidence(args) -> None:
    from contradish.evidence_check import check, check_legitimacy, check_run
    results = []
    for path in args.files:
        with open(path) as f:
            cert = json.load(f)
        sch = str(cert.get("schema", ""))
        r = (check_run(cert) if sch.startswith("contradish.run_certificate/") else
             check_legitimacy(cert) if sch.startswith("contradish.legitimacy_certificate/") else check(cert))
        if args.rerun and r["verdict"] == "VERIFIED":
            extra = _rerun(cert)
            hard = [x for x in extra if x.startswith("rerun:")]
            r["notes"] += [x for x in extra if not x.startswith("rerun:")]
            if hard:
                r["problems"] += hard
                r["verdict"] = "REJECTED"
            else:
                r["notes"].append("re-run: the scripted agent reproduced the observed calls"
                                  + (" and the minimal scenario" if cert.get("minimization") else ""))
        r["file"] = path
        results.append(r)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            label = "certificate " + r["verdict"]
            c = r.get("claim")
            if isinstance(c, dict) and "statement" in c:
                print(f"{label}  {r['file']}")
                print(f"  {c['statement']}")
            else:
                print(f"{r['verdict']}  {r['file']}")
                print(f"  claim: {c}")
            for x in r["problems"]:
                print(f"  problem: {x}")
            for x in r["notes"]:
                print(f"  note: {x}")
    sys.exit(0 if all(r["verdict"] == "VERIFIED" for r in results) else 1)


def _index() -> list:
    with open(os.path.join(EXHIBITS_DIR, "index.json")) as f:
        return json.load(f)["exhibits"]


def cmd_exhibits(args) -> None:
    from contradish.evidence_check import check as _check1, check_legitimacy, check_run

    def check(cert):
        sch = str(cert.get("schema", ""))
        if sch.startswith("contradish.run_certificate/"):
            return check_run(cert)
        if sch.startswith("contradish.legitimacy_certificate/"):
            return check_legitimacy(cert)
        return _check1(cert)
    ex = _index()
    if args.exhibits_cmd == "path":
        print(EXHIBITS_DIR)
        return
    if args.exhibits_cmd == "list":
        if args.json:
            print(json.dumps(ex, indent=2))
            return
        for e in ex:
            print(f"{e['id']}  {e['kind']:<20} agent={e['agent']:<18} {e['title']}")
        print("\nshow one:   contradish exhibits show EX-0001")
        print("verify all: contradish exhibits verify   (independent checker; exit 0 iff all VERIFIED)")
        return
    if args.exhibits_cmd == "show":
        if not args.id:
            sys.exit("usage: contradish exhibits show EX-0001")
        e = next((x for x in ex if x["id"] == args.id), None)
        if e is None:
            sys.exit(f"no exhibit {args.id}")
        with open(os.path.join(EXHIBITS_DIR, e["file"])) as f:
            cert = json.load(f)
        if args.json:
            print(json.dumps(cert, indent=2, sort_keys=True))
            return
        r = check(cert)
        print(f"{e['id']}: {e['title']}")
        print(f"  {e['summary']}")
        print(f"\n  kind        {cert['kind']}")
        print(f"  agent       {cert['provenance']['agent']} ({cert['provenance']['agent_kind']})")
        print(f"  update      from '{cert['update']['source']}' via {cert['update']['channel']}: {cert['update']['content']!r}")
        print(f"  situation   {cert['situation']}")
        print(f"  step        {cert['step']}")
        print(f"  warranted   {_show(cert['derivation']['warranted_before'])}  ->  {_show(cert['derivation']['warranted_after'])}")
        print(f"  observed    {_show(cert['observed']['before'])}  ->  {_show(cert['observed']['after'])}")
        print(f"  claim       {cert['claim']}")
        if cert.get("minimization"):
            m = cert["minimization"]
            print(f"  minimal     kept {[k['component'] for k in m['kept']]}; removed {len(m['removed'])} components")
        print(f"\n  independent check: {r['verdict']}")
        for x in r["problems"] + r["notes"]:
            print(f"    - {x}")
        print(f"\n  {cert['attribution']}")
        print(f"  file: {os.path.join(EXHIBITS_DIR, e['file'])}")
        return
    if args.exhibits_cmd == "verify":
        ok = True
        for e in ex:
            with open(os.path.join(EXHIBITS_DIR, e["file"])) as f:
                cert = json.load(f)
            r = check(cert)
            extra = _rerun(cert)
            hard = [x for x in extra if x.startswith("rerun:")]
            verdict = "VERIFIED" if r["verdict"] == "VERIFIED" and not hard else "REJECTED"
            ok &= verdict == "VERIFIED"
            c = cert.get("claim")
            tag = ""
            if isinstance(c, dict) and "verified" in c:
                tag = "  [claim: " + ("verified" if c["verified"] else "proved, not authenticated" if c["proved"]
                                      else "NOT proved") + "]"
            elif "legitimate" in cert:
                tag = "  [change: " + ("legitimate" if cert["legitimate"] else "NOT legitimate") + "]"
            print(f"{verdict}  {e['id']}  {e['title']}{tag}")
            for x in r["problems"] + hard:
                print(f"  problem: {x}")
        sys.exit(0 if ok else 1)


def cai_bench_manifest(version: str = "v2") -> dict:
    """Counts and sha256 of every frozen CAI-Bench file, read from the files themselves."""
    import glob
    import hashlib
    root = os.path.join(_HERE, "benchmarks", version)
    domains, cases, variants = [], 0, 0
    whole = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(root, "*.json"))):
        with open(path, "rb") as f:
            raw = f.read()
        whole.update(raw)
        d = json.loads(raw)
        n = len(d.get("cases", []))
        v = sum(len(c.get("adversarial", [])) for c in d.get("cases", []))
        cases += n
        variants += v
        domains.append({"domain": os.path.basename(path)[:-5], "cases": n, "adversarial_variants": v,
                        "version_field": d.get("version"), "sha256": hashlib.sha256(raw).hexdigest()})
    return {"benchmark": "CAI-Bench", "version": version, "domains": len(domains), "cases": cases,
            "adversarial_variants": variants, "prompts_per_full_run": cases + variants,
            "sha256": whole.hexdigest(), "per_domain": domains,
            "run": "contradish benchmark --model <model> --provider anthropic|openai  (needs that provider's API key; "
                   "the judge defaults to the other provider)",
            "introduced_by": "Michele Joseph"}


def cmd_cai_bench(args) -> None:
    m = cai_bench_manifest(args.bench_version)
    if args.json:
        print(json.dumps(m, indent=2))
        return
    print(f"CAI-Bench {m['version']}: {m['domains']} domains, {m['cases']} cases, "
          f"{m['adversarial_variants']} adversarial variants, {m['prompts_per_full_run']} prompts per full run")
    print(f"content sha256 {m['sha256']}")
    for d in m["per_domain"]:
        print(f"  {d['domain']:<22} {d['cases']:>3} cases  {d['adversarial_variants']:>4} variants  {d['sha256'][:16]}")
    print(f"\nrun it: {m['run']}")


def _rerun_run(cert: dict) -> list:
    from contradish.versions import load_pinned, run_certificate, version_witness
    prov = cert.get("provenance", {})
    name = str(prov.get("agent", ""))
    if prov.get("agent_kind") != "scripted" or not name.startswith("witness:"):
        return ["--rerun applies to scripted witness agents only"]
    from contradish.versions import VerificationScope
    again = run_certificate(load_pinned(cert["before"]), load_pinned(cert["after"]),
                            version_witness(name.split(":", 1)[1]), name,
                            scope=VerificationScope.from_dict(cert.get("scope") or {}),
                            trust_anchors=(cert.get("authentication") or {}).get("trust_anchors"))
    probs = []
    canon = lambda obs: sorted(json.dumps(o, sort_keys=True) for o in obs)
    if canon(again["observations"]) != canon(cert["observations"]):
        probs.append("rerun: the scripted agent did not reproduce the observations")
    if again["claim"]["proved"] != cert["claim"]["proved"]:
        probs.append("rerun: the claim did not reproduce")
    return probs


def _version_agent(args):
    from contradish.action_frontier import llm_tool_agent
    from contradish.versions import version_witness
    if getattr(args, "app", None):
        mod, fn = args.app.split(":")
        sys.path.insert(0, os.getcwd())
        chat = getattr(importlib.import_module(mod), fn)
        return llm_tool_agent(chat, args.model_name or args.app), "model"
    name = (args.agent or "witness:faithful").replace("witness:", "")
    if name not in ("faithful", "rigid", "stale_permissions", "leaky"):
        sys.exit("--agent must be witness:faithful|rigid|stale_permissions|leaky (or pass --app)")
    return version_witness(name), "scripted"


def _print_diff(diff) -> None:
    from contradish.symbolic import regions
    s = diff.summary()
    print(f"{s['cells']} cells cover every situation (proved complete); redefined: {s['redefined'] or 'nothing'}")
    if not s["steps_that_change"]:
        print("no step changes anywhere: the null transition")
    for k in diff.steps:
        for r in regions(diff, k):
            print(f"\n  {k}: {r['change']}")
            for t in r["conditions"]:
                print(f"      when {t}")
    unchanged = [k for k in diff.steps if k not in s["steps_that_change"]]
    if unchanged:
        print(f"\n  proved unchanged in every situation: {', '.join(unchanged)}")


def cmd_versions(args) -> None:
    from contradish.evidence import save
    from contradish.policy_program import PolicyProgram, load_builtin_program
    from contradish.symbolic import diff_versions
    from contradish.versions import (
        VerificationScope, authenticate_transition, fixture_key, load_pinned, new_key, pin, run_certificate,
        sign_pin,
    )

    def read_pin(path):
        with open(path) as f:
            return load_pinned(json.load(f))

    if args.versions_cmd == "keygen":
        if not args.source:
            sys.exit("usage: contradish versions keygen --source SOURCE --out KEY.json")
        k = new_key(args.source)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(k, f, indent=2)
            os.chmod(args.out, 0o600)
            print(f"key for {args.source!r} -> {args.out} (secret; keep it private)")
            print(f"public key, to list under sources.{args.source}.keys: {k['public']}")
        else:
            print(json.dumps(k, indent=2))
        return
    if args.versions_cmd == "approve":
        from contradish.charter import approve_pin
        if len(args.files) != 1 or not (args.key and args.role):
            sys.exit("usage: contradish versions approve V2.pin.json --key KEY.json --role ROLE [--out F]")
        pv = read_pin(args.files[0])
        with open(args.key) as f:
            approve_pin(pv, json.load(f), args.role)
        with open(args.out or args.files[0], "w") as f:
            f.write(json.dumps(pv.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"{args.role} approved {pv.id} -> {args.out or args.files[0]}")
        return
    if args.versions_cmd == "pin":
        if len(args.files) != 1 or not (args.id and args.verified_by and args.method):
            sys.exit("usage: contradish versions pin PROGRAM.json --id ID --verified-by WHO --method HOW "
                     "[--issued-by SOURCE --supersedes V1.pin.json] [--sign KEY.json] [--out F]")
        with open(args.files[0]) as f:
            spec = json.load(f)
        spec.pop("updates", None)
        sup = read_pin(args.supersedes).digest if args.supersedes else None
        pv = pin(PolicyProgram(spec), args.id, args.verified_by, args.method, evidence=args.evidence,
                 issued_by=args.issued_by, supersedes=sup, proposed_at=args.proposed_at,
                 effective_at=args.effective_at)
        if args.sign:
            with open(args.sign) as f:
                sign_pin(pv, json.load(f))
        out = json.dumps(pv.to_dict(), indent=2, sort_keys=True)
        if args.out:
            with open(args.out, "w") as f:
                f.write(out + "\n")
            print(f"pinned {args.id} ({pv.digest}){' signed' if args.sign else ''} -> {args.out}")
        else:
            print(out)
        return
    anchors = list(args.trust)
    if args.versions_cmd == "demo":
        p, U = load_builtin_program("returns")
        u = U[args.update or "exchange_closed_late"]
        ok, _ = p.split_edits(u)
        k = fixture_key("policy_owner")
        v1 = sign_pin(pin(p, "returns-v1", "contradish fixture", "built-in example"), k)
        v2 = pin(p.apply(ok), "returns-" + u.id, "contradish fixture", "built-in example",
                 issued_by=u.source, supersedes=v1.digest)
        if u.source == "policy_owner":
            sign_pin(v2, k)
        anchors = anchors or [k["public"]]
    else:
        if len(args.files) != 2:
            sys.exit("usage: contradish versions diff|certify V1.pin.json V2.pin.json")
        v1, v2 = read_pin(args.files[0]), read_pin(args.files[1])
    auth = authenticate_transition(v1, v2, anchors)
    diff = diff_versions(v1.program, v2.program)
    if args.versions_cmd in ("diff", "demo") and not (args.agent or args.app):
        if args.json:
            print(json.dumps({"authentication": auth, "diff": diff.to_json()}, indent=2, default=str))
            return
        print(f"{v1.id} -> {v2.id}")
        print(f"  authenticated: {auth['authenticated']}   authorized: {auth['authorized']}"
              + (f"   ({'; '.join(auth['reasons'])})" if auth["reasons"] else ""))
        _print_diff(diff)
        return
    scope = VerificationScope()
    if args.scope:
        with open(args.scope) as f:
            scope = VerificationScope.from_dict(json.load(f))
    if args.trials:
        scope.trials = args.trials
    agent, kind = _version_agent(args)
    cert = run_certificate(v1, v2, agent, getattr(agent, "__name__", "agent"), kind, args.model_name,
                           scope=scope, trust_anchors=anchors)
    if args.out:
        save(cert, args.out)
    r, c = cert["result"], cert["claim"]
    print(c["statement"])
    print(f"  authenticated / authorized {c['authenticated']} / {c['authorized']}")
    print(f"  regions exercised          {r['cells_covered']}/{r['cells']} (x{r['trials_per_cell']} trials)")
    print(f"  required changes made      {r['required_changes_made']}/{r['required_changes']}")
    print(f"  unaffected constraints     {r['preserved_held']}/{r['preserved_cases']} held")
    print(f"  granted permissions used   {r['granted_permissions_exercised']}/{r['permissions_granted']} (optional)")
    for f in cert["failures"][:5]:
        print(f"    - {f.get('step', '')}: {f['problem']} when {f['condition']}")
    for reason in cert["authentication"]["reasons"]:
        print(f"    - authentication: {reason}")
    if args.out:
        print(f"  certificate: {args.out}   (independent check: contradish evidence check {args.out})")


def cmd_perspectives(args) -> None:
    from contradish.versions import atlas, list_perspectives, load_perspectives, perspective_matrix, version_witness
    if args.perspectives_cmd == "list":
        for n in list_perspectives():
            about, frames = load_perspectives(n)
            print(f"{n}: {about['title']} ({len(frames)} frames)")
            for fr in frames:
                print(f"   {fr.id:<20} {fr.program.spec['title']}")
        return
    about, frames = load_perspectives(args.name)
    if args.perspectives_cmd == "atlas":
        a = atlas(frames)
        if args.json:
            print(json.dumps(a, indent=2, default=str))
            return
        print(about["title"])
        print(f"  {about['about']}\n")
        print(f"{a['cells']} cells x {len(a['per_step'])} actions: {a['consensus_cases']} consensus, "
              f"{a['contested_cases']} contested")
        for k, v in a["per_step"].items():
            print(f"\n  {k}: consensus {v['consensus']}, contested {v['contested']}, "
                  f"allowed by every frame {v['allowed_by_all']}, required by every frame {v['required_by_all']}")
            for bloc, n in sorted(v["blocs"].items(), key=lambda t: -t[1]):
                print(f"      {n:>3}  {bloc}")
        ids = [f.id for f in frames]
        w = max(len(i) for i in ids)
        print("\n  disagreement distance (share of cases where two frames' norms differ; a pseudometric)")
        for i in ids:
            print("   " + i.ljust(w) + "  " + " ".join(f"{a['distance'][i][j]:.2f}" for j in ids))
        return
    name = args.agent.replace("witness:", "")
    if args.app:
        from contradish.action_frontier import llm_tool_agent
        mod, fn = args.app.split(":")
        sys.path.insert(0, os.getcwd())
        chat = getattr(importlib.import_module(mod), fn)
        factory = lambda: llm_tool_agent(chat, args.app)
    else:
        factory = lambda: version_witness(name)
    m = perspective_matrix(frames, factory)
    ids = [f.id for f in frames]
    w = max(len(i) for i in ids)
    proved = sum(v["proved"] for v in m.values())
    print(f"frame switching, agent {args.agent if not args.app else args.app}: {proved}/{len(m)} transitions proved faithful")
    print("   frames: " + ", ".join(f"[{i + 1}] {x}" for i, x in enumerate(ids)))
    print("   from \\ to " + " ".join(f"[{j + 1}]".rjust(4) for j in range(len(ids))))
    for i, x in enumerate(ids):
        print(f"   [{i + 1}]".ljust(13) + " " + " ".join((" yes" if m[(x, y)]["proved"] else "  NO") for y in ids))


def _print_legitimacy(c: dict, desc: str = "") -> None:
    print(c["statement"])
    if desc:
        print(f"  ({desc})")
    ck = c["checks"]
    mark = lambda b: "pass" if b else "FAIL"
    a = c["authentication"]
    print(f"  provenance           {mark(ck['provenance'])}" + ("" if ck["provenance"] else f"  {'; '.join(a['reasons'])}"))
    pr = c["procedure"]
    print(f"  procedure            {mark(ck['procedure'])}  approved by {pr['approved_by'] or 'nobody'}")
    for r in pr["rules"]:
        print(f"      {r['id']}: {len(r['approved_by'])}/{r['approvals_required']} of {r['approvers']}"
              f"{'' if r['approvals_ok'] else ' (too few)'}; review {r['review_days_required']}d "
              f"{'ok' if r['review_ok'] else 'NOT met'}")
    e = c["effects"]
    cl = c["clause_level_authority"]["status"]
    print(f"  authority by effect  {mark(ck['authority_by_effect'])}  issuer {e['issuer']!r} granted {e['granted_domains']}"
          f"   (clause-level authority alone: {cl})")
    for x in e["unauthorized"][:6]:
        print(f"      outside grant: {x['step']} {x['channel']} {x['change']} {x['args'] or ''} in {x['regions']} regions,"
              f" licensable only by {x['domains'] or 'no domain'}; e.g. {x['example']}")
    print(f"  invariants           {mark(ck['invariants'])}  (proved over all {e['regions']} regions)")
    for i in c["invariants"]:
        line = f"      {'entrenched ' if i['entrenched'] else ''}{i['id']}: {'holds' if i['holds'] else 'VIOLATED'}"
        if not i["holds"]:
            v = i["violations"][0]
            line += f" at {v['situation']} ({v['found']})"
        print(line)
    if c["charter_amendment"]:
        am = c["charter_amendment"]
        print(f"  charter amendment    {mark(ck['charter_amendment'])}  entrenched altered/removed: "
              f"{am['entrenched_altered_or_removed'] or 'none'}")


def cmd_legitimacy(args) -> None:
    from contradish.charter import DEMO_CASES, demo_transition, legitimacy_certificate
    from contradish.evidence import save
    from contradish.versions import load_pinned
    if args.legitimacy_cmd == "demo":
        cases = [args.case] if args.case else list(DEMO_CASES)
        for case in cases:
            v1, v2, anchors, desc = demo_transition(case)
            c = legitimacy_certificate(v1, v2, anchors)
            if args.json:
                print(json.dumps(c, indent=2, default=str))
            else:
                print(f"\n[{case}]")
                _print_legitimacy(c, desc)
            if args.out and len(cases) == 1:
                save(c, args.out)
        return
    if len(args.files) != 2:
        sys.exit("usage: contradish legitimacy check V1.pin.json V2.pin.json --trust PUBKEY [--out F]")
    with open(args.files[0]) as f:
        v1 = load_pinned(json.load(f))
    with open(args.files[1]) as f:
        v2 = load_pinned(json.load(f))
    c = legitimacy_certificate(v1, v2, args.trust)
    if args.out:
        save(c, args.out)
    if args.json:
        print(json.dumps(c, indent=2, default=str))
    else:
        _print_legitimacy(c)
        if args.out:
            print(f"  certificate: {args.out}   (independent check: contradish evidence check {args.out})")


def cmd_justify(args) -> None:
    from contradish import justification as J
    from contradish.policy_program import Call, PolicyProgram, load_builtin_program
    if args.program.endswith(".json") or os.path.sep in args.program:
        with open(args.program) as f:
            spec = json.load(f)
        spec.pop("updates", None)
        program = PolicyProgram(spec)
        default_auth, irreversible = {}, set(spec.get("irreversible_tools", []))
    else:
        program = load_builtin_program(args.program)[0]
        default_auth, irreversible = J.RETURNS_FACT_AUTHORITY, J.RETURNS_IRREVERSIBLE

    def state():
        if not args.state:
            sys.exit("--state is required")
        with open(args.state) as f:
            st = J.InformationState.from_dict(json.load(f), default_auth)
        if not st.fact_authority:
            sys.exit("no fact authority: add \"fact_authority\" to the state file")
        return st

    if args.justify_cmd == "demo":
        st = J.InformationState([J.Evidence("category", "order_system", value="electronics"),
                                 J.Evidence("price", "order_system", value=100.0),
                                 J.Evidence("days_since_delivery", "customer", value=12)], default_auth)
        calls = J.scripted_agent("assumes_defaults", program)(st)
        g = J.check_actions(program, st, calls, irreversible)
        print("Recorded about the case:")
        print(st.render())
        print("\nThe agent proposes: " + ", ".join(f"{c.tool}({', '.join(f'{k}={v}' for k, v in c.args)})" for c in calls))
        print(f"Gate: {g.decision}" + (" (it acted as if it knew enough; it did not)" if g.unjustified_certainty else ""))
        for c in g.checks:
            if c.proposed is not None:
                print(f"  {c.step:<9} {c.verdict:<12} permitted in {c.permitted_in} of {c.of} situations still possible -> {c.decision}")
        if g.plan:
            print()
            print(g.plan.render(program))
        print("\nScripted agents on paired states (each pair differs in one fact: established, or missing / vague / only claimed):")
        sc = J.build_scenarios(program, default_auth, irreversible)
        print(f"  {'agent':<18}{'awareness':>10}{'commits|enough':>16}{'commits|not':>13}{'lucky':>8}{'blocked':>9}{'plans restore':>15}")
        for m in J.SCRIPTED:
            r = J.evaluate_agent(program, J.scripted_agent(m, program), sc, irreversible)
            fmt = lambda x: "-" if x is None else f"{x:.2f}"
            print(f"  {m:<18}{fmt(r['sufficiency_awareness_index']):>10}{fmt(r['commits_when_sufficient']):>16}"
                  f"{fmt(r['commits_when_insufficient']):>13}{fmt(r['lucky_share_of_unjustified']):>8}"
                  f"{fmt(r['gate_blocked']):>9}{fmt(r['plans_restore']):>15}")
        print(f"  ({len(sc)} pairs. 'lucky': unjustified commitments that happened to match the true situation, which"
              " outcome scoring would pass.)")
        return

    if args.justify_cmd == "check":
        st = state()
        with open(args.calls) as f:
            raw = json.load(f)
        calls = [Call.of(d["tool"], d.get("args") or {}) for d in raw]
        g = J.check_actions(program, st, calls, irreversible)
        if args.json:
            print(json.dumps(g.to_dict(), indent=2, default=str))
        else:
            print(f"Gate: {g.decision}")
            for c in g.checks:
                if c.proposed is not None:
                    print(f"  {c.step:<9} {c.verdict:<12} permitted in {c.permitted_in} of {c.of} -> {c.decision}")
            if g.plan:
                print(g.plan.render(program))
        sys.exit(1 if g.decision == J.BLOCK else 2 if g.decision == J.REVIEW else 0)

    if args.justify_cmd == "plan":
        st = state()
        steps = args.steps.split(",") if args.steps else program.step_ids()
        pl = J.verification_plan(program, st, steps)
        print(json.dumps(pl.to_dict(), indent=2, default=str) if args.json else pl.render(program))
        return

    if args.justify_cmd == "evaluate":
        if args.app:
            mod, fn = args.app.split(":")
            sys.path.insert(0, os.getcwd())
            chat = getattr(importlib.import_module(mod), fn)
            agent = J.llm_justified_agent(chat, program, args.model_name or args.app)
        else:
            mode = (args.agent or "").replace("scripted:", "")
            if mode not in J.SCRIPTED:
                sys.exit(f"--agent must be one of {['scripted:' + m for m in J.SCRIPTED]} (or pass --app)")
            agent = J.scripted_agent(mode, program)
        sc = J.build_scenarios(program, default_auth, irreversible, limit=args.limit)
        r = J.evaluate_agent(program, agent, sc, irreversible)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(r, f, indent=1, default=str)
        summary = {k: v for k, v in r.items() if k != "rows"}
        print(json.dumps(summary, indent=2))
        return


def cmd_entitled(args) -> None:
    from contradish import entitlement as E
    from contradish.justification import Evidence, InformationState
    from contradish.policy_program import Call

    if args.entitled_cmd == "demo":
        print("A patient asks for a refill on day 22 of a 30-day supply. The pharmacy's own systems have established")
        print("every fact. The insurer lowered the early-refill threshold to 70% (day 21), effective 2026-10-01.\n")
        g = E.governing_version(args.policy, args.as_of)
        a = g.fact_authority
        st = InformationState([Evidence("days_since_fill", "pharmacy_record", value=22),
                               Evidence("refills_remaining", "pharmacy_record", value=1),
                               Evidence("schedule", "pharmacy_record", value="none"),
                               Evidence("prescriber_reachable", "prescriber_office", value=True),
                               Evidence("identity_verified", "identity_check", value=True)], a)
        for mode in ("stale", "over_cautious"):
            calls = E.scripted_agent(mode)(g, st)
            print(f"-- scripted:{mode} proposes: " + (", ".join(c.tool for c in calls) or "nothing"))
            print(E.check_entitlement(g, st, calls).render())
            print()
        print("Scripted agents across every situation, three dates and three evidence states (1 = never withholds what is")
        print("owed, never refuses without asking for what would settle it):")
        sc = E.build_scenarios(args.policy, limit=args.limit or 3000)
        print(f"  {'agent':<20}{'guard':>7}{'entitlement':>13}{'settlement':>12}{'withheld when owed':>20}")
        for m in E.SCRIPTED:
            r = E.evaluate_agent(E.scripted_agent(m), sc, args.policy)
            print(f"  {m:<20}{r['withholding_guard']:>7.2f}{r['entitlement_index']:>13.2f}{r['settlement_index']:>12.2f}"
                  f"{r['withheld_when_entitled']:>20.2f}")
        return

    if args.entitled_cmd == "check":
        if not (args.state and args.calls):
            sys.exit("--state and --calls are required")
        g = E.governing_version(args.policy, args.as_of)
        with open(args.state) as f:
            st = InformationState.from_dict(json.load(f), g.fact_authority)
        with open(args.calls) as f:
            calls = [Call.of(d["tool"], d.get("args") or {}) for d in json.load(f)]
        gate = E.check_entitlement(g, st, calls)
        print(json.dumps(gate.to_dict(), indent=2, default=str) if args.json else gate.render())
        sys.exit({"ALLOW": 0, "DELIVER": 1, "VERIFY": 1}.get(gate.decision, 2))

    if args.entitled_cmd == "evaluate":
        if args.app:
            mod, fn = args.app.split(":")
            sys.path.insert(0, os.getcwd())
            agent = E.llm_entitlement_agent(getattr(importlib.import_module(mod), fn), args.model_name or args.app)
        else:
            mode = (args.agent or "").replace("scripted:", "")
            if mode not in E.SCRIPTED:
                sys.exit(f"--agent must be one of {['scripted:' + m for m in E.SCRIPTED]} (or pass --app)")
            agent = E.scripted_agent(mode)
        r = E.evaluate_agent(agent, E.build_scenarios(args.policy, limit=args.limit), args.policy)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(r, f, indent=1, default=str)
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2))


def _guard_draft(args) -> None:
    from contradish import drafting as D
    tpl = os.path.join(os.path.dirname(__file__), "templates")
    if args.demo:
        with open(os.path.join(tpl, "drafting_demo.json")) as f:
            demo = json.load(f)
        src = args.target or os.path.join(tpl, "refill_policy.md")
        encoder = lambda m: json.dumps(demo["encoder"])       # noqa: E731
        examiner = lambda m: json.dumps(demo["examiner"])     # noqa: E731
        models = {"encoder": "scripted stand-in (demo)", "examiner": "scripted stand-in (demo)"}
        print("Demo: the encoder and examiner replies are scripted stand-ins, fixed so this runs offline. The")
        print("encoder's reply has one deliberate error. Everything after the replies is the real pipeline.\n")
    else:
        if not args.target:
            sys.exit("contradish guard draft POLICY_TEXT --model provider:model  (or --demo)")
        if not args.model:
            sys.exit("contradish guard draft: --model provider:model is required (or --demo)")
        src = args.target
        encoder = D.chat_for_model(args.model)
        examiner = D.chat_for_model(args.examiner or args.model)
        models = {"encoder": args.model, "examiner": args.examiner or args.model}
    with open(src, encoding="utf-8") as f:
        text = f.read()
    tools = [t.strip() for t in args.tools.split(",")] if args.tools else None
    d = D.draft_policy(text, encoder, tools=tools, examiner=examiner, models=models)
    out = args.out or ("policy.draft.json" if args.demo else os.path.splitext(src)[0] + ".policy.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d.spec, f, indent=2, ensure_ascii=False)
        f.write("\n")
    review = os.path.splitext(out)[0] + ".review.md"
    with open(review, "w", encoding="utf-8") as f:
        f.write(D.review_markdown(d.spec, text, d) + "\n")
    n_rules = len(d.spec.get("clauses") or {})
    print(f"  drafted {n_rules} rules over {len(d.spec.get('facts') or {})} facts from {src}")
    print(f"  {len(d.errors)} error(s): source quotes not in the text, numbers not in their quote, or a policy that does not load")
    print(f"  {len(d.disagreements)} disagreement(s) with an independent reading of the text")
    print(f"  {len(d.boundary)} boundary question(s) for a person to confirm")
    print(f"  {len(d.uncovered)} sentence(s) of the text no rule cites")
    for x in d.disagreements[:5]:
        print(f"    disagreement: {x['case']}: missing {x.get('missing')}, unexpected {x.get('unexpected')}")
    print(f"\n  wrote {out} (status: draft; the guard will not enforce it)")
    print(f"  wrote {review}")
    print(f"\nNext: work through {review}, fix {out}, then\n"
          f"  contradish guard review {out}\n  contradish guard approve {out} --by \"Your Name\" --role \"...\"")


def cmd_guard(args) -> None:
    from contradish import guard as G

    cmd = args.guard_cmd
    if cmd == "demo":
        import tempfile
        from contradish import entitlement as E
        d = tempfile.mkdtemp(prefix="contradish-guard-")
        log = os.path.join(d, "decisions.jsonl")
        print("A pharmacy's refill agent runs behind the guard in shadow mode for three dates, across every situation")
        print("the policy distinguishes, with five scripted agents of known behavior standing in for models.\n")
        g = G.Guard("pharmacy_refills", mode="shadow", log=log)
        import random
        rng = random.Random(7)
        for sc in E.build_scenarios(limit=600):
            gv = E.governing_version("pharmacy_refills", sc["as_of"])
            calls = E.scripted_agent(rng.choice(E.SCRIPTED))(gv, sc["state"])
            g.check(sc["state"].evidence, calls, as_of=sc["as_of"], request_id=sc["id"])
        md, _ = G.value_report(log)
        print(md)
        print(f"\nThe log is {log}. Re-check it yourself: python {os.path.join(os.path.dirname(__file__), 'evidence_check.py')} {log}")
        return

    if cmd == "init":
        import shutil
        out = args.target or "contradish_guard"
        os.makedirs(out, exist_ok=True)
        tpl = os.path.join(os.path.dirname(__file__), "templates")
        for name in ("starter_policy.json", "guard_example.py"):
            dst = os.path.join(out, "policy.json" if name == "starter_policy.json" else name)
            if os.path.exists(dst):
                print(f"  kept existing {dst}")
            else:
                shutil.copy(os.path.join(tpl, name), dst)
                print(f"  wrote {dst}")
        print(f"\nNext: edit {out}/policy.json (facts, who establishes them, clauses, examples), then\n"
              f"  contradish guard lint {out}/policy.json\n  contradish guard test {out}/policy.json\n"
              f"  python {out}/guard_example.py\nand see docs/INTEGRATE.md.")
        return

    if cmd == "draft":
        return _guard_draft(args)

    if not args.target:
        sys.exit(f"contradish guard {cmd}: give the "
                 f"{'policy' if cmd in ('lint', 'test', 'check', 'review', 'approve') else 'log'} file")

    if cmd == "review":
        from contradish import drafting as D
        with open(args.target) as f:
            spec = json.load(f)
        text = None
        if args.text:
            with open(args.text, encoding="utf-8") as f:
                text = f.read()
        md = D.review_markdown(spec, text)
        out = args.out or os.path.splitext(args.target)[0] + ".review.md"
        with open(out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        b = D.blockers(spec)
        print(f"  status: {D.approval_status(spec)['state']}")
        print(f"  {len(b['errors'])} error(s) to fix, {len(b['disagreements'])} disagreement(s) to decide")
        print(f"  wrote {out}")
        sys.exit(1 if b["errors"] or b["disagreements"] else 0)

    if cmd == "approve":
        from contradish import drafting as D
        if not args.by:
            sys.exit("contradish guard approve: --by NAME is required (the person taking responsibility)")
        with open(args.target) as f:
            spec = json.load(f)
        try:
            out = D.approve(spec, args.by, args.role, args.note, despite_disagreements=args.despite_disagreements)
        except ValueError as e:
            sys.exit(f"  {e}")
        with open(args.target, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            f.write("\n")
        ap = out["approval"]
        print(f"  approved by {ap['by']}" + (f" ({ap['role']})" if ap["role"] else "") + f" at {ap['at']}")
        print(f"  bound to {ap['digest']}: any later change to the rules or updates needs a new approval")
        if ap.get("overrode"):
            print(f"  recorded: approved despite {len(ap['overrode'])} disagreement(s): {', '.join(map(str, ap['overrode']))}")
        return

    if cmd == "lint":
        issues = G.check_policy(args.target)
        for lvl, msg in issues:
            print(f"  {lvl.upper():7s} {msg}")
        sys.exit(1 if any(l == "error" for l, _ in issues) else 0)

    if cmd == "test":
        ex = None
        if args.examples:
            with open(args.examples) as f:
                ex = json.load(f)
        problems = G.run_examples(args.target, ex)
        n = len(ex) if ex is not None else len(G.load_policy_examples(args.target))
        for name, msg in problems:
            print(f"  FAIL  {name}: {msg}")
        print(f"  {n - len({p[0] for p in problems})} of {n} examples pass")
        sys.exit(1 if problems else 0)

    if cmd == "check":
        if not (args.evidence and args.calls):
            sys.exit("--evidence and --calls are required")
        with open(args.evidence) as f:
            ev = json.load(f)
        with open(args.calls) as f:
            calls = json.load(f)
        d = G.Guard(args.target, mode=args.mode, log=args.log).check(ev, calls, as_of=args.as_of)
        print(json.dumps(d.to_dict(), indent=2, default=str) if args.json else d.explain())
        sys.exit({"allow": 0, "deliver": 1, "block": 1}.get(d.outcome, 2))

    if cmd == "verify":
        from contradish.evidence_check import main as check_main
        sys.exit(check_main([args.target] + (["--json"] if args.json else [])))

    if cmd == "report":
        md, summary = G.value_report(args.target, verify=not args.no_verify)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(md + "\n")
            print(f"  wrote {args.out}")
        print(json.dumps(summary, indent=2) if args.json else md)
