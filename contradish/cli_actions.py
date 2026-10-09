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

ABOUT = {
    "name": "contradish",
    "top_line": "Contradish measures whether AI transitions remain faithful to their governing information.",
    "principle": "Change what truth requires. Preserve what truth does not require changing.",
    "definition": ("Behavioral Update Fidelity measures whether an AI changes its behavior exactly when, "
                   "and only as far as, changes in governing information warrant."),
    "author": "Michele Joseph",
    "attribution": "Behavioral Update Fidelity was introduced by Michele Joseph in 2026.",
    "also_introduced_by_the_author": ["CAI Strain", "CAI-Bench", "the policy evaluation contract",
                                      "the transition contract", "action-level transition verification",
                                      "whole-run warranted-transition proofs", "the perspective atlas"],
    "repository": "https://github.com/michelejoseph/contradish",
    "website": "https://contradish.com",
    "license": "MIT",
    "cite": "CITATION.cff / CITATION.bib in the repository (key: joseph2026buf)",
    "spec": "docs/BUF-SPEC.md",
    "entry_points": {
        "derive what must change": "contradish actions derive --update restocking_fee_20",
        "verify an agent": "contradish actions verify --update customer_claims --app mymodule:chat",
        "complete difference of two versions": "contradish versions demo --update exchange_closed_late",
        "whole-run proof": "contradish versions certify v1.pin.json v2.pin.json --app mymodule:chat",
        "alternative frames": "contradish perspectives atlas",
        "inspect a verified failure": "contradish exhibits show EX-0001",
        "re-verify independently": "contradish exhibits verify",
        "run CAI-Bench": "contradish benchmark --model <model> --provider anthropic|openai",
    },
}

_HERE = os.path.dirname(__file__)
EXHIBITS_DIR = os.path.join(_HERE, "exhibits")


def register(sub) -> None:
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
    vs.add_argument("versions_cmd", choices=["pin", "diff", "certify", "demo"])
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
    vs.add_argument("--out", default=None)
    vs.add_argument("--json", action="store_true", default=False)

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


def _program_args(p) -> None:
    p.add_argument("--program", default="returns", help="Built-in program name or a .json path (default: returns).")
    p.add_argument("--update", default=None, help="Built-in update id or an update .json path.")


COMMANDS = ("about", "actions", "counterexample", "evidence", "exhibits", "cai-bench", "versions", "perspectives")


def dispatch(args) -> bool:
    if args.command not in COMMANDS:
        return False
    {"about": cmd_about, "actions": cmd_actions, "counterexample": cmd_counterexample,
     "evidence": cmd_evidence, "exhibits": cmd_exhibits, "cai-bench": cmd_cai_bench,
     "versions": cmd_versions, "perspectives": cmd_perspectives}[args.command](args)
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
    if cert.get("schema") == "contradish.run_certificate/1.0":
        return _rerun_run(cert)
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
    from contradish.evidence_check import check, check_run
    results = []
    for path in args.files:
        with open(path) as f:
            cert = json.load(f)
        r = check_run(cert) if cert.get("schema") == "contradish.run_certificate/1.0" else check(cert)
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
            print(f"{r['verdict']}  {r['file']}")
            print(f"  claim: {r.get('claim')}")
            for x in r["problems"]:
                print(f"  problem: {x}")
            for x in r["notes"]:
                print(f"  note: {x}")
    sys.exit(0 if all(r["verdict"] == "VERIFIED" for r in results) else 1)


def _index() -> list:
    with open(os.path.join(EXHIBITS_DIR, "index.json")) as f:
        return json.load(f)["exhibits"]


def cmd_exhibits(args) -> None:
    from contradish.evidence_check import check as _check1, check_run

    def check(cert):
        return check_run(cert) if cert.get("schema") == "contradish.run_certificate/1.0" else _check1(cert)
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
            print(f"{verdict}  {e['id']}  {e['title']}")
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
    again = run_certificate(load_pinned(cert["before"]), load_pinned(cert["after"]),
                            version_witness(name.split(":", 1)[1]), name)
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
    from contradish.versions import load_pinned, pin, run_certificate, transition_authority

    def read_pin(path):
        with open(path) as f:
            return load_pinned(json.load(f))

    if args.versions_cmd == "pin":
        if len(args.files) != 1 or not (args.id and args.verified_by and args.method):
            sys.exit("usage: contradish versions pin PROGRAM.json --id ID --verified-by WHO --method HOW [--out F]")
        with open(args.files[0]) as f:
            spec = json.load(f)
        spec.pop("updates", None)
        pv = pin(PolicyProgram(spec), args.id, args.verified_by, args.method, evidence=args.evidence,
                 issued_by=args.issued_by)
        out = json.dumps(pv.to_dict(), indent=2, sort_keys=True)
        if args.out:
            with open(args.out, "w") as f:
                f.write(out + "\n")
            print(f"pinned {args.id} ({pv.digest}) -> {args.out}")
        else:
            print(out)
        return
    if args.versions_cmd == "demo":
        p, U = load_builtin_program("returns")
        u = U[args.update or "exchange_closed_late"]
        ok, _ = p.split_edits(u)
        v1 = pin(p, "returns-v1", "contradish fixture", "built-in example")
        v2 = pin(p.apply(ok), "returns-" + u.id, "contradish fixture", "built-in example", issued_by=u.source)
    else:
        if len(args.files) != 2:
            sys.exit("usage: contradish versions diff|certify V1.pin.json V2.pin.json")
        v1, v2 = read_pin(args.files[0]), read_pin(args.files[1])
    diff = diff_versions(v1.program, v2.program)
    if args.versions_cmd in ("diff", "demo") and not (args.agent or args.app):
        if args.json:
            print(json.dumps(diff.to_json(), indent=2, default=str))
            return
        print(f"{v1.id} -> {v2.id}   authority: {transition_authority(v1, v2)['status']}")
        _print_diff(diff)
        return
    agent, kind = _version_agent(args)
    cert = run_certificate(v1, v2, agent, getattr(agent, "__name__", "agent"), kind, args.model_name)
    if args.out:
        save(cert, args.out)
    r, c = cert["result"], cert["claim"]
    print(f"{v1.id} -> {v2.id}, agent {cert['provenance']['agent']}")
    print(f"  cells exercised          {r['cells_covered']}/{r['cells']}")
    print(f"  required changes made    {r['required_changes_made']}/{r['required_changes']}")
    print(f"  unrelated obligations    {r['preserved_held']}/{r['preserved_cases']} held")
    print(f"  granted permissions used {r['granted_permissions_exercised']}/{r['permissions_granted']} (optional)")
    print(f"  PROVED: {c['proved']}")
    for f in cert["failures"][:5]:
        print(f"    - {f.get('step', '')}: {f['problem']} when {f['condition']}")
    if args.out:
        print(f"  certificate: {args.out}   (verify: contradish evidence check {args.out})")


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
