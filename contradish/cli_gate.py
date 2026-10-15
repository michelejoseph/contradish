"""`contradish gate`: score a compiled policy gate across a policy amendment."""
from __future__ import annotations

import argparse
import json
import os
import sys

COMMANDS = ("gate",)


def register(sub) -> None:
    g = sub.add_parser(
        "gate",
        help="Did a compiled policy gate change exactly where a policy amendment warrants? Exhaustive, no model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Score a pair of compiled tool-call gates (before and after a policy amendment) against the\n"
            "change the amendment warrants, over every situation in the program's exact partition.\n\n"
            "  contradish gate check --program returns --update window_45 --before old.json --after new.json\n"
            "  contradish gate check --program returns --update customer_claims --mutant obeys_everything\n"
            "  contradish gate compile --program returns [--update window_45] > gate.json\n"
            "  contradish gate demo [--program returns] [--updates window_45,customer_claims]\n\n"
            "A gate file is JSON rules ({\"rules\": [{\"tool\", \"allow_when\", \"args\"?, \"required_when\"?}]});\n"
            "see contradish/gate_fidelity.py. --program and --update take a built-in name or a JSON path."
        ),
    )
    g.add_argument("gate_action", choices=["check", "compile", "demo"])
    g.add_argument("--program", default="returns")
    g.add_argument("--update", default=None, help="Update id in the program, or a ProgramUpdate JSON file.")
    g.add_argument("--updates", default=None, help="demo: comma-separated update ids (default: all).")
    g.add_argument("--before", default=None, help="check: the gate compiled from the old policy (JSON rules).")
    g.add_argument("--after", default=None, help="check: the gate compiled after the amendment (JSON rules).")
    g.add_argument("--mutant", default=None, help="check: use a built-in gate pair instead of files.")
    g.add_argument("--obligations", action="store_true", default=False,
                   help="Also check required calls (both gates must define required_when).")
    g.add_argument("--json", action="store_true", default=False)


def dispatch(args) -> bool:
    if args.command not in COMMANDS:
        return False
    cmd_gate(args)
    return True


def _program_and_updates(name: str):
    from contradish.policy_program import load_builtin_program, list_builtin_programs, PolicyProgram, ProgramUpdate
    if os.path.exists(name):
        with open(name, encoding="utf-8") as f:
            spec = json.load(f)
        updates = {u["id"]: ProgramUpdate.from_dict(u) for u in spec.pop("updates", [])}
        return PolicyProgram(spec), updates
    if name not in list_builtin_programs():
        sys.exit(f"  no program {name!r}; built-in: {', '.join(list_builtin_programs())}")
    return load_builtin_program(name)


def _update(arg: str, updates: dict):
    from contradish.policy_program import ProgramUpdate
    if arg is None:
        sys.exit(f"  --update is required; this program has: {', '.join(updates)}")
    if os.path.exists(arg):
        with open(arg, encoding="utf-8") as f:
            return ProgramUpdate.from_dict(json.load(f))
    if arg not in updates:
        sys.exit(f"  no update {arg!r}; this program has: {', '.join(updates)}")
    return updates[arg]


def cmd_gate(args) -> None:
    from contradish import gate_fidelity as G
    program, updates = _program_and_updates(args.program)

    if args.gate_action == "compile":
        p = program
        if args.update:
            u = _update(args.update, updates)
            authorized, _ = program.split_edits(u)
            p = program.apply(authorized)
        print(json.dumps(G.compile_rules(p), indent=2))
        return

    if args.gate_action == "demo":
        ids = [x.strip() for x in args.updates.split(",")] if args.updates else list(updates)
        rows = []
        print(f"\n  GATE AMENDMENT FIDELITY  *  {program.id}: {len(ids)} amendments x {len(G.MUTANTS)} gate pairs\n")
        print(f"  {'amendment':24} {'gate pair':17} {'probes':>6} {'change':>7} {'stale':>6} {'overreach':>9} "
              f"{'captured':>8}  verified")
        for uid in ids:
            u = _update(uid, updates)
            for kind in G.MUTANTS:
                b, a = G.mutant_gates(kind, program, u)
                r = G.evaluate_gates(program, u, b, a, obligations=args.obligations)
                s = r.summary()
                c = s["counts"]
                rows.append({"update": uid, "mutant": kind, **s})
                print(f"  {uid:24} {kind:17} {s['probes']:6d} {s['must_change']:7d} {c['rigid']:6d} "
                      f"{c['drift']:9d} {c['captured']:8d}  {'yes' if s['verified'] else 'no'}")
        if args.json:
            print(json.dumps(rows, indent=2))
        print("\n  faithful and compiled must be verified on every amendment; stale, overreach and\n"
              "  obeys_everything must fail wherever the amendment gives them something to get wrong.\n")
        return

    u = _update(args.update, updates)
    if args.mutant:
        if args.mutant not in G.MUTANTS:
            sys.exit(f"  --mutant must be one of {', '.join(G.MUTANTS)}")
        before, after = G.mutant_gates(args.mutant, program, u)
    else:
        if not (args.before and args.after):
            sys.exit("  check needs --before and --after gate files, or --mutant")
        before, after = G.load_gate(args.before, program), G.load_gate(args.after, program)
    r = G.evaluate_gates(program, u, before, after, obligations=args.obligations or None)
    if args.json:
        print(json.dumps(r.to_dict(), indent=2, default=str))
    else:
        print()
        print(r.render())
        print()
    sys.exit(0 if r.verified else 1)
