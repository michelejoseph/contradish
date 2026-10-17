"""`contradish agency`: who can know, decide, act, challenge and correct after an action?"""
from __future__ import annotations

import argparse
import json
import sys

COMMANDS = ("agency",)


def register(sub) -> None:
    p = sub.add_parser(
        "agency",
        help="Evaluate what an action does to the affected people's ability to know, decide, act, challenge "
             "and correct.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Whether an action was authorized and whether it met its objective are two questions. A third is\n"
            "what it did to the people it affects. Each option declares which of five dimensions it grants or\n"
            "removes for which party (know, decide, act, challenge, correct). A loss is warranted only when a\n"
            "source with authority over it requires it: the party itself, or a source that declares it\n"
            "governs `agency:<party>:<dimension>`. A requirement with authority is also binding.\n\n"
            "  contradish agency demo                  built-in cases, every option evaluated\n"
            "  contradish agency template > case.json  a case file to fill in\n"
            "  contradish agency check case.json       evaluate the case's \"chosen\" option (exit 1 if not governed)\n"
            "  contradish agency check case.json --json"),
    )
    p.add_argument("agency_action", choices=["demo", "template", "check"])
    p.add_argument("file", nargs="?", default=None)
    p.add_argument("--json", action="store_true", default=False)


def dispatch(args) -> bool:
    if args.command not in COMMANDS:
        return False
    from contradish import agency as A
    if args.agency_action == "demo":
        print()
        print(A.demo())
        return True
    if args.agency_action == "template":
        print(json.dumps(A.TEMPLATE, indent=2))
        return True
    if not args.file:
        sys.exit("  usage: contradish agency check CASE.json")
    try:
        v = A.check_file(args.file)
    except (ValueError, KeyError) as e:
        sys.exit(f"  {e}")
    print(json.dumps(v.to_dict(), indent=2) if args.json else v.summary())
    if v.verdict != "governed":
        sys.exit(1)
    return True
