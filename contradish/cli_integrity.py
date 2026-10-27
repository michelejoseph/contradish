"""CLI for the decision-integrity layer: `contradish integrity ...` and `contradish gateway ...`."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings

COMMANDS = ("integrity", "gateway", "license", "memory", "transform")


def register(sub) -> None:
    ip = sub.add_parser(
        "integrity",
        help="Did the AI change exactly what it should, for the right reasons, at the right time? Compile a change "
             "plan from the policy file, verify any models against it, gate the release, keep a signed audit ledger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  contradish integrity demo                                   the four layers offline, in seconds\n"
            "  contradish integrity plan POLICY --as-of D [--since D]      which actions must change, stay, resist, wait\n"
            "  contradish integrity release POLICY --as-of D --models M1,M2 [--ledger L --key K]\n"
            "        M = provider:model (openai:gpt-4.1, anthropic:claude-sonnet-4-5, gemini:..., ollama:...),\n"
            "            or witness:faithful|rigid|credulous|overreach. Exit 0 if every model ships, 1 if any is held.\n"
            "  contradish integrity ledger verify L [--trust PUBKEY]       re-check the chain, signatures, every decision\n"
            "  contradish integrity ledger checkpoint L --key K            sign the current head\n\n"
            "Keys: contradish versions keygen --source auditor --out auditor.key.json. See docs/INTEGRITY.md."
        ),
    )
    ip.add_argument("integrity_cmd", choices=["demo", "plan", "release", "ledger"])
    ip.add_argument("target", nargs="*", default=[])
    ip.add_argument("--as-of", dest="as_of", default=None, help="the release date: decides what is in force")
    ip.add_argument("--since", default=None, help="the previous release date: updates in force then are the baseline")
    ip.add_argument("--updates", default=None, help="comma-separated update ids (default: all after --since)")
    ip.add_argument("--models", default=None)
    ip.add_argument("--max-situations", dest="max_situations", type=int, default=60)
    ip.add_argument("--seed", type=int, default=0)
    ip.add_argument("--gate", default=None, help="e.g. complete=0.95,minimal=0.99,authorized=0.99,timely=0.99")
    ip.add_argument("--ledger", default=None)
    ip.add_argument("--key", default=None)
    ip.add_argument("--trust", action="append", default=None)
    ip.add_argument("--by", default=None, help="release: who recorded the release")
    ip.add_argument("--out", default=None)
    ip.add_argument("--json", action="store_true", default=False)

    lp = sub.add_parser(
        "license",
        help="What must a representation (memory, summary, case file) keep to decide safely in the future, not only "
             "today? Computed from the policy file and the changes it makes possible.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  contradish license demo                                  two notes that agree today; one fails tomorrow\n"
            "  contradish license POLICY --as-of D                      per fact: what today needs, what the future adds\n"
            "  contradish license check POLICY --case C.json --note N.json --as-of D\n"
            "        is this note sufficient now, and for every future version? (exit 0 yes, 1 no)\n"
            "  contradish license notes POLICY --model provider:model --as-of D [--cases 24]\n"
            "        have a model write its own case notes; how often are they sufficient now but not later?\n\n"
            "A note gives each fact exactly, as a range {\"lo\": a, \"hi\": b}, as {\"values\": [...]}, or leaves it out."
        ),
    )
    lp.add_argument("target", nargs="+")
    lp.add_argument("--as-of", dest="as_of", default=None)
    lp.add_argument("--case", default=None)
    lp.add_argument("--note", default=None)
    lp.add_argument("--model", default=None)
    lp.add_argument("--cases", type=int, default=24)
    lp.add_argument("--seed", type=int, default=0)
    lp.add_argument("--out", default=None)
    lp.add_argument("--json", action="store_true", default=False)

    mp = sub.add_parser(
        "memory",
        help="Compile an agent's history into the smallest memory that decides every future action its policy "
             "permits exactly as the full history would; certify it; revalidate it as authorization evolves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  contradish memory demo                                        compile, check, forge, revalidate\n"
            "  contradish memory compile POLICY --history H.json --agent A --as-of D [--horizon N] --out CERT.json\n"
            "  contradish memory check CERT.json [--history H.json] [--previous OLD.json]   (independent checker)\n"
            "  contradish memory revalidate CERT.json POLICY [--events E.json] [--updates U.json] [--as-of D]\n"
            "                    [--history H.json] --out NEW.json     exit 0 valid, 1 invalidated\n"
            "  contradish memory show CERT.json                              the memory as the agent carries it\n"
            "  contradish memory obligations CERT.json POLICY [--as-of D]    can it still pay what it owes? exit 1 if not\n\n"
            "Add --ledger L --key K to record certificates in the signed audit ledger. See docs/MEMORY.md."
        ),
    )
    mp.add_argument("target", nargs="+")
    mp.add_argument("--history", default=None)
    mp.add_argument("--agent", default="agent")
    mp.add_argument("--as-of", dest="as_of", default=None)
    mp.add_argument("--horizon", type=int, default=None)
    mp.add_argument("--events", default=None)
    mp.add_argument("--updates", default=None)
    mp.add_argument("--previous", default=None)
    mp.add_argument("--ledger", default=None)
    mp.add_argument("--key", default=None)
    mp.add_argument("--out", default=None)
    mp.add_argument("--json", action="store_true", default=False)

    tp = sub.add_parser(
        "transform",
        help="Is a change to what an agent remembers authorization-preserving? (summaries, deletions, merges, "
             "consolidations) Verdict, counterexample, the safe part of the change, an independently checkable receipt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  contradish transform demo\n"
            "  contradish transform check POLICY --before B.json --after A.json --as-of D [--out V.json]   exit 0 SAFE\n"
            "  contradish transform repair POLICY --before B.json --after A.json --as-of D [--out R.json]\n"
            "  contradish transform compress POLICY --before B.json --as-of D [--out C.json]\n"
            "  contradish transform serve POLICY [--port 8788] [--ledger L --key K]       JSON over HTTP\n"
            "  python contradish/memory_check.py V.json        re-derive a verdict independently\n\n"
            "Memories are contradish.memory/1 (the `memory` of a memory certificate). See docs/TRANSFORM.md."
        ),
    )
    tp.add_argument("target", nargs="+")
    tp.add_argument("--before", default=None)
    tp.add_argument("--after", default=None)
    tp.add_argument("--as-of", dest="as_of", default=None)
    tp.add_argument("--horizon", type=int, default=None)
    tp.add_argument("--reason", default="")
    tp.add_argument("--port", type=int, default=8788)
    tp.add_argument("--host", default="127.0.0.1")
    tp.add_argument("--ledger", default=None)
    tp.add_argument("--key", default=None)
    tp.add_argument("--out", default=None)
    tp.add_argument("--json", action="store_true", default=False)

    gp = sub.add_parser(
        "gateway",
        help="Put the guard in front of any OpenAI- or Anthropic-format model endpoint (change only your base URL).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "  contradish gateway POLICY --upstream https://api.openai.com [--mode shadow|enforce]\n"
            "                    [--port 8787] [--log decisions.jsonl] [--ledger audit.jsonl --key K]\n\n"
            "Clients: OpenAI(base_url='http://localhost:8787/v1'), Anthropic(base_url='http://localhost:8787').\n"
            "Your backend sends the request's evidence in X-Contradish-Context or a 'contradish' body field."
        ),
    )
    gp.add_argument("policy")
    gp.add_argument("--upstream", required=True)
    gp.add_argument("--mode", default="shadow", choices=["shadow", "enforce"])
    gp.add_argument("--host", default="127.0.0.1")
    gp.add_argument("--port", type=int, default=8787)
    gp.add_argument("--log", default=None)
    gp.add_argument("--ledger", default=None)
    gp.add_argument("--key", default=None)
    gp.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=100)


def dispatch(args) -> bool:
    if getattr(args, "command", None) not in COMMANDS:
        return False
    {"integrity": cmd_integrity, "gateway": cmd_gateway, "license": cmd_license,
     "memory": cmd_memory, "transform": cmd_transform}[args.command](args)
    return True


def _gate(text):
    if not text:
        return None
    return {k.strip(): float(v) for k, v in (x.split("=") for x in text.split(","))}


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def cmd_integrity(args) -> None:
    from contradish.integrity import IntegrityLedger, compile_change, verify_change, verify_ledger
    c, t = args.integrity_cmd, args.target
    if c == "demo":
        return _demo()
    if c == "ledger":
        if len(t) != 2 or t[0] not in ("verify", "checkpoint"):
            sys.exit("contradish integrity ledger verify|checkpoint LEDGER")
        if t[0] == "checkpoint":
            if not args.key:
                sys.exit("checkpoint needs --key KEY.json")
            e = IntegrityLedger(t[1], key=args.key).checkpoint()
            print(f"signed head {e['record']['head']} at seq {e['record']['covers_seq']}")
            return
        r = verify_ledger(t[1], trust=args.trust)
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            print(f"{r['verdict']}: {r['entries']} entries {r['kinds']}; signed through seq {r['signed_through_seq']}, "
                  f"{r['unsigned_tail']} after the last signature; decisions re-derived {r['decisions_rederived']}")
            for p in r["problems"][:20]:
                print("  " + p)
        sys.exit(0 if r["verdict"] == "VERIFIED" else 1)
    if not t:
        sys.exit(f"contradish integrity {c} POLICY ...")
    ups = args.updates.split(",") if args.updates else None
    plan = compile_change(t[0], as_of=args.as_of, since=args.since, updates=ups,
                          max_situations=args.max_situations, seed=args.seed)
    led = IntegrityLedger(args.ledger, key=args.key) if args.ledger else None
    if c == "plan":
        if led:
            led.record_plan(plan)
        print(json.dumps(plan.to_dict(), indent=2, default=str) if args.json else plan.render())
        if args.out:
            _dump(plan.to_dict(), args.out)
        return
    if not args.models:
        sys.exit("release needs --models (e.g. openai:gpt-4.1,anthropic:claude-sonnet-4-5 or witness:faithful)")
    print(plan.render(), "\n", file=sys.stderr)
    models = {m: m for m in args.models.split(",") if m}

    def progress(m, u, i, n):
        if i == n or i % 10 == 0:
            print(f"  {m}  {u}  {i}/{n}", file=sys.stderr)
    rel = verify_change(plan, models, gate=_gate(args.gate), on_progress=progress)
    if led:
        led.record_plan(plan)
        led.record_release(plan, rel, by=args.by)
        if led.key:
            led.checkpoint()
    print(json.dumps(rel.to_dict(), indent=2) if args.json else rel.render())
    if args.out:
        _dump({"plan": plan.to_dict(), "release": rel.to_dict()}, args.out)
    sys.exit(0 if len(rel.passed()) == len(models) else 1)


def _memory_checker():
    import importlib.util
    spec = importlib.util.spec_from_file_location("contradish_memory_check_standalone",
                                                  os.path.join(os.path.dirname(__file__), "memory_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_memory(args) -> None:
    from contradish.memory_compiler import compile_memory, render_memory, revalidate
    t = args.target
    led = None
    if args.ledger:
        from contradish.integrity import IntegrityLedger
        led = IntegrityLedger(args.ledger, key=args.key)
    if t[0] == "demo":
        return _memory_demo()
    if t[0] == "compile":
        if len(t) < 2 or not args.history or not args.as_of:
            sys.exit("contradish memory compile POLICY --history H.json --agent A --as-of D [--horizon N] --out CERT.json")
        c = compile_memory(t[1], _read(args.history), args.agent, args.as_of, args.horizon)
    elif t[0] == "revalidate":
        if len(t) < 3:
            sys.exit("contradish memory revalidate CERT.json POLICY [--events E] [--updates U] [--as-of D] [--history H]")
        c = revalidate(_read(t[1]), t[2], _read(args.updates) if args.updates else None,
                       _read(args.events) if args.events else None, args.as_of,
                       _read(args.history) if args.history else None)
    elif t[0] == "check":
        r = _memory_checker().check(_read(t[1]), _read(args.history) if args.history else None,
                                    _read(args.previous) if args.previous else None)
        print(json.dumps(r, indent=2) if args.json else
              f"{r['verdict']}  {r['agent']} as of {r['as_of']}\n" + "\n".join("  - " + p for p in r["problems"]) +
              "".join("\n  . " + n for n in r["notes"]))
        sys.exit(0 if r["verdict"] == "VERIFIED" else 1)
    elif t[0] == "show":
        print(render_memory(_read(t[1])))
        return
    elif t[0] == "obligations":
        if len(t) < 3:
            sys.exit("contradish memory obligations CERT.json POLICY [--as-of D]")
        from contradish.fulfillment import assess
        cert = _read(t[1])
        r = assess(t[2], cert.get("memory", cert), args.as_of or cert.get("as_of"))
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            _print_obligations(r)
        sys.exit(1 if any(o["status"] in ("stranded_by_schedule", "params_missing", "overdue") for o in r["obligations"])
                 else 0)
    else:
        sys.exit("contradish memory demo | compile | check | revalidate | show")
    print(render_memory(c))
    print(f"  {c['size']}" + (f"\n  status: {c.get('status')}" if c.get("status") else ""))
    for i in c.get("issues", []):
        print("  ISSUE: " + i["problem"] + (f" (load-bearing under {i['load_bearing_under']} on {i['day']})"
                                            if i.get("load_bearing_under") else "") + f"; {i['recover']}")
    if args.out:
        _dump(c, args.out)
    if led:
        led.record_memory(c)
        if led.key:
            led.checkpoint()
    sys.exit(1 if c.get("status") == "invalidated" else 0)


def _print_obligations(r):
    for o in r["obligations"]:
        head = {"fulfillable": "can be fulfilled", "act_by": f"ACT BY {o.get('act_by')}",
                "stranded_by_schedule": "STRANDED", "params_missing": "CANNOT BE PAID FROM MEMORY",
                "overdue": "OVERDUE"}[o["status"]]
        print(f"  {o['id']} {o['type']} due {o['due']}: {head}" + (f" (via {', '.join(o['routes'])})" if o["routes"] else ""))
        if o["stranding_futures"]:
            print(f"     fragile: no route remains if {o['stranding_futures'][0]['future']}"
                  + (f" (and {len(o['stranding_futures']) - 1} other legitimate revocations)" if len(o["stranding_futures"]) > 1 else ""))
        for x in o["remedy"]:
            print(f"     -> {x}")


def _memory_demo() -> None:
    import copy
    from contradish.memory_compiler import compile_memory, render_memory, revalidate
    here = os.path.join(os.path.dirname(__file__), "templates")
    pol, hist = os.path.join(here, "memory_policy.json"), _read(os.path.join(here, "memory_history.json"))
    MC = _memory_checker()
    print(f"An agent's history: {len(hist)} events (records, a patient's claim, delegations, revocations).\n")
    c = compile_memory(pol, hist, "agent", "2026-09-30")
    print("1. COMPILE: the smallest memory that decides every permitted future the same way\n")
    print(render_memory(c))
    print("  discarded: " + "; ".join(f"#{d['event']} {d['reason']}" for d in c["discarded"]))
    print(f"\n2. CHECK (memory_check.py, a separate implementation): {MC.check(c, hist)['verdict']}")
    x = copy.deepcopy(c)
    p = x["partitions"]["days_since_fill"]
    i = [k for k, q in enumerate(p) if q.get("lo") == x["memory"]["facts"]["days_since_fill"]["lo"]][0]
    p[i - 1:i + 1] = [{"lo": p[i - 1]["lo"], "hi": p[i]["hi"]}]
    x["memory"]["facts"]["days_since_fill"] = dict(p[i - 1], on="2026-09-30")
    x["witnesses"] = [w for w in x["witnesses"] if not (w["fact"] == "days_since_fill" and w["b"] == p[i - 1]["hi"] - 3)]
    x.pop("digest")
    x["digest"] = MC.sha(x)
    r = MC.check(x, hist)
    print(f"   a forged, coarser memory (re-digested): {r['verdict']}: {r['problems'][0]}")
    print("\n3. CAN IT STILL PAY WHAT IT OWES? (contradish memory obligations)\n")
    from contradish.fulfillment import assess
    _print_obligations(assess(pol, c["memory"], "2026-09-30"))
    print("\n4. REVALIDATE as authorization evolves (from the kept memory alone)\n")
    steps = [("the owner revokes G3 early, effective 2026-10-05",
              dict(new_events=[{"kind": "revoke", "id": "R3", "target": "G3", "by": "pharmacy_owner",
                                "effective": "2026-10-05"}])),
             ("marcus delegates to the agent under G4, a grant the memory dropped",
              dict(new_events=[{"kind": "grant", "id": "G7", "by": "marcus", "to": "agent", "scope": "dispense",
                                "under": "G4", "from": "2026-10-02"}])),
             ("the insurer announces a return to 90% from 2026-10-03",
              dict(new_updates=[{"id": "insurer_back_to_90", "source": "insurer", "channel": "system",
                                 "effective": "2026-10-03", "content": "From 2026-10-03 refills need 90% elapsed.",
                                 "edits": [{"clause": "R1", "params": {"early_refill_pct": 90}}]}]))]
    for label, kw in steps:
        n = revalidate(c, pol, **kw)
        v = MC.check(n, None, c)["verdict"]
        print(f"  {label}: {n['status'].upper()} (reused {len(n['reused_partitions'])} of {len(n['partitions'])} "
              f"partitions; checker {v})")
        for iss in n.get("issues", []):
            print(f"     {iss['problem']}" + (f", load-bearing under {iss['load_bearing_under']} on {iss['day']}"
                                               if iss.get("load_bearing_under") else "") + f"; {iss['recover']}")
        if n["status"] == "invalidated":
            h = revalidate(c, pol, history=hist, **kw)
            print(f"     with the history: {h['status'].upper()}; checker {MC.check(h, None, c)['verdict']}; "
                  + "; ".join(l[2:] for l in render_memory(h).splitlines()[1:] if "days_since_fill" in l or "G7" in l))


def _mem(path):
    d = _read(path)
    return d["memory"] if isinstance(d, dict) and d.get("schema") == "contradish.memory_certificate/1" else d


def _print_verdict(v):
    print(f"{v['verdict']}  ({v['checked']['days_x_futures']} day x future combinations; "
          f"{len(v['futures_checked'])} futures)" + (f"  {v['violations_by_kind']}" if v["violations_by_kind"] else ""))
    for c in v["counterexamples"]:
        if c["kind"] in ("obligation_forgotten", "fulfillment_loss"):
            print(f"  {c['kind']}: obligation {c['obligation']} due {c['due']}"
                  + (f", if {c['future']}" if c.get("future") and c["future"] != "no revocation" else "")
                  + (f": {c['detail']}" if c.get("detail") else ""))
        elif "after_does" in c:
            print(f"  {c['kind']}: on {c['day']}, if {c['future']}: before {c['before_does']}, after {c['after_does']}")
        else:
            print(f"  {c['kind']}: on {c['day']}, if {c['future']}: {c['before_outcomes']} possible outcome(s) before,"
                  f" {c['after_outcomes']} after")


def cmd_transform(args) -> None:
    from contradish.transform import check_transform, compress, repair, serve
    t = args.target
    if t[0] == "demo":
        return _transform_demo()
    if len(t) < 2:
        sys.exit("contradish transform check|repair|compress|serve POLICY ...")
    pol = t[1]
    if t[0] == "serve":
        led = None
        if args.ledger:
            from contradish.integrity import IntegrityLedger
            led = IntegrityLedger(args.ledger, key=args.key)
        srv = serve(pol, args.host, args.port, led)
        print(f"contradish memory-transform service on http://{args.host}:{srv.server_address[1]}  "
              "POST /v1/memory/transform | /v1/memory/compress")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return
    if not args.before or not args.as_of:
        sys.exit("--before and --as-of are required")
    before = _mem(args.before)
    if t[0] == "compress":
        r = compress(pol, before, args.as_of, args.horizon)
        print(f"dropped: {', '.join(r['dropped']) or 'nothing'}; whole compression {r['verdict']['verdict']}")
        if args.out:
            _dump(r, args.out)
        return
    if not args.after:
        sys.exit("--after is required")
    after = _mem(args.after)
    if t[0] == "check":
        v = check_transform(pol, before, after, args.as_of, args.horizon, args.reason)
        print(json.dumps(v, indent=2) if args.json else "")
        if not args.json:
            _print_verdict(v)
        if args.out:
            _dump(v, args.out)
        sys.exit(0 if v["verdict"] == "SAFE" else 1)
    if t[0] == "repair":
        r = repair(pol, before, after, args.as_of, args.horizon)
        print(f"applied: {', '.join(r['applied']) or 'nothing'}")
        for x in r["refused"]:
            print(f"refused: {x['component']} ({', '.join(x['kinds'])})")
        print(f"repaired change: {r['verdict']['verdict']}")
        if args.out:
            _dump(r, args.out)
        return
    sys.exit("contradish transform demo | check | repair | compress | serve")


def _transform_demo() -> None:
    import copy
    from contradish.memory_compiler import compile_memory, render_memory
    from contradish.transform import MemoryGuard, check_transform, compress
    here = os.path.join(os.path.dirname(__file__), "templates")
    pol = os.path.join(here, "memory_policy.json")
    hist = _read(os.path.join(here, "memory_history.json"))
    MC = _memory_checker()
    day = "2026-09-30"
    M = compile_memory(pol, hist, "agent", day)["memory"]
    print("The agent's certified memory (contradish memory compile):\n")
    print(render_memory({"memory": M, "agent": "agent", "digest": "-" * 19, "as_of": day, "horizon_days": 45}))
    print("\nChanges a framework might make to it, each checked against every permitted future:\n")
    cases = []
    a = copy.deepcopy(M)
    a["authority"]["agent_may_dispense"]["chains"] = [{"links": [], "from": day, "until": "2026-10-20"}]
    cases.append(("summarize authority as 'may dispense until 2026-10-20'", a))
    a = copy.deepcopy(M)
    a["facts"]["days_since_fill"] = {"lo": 18, "hi": 23, "on": day}
    cases.append(("round the fill day to 'about three weeks ago' (18-23)", a))
    a = copy.deepcopy(M)
    a["authority"]["agent_may_dispense"]["chains"] = a["authority"]["agent_may_dispense"]["chains"][:1]
    cases.append(("forget the direct grant G3 (looks redundant)", a))
    a = copy.deepcopy(M)
    a["facts"]["prescriber_reachable"] = {"drop": True}
    cases.append(("forget whether the prescriber is reachable", a))
    a = copy.deepcopy(M)
    for o in a["obligations"]:
        if o["type"] == "refund":
            o["params"].pop("card_token", None)
    cases.append(("summarize the refund owed as 'owes the patient $40' (card dropped)", a))
    for label, after in cases:
        v = check_transform(pol, M, after, day, reason=label)
        print(f"- {label}")
        print("  ", end="")
        _print_verdict(v)
        print(f"   independent re-derivation: {MC.check_verdict(v)['verdict']}")
    c = compress(pol, M, day)
    print(f"\ncompress (joint, beyond the compiler's per-fact memory): drops {', '.join(c['dropped'])}; "
          f"{c['verdict']['verdict']}")
    g = MemoryGuard(pol, M, as_of=lambda: day)
    allin = copy.deepcopy(M)
    for _, after in cases:
        for part in ("facts", "authority"):
            for k, val in after[part].items():
                if val != M[part][k]:
                    allin[part][k] = val
        if after.get("obligations") != M.get("obligations"):
            allin["obligations"] = copy.deepcopy(after["obligations"])
    r = g.propose(allin, reason="nightly consolidation: all four at once")
    print(f"\nMemoryGuard (enforce) given all {len(cases)} at once: {r['action']}")


def _read(path):
    with open(path) as f:
        return json.load(f)


def cmd_license(args) -> None:
    from contradish.sufficiency import check_note, compression_license, evaluate_notes, future_equivalent
    t = args.target
    if t[0] == "demo":
        return _license_demo()
    if t[0] == "check":
        if len(t) < 2 or not (args.case and args.note):
            sys.exit("contradish license check POLICY --case CASE.json --note NOTE.json [--as-of D]")
        v = check_note(t[1], _read(args.case), _read(args.note), args.as_of)
        if args.json:
            print(json.dumps(v.to_dict(), indent=2))
        else:
            for x in v.per_version:
                print(f"  {x['label']:<28} {x['note']:<34} " + ("sufficient" if x["correct"] else
                      "NOT sufficient (" + (f"{x['outcomes_consistent_with_note']} different decisions fit the note"
                                            if not x["determined"] else "settles it, wrongly") + ")"))
            if v.counterexample:
                c = v.counterexample
                print(f"  counterexample under {c['version']} {c['when']}: the note cannot tell this case from one where "
                      + ", ".join(f"{f} is {b} instead of {a}" for f, (a, b) in c["facts_that_differ"].items())
                      + f"; they differ on {', '.join(c['steps_that_differ'])}")
            print("  BLIND SPOT: sufficient today, not for the future" if v.blind_spot else "")
        sys.exit(0 if v.sufficient_future else 1)
    if t[0] == "notes":
        if len(t) < 2 or not args.model:
            sys.exit("contradish license notes POLICY --model provider:model [--as-of D] [--cases N]")
        from contradish.drafting import chat_for_model
        r = evaluate_notes(t[1], chat_for_model(args.model, max_tokens=900), args.as_of, args.cases, args.seed, args.model)
        if args.out:
            _dump(r, args.out)
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2))
        return
    lic = compression_license(t[0], args.as_of)
    print(json.dumps(lic.to_dict(), indent=2) if args.json else lic.render())
    if args.out:
        _dump(lic.to_dict(), args.out)


def _license_demo() -> None:
    from contradish.sufficiency import check_note, compression_license, future_equivalent
    pol = os.path.join(os.path.dirname(__file__), "templates", "starter_policy.json")
    day = "2026-09-30"
    print(compression_license(pol, day).render())
    case = {"days_since_fill": 22, "refills_remaining": 2, "schedule": "none", "identity_verified": True,
            "prescriber_reachable": True}
    a = dict(case, days_since_fill={"lo": 0, "hi": 23}, refills_remaining={"lo": 1, "hi": 11})
    print(f"\nOne case on {day}: day 22 since the last fill, 2 refills left. Two notes about it:")
    print("  A  keeps what today's decision needs: 'not due yet (before day 24)', 'has refills'")
    print("  B  keeps the record: day 22, 2 refills")
    fe = future_equivalent(pol, case, a, case, day)
    print(f"\n  equivalent now: {fe['equivalent_now']}    future-equivalent: {fe['future_equivalent']}    "
          f"first divergence: {fe['first_divergence']}")
    c = check_note(pol, case, a, day).counterexample
    print(f"  why: under {c['version']} {c['when']}, note A cannot tell day 22 from day "
          f"{c['facts_that_differ']['days_since_fill'][1]}, and the policy then dispenses on one and not the other.")


def cmd_gateway(args) -> None:
    from contradish.gateway import Gateway, serve
    from contradish.guard import Guard
    from contradish.integrity import IntegrityLedger
    led = IntegrityLedger(args.ledger, key=args.key, checkpoint_every=args.checkpoint_every if args.key else 0) \
        if args.ledger else None
    g = Guard(args.policy, mode=args.mode, log=args.log, ledger=led)
    srv = serve(Gateway(g, args.upstream), args.host, args.port)
    print(f"contradish gateway ({args.mode}) on http://{args.host}:{srv.server_address[1]} -> {args.upstream}; "
          f"policy {g.base_spec.get('id')} ({g.approval.get('state')})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if led and led.key:
            led.checkpoint()


def _demo() -> None:
    """All four layers offline on the starter pharmacy policy, with scripted witness agents."""
    warnings.simplefilter("ignore")
    from contradish.gateway import Gateway
    from contradish.guard import Guard
    from contradish.integrity import IntegrityLedger, compile_change, verify_change, verify_ledger
    from contradish.versions import fixture_key
    here = os.path.dirname(__file__)
    pol = os.path.join(here, "templates", "starter_policy.json")
    d = tempfile.mkdtemp(prefix="contradish-integrity-")
    L = os.path.join(d, "audit.jsonl")
    key = fixture_key("auditor")
    led = IntegrityLedger(L, key=key)
    print("1. DECISION CHANGE COMPILER  (release of 2026-10-10; previous release 2026-09-30)\n")
    plan = compile_change(pol, as_of="2026-10-10", since="2026-09-30", max_situations=40)
    print(plan.render())
    led.record_plan(plan)
    print("\n2. DIFFERENTIAL VERIFICATION  (scripted witnesses, not models: each shows one failure)\n")
    rel = verify_change(plan, {f"witness:{m}": f"witness:{m}" for m in ("faithful", "rigid", "credulous")})
    print(rel.render())
    led.record_release(plan, rel, by="demo")
    print("\n3. DECISION INTEGRITY GATEWAY  (enforce; a model proposes dispensing, three requests)\n")
    g = Guard(pol, mode="shadow", ledger=led)
    g.mode = "enforce"     # demo only: the starter policy is unreviewed, so Guard refuses enforce; approve yours first
    rec = {"days_since_fill": {"value": 27, "source": "pharmacy_record"},
           "refills_remaining": {"value": 2, "source": "pharmacy_record"},
           "schedule": {"value": "c3_c5", "source": "pharmacy_record"},
           "identity_verified": {"value": True, "source": "identity_check"},
           "prescriber_reachable": {"value": True, "source": "prescriber_office"}}
    calls = [{"id": "c0", "type": "function", "function": {"name": "dispense_refill", "arguments": '{"days": 30}'}},
             {"id": "c1", "type": "function", "function": {"name": "notify_patient",
                                                           "arguments": '{"outcome": "refill_ready"}'}}]
    resp = {"choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None, "tool_calls": calls}}]}
    gw = Gateway(g, "https://any-model.example", lambda u, h, b: (200, json.dumps(resp).encode()))
    for label, ev in (("record says day 27", rec),
                      ("record says day 10", dict(rec, days_since_fill={"value": 10, "source": "pharmacy_record"})),
                      ("only the patient says day 27", dict(rec, days_since_fill={"value": 27, "source": "patient"}))):
        body = {"model": "any", "messages": [], "tools": [{}],
                "contradish": {"evidence": ev, "as_of": "2026-10-10", "request_id": label}}
        st, h, out = gw.handle("/v1/chat/completions", {}, json.dumps(body).encode())
        o = json.loads(out)
        left = [x["function"]["name"] for x in (o["choices"][0]["message"].get("tool_calls") or [])]
        print(f"  {label:<30} {h['X-Contradish-Outcome']:<7} runs: {left or 'nothing'}")
    led.checkpoint()
    print("\n4. EVIDENCE AND AUDIT LEDGER\n")
    r = verify_ledger(L, trust=[key["public"]])
    print(f"  {r['verdict']}: {r['entries']} entries {r['kinds']}, signed through seq {r['signed_through_seq']}, "
          f"decisions re-derived independently {r['decisions_rederived']}")
    print(f"  ledger: {L}\n  (signed with the public FIXTURE key; never use it for anything real)")
