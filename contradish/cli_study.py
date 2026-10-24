"""`contradish study`: run Behavioral Update Fidelity measurements against real models."""
from __future__ import annotations

import argparse
import json
import os
import sys

COMMANDS = ("study", "governed", "repair")

DEFAULT_JUDGES = (("openai", "gpt-4.1-mini"), ("anthropic", "claude-haiku-4-5"))


def register(sub) -> None:
    rp = sub.add_parser(
        "repair",
        help="Find the smallest missing distinction and verify a repair that restores it without blocking legitimate actions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Find which distinction a system fails to keep (who has standing, when a change applies, what a memory\n"
            "must keep, whether rules conflict), show a minimal counterexample, and try repairs from least to most\n"
            "intrusive (an instruction, a memory field, a code guard). A repair passes only if, on held-out cases,\n"
            "the distinction is restored, no legitimate change is newly refused, and nothing else moved.\n\n"
            "  contradish repair --demo                          scripted agents of known behavior, no key\n"
            "  contradish repair --model ollama:qwen2.5:7b       a real model (judge: the model labels itself)\n"
            "  contradish repair --model ... --components standing,claims,tone --json report.json"),
    )
    rp.add_argument("--demo", action="store_true")
    rp.add_argument("--model", default=None, help="PROVIDER:MODEL, as in `contradish study`.")
    rp.add_argument("--components", default="standing,claims,tone,conflicts,future,contrast,temporal")
    rp.add_argument("--json", default=None, help="Write the full report, with its certificate, here.")
    gv = sub.add_parser(
        "governed",
        help="The one question: does the system change exactly when someone with authority changed it, and only then?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Answer contradish's one question for every model in a study:\n\n"
            "  When something changes, does the system change exactly when someone with the authority\n"
            "  to change it did, and only then?\n\n"
            "Governed Score = the lowest score of any part. The profile groups the parts into WHO (only someone\n"
            "with standing), WHEN (only from when it takes effect), WHAT (exactly what it covers, nothing else)\n"
            "and ACT (what it does follows what it says governs).\n\n"
            "  contradish governed STUDY_DIR        a study written by `contradish study run|local`\n"
            "  contradish governed --demo           scripted agents of known behavior\n"
            "  contradish governed --audit          the instrument check: no shortcut strategy can score above 0"),
    )
    gv.add_argument("study_dir", nargs="?", default=None)
    gv.add_argument("--demo", action="store_true")
    gv.add_argument("--audit", action="store_true")
    gv.add_argument("--json", action="store_true")
    gv.add_argument("--delivery", default="fresh")
    st = sub.add_parser(
        "study",
        help="Run the authority suites and the policy contract against real models; write a report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Put one or more models through the measurements that need a live system, keep every\n"
            "request and reply, and write REPORT.md and results.json.\n\n"
            "  contradish study local                     free: your local Ollama models, no key, no credits\n"
            "  contradish study demo                      offline, scripted agents, no key\n"
            "  contradish study contrasts                 list the matched-provenance authority contrasts\n"
            "  contradish study temporal                  list the temporal-scope contrasts\n"
            "  contradish study relay                     list the relayed-change (verify) contrasts\n"
            "  contradish study relational                list the relational (care survives the relationship) contrasts\n"
            "  contradish study protected                 list the protected obligations (what an operator cannot change)\n"
            "  contradish study experience                list the experience contrasts (only the person governs what they feel)\n"
            "  contradish study capability                list the capability contrasts (power gained only from authority)\n"
            "  contradish study observer                  list the observer contrasts (behavior when no one seems to watch)\n"
            "  contradish study recoverability            list the recoverability contrasts (correct today, correctable tomorrow)\n"
            "  contradish study obligations               list the obligation contrasts (duties a decision creates)\n"
            "  contradish study agency                    list the agency contrasts (who can know, decide, act, challenge, correct)\n"
            "  contradish study distinctions              list the distinction contrasts (felt, suspected, established, warranted)\n"
            "  contradish study coupling                  list the governance coupling contrasts (decide by what you say governs)\n"
            "  contradish study self_governance           list the self-governance contrasts (its own earlier word has no standing)\n"
            "  contradish study standing                  list the standing contrasts (authority that moves to someone else)\n"
            "  contradish study claims                    list the claims contrasts (pay, deny or escalate on the right date and authority)\n"
            "  contradish study conflicts                 list the conflict contrasts (flag contradictory rules; only the owner resolves)\n"
            "  contradish study tone                      list the tone contrasts (authority counts, tone does not)\n"
            "  contradish study future                    list the future contrasts (memory keeps what later decisions need)\n"
            "  contradish study provider                  show the encoded provider precedence rules\n"
            "  contradish study run --models anthropic:claude-haiku-4-5,openai:gpt-4.1-mini \\\n"
            "        --judge openai:gpt-4.1-mini --out studies/run1\n"
            "  contradish study run --models ollama:llama3.1:8b --judge ollama:qwen2.5:14b --out studies/local\n"
            "  contradish study run ... --offline          re-score from the logged replies, no API calls\n"
            "  contradish study report studies/run1 --human-labels labeled.csv\n\n"
            "Providers: anthropic, openai, groq, openrouter, together, mistral, deepseek, xai, gemini,\n"
            "ollama. Keys come from the usual environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY,\n"
            "GROQ_API_KEY, ...). See contradish/study.py."
        ),
    )
    st.add_argument("study_action", choices=["local", "run", "demo", "report", "plan", "contrasts", "temporal", "relay", "relational",
                                                    "protected", "experience", "capability", "observer",
                                                    "recoverability", "obligations", "agency", "distinctions", "coupling", "self_governance", "standing", "claims", "conflicts", "tone", "future", "provider"])
    st.add_argument("out_dir", nargs="?", default=None, help="report: the study directory.")
    st.add_argument("--models", default=None, help="Comma-separated PROVIDER:MODEL list.")
    st.add_argument("--judge", default=None,
                    help="PROVIDER:MODEL that labels each answer. Default: openai:gpt-4.1-mini, else "
                         "anthropic:claude-haiku-4-5, whichever key is set. Prefer a judge from a "
                         "different provider than the models under test.")
    st.add_argument("--judge2", default=None, help="Optional second judge; reports judge agreement.")
    st.add_argument("--components", default=None,
                    help="Any of authority, contrast, temporal, relay, relational, protected, experience, capability, "
                         "observer, recoverability, obligations, agency, distinctions, coupling, self_governance, standing, claims, conflicts, tone, future, provider, contract. Default: all twenty-two "
                         "(local: contrast,temporal,relay,capability,recoverability,obligations,agency).")
    st.add_argument("--provider-spec", dest="provider_spec", default=None,
                    help="Score the provider component against this provider's published rules instead of each "
                         "model's own (reported as reference only). Encoded: anthropic.")
    st.add_argument("--deliveries", default=None, help="Default: fresh,in_conversation (local: fresh).")
    st.add_argument("--samples", type=int, default=None,
                    help="Repeats per input; the modal label is used. Default 3 (local: 1).")
    st.add_argument("--workers", type=int, default=None, help="Parallel requests. Default 4 (local: 1).")
    st.add_argument("--self-label", dest="self_label", action="store_true", default=False,
                    help="No judge: each model ends its reply with 'OUTCOME: <label>'. Default for local.")
    st.add_argument("--max-tokens", dest="max_tokens", type=int, default=600)
    st.add_argument("--temperature", type=float, default=None,
                    help="Omit to use each provider's default (recorded in the results).")
    st.add_argument("--human-sample", dest="human_sample", type=int, default=60)
    st.add_argument("--out", default=None, help="run/demo: output directory.")
    st.add_argument("--offline", action="store_true", default=False)
    st.add_argument("--human-labels", dest="human_labels", default=None)


def dispatch(args) -> bool:
    if args.command not in COMMANDS:
        return False
    if args.command == "repair":
        from contradish import distinction_repair as DR
        if args.demo or not args.model:
            print(DR.demo())
            return True
        import tempfile
        from pathlib import Path
        from contradish import study as S
        spec = S.parse_model(args.model)
        log, tally = S.CallLog(Path(tempfile.mkdtemp(prefix="contradish-repair-")) / "calls.jsonl"), S.Tally()

        def chat_for(t, delivery=None):
            fn = S.http_chat_fn(spec, log, tally, 600, 0.0, context=f"repair/{t.id}/{delivery}")
            return S.self_labelled(fn, t.outcomes)
        report = DR.find_and_repair(chat_for, S.self_label_classifier, components=tuple(_split(args.components)))
        print(f"  ({tally.calls} calls, {tally.errors} failed)")
        print(DR.render(report))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(report, f, indent=1, default=str)
        return True
    if args.command == "governed":
        from contradish.governed import main as governed_main
        argv = ([args.study_dir] if args.study_dir else []) + (["--demo"] if args.demo else []) + \
            (["--audit"] if args.audit else []) + (["--json"] if args.json else []) + ["--delivery", args.delivery]
        raise SystemExit(governed_main(argv))
    cmd_study(args)
    return True


def _split(s: str) -> list:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _default_judge():
    import os
    from contradish.study import ModelSpec, PROVIDERS
    for provider, model in DEFAULT_JUDGES:
        if os.environ.get(PROVIDERS[provider][2]):
            return ModelSpec(provider, model)
    return None


def _plan(cfg) -> str:
    from contradish.contract import load_builtin_contract
    from contradish.study import load_authority_suites
    per_model = 0
    provider_extra = 0      # provider-spec calls happen only for models whose provider's rules are encoded
    lines = []
    if "authority" in cfg.components:
        from contradish.study import placebo_transitions
        suite = load_authority_suites() + placebo_transitions()
        n = sum(len(t.cases) for t in suite) * 2 * cfg.samples * len(cfg.deliveries)
        per_model += n
        lines.append(f"    authority suites  {n:5d} model calls  ({len(suite)} transitions, 4 of them placebos, x cases "
                     f"x 2 turns x {cfg.samples} samples x {len(cfg.deliveries)} deliveries)")
    if "contrast" in cfg.components:
        from contradish.authority_contrast import build_contrasts
        con = build_contrasts()
        n = sum(len(t.cases) for t in con) * 2 * cfg.samples * len(cfg.deliveries)
        per_model += n
        lines.append(f"    authority contrasts {n:3d} model calls  ({len(con)} deliveries of 8 claims x cases x 2 "
                     f"turns x {cfg.samples} samples x {len(cfg.deliveries)} deliveries)")
    if "temporal" in cfg.components:
        from contradish.temporal_contrast import build_temporal, CLAIMS as TCLAIMS
        tem = build_temporal()
        n = sum(len(t.cases) for t in tem) * 2 * cfg.samples * len(cfg.deliveries)
        per_model += n
        lines.append(f"    temporal contrasts {n:4d} model calls  ({len(tem)} scopes of {len(TCLAIMS)} amendments x cases "
                     f"x 2 turns x {cfg.samples} samples x {len(cfg.deliveries)} deliveries)")
    if "relay" in cfg.components:
        from contradish.relay_contrast import build_relay
        rel = build_relay()
        n = sum(len(t.cases) for t in rel) * 2 * cfg.samples * len(cfg.deliveries)
        per_model += n
        lines.append(f"    relay contrasts   {n:5d} model calls  ({len(rel)} deliveries of the temporal amendments x "
                     f"cases x 2 turns x {cfg.samples} samples x {len(cfg.deliveries)} deliveries)")
    if "relational" in cfg.components:
        from contradish.relational_contrast import build_relational
        rc = build_relational()
        n = sum(len(t.cases) for t in rc) * 2 * cfg.samples * len(cfg.deliveries)
        per_model += n
        lines.append(f"    relational contrasts {n:3d} model calls  ({len(rc)} frames x clause changes x cases "
                     f"x 2 turns x {cfg.samples} samples x {len(cfg.deliveries)} deliveries)")
    if "protected" in cfg.components:
        from contradish.protected_contrast import build_protected
        pt = build_protected()
        n = sum(len(t.cases) for t in pt) * 2 * cfg.samples * len(cfg.deliveries)
        per_model += n
        lines.append(f"    protected obligations {n:2d} model calls  ({len(pt)} operator briefs x 2 cases x 2 turns x "
                     f"{cfg.samples} samples x {len(cfg.deliveries)} deliveries)")
    if "experience" in cfg.components:
        from contradish.experience_contrast import build_experience
        et = build_experience()
        n = sum(len(t.cases) for t in et) * 2 * cfg.samples * len(cfg.deliveries)
        per_model += n
        lines.append(f"    experience contrasts {n:3d} model calls  ({len(et)} deliveries x 2 turns x {cfg.samples} samples "
                     f"x {len(cfg.deliveries)} deliveries)")
    import importlib
    from contradish.study import BEYOND
    for name in BEYOND:
        if name not in cfg.components:
            continue
        mod = importlib.import_module(BEYOND[name][0])
        built = getattr(mod, "build_" + name)()
        n = sum(len(t.cases) for t in built) * 2 * cfg.samples * len(cfg.deliveries)
        per_model += n
        lines.append(f"    {name + ' contrasts':<22}{n:4d} model calls  ({len(built)} transitions x cases x 2 turns x "
                     f"{cfg.samples} samples x {len(cfg.deliveries)} deliveries)")
    if "provider" in cfg.components:
        from contradish.provider_spec import build_provider_contrasts, SPECS
        pc = build_provider_contrasts()
        n = sum(len(t.cases) for t in pc) * 2 * cfg.samples * len(cfg.deliveries)
        applies = [m for m in cfg.models if (cfg.provider_spec or m.provider) in SPECS]
        provider_extra = n * len(applies)
        lines.append(f"    provider spec     {n:5d} model calls  ({len(pc)} transitions; runs for "
                     f"{len(applies)} of {len(cfg.models)} model(s): those whose provider's rules are encoded"
                     + (", or all with --provider-spec" if not cfg.provider_spec else "") + ")")
    if "contract" in cfg.components:
        c = load_builtin_contract("ecommerce_returns")
        n = 0
        for st in c.states():
            for case in c.cases:
                n += len(list(case.inputs(True))) + (len(case.contrasts) if st == "base" else 0)
        n *= cfg.samples
        per_model += n
        lines.append(f"    policy contract   {n:5d} model calls  ({len(c.states())} policy states, "
                     f"{len(c.cases)} cases with rewordings and contrasts, x {cfg.samples} samples)")
    total = per_model * len(cfg.models) + provider_extra
    jc = 0 if cfg.judge is None else total
    return ("\n".join(lines) + f"\n    per model         {per_model:5d} model calls"
            + (" (plus the provider-spec calls, where they apply)" if provider_extra else "")
            + (", and as many judge calls" if cfg.judge is not None else ", no judge calls (self-label)")
            + f"\n    all models        {total:5d} model calls + {jc} judge calls"
            + (f" (+{total} for the second judge)" if cfg.judge2 else ""))


def cmd_study(args) -> None:
    from contradish import study as S

    local = args.study_action == "local"
    defaults = ({"components": "contrast,temporal,relay,capability,recoverability,obligations,agency", "deliveries": "fresh",
                 "samples": 1, "workers": 1} if local else
                {"components": "authority,contrast,temporal,relay,relational,protected,experience,capability,observer,"
                               "recoverability,obligations,agency,distinctions,coupling,self_governance,standing,claims,conflicts,tone,future,provider,contract", "deliveries": "fresh,in_conversation",
                 "samples": 3, "workers": 4})
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    if args.study_action == "demo":
        out = args.out or "studies/demo"
        S.run_demo(out, deliveries=tuple(_split(args.deliveries)))
        print(open(f"{out}/REPORT.md", encoding="utf-8").read())
        print(f"  wrote {out}/REPORT.md and {out}/results.json")
        return

    if args.study_action == "contrasts":
        from contradish.authority_contrast import describe
        print()
        print(describe())
        return

    if args.study_action == "temporal":
        from contradish.temporal_contrast import describe as describe_temporal
        print()
        print(describe_temporal())
        return

    if args.study_action == "relay":
        from contradish.relay_contrast import describe as describe_relay
        print()
        print(describe_relay())
        return

    if args.study_action == "relational":
        from contradish.relational_contrast import describe as describe_relational
        print()
        print(describe_relational())
        return

    if args.study_action == "protected":
        from contradish.protected_contrast import describe as describe_protected
        print()
        print(describe_protected())
        return

    if args.study_action == "experience":
        from contradish.experience_contrast import describe as describe_experience
        print()
        print(describe_experience())
        return

    if args.study_action in ("capability", "observer", "recoverability", "obligations", "agency", "distinctions",
                             "coupling", "self_governance", "standing", "claims", "conflicts", "tone", "future"):
        import importlib
        from contradish.study import BEYOND
        print()
        print(importlib.import_module(BEYOND[args.study_action][0]).describe())
        return

    if args.study_action == "provider":
        from contradish.provider_spec import SPECS, describe as describe_provider
        print()
        for sp in SPECS.values():
            print(describe_provider(sp))
            print()
        return

    if args.study_action == "report":
        if not args.out_dir:
            sys.exit("  usage: contradish study report STUDY_DIR [--human-labels FILE]")
        S.rebuild_report(args.out_dir, args.human_labels)
        print(open(f"{args.out_dir}/REPORT.md", encoding="utf-8").read())
        return

    if local and args.offline:
        # Re-scoring from the logged replies needs no Ollama: take the models from --models or from the study itself.
        args.out = args.out or "studies/local"
        wanted = _split(args.models)
        if not wanted:
            try:
                with open(os.path.join(args.out, "results.json"), encoding="utf-8") as f:
                    wanted = list(json.load(f)["config"]["models"])
            except (OSError, KeyError, ValueError):
                sys.exit(f"  --offline: pass --models, or point --out at a study directory with results.json")
        args.models = ",".join(m if m.startswith("ollama:") else f"ollama:{m}" for m in wanted)
        if not args.judge:
            args.self_label = True
    elif local:
        installed = S.ollama_models()
        if installed is None:
            sys.exit(
                "\n  Ollama is not running, so there are no local models to test.\n\n"
                "  1. Install it from https://ollama.com/download (free) and open the Ollama app,\n"
                "     or run `ollama serve` in another Terminal window.\n"
                "  2. Download a model or two (one-time, about 2 GB each):\n"
                "       ollama pull qwen2.5:3b\n"
                "       ollama pull llama3.2:3b\n"
                "  3. Run `contradish study local` again.\n")
        wanted = _split(args.models) or installed
        wanted = [m if m.startswith("ollama:") else f"ollama:{m}" for m in wanted]
        if not wanted:
            sys.exit("\n  Ollama is running but has no models. Download one (about 2 GB):\n"
                     "    ollama pull qwen2.5:3b\n  then run `contradish study local` again.\n")
        def _have(name):
            return name in installed or (":" not in name and f"{name}:latest" in installed)
        missing_local = [m for m in wanted if not _have(m.split(":", 1)[1])]
        if missing_local:
            sys.exit(f"  not downloaded yet: {', '.join(missing_local)}. Run: "
                     + "  ".join(f"ollama pull {m.split(':', 1)[1]}" for m in missing_local))
        args.models = ",".join(wanted)
        if not args.judge:
            args.self_label = True
        args.out = args.out or "studies/local"

    if args.offline and args.out and os.path.exists(os.path.join(args.out, "results.json")):
        try:
            with open(os.path.join(args.out, "results.json"), encoding="utf-8") as f:
                recorded = json.load(f).get("config", {})
        except (OSError, ValueError):
            recorded = {}
        # The logged replies are keyed by the exact request, settings included: re-score with the run's own
        # settings unless told otherwise, or every lookup misses.
        if args.temperature is None and recorded.get("temperature") is not None:
            args.temperature = recorded["temperature"]
        if recorded.get("max_tokens") is not None and args.max_tokens == 600 and recorded["max_tokens"] != 600:
            args.max_tokens = recorded["max_tokens"]

    try:
        models = [S.parse_model(m) for m in _split(args.models)]
        judge = None if args.self_label else (S.parse_model(args.judge) if args.judge else _default_judge())
        judge2 = S.parse_model(args.judge2) if args.judge2 else None
    except ValueError as e:
        sys.exit(f"  {e}")
    if not models:
        sys.exit("  --models is required, e.g. --models anthropic:claude-haiku-4-5,openai:gpt-4.1-mini")
    if judge is None and not args.self_label:
        sys.exit("  no judge: pass --judge PROVIDER:MODEL, set OPENAI_API_KEY or ANTHROPIC_API_KEY, or use\n"
                 "  --self-label (each model labels its own answers). For a free run: contradish study local")
    components = tuple(_split(args.components))
    bad = [c for c in components if c not in S.COMPONENTS]
    if bad:
        sys.exit(f"  unknown component(s): {bad}")
    deliveries = tuple(_split(args.deliveries))
    if any(d not in ("fresh", "in_conversation") for d in deliveries):
        sys.exit("  --deliveries takes fresh and/or in_conversation")
    cfg = S.StudyConfig(models=models, judge=judge, judge2=judge2, components=components,
                        deliveries=deliveries, samples=args.samples, workers=args.workers,
                        max_tokens=args.max_tokens, temperature=args.temperature,
                        offline=args.offline, human_sample=args.human_sample,
                        provider_spec=args.provider_spec)

    print()
    who = judge.id if judge else "none (each model labels its own answers)"
    print(f"  STUDY  *  {len(models)} model(s)  *  judge {who}" + (f", second judge {judge2.id}" if judge2 else ""))
    print(_plan(cfg))
    if local:
        print("  free: runs on this computer through Ollama; no API key, no credits. Small models take a few")
        print("  seconds per call, so expect roughly 10-30 minutes per model. Rerunning resumes where it stopped.")
    same = [m.id for m in models if judge and m.provider == judge.provider and m.provider != "ollama"]
    if same:
        print(f"  note: judge shares a provider with {', '.join(same)}; consider --judge from another provider")
    if args.study_action == "plan":
        return
    if not args.offline:
        everyone = models + [x for x in (judge, judge2) if x]
        missing = S.missing_keys(everyone)
        if missing:
            sys.exit(f"  missing API key(s): {', '.join(missing)}")
        keys = {m.key_env for m in everyone if m.key_env}
        placeholders = sorted(k for k in keys if os.environ.get(k, "").endswith("..."))
        if placeholders:
            sys.exit(f"  {', '.join(sorted(placeholders))} still hold a placeholder ending in '...'; "
                     f"export the real key(s) and run again")
        print("  checking each model and judge with one short request ...")
        problems = S.preflight(everyone)
        if problems:
            for mid, err in problems:
                print(f"  {mid}: {err}")
            sys.exit("  stopped before the run: fix the key or model name above (nothing was spent but the checks)")
    out = args.out or "studies/run"
    print()
    study = S.run_study(cfg, out)
    print()
    print(open(f"{out}/REPORT.md", encoding="utf-8").read())
    failed = sum(r["calls"]["failed"] for r in study["results"])
    if failed:
        print(f"  {failed} model call(s) failed and were scored as unclassifiable; rerun the same command "
              f"to retry only those.")
