"""`contradish study`: run Behavioral Update Fidelity measurements against real models."""
from __future__ import annotations

import argparse
import json
import sys

COMMANDS = ("study",)

DEFAULT_JUDGES = (("openai", "gpt-4.1-mini"), ("anthropic", "claude-haiku-4-5"))


def register(sub) -> None:
    st = sub.add_parser(
        "study",
        help="Run the authority suites and the policy contract against real models; write a report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Put one or more models through the measurements that need a live system, keep every\n"
            "request and reply, and write REPORT.md and results.json.\n\n"
            "  contradish study demo                      offline, scripted agents, no key\n"
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
    st.add_argument("study_action", choices=["run", "demo", "report", "plan"])
    st.add_argument("out_dir", nargs="?", default=None, help="report: the study directory.")
    st.add_argument("--models", default=None, help="Comma-separated PROVIDER:MODEL list.")
    st.add_argument("--judge", default=None,
                    help="PROVIDER:MODEL that labels each answer. Default: openai:gpt-4.1-mini, else "
                         "anthropic:claude-haiku-4-5, whichever key is set. Prefer a judge from a "
                         "different provider than the models under test.")
    st.add_argument("--judge2", default=None, help="Optional second judge; reports judge agreement.")
    st.add_argument("--components", default="authority,contract", help="authority, contract, or both.")
    st.add_argument("--deliveries", default="fresh,in_conversation")
    st.add_argument("--samples", type=int, default=3, help="Repeats per input; the modal label is used.")
    st.add_argument("--workers", type=int, default=4)
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
    lines = []
    if "authority" in cfg.components:
        suite = load_authority_suites()
        n = sum(len(t.cases) for t in suite) * 2 * cfg.samples * len(cfg.deliveries)
        per_model += n
        lines.append(f"    authority suites  {n:5d} model calls  ({len(suite)} transitions x cases x 2 turns "
                     f"x {cfg.samples} samples x {len(cfg.deliveries)} deliveries)")
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
    total = per_model * len(cfg.models)
    return ("\n".join(lines) + f"\n    per model         {per_model:5d} model calls, and as many judge calls"
            + f"\n    all models        {total:5d} model calls + {total} judge calls"
            + (f" (+{total} for the second judge)" if cfg.judge2 else ""))


def cmd_study(args) -> None:
    from contradish import study as S

    if args.study_action == "demo":
        out = args.out or "studies/demo"
        S.run_demo(out, deliveries=tuple(_split(args.deliveries)))
        print(open(f"{out}/REPORT.md", encoding="utf-8").read())
        print(f"  wrote {out}/REPORT.md and {out}/results.json")
        return

    if args.study_action == "report":
        if not args.out_dir:
            sys.exit("  usage: contradish study report STUDY_DIR [--human-labels FILE]")
        S.rebuild_report(args.out_dir, args.human_labels)
        print(open(f"{args.out_dir}/REPORT.md", encoding="utf-8").read())
        return

    try:
        models = [S.parse_model(m) for m in _split(args.models)]
        judge = S.parse_model(args.judge) if args.judge else _default_judge()
        judge2 = S.parse_model(args.judge2) if args.judge2 else None
    except ValueError as e:
        sys.exit(f"  {e}")
    if not models:
        sys.exit("  --models is required, e.g. --models anthropic:claude-haiku-4-5,openai:gpt-4.1-mini")
    if judge is None:
        sys.exit("  no judge: pass --judge PROVIDER:MODEL, or set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    components = tuple(_split(args.components))
    bad = [c for c in components if c not in ("authority", "contract")]
    if bad:
        sys.exit(f"  unknown component(s): {bad}")
    deliveries = tuple(_split(args.deliveries))
    if any(d not in ("fresh", "in_conversation") for d in deliveries):
        sys.exit("  --deliveries takes fresh and/or in_conversation")
    cfg = S.StudyConfig(models=models, judge=judge, judge2=judge2, components=components,
                        deliveries=deliveries, samples=args.samples, workers=args.workers,
                        max_tokens=args.max_tokens, temperature=args.temperature,
                        offline=args.offline, human_sample=args.human_sample)

    print()
    print(f"  STUDY  *  {len(models)} model(s)  *  judge {judge.id}" + (f", second judge {judge2.id}" if judge2 else ""))
    print(_plan(cfg))
    same = [m.id for m in models if m.provider == judge.provider]
    if same:
        print(f"  note: judge shares a provider with {', '.join(same)}; consider --judge from another provider")
    if args.study_action == "plan":
        return
    if not args.offline:
        missing = S.missing_keys(models + [judge] + ([judge2] if judge2 else []))
        if missing:
            sys.exit(f"  missing API key(s): {', '.join(missing)}")
    out = args.out or "studies/run"
    print()
    study = S.run_study(cfg, out)
    print()
    print(open(f"{out}/REPORT.md", encoding="utf-8").read())
    failed = sum(r["calls"]["failed"] for r in study["results"])
    if failed:
        print(f"  {failed} model call(s) failed and were scored as unclassifiable; rerun the same command "
              f"to retry only those.")
