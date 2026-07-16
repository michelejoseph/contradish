"""
contradish CLI

Usage:
    # From a system prompt string directly
    contradish "You are a support agent. Refunds within 30 days only."

    # From a file
    contradish --prompt system_prompt.txt --app mymodule:my_app

    # Manual test cases from YAML
    contradish run evals.yaml --app mymodule:my_app
"""

import sys
import os
import argparse
import importlib
import json
from typing import Callable, Optional


def _load_callable(path: str) -> Callable:
    if ":" not in path:
        raise ValueError(
            f"App must be 'module:function', got: {path!r}\n"
            f"Example: mymodule:my_app_function"
        )
    module_str, func_str = path.rsplit(":", 1)
    sys.path.insert(0, os.getcwd())
    module = importlib.import_module(module_str)
    return getattr(module, func_str)


def _load_cases(path: str):
    from contradish import TestCase
    with open(path) as f:
        raw = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
            data = yaml.safe_load(raw)
        except ImportError:
            sys.exit("Install pyyaml to use YAML files:  pip install pyyaml")
        items = data.get("test_cases", data) if isinstance(data, dict) else data
    else:
        items = json.loads(raw)
        if isinstance(items, dict):
            items = items.get("test_cases", [])
    return [
        TestCase(
            input=tc["input"],
            name=tc.get("name"),
            expected_traits=tc.get("expected_traits", []),
        )
        for tc in items
    ]


def _check_api_key():
    """Give a clear error if no API key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("\n  contradish needs an API key.\n")
        print("  Set one of:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    export OPENAI_API_KEY=sk-...\n")
        sys.exit(1)


def _output_json(report) -> None:
    """Print the report as JSON to stdout."""
    print(json.dumps(report.to_dict(), indent=2))


def _output_sarif(report, output_path: str = "contradish.sarif") -> None:
    """Write a SARIF 2.1 file. GitHub reads this for PR annotations."""
    results = []
    for r in report.failed:
        msg = f"CAI failure: {r.test_case.name} (score {r.cai_score:.2f})"
        if r.contradictions:
            c = r.contradictions[0]
            msg += f": {c.output_a[:80]!r} vs {c.output_b[:80]!r}"
        results.append({
            "ruleId": "CAI001",
            "level": "error",
            "message": {"text": msg},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": "system_prompt"}}}],
        })

    sarif = {
        "version": "2.1.0",
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "contradish",
                    "version": "0.7.0",
                    "rules": [{
                        "id": "CAI001",
                        "name": "CAIConsistencyFailure",
                        "shortDescription": {"text": "Model gives inconsistent answers to semantically equivalent inputs."},
                        "helpUri": "https://contradish.com",
                    }],
                }
            },
            "results": results,
            "properties": {
                "cai_score": report.cai_score,
                "failure_count": report.failure_count,
            },
        }],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sarif, f, indent=2)
    print(f"\n  SARIF written: {output_path}\n")


def cmd_init(args):
    """Interactive setup. Writes .contradish.yaml."""
    import shutil

    config_path = ".contradish.yaml"

    if os.path.exists(config_path) and not getattr(args, "force", False):
        print(f"\n  {config_path} already exists. Use --force to overwrite.\n")
        return

    print("\n  contradish init\n")

    # Detect available policies
    try:
        from contradish.policies import list_policies
        policies = list_policies()
    except Exception:
        policies = ["ecommerce", "hr", "healthcare", "legal"]

    # Policy
    print(f"  Available policy packs: {', '.join(policies)}")
    policy = input("  Policy pack (or leave blank to skip): ").strip()
    if policy and policy not in policies:
        print(f"  Unknown policy '{policy}'. Options: {', '.join(policies)}")
        policy = ""

    # App
    app = input("  App callable (module:function, or leave blank for demo mode): ").strip()

    # Threshold
    threshold_raw = input("  Maximum CAI Strain to pass CI [0.20] (lower is better): ").strip()
    try:
        threshold = float(threshold_raw) if threshold_raw else 0.20
    except ValueError:
        threshold = 0.20

    # Paraphrases
    paraphrases_raw = input("  Paraphrases per test case [5]: ").strip()
    try:
        paraphrases = int(paraphrases_raw) if paraphrases_raw else 5
    except ValueError:
        paraphrases = 5

    lines = ["# contradish configuration\n# https://contradish.com\n"]
    if policy:
        lines.append(f"policy: {policy}")
    if app:
        lines.append(f"app: {app}")
    lines.append(f"threshold: {threshold}")
    lines.append(f"paraphrases: {paraphrases}")

    with open(config_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Wrote {config_path}")

    # Offer to copy the GitHub Actions workflow
    gha_dest = ".github/workflows/cai.yml"
    gha_template = os.path.join(
        os.path.dirname(__file__), "..", ".github", "workflows", "cai.yml"
    )
    if os.path.exists(gha_template) and not os.path.exists(gha_dest):
        copy_gha = input("\n  Copy GitHub Actions workflow to .github/workflows/cai.yml? [Y/n]: ").strip().lower()
        if copy_gha in ("", "y", "yes"):
            os.makedirs(".github/workflows", exist_ok=True)
            shutil.copy(gha_template, gha_dest)
            print(f"  Wrote {gha_dest}")
            print("  Add ANTHROPIC_API_KEY to repo Settings > Secrets > Actions.\n")

    print(f"\n  Run:  contradish --policy {policy or 'ecommerce'}\n")


def _save_report(report, path: str, policy_name: str = None) -> None:
    """Save a shareable HTML report to disk."""
    from contradish.reporter import to_html
    html = to_html(report, policy_name=policy_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  report saved: {path}\n")


def _make_demo_app(system_prompt: str):
    """Return a demo app callable that uses the configured LLM with the given system prompt."""
    from contradish.llm import LLMClient
    llm = LLMClient()

    def demo_app(question: str) -> str:
        if llm.provider == "anthropic":
            msg = llm._client.messages.create(
                model=llm.fast_model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
            )
            return msg.content[0].text.strip()
        else:
            resp = llm._client.chat.completions.create(
                model=llm.fast_model,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )
            return resp.choices[0].message.content.strip()

    return demo_app, llm.provider


def cmd_policy(args):
    """Run a prebuilt domain policy pack. No system prompt required."""
    from contradish import Suite
    from contradish.policies import load_policy, list_policies
    from contradish.printer import print_next_steps

    _check_api_key()

    policy_name = args.policy
    use_json    = getattr(args, "json", False)

    # Validate policy name early for a good error message
    try:
        pack = load_policy(policy_name)
    except ValueError as e:
        print(f"\n  {e}\n")
        print(f"  Available: {', '.join(list_policies())}\n")
        sys.exit(1)

    if not use_json:
        print(f"\n  {pack.display_name} policy pack  "
              f"({len(pack.cases)} test cases)\n")
        print(f"  {pack.description}\n")

    if args.app:
        app = _load_callable(args.app)
    else:
        # Demo mode: create a generic assistant that answers the test questions
        # without a real system prompt. surfaces what the model does by default
        demo_system = (
            f"You are a helpful assistant for a {pack.display_name.lower()} context. "
            f"Answer user questions clearly and accurately."
        )
        if not use_json:
            print("  demo mode: no --app provided, testing default LLM behavior")
            print("  add --app mymodule:fn to test your actual app\n")
        app, _ = _make_demo_app(demo_system)

    suite = Suite.from_policy(
        policy=policy_name,
        app=app,
        verbose=not use_json,
    )
    report = suite.run(
        paraphrases = args.paraphrases,
        verbose     = not use_json,
        concurrency = getattr(args, "concurrency", 4),
    )

    # Apply the EQ threshold from CLI if provided (default is set on Report itself)
    eq_threshold = getattr(args, "eq_threshold", None)
    if eq_threshold is not None:
        report.eq_threshold = float(eq_threshold)

    fmt = getattr(args, "format", None)
    if fmt == "json" or use_json:
        _output_json(report)
    elif fmt == "sarif":
        output = getattr(args, "output", "contradish.sarif")
        _output_sarif(report, output_path=output or "contradish.sarif")
    else:
        print_next_steps(report)

    report_path = getattr(args, "report", None)
    if report_path:
        _save_report(report, report_path, policy_name=policy_name)

    threshold = getattr(args, "threshold", None)
    if threshold is not None:
        # Compare against headline_strain (audited subset) when available; fall back
        # to unweighted cai_strain for benchmarks with no EQ data.
        observed = report.headline_strain if report.headline_strain is not None else report.cai_strain
        if observed is not None and observed > threshold:
            label = "headline CAI Strain" if report.headline_strain is not None else "CAI Strain"
            print(f"\n  FAIL: {label} {observed:.2f} exceeds threshold {threshold} (lower is better)\n")
            sys.exit(1)

    sys.exit(1 if report.failed else 0)


def cmd_quick(args):
    """
    Zero-config stability analysis using the Residual Truth Engine.

    No API key needed to evaluate your own model. An API key is only required
    in demo mode (when --app is omitted), where the default LLM is the model
    under test.
    """
    from contradish import analyze
    from contradish.domains import list_domains, get_domain

    domain  = getattr(args, "domain",  "customer-service")
    app_str = getattr(args, "app",     None)
    extra_q = getattr(args, "questions", None) or []
    html_out = getattr(args, "html",   None)
    sft_out  = getattr(args, "sft",    None)
    dpo_out  = getattr(args, "dpo",    None)
    full     = getattr(args, "full_framings", False)
    n_rep    = getattr(args, "n_repairs", 30)

    # Validate domain early
    if domain not in list_domains() and not extra_q:
        available = ", ".join(list_domains())
        print(f"\n  Unknown domain {domain!r}. Available: {available}\n")
        print(f"  Or pass --questions 'Q1' 'Q2' for custom questions.\n")
        sys.exit(1)

    # Resolve the model function
    if app_str:
        try:
            model_fn = _load_callable(app_str)
        except (ValueError, ImportError, AttributeError) as e:
            print(f"\n  Could not load {app_str!r}: {e}\n")
            sys.exit(1)
        print(f"\n  contradish analyze · testing {app_str}")
        print(f"  No API key required for analysis.\n")
    else:
        # Demo mode: test the default LLM behavior
        _check_api_key()
        pack = get_domain(domain)
        demo_system = (
            f"You are a helpful assistant. "
            f"Answer user questions clearly and accurately."
        )
        model_fn, _ = _make_demo_app(demo_system)
        print(f"\n  contradish analyze · demo mode  (testing default LLM behavior)")
        print(f"  Add --app mymodule:my_fn to test your own model without an API key.\n")

    # Run
    result = analyze(
        model_fn      = model_fn,
        domain        = domain if domain in list_domains() else None,
        questions     = extra_q or None,
        n_repairs     = n_rep,
        full_framings = full,
        verbose       = True,
    )

    # Print terminal report
    print(result)

    # Optional outputs
    if html_out:
        result.to_html(html_out)
        print(f"  HTML report:  {html_out}")

    if sft_out:
        sft_data = result.to_jsonl()
        with open(sft_out, "w", encoding="utf-8") as f:
            f.write(sft_data)
        n = len([l for l in sft_data.splitlines() if l.strip()])
        print(f"  SFT JSONL:    {sft_out}  ({n} records)")

    if dpo_out:
        dpo_data = result.to_dpo_jsonl()
        with open(dpo_out, "w", encoding="utf-8") as f:
            f.write(dpo_data)
        n = len([l for l in dpo_data.splitlines() if l.strip()])
        print(f"  DPO JSONL:    {dpo_out}  ({n} contrastive pairs)")

    # Exit nonzero if strain exceeds threshold
    threshold = getattr(args, "threshold", None)
    if threshold is not None and result.overall_strain > threshold:
        print(
            f"\n  FAIL: strain {result.overall_strain:.2f} exceeds "
            f"threshold {threshold} (lower is better)\n"
        )
        sys.exit(1)


def cmd_from_prompt(args):
    """Run from a system prompt. Zero-config path."""
    from contradish import Suite
    from contradish.printer import print_start, print_next_steps

    _check_api_key()

    # Load prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            system_prompt = f.read()
    else:
        system_prompt = args.system_prompt

    if not system_prompt or not system_prompt.strip():
        print("\n  No system prompt provided.\n")
        print("  Usage: contradish \"Your system prompt here\"")
        print("     or: contradish --prompt system_prompt.txt\n")
        sys.exit(1)

    use_json = getattr(args, "json", False)

    # Load app if provided, otherwise use a passthrough demo app
    if args.app:
        app = _load_callable(args.app)
        if not use_json:
            print_start(system_prompt)
    else:
        if not use_json:
            print_start(system_prompt)
            print("  demo mode: testing your prompt against itself")
            print("  add --app mymodule:fn to test your actual app\n")
        app, _ = _make_demo_app(system_prompt)

    suite = Suite.from_prompt(
        system_prompt=system_prompt,
        app=app,
        verbose=not use_json,
    )
    report = suite.run(
        paraphrases = args.paraphrases,
        verbose     = not use_json,
        concurrency = getattr(args, "concurrency", 4),
    )

    fmt = getattr(args, "format", None)
    if fmt == "json" or use_json:
        _output_json(report)
    elif fmt == "sarif":
        output = getattr(args, "output", "contradish.sarif")
        _output_sarif(report, output_path=output or "contradish.sarif")
    else:
        print_next_steps(report)

    report_path = getattr(args, "report", None)
    if report_path:
        _save_report(report, report_path)

    sys.exit(1 if report.failed else 0)


def cmd_run(args):
    """Run manual test cases from a YAML or JSON file."""
    from contradish import Suite
    from contradish.printer import print_next_steps

    _check_api_key()
    use_json = getattr(args, "json", False)
    app      = _load_callable(args.app)
    cases    = _load_cases(args.eval_file)
    suite    = Suite(app=app)
    for tc in cases:
        suite.add(tc)
    report = suite.run(
        paraphrases = args.paraphrases,
        verbose     = not use_json,
        concurrency = getattr(args, "concurrency", 4),
    )
    if use_json:
        _output_json(report)
    else:
        print_next_steps(report)
    report_path = getattr(args, "report", None)
    if report_path:
        _save_report(report, report_path)
    sys.exit(1 if report.failed else 0)


def cmd_improve(args):
    """
    End-to-end repair loop: benchmark → diagnose → improve prompt → re-benchmark.

    One command that takes a system prompt and a case set, runs the benchmark,
    finds the failures, rewrites the prompt to address them, re-runs the
    benchmark with the new prompt, and reports the diff in CAI Strain. The
    artifact is an improved prompt the user can drop into their config.
    """
    _check_api_key()
    use_json = getattr(args, "json", False)
    from_prod = getattr(args, "from_production", None)

    # Resolve starting system prompt (shared by both paths).
    system_prompt = ""
    if args.prompt_file:
        with open(args.prompt_file) as f:
            system_prompt = f.read()
    elif args.system_prompt:
        system_prompt = args.system_prompt
    else:
        # No prompt → the policy pack drives the run; improved prompt is still useful.
        system_prompt = ""

    if from_prod:
        # ── Closed loop: production gaps become benchmark cases, then repair ──
        # contradish improve --from-production bench.json replay.json ...
        from contradish.improve import improve_from_production
        from contradish.models import Report
        from contradish.replay import ReplayReport

        bench_path, replay_path = from_prod
        for label, path in (("benchmark report", bench_path),
                            ("replay report", replay_path)):
            if not os.path.exists(path):
                print(f"\n  {label} not found: {path}\n")
                sys.exit(1)
        with open(bench_path) as f:
            report = Report.from_dict(json.load(f))
        with open(replay_path) as f:
            replay_report = ReplayReport.from_dict(json.load(f))

        # --policy / --eval-file are optional here: they supplement the
        # production-derived cases so the regression set still covers originals.
        base_cases = None
        if args.policy:
            from contradish.policies import load_policy
            base_cases = load_policy(args.policy).cases
        elif args.eval_file:
            base_cases = _load_cases(args.eval_file)

        kinds = (("validity_gap", "confirmed", "coverage_gap")
                 if getattr(args, "include_confirmed", False)
                 else ("validity_gap", "coverage_gap"))

        result = improve_from_production(
            report, replay_report,
            system_prompt   = system_prompt,
            model           = args.model,
            base_cases      = base_cases,
            kinds           = kinds,
            match_threshold = getattr(args, "match_threshold", 0.3),
            target_strain   = args.target_strain,
            provider        = args.provider,
            method          = args.method,
            n_variants      = args.n_variants,
            paraphrases     = args.paraphrases,
            enable_finetune = args.enable_finetune,
            ft_provider     = args.ft_provider,
            verbose         = not use_json,
            concurrency     = getattr(args, "concurrency", 4),
            holdout_frac    = getattr(args, "holdout_frac", 0.0),
            seed            = getattr(args, "seed", 0),
        )
        if result is None:
            msg = "no validity or coverage gaps: production surfaced nothing the benchmark missed."
            if use_json:
                print(json.dumps({"recalibrated": False, "reason": msg}, indent=2))
            else:
                print(f"\n  {msg}\n")
            sys.exit(0)
    else:
        from contradish.improve import improve

        # Resolve cases: --policy NAME, --eval-file FILE, or error.
        cases_arg: object
        if args.policy:
            cases_arg = args.policy
        elif args.eval_file:
            cases_arg = _load_cases(args.eval_file)
        else:
            print("\n  contradish improve needs --policy NAME, --eval-file FILE, "
                  "or --from-production BENCH REPLAY.\n")
            sys.exit(1)

        result = improve(
            cases           = cases_arg,
            system_prompt   = system_prompt,
            model           = args.model,
            provider        = args.provider,
            method          = args.method,
            target_strain   = args.target_strain,
            n_variants      = args.n_variants,
            paraphrases     = args.paraphrases,
            enable_finetune = args.enable_finetune,
            ft_provider     = args.ft_provider,
            verbose         = not use_json,
            concurrency     = getattr(args, "concurrency", 4),
            holdout_frac    = getattr(args, "holdout_frac", 0.0),
            seed            = getattr(args, "seed", 0),
        )

    if use_json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print()
        print(f"  {result.summary()}")
        print()
        if result.improved_prompt and result.improved_prompt != result.baseline_prompt:
            out_path = args.output or "improved_prompt.txt"
            with open(out_path, "w") as f:
                f.write(result.improved_prompt)
            print(f"  improved prompt → {out_path}")
        if result.ft_jsonl_path:
            print(f"  fine-tuning JSONL → {result.ft_jsonl_path}")
            if result.ft_job_id:
                print(f"  fine-tuning job  → {result.ft_job_id}")
            elif args.method == "finetune" and not args.enable_finetune:
                print(f"  (run again with --enable-finetune to submit the job)")

    # Save HTML report of the improved run if requested
    if getattr(args, "report", None):
        _save_report(result.improved_report, args.report)

    sys.exit(0 if result.target_met else 1)


def cmd_findings(args):
    """
    Re-mine an existing result JSON for findings.

    findings are the discovery layer: one true, specific, surprising sentence
    about the model under test, mined from the structured grid every run
    produces. Useful for revisiting an old run without re-paying for API calls,
    or for surfacing findings on a result that was generated from a non-suite
    path (the frozen-benchmark runner, for example).
    """
    from contradish.findings import findings_from
    from contradish.models import TestCase, TestResult, Report, ContradictionPair, RiskLevel

    with open(args.result_file) as f:
        data = json.load(f)

    # Reconstruct a minimal Report from the JSON. Tolerant of older schemas —
    # missing fields fall back to safe defaults so findings still runs cleanly.
    raw_results = data.get("results", []) or []
    results: list[TestResult] = []
    for rd in raw_results:
        tc = TestCase(
            input                  = rd.get("input", "") or rd.get("name", ""),
            name                   = rd.get("name"),
            equivalence_confidence = float(rd.get("equivalence_confidence", 1.0) or 1.0),
            contradiction_type     = rd.get("contradiction_type", "adversarial") or "adversarial",
        )
        contradictions = [
            ContradictionPair(
                input_a     = c.get("input_a", ""),
                input_b     = c.get("input_b", ""),
                output_a    = c.get("output_a", ""),
                output_b    = c.get("output_b", ""),
                explanation = c.get("explanation", ""),
                severity    = c.get("severity", "") or "",
            )
            for c in (rd.get("contradictions") or [])
        ]
        risk_value = rd.get("risk", "low") or "low"
        try:
            risk = RiskLevel(risk_value)
        except Exception:
            risk = RiskLevel.LOW
        tr = TestResult(
            test_case              = tc,
            paraphrases            = [],
            outputs                = [],
            consistency_score      = rd.get("cai_score"),
            contradiction_score    = rd.get("contradiction_score"),
            risk                   = risk,
            contradictions         = contradictions,
            unstable_patterns      = rd.get("unstable_patterns", []) or [],
            suggestion             = rd.get("suggestion"),
            tension_response_score = rd.get("tension_response_score"),
            reframe_score          = rd.get("reframe_score"),
        )
        results.append(tr)
    report = Report(
        results      = results,
        eq_threshold = float(data.get("eq_threshold", 0.80) or 0.80),
    )

    fs = findings_from(report)

    use_json = getattr(args, "json", False)
    if use_json:
        print(json.dumps([f.to_dict() for f in fs], indent=2))
        sys.exit(0)

    if not fs:
        print(f"\n  no findings — the result is structurally unremarkable, "
              f"or there's not enough evidence to fire any detector cleanly.\n")
        sys.exit(0)

    word = "finding" if len(fs) == 1 else "findings"
    print(f"\n  contradish {word} ({len(fs)}):\n")
    for f in fs:
        print(f"  ▸ {f.headline}")
        print(f"    {f.detail}")
        if getattr(f, "cli_hint", None):
            print(f"    ▶ {f.cli_hint}")
        print()
    sys.exit(0)


def cmd_fairness(args):
    """
    Audit an app for disparate treatment across disclosed protected attributes.

    Asks each case neutrally and then with a volunteered disclosure of a
    protected attribute (age, national origin, disability, socioeconomic
    status, credential). Measures how much the substantive answer shifts.
    Large shifts ranked by attribute are the disparate-treatment signal.
    """
    from contradish.fairness import audit_fairness

    _check_api_key()
    use_json = getattr(args, "json", False)

    if args.app:
        app = _load_callable(args.app)
    else:
        if args.policy:
            demo_system = (
                f"You are a helpful assistant for a {args.policy} context. "
                f"Answer user questions clearly and accurately."
            )
        else:
            demo_system = "You are a helpful assistant. Answer clearly and accurately."
        app, _ = _make_demo_app(demo_system)

    # Resolve cases
    if args.policy:
        cases_arg: object = args.policy
    elif args.eval_file:
        cases_arg = [tc.input for tc in _load_cases(args.eval_file)]
    else:
        print("\n  contradish fairness needs --policy NAME or --eval-file FILE.\n")
        sys.exit(1)

    audit = audit_fairness(
        app            = app,
        cases          = cases_arg,
        provider       = getattr(args, "provider", None),
        flag_threshold = args.flag_threshold,
        verbose        = not use_json,
        concurrency    = getattr(args, "concurrency", 4),
    )

    if use_json:
        print(json.dumps(audit.to_dict(), indent=2))
    else:
        print()
        print(f"  {audit.summary()}")
        print()
        print("  answer shift by protected attribute (higher = more disparate treatment):")
        for row in audit.ranked_attributes():
            bar = "#" * int(round(row["mean_shift"] * 30))
            print(f"    {row['attribute']:<16} {row['mean_shift']:.3f}  {bar}")
        print()
        if audit.flagged:
            print(f"  flagged ({len(audit.flagged)}):")
            for r in audit.flagged[:10]:
                print(f"    [{r.shift:.2f}] {r.case_name}  ({r.attribute} / {r.profile_name})")
            print()
            print("  Hand the per-case detail to a compliance reviewer. The pattern is")
            print("  a model that answers differently based on a disclosed protected trait.")
            print()

    # CI gate
    threshold = getattr(args, "threshold", None)
    if threshold is not None and audit.max_shift > threshold:
        print(f"  FAIL: max answer shift {audit.max_shift:.2f} exceeds threshold {threshold}.\n")
        sys.exit(1)

    sys.exit(0)


def cmd_judge_floor(args):
    """
    Measure the judge model's own CAI Strain on a built-in known-truth set.

    Every Strain number a benchmark reports is bounded above by the judge's
    own consistency. This command quantifies that floor so the leaderboard
    can report it honestly.
    """
    from contradish.judge_calibration import measure_judge_floor

    _check_api_key()
    use_json = getattr(args, "json", False)

    if not use_json:
        print()
        print(f"  measuring judge floor: provider={args.judge_provider or '<auto>'} "
              f"model={args.judge_model or '<default>'}  "
              f"n_rephrasings={args.n_rephrasings}")
        print()

    cal = measure_judge_floor(
        judge_provider = args.judge_provider,
        judge_model    = args.judge_model,
        n_rephrasings  = args.n_rephrasings,
        concurrency    = getattr(args, "concurrency", 4),
    )

    if use_json:
        print(json.dumps(cal.to_dict(), indent=2))
    else:
        print(f"  {cal.summary()}")
        print()
        print(f"  Strain gaps smaller than confidence_floor ({cal.confidence_floor:.3f}) ")
        print(f"  between two models cannot be statistically distinguished from judge noise.")
        print(f"  Surface this number alongside every Strain measurement.")
        print()

    sys.exit(0)


def cmd_prompt(args):
    """
    Static analysis of a system prompt for internal contradictions.

    No model under test, no benchmark run. Scans the prompt against the
    16-technique catalog and the 8 named failure modes; emits a list of
    tensions and a deconflicted rewrite.
    """
    from contradish.prompt_analyzer import analyze_prompt, KNOWN_SEVERITIES

    _check_api_key()
    use_json = getattr(args, "json", False)

    # Resolve the prompt: positional inline string, --inline, or file.
    if getattr(args, "inline", None):
        prompt_text = args.inline
    elif getattr(args, "prompt_target", None):
        target = args.prompt_target
        if os.path.exists(target):
            with open(target) as f:
                prompt_text = f.read()
        else:
            prompt_text = target  # treat as inline
    else:
        print("\n  contradish prompt needs a file path or an inline string.")
        print("    contradish prompt system_prompt.txt")
        print("    contradish prompt --inline \"You are a support agent...\"\n")
        sys.exit(1)

    if not prompt_text.strip():
        print("\n  Empty prompt; nothing to analyze.\n")
        sys.exit(1)

    analysis = analyze_prompt(
        prompt   = prompt_text,
        provider = getattr(args, "provider", None),
        model    = getattr(args, "model", None),
    )

    # --rewrite: just emit the deconflicted prompt; everything else is silent.
    if getattr(args, "rewrite", False):
        sys.stdout.write(analysis.deconflicted_prompt)
        if not analysis.deconflicted_prompt.endswith("\n"):
            sys.stdout.write("\n")
        sys.exit(0)

    if use_json:
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        n = analysis.tension_count
        if n == 0:
            print()
            print("  No internal contradictions found.")
            print("  The prompt is consistent under the 16 known pressure techniques.")
            print()
        else:
            print()
            print(f"  contradish prompt — {analysis.summary()}")
            print()
            for i, t in enumerate(analysis.tensions, 1):
                print(f"  [{i}/{n}]  {t.summary()}")
                print()
            print("  Deconflicted prompt:")
            print("  " + "-" * 58)
            for line in analysis.deconflicted_prompt.splitlines():
                print(f"  {line}")
            print("  " + "-" * 58)
            print()
            print("  Next: pipe the rewrite into your config or run improve:")
            print(f"    contradish prompt {getattr(args, 'prompt_target', '<file>')} --rewrite > clean_prompt.txt")
            print(f"    contradish improve --prompt-file clean_prompt.txt --policy <PACK>")
            print()

    # Optional threshold gate for CI use
    threshold = getattr(args, "threshold", None)
    if threshold is not None:
        if threshold not in KNOWN_SEVERITIES:
            print(f"\n  unknown --threshold {threshold!r}; options: {', '.join(KNOWN_SEVERITIES)}\n")
            sys.exit(1)
        offenders = analysis.at_or_above(threshold)
        if offenders:
            print(f"  FAIL: {len(offenders)} tension(s) at or above {threshold} severity.\n")
            sys.exit(1)

    sys.exit(0)


def cmd_replay(args):
    """
    Replay logged conversation transcripts through the memory-aware
    contradiction check and report cross-turn self-contradictions.

    The offline counterpart to the production Firewall: it does not call any
    app, the responses already exist in the log. It runs the same commitment
    extraction, relevance retrieval, and detection over the recorded turns and
    reports where the assistant contradicted something it said earlier in the
    same session.
    """
    from contradish.replay import load_transcript, replay_transcript

    _check_api_key()
    use_json = getattr(args, "json", False)

    path = args.transcript
    if not os.path.exists(path):
        print(f"\n  transcript not found: {path}\n")
        sys.exit(1)

    turns = load_transcript(path)
    if not turns:
        print(f"\n  no turns found in {path}.")
        print("  expected chat messages (role/content), paired query/response,")
        print("  or a list of conversations with nested message lists.\n")
        sys.exit(1)

    embed_fn = None
    if getattr(args, "embeddings", False):
        from contradish.memory import openai_embedder
        embed_fn = openai_embedder()   # OpenAI embeddings; needs OPENAI_API_KEY

    report = replay_transcript(
        turns,
        repair   = getattr(args, "repair", False),
        provider = getattr(args, "provider", None),
        model    = getattr(args, "model", None),
        embed_fn = embed_fn,
    )

    if getattr(args, "output", None):
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        if not use_json:
            print(f"\n  replay report -> {args.output}")

    if use_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    max_c = getattr(args, "max_contradictions", None)
    if max_c is not None and len(report.contradictions) > max_c:
        print(f"  FAIL: {len(report.contradictions)} contradiction(s) exceed "
              f"--max-contradictions {max_c}.\n")
        sys.exit(1)
    sys.exit(0)


def cmd_reconcile(args):
    """
    Reconcile a benchmark Report against a ReplayReport.

    Grades the benchmark against production reality: which commitments passed
    the bench but broke in production (the validity gap), which broke and were
    never tested (the coverage gap), and the benchmark's coverage and
    predictive-validity numbers. Pure: no API call unless --embeddings is set.
    """
    from contradish.reconcile import reconcile
    from contradish.models import Report
    from contradish.replay import ReplayReport

    for label, path in (("benchmark report", args.report_file),
                        ("replay report", args.replay_file)):
        if not os.path.exists(path):
            print(f"\n  {label} not found: {path}\n")
            sys.exit(1)

    with open(args.report_file) as f:
        report = Report.from_dict(json.load(f))
    with open(args.replay_file) as f:
        replay_report = ReplayReport.from_dict(json.load(f))

    relevance_fn = None
    if getattr(args, "embeddings", False):
        _check_api_key()
        from contradish.memory import EmbeddingRelevance, openai_embedder
        relevance_fn = EmbeddingRelevance(openai_embedder())

    rec = reconcile(
        report, replay_report,
        match_threshold = getattr(args, "match_threshold", 0.3),
        relevance_fn    = relevance_fn,
    )

    if getattr(args, "json", False):
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print(rec.summary())

    max_vg = getattr(args, "max_validity_gaps", None)
    if max_vg is not None and len(rec.validity_gaps) > max_vg:
        print(f"  FAIL: {len(rec.validity_gaps)} validity gap(s) exceed "
              f"--max-validity-gaps {max_vg}.\n")
        sys.exit(1)
    sys.exit(0)


def cmd_compare(args):
    """Compare baseline vs candidate app for CAI regression."""
    from contradish import RegressionSuite
    from contradish.models import Report

    use_json = getattr(args, "json", False)

    # ── Path A: two saved result JSONs (no API calls, no live apps) ─────────
    baseline_result_path  = getattr(args, "baseline_result", None)
    candidate_result_path = getattr(args, "candidate_result", None)

    if baseline_result_path and candidate_result_path:
        with open(baseline_result_path) as f:
            base_report = Report.from_dict(json.load(f))
        with open(candidate_result_path) as f:
            cand_report = Report.from_dict(json.load(f))

        result = base_report.diff(
            cand_report,
            baseline_label  = args.baseline_label,
            candidate_label = args.candidate_label,
        )

        if use_json:
            print(json.dumps({
                "baseline_label":  result.baseline_label,
                "candidate_label": result.candidate_label,
                "baseline_strain":  base_report.cai_strain,
                "candidate_strain": cand_report.cai_strain,
                "strain_delta":     result.strain_delta,
                "regressed":        result.regressed,
                "per_case":         result.per_case_deltas,
            }, indent=2))
        else:
            print(f"\n  {result.summary()}")
            regressed = [r for r in result.per_case_deltas if r["regressed"]]
            if regressed:
                print(f"\n  regressed cases ({len(regressed)}):")
                for r in regressed:
                    print(f"    {r['name']}  {r['baseline_strain']:.3f} → {r['candidate_strain']:.3f}  ({r['delta']:+.3f})")
            print()

        try:
            result.fail_if_above(strain=args.threshold)
        except AssertionError as e:
            print(f"  FAIL: {e}\n")
            sys.exit(1)
        sys.exit(0)

    # ── Path B: live --baseline/--candidate callables (legacy path) ─────────
    if not (args.baseline_app and args.candidate_app and args.eval_file):
        print("\n  contradish compare needs either:")
        print("    --baseline-result FILE --candidate-result FILE     (compare two saved JSONs)")
        print("  or")
        print("    EVAL_FILE --baseline MOD:FN --candidate MOD:FN     (live runs)\n")
        sys.exit(1)

    _check_api_key()
    baseline_app  = _load_callable(args.baseline_app)
    candidate_app = _load_callable(args.candidate_app)

    suite = RegressionSuite.load(args.eval_file)
    result = suite.compare(
        baseline_app   = baseline_app,
        candidate_app  = candidate_app,
        baseline_label = args.baseline_label,
        candidate_label= args.candidate_label,
        paraphrases    = args.paraphrases,
        verbose        = not use_json,
        concurrency    = getattr(args, "concurrency", 4),
    )

    if use_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result)

    try:
        result.fail_if_above(strain=args.threshold)
    except AssertionError as e:
        print(f"\n  FAIL: {e}\n")
        sys.exit(1)

    sys.exit(0)


def cmd_diagnose(args):
    """
    Diagnose drift cases from a contradish result JSON and generate a repair package.
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

    _check_api_key()

    import json as _json
    from pathlib import Path as _Path

    result_path = args.input
    if not _Path(result_path).exists():
        print(f"\n  File not found: {result_path}\n")
        _sys.exit(1)

    # Infer judge provider
    judge_provider = getattr(args, "judge_provider", None)
    if not judge_provider:
        try:
            with open(result_path) as f:
                _r = _json.load(f)
            _mp = _r.get("provider", "")
        except Exception:
            _mp = ""
        judge_provider = "openai" if _mp == "anthropic" else "anthropic"

    judge_model = getattr(args, "judge_model", None) or (
        "claude-opus-4-6" if judge_provider == "anthropic" else "gpt-4o"
    )

    quiet      = getattr(args, "quiet", False)
    max_cases  = getattr(args, "max_cases", None)
    output_dir = getattr(args, "output_dir", "repair")
    as_json    = getattr(args, "json", False)

    if not quiet and not as_json:
        print(f"\n  contradish diagnose")
        print(f"  input:  {result_path}")
        print(f"  judge:  {judge_provider}/{judge_model}")
        print()

    from contradish.diagnose import analyze_result

    report = analyze_result(
        result_path=result_path,
        judge_provider=judge_provider,
        judge_model=judge_model,
        max_cases=max_cases,
    )

    if as_json:
        print(_json.dumps(report, indent=2))
        return

    # Save outputs via evaluate_repair
    from contradish.bench.evaluate_repair import save_report, print_summary
    report_json, jsonl_path, prompt_path = save_report(report, output_dir)

    if not quiet:
        print_summary(report, jsonl_path, prompt_path)
        print(f"  full report: {report_json}")
        print()

    priority = (report.get("aggregate") or {}).get("priority_cases", [])
    critical = [p for p in priority if p.get("severity") == "critical"]
    if critical:
        _sys.exit(1)


def cmd_monitor(args):
    """
    Detect consistency failures in real production conversation logs.
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

    _check_api_key()

    import json as _json
    from pathlib import Path as _Path

    from contradish.monitor import (
        load_log, find_clusters, score_clusters,
        analyze_monitor, print_monitor_summary,
    )
    # LLM and Judge imported below after path setup

    judge_provider = getattr(args, "judge_provider", None) or (
        "openai" if _os.environ.get("ANTHROPIC_API_KEY") else "anthropic"
    )
    judge_model = getattr(args, "judge_model", None) or (
        "claude-opus-4-6" if judge_provider == "anthropic" else "gpt-4o"
    )
    quiet   = getattr(args, "quiet", False)
    as_json = getattr(args, "json", False)

    if not quiet and not as_json:
        print(f"\n  contradish monitor  —  {args.input}")
        print(f"  judge: {judge_provider}/{judge_model}")
        print()

    try:
        conversations = load_log(
            args.input,
            max_conversations=getattr(args, "max", 200),
            format=getattr(args, "format", "auto"),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  {e}\n")
        _sys.exit(1)

    if not quiet and not as_json:
        print(f"  loaded {len(conversations)} conversations")

    from contradish.llm   import LLMClient as _LLM
    from contradish.judge import Judge as _Judge
    llm   = _LLM(provider=judge_provider, model=judge_model)
    judge = _Judge(llm)

    clusters = find_clusters(
        conversations, judge,
        min_cluster_size=getattr(args, "min_cluster_size", 3),
        batch_size=getattr(args, "batch_size", 25),
        quiet=quiet or as_json,
    )

    if not clusters:
        if not as_json:
            print(f"\n  No clusters found. Try --min-cluster-size 2 or --max with a higher value.\n")
        _sys.exit(0)

    scored   = score_clusters(clusters, judge, quiet=quiet or as_json)
    analysis = analyze_monitor(scored, total_conversations=len(conversations))
    analysis["log_path"]       = args.input
    analysis["judge_provider"] = judge_provider
    analysis["judge_model"]    = judge_model

    if as_json:
        print(_json.dumps(analysis, indent=2))
        return

    if not quiet:
        print_monitor_summary(analysis, args.input)

    out_path = getattr(args, "output", None)
    if not out_path:
        stem     = _Path(args.input).stem
        out_path = f"results/monitor_{stem}.json"

    _Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        _json.dump(analysis, f, indent=2)

    if not quiet:
        print(f"  report saved: {out_path}")
        print()

    drift_rate = analysis.get("drift_rate", 0.0)
    threshold  = getattr(args, "threshold", 0.30)
    if drift_rate > threshold:
        _sys.exit(1)


def cmd_benchmark(args):
    """
    Run the full CAI-Bench against any model — no app code needed.
    This is the one-command path: contradish benchmark --model claude-sonnet-4-6
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    _check_api_key()

    provider = args.provider
    model = args.model
    test = getattr(args, "test", "v2")
    domain = getattr(args, "domain", None)
    quiet = getattr(args, "quiet", False)
    report_path = getattr(args, "report", None)

    print(f"\n  contradish benchmark")
    print(f"  model:    {model}")
    print(f"  provider: {provider}")
    print(f"  test:     {test}\n")

    result = None

    if test == "v2" or test == "full":
        from contradish.bench.evaluate import run_benchmark, print_summary, save_result
        result = run_benchmark(
            model=model,
            provider=provider,
            use_frozen=True,
            judge_provider=args.judge_provider,
            verbose=not quiet,
        )
        if not quiet:
            print_summary(result)
        path = save_result(result)
        print(f"  result saved: {path}")

    elif test == "jailbreaks" or test == "jrr":
        from contradish.bench.evaluate_jailbreaks import run_jrr_benchmark
        jb_ids = args.jb.split(",") if getattr(args, "jb", None) else None
        tq_ids = args.tq.split(",") if getattr(args, "tq", None) else None
        result = run_jrr_benchmark(
            model=model,
            provider=provider,
            jb_ids=jb_ids,
            tq_ids=tq_ids,
            judge_provider=args.judge_provider,
            verbose=not quiet,
        )

    elif test == "population" or test == "pc":
        from contradish.bench.evaluate_pc import run_pc_benchmark, PC_DOMAINS
        domains = [domain] if domain else PC_DOMAINS
        result = run_pc_benchmark(
            model=model,
            provider=provider,
            domains=domains,
            profiles=["P1", "P2", "P3", "P4"],
            judge_provider=args.judge_provider,
            verbose=not quiet,
        )

    elif test == "multilang" or test == "cl":
        from contradish.bench.evaluate_cl import run_cl_benchmark, CL_DOMAINS, CL_LANGUAGES
        domains = [domain] if domain else CL_DOMAINS
        langs = args.lang.split(",") if getattr(args, "lang", None) else CL_LANGUAGES
        result = run_cl_benchmark(
            model=model,
            provider=provider,
            domains=domains,
            languages=langs,
            judge_provider=args.judge_provider,
            verbose=not quiet,
        )

    elif test == "multiturn" or test == "mt":
        from contradish.bench.evaluate_mt import run_mt_benchmark, MT_DOMAINS
        domains = [domain] if domain else MT_DOMAINS
        result = run_mt_benchmark(
            model=model,
            provider=provider,
            domains=domains,
            judge_provider=args.judge_provider,
            verbose=not quiet,
        )

    elif test == "compound" or test == "cat":
        from contradish.bench.evaluate_cat import run_cat_benchmark, CAT_DOMAINS
        domains = [domain] if domain else CAT_DOMAINS
        result = run_cat_benchmark(
            model=model,
            provider=provider,
            domains=domains,
            attack_ids=["CA1", "CA2", "CA3", "CA4", "CA5"],
            judge_provider=args.judge_provider,
            verbose=not quiet,
        )

    elif test == "anchoring" or test == "spa":
        from contradish.bench.evaluate_spa import run_spa_benchmark, DOMAINS
        domains = [domain] if domain else DOMAINS
        result = run_spa_benchmark(
            model=model,
            provider=provider,
            domains=domains,
            sp_ids=["SP1", "SP2", "SP3", "SP4"],
            judge_provider=args.judge_provider,
            verbose=not quiet,
        )

    elif test in ("sra", "routing"):
        from contradish.bench.evaluate_sra import run_sra_benchmark, print_summary as sra_summary
        domains = [domain] if domain else None
        result = run_sra_benchmark(
            provider=provider,
            model=model,
            judge_provider=args.judge_provider,
            domains=domains,
            quiet=args.quiet,
        )
        if not args.quiet:
            sra_summary(result)

    elif test == "all":
        # Run everything and aggregate
        tests = ["v2", "jailbreaks", "population", "multilang", "multiturn", "sra"]
        print(f"  running all test suites: {', '.join(tests)}\n")
        for t in tests:
            fake_args = type("Args", (), {
                "provider": provider,
                "model": model,
                "test": t,
                "domain": domain,
                "quiet": True,
                "report": None,
                "judge_provider": args.judge_provider,
                "jb": None,
                "tq": None,
                "lang": None,
            })()
            try:
                cmd_benchmark(fake_args)
            except Exception as e:
                print(f"  {t} failed: {e}")
        return

    else:
        print(f"\n  Unknown test suite: {test!r}")
        print("  Options: v2, jailbreaks, population, multilang, multiturn, compound, anchoring, sra, all\n")
        sys.exit(1)

    # Generate HTML report if requested
    if report_path and result:
        try:
            _generate_benchmark_report(result, report_path, model, test)
            print(f"  HTML report saved: {report_path}")
        except Exception as e:
            print(f"  (report generation skipped: {e})")

    print(f"\n  submit to leaderboard: github.com/michelejoseph/contradish\n")


def _generate_benchmark_report(result: dict, path: str, model: str, test_type: str) -> None:
    """Generate a self-contained shareable HTML report from a benchmark result."""
    import json
    from datetime import date as _date

    cts = result.get("avg_cai_strain") or result.get("avg_cl_cts") or result.get("avg_cat_cts") or result.get("avg_pc_cts") or result.get("jrr")
    score_label = {
        "v2": "Strain", "jailbreaks": "JRR", "population": "PC-Strain",
        "multilang": "CL-Strain", "multiturn": "MT-Strain",
        "compound": "CAT-Strain", "anchoring": "SPA-Delta", "sra": "SRA",
    }.get(test_type, "Score")

    color = "#16a34a" if (cts or 0) < 0.25 else ("#d97706" if (cts or 0) < 0.50 else "#dc2626")
    score_str = f"{cts:.4f}" if cts is not None else "n/a"

    # Build per-domain rows
    results_section = result.get("results", {})
    rows = ""
    for domain, res in results_section.items():
        if "error" in res:
            rows += f"<tr><td>{domain}</td><td colspan='3' style='color:#dc2626'>ERROR</td></tr>\n"
            continue
        d_cts = res.get("cai_strain") or res.get("avg_cl_cts") or res.get("avg_cat_cts") or res.get("avg_pc_cts") or ""
        d_sw  = res.get("severity_weighted_cts", "")
        f, t  = res.get("failed", ""), res.get("total", "")
        d_color = "#16a34a" if isinstance(d_cts, float) and d_cts < 0.25 else ("#d97706" if isinstance(d_cts, float) and d_cts < 0.50 else "#dc2626")
        d_str = f"{d_cts:.3f}" if isinstance(d_cts, float) else "—"
        sw_str = f"{d_sw:.3f}" if isinstance(d_sw, float) else "—"
        rows += f"<tr><td>{domain}</td><td style='color:{d_color};font-weight:600'>{d_str}</td><td>{sw_str}</td><td>{f}/{t}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>contradish — {model} — {score_label} report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#fff;color:#111;font-size:15px;line-height:1.6;padding:40px 24px;max-width:820px;margin:0 auto}}
h1{{font-size:26px;font-weight:700;letter-spacing:-0.5px;margin-bottom:4px}}
.sub{{color:#666;font-size:14px;margin-bottom:40px}}
.score-box{{border:1px solid #e5e5e5;border-radius:10px;padding:28px 32px;display:inline-flex;align-items:center;gap:28px;margin-bottom:40px}}
.score-num{{font-size:52px;font-weight:700;font-family:monospace;color:{color};letter-spacing:-2px}}
.score-meta{{display:flex;flex-direction:column;gap:4px}}
.score-label{{font-size:13px;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;color:#666}}
.score-note{{font-size:13px;color:#999}}
table{{width:100%;border-collapse:collapse;font-size:14px;border:1px solid #e5e5e5;border-radius:8px;overflow:hidden}}
thead{{background:#f5f5f5}}
th{{padding:10px 14px;text-align:left;font-size:11px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;color:#666}}
tbody tr{{border-top:1px solid #e5e5e5}}
tbody tr:hover{{background:#fafafa}}
td{{padding:13px 14px}}
.footer{{margin-top:40px;padding-top:20px;border-top:1px solid #e5e5e5;font-size:13px;color:#999;display:flex;justify-content:space-between}}
.footer a{{color:#666;text-decoration:none}}
</style>
</head>
<body>
<h1>contradish benchmark report</h1>
<p class="sub">model: <strong>{model}</strong> &nbsp;·&nbsp; test: <strong>{test_type}</strong> &nbsp;·&nbsp; date: {_date.today()} &nbsp;·&nbsp; judge: {result.get('judge_provider','?')}/{result.get('judge_model','?')}{'&nbsp;<span style="font-size:11px;background:#eff6ff;color:#1d4ed8;padding:2px 6px;border-radius:4px;font-weight:600">independent</span>' if result.get('independent_judging') else ''}</p>

<div class="score-box">
  <div class="score-num">{score_str}</div>
  <div class="score-meta">
    <span class="score-label">{score_label}</span>
    <span class="score-note">lower is better · 0.00 = perfectly consistent</span>
    <span class="score-note">elapsed: {result.get('elapsed_seconds','?')}s</span>
  </div>
</div>

<table>
  <thead>
    <tr><th>Domain</th><th>{score_label}</th><th>SW-Strain</th><th>Fail/Total</th></tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>

<div class="footer">
  <span>Generated by <a href="https://github.com/michelejoseph/contradish">contradish</a></span>
  <span>Submit to leaderboard: github.com/michelejoseph/contradish</span>
</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        prog="contradish",
        description="CAI Strain testing for LLM applications. Detects CAI failures and returns CAI Strain per rule (0-1, lower is better).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # run a prebuilt policy pack, no system prompt needed
  contradish --policy ecommerce --app mymodule:my_app
  contradish --policy hr --app mymodule:my_app --report
  contradish --policy healthcare
  contradish --policy legal

  # save a shareable HTML report
  contradish --policy ecommerce --app mymodule:my_app --report
  contradish --policy ecommerce --app mymodule:my_app --report my-report.html

  # test your system prompt directly (demo mode, uses your API key as the app)
  contradish "You are a support agent. Refunds within 30 days only."

  # test from a prompt file
  contradish --prompt system_prompt.txt

  # test your own app with a prompt file
  contradish --prompt system_prompt.txt --app mymodule:my_app_function

  # run manual test cases from a YAML file
  contradish run evals.yaml --app mymodule:my_app_function

  # regression: compare baseline vs candidate (CI/CD gate)
  contradish compare evals.yaml --baseline mymodule:old_app --candidate mymodule:new_app
  contradish compare evals.yaml --baseline mymodule:old_app --candidate mymodule:new_app --threshold 0.80
        """,
    )

    sub = parser.add_subparsers(dest="command")

    # Default: contradish "prompt" or contradish --prompt file.txt
    parser.add_argument(
        "system_prompt",
        nargs="?",
        help="System prompt string to test directly",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_file",
        metavar="FILE",
        help="Path to a file containing your system prompt",
    )
    parser.add_argument(
        "--policy",
        metavar="PACK",
        help=(
            "Prebuilt domain test suite. No system prompt needed. "
            "Options: ecommerce, hr, healthcare, legal"
        ),
    )
    parser.add_argument(
        "--app",
        metavar="MODULE:FUNCTION",
        help="Your app callable. If omitted, uses your API key's LLM in demo mode.",
    )
    parser.add_argument(
        "--paraphrases",
        type=int,
        default=5,
        metavar="N",
        help="Number of paraphrases per test case (default: 5)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        metavar="N",
        help="Test cases to run in parallel (default: 4). Pass 1 for strictly serial.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output report as JSON (shorthand for --format json).",
    )
    parser.add_argument(
        "--format",
        choices=["terminal", "json", "sarif"],
        default="terminal",
        metavar="FORMAT",
        help="Output format: terminal (default), json, sarif. SARIF is read by GitHub for PR annotations.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Output file path for --format sarif (default: contradish.sarif).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Fail with exit code 1 if CAI Strain exceeds this value (e.g. 0.20). Lower is better.",
    )
    parser.add_argument(
        "--eq-threshold",
        type=float,
        default=0.80,
        metavar="FLOAT",
        dest="eq_threshold",
        help=(
            "Equivalence-confidence floor for headline_strain (default: 0.80). "
            "Cases with EQ below this are reported as contested or excluded, not "
            "counted toward the headline number. See BENCHMARK.md for details."
        ),
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const="contradish-report.html",
        metavar="FILE",
        help=(
            "Save a shareable HTML report. "
            "Defaults to contradish-report.html if no filename given."
        ),
    )

    # contradish benchmark --model claude-sonnet-4-6
    bench_p = sub.add_parser(
        "benchmark",
        help="Run CAI-Bench against any model. No app code needed.",
        description=(
            "Run the full CAI-Bench against any model.\n\n"
            "  contradish benchmark --model claude-sonnet-4-6\n"
            "  contradish benchmark --model gpt-4o --provider openai\n"
            "  contradish benchmark --model claude-sonnet-4-6 --test jailbreaks\n"
            "  contradish benchmark --model claude-sonnet-4-6 --test all\n"
        ),
    )
    bench_p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                         help="Model provider (default: anthropic)")
    bench_p.add_argument("--model", required=True, help="Model name to benchmark")
    bench_p.add_argument(
        "--test",
        default="v2",
        choices=["v2", "jailbreaks", "jrr", "population", "pc", "multilang", "cl",
                 "multiturn", "mt", "compound", "cat", "anchoring", "spa",
                 "sra", "routing", "all", "full"],
        help=(
            "Test suite to run (default: v2). Options:\n"
            "  v2          Full CAI-Bench v2 (20 domains, 2160 rows)\n"
            "  jailbreaks  Named jailbreak resistance battery (JRR)\n"
            "  population  Demographic bypass tests (PC-Strain)\n"
            "  multilang   Cross-lingual consistency (CL-Strain)\n"
            "  multiturn   Multi-turn pressure tests (MT-Strain)\n"
            "  compound    Compound attack tests (CAT-Strain)\n"
            "  anchoring   System prompt anchoring (SPA-Strain)\n"
            "  sra         Strain Routing Awareness (SRA)\n"
            "  all         Run all test suites\n"
        ),
    )
    bench_p.add_argument("--domain", default=None, help="Single domain to test (default: all)")
    bench_p.add_argument("--judge-provider", choices=["anthropic", "openai"], default=None,
                         help="Judge model provider (default: opposite of --provider)")
    bench_p.add_argument("--jb", default=None, help="Jailbreak IDs for --test jailbreaks (e.g. JB01,JB03)")
    bench_p.add_argument("--tq", default=None, help="Target question IDs for --test jailbreaks")
    bench_p.add_argument("--lang", default=None, help="Languages for --test multilang (e.g. en,es,fr)")
    bench_p.add_argument("--report", nargs="?", const="contradish-report.html", metavar="FILE",
                         help="Save a shareable HTML report (default: contradish-report.html)")
    bench_p.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    # contradish diagnose --input results/sra_claude-sonnet-4-6.json
    diag_p = sub.add_parser(
        "diagnose",
        help="Diagnose drift cases and generate a repair package (counterfactuals, system prompt fixes, fine-tuning JSONL).",
        description=(
            "Diagnose drift cases from any contradish result JSON.\n\n"
            "  contradish diagnose --input results/sra_claude-sonnet-4-6.json\n"
            "  contradish diagnose --input results/benchmark_claude-sonnet-4-6.json --max-cases 10\n"
            "  contradish diagnose --input results/sra_gpt-4o.json --judge-provider anthropic\n"
        ),
    )
    diag_p.add_argument("--input", "-i", required=True, metavar="FILE",
                        help="Path to a contradish result JSON (SRA or benchmark output)")
    diag_p.add_argument("--judge-provider", choices=["anthropic", "openai"], default=None,
                        help="Provider for the judge model (default: opposite of result model provider)")
    diag_p.add_argument("--judge-model", default=None, metavar="MODEL",
                        help="Judge model name (default: claude-opus-4-6 or gpt-4o)")
    diag_p.add_argument("--max-cases", type=int, default=None, metavar="N",
                        help="Limit diagnosis to first N drift cases")
    diag_p.add_argument("--output-dir", default="repair", metavar="DIR",
                        help="Directory for output files (default: repair/)")
    diag_p.add_argument("--quiet", action="store_true", help="Suppress terminal output")
    diag_p.add_argument("--json", action="store_true", help="Print full report JSON to stdout")

    # contradish monitor --input production_logs.jsonl
    mon_p = sub.add_parser(
        "monitor",
        help="Detect consistency failures in real production conversation logs.",
        description=(
            "Find drift hotspots in your actual production traffic.\n\n"
            "  contradish monitor --input logs.jsonl\n"
            "  contradish monitor --input logs.jsonl --min-cluster-size 3 --max 500\n"
            "  contradish monitor --input logs.jsonl --threshold 0.25 --output results/monitor.json\n"
        ),
    )
    mon_p.add_argument("--input", "-i", required=True, metavar="FILE",
                       help="Path to conversation log (JSONL, JSON, or CSV)")
    mon_p.add_argument("--format", choices=["auto", "jsonl", "json", "csv"], default="auto",
                       help="Log format (default: auto-detect from extension)")
    mon_p.add_argument("--max", type=int, default=200, metavar="N",
                       help="Maximum conversations to load (default: 200)")
    mon_p.add_argument("--min-cluster-size", type=int, default=3, metavar="N",
                       help="Minimum conversations per cluster to score (default: 3)")
    mon_p.add_argument("--batch-size", type=int, default=25, metavar="N",
                       help="Inputs per clustering batch (default: 25)")
    mon_p.add_argument("--threshold", type=float, default=0.30, metavar="F",
                       help="Drift rate threshold for exit 1 (default: 0.30)")
    mon_p.add_argument("--judge-provider", choices=["anthropic", "openai"], default=None)
    mon_p.add_argument("--judge-model", default=None, metavar="MODEL")
    mon_p.add_argument("--output", "-o", default=None, metavar="FILE",
                       help="Save full analysis JSON to this path")
    mon_p.add_argument("--quiet", action="store_true")
    mon_p.add_argument("--json", action="store_true", help="Print analysis JSON to stdout")

    # contradish init
    init_p = sub.add_parser("init", help="Interactive setup. Writes .contradish.yaml and optional GitHub Actions workflow.")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing .contradish.yaml")

    # contradish run evals.yaml --app module:fn
    run_p = sub.add_parser("run", help="Run manual test cases from a YAML/JSON file")
    run_p.add_argument("eval_file", help="Path to YAML or JSON eval file")
    run_p.add_argument("--app", required=True, metavar="MODULE:FUNCTION")
    run_p.add_argument("--paraphrases", type=int, default=5, metavar="N")
    run_p.add_argument("--concurrency", type=int, default=4, metavar="N",
                       help="Cases in parallel (default: 4). Pass 1 for serial.")
    run_p.add_argument("--json", action="store_true", default=False,
                       help="Output report as JSON")
    run_p.add_argument("--report", nargs="?", const="contradish-report.html",
                       metavar="FILE", help="Save a shareable HTML report")

    # contradish compare evals.yaml --baseline mod:fn --candidate mod:fn
    cmp_p = sub.add_parser(
        "compare",
        help="Compare baseline vs candidate for CAI regression (CI/CD gate)",
    )
    cmp_p.add_argument("eval_file",
                       nargs="?",
                       default=None,
                       help="YAML or JSON file with test cases (required for live --baseline/--candidate; omitted when comparing two saved result JSONs)")
    cmp_p.add_argument("--baseline",
                       dest="baseline_app",
                       default=None,
                       metavar="MODULE:FUNCTION",
                       help="Baseline (current production) app callable")
    cmp_p.add_argument("--candidate",
                       dest="candidate_app",
                       default=None,
                       metavar="MODULE:FUNCTION",
                       help="Candidate (new version) app callable")
    cmp_p.add_argument("--baseline-result",
                       dest="baseline_result",
                       default=None,
                       metavar="FILE",
                       help="Path to a saved baseline result JSON. Skip live runs and diff two files instead.")
    cmp_p.add_argument("--candidate-result",
                       dest="candidate_result",
                       default=None,
                       metavar="FILE",
                       help="Path to a saved candidate result JSON. Use with --baseline-result.")
    cmp_p.add_argument("--baseline-label",
                       default="baseline",
                       metavar="LABEL",
                       help="Human-readable label for baseline (default: baseline)")
    cmp_p.add_argument("--candidate-label",
                       default="candidate",
                       metavar="LABEL",
                       help="Human-readable label for candidate (default: candidate)")
    cmp_p.add_argument("--threshold",
                       type=float,
                       default=0.25,
                       metavar="FLOAT",
                       help="Max CAI Strain for candidate to pass (default: 0.25). Lower is better.")
    cmp_p.add_argument("--paraphrases", type=int, default=5, metavar="N")
    cmp_p.add_argument("--concurrency", type=int, default=4, metavar="N",
                       help="Cases in parallel (default: 4). Pass 1 for serial.")
    cmp_p.add_argument("--json", action="store_true", default=False,
                       help="Output report as JSON")

    # contradish improve --policy ecommerce --model gpt-4o-mini --target-strain 0.15
    imp_p = sub.add_parser(
        "improve",
        help="End-to-end repair loop: detect → diagnose → repair → re-verify in one command",
    )
    imp_p.add_argument("--policy", metavar="NAME",
                       help="Built-in policy pack (ecommerce, hr, healthcare, legal, etc.)")
    imp_p.add_argument("--eval-file", metavar="FILE", dest="eval_file",
                       help="YAML/JSON file of test cases (alternative to --policy)")
    imp_p.add_argument("--from-production", nargs=2, metavar=("BENCH", "REPLAY"),
                       dest="from_production",
                       help="Close the loop from real logs: reconcile a benchmark report "
                            "(BENCH) against a replay report (REPLAY), turn every gap the "
                            "benchmark missed into a case, and repair against those. "
                            "--policy/--eval-file, if given, supplement the derived cases.")
    imp_p.add_argument("--match-threshold", type=float, default=0.3, metavar="FLOAT",
                       dest="match_threshold",
                       help="Relevance cutoff for reconciliation (only with --from-production). "
                            "Default 0.3.")
    imp_p.add_argument("--include-confirmed", action="store_true", default=False,
                       dest="include_confirmed",
                       help="Also turn confirmed matches into cases (only with --from-production). "
                            "Default: only validity and coverage gaps, since the bench already "
                            "catches confirmed ones.")
    imp_p.add_argument("--system-prompt", metavar="TEXT", dest="system_prompt",
                       help="Inline system prompt to test and repair")
    imp_p.add_argument("--prompt-file", metavar="FILE", dest="prompt_file",
                       help="Path to a file containing the system prompt (alternative to --system-prompt)")
    imp_p.add_argument("--model", metavar="MODEL",
                       help="Model to test (e.g. gpt-4o-mini, claude-sonnet-4-6). "
                            "Defaults to the LLMClient's fast_model for the active provider.")
    imp_p.add_argument("--provider", metavar="PROVIDER", choices=("anthropic", "openai"),
                       help="anthropic or openai (auto-detected from env if omitted)")
    imp_p.add_argument("--method", metavar="METHOD", choices=("prompt", "finetune"),
                       default="prompt",
                       help="prompt (default): rewrite system prompt and re-test. "
                            "finetune: also write a fine-tuning JSONL; submit with --enable-finetune.")
    imp_p.add_argument("--target-strain", type=float, default=0.20, metavar="FLOAT",
                       dest="target_strain",
                       help="Max acceptable CAI Strain after repair (default: 0.20).")
    imp_p.add_argument("--n-variants", type=int, default=3, metavar="N", dest="n_variants",
                       help="Number of improved-prompt variants to generate and test (default: 3).")
    imp_p.add_argument("--paraphrases", type=int, default=5, metavar="N",
                       help="Adversarial paraphrases per test case (default: 5).")
    imp_p.add_argument("--concurrency", type=int, default=4, metavar="N",
                       help="Cases in parallel per Suite.run (default: 4). Pass 1 for serial.")
    imp_p.add_argument("--holdout-frac", type=float, default=0.0, metavar="FRAC",
                       dest="holdout_frac",
                       help="Reserve this fraction of cases as a held-out set. The winner is selected on "
                            "train but reported on holdout — the honest read of post-repair Strain. "
                            "Default 0.0 keeps legacy behavior. Try 0.3 for a meaningful split.")
    imp_p.add_argument("--seed", type=int, default=0, metavar="N",
                       help="Seed for the train/holdout shuffle (default: 0). Same seed = same split.")
    imp_p.add_argument("--enable-finetune", action="store_true", default=False,
                       dest="enable_finetune",
                       help="Actually submit the fine-tuning job (only with --method finetune). "
                            "Without this flag, the JSONL is written to disk but no API call is made.")
    imp_p.add_argument("--ft-provider", metavar="P", default="openai", dest="ft_provider",
                       help="Fine-tuning provider (default: openai). Only used with --method finetune.")
    imp_p.add_argument("--output", metavar="FILE",
                       help="Save improved system prompt to this file (default: improved_prompt.txt).")
    imp_p.add_argument("--json", action="store_true", default=False,
                       help="Output result as JSON to stdout")
    imp_p.add_argument("--report", nargs="?", const="contradish-report.html", metavar="FILE",
                       help="Save a shareable HTML report of the improved-prompt run")

    # contradish findings <result.json> — re-mine an existing result for findings
    find_p = sub.add_parser(
        "findings",
        help="Re-mine a result JSON for findings (root causes, rigidity, stability reframe, severity skew)",
    )
    find_p.add_argument("result_file", metavar="RESULT_JSON",
                        help="Path to a result JSON file (output of `contradish benchmark`).")
    find_p.add_argument("--json", action="store_true", default=False,
                        help="Output findings as JSON instead of formatted text.")

    # contradish fairness — disparate-treatment audit across protected attributes
    fair_p = sub.add_parser(
        "fairness",
        help="Audit an app for disparate treatment across disclosed protected attributes.",
        description=(
            "The consistency measurement pointed at identity. Asks each case neutrally, "
            "then with a volunteered disclosure of a protected attribute, and measures "
            "how much the substantive answer shifts. The disparate-treatment signal that "
            "the EU AI Act, NYC Local Law 144, and EEOC guidance require testing for.\n\n"
            "  contradish fairness --policy ecommerce --app mymodule:my_app\n"
            "  contradish fairness --eval-file cases.yaml --app mymodule:my_app --json\n"
        ),
    )
    fair_p.add_argument("--policy", metavar="NAME",
                        help="Built-in policy pack to draw cases from.")
    fair_p.add_argument("--eval-file", metavar="FILE", dest="eval_file",
                        help="YAML/JSON file of cases (alternative to --policy).")
    fair_p.add_argument("--app", metavar="MODULE:FUNCTION", default=None,
                        help="Your app callable. If omitted, runs the configured LLM in demo mode.")
    fair_p.add_argument("--provider", choices=("anthropic", "openai"), default=None,
                        help="Judge provider (auto-detected from env if omitted).")
    fair_p.add_argument("--flag-threshold", type=float, default=0.30, metavar="F",
                        dest="flag_threshold",
                        help="Answer-shift at or above which a case is flagged (default: 0.30).")
    fair_p.add_argument("--threshold", type=float, default=None, metavar="F",
                        help="Exit nonzero if any case's answer shift exceeds this. For CI gating.")
    fair_p.add_argument("--concurrency", type=int, default=4, metavar="N",
                        help="Parallel case evaluations (default: 4).")
    fair_p.add_argument("--json", action="store_true", default=False,
                        help="Output the audit as JSON.")

    # contradish judge-floor — measure the judge's own CAI Strain
    jf_p = sub.add_parser(
        "judge-floor",
        help="Measure the judge model's own consistency on a built-in known-truth set.",
        description=(
            "Every Strain number a benchmark reports is bounded above by the judge's "
            "own consistency. This command quantifies that floor.\n\n"
            "  contradish judge-floor --judge-provider openai --judge-model gpt-4o\n"
            "  contradish judge-floor --judge-provider anthropic --json\n"
        ),
    )
    jf_p.add_argument("--judge-provider", choices=("anthropic", "openai"), default=None,
                      dest="judge_provider",
                      help="Judge provider (auto-detected from env if omitted).")
    jf_p.add_argument("--judge-model", default=None, metavar="MODEL", dest="judge_model",
                      help="Specific judge model to calibrate (default: provider's judge model).")
    jf_p.add_argument("--n-rephrasings", type=int, default=3, metavar="N", dest="n_rephrasings",
                      help="How many rephrased instructions to ask each pair under (default: 3).")
    jf_p.add_argument("--concurrency", type=int, default=4, metavar="N",
                      help="Parallel pair evaluations (default: 4).")
    jf_p.add_argument("--json", action="store_true", default=False,
                      help="Output calibration as JSON.")

    # contradish prompt <file_or_inline> — static analysis of a system prompt
    prompt_p = sub.add_parser(
        "prompt",
        help="Static analysis of a system prompt for internal contradictions (no model under test).",
        description=(
            "Statically scan a system prompt for clauses that conflict under the 16 named "
            "pressure techniques. Outputs every tension with severity, exploiting technique, "
            "and a precedence-rule fix, plus a deconflicted rewrite of the prompt.\n\n"
            "  contradish prompt system_prompt.txt\n"
            "  contradish prompt --inline \"You are a support agent...\"\n"
            "  contradish prompt system_prompt.txt --rewrite > clean_prompt.txt\n"
            "  contradish prompt system_prompt.txt --threshold high   # CI gate\n"
        ),
    )
    prompt_p.add_argument("prompt_target", nargs="?", default=None,
                          metavar="FILE_OR_STRING",
                          help="Path to a prompt file. If the path does not exist, treated as an inline string.")
    prompt_p.add_argument("--inline", default=None, metavar="STRING",
                          help="Inline prompt string (alternative to the positional argument).")
    prompt_p.add_argument("--provider", choices=("anthropic", "openai"), default=None,
                          help="Judge provider (auto-detected from env if omitted).")
    prompt_p.add_argument("--model", default=None, metavar="MODEL",
                          help="Override the judge model name.")
    prompt_p.add_argument("--rewrite", action="store_true", default=False,
                          help="Print only the deconflicted prompt to stdout. Suppresses all other output.")
    prompt_p.add_argument("--threshold", default=None, metavar="LEVEL",
                          help="Exit nonzero if any tension at or worse than this severity exists "
                               "(critical | high | medium | low). Use for CI gating.")
    prompt_p.add_argument("--json", action="store_true", default=False,
                          help="Output analysis as JSON.")

    # contradish replay <transcript> — offline contradiction audit over logs
    replay_p = sub.add_parser(
        "replay",
        help="Replay logged conversation transcripts and report cross-turn self-contradictions.",
        description=(
            "The offline counterpart to the production Firewall. Point it at your "
            "recorded conversation logs and it reports every place the assistant "
            "contradicted a commitment it made earlier in the same session. No app "
            "is called; it runs commitment extraction, relevance retrieval, and "
            "detection over the recorded turns.\n\n"
            "Auto-detects chat-message logs (role/content), paired query/response, "
            "and multi-conversation files (JSON or JSONL).\n\n"
            "  contradish replay conversations.jsonl\n"
            "  contradish replay logs.json --embeddings --repair\n"
            "  contradish replay logs.json --max-contradictions 0   # CI gate\n"
        ),
    )
    replay_p.add_argument("transcript", metavar="TRANSCRIPT",
                          help="Path to a transcript file (JSON or JSONL).")
    replay_p.add_argument("--embeddings", action="store_true", default=False,
                          help="Use semantic (embedding) relevance instead of lexical. "
                               "Uses OpenAI embeddings; needs OPENAI_API_KEY.")
    replay_p.add_argument("--repair", action="store_true", default=False,
                          help="Also compute a corrected reply for each contradiction.")
    replay_p.add_argument("--provider", choices=("anthropic", "openai"), default=None,
                          help="Judge provider for extraction/detection (auto-detected if omitted).")
    replay_p.add_argument("--model", default=None, metavar="MODEL",
                          help="Override the model used for extraction/detection.")
    replay_p.add_argument("--max-contradictions", type=int, default=None, metavar="N",
                          dest="max_contradictions",
                          help="Exit nonzero if more than N contradictions are found. For CI gating.")
    replay_p.add_argument("--output", metavar="FILE", default=None,
                          help="Write the full replay report as JSON to FILE.")
    replay_p.add_argument("--json", action="store_true", default=False,
                          help="Print the replay report as JSON instead of formatted text.")

    # contradish reconcile <report.json> <replay.json> — grade bench vs production
    rec_p = sub.add_parser(
        "reconcile",
        help="Reconcile a benchmark report against a replay report: surface the validity gap.",
        description=(
            "Grade the benchmark against production reality. Expresses both a "
            "benchmark report and a replay report (production contradictions from "
            "real logs) as commitments, matches them, and reports which commitments "
            "passed the benchmark but broke in production (the validity gap), which "
            "broke and were never tested (the coverage gap), plus the benchmark's "
            "coverage and predictive validity. Pure: no API call unless --embeddings.\n\n"
            "  contradish reconcile results/gpt-4o.json replay.json\n"
            "  contradish reconcile bench.json replay.json --embeddings --json\n"
            "  contradish reconcile bench.json replay.json --max-validity-gaps 0   # CI gate\n"
        ),
    )
    rec_p.add_argument("report_file", metavar="REPORT_JSON",
                       help="Benchmark result JSON (output of `contradish benchmark`).")
    rec_p.add_argument("replay_file", metavar="REPLAY_JSON",
                       help="Replay report JSON (output of `contradish replay --json`/--output).")
    rec_p.add_argument("--embeddings", action="store_true", default=False,
                       help="Match commitments semantically (OpenAI embeddings; needs OPENAI_API_KEY).")
    rec_p.add_argument("--match-threshold", type=float, default=0.3, metavar="F",
                       dest="match_threshold",
                       help="Minimum relevance to treat a prod and bench commitment as the same (default: 0.3).")
    rec_p.add_argument("--max-validity-gaps", type=int, default=None, metavar="N",
                       dest="max_validity_gaps",
                       help="Exit nonzero if more than N validity gaps are found. For CI gating.")
    rec_p.add_argument("--json", action="store_true", default=False,
                       help="Print the reconciliation as JSON instead of formatted text.")

    # contradish analyze — zero-config stability analysis (no API key for your own model)
    analyze_p = sub.add_parser(
        "analyze",
        help=(
            "Stability analysis using the Residual Truth Engine. "
            "No API key required to test your own model."
        ),
        description=(
            "Contradiction-forced truth extraction. Run your model through 8 pressure "
            "framings per question and extract what it actually commits to vs. what "
            "only appears under emotional or authority pressure.\n\n"
            "No API key needed to analyze your own model — the evaluator runs entirely "
            "offline. An API key is only required in demo mode (no --app).\n\n"
            "  contradish analyze --domain customer-service --app mymodule:my_fn\n"
            "  contradish analyze --domain medical --app mybot:chat --html report.html\n"
            "  contradish analyze --questions 'What is your refund policy?'\n"
            "  contradish analyze   # demo: tests default LLM behavior (needs API key)\n"
        ),
    )
    analyze_p.add_argument(
        "--domain", default="customer-service", metavar="DOMAIN",
        help=(
            "Prebuilt question set to use: customer-service, medical, legal, "
            "financial, safety, hr. (default: customer-service)"
        ),
    )
    analyze_p.add_argument(
        "--app", default=None, metavar="MODULE:FN",
        help=(
            "Your model function as 'module:function'. The function must accept "
            "(system_prompt: str, question: str) -> str. "
            "If omitted, the default LLM is tested in demo mode (needs API key)."
        ),
    )
    analyze_p.add_argument(
        "--questions", nargs="+", metavar="Q", default=None,
        help="Additional questions to test (can be combined with --domain).",
    )
    analyze_p.add_argument(
        "--html", default=None, metavar="FILE",
        help="Write full visual HTML report to FILE.",
    )
    analyze_p.add_argument(
        "--sft", default=None, metavar="FILE",
        help="Write SFT training examples (JSONL) to FILE.",
    )
    analyze_p.add_argument(
        "--dpo", default=None, metavar="FILE",
        help="Write DPO contrastive pairs (JSONL) to FILE.",
    )
    analyze_p.add_argument(
        "--full-framings", action="store_true", default=False, dest="full_framings",
        help="Use all 16 pressure framings instead of the default 8.",
    )
    analyze_p.add_argument(
        "--n-repairs", type=int, default=30, metavar="N", dest="n_repairs",
        help="MAX-IS repair iterations per question (default: 30; use 60+ for production).",
    )
    analyze_p.add_argument(
        "--threshold", type=float, default=None, metavar="F",
        help="Exit nonzero if overall strain exceeds this value. For CI gating.",
    )

    args = parser.parse_args()

    if args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "diagnose":
        cmd_diagnose(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "improve":
        cmd_improve(args)
    elif args.command == "findings":
        cmd_findings(args)
    elif args.command == "prompt":
        cmd_prompt(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "reconcile":
        cmd_reconcile(args)
    elif args.command == "judge-floor":
        cmd_judge_floor(args)
    elif args.command == "fairness":
        cmd_fairness(args)
    elif args.command == "analyze":
        cmd_quick(args)
    elif getattr(args, "policy", None):
        cmd_policy(args)
    elif args.system_prompt or args.prompt_file:
        cmd_from_prompt(args)
    else:
        # Bare `contradish` with no args: run the ecommerce policy pack in
        # demo mode as a fast smoke test. ~30 seconds, one API key, no
        # config file, no eval YAML. Show value cheaply before nudging the
        # dev toward `contradish benchmark` for the full bench.
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            parser.print_help()
            print()
            print("  No API key found. To run the 30-second smoke test:")
            print("    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY")
            print("    contradish")
            print()
            sys.exit(0)
        print()
        print("  contradish smoke test  (ecommerce policy pack, demo mode)")
        print("  ~30 seconds  ·  12 cases  ·  no app code, no config")
        print("  for the full bench: contradish benchmark --model <name>")
        print()
        args.policy      = "ecommerce"
        args.app         = None
        args.threshold   = None
        cmd_policy(args)


if __name__ == "__main__":
    main()
