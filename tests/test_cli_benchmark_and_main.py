"""
Coverage for the two largest untested chunks of contradish/cli.py:
cmd_benchmark (the `contradish benchmark` dispatcher across ~10 bench
suites) and main() (the ~700-line argparse construction + subcommand
dispatch chain).

cmd_benchmark does function-local `from contradish.bench.evaluate_X import
run_X_benchmark` for whichever suite --test selects. contradish.bench.* are
themselves large, separately-scoped modules (not covered by this file, or
by this session's "harden the core" pass) -- so rather than importing the
real ones (slow, and would require real API calls), each test injects a
fake module into sys.modules before calling cmd_benchmark, matching the
existing pattern used for optional SDKs elsewhere in this suite (e.g.
tests/test_exporters.py's fake `phoenix` module). contradish.bench itself
(the parent package, just a docstring) is cheap to really import, so only
the leaf evaluate_* modules are faked.

main() reads real sys.argv and dispatches via an if/elif chain on
args.command to one of 16 cmd_* handlers (imported already-tested
elsewhere) or a policy/prompt/bare-smoke-test fallback. Since parser
construction runs top-to-bottom on every call regardless of which
subcommand is invoked, a single successful call already exercises every
add_argument/add_parser line; what actually needs one test per branch is
the dispatch chain itself. Each test monkeypatches sys.argv and every
cmd_* attribute on the cli module with a recorder, so main() never runs
real command logic -- this file tests routing, not the handlers (those
have their own test files already).

REAL BUG found while writing these tests, flagged but NOT fixed here (see
test_main_bare_system_prompt_argument_is_currently_broken for the full
writeup): the bare positional form of the CLI --
    contradish "You are a support agent. Refunds within 30 days only."
-- the very first example in the README's Quickstart -- currently crashes
with argparse exit code 2. It's a real argparse limitation (subparsers'
nargs=PARSER positional always claims an unrecognized lone token before
the fallback `system_prompt` positional gets a chance, regardless of
declaration order), not a typo or a one-line fix -- correcting it means
restructuring main()'s single-parser design, which deserves its own
careful pass rather than a fix bundled into a test-coverage commit. The
other two documented ways to pass a prompt (`--policy X` and `--prompt
file.txt`) are unaffected and were verified working directly.

Not covered (documented, not papered over): the trailing
`if __name__ == "__main__": main()` two-line block -- see the comment at
the bottom of this file, after the last test.
"""
import json
import sys
import types

import pytest

import contradish.cli as cli
from contradish.cli import cmd_benchmark, _generate_benchmark_report


# ── shared helpers ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


def _run(fn, *a, **kw):
    """Call a cmd_* handler and capture its sys.exit code (None if it didn't exit)."""
    code = None
    try:
        fn(*a, **kw)
    except SystemExit as e:
        code = e.code
    return code


def _fake_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _bench_args(**over):
    ns = types.SimpleNamespace(
        provider="anthropic", model="claude-sonnet-4-6", test="v2", domain=None,
        quiet=True, report=None, output_json=None, judge_provider=None,
        judge_votes=1, jb=None, tq=None, lang=None,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _result(**over):
    base = {"avg_cai_strain": 0.12, "results": {}}
    base.update(over)
    return base


# ── cmd_benchmark: v2 / full ─────────────────────────────────────────────

def test_cmd_benchmark_v2_calls_run_benchmark_and_saves(monkeypatch, capsys):
    calls = {}
    def run_benchmark(**kw):
        calls["run_kw"] = kw
        return _result()
    def print_summary(result):
        calls["printed"] = result
    def save_result(result):
        calls["saved"] = result
        return "results/out.json"
    _fake_module(monkeypatch, "contradish.bench.evaluate",
                 run_benchmark=run_benchmark, print_summary=print_summary, save_result=save_result)

    code = _run(cmd_benchmark, _bench_args(test="v2", quiet=False))
    assert code is None
    assert calls["run_kw"]["model"] == "claude-sonnet-4-6"
    assert calls["run_kw"]["provider"] == "anthropic"
    assert calls["run_kw"]["use_frozen"] is True
    assert "printed" in calls  # not quiet -> print_summary called
    out = capsys.readouterr().out
    assert "result saved: results/out.json" in out
    assert "submit to leaderboard" in out


def test_cmd_benchmark_full_alias_and_quiet_skips_print_summary(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate",
                 run_benchmark=lambda **kw: _result(),
                 print_summary=lambda r: calls.setdefault("printed", True),
                 save_result=lambda r: "out.json")
    _run(cmd_benchmark, _bench_args(test="full", quiet=True))
    assert "printed" not in calls  # quiet=True -> no print_summary


# ── cmd_benchmark: jailbreaks / jrr ──────────────────────────────────────

def test_cmd_benchmark_jailbreaks_splits_jb_and_tq_ids(monkeypatch):
    calls = {}
    def run_jrr_benchmark(**kw):
        calls["kw"] = kw
        return _result(jrr=0.5)
    _fake_module(monkeypatch, "contradish.bench.evaluate_jailbreaks", run_jrr_benchmark=run_jrr_benchmark)
    _run(cmd_benchmark, _bench_args(test="jailbreaks", jb="JB1,JB2", tq="TQ1"))
    assert calls["kw"]["jb_ids"] == ["JB1", "JB2"]
    assert calls["kw"]["tq_ids"] == ["TQ1"]


def test_cmd_benchmark_jrr_alias_defaults_ids_to_none(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_jailbreaks",
                 run_jrr_benchmark=lambda **kw: calls.setdefault("kw", kw) or _result())
    _run(cmd_benchmark, _bench_args(test="jrr", jb=None, tq=None))
    assert calls["kw"]["jb_ids"] is None
    assert calls["kw"]["tq_ids"] is None


# ── cmd_benchmark: population / pc ───────────────────────────────────────

def test_cmd_benchmark_population_defaults_to_all_pc_domains(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_pc",
                 run_pc_benchmark=lambda **kw: calls.setdefault("kw", kw) or _result(),
                 PC_DOMAINS=["ecommerce", "hr"])
    _run(cmd_benchmark, _bench_args(test="population", domain=None))
    assert calls["kw"]["domains"] == ["ecommerce", "hr"]
    assert calls["kw"]["profiles"] == ["P1", "P2", "P3", "P4"]


def test_cmd_benchmark_pc_alias_narrows_to_one_domain(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_pc",
                 run_pc_benchmark=lambda **kw: calls.setdefault("kw", kw) or _result(),
                 PC_DOMAINS=["ecommerce", "hr"])
    _run(cmd_benchmark, _bench_args(test="pc", domain="hr"))
    assert calls["kw"]["domains"] == ["hr"]


# ── cmd_benchmark: multilang / cl ────────────────────────────────────────

def test_cmd_benchmark_multilang_splits_langs_or_defaults(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_cl",
                 run_cl_benchmark=lambda **kw: calls.setdefault("kw", kw) or _result(),
                 CL_DOMAINS=["ecommerce"], CL_LANGUAGES=["en", "es", "fr"])
    _run(cmd_benchmark, _bench_args(test="multilang", lang="en,de"))
    assert calls["kw"]["languages"] == ["en", "de"]

    calls.clear()
    _run(cmd_benchmark, _bench_args(test="cl", lang=None))
    assert calls["kw"]["languages"] == ["en", "es", "fr"]


# ── cmd_benchmark: multiturn / mt ────────────────────────────────────────

def test_cmd_benchmark_multiturn_and_mt_alias(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_mt",
                 run_mt_benchmark=lambda **kw: calls.__setitem__("kw", kw) or _result(),
                 MT_DOMAINS=["ecommerce"])
    _run(cmd_benchmark, _bench_args(test="multiturn", domain="hr"))
    assert calls["kw"]["domains"] == ["hr"]
    _run(cmd_benchmark, _bench_args(test="mt", domain=None))
    assert calls["kw"]["domains"] == ["ecommerce"]


# ── cmd_benchmark: compound / cat ────────────────────────────────────────

def test_cmd_benchmark_compound_and_cat_alias(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_cat",
                 run_cat_benchmark=lambda **kw: calls.setdefault("kw", kw) or _result(),
                 CAT_DOMAINS=["ecommerce"])
    _run(cmd_benchmark, _bench_args(test="compound"))
    assert calls["kw"]["attack_ids"] == ["CA1", "CA2", "CA3", "CA4", "CA5"]
    _run(cmd_benchmark, _bench_args(test="cat"))
    assert "kw" in calls


# ── cmd_benchmark: anchoring / spa ───────────────────────────────────────

def test_cmd_benchmark_anchoring_and_spa_alias(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_spa",
                 run_spa_benchmark=lambda **kw: calls.setdefault("kw", kw) or _result(),
                 DOMAINS=["ecommerce"])
    _run(cmd_benchmark, _bench_args(test="anchoring"))
    assert calls["kw"]["sp_ids"] == ["SP1", "SP2", "SP3", "SP4"]
    _run(cmd_benchmark, _bench_args(test="spa"))
    assert "kw" in calls


# ── cmd_benchmark: witness / wa ──────────────────────────────────────────

def test_cmd_benchmark_witness_and_wa_alias(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_wa",
                 run_wa_benchmark=lambda **kw: calls.setdefault("kw", kw) or _result(),
                 DOMAINS=["ecommerce"])
    _run(cmd_benchmark, _bench_args(test="witness"))
    assert calls["kw"]["wa_ids"] == ["WA1", "WA2", "WA3", "WA4"]
    _run(cmd_benchmark, _bench_args(test="wa"))
    assert "kw" in calls


# ── cmd_benchmark: cache-invalidation / ci ───────────────────────────────

def test_cmd_benchmark_cache_invalidation_and_ci_alias(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_ci",
                 run_ci_benchmark=lambda **kw: calls.setdefault("kw", kw) or _result(),
                 CI_DOMAINS=["ecommerce"])
    _run(cmd_benchmark, _bench_args(test="cache-invalidation"))
    assert "kw" in calls
    _run(cmd_benchmark, _bench_args(test="ci"))
    assert "kw" in calls


# ── cmd_benchmark: sra / routing ─────────────────────────────────────────

def test_cmd_benchmark_sra_calls_summary_when_not_quiet(monkeypatch):
    calls = {}
    def run_sra_benchmark(**kw):
        calls["kw"] = kw
        return _result()
    _fake_module(monkeypatch, "contradish.bench.evaluate_sra",
                 run_sra_benchmark=run_sra_benchmark,
                 print_summary=lambda r: calls.setdefault("summarized", True))
    _run(cmd_benchmark, _bench_args(test="routing", quiet=False))
    assert "summarized" in calls
    assert calls["kw"]["domains"] is None  # sra passes domain=None straight through unless set


def test_cmd_benchmark_sra_quiet_skips_summary(monkeypatch):
    calls = {}
    _fake_module(monkeypatch, "contradish.bench.evaluate_sra",
                 run_sra_benchmark=lambda **kw: _result(),
                 print_summary=lambda r: calls.setdefault("summarized", True))
    _run(cmd_benchmark, _bench_args(test="sra", quiet=True))
    assert "summarized" not in calls


# ── cmd_benchmark: all ────────────────────────────────────────────────────

def test_cmd_benchmark_all_runs_each_suite_and_survives_failures(monkeypatch, capsys):
    _fake_module(monkeypatch, "contradish.bench.evaluate",
                 run_benchmark=lambda **kw: _result(), print_summary=lambda r: None,
                 save_result=lambda r: "out.json")
    _fake_module(monkeypatch, "contradish.bench.evaluate_jailbreaks",
                 run_jrr_benchmark=lambda **kw: _result())

    def boom(**kw):
        raise RuntimeError("population suite exploded")
    _fake_module(monkeypatch, "contradish.bench.evaluate_pc",
                 run_pc_benchmark=boom, PC_DOMAINS=["ecommerce"])
    _fake_module(monkeypatch, "contradish.bench.evaluate_cl",
                 run_cl_benchmark=lambda **kw: _result(), CL_DOMAINS=["e"], CL_LANGUAGES=["en"])
    _fake_module(monkeypatch, "contradish.bench.evaluate_mt",
                 run_mt_benchmark=lambda **kw: _result(), MT_DOMAINS=["e"])
    _fake_module(monkeypatch, "contradish.bench.evaluate_sra",
                 run_sra_benchmark=lambda **kw: _result(), print_summary=lambda r: None)

    code = _run(cmd_benchmark, _bench_args(test="all"))
    assert code is None
    out = capsys.readouterr().out
    assert "running all test suites" in out
    assert "population failed: population suite exploded" in out


# ── cmd_benchmark: unknown suite ─────────────────────────────────────────

def test_cmd_benchmark_unknown_test_suite_exits_one(capsys):
    code = _run(cmd_benchmark, _bench_args(test="not-a-real-suite"))
    assert code == 1
    assert "Unknown test suite" in capsys.readouterr().out


# ── cmd_benchmark: report / json output ──────────────────────────────────

def test_cmd_benchmark_writes_html_report(monkeypatch, tmp_path, capsys):
    _fake_module(monkeypatch, "contradish.bench.evaluate",
                 run_benchmark=lambda **kw: _result(), print_summary=lambda r: None,
                 save_result=lambda r: "out.json")
    report_path = str(tmp_path / "report.html")
    _run(cmd_benchmark, _bench_args(report=report_path))
    assert "HTML report saved" in capsys.readouterr().out
    assert (tmp_path / "report.html").exists()


def test_cmd_benchmark_report_generation_failure_is_caught(monkeypatch, tmp_path, capsys):
    _fake_module(monkeypatch, "contradish.bench.evaluate",
                 run_benchmark=lambda **kw: _result(), print_summary=lambda r: None,
                 save_result=lambda r: "out.json")
    monkeypatch.setattr(cli, "_generate_benchmark_report",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    _run(cmd_benchmark, _bench_args(report=str(tmp_path / "r.html")))
    assert "report generation skipped: boom" in capsys.readouterr().out


def test_cmd_benchmark_writes_json_output_creating_nested_dirs(monkeypatch, tmp_path, capsys):
    _fake_module(monkeypatch, "contradish.bench.evaluate",
                 run_benchmark=lambda **kw: _result(), print_summary=lambda r: None,
                 save_result=lambda r: "out.json")
    out_path = str(tmp_path / "nested" / "dir" / "out.json")
    _run(cmd_benchmark, _bench_args(output_json=out_path))
    assert "JSON written" in capsys.readouterr().out
    with open(out_path) as f:
        assert json.load(f)["avg_cai_strain"] == 0.12


# ── _generate_benchmark_report ────────────────────────────────────────────

def test_generate_benchmark_report_basic_shape(tmp_path):
    result = _result(avg_cai_strain=0.42, judge_provider="openai", judge_model="gpt-4o",
                      elapsed_seconds=12.5,
                      results={"ecommerce": {"cai_strain": 0.1, "severity_weighted_cts": 0.2,
                                              "failed": 1, "total": 10}})
    path = str(tmp_path / "r.html")
    _generate_benchmark_report(result, path, "claude-sonnet-4-6", "v2")
    html = open(path).read()
    assert "claude-sonnet-4-6" in html
    assert "0.4200" in html
    assert "ecommerce" in html
    assert "1/10" in html


def test_generate_benchmark_report_marks_independent_judging(tmp_path):
    result = _result(independent_judging=True)
    path = str(tmp_path / "r.html")
    _generate_benchmark_report(result, path, "m", "v2")
    assert "independent" in open(path).read()


def test_generate_benchmark_report_domain_error_row(tmp_path):
    result = _result(results={"hr": {"error": "boom"}})
    path = str(tmp_path / "r.html")
    _generate_benchmark_report(result, path, "m", "v2")
    html = open(path).read()
    assert "ERROR" in html
    assert "hr" in html


def test_generate_benchmark_report_handles_missing_score_as_na(tmp_path):
    result = {"results": {}}  # no avg_* keys at all -> cts is None
    path = str(tmp_path / "r.html")
    _generate_benchmark_report(result, path, "m", "jailbreaks")
    assert "n/a" in open(path).read()


def test_generate_benchmark_report_color_thresholds(tmp_path):
    for strain, expect_color in [(0.10, "#16a34a"), (0.35, "#d97706"), (0.75, "#dc2626")]:
        result = _result(avg_cai_strain=strain)
        path = str(tmp_path / f"r_{strain}.html")
        _generate_benchmark_report(result, path, "m", "v2")
        assert expect_color in open(path).read()


def test_generate_benchmark_report_score_label_by_test_type(tmp_path):
    for test_type, label in [("v2", "Strain"), ("jailbreaks", "JRR"), ("population", "PC-Strain"),
                              ("multilang", "CL-Strain"), ("sra", "SRA"), ("unknown-type", "Score")]:
        result = _result()
        path = str(tmp_path / f"r_{test_type}.html")
        _generate_benchmark_report(result, path, "m", test_type)
        assert f">{label}<" in open(path).read()


# ── main(): dispatch chain ────────────────────────────────────────────────

_ALL_CMD_NAMES = [
    "cmd_benchmark", "cmd_monitor", "cmd_ledger", "cmd_diagnose", "cmd_init",
    "cmd_run", "cmd_compare", "cmd_improve", "cmd_findings", "cmd_prompt",
    "cmd_replay", "cmd_reconcile", "cmd_judge_floor", "cmd_fairness",
    "cmd_quick", "cmd_calibrate", "cmd_policy", "cmd_from_prompt",
]


def _dispatch(monkeypatch, argv):
    calls = {}
    for name in _ALL_CMD_NAMES:
        def make(n):
            def recorder(args):
                calls["name"] = n
                calls["args"] = args
            return recorder
        monkeypatch.setattr(cli, name, make(name))
    monkeypatch.setattr(sys, "argv", ["contradish"] + argv)
    cli.main()
    return calls


@pytest.mark.parametrize("argv,expected_cmd", [
    (["benchmark", "--model", "claude-sonnet-4-6"], "cmd_benchmark"),
    (["monitor", "--input", "logs.jsonl"], "cmd_monitor"),
    (["ledger"], "cmd_ledger"),
    (["diagnose", "--input", "result.json"], "cmd_diagnose"),
    (["init"], "cmd_init"),
    (["run", "evals.yaml", "--app", "mod:fn"], "cmd_run"),
    (["compare"], "cmd_compare"),
    (["improve"], "cmd_improve"),
    (["findings", "result.json"], "cmd_findings"),
    (["prompt"], "cmd_prompt"),
    (["replay", "transcript.jsonl"], "cmd_replay"),
    (["reconcile", "report.json", "replay.json"], "cmd_reconcile"),
    (["judge-floor"], "cmd_judge_floor"),
    (["fairness"], "cmd_fairness"),
    (["analyze"], "cmd_quick"),
    (["calibrate", "--model", "claude-sonnet-4-6"], "cmd_calibrate"),
])
def test_main_dispatches_each_subcommand(monkeypatch, argv, expected_cmd):
    calls = _dispatch(monkeypatch, argv)
    assert calls["name"] == expected_cmd


def test_main_dispatches_policy_only_to_cmd_policy(monkeypatch):
    calls = _dispatch(monkeypatch, ["--policy", "ecommerce"])
    assert calls["name"] == "cmd_policy"
    assert calls["args"].policy == "ecommerce"


def test_main_bare_system_prompt_argument_is_currently_broken(monkeypatch):
    # REAL BUG, found while writing this test, NOT fixed here (flagged to
    # the user -- the fix requires restructuring main()'s single-parser
    # design, out of scope for a test-coverage pass): the README's very
    # first Quickstart line --
    #     contradish "You are a support agent. Refunds within 30 days only."
    # -- currently crashes. parser.add_subparsers(dest="command") is added
    # to the top-level parser *alongside* the `system_prompt` positional
    # (nargs="?"), meant to catch exactly this bare-string invocation. But
    # argparse's positional-matching always lets the subparsers action
    # (nargs=PARSER, effectively "one-or-more, greedy") claim a lone
    # unrecognized token first, regardless of declaration order (verified:
    # swapping which is added first does not change this) -- so instead of
    # falling through to `system_prompt`, argparse hard-errors with
    # "invalid choice" and exits 2. The other two documented ways to pass a
    # prompt -- `contradish --policy X` and `contradish --prompt file.txt`
    # -- are unaffected (they're named options, not competing positionals)
    # and both dispatch correctly; only the bare freeform-string form is
    # broken. This test pins the actual current behavior.
    for name in _ALL_CMD_NAMES:
        monkeypatch.setattr(cli, name, lambda args: pytest.fail("should not dispatch"))
    monkeypatch.setattr(sys, "argv", ["contradish", "You are a helpful assistant."])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2


def test_main_dispatches_prompt_file_to_cmd_from_prompt(monkeypatch, tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("be nice")
    calls = _dispatch(monkeypatch, ["--prompt", str(f)])
    assert calls["name"] == "cmd_from_prompt"


def test_main_bare_no_args_with_api_key_runs_smoke_test(monkeypatch, capsys):
    calls = _dispatch(monkeypatch, [])
    assert calls["name"] == "cmd_policy"
    assert calls["args"].policy == "ecommerce"
    assert calls["args"].app is None
    assert calls["args"].threshold is None
    assert "smoke test" in capsys.readouterr().out


def test_main_bare_no_args_without_api_key_prints_help_and_exits_zero(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in _ALL_CMD_NAMES:
        monkeypatch.setattr(cli, name, lambda args: pytest.fail("should not dispatch"))
    monkeypatch.setattr(sys, "argv", ["contradish"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "No API key found" in out


# Not covered (documented, not papered over): the trailing
# `if __name__ == "__main__": main()` two-line block. runpy.run_module
# re-executes cli.py fresh in a new "__main__" namespace rather than
# reusing the already-imported contradish.cli module object, so
# monkeypatch.setattr(cli, "cmd_X", ...) wouldn't apply to that fresh
# execution -- it would run the *real* cmd_init (or whichever command),
# including cmd_init's real side effect of writing .contradish.yaml to
# the working directory. Not worth the risk for two lines of standard
# CLI boilerplate.
