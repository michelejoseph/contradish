"""
Tests for the remaining contradish.cli entry points not covered by
tests/test_cli.py: cmd_init, cmd_policy, cmd_quick, cmd_from_prompt,
cmd_run, cmd_calibrate, cmd_diagnose, cmd_monitor.

No API key or live model call required. Suite/analyze/judge orchestration
is faked the same way tests/test_cli.py fakes it: patch the underlying
attribute on the module it's function-locally imported from (or on
`contradish` itself, for `from contradish import X`), then drive the
handler directly with a constructed argparse.Namespace and catch
SystemExit for the exit code.
"""
import argparse
import importlib
import json

import pytest

import contradish
import contradish.cli as cli
from contradish.cli import cmd_init, cmd_policy, cmd_quick, cmd_from_prompt, cmd_run, cmd_calibrate, cmd_diagnose, cmd_monitor
from contradish.models import TestCase, TestResult, Report, RiskLevel, ContradictionPair
from contradish.calibration import CalibrationResult


def _run(fn, args):
    """Call a cmd_* handler and capture its sys.exit code (None if it didn't exit)."""
    code = None
    try:
        fn(args)
    except SystemExit as e:
        code = e.code
    return code


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


def _bench_result(name, canonical, consistency, contradiction=0.0):
    tc = TestCase(input=name, name=name, canonical_answer=canonical)
    contradictions = []
    if contradiction > 0:
        contradictions = [ContradictionPair(output_a="a", output_b="b",
                                            input_a=name, input_b=name,
                                            explanation="conflict", severity="factual")]
    return TestResult(test_case=tc, paraphrases=[], outputs=[],
                      consistency_score=consistency, contradiction_score=contradiction,
                      risk=RiskLevel.HIGH if contradiction else RiskLevel.LOW,
                      contradictions=contradictions)


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


# ─────────────────────────────────────────────────────────────────────────
# cmd_init
# ─────────────────────────────────────────────────────────────────────────

def test_cmd_init_refuses_overwrite_without_force(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".contradish.yaml").write_text("existing: true\n")
    cmd_init(_ns(force=False))
    out = capsys.readouterr().out
    assert "already exists" in out
    assert (tmp_path / ".contradish.yaml").read_text() == "existing: true\n"


def test_cmd_init_writes_config_from_answers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter([
        "ecommerce",  # policy
        "",           # app (demo mode)
        "0.15",       # threshold
        "8",          # paraphrases
        "n",          # decline GHA workflow copy
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cmd_init(_ns(force=False))
    content = (tmp_path / ".contradish.yaml").read_text()
    assert "policy: ecommerce" in content
    assert "threshold: 0.15" in content
    assert "paraphrases: 8" in content
    assert not (tmp_path / ".github" / "workflows" / "cai.yml").exists()


def test_cmd_init_invalid_policy_is_dropped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter(["not-a-real-policy", "", "", "", "n"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cmd_init(_ns(force=False))
    content = (tmp_path / ".contradish.yaml").read_text()
    assert "policy:" not in content


def test_cmd_init_blank_threshold_and_paraphrases_use_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter(["", "", "not-a-number", "also-not-a-number", "n"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cmd_init(_ns(force=False))
    content = (tmp_path / ".contradish.yaml").read_text()
    assert "threshold: 0.2" in content
    assert "paraphrases: 5" in content


def test_cmd_init_accepts_gha_workflow_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter(["", "", "", "", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cmd_init(_ns(force=False))
    assert (tmp_path / ".github" / "workflows" / "cai.yml").exists()


# ─────────────────────────────────────────────────────────────────────────
# Fake Suite (shared by cmd_policy / cmd_quick / cmd_from_prompt / cmd_run)
# ─────────────────────────────────────────────────────────────────────────

class FakeSuite:
    """
    Stand-in for contradish.Suite. Captures the kwargs each entry point
    passed so tests can assert on wiring, and returns a fixed pre-built
    Report from .run() so no real judge/model call ever happens.
    """
    report = None
    last_from_policy_kwargs = None
    last_from_prompt_kwargs = None
    last_init_kwargs = None
    last_run_kwargs = None

    def __init__(self, app=None, **kwargs):
        self.app = app
        self.cases = []
        FakeSuite.last_init_kwargs = {"app": app, **kwargs}

    def add(self, tc):
        self.cases.append(tc)

    def run(self, **kwargs):
        FakeSuite.last_run_kwargs = kwargs
        return FakeSuite.report

    @classmethod
    def from_policy(cls, **kwargs):
        cls.last_from_policy_kwargs = kwargs
        return cls(app=kwargs.get("app"))

    @classmethod
    def from_prompt(cls, **kwargs):
        cls.last_from_prompt_kwargs = kwargs
        return cls(app=kwargs.get("app"))


@pytest.fixture(autouse=True)
def _fake_suite(monkeypatch):
    FakeSuite.report = Report(results=[_bench_result("c1", "answer", 0.95)])
    FakeSuite.last_from_policy_kwargs = None
    FakeSuite.last_from_prompt_kwargs = None
    FakeSuite.last_init_kwargs = None
    FakeSuite.last_run_kwargs = None
    monkeypatch.setattr(contradish, "Suite", FakeSuite)
    yield


# ─────────────────────────────────────────────────────────────────────────
# cmd_policy
# ─────────────────────────────────────────────────────────────────────────

def test_cmd_policy_invalid_policy_name_exits_1(capsys):
    code = _run(cmd_policy, _ns(policy="not-a-real-policy", app=None, json=False,
                                 paraphrases=5, format=None, report=None, threshold=None))
    assert code == 1
    assert "Available:" in capsys.readouterr().out


def test_cmd_policy_happy_path_with_app(monkeypatch):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    code = _run(cmd_policy, _ns(policy="ecommerce", app="mymod:fn", json=False,
                                 paraphrases=5, format=None, report=None, threshold=None))
    assert code == 0  # FakeSuite.report has no failures
    assert FakeSuite.last_from_policy_kwargs["policy"] == "ecommerce"


def test_cmd_policy_json_format_prints_valid_json(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    _run(cmd_policy, _ns(policy="ecommerce", app="mymod:fn", json=False,
                          paraphrases=5, format="json", report=None, threshold=None))
    out = capsys.readouterr().out
    # format="json" (unlike the top-level --json flag) still prints the human
    # header first; the JSON payload is the trailing `{...}` blob.
    data = json.loads(out[out.index("{"):])
    assert "cai_score" in data


def test_cmd_policy_sarif_format_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    out_path = str(tmp_path / "out.sarif")
    _run(cmd_policy, _ns(policy="ecommerce", app="mymod:fn", json=False,
                          paraphrases=5, format="sarif", output=out_path,
                          report=None, threshold=None))
    assert (tmp_path / "out.sarif").exists()


def test_cmd_policy_threshold_exceeded_fails(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    FakeSuite.report = Report(results=[_bench_result("c1", "answer", 0.1, contradiction=0.9)])
    code = _run(cmd_policy, _ns(policy="ecommerce", app="mymod:fn", json=False,
                                 paraphrases=5, format=None, report=None, threshold=0.01))
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_cmd_policy_saves_report_when_path_given(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    calls = []
    monkeypatch.setattr(cli, "_save_report", lambda report, path, policy_name=None: calls.append((path, policy_name)))
    report_path = str(tmp_path / "report.html")
    _run(cmd_policy, _ns(policy="ecommerce", app="mymod:fn", json=False,
                          paraphrases=5, format=None, report=report_path, threshold=None))
    assert calls == [(report_path, "ecommerce")]


def test_cmd_policy_demo_mode_when_no_app(monkeypatch):
    # No --app: falls into _make_demo_app, which constructs a real LLMClient()
    # (safe: no network call at construction) but never actually calls it,
    # since FakeSuite.run() never invokes the app.
    code = _run(cmd_policy, _ns(policy="ecommerce", app=None, json=False,
                                 paraphrases=5, format=None, report=None, threshold=None))
    assert code == 0
    assert FakeSuite.last_from_policy_kwargs["app"] is not None


# ─────────────────────────────────────────────────────────────────────────
# cmd_quick
# ─────────────────────────────────────────────────────────────────────────

class FakeQuickResult:
    def __init__(self, overall_strain=0.1):
        self.overall_strain = overall_strain
        self.html_written = None
        self.sft_calls = 0
        self.dpo_calls = 0

    def __str__(self):
        return "QUICK-RESULT-SUMMARY"

    def to_html(self, path):
        with open(path, "w") as f:
            f.write("<html></html>")

    def to_jsonl(self):
        return '{"a": 1}\n{"b": 2}\n'

    def to_dpo_jsonl(self):
        return '{"chosen": 1, "rejected": 0}\n'


def test_cmd_quick_unknown_domain_without_questions_exits_1(capsys):
    code = _run(cmd_quick, _ns(domain="not-a-real-domain", app=None, questions=None,
                                html=None, sft=None, dpo=None, full_framings=False,
                                n_repairs=30, threshold=None))
    assert code == 1
    assert "Unknown domain" in capsys.readouterr().out


def test_cmd_quick_happy_path_with_app(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    monkeypatch.setattr(contradish, "analyze", lambda **kwargs: FakeQuickResult())
    code = _run(cmd_quick, _ns(domain="customer-service", app="mymod:fn", questions=None,
                                html=None, sft=None, dpo=None, full_framings=False,
                                n_repairs=30, threshold=None))
    assert code is None
    assert "QUICK-RESULT-SUMMARY" in capsys.readouterr().out


def test_cmd_quick_custom_questions_bypass_unknown_domain_check(monkeypatch):
    captured = {}

    def fake_analyze(**kwargs):
        captured.update(kwargs)
        return FakeQuickResult()

    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    monkeypatch.setattr(contradish, "analyze", fake_analyze)
    code = _run(cmd_quick, _ns(domain="not-a-real-domain", app="mymod:fn",
                                questions=["custom Q1"], html=None, sft=None, dpo=None,
                                full_framings=False, n_repairs=30, threshold=None))
    assert code is None
    assert captured["domain"] is None  # not in list_domains() -> None
    assert captured["questions"] == ["custom Q1"]


def test_cmd_quick_bad_app_string_exits_1(capsys):
    code = _run(cmd_quick, _ns(domain="customer-service", app="not:a:real:path", questions=None,
                                html=None, sft=None, dpo=None, full_framings=False,
                                n_repairs=30, threshold=None))
    assert code == 1
    assert "Could not load" in capsys.readouterr().out


def test_cmd_quick_writes_html_sft_dpo_outputs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    monkeypatch.setattr(contradish, "analyze", lambda **kwargs: FakeQuickResult())
    html_path = str(tmp_path / "r.html")
    sft_path  = str(tmp_path / "r.sft.jsonl")
    dpo_path  = str(tmp_path / "r.dpo.jsonl")
    _run(cmd_quick, _ns(domain="customer-service", app="mymod:fn", questions=None,
                         html=html_path, sft=sft_path, dpo=dpo_path, full_framings=False,
                         n_repairs=30, threshold=None))
    assert (tmp_path / "r.html").exists()
    assert (tmp_path / "r.sft.jsonl").read_text() == '{"a": 1}\n{"b": 2}\n'
    assert (tmp_path / "r.dpo.jsonl").exists()
    out = capsys.readouterr().out
    assert "HTML report" in out
    assert "SFT JSONL" in out
    assert "DPO JSONL" in out


def test_cmd_quick_threshold_exceeded_exits_1(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    monkeypatch.setattr(contradish, "analyze", lambda **kwargs: FakeQuickResult(overall_strain=0.9))
    code = _run(cmd_quick, _ns(domain="customer-service", app="mymod:fn", questions=None,
                                html=None, sft=None, dpo=None, full_framings=False,
                                n_repairs=30, threshold=0.2))
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────
# cmd_from_prompt
# ─────────────────────────────────────────────────────────────────────────

def test_cmd_from_prompt_missing_prompt_exits_1(capsys):
    code = _run(cmd_from_prompt, _ns(prompt_file=None, system_prompt="   ", app=None,
                                      json=False, paraphrases=5, format=None,
                                      report=None, judge_votes=1))
    assert code == 1
    assert "No system prompt" in capsys.readouterr().out


def test_cmd_from_prompt_reads_prompt_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    prompt_path = tmp_path / "sp.txt"
    prompt_path.write_text("You are a refund bot.")
    code = _run(cmd_from_prompt, _ns(prompt_file=str(prompt_path), system_prompt=None,
                                      app="mymod:fn", json=False, paraphrases=5, format=None,
                                      report=None, judge_votes=1))
    assert code == 0
    assert FakeSuite.last_from_prompt_kwargs["system_prompt"] == "You are a refund bot."


def test_cmd_from_prompt_demo_mode(monkeypatch):
    code = _run(cmd_from_prompt, _ns(prompt_file=None, system_prompt="Be helpful.", app=None,
                                      json=False, paraphrases=5, format=None,
                                      report=None, judge_votes=1))
    assert code == 0
    assert FakeSuite.last_from_prompt_kwargs["app"] is not None


def test_cmd_from_prompt_json_format(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    _run(cmd_from_prompt, _ns(prompt_file=None, system_prompt="Be helpful.", app="mymod:fn",
                               json=True, paraphrases=5, format=None, report=None, judge_votes=1))
    data = json.loads(capsys.readouterr().out)
    assert "cai_score" in data


def test_cmd_from_prompt_saves_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    calls = []
    monkeypatch.setattr(cli, "_save_report", lambda report, path, policy_name=None: calls.append(path))
    report_path = str(tmp_path / "r.html")
    _run(cmd_from_prompt, _ns(prompt_file=None, system_prompt="Be helpful.", app="mymod:fn",
                               json=False, paraphrases=5, format=None, report=report_path,
                               judge_votes=1))
    assert calls == [report_path]


# ─────────────────────────────────────────────────────────────────────────
# cmd_run
# ─────────────────────────────────────────────────────────────────────────

def test_cmd_run_loads_cases_and_runs_suite(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([{"input": "hello", "name": "n1"}, {"input": "world"}]))
    code = _run(cmd_run, _ns(app="mymod:fn", eval_file=str(cases_path), json=False,
                              paraphrases=5, report=None))
    assert code == 0
    assert len(FakeSuite.last_init_kwargs) >= 1
    # both cases got added to the suite before .run()
    assert FakeSuite.last_run_kwargs is not None


def test_cmd_run_json_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([{"input": "hello"}]))
    _run(cmd_run, _ns(app="mymod:fn", eval_file=str(cases_path), json=True,
                       paraphrases=5, report=None))
    data = json.loads(capsys.readouterr().out)
    assert "cai_score" in data


def test_cmd_run_exits_1_when_report_has_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    FakeSuite.report = Report(results=[_bench_result("c1", "answer", 0.1, contradiction=0.9)])
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([{"input": "hello"}]))
    code = _run(cmd_run, _ns(app="mymod:fn", eval_file=str(cases_path), json=False,
                              paraphrases=5, report=None))
    assert code == 1


def test_cmd_run_saves_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))
    calls = []
    monkeypatch.setattr(cli, "_save_report", lambda report, path, policy_name=None: calls.append(path))
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([{"input": "hello"}]))
    report_path = str(tmp_path / "r.html")
    _run(cmd_run, _ns(app="mymod:fn", eval_file=str(cases_path), json=False,
                       paraphrases=5, report=report_path))
    assert calls == [report_path]


# ─────────────────────────────────────────────────────────────────────────
# cmd_calibrate
# ─────────────────────────────────────────────────────────────────────────

def test_cmd_calibrate_computable_saves_result(monkeypatch, tmp_path):
    result = CalibrationResult(model="m1", computable=True, reason=None,
                                stability=0.8, responsiveness=0.7, score=0.75, verdict="calibrated")
    monkeypatch.setattr("contradish.calibration.compute_calibration_score", lambda model, results_dir: result)
    code = _run(cmd_calibrate, _ns(model="m1", results_dir=str(tmp_path), json=False, threshold=None))
    assert code == 0
    saved = list(tmp_path.glob("calibration_m1_*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["calibration_score"] == 0.75


def test_cmd_calibrate_json_output(monkeypatch, tmp_path, capsys):
    result = CalibrationResult(model="m1", computable=True, reason=None,
                                stability=0.8, responsiveness=0.7, score=0.75, verdict="calibrated")
    monkeypatch.setattr("contradish.calibration.compute_calibration_score", lambda model, results_dir: result)
    _run(cmd_calibrate, _ns(model="m1", results_dir=str(tmp_path), json=True, threshold=None))
    out = capsys.readouterr().out.split("result saved")[0]  # isolate the JSON blob
    data = json.loads(out)
    assert data["calibration_score"] == 0.75


def test_cmd_calibrate_not_computable_with_threshold_fails(monkeypatch, tmp_path, capsys):
    result = CalibrationResult(model="m1", computable=False, reason="no data")
    monkeypatch.setattr("contradish.calibration.compute_calibration_score", lambda model, results_dir: result)
    code = _run(cmd_calibrate, _ns(model="m1", results_dir=str(tmp_path), json=False, threshold=0.5))
    assert code == 1
    assert "not computable" in capsys.readouterr().out


def test_cmd_calibrate_below_threshold_fails(monkeypatch, tmp_path, capsys):
    result = CalibrationResult(model="m1", computable=True, reason=None,
                                stability=0.5, responsiveness=0.5, score=0.5, verdict="mediocre")
    monkeypatch.setattr("contradish.calibration.compute_calibration_score", lambda model, results_dir: result)
    code = _run(cmd_calibrate, _ns(model="m1", results_dir=str(tmp_path), json=False, threshold=0.9))
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_cmd_calibrate_above_threshold_passes(monkeypatch, tmp_path):
    result = CalibrationResult(model="m1", computable=True, reason=None,
                                stability=0.9, responsiveness=0.9, score=0.9, verdict="excellent")
    monkeypatch.setattr("contradish.calibration.compute_calibration_score", lambda model, results_dir: result)
    code = _run(cmd_calibrate, _ns(model="m1", results_dir=str(tmp_path), json=False, threshold=0.5))
    assert code == 0


# ─────────────────────────────────────────────────────────────────────────
# cmd_diagnose
# ─────────────────────────────────────────────────────────────────────────

_DIAGNOSE_REPORT = {
    "result_path": "r.json",
    "source_type": "sra",
    "drift_count": 1,
    "diagnoses": [{"failure_mode": "AUTHORITY_CAPITULATION"}],
    "aggregate": {"priority_cases": []},
}


def test_cmd_diagnose_missing_file_exits_1(tmp_path, capsys):
    code = _run(cmd_diagnose, _ns(input=str(tmp_path / "nope.json"), judge_provider=None,
                                   judge_model=None, quiet=False, max_cases=None,
                                   output_dir="repair", json=False))
    assert code == 1
    assert "File not found" in capsys.readouterr().out


def test_cmd_diagnose_json_output_returns_without_exit(monkeypatch, tmp_path, capsys):
    result_path = tmp_path / "r.json"
    result_path.write_text(json.dumps({"provider": "anthropic"}))
    monkeypatch.setattr("contradish.diagnose.analyze_result", lambda **kwargs: _DIAGNOSE_REPORT)
    code = _run(cmd_diagnose, _ns(input=str(result_path), judge_provider=None, judge_model=None,
                                   quiet=False, max_cases=None, output_dir="repair", json=True))
    assert code is None
    data = json.loads(capsys.readouterr().out)
    assert data["drift_count"] == 1


def test_cmd_diagnose_infers_openai_judge_when_result_is_anthropic(monkeypatch, tmp_path):
    result_path = tmp_path / "r.json"
    result_path.write_text(json.dumps({"provider": "anthropic"}))
    captured = {}

    def fake_analyze_result(**kwargs):
        captured.update(kwargs)
        return _DIAGNOSE_REPORT

    monkeypatch.setattr("contradish.diagnose.analyze_result", fake_analyze_result)
    _run(cmd_diagnose, _ns(input=str(result_path), judge_provider=None, judge_model=None,
                            quiet=True, max_cases=None, output_dir="repair", json=True))
    assert captured["judge_provider"] == "openai"


def test_cmd_diagnose_saves_report_via_evaluate_repair(monkeypatch, tmp_path):
    result_path = tmp_path / "r.json"
    result_path.write_text(json.dumps({"provider": "anthropic"}))
    monkeypatch.setattr("contradish.diagnose.analyze_result", lambda **kwargs: _DIAGNOSE_REPORT)
    save_calls = []
    monkeypatch.setattr(
        "contradish.bench.evaluate_repair.save_report",
        lambda report, output_dir: (save_calls.append((report, output_dir)) or ("full.json", "ft.jsonl", "prompt.txt")),
    )
    monkeypatch.setattr("contradish.bench.evaluate_repair.print_summary", lambda *a, **k: None)
    code = _run(cmd_diagnose, _ns(input=str(result_path), judge_provider="openai", judge_model="gpt-4o",
                                   quiet=False, max_cases=None, output_dir=str(tmp_path / "out"), json=False))
    assert code is None
    assert save_calls == [(_DIAGNOSE_REPORT, str(tmp_path / "out"))]


def test_cmd_diagnose_critical_priority_cases_exit_1(monkeypatch, tmp_path):
    result_path = tmp_path / "r.json"
    result_path.write_text(json.dumps({"provider": "anthropic"}))
    report_with_critical = dict(_DIAGNOSE_REPORT, aggregate={"priority_cases": [{"severity": "critical"}]})
    monkeypatch.setattr("contradish.diagnose.analyze_result", lambda **kwargs: report_with_critical)
    monkeypatch.setattr("contradish.bench.evaluate_repair.save_report",
                         lambda report, output_dir: ("full.json", "ft.jsonl", "prompt.txt"))
    monkeypatch.setattr("contradish.bench.evaluate_repair.print_summary", lambda *a, **k: None)
    code = _run(cmd_diagnose, _ns(input=str(result_path), judge_provider="openai", judge_model="gpt-4o",
                                   quiet=True, max_cases=None, output_dir="repair", json=False))
    assert code == 1


# ─────────────────────────────────────────────────────────────────────────
# cmd_monitor
# ─────────────────────────────────────────────────────────────────────────

def test_cmd_monitor_missing_log_exits_1(monkeypatch, capsys):
    def fake_load_log(*a, **k):
        raise FileNotFoundError("no such log")
    monkeypatch.setattr("contradish.monitor.load_log", fake_load_log)
    code = _run(cmd_monitor, _ns(input="nope.jsonl", judge_provider=None, judge_model=None,
                                  quiet=False, json=False, max=200, format="auto",
                                  min_cluster_size=3, batch_size=25, no_ledger=True,
                                  ledger=None, output=None, threshold=0.30))
    assert code == 1
    assert "no such log" in capsys.readouterr().out


def test_cmd_monitor_no_clusters_exits_0(monkeypatch, capsys):
    monkeypatch.setattr("contradish.monitor.load_log", lambda *a, **k: ["conv1", "conv2"])
    monkeypatch.setattr("contradish.monitor.find_clusters", lambda *a, **k: [])
    code = _run(cmd_monitor, _ns(input="log.jsonl", judge_provider=None, judge_model=None,
                                  quiet=False, json=False, max=200, format="auto",
                                  min_cluster_size=3, batch_size=25, no_ledger=True,
                                  ledger=None, output=None, threshold=0.30))
    assert code == 0
    assert "No clusters found" in capsys.readouterr().out


def test_cmd_monitor_json_output_returns_without_saving_file(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("contradish.monitor.load_log", lambda *a, **k: ["conv1"])
    monkeypatch.setattr("contradish.monitor.find_clusters", lambda *a, **k: [{"topic": "t"}])
    monkeypatch.setattr("contradish.monitor.score_clusters", lambda *a, **k: [{"topic": "t", "cts": 0.9}])
    monkeypatch.setattr("contradish.monitor.analyze_monitor",
                         lambda scored, total_conversations: {"drift_rate": 0.1, "hotspots": [], "clean_clusters": []})
    code = _run(cmd_monitor, _ns(input="log.jsonl", judge_provider="anthropic", judge_model="claude-opus-4-6",
                                  quiet=False, json=True, max=200, format="auto",
                                  min_cluster_size=3, batch_size=25, no_ledger=True,
                                  ledger=None, output=None, threshold=0.30))
    assert code is None
    data = json.loads(capsys.readouterr().out)
    assert data["drift_rate"] == 0.1
    assert not (tmp_path / "results").exists()  # json path returns before saving to disk


def test_cmd_monitor_saves_report_and_passes_below_threshold(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("contradish.monitor.load_log", lambda *a, **k: ["conv1"])
    monkeypatch.setattr("contradish.monitor.find_clusters", lambda *a, **k: [{"topic": "t"}])
    monkeypatch.setattr("contradish.monitor.score_clusters", lambda *a, **k: [{"topic": "t", "cts": 0.9}])
    monkeypatch.setattr("contradish.monitor.analyze_monitor",
                         lambda scored, total_conversations: {"drift_rate": 0.1, "hotspots": [], "clean_clusters": []})
    monkeypatch.setattr("contradish.monitor.print_monitor_summary", lambda *a, **k: None)
    out_path = str(tmp_path / "out.json")
    code = _run(cmd_monitor, _ns(input="log.jsonl", judge_provider="anthropic", judge_model="claude-opus-4-6",
                                  quiet=False, json=False, max=200, format="auto",
                                  min_cluster_size=3, batch_size=25, no_ledger=True,
                                  ledger=None, output=out_path, threshold=0.30))
    assert code is None
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved["drift_rate"] == 0.1


def test_cmd_monitor_drift_rate_above_threshold_exits_1(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("contradish.monitor.load_log", lambda *a, **k: ["conv1"])
    monkeypatch.setattr("contradish.monitor.find_clusters", lambda *a, **k: [{"topic": "t"}])
    monkeypatch.setattr("contradish.monitor.score_clusters", lambda *a, **k: [{"topic": "t", "cts": 0.2}])
    monkeypatch.setattr("contradish.monitor.analyze_monitor",
                         lambda scored, total_conversations: {"drift_rate": 0.9, "hotspots": [], "clean_clusters": []})
    monkeypatch.setattr("contradish.monitor.print_monitor_summary", lambda *a, **k: None)
    out_path = str(tmp_path / "out.json")
    code = _run(cmd_monitor, _ns(input="log.jsonl", judge_provider="anthropic", judge_model="claude-opus-4-6",
                                  quiet=False, json=False, max=200, format="auto",
                                  min_cluster_size=3, batch_size=25, no_ledger=True,
                                  ledger=None, output=out_path, threshold=0.30))
    assert code == 1
