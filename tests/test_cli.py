"""
Tests for contradish.cli — the command-line entry points.
Run with: pytest tests/test_cli.py

No API key or live model call required. Each cmd_* handler does a
function-local `from contradish.X import Y` (or, for _load_callable, calls
a plain module-level name), so tests patch the underlying attribute on the
source module / on contradish.cli and drive the handler directly with a
constructed argparse.Namespace, matching the existing pattern in
tests/test_cli_from_production.py. cmd_* handlers exit via sys.exit(); tests
catch SystemExit and assert on the code.
"""
import argparse
import json
import os
import sys
import tempfile

import pytest

import contradish.cli as cli
from contradish.cli import (
    _load_callable, _load_cases, _check_api_key, _output_json, _output_sarif,
    _save_report, cmd_findings, cmd_reconcile, cmd_compare, cmd_fairness,
    cmd_judge_floor, cmd_prompt, cmd_replay,
)
from contradish.models import TestCase, TestResult, Report, ContradictionPair, RiskLevel
from contradish import ReplayReport, ReplayContradiction
from contradish.fairness import FairnessAudit, CaseProfileResult
from contradish.judge_calibration import JudgeCalibration
from contradish.prompt_analyzer import PromptAnalysis, PromptTension


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
    # Satisfy _check_api_key() for handlers that call it, without touching
    # the real environment across tests.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


# ── _load_callable ───────────────────────────────────────────────────────────

def test_load_callable_rejects_missing_colon():
    with pytest.raises(ValueError, match="module:function"):
        _load_callable("not_a_valid_path")


def test_load_callable_imports_and_resolves(tmp_path, monkeypatch):
    mod_file = tmp_path / "my_temp_app.py"
    mod_file.write_text("def my_fn(x):\n    return x\n")
    monkeypatch.chdir(tmp_path)
    fn = _load_callable("my_temp_app:my_fn")
    assert fn("hi") == "hi"


def test_load_callable_raises_for_missing_module():
    with pytest.raises(ImportError):
        _load_callable("this_module_does_not_exist_xyz:fn")


# ── _load_cases ──────────────────────────────────────────────────────────────

def test_load_cases_from_json_list(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps([{"input": "hello", "name": "n1"}]))
    cases = _load_cases(str(p))
    assert len(cases) == 1 and cases[0].input == "hello" and cases[0].name == "n1"


def test_load_cases_from_json_wrapped_dict(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps({"test_cases": [{"input": "a"}, {"input": "b"}]}))
    cases = _load_cases(str(p))
    assert [c.input for c in cases] == ["a", "b"]


def test_load_cases_from_yaml(tmp_path):
    p = tmp_path / "cases.yaml"
    p.write_text("test_cases:\n  - input: hi there\n    name: greet\n")
    cases = _load_cases(str(p))
    assert len(cases) == 1 and cases[0].input == "hi there"


def test_load_cases_default_expected_traits_is_empty_list(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps([{"input": "x"}]))
    cases = _load_cases(str(p))
    assert cases[0].expected_traits == []


# ── _check_api_key ────────────────────────────────────────────────────────────

def test_check_api_key_passes_when_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _check_api_key()   # no exception, no SystemExit


def test_check_api_key_exits_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        _check_api_key()
    assert exc.value.code == 1


# ── _output_json / _output_sarif / _save_report ────────────────────────────────

def _bench_result(name, canonical, consistency, contradiction=0.0, output_a="a", output_b="b"):
    tc = TestCase(input=name, name=name, canonical_answer=canonical)
    contradictions = []
    if contradiction > 0:
        contradictions = [ContradictionPair(output_a=output_a, output_b=output_b,
                                            input_a=name, input_b=name,
                                            explanation="conflict", severity="factual")]
    return TestResult(test_case=tc, paraphrases=[], outputs=[],
                      consistency_score=consistency, contradiction_score=contradiction,
                      risk=RiskLevel.HIGH if contradiction else RiskLevel.LOW,
                      contradictions=contradictions)


def test_output_json_prints_report_dict(capsys):
    report = Report(results=[_bench_result("case1", "answer", 0.9)])
    _output_json(report)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "cai_score" in data


def test_output_sarif_writes_valid_sarif(tmp_path, capsys):
    report = Report(results=[
        _bench_result("refund window", "30 days", 0.2, contradiction=0.8,
                      output_a="yes", output_b="no"),
    ])
    out_path = str(tmp_path / "out.sarif")
    _output_sarif(report, out_path)
    with open(out_path) as f:
        sarif = json.load(f)
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"], "expected at least one SARIF result for the failed case"
    assert sarif["runs"][0]["results"][0]["ruleId"] == "CAI001"
    assert "SARIF written" in capsys.readouterr().out


def test_save_report_writes_html(tmp_path, monkeypatch, capsys):
    import contradish.reporter as reporter_mod
    monkeypatch.setattr(reporter_mod, "to_html", lambda report, policy_name=None: "<html>ok</html>")
    report = Report(results=[_bench_result("c", "a", 0.9)])
    out_path = str(tmp_path / "report.html")
    _save_report(report, out_path, policy_name="ecommerce")
    assert open(out_path).read() == "<html>ok</html>"
    assert "report saved" in capsys.readouterr().out


# ── cmd_findings ──────────────────────────────────────────────────────────────

def _write_result_json(tmp_path, results):
    p = tmp_path / "result.json"
    p.write_text(json.dumps({"results": results}))
    return str(p)


def test_findings_json_output(tmp_path, monkeypatch, capsys):
    import contradish.findings as findings_mod

    class _Finding:
        def to_dict(self):
            return {"headline": "h"}

    monkeypatch.setattr(findings_mod, "findings_from", lambda report: [_Finding()])
    path = _write_result_json(tmp_path, [{"input": "q", "name": "q", "cai_score": 0.5}])
    code = _run(cmd_findings, argparse.Namespace(result_file=path, json=True))
    assert code == 0
    assert json.loads(capsys.readouterr().out) == [{"headline": "h"}]


def test_findings_text_output_no_findings(tmp_path, monkeypatch, capsys):
    import contradish.findings as findings_mod
    monkeypatch.setattr(findings_mod, "findings_from", lambda report: [])
    path = _write_result_json(tmp_path, [{"input": "q"}])
    code = _run(cmd_findings, argparse.Namespace(result_file=path, json=False))
    assert code == 0
    assert "no findings" in capsys.readouterr().out


def test_findings_text_output_with_findings(tmp_path, monkeypatch, capsys):
    import contradish.findings as findings_mod

    class _Finding:
        headline = "Model flips under authority framing"
        detail = "some detail"
        cli_hint = "try --policy ecommerce"

    monkeypatch.setattr(findings_mod, "findings_from", lambda report: [_Finding()])
    path = _write_result_json(tmp_path, [{"input": "q"}])
    code = _run(cmd_findings, argparse.Namespace(result_file=path, json=False))
    out = capsys.readouterr().out
    assert code == 0
    assert "Model flips under authority framing" in out
    assert "try --policy ecommerce" in out


def test_findings_tolerates_missing_fields(tmp_path, monkeypatch):
    import contradish.findings as findings_mod
    captured = {}
    def fake_findings_from(report):
        captured["report"] = report
        return []
    monkeypatch.setattr(findings_mod, "findings_from", fake_findings_from)
    # No "input", "risk", "contradictions" -- must fall back to safe defaults.
    path = _write_result_json(tmp_path, [{"name": "bare case"}])
    code = _run(cmd_findings, argparse.Namespace(result_file=path, json=False))
    assert code == 0
    report = captured["report"]
    assert report.results[0].risk == RiskLevel.LOW
    assert report.results[0].test_case.input == "bare case"


# ── cmd_reconcile ─────────────────────────────────────────────────────────────

def _contradiction(prior_claim, query="q", session="u1", turn=1):
    return ReplayContradiction(
        session=session, turn_index=turn + 3, query=query, response="r",
        prior_turn_index=turn, prior_query="p", new_claim="changed",
        prior_claim=prior_claim, explanation="conflict", confidence=0.9)


def _write_reconcile_inputs(tmp_path):
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days, no exceptions", 0.95)])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days, no exceptions")], n_turns=10, sessions=["u1"])
    bench_path = str(tmp_path / "bench.json")
    replay_path = str(tmp_path / "replay.json")
    with open(bench_path, "w") as f:
        json.dump(report.to_dict(), f)
    with open(replay_path, "w") as f:
        json.dump(replay.to_dict(), f)
    return bench_path, replay_path


def test_reconcile_missing_file_exits_one(tmp_path, capsys):
    code = _run(cmd_reconcile, argparse.Namespace(
        report_file=str(tmp_path / "nope.json"), replay_file=str(tmp_path / "also_nope.json"),
        json=False, embeddings=False, match_threshold=0.3, max_validity_gaps=None))
    assert code == 1
    assert "not found" in capsys.readouterr().out


def test_reconcile_text_summary(tmp_path, capsys):
    bench, replay = _write_reconcile_inputs(tmp_path)
    code = _run(cmd_reconcile, argparse.Namespace(
        report_file=bench, replay_file=replay, json=False, embeddings=False,
        match_threshold=0.3, max_validity_gaps=None))
    assert code == 0
    assert "contradish reconcile" in capsys.readouterr().out


def test_reconcile_json_output(tmp_path, capsys):
    bench, replay = _write_reconcile_inputs(tmp_path)
    code = _run(cmd_reconcile, argparse.Namespace(
        report_file=bench, replay_file=replay, json=True, embeddings=False,
        match_threshold=0.3, max_validity_gaps=None))
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_validity_gaps"] == 1


def test_reconcile_max_validity_gaps_gate_fails(tmp_path, capsys):
    bench, replay = _write_reconcile_inputs(tmp_path)
    code = _run(cmd_reconcile, argparse.Namespace(
        report_file=bench, replay_file=replay, json=False, embeddings=False,
        match_threshold=0.3, max_validity_gaps=0))
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_reconcile_max_validity_gaps_gate_passes_when_under_limit(tmp_path):
    bench, replay = _write_reconcile_inputs(tmp_path)
    code = _run(cmd_reconcile, argparse.Namespace(
        report_file=bench, replay_file=replay, json=False, embeddings=False,
        match_threshold=0.3, max_validity_gaps=5))
    assert code == 0


# ── cmd_compare (Path A: two saved result JSONs, no live app) ──────────────────

def _write_report_json(tmp_path, name, results):
    p = tmp_path / name
    p.write_text(json.dumps(Report(results=results).to_dict()))
    return str(p)


def _compare_args(**over):
    ns = argparse.Namespace(
        eval_file=None, baseline_app=None, candidate_app=None,
        baseline_result=None, candidate_result=None,
        baseline_label="baseline", candidate_label="candidate",
        threshold=0.25, paraphrases=5, concurrency=4, json=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_compare_result_files_pass_under_threshold(tmp_path, capsys):
    baseline = _write_report_json(tmp_path, "b.json", [_bench_result("c", "a", 0.9)])
    candidate = _write_report_json(tmp_path, "c.json", [_bench_result("c", "a", 0.9)])
    code = _run(cmd_compare, _compare_args(baseline_result=baseline, candidate_result=candidate))
    assert code == 0


def test_compare_result_files_json_output_has_expected_keys(tmp_path, capsys):
    baseline = _write_report_json(tmp_path, "b.json", [_bench_result("c", "a", 0.9)])
    candidate = _write_report_json(tmp_path, "c.json", [_bench_result("c", "a", 0.2, contradiction=0.9)])
    # threshold=1.0 so the regression gate doesn't also print a trailing
    # "FAIL: ..." line after the JSON, which would break json.loads below.
    code = _run(cmd_compare, _compare_args(baseline_result=baseline, candidate_result=candidate,
                                           json=True, threshold=1.0))
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["baseline_label"] == "baseline" and data["candidate_label"] == "candidate"
    assert "strain_delta" in data and "per_case" in data


def test_compare_result_files_threshold_gate_fails_and_still_prints_json(tmp_path, capsys):
    baseline = _write_report_json(tmp_path, "b.json", [_bench_result("c", "a", 0.9)])
    candidate = _write_report_json(tmp_path, "c.json", [_bench_result("c", "a", 0.2, contradiction=0.9)])
    code = _run(cmd_compare, _compare_args(baseline_result=baseline, candidate_result=candidate, json=True))
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_compare_missing_both_paths_prints_usage_and_exits_one(capsys):
    code = _run(cmd_compare, _compare_args())
    assert code == 1
    assert "contradish compare needs" in capsys.readouterr().out


# ── cmd_fairness ──────────────────────────────────────────────────────────────

def _fairness_args(**over):
    ns = argparse.Namespace(
        policy=None, eval_file=None, app="fake_app_module:fake_app",
        provider=None, flag_threshold=0.30, threshold=None, concurrency=4, json=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def _fake_app(monkeypatch):
    monkeypatch.setattr(cli, "_load_callable", lambda path: (lambda q: "answer"))


def _fairness_audit(flagged=False):
    result = CaseProfileResult(case_name="c1", profile_name="age65", attribute="age",
                               baseline_output="a", variant_output="b",
                               shift=0.9 if flagged else 0.05)
    return FairnessAudit(
        results=[result], per_attribute={"age": result.shift}, per_profile={"age65": result.shift},
        flag_threshold=0.30, flagged=[result] if flagged else [],
    )


def test_fairness_needs_policy_or_eval_file(_fake_app, capsys):
    code = _run(cmd_fairness, _fairness_args(policy=None, eval_file=None))
    assert code == 1
    assert "needs --policy" in capsys.readouterr().out


def test_fairness_text_output_no_flags(_fake_app, monkeypatch, capsys):
    import contradish.fairness as fairness_mod
    monkeypatch.setattr(fairness_mod, "audit_fairness", lambda **kw: _fairness_audit(flagged=False))
    code = _run(cmd_fairness, _fairness_args(policy="ecommerce"))
    assert code == 0
    assert "No disparate treatment detected" in capsys.readouterr().out


def test_fairness_threshold_gate_fails_when_shift_exceeds(_fake_app, monkeypatch, capsys):
    import contradish.fairness as fairness_mod
    monkeypatch.setattr(fairness_mod, "audit_fairness", lambda **kw: _fairness_audit(flagged=True))
    code = _run(cmd_fairness, _fairness_args(policy="ecommerce", threshold=0.5))
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_fairness_json_output(_fake_app, monkeypatch, capsys):
    import contradish.fairness as fairness_mod
    monkeypatch.setattr(fairness_mod, "audit_fairness", lambda **kw: _fairness_audit(flagged=True))
    code = _run(cmd_fairness, _fairness_args(policy="ecommerce", json=True))
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["flagged"][0]["attribute"] == "age"


# ── cmd_judge_floor ───────────────────────────────────────────────────────────

def _judge_floor_args(**over):
    ns = argparse.Namespace(
        judge_provider=None, judge_model=None, n_rephrasings=3, concurrency=4, json=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _judge_calibration():
    return JudgeCalibration(judge_provider="openai", judge_model="gpt-4o", n_pairs=20,
                            n_rephrasings=3, accuracy=0.95, floor_strain=0.04,
                            confidence_floor=0.08)


def test_judge_floor_text_output(monkeypatch, capsys):
    import contradish.judge_calibration as jc_mod
    monkeypatch.setattr(jc_mod, "measure_judge_floor", lambda **kw: _judge_calibration())
    code = _run(cmd_judge_floor, _judge_floor_args())
    assert code == 0
    assert "confidence_floor" in capsys.readouterr().out


def test_judge_floor_json_output(monkeypatch, capsys):
    import contradish.judge_calibration as jc_mod
    monkeypatch.setattr(jc_mod, "measure_judge_floor", lambda **kw: _judge_calibration())
    code = _run(cmd_judge_floor, _judge_floor_args(json=True))
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["judge_model"] == "gpt-4o"


# ── cmd_prompt ────────────────────────────────────────────────────────────────

def _prompt_args(**over):
    ns = argparse.Namespace(
        prompt_target=None, inline=None, provider=None, model=None,
        rewrite=False, threshold=None, json=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _prompt_analysis(with_tension=True):
    tensions = []
    if with_tension:
        tensions = [PromptTension(clauses=["Be empathetic.", "Refunds within 30 days only."],
                                  description="sympathy tips policy",
                                  exploiting_techniques=["sympathy"], severity="high")]
    return PromptAnalysis(prompt="p", tensions=tensions, deconflicted_prompt="CLEAN PROMPT")


def test_prompt_needs_a_target(capsys):
    code = _run(cmd_prompt, _prompt_args())
    assert code == 1
    assert "needs a file path" in capsys.readouterr().out


def test_prompt_empty_inline_exits_one(capsys):
    code = _run(cmd_prompt, _prompt_args(inline="   "))
    assert code == 1
    assert "Empty prompt" in capsys.readouterr().out


def test_prompt_inline_no_tensions(monkeypatch, capsys):
    import contradish.prompt_analyzer as pa_mod
    monkeypatch.setattr(pa_mod, "analyze_prompt", lambda **kw: _prompt_analysis(with_tension=False))
    code = _run(cmd_prompt, _prompt_args(inline="Be nice."))
    assert code == 0
    assert "No internal contradictions found" in capsys.readouterr().out


def test_prompt_rewrite_prints_only_deconflicted(monkeypatch, capsys):
    import contradish.prompt_analyzer as pa_mod
    monkeypatch.setattr(pa_mod, "analyze_prompt", lambda **kw: _prompt_analysis())
    code = _run(cmd_prompt, _prompt_args(inline="Be empathetic.", rewrite=True))
    assert code == 0
    assert capsys.readouterr().out == "CLEAN PROMPT\n"


def test_prompt_file_target_reads_file_contents(tmp_path, monkeypatch, capsys):
    import contradish.prompt_analyzer as pa_mod
    captured = {}
    def fake_analyze(prompt, provider, model):
        captured["prompt"] = prompt
        return _prompt_analysis(with_tension=False)
    monkeypatch.setattr(pa_mod, "analyze_prompt", fake_analyze)
    p = tmp_path / "sp.txt"
    p.write_text("You are a support agent.")
    code = _run(cmd_prompt, _prompt_args(prompt_target=str(p)))
    assert code == 0
    assert captured["prompt"] == "You are a support agent."


def test_prompt_threshold_gate_fails_on_offending_tension(monkeypatch, capsys):
    import contradish.prompt_analyzer as pa_mod
    monkeypatch.setattr(pa_mod, "analyze_prompt", lambda **kw: _prompt_analysis(with_tension=True))
    code = _run(cmd_prompt, _prompt_args(inline="x", threshold="high"))
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_prompt_threshold_unknown_value_exits_one(monkeypatch, capsys):
    import contradish.prompt_analyzer as pa_mod
    monkeypatch.setattr(pa_mod, "analyze_prompt", lambda **kw: _prompt_analysis(with_tension=False))
    code = _run(cmd_prompt, _prompt_args(inline="x", threshold="not_a_real_severity"))
    assert code == 1
    assert "unknown --threshold" in capsys.readouterr().out


# ── cmd_replay ────────────────────────────────────────────────────────────────

def _replay_args(**over):
    ns = argparse.Namespace(
        transcript="conversations.jsonl", embeddings=False, repair=False,
        provider=None, model=None, max_contradictions=None, output=None, json=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_replay_missing_transcript_exits_one(tmp_path, capsys):
    code = _run(cmd_replay, _replay_args(transcript=str(tmp_path / "nope.jsonl")))
    assert code == 1
    assert "transcript not found" in capsys.readouterr().out


def test_replay_no_turns_found_exits_one(tmp_path, monkeypatch, capsys):
    # cmd_replay does a function-local `from contradish._replay import
    # load_transcript, replay_transcript` -- contradish/replay.py was renamed
    # to contradish/_replay.py precisely so this plain import (no importlib
    # trick needed) unambiguously gets the submodule, not the re-exported
    # `replay` function.
    import contradish._replay as replay_mod
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    monkeypatch.setattr(replay_mod, "load_transcript", lambda path: [])
    code = _run(cmd_replay, _replay_args(transcript=str(p)))
    assert code == 1
    assert "no turns found" in capsys.readouterr().out


def test_replay_text_summary_and_max_contradictions_gate(tmp_path, monkeypatch, capsys):
    # cmd_replay does a function-local `from contradish._replay import
    # load_transcript, replay_transcript` -- contradish/replay.py was renamed
    # to contradish/_replay.py precisely so this plain import (no importlib
    # trick needed) unambiguously gets the submodule, not the re-exported
    # `replay` function.
    import contradish._replay as replay_mod
    p = tmp_path / "log.jsonl"
    p.write_text('{"role": "user", "content": "hi"}\n')
    monkeypatch.setattr(replay_mod, "load_transcript", lambda path: [{"role": "user", "content": "hi"}])
    report = ReplayReport(contradictions=[_contradiction("x")], n_turns=5, sessions=["u1"])
    monkeypatch.setattr(replay_mod, "replay_transcript", lambda turns, **kw: report)
    code = _run(cmd_replay, _replay_args(transcript=str(p), max_contradictions=0))
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_replay_writes_output_file(tmp_path, monkeypatch, capsys):
    # cmd_replay does a function-local `from contradish._replay import
    # load_transcript, replay_transcript` -- contradish/replay.py was renamed
    # to contradish/_replay.py precisely so this plain import (no importlib
    # trick needed) unambiguously gets the submodule, not the re-exported
    # `replay` function.
    import contradish._replay as replay_mod
    p = tmp_path / "log.jsonl"
    p.write_text('{"role": "user", "content": "hi"}\n')
    monkeypatch.setattr(replay_mod, "load_transcript", lambda path: [{"role": "user", "content": "hi"}])
    report = ReplayReport(contradictions=[], n_turns=1, sessions=["u1"])
    monkeypatch.setattr(replay_mod, "replay_transcript", lambda turns, **kw: report)
    out_path = str(tmp_path / "out.json")
    code = _run(cmd_replay, _replay_args(transcript=str(p), output=out_path))
    assert code == 0
    with open(out_path) as f:
        saved = json.load(f)
    assert saved["n_turns"] == 1
