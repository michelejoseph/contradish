"""
Tests for the CLI surface of the closed loop: `contradish improve --from-production`.
Run with: pytest tests/test_cli_from_production.py

No API key or model call required. improve_from_production is mocked, so these
tests cover only the CLI wiring: report loading, base-case threading, kinds
selection, and the no-gaps exit path.
"""
import argparse
import importlib
import json
import os
import tempfile

from contradish.cli import cmd_improve
from contradish.models import TestCase, TestResult, Report, RiskLevel
from contradish.replay import ReplayReport, ReplayContradiction

# cmd_improve does a function-local `from contradish.improve import
# improve_from_production`, so patching this module attribute is enough.
improve_mod = importlib.import_module("contradish.improve")


# ── fixtures ────────────────────────────────────────────────────────────────

def _bench_result(name, canonical, consistency, contradiction=0.0):
    tc = TestCase(input=name, name=name, canonical_answer=canonical)
    return TestResult(test_case=tc, paraphrases=[], outputs=[],
                      consistency_score=consistency, contradiction_score=contradiction,
                      risk=RiskLevel.LOW)


def _contradiction(prior_claim, query="q", session="u1", turn=1):
    return ReplayContradiction(
        session=session, turn_index=turn + 3, query=query, response="r",
        prior_turn_index=turn, prior_query="p", new_claim="changed",
        prior_claim=prior_claim, explanation="conflict", confidence=0.9)


def _write_reports(tmpdir):
    report = Report(results=[
        _bench_result("refund window", "Refund window is 30 days, no exceptions", 0.95)])
    replay = ReplayReport(contradictions=[
        _contradiction("Refund window is 30 days, no exceptions", query="refund after 45?")],
        n_turns=10, sessions=["u1"])
    bench_path = os.path.join(tmpdir, "bench.json")
    replay_path = os.path.join(tmpdir, "replay.json")
    with open(bench_path, "w") as f:
        json.dump(report.to_dict(), f)
    with open(replay_path, "w") as f:
        json.dump(replay.to_dict(), f)
    return bench_path, replay_path


def _args(**over):
    ns = argparse.Namespace(
        json=True, from_production=None, prompt_file=None, system_prompt="SP",
        policy=None, eval_file=None, model="gpt-4o-mini", provider="openai",
        method="prompt", target_strain=0.2, n_variants=3, paraphrases=5,
        enable_finetune=False, ft_provider="openai", match_threshold=0.3,
        include_confirmed=False, concurrency=4, holdout_frac=0.0, seed=0,
        output=None, report=None,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


class _FakeResult:
    target_met = True
    def to_dict(self):
        return {"ok": True}


class _Recorder:
    def __init__(self, ret):
        self.ret = ret
        self.calls = []
    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.ret


def _run(args, ret):
    """Drive cmd_improve with improve_from_production mocked. Returns (recorder, exit_code)."""
    orig_fn = improve_mod.improve_from_production
    orig_key = os.environ.get("OPENAI_API_KEY")
    rec = _Recorder(ret)
    improve_mod.improve_from_production = rec
    os.environ["OPENAI_API_KEY"] = "sk-test"          # satisfy _check_api_key
    code = None
    try:
        cmd_improve(args)
    except SystemExit as e:
        code = e.code
    finally:
        improve_mod.improve_from_production = orig_fn
        if orig_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = orig_key
    return rec, code


# ── tests ─────────────────────────────────────────────────────────────────

def test_from_production_dispatches_and_loads_reports():
    with tempfile.TemporaryDirectory() as d:
        bench, replay = _write_reports(d)
        rec, code = _run(_args(from_production=[bench, replay]), _FakeResult())
        assert code == 0
        assert len(rec.calls) == 1
        (report_arg, replay_arg), kwargs = rec.calls[0]
        assert isinstance(report_arg, Report)
        assert isinstance(replay_arg, ReplayReport)
        assert kwargs["system_prompt"] == "SP" and kwargs["model"] == "gpt-4o-mini"
        assert kwargs["match_threshold"] == 0.3
        assert kwargs["kinds"] == ("validity_gap", "coverage_gap")
        assert kwargs["base_cases"] is None


def test_from_production_threads_policy_as_base_cases():
    with tempfile.TemporaryDirectory() as d:
        bench, replay = _write_reports(d)
        rec, code = _run(_args(from_production=[bench, replay], policy="ecommerce"), _FakeResult())
        assert code == 0
        _, kwargs = rec.calls[0]
        assert isinstance(kwargs["base_cases"], list) and len(kwargs["base_cases"]) > 0


def test_from_production_include_confirmed_widens_kinds():
    with tempfile.TemporaryDirectory() as d:
        bench, replay = _write_reports(d)
        rec, code = _run(_args(from_production=[bench, replay], include_confirmed=True), _FakeResult())
        assert code == 0
        _, kwargs = rec.calls[0]
        assert kwargs["kinds"] == ("validity_gap", "confirmed", "coverage_gap")


def test_from_production_no_gaps_exits_zero():
    with tempfile.TemporaryDirectory() as d:
        bench, replay = _write_reports(d)
        rec, code = _run(_args(from_production=[bench, replay]), None)   # None = nothing to repair
        assert code == 0
        assert len(rec.calls) == 1                                       # called, returned None


def test_from_production_missing_file_exits_one():
    with tempfile.TemporaryDirectory() as d:
        bench, _ = _write_reports(d)
        rec, code = _run(_args(from_production=[bench, os.path.join(d, "nope.json")]), _FakeResult())
        assert code == 1
        assert rec.calls == []                                           # never reached the orchestrator


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
