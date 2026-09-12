"""
Full test coverage for contradish/_improve.py (public API: contradish.improve)
— the detect -> diagnose -> repair -> re-verify loop.

improve() orchestrates two other modules (Suite, PromptRepair) via
function-local imports (`from .suite import Suite`, `from .repair import
PromptRepair` inside improve() itself; `from .llm import LLMClient` inside
_make_app_for_prompt). Per this session's established pattern, that means
tests must patch the attribute on the *source* module
(contradish.suite.Suite, contradish.repair.PromptRepair,
contradish.llm.LLMClient) rather than any local alias — the function body
re-resolves the name from that module's namespace every call.

FakeSuite/FakePromptRepair are queue-driven: each test sets the exact
sequence of Report/RepairResult objects to hand back, in the order
improve()'s own docstring says Suite.run() is invoked (train baseline ->
holdout baseline, if any -> holdout re-score of the winner, if any), so
each scenario's assertions can be pinned to a specific, known Report rather
than inferred from real LLM/Suite behavior.
"""
import json
from types import SimpleNamespace

import pytest

from contradish.models import ContradictionPair, Report, RepairResult, TestCase, TestResult
from contradish._improve import (
    ImprovementResult,
    _make_app_for_prompt,
    _resolve_cases,
    _submit_finetune_job,
    _write_finetune_jsonl,
    improve,
    improve_from_production,
)

# NOTE on a real bug found while first writing these tests, since fixed:
# contradish/__init__.py used to do `from .improve import improve` and
# `from .reconcile import reconcile` -- each re-export's NAME collided with
# its own submodule's name (improve.py's `improve` function vs. the
# `contradish.improve` submodule), so the import statement's final binding
# overwrote the submodule reference on the `contradish` package object.
# `contradish.improve.<anything>` (attribute-chain access into the
# submodule) raised AttributeError, even though `from contradish.improve
# import improve` (a real import statement) worked fine -- and critically,
# `import contradish.improve as m` did NOT dodge it either, since Python
# compiles a dotted "import ... as" into an attribute getattr on the
# already-imported parent package, not a sys.modules lookup.
#
# Fixed by renaming the files on disk (improve.py -> _improve.py,
# reconcile.py -> _reconcile.py, replay.py -> _replay.py) so the collision
# can't happen: `contradish.improve` is now unambiguously the re-exported
# function (the documented public API, unchanged), and the submodule lives
# at `contradish._improve`, a distinct attribute slot. See the comment
# block in contradish/__init__.py above the _improve/_reconcile/_replay
# imports for the full explanation, and the regression tests below
# (renamed from "..._is_shadowed_by_its_own_reexport" to
# "..._no_longer_shadowed...") which now assert the fix instead of pinning
# the bug.

import contradish._improve as improve_module
import contradish._reconcile as reconcile_module


# ── Fixture builders ────────────────────────────────────────────────────────

def _tc(input="a test question", name=None, canonical_answer=None, **kw):
    return TestCase(input=input, name=name, canonical_answer=canonical_answer, **kw)


def _cp(output_a="out a", output_b="out b", severity="policy",
        explanation="they disagree", input_a="in a", input_b="in b"):
    return ContradictionPair(
        input_a=input_a, input_b=input_b,
        output_a=output_a, output_b=output_b,
        explanation=explanation, severity=severity,
    )


def _tr(tc=None, consistency_score=0.9, contradictions=None,
        suggestion=None, truth_score=None, truth_strain=None, **kw):
    return TestResult(
        test_case=tc or _tc(),
        paraphrases=["p1"],
        outputs=["o0", "o1"],
        consistency_score=consistency_score,
        contradiction_score=0.0,
        contradictions=contradictions or [],
        suggestion=suggestion,
        truth_score=truth_score,
        truth_strain=truth_strain,
        **kw,
    )


def _report(results, **kw):
    return Report(results=results, **kw)


def _repair_result(improved_prompt="improved prompt", report=None, rank=1,
                    original_cai_score=0.4, improved_cai_score=0.95):
    return RepairResult(
        original_prompt="original prompt",
        improved_prompt=improved_prompt,
        original_cai_score=original_cai_score,
        improved_cai_score=improved_cai_score,
        delta=improved_cai_score - original_cai_score,
        report=report or _report([_tr(consistency_score=improved_cai_score)]),
        rank=rank,
    )


# ── Fake LLMClient (anthropic/openai) for _make_app_for_prompt ─────────────

class _FakeAnthropicLLM:
    provider = "anthropic"
    fast_model = "claude-haiku-x"

    def __init__(self, *a, **kw):
        self.calls = []
        self._client = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        block = SimpleNamespace(text="  anthropic reply  ")
        return SimpleNamespace(content=[block])


class _FakeOpenAILLM:
    provider = "openai"
    fast_model = "gpt-fast-x"

    def __init__(self, *a, **kw):
        self.calls = []
        self._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self._create))
        )

    def _create(self, **kw):
        self.calls.append(kw)
        msg = SimpleNamespace(content="  openai reply  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ── Fake Suite / PromptRepair for improve() orchestration tests ────────────

class FakeSuite:
    """Queue-driven: each .run() call pops the next Report off the class
    queue, in the order improve() actually invokes Suite.run()."""
    queue: list = []
    instances: list = []

    def __init__(self, app, api_key=None, provider=None, **kw):
        self.app = app
        self.api_key = api_key
        self.provider = provider
        self.cases = []
        FakeSuite.instances.append(self)

    def add(self, tc):
        self.cases.append(tc)

    def run(self, paraphrases=5, verbose=True, concurrency=4):
        return FakeSuite.queue.pop(0)


class FakePromptRepair:
    variants: list = []
    last_kwargs: dict = {}

    def __init__(self, api_key=None, provider=None, n=3, **kw):
        self.api_key = api_key
        self.provider = provider
        self.n = n

    def fix(self, **kwargs):
        FakePromptRepair.last_kwargs = kwargs
        return FakePromptRepair.variants


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch):
    FakeSuite.queue = []
    FakeSuite.instances = []
    FakePromptRepair.variants = []
    FakePromptRepair.last_kwargs = {}
    monkeypatch.setattr("contradish.suite.Suite", FakeSuite)
    monkeypatch.setattr("contradish.repair.PromptRepair", FakePromptRepair)
    monkeypatch.setattr("contradish.llm.LLMClient", _FakeAnthropicLLM)


# ── ImprovementResult.summary() ─────────────────────────────────────────────

def test_summary_shows_down_arrow_and_target_met():
    r = ImprovementResult(
        baseline_strain=0.50, improved_strain=0.20, strain_delta=-0.30,
        target_strain=0.25, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
    )
    out = r.summary()
    assert "↓" in out
    assert "target met" in out
    assert "60%" in out  # 0.30/0.50 = 60%


def test_summary_shows_up_arrow_and_target_not_met():
    r = ImprovementResult(
        baseline_strain=0.20, improved_strain=0.30, strain_delta=0.10,
        target_strain=0.10, target_met=False, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
    )
    out = r.summary()
    assert "↑" in out
    assert "target 0.10 not yet hit" in out


def test_summary_handles_zero_baseline_strain_without_division_error():
    r = ImprovementResult(
        baseline_strain=0.0, improved_strain=0.0, strain_delta=0.0,
        target_strain=0.10, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
    )
    out = r.summary()
    assert "0%" in out


def test_summary_includes_holdout_scope():
    r = ImprovementResult(
        baseline_strain=0.5, improved_strain=0.2, strain_delta=-0.3,
        target_strain=0.25, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
        holdout_size=3, train_size=7,
    )
    assert "[holdout n=3, train n=7]" in r.summary()


def test_summary_omits_scope_when_no_holdout():
    r = ImprovementResult(
        baseline_strain=0.5, improved_strain=0.2, strain_delta=-0.3,
        target_strain=0.25, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
    )
    assert "holdout" not in r.summary()


def test_summary_shows_rejected_when_truth_regressed():
    r = ImprovementResult(
        baseline_strain=0.5, improved_strain=0.05, strain_delta=-0.45,
        target_strain=0.10, target_met=False, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
        baseline_truth_strain=0.10, improved_truth_strain=0.40, truth_regressed=True,
    )
    out = r.summary()
    assert "REJECTED" in out
    assert "0.100" in out and "0.400" in out


def test_summary_shows_truth_strain_when_not_regressed():
    r = ImprovementResult(
        baseline_strain=0.5, improved_strain=0.05, strain_delta=-0.45,
        target_strain=0.10, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
        baseline_truth_strain=0.30, improved_truth_strain=0.10, truth_regressed=False,
    )
    out = r.summary()
    assert "REJECTED" not in out
    assert "truth_strain 0.300 to 0.100" in out


def test_summary_omits_truth_section_when_no_truth_data():
    r = ImprovementResult(
        baseline_strain=0.5, improved_strain=0.2, strain_delta=-0.3,
        target_strain=0.25, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
    )
    assert "truth_strain" not in r.summary()


# ── ImprovementResult.to_dict() ─────────────────────────────────────────────

def test_to_dict_includes_variant_strains():
    variant = _repair_result(rank=1, original_cai_score=0.4, improved_cai_score=0.9)
    r = ImprovementResult(
        baseline_strain=0.6, improved_strain=0.1, strain_delta=-0.5,
        target_strain=0.15, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
        variant_results=[variant],
    )
    d = r.to_dict()
    assert d["method"] == "prompt"
    assert d["target_met"] is True
    assert d["baseline_report"] == r.baseline_report.to_dict()
    assert len(d["variant_strains"]) == 1
    vs = d["variant_strains"][0]
    assert vs["rank"] == 1
    assert vs["original_strain"] == variant.original_cai_strain
    assert vs["improved_strain"] == variant.improved_cai_strain
    assert vs["strain_delta"] == variant.strain_delta


def test_to_dict_empty_variant_results():
    r = ImprovementResult(
        baseline_strain=0.6, improved_strain=0.6, strain_delta=0.0,
        target_strain=0.15, target_met=False, method="prompt",
        baseline_prompt="a", improved_prompt="a",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
    )
    assert r.to_dict()["variant_strains"] == []


# ── _make_app_for_prompt ────────────────────────────────────────────────────

def test_make_app_for_prompt_anthropic_uses_fast_model_and_strips_reply(monkeypatch):
    monkeypatch.setattr("contradish.llm.LLMClient", _FakeAnthropicLLM)
    app = _make_app_for_prompt(system_prompt="be nice")
    out = app("hello?")
    assert out == "anthropic reply"


def test_make_app_for_prompt_openai_uses_explicit_model(monkeypatch):
    monkeypatch.setattr("contradish.llm.LLMClient", _FakeOpenAILLM)
    app = _make_app_for_prompt(system_prompt="be nice", model="gpt-explicit")
    out = app("hello?")
    assert out == "openai reply"


def test_make_app_for_prompt_openai_message_shape(monkeypatch):
    captured = {}

    class _CapturingOpenAILLM(_FakeOpenAILLM):
        def _create(self, **kw):
            captured.update(kw)
            return super()._create(**kw)

    monkeypatch.setattr("contradish.llm.LLMClient", _CapturingOpenAILLM)
    app = _make_app_for_prompt(system_prompt="be nice", model="gpt-explicit")
    app("hello?")
    assert captured["model"] == "gpt-explicit"
    assert captured["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello?"},
    ]


def test_make_app_for_prompt_anthropic_message_shape(monkeypatch):
    captured = {}

    class _CapturingAnthropicLLM(_FakeAnthropicLLM):
        def _create(self, **kw):
            captured.update(kw)
            return super()._create(**kw)

    monkeypatch.setattr("contradish.llm.LLMClient", _CapturingAnthropicLLM)
    app = _make_app_for_prompt(system_prompt="be nice")
    app("hello?")
    assert captured["system"] == "be nice"
    assert captured["model"] == _FakeAnthropicLLM.fast_model
    assert captured["messages"] == [{"role": "user", "content": "hello?"}]


# ── _resolve_cases ───────────────────────────────────────────────────────────

def test_resolve_cases_passes_through_a_list():
    cases = [_tc(), _tc()]
    assert _resolve_cases(cases) is cases


def test_resolve_cases_loads_a_policy_by_name(monkeypatch):
    pack_cases = [_tc(name="from pack")]
    fake_pack = SimpleNamespace(cases=pack_cases)
    monkeypatch.setattr("contradish.policies.load_policy", lambda name: fake_pack)
    assert _resolve_cases("medication") is pack_cases


def test_resolve_cases_rejects_other_types():
    with pytest.raises(TypeError, match="cases must be"):
        _resolve_cases(123)


# ── _write_finetune_jsonl ───────────────────────────────────────────────────

def test_write_finetune_jsonl_writes_one_line_per_contradiction_pair(tmp_path):
    tc = _tc(name="refund rule")
    failed = _tr(
        tc=tc, consistency_score=0.4, suggestion="be firmer",
        contradictions=[_cp(input_b="q1", output_a="a1", severity="policy", explanation="e1"),
                        _cp(input_b="q2", output_a="a2", severity="factual", explanation="e2")],
    )
    report = _report([failed])
    out_path = str(tmp_path / "out.jsonl")
    result_path = _write_finetune_jsonl(report, system_prompt="Base prompt.", out_path=out_path)
    assert result_path == str((tmp_path / "out.jsonl").resolve())
    lines = (tmp_path / "out.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    ex0 = json.loads(lines[0])
    assert ex0["messages"][0]["content"] == "Base prompt.\n\nbe firmer"
    assert ex0["messages"][1] == {"role": "user", "content": "q1"}
    assert ex0["messages"][2] == {"role": "assistant", "content": "a1"}
    assert ex0["meta"]["rule"] == "refund rule"
    assert ex0["meta"]["severity"] == "policy"
    assert ex0["meta"]["explanation"] == "e1"


def test_write_finetune_jsonl_skips_failures_without_contradictions(tmp_path):
    failed = _tr(consistency_score=0.4, contradictions=[])
    report = _report([failed])
    out_path = str(tmp_path / "empty.jsonl")
    result_path = _write_finetune_jsonl(report, system_prompt="p", out_path=out_path)
    assert result_path == ""


def test_write_finetune_jsonl_handles_blank_prompt_and_no_suggestion(tmp_path):
    failed = _tr(consistency_score=0.4, suggestion=None, contradictions=[_cp()])
    report = _report([failed])
    out_path = str(tmp_path / "blank.jsonl")
    _write_finetune_jsonl(report, system_prompt="", out_path=out_path)
    ex = json.loads((tmp_path / "blank.jsonl").read_text().strip())
    assert ex["messages"][0]["content"] == ""


def test_write_finetune_jsonl_default_out_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    failed = _tr(consistency_score=0.4, contradictions=[_cp()])
    report = _report([failed])
    result_path = _write_finetune_jsonl(report, system_prompt="p")
    assert result_path.endswith("repair_finetune.jsonl")
    assert (tmp_path / "repair_finetune.jsonl").exists()


# ── improve(): validation ───────────────────────────────────────────────────

def test_improve_rejects_empty_case_list():
    with pytest.raises(ValueError, match="No test cases provided"):
        improve(cases=[], system_prompt="p", verbose=False)


def test_improve_rejects_out_of_range_holdout_frac():
    with pytest.raises(ValueError, match="holdout_frac must be between"):
        improve(cases=[_tc()], system_prompt="p", holdout_frac=1.0, verbose=False)


# ── improve(): no-holdout paths ─────────────────────────────────────────────

def test_improve_early_exit_when_baseline_already_meets_target():
    baseline = _report([_tr(consistency_score=0.95)])  # cai_strain ~0.05
    FakeSuite.queue = [baseline]
    result = improve(cases=[_tc()], system_prompt="p", target_strain=0.20, verbose=False)
    assert result.target_met is True
    assert result.baseline_strain == result.improved_strain == baseline.cai_strain
    assert result.variant_results == []
    assert result.improved_prompt == "p"


def test_improve_returns_baseline_when_no_variants_generated():
    baseline = _report([_tr(consistency_score=0.4)])  # cai_strain ~0.6, above target
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = []
    result = improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False)
    assert result.target_met is False
    assert result.variant_results == []
    assert result.improved_prompt == "p"
    assert result.baseline_strain == baseline.cai_strain


def test_improve_picks_best_variant_and_hits_target():
    baseline = _report([_tr(consistency_score=0.4)])   # strain ~0.6
    improved = _report([_tr(consistency_score=0.95)])  # strain ~0.05
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(
        improved_prompt="better prompt", report=improved, rank=1,
        original_cai_score=0.4, improved_cai_score=0.95,
    )]
    result = improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False)
    assert result.target_met is True
    assert result.improved_prompt == "better prompt"
    assert result.improved_strain == improved.cai_strain
    assert result.strain_delta == round(improved.cai_strain - baseline.cai_strain, 4)
    assert len(result.variant_results) == 1
    # holdout fields stay None when no holdout was requested
    assert result.holdout_size is None
    assert result.train_size is None


def test_improve_passes_a_working_app_factory_to_prompt_repair(monkeypatch):
    # PromptRepair.fix() receives an app_factory closure (improve.py's
    # internal `app_factory`) that it uses to build a fresh app callable per
    # candidate prompt. Verify that closure actually works end-to-end by
    # having the fake PromptRepair call it, the way the real one does.
    baseline = _report([_tr(consistency_score=0.4)])
    FakeSuite.queue = [baseline]

    captured = {}

    class _CallingPromptRepair(FakePromptRepair):
        def fix(self, **kwargs):
            captured.update(kwargs)
            app = kwargs["app_factory"]("candidate prompt")
            captured["app_reply"] = app("a question")
            return []

    monkeypatch.setattr("contradish.repair.PromptRepair", _CallingPromptRepair)
    improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False)
    assert captured["app_reply"] == "anthropic reply"  # from _FakeAnthropicLLM


def test_improve_target_not_met_when_variant_still_above_target():
    baseline = _report([_tr(consistency_score=0.4)])
    improved = _report([_tr(consistency_score=0.5)])  # still above target
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(report=improved, original_cai_score=0.4, improved_cai_score=0.5)]
    result = improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False)
    assert result.target_met is False


# ── improve(): holdout paths ────────────────────────────────────────────────

def test_improve_holdout_early_exit_uses_holdout_strain_as_headline():
    train_baseline = _report([_tr(consistency_score=0.4)])    # strain ~0.6 (irrelevant to early exit)
    holdout_baseline = _report([_tr(consistency_score=0.97)])  # strain ~0.03, meets target
    FakeSuite.queue = [train_baseline, holdout_baseline]
    result = improve(
        cases=[_tc(), _tc()], system_prompt="p", target_strain=0.10,
        holdout_frac=0.5, seed=0, verbose=False,
    )
    assert result.target_met is True
    assert result.baseline_strain == holdout_baseline.cai_strain
    assert result.holdout_size == 1
    assert result.train_size == 1


def test_improve_holdout_scores_winner_on_holdout_not_train():
    train_baseline    = _report([_tr(consistency_score=0.3)])   # strain ~0.7
    holdout_baseline  = _report([_tr(consistency_score=0.3)])   # strain ~0.7, above target
    train_improved    = _report([_tr(consistency_score=0.99)])  # strain ~0.01 (train winner pick)
    holdout_improved  = _report([_tr(consistency_score=0.85)])  # strain ~0.15 (honest holdout read)
    FakeSuite.queue = [train_baseline, holdout_baseline, holdout_improved]
    FakePromptRepair.variants = [_repair_result(
        improved_prompt="winner", report=train_improved,
        original_cai_score=0.3, improved_cai_score=0.99,
    )]
    result = improve(
        cases=[_tc(), _tc()], system_prompt="p", target_strain=0.10,
        holdout_frac=0.5, seed=0, verbose=False,
    )
    # Headline numbers are the HOLDOUT ones, not the train winner's own strain.
    assert result.baseline_strain == holdout_baseline.cai_strain
    assert result.improved_strain == holdout_improved.cai_strain
    assert result.train_baseline_strain == train_baseline.cai_strain
    assert result.train_improved_strain == train_improved.cai_strain
    assert result.improved_prompt == "winner"
    # holdout strain (~0.15) misses the 0.10 target even though train strain (~0.01) would have hit it.
    assert result.target_met is False


# ── improve(): truth gate ───────────────────────────────────────────────────

def test_improve_truth_gate_rejects_a_consistency_win_that_costs_truth():
    baseline = _report([_tr(consistency_score=0.4, truth_score=0.9, truth_strain=0.10)])
    improved_result_report = _report([_tr(consistency_score=0.99, truth_score=0.2, truth_strain=0.50)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(
        report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.99,
    )]
    result = improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False)
    # CAI Strain target would be met (~0.01 <= 0.10), but truth regressed -> rejected.
    assert result.truth_regressed is True
    assert result.target_met is False
    assert result.baseline_truth_strain == 0.10
    assert result.improved_truth_strain == 0.50


def test_improve_truth_gate_allows_a_small_improvement_within_tolerance():
    baseline = _report([_tr(consistency_score=0.4, truth_score=0.9, truth_strain=0.10)])
    improved_result_report = _report([_tr(consistency_score=0.99, truth_score=0.89, truth_strain=0.11)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(
        report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.99,
    )]
    result = improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False)
    # 0.11 - 0.10 = 0.01, within the 0.02 tolerance -> not regressed.
    assert result.truth_regressed is False
    assert result.target_met is True


def test_improve_no_truth_gate_when_no_canonical_answers():
    baseline = _report([_tr(consistency_score=0.4)])  # truth_strain None
    improved_result_report = _report([_tr(consistency_score=0.99)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.99)]
    result = improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False)
    assert result.truth_regressed is False
    assert result.baseline_truth_strain is None
    assert result.improved_truth_strain is None
    assert result.target_met is True


# ── improve(): finetune method ──────────────────────────────────────────────

def test_improve_finetune_writes_jsonl_but_does_not_submit_by_default(monkeypatch):
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.95)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.95)]

    write_calls = []

    def _fake_write(report, system_prompt, out_path=None):
        write_calls.append((report, system_prompt, out_path))
        return "/tmp/fake.jsonl"

    def _boom_submit(*a, **kw):
        raise AssertionError("must not submit when enable_finetune=False")

    monkeypatch.setattr(improve_module, "_write_finetune_jsonl", _fake_write)
    monkeypatch.setattr(improve_module, "_submit_finetune_job", _boom_submit)

    result = improve(
        cases=[_tc()], system_prompt="p", target_strain=0.10,
        method="finetune", enable_finetune=False, verbose=False,
    )
    assert result.ft_jsonl_path == "/tmp/fake.jsonl"
    assert result.ft_job_id is None
    assert len(write_calls) == 1


def test_improve_finetune_submits_job_when_enabled(monkeypatch):
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.95)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.95)]

    monkeypatch.setattr(improve_module, "_write_finetune_jsonl", lambda *a, **kw: "/tmp/fake.jsonl")
    monkeypatch.setattr(improve_module, "_submit_finetune_job", lambda *a, **kw: "job-123")

    result = improve(
        cases=[_tc()], system_prompt="p", target_strain=0.10,
        method="finetune", enable_finetune=True, verbose=False,
    )
    assert result.ft_job_id == "job-123"


def test_improve_finetune_submission_not_implemented_is_caught(monkeypatch):
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.95)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.95)]

    monkeypatch.setattr(improve_module, "_write_finetune_jsonl", lambda *a, **kw: "/tmp/fake.jsonl")

    def _raise(*a, **kw):
        raise NotImplementedError("provider not wired")

    monkeypatch.setattr(improve_module, "_submit_finetune_job", _raise)

    result = improve(  # must not raise
        cases=[_tc()], system_prompt="p", target_strain=0.10,
        method="finetune", enable_finetune=True, ft_provider="vertex", verbose=False,
    )
    assert result.ft_job_id is None


def test_improve_prompt_method_never_writes_jsonl(monkeypatch):
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.95)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.95)]

    def _boom(*a, **kw):
        raise AssertionError("prompt method must not write finetune jsonl")

    monkeypatch.setattr(improve_module, "_write_finetune_jsonl", _boom)
    result = improve(cases=[_tc()], system_prompt="p", target_strain=0.10, method="prompt", verbose=False)
    assert result.ft_jsonl_path is None


# ── improve_from_production() ───────────────────────────────────────────────

def _fake_rec(validity_gaps=(), coverage_gaps=(), confirmed=()):
    return SimpleNamespace(
        validity_gaps=list(validity_gaps),
        coverage_gaps=list(coverage_gaps),
        confirmed=list(confirmed),
    )


def test_improve_from_production_returns_none_when_nothing_derived(monkeypatch):
    monkeypatch.setattr(reconcile_module, "reconcile", lambda *a, **kw: _fake_rec())
    monkeypatch.setattr(reconcile_module, "cases_from_reconciliation", lambda *a, **kw: [])
    called = []
    monkeypatch.setattr(improve_module, "improve", lambda **kw: called.append(kw))
    result = improve_from_production(report=object(), replay_report=object(), verbose=False)
    assert result is None
    assert called == []


def test_improve_from_production_forwards_merged_cases_and_kwargs(monkeypatch):
    derived = [_tc(input="new case", name="derived")]
    monkeypatch.setattr(
        reconcile_module, "reconcile",
        lambda *a, **kw: _fake_rec(validity_gaps=[1], coverage_gaps=[2, 3]),
    )
    monkeypatch.setattr(reconcile_module, "cases_from_reconciliation", lambda *a, **kw: derived)

    captured = {}

    def _fake_improve(**kw):
        captured.update(kw)
        return "SENTINEL_RESULT"

    monkeypatch.setattr(improve_module, "improve", _fake_improve)

    result = improve_from_production(
        report=object(), replay_report=object(),
        system_prompt="sys", model="my-model", target_strain=0.15,
        verbose=False, method="finetune",
    )
    assert result == "SENTINEL_RESULT"
    assert captured["cases"] == derived
    assert captured["system_prompt"] == "sys"
    assert captured["model"] == "my-model"
    assert captured["target_strain"] == 0.15
    assert captured["method"] == "finetune"


def test_improve_from_production_base_case_wins_dedup_collision(monkeypatch):
    base = _tc(input="Same Question", canonical_answer="X", name="base-case")
    dup  = _tc(input="same question", canonical_answer="x", name="derived-dup")  # same key, case-insensitive
    unique_derived = _tc(input="different question", name="derived-unique")

    monkeypatch.setattr(reconcile_module, "reconcile", lambda *a, **kw: _fake_rec())
    monkeypatch.setattr(reconcile_module, "cases_from_reconciliation", lambda *a, **kw: [dup, unique_derived])

    captured = {}
    monkeypatch.setattr(improve_module, "improve", lambda **kw: captured.update(kw) or "R")

    improve_from_production(
        report=object(), replay_report=object(),
        base_cases=[base], verbose=False,
    )
    merged_names = [tc.name for tc in captured["cases"]]
    assert merged_names == ["base-case", "derived-unique"]


# ── _submit_finetune_job ────────────────────────────────────────────────────

def test_submit_finetune_job_rejects_non_openai_provider():
    with pytest.raises(NotImplementedError, match="vertex"):
        _submit_finetune_job("some.jsonl", "vertex", None, False)


def test_submit_finetune_job_requires_jsonl_path(monkeypatch):
    class _FakeOpenAIModule:
        class OpenAI:
            def __init__(self):
                pass

    monkeypatch.setitem(__import__("sys").modules, "openai", _FakeOpenAIModule)
    with pytest.raises(ValueError, match="no JSONL path"):
        _submit_finetune_job("", "openai", None, False)


def test_submit_finetune_job_raises_import_error_when_openai_not_installed(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="openai SDK is required"):
        _submit_finetune_job("some.jsonl", "openai", None, False)


def test_submit_finetune_job_success_uses_default_base_model(monkeypatch, tmp_path):
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text('{"messages": []}\n')

    created_files = []
    created_jobs = []

    class _FakeFiles:
        def create(self, file, purpose):
            created_files.append(purpose)
            return SimpleNamespace(id="file-abc")

    class _FakeJobs:
        def create(self, training_file, model):
            created_jobs.append((training_file, model))
            return SimpleNamespace(id="job-xyz")

    class _FakeClient:
        def __init__(self):
            self.files = _FakeFiles()
            self.fine_tuning = SimpleNamespace(jobs=_FakeJobs())

    fake_module = SimpleNamespace(OpenAI=lambda: _FakeClient())
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)

    job_id = _submit_finetune_job(str(jsonl), "openai", None, verbose=False)
    assert job_id == "job-xyz"
    assert created_files == ["fine-tune"]
    assert created_jobs[0][1] == "gpt-4o-mini-2024-07-18"  # _DEFAULT_FT_BASE_MODEL


def test_submit_finetune_job_success_uses_explicit_base_model(monkeypatch, tmp_path):
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text('{"messages": []}\n')

    created_jobs = []

    class _FakeClient:
        def __init__(self):
            self.files = SimpleNamespace(create=lambda file, purpose: SimpleNamespace(id="f1"))
            self.fine_tuning = SimpleNamespace(
                jobs=SimpleNamespace(create=lambda training_file, model: created_jobs.append(model) or SimpleNamespace(id="j1"))
            )

    fake_module = SimpleNamespace(OpenAI=lambda: _FakeClient())
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)

    job_id = _submit_finetune_job(str(jsonl), "openai", "my-base-model", verbose=True)
    assert job_id == "j1"
    assert created_jobs == ["my-base-model"]


# ── verbose=True print-branch coverage ──────────────────────────────────────
# The scenarios above all pass verbose=False to keep test output quiet; these
# mirror a few of them with verbose=True specifically to exercise the
# `if verbose: print(...)` branches, which are otherwise never hit.

def test_improve_verbose_prints_early_exit_message(capsys):
    baseline = _report([_tr(consistency_score=0.95)])
    FakeSuite.queue = [baseline]
    improve(cases=[_tc()], system_prompt="p", target_strain=0.20, verbose=True)
    out = capsys.readouterr().out
    assert "baseline already meets target" in out
    assert "train CAI Strain (baseline)" in out


def test_improve_verbose_prints_no_variants_message(capsys):
    baseline = _report([_tr(consistency_score=0.4)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = []
    improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=True)
    out = capsys.readouterr().out
    assert "variant generation failed" in out
    assert "generating 3 improved-prompt variants" in out


def test_improve_verbose_prints_holdout_and_winner_messages(capsys):
    train_baseline   = _report([_tr(consistency_score=0.3)])
    holdout_baseline = _report([_tr(consistency_score=0.3)])
    train_improved   = _report([_tr(consistency_score=0.99)])
    holdout_improved = _report([_tr(consistency_score=0.85)])
    FakeSuite.queue = [train_baseline, holdout_baseline, holdout_improved]
    FakePromptRepair.variants = [_repair_result(
        improved_prompt="winner", report=train_improved,
        original_cai_score=0.3, improved_cai_score=0.99,
    )]
    improve(
        cases=[_tc(), _tc()], system_prompt="p", target_strain=0.10,
        holdout_frac=0.5, seed=0, verbose=True,
    )
    out = capsys.readouterr().out
    assert "holdout split: train n=1, holdout n=1" in out
    assert "baseline run on holdout" in out
    assert "train winner: Strain" in out
    assert "scoring winner on holdout" in out
    assert "holdout result: Strain" in out


def test_improve_verbose_prints_truth_rejection_message(capsys):
    baseline = _report([_tr(consistency_score=0.4, truth_score=0.9, truth_strain=0.10)])
    improved_result_report = _report([_tr(consistency_score=0.99, truth_score=0.2, truth_strain=0.50)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(
        report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.99,
    )]
    improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=True)
    out = capsys.readouterr().out
    assert "REJECTED: CAI Strain fell but truth_strain rose" in out


def test_improve_verbose_finetune_prints_write_and_no_submit_message(monkeypatch, capsys):
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.95)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.95)]
    monkeypatch.setattr(improve_module, "_write_finetune_jsonl", lambda *a, **kw: "/tmp/fake.jsonl")
    improve(cases=[_tc()], system_prompt="p", target_strain=0.10, method="finetune",
            enable_finetune=False, verbose=True)
    out = capsys.readouterr().out
    assert "wrote fine-tuning pairs: /tmp/fake.jsonl" in out
    assert "pass enable_finetune=True to actually submit" in out


def test_improve_verbose_finetune_prints_submission_success_message(monkeypatch, capsys):
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.95)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.95)]
    monkeypatch.setattr(improve_module, "_write_finetune_jsonl", lambda *a, **kw: "/tmp/fake.jsonl")
    monkeypatch.setattr(improve_module, "_submit_finetune_job", lambda *a, **kw: "job-123")
    improve(cases=[_tc()], system_prompt="p", target_strain=0.10, method="finetune",
            enable_finetune=True, verbose=True)
    out = capsys.readouterr().out
    assert "fine-tuning job submitted: job-123" in out


def test_improve_verbose_finetune_prints_not_implemented_message(monkeypatch, capsys):
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.95)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.95)]
    monkeypatch.setattr(improve_module, "_write_finetune_jsonl", lambda *a, **kw: "/tmp/fake.jsonl")

    def _raise(*a, **kw):
        raise NotImplementedError("nope")

    monkeypatch.setattr(improve_module, "_submit_finetune_job", _raise)
    improve(cases=[_tc()], system_prompt="p", target_strain=0.10, method="finetune",
            enable_finetune=True, ft_provider="vertex", verbose=True)
    out = capsys.readouterr().out
    assert "fine-tune submission not yet implemented for provider=vertex" in out


def test_improve_from_production_verbose_prints_reconcile_and_nothing_to_repair(monkeypatch, capsys):
    monkeypatch.setattr(reconcile_module, "reconcile", lambda *a, **kw: _fake_rec())
    monkeypatch.setattr(reconcile_module, "cases_from_reconciliation", lambda *a, **kw: [])
    improve_from_production(report=object(), replay_report=object(), verbose=True)
    out = capsys.readouterr().out
    assert "reconcile: 0 validity gap(s), 0 coverage gap(s), 0 confirmed -> 0 new case(s)" in out
    assert "nothing to repair" in out


def test_improve_from_production_verbose_prints_repairing_over_message(monkeypatch, capsys):
    derived = [_tc(input="new case", name="derived")]
    monkeypatch.setattr(reconcile_module, "reconcile", lambda *a, **kw: _fake_rec())
    monkeypatch.setattr(reconcile_module, "cases_from_reconciliation", lambda *a, **kw: derived)
    monkeypatch.setattr(improve_module, "improve", lambda **kw: "R")
    improve_from_production(report=object(), replay_report=object(), verbose=True)
    out = capsys.readouterr().out
    assert "repairing over 1 case(s) (0 supplied + 1 from production)" in out


# ── Regression: package-attribute shadowing bug (fixed) ─────────────────────
#
# See the module-level NOTE above. contradish/__init__.py used to do
# `from .improve import improve` / `from .reconcile import reconcile`,
# whose re-exports shadowed the submodule reference on the `contradish`
# package object because the imported name was identical to its own
# submodule's name -- `contradish.improve.<anything>` (attribute-chain
# access into the submodule) raised AttributeError. Fixed by renaming the
# files (improve.py -> _improve.py, reconcile.py -> _reconcile.py,
# replay.py -> _replay.py), which removes the collision entirely: the
# function and the submodule now live at different attribute names. These
# tests now assert the fixed behavior rather than pinning the old bug.

def test_improve_attribute_is_the_function_not_shadowed():
    import contradish
    assert callable(contradish.improve)
    assert contradish.improve is improve  # the documented top-level export


def test_improve_submodule_is_reachable_at_its_own_underscore_name():
    import contradish
    assert contradish._improve is improve_module
    assert contradish._improve.improve is contradish.improve  # same function, both paths


def test_reconcile_attribute_is_the_function_not_shadowed():
    import contradish
    assert callable(contradish.reconcile)


def test_reconcile_submodule_is_reachable_at_its_own_underscore_name():
    import contradish
    assert contradish._reconcile is reconcile_module
    assert callable(contradish._reconcile.cases_from_reconciliation)


def test_improve_and_reconcile_both_importable_the_documented_way():
    # The README's documented usage: `from contradish import improve` /
    # `from contradish import reconcile`. No importlib workaround needed --
    # this is exactly what broke before the fix (it used to require
    # importlib.import_module to dodge the shadowing on the submodule
    # side, though the top-level function import itself was never broken).
    from contradish import improve as improve_fn
    from contradish import cases_from_reconciliation
    assert callable(improve_fn)
    assert callable(cases_from_reconciliation)


# ── ImprovementResult.summary() / to_dict(): distinctions gate ─────────────

def test_summary_shows_rejected_when_distinctions_regressed():
    r = ImprovementResult(
        baseline_strain=0.5, improved_strain=0.05, strain_delta=-0.45,
        target_strain=0.10, target_met=False, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
        distinction_diff={"newly_collapsed": ["x"], "summary": "s"}, distinctions_regressed=True,
    )
    out = r.summary()
    assert "REJECTED" in out
    assert "x" in out


def test_summary_shows_distinctions_summary_when_not_regressed():
    r = ImprovementResult(
        baseline_strain=0.5, improved_strain=0.2, strain_delta=-0.3,
        target_strain=0.25, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
        distinction_diff={
            "newly_collapsed": [],
            "summary": "baseline_prompt vs improved_prompt  (medication)  pairs=1  regressed=0  newly_collapsed=0",
        },
        distinctions_regressed=False,
    )
    out = r.summary()
    assert "pairs=1" in out


def test_summary_omits_distinctions_section_when_none():
    r = ImprovementResult(
        baseline_strain=0.5, improved_strain=0.2, strain_delta=-0.3,
        target_strain=0.25, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
    )
    assert "collapsed" not in r.summary()


def test_to_dict_includes_distinction_fields():
    r = ImprovementResult(
        baseline_strain=0.5, improved_strain=0.2, strain_delta=-0.3,
        target_strain=0.25, target_met=True, method="prompt",
        baseline_prompt="a", improved_prompt="b",
        baseline_report=_report([_tr()]), improved_report=_report([_tr()]),
        distinction_diff={"newly_collapsed": [], "summary": "s"}, distinctions_regressed=False,
    )
    d = r.to_dict()
    assert d["distinction_diff"] == {"newly_collapsed": [], "summary": "s"}
    assert d["distinctions_regressed"] is False


# ── improve(): distinctions gate ────────────────────────────────────────────
#
# FakeDistinctionProber mirrors FakeSuite/FakePromptRepair above: queue-driven,
# each .measure() call pops the next to_dict()-shaped payload off the class
# queue, in the order improve()'s distinction gate calls it (baseline prompt
# measured first, then the improved prompt). This tests the gate's wiring
# (does it call DistinctionProber twice, diff the results, and force
# target_met False on a collapse) without running real pressure-framing
# probes -- diff_distinction_reports() itself is already covered directly in
# tests/test_distinction.py.

class FakeDistinctionProber:
    queue: list = []
    instances: list = []

    def __init__(self, model_fn, pairs, commitment_extractor, domain="general", **kw):
        self.model_fn = model_fn
        self.pairs = pairs
        self.domain = domain
        FakeDistinctionProber.instances.append(self)

    def measure(self, verbose=True):
        data = FakeDistinctionProber.queue.pop(0)
        return SimpleNamespace(to_dict=lambda: data)


def _dist_payload(domain="medication", hold_rates=None):
    hold_rates = hold_rates or {}
    return {
        "domain": domain,
        "profiles": {
            pid: {"overall_hold_rate": rate, "description": f"{pid} description"}
            for pid, rate in hold_rates.items()
        },
    }


def test_improve_distinctions_gate_rejects_when_a_distinction_collapses(monkeypatch):
    monkeypatch.setattr("contradish.distinction.DistinctionProber", FakeDistinctionProber)
    FakeDistinctionProber.queue = [
        _dist_payload(hold_rates={"schedule_ii_vs_routine_refill": 0.9}),  # baseline: held
        _dist_payload(hold_rates={"schedule_ii_vs_routine_refill": 0.2}),  # improved: collapsed
    ]
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.99)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(
        report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.99,
    )]
    result = improve(
        cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False,
        distinctions="medication",
    )
    # CAI Strain target would be met (~0.01 <= 0.10), but a distinction that
    # held in baseline collapsed in the improved prompt -> rejected.
    assert result.distinctions_regressed is True
    assert result.target_met is False
    assert result.distinction_diff["newly_collapsed"] == ["schedule_ii_vs_routine_refill"]
    assert len(FakeDistinctionProber.instances) == 2


def test_improve_distinctions_gate_allows_when_nothing_collapses(monkeypatch):
    monkeypatch.setattr("contradish.distinction.DistinctionProber", FakeDistinctionProber)
    FakeDistinctionProber.queue = [
        _dist_payload(hold_rates={"a": 0.9}),
        _dist_payload(hold_rates={"a": 0.85}),  # ordinary drop, not a collapse
    ]
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.99)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(
        report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.99,
    )]
    result = improve(
        cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False,
        distinctions="medication",
    )
    assert result.distinctions_regressed is False
    assert result.target_met is True
    assert result.distinction_diff is not None


def test_improve_no_distinctions_gate_when_domain_not_passed():
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.99)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(
        report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.99,
    )]
    result = improve(cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=False)
    assert result.distinction_diff is None
    assert result.distinctions_regressed is False
    assert result.target_met is True


def test_improve_unknown_distinctions_domain_warns_and_skips(capsys):
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.99)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(
        report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.99,
    )]
    result = improve(
        cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=True,
        distinctions="not_a_real_domain",
    )
    out = capsys.readouterr().out
    assert "no built-in distinction pairs for domain" in out
    assert result.distinction_diff is None
    assert result.distinctions_regressed is False
    assert result.target_met is True


def test_improve_verbose_prints_distinctions_rejection_message(monkeypatch, capsys):
    monkeypatch.setattr("contradish.distinction.DistinctionProber", FakeDistinctionProber)
    FakeDistinctionProber.queue = [
        _dist_payload(hold_rates={"a": 0.9}),
        _dist_payload(hold_rates={"a": 0.1}),
    ]
    baseline = _report([_tr(consistency_score=0.4)])
    improved_result_report = _report([_tr(consistency_score=0.99)])
    FakeSuite.queue = [baseline]
    FakePromptRepair.variants = [_repair_result(
        report=improved_result_report, original_cai_score=0.4, improved_cai_score=0.99,
    )]
    improve(
        cases=[_tc()], system_prompt="p", target_strain=0.10, verbose=True,
        distinctions="medication",
    )
    out = capsys.readouterr().out
    assert "REJECTED: CAI Strain fell but" in out
    assert "distinction(s) that held in the baseline prompt collapsed" in out
