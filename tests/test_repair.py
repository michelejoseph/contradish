"""
Full test coverage for contradish/repair.py -- PromptRepair, the "generate
N improved prompt variants, test each, rank by CAI Strain reduction" step
of the repair loop.

Unlike improve.py, repair.py imports LLMClient and Suite at module level
(`from .llm import LLMClient`, `from .suite import Suite`), so these tests
patch `contradish.repair.LLMClient` / `contradish.repair.Suite` directly --
no function-local-import indirection to route around here.

FakeSuite resolves which canned Report to hand back by looking at which
prompt its app was built for (app_factory tags the app with `.prompt`),
rather than a call-order queue -- this makes the parallel (concurrency>1,
ThreadPoolExecutor) path deterministic to assert on regardless of thread
scheduling.
"""
import contradish.repair as repair_module
import pytest

from contradish.models import ContradictionPair, Report, TestCase, TestResult
from contradish.repair import PromptRepair, _REPAIR_PROMPT


# ── Fixture builders ────────────────────────────────────────────────────────

def _tc(input="a test question", name=None, **kw):
    return TestCase(input=input, name=name, **kw)


def _tr(tc=None, consistency_score=0.9, unstable_patterns=None, suggestion=None, **kw):
    return TestResult(
        test_case=tc or _tc(),
        paraphrases=["p1"],
        outputs=["o0", "o1"],
        consistency_score=consistency_score,
        contradiction_score=0.0,
        unstable_patterns=unstable_patterns or [],
        suggestion=suggestion,
        **kw,
    )


def _report(results):
    return Report(results=results)


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakeLLM:
    """Records the prompt it was called with; behavior configured per-test
    via the `response` / `raises` class attributes."""
    response = None
    raises = None
    last_prompt = None

    def __init__(self, *a, **kw):
        self.init_kwargs = kw
        FakeLLM.last_init_kwargs = kw

    def complete_json(self, prompt, model=None):
        FakeLLM.last_prompt = prompt
        if FakeLLM.raises is not None:
            raise FakeLLM.raises
        return FakeLLM.response


def _fake_app_factory(prompt):
    def app(question):
        return f"reply to {question!r} for {prompt!r}"
    app.prompt = prompt
    return app


class FakeSuite:
    """Resolves its canned Report by the prompt its app was built for
    (via REPORTS_BY_PROMPT), not by call order -- safe under concurrency."""
    REPORTS_BY_PROMPT: dict = {}
    instances: list = []

    def __init__(self, app, **kw):
        self.app = app
        self.kw = kw
        self.cases = []
        FakeSuite.instances.append(self)

    def add(self, tc):
        self.cases.append(tc)

    def run(self, paraphrases=5, verbose=True, concurrency=4):
        return FakeSuite.REPORTS_BY_PROMPT[self.app.prompt]


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch):
    FakeLLM.response = None
    FakeLLM.raises = None
    FakeLLM.last_prompt = None
    FakeSuite.REPORTS_BY_PROMPT = {}
    FakeSuite.instances = []
    monkeypatch.setattr(repair_module, "LLMClient", FakeLLM)
    monkeypatch.setattr(repair_module, "Suite", FakeSuite)


# ── PromptRepair.__init__ ───────────────────────────────────────────────────

def test_init_defaults_n_to_three():
    r = PromptRepair()
    assert r.n == 3


def test_init_forwards_api_key_and_provider_to_llm_client():
    PromptRepair(api_key="k", provider="openai", n=5)
    assert FakeLLM.last_init_kwargs == {"api_key": "k", "provider": "openai"}


# ── _generate_variants ──────────────────────────────────────────────────────

def test_generate_variants_returns_list_from_llm():
    FakeLLM.response = ["prompt A", "prompt B"]
    r = PromptRepair(n=3)
    out = r._generate_variants("orig", "failures text")
    assert out == ["prompt A", "prompt B"]


def test_generate_variants_truncates_to_n():
    FakeLLM.response = ["a", "b", "c", "d"]
    r = PromptRepair(n=2)
    assert r._generate_variants("orig", "f") == ["a", "b"]


def test_generate_variants_drops_blank_and_whitespace_entries():
    FakeLLM.response = ["good one", "  ", "", "also good"]
    r = PromptRepair(n=10)
    assert r._generate_variants("orig", "f") == ["good one", "also good"]


def test_generate_variants_coerces_non_string_items():
    FakeLLM.response = [123, "text"]
    r = PromptRepair(n=10)
    assert r._generate_variants("orig", "f") == ["123", "text"]


def test_generate_variants_empty_when_llm_returns_non_list():
    FakeLLM.response = {"not": "a list"}
    r = PromptRepair()
    assert r._generate_variants("orig", "f") == []


def test_generate_variants_empty_when_llm_raises():
    FakeLLM.raises = RuntimeError("boom")
    r = PromptRepair()
    assert r._generate_variants("orig", "f") == []


def test_generate_variants_prompt_includes_system_prompt_failures_and_n():
    FakeLLM.response = ["x"]
    r = PromptRepair(n=7)
    r._generate_variants("MY SYSTEM PROMPT", "MY FAILURES TEXT")
    assert "MY SYSTEM PROMPT" in FakeLLM.last_prompt
    assert "MY FAILURES TEXT" in FakeLLM.last_prompt
    assert "7 improved versions" in FakeLLM.last_prompt


def test_generate_variants_truncates_long_system_prompt_to_2000_chars():
    FakeLLM.response = ["x"]
    r = PromptRepair(n=1)
    long_prompt = "A" * 3000
    r._generate_variants(long_prompt, "f")
    # Only the first 2000 chars of the system prompt should appear.
    assert "A" * 2000 in FakeLLM.last_prompt
    assert "A" * 2001 not in FakeLLM.last_prompt


# ── _format_failures ─────────────────────────────────────────────────────────

def test_format_failures_no_failures_message():
    report = _report([_tr(consistency_score=0.95)])  # passes -> not in .failed
    assert PromptRepair._format_failures(report) == "No failures detected (prompt may already be stable)."


def test_format_failures_lists_rule_name_and_strain():
    failed = _tr(tc=_tc(name="refund rule"), consistency_score=0.4)
    out = PromptRepair._format_failures(_report([failed]))
    assert "Rule: refund rule" in out
    assert "CAI Strain: 0.60" in out  # 1 - 0.4


def test_format_failures_defaults_strain_to_one_when_cai_strain_none():
    # cai_strain is None only when consistency_score is None.
    failed = TestResult(
        test_case=_tc(name="unscored rule"), paraphrases=["p"], outputs=["o0", "o1"],
        consistency_score=None, contradiction_score=0.9,  # fails via contradiction_score
    )
    out = PromptRepair._format_failures(_report([failed]))
    assert "CAI Strain: 1.00" in out


def test_format_failures_includes_pattern_line_when_present():
    failed = _tr(consistency_score=0.4, unstable_patterns=["numeric drift", "second pattern"])
    out = PromptRepair._format_failures(_report([failed]))
    assert "Pattern: numeric drift" in out
    assert "second pattern" not in out  # only the first pattern is shown


def test_format_failures_omits_pattern_line_when_absent():
    failed = _tr(consistency_score=0.4, unstable_patterns=[])
    out = PromptRepair._format_failures(_report([failed]))
    assert "Pattern:" not in out


def test_format_failures_includes_fix_hint_when_present():
    failed = _tr(consistency_score=0.4, suggestion="anchor the number")
    out = PromptRepair._format_failures(_report([failed]))
    assert "Fix hint: anchor the number" in out


def test_format_failures_omits_fix_hint_when_absent():
    failed = _tr(consistency_score=0.4, suggestion=None)
    out = PromptRepair._format_failures(_report([failed]))
    assert "Fix hint:" not in out


def test_format_failures_multiple_failures_all_listed():
    f1 = _tr(tc=_tc(name="rule one"), consistency_score=0.4)
    f2 = _tr(tc=_tc(name="rule two"), consistency_score=0.3)
    out = PromptRepair._format_failures(_report([f1, f2]))
    assert "Rule: rule one" in out
    assert "Rule: rule two" in out


# ── fix(): variant generation failure ───────────────────────────────────────

def test_fix_returns_empty_when_no_variants_generated(capsys):
    FakeLLM.response = None  # not a list -> _generate_variants returns []
    report = _report([_tr(consistency_score=0.4)])
    r = PromptRepair(n=3)
    out = r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory, verbose=True)
    assert out == []
    printed = capsys.readouterr().out
    assert "variant generation failed" in printed


def test_fix_returns_empty_silently_when_verbose_false():
    FakeLLM.response = None
    report = _report([_tr(consistency_score=0.4)])
    r = PromptRepair(n=3)
    out = r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory, verbose=False)
    assert out == []


# ── fix(): sequential path (concurrency<=1 or a single variant) ────────────

def test_fix_single_variant_sequential_path_builds_correct_result():
    FakeLLM.response = ["improved prompt"]
    tc = _tc(name="rule a")
    report = _report([_tr(tc=tc, consistency_score=0.4)])   # original_cai_score 0.4
    improved_report = _report([_tr(tc=tc, consistency_score=0.9)])  # improved_cai_score 0.9
    FakeSuite.REPORTS_BY_PROMPT = {"improved prompt": improved_report}

    r = PromptRepair(n=1)
    results = r.fix(system_prompt="orig prompt", report=report,
                     app_factory=_fake_app_factory, verbose=False)

    assert len(results) == 1
    res = results[0]
    assert res.original_prompt == "orig prompt"
    assert res.improved_prompt == "improved prompt"
    assert res.original_cai_score == 0.4
    assert res.improved_cai_score == 0.9
    assert res.delta == round(0.9 - 0.4, 3)
    assert res.rank == 1
    assert res.report is improved_report


def test_fix_uses_default_retest_cases_from_report_when_cases_none():
    FakeLLM.response = ["v1"]
    tc = _tc(name="rule a")
    report = _report([_tr(tc=tc, consistency_score=0.4)])
    FakeSuite.REPORTS_BY_PROMPT = {"v1": _report([_tr(tc=tc, consistency_score=0.9)])}

    r = PromptRepair(n=1)
    r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory, verbose=False)

    assert len(FakeSuite.instances) == 1
    assert FakeSuite.instances[0].cases == [tc]


def test_fix_uses_explicit_cases_param_over_report_cases():
    FakeLLM.response = ["v1"]
    report_tc = _tc(name="report case")
    explicit_tc = _tc(name="explicit case")
    report = _report([_tr(tc=report_tc, consistency_score=0.4)])
    FakeSuite.REPORTS_BY_PROMPT = {"v1": _report([_tr(tc=report_tc, consistency_score=0.9)])}

    r = PromptRepair(n=1)
    r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory,
          cases=[explicit_tc], verbose=False)

    assert FakeSuite.instances[0].cases == [explicit_tc]


def test_fix_sequential_path_used_when_concurrency_is_one():
    FakeLLM.response = ["v1", "v2"]
    tc = _tc()
    report = _report([_tr(tc=tc, consistency_score=0.4)])
    FakeSuite.REPORTS_BY_PROMPT = {
        "v1": _report([_tr(tc=tc, consistency_score=0.6)]),
        "v2": _report([_tr(tc=tc, consistency_score=0.9)]),
    }
    r = PromptRepair(n=2)
    results = r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory,
                     concurrency=1, verbose=False)
    assert len(results) == 2
    assert len(FakeSuite.instances) == 2


def test_fix_original_cai_score_zero_when_report_cai_score_none():
    FakeLLM.response = ["v1"]
    report = _report([])  # cai_score None -> original_cai defaults to 0.0
    FakeSuite.REPORTS_BY_PROMPT = {"v1": _report([_tr(consistency_score=0.5)])}
    r = PromptRepair(n=1)
    results = r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory, verbose=False)
    assert results[0].original_cai_score == 0.0


def test_fix_improved_cai_score_zero_when_new_report_cai_score_none():
    FakeLLM.response = ["v1"]
    tc = _tc()
    report = _report([_tr(tc=tc, consistency_score=0.4)])
    FakeSuite.REPORTS_BY_PROMPT = {"v1": _report([])}  # improved cai_score -> None -> 0.0
    r = PromptRepair(n=1)
    results = r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory, verbose=False)
    assert results[0].improved_cai_score == 0.0


# ── fix(): parallel path (concurrency>1 and multiple variants) ─────────────

def test_fix_parallel_path_scores_all_variants_and_ranks_best_first():
    FakeLLM.response = ["worst", "mid", "best"]
    tc = _tc()
    report = _report([_tr(tc=tc, consistency_score=0.4)])
    FakeSuite.REPORTS_BY_PROMPT = {
        "worst": _report([_tr(tc=tc, consistency_score=0.3)]),
        "mid":   _report([_tr(tc=tc, consistency_score=0.6)]),
        "best":  _report([_tr(tc=tc, consistency_score=0.95)]),
    }
    r = PromptRepair(n=3)
    results = r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory,
                     concurrency=4, verbose=False)

    assert len(results) == 3
    assert [res.improved_prompt for res in results] == ["best", "mid", "worst"]
    assert [res.rank for res in results] == [1, 2, 3]


def test_fix_parallel_path_verbose_prints_progress(capsys):
    FakeLLM.response = ["a", "b"]
    tc = _tc()
    report = _report([_tr(tc=tc, consistency_score=0.4)])
    FakeSuite.REPORTS_BY_PROMPT = {
        "a": _report([_tr(tc=tc, consistency_score=0.5)]),
        "b": _report([_tr(tc=tc, consistency_score=0.7)]),
    }
    r = PromptRepair(n=2)
    r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory,
          concurrency=4, verbose=True)
    out = capsys.readouterr().out
    assert "original CAI Strain" in out
    assert "generating 2 improved prompt variants" in out
    assert "testing 2 variants in parallel" in out
    assert "done" in out
    assert "Prompt repair results:" in out
    assert "#1: CAI Strain" in out


def test_fix_sequential_path_verbose_prints_per_variant_progress(capsys):
    FakeLLM.response = ["only one"]
    tc = _tc()
    report = _report([_tr(tc=tc, consistency_score=0.4)])
    FakeSuite.REPORTS_BY_PROMPT = {"only one": _report([_tr(tc=tc, consistency_score=0.8)])}
    r = PromptRepair(n=1)
    r.fix(system_prompt="p", report=report, app_factory=_fake_app_factory,
          concurrency=1, verbose=True)
    out = capsys.readouterr().out
    assert "testing variant 1/1" in out
