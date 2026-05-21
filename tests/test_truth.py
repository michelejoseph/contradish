"""
Tests for the truth axis: truth_score on TestResult/Report, the confident_wrong
finding, and the truth-gate in improve().
Run with: pytest tests/test_truth.py
No API key required.
"""
import sys

from contradish.models import Report, TestResult, TestCase, RiskLevel
from contradish.findings import findings_from


def _result(name, cai_strain, truth_strain=None, canonical=None):
    tc = TestCase(input=name, name=name, canonical_answer=canonical)
    return TestResult(
        test_case=tc, paraphrases=[], outputs=[],
        consistency_score=1.0 - cai_strain, contradiction_score=0.0,
        risk=RiskLevel.LOW,
        truth_score=(None if truth_strain is None else round(1.0 - truth_strain, 3)),
        truth_strain=truth_strain,
    )


def test_report_truth_strain_aggregates_only_scored_cases():
    rep = Report(results=[
        _result("a", 0.1, truth_strain=0.8, canonical="x"),
        _result("b", 0.1, truth_strain=0.6, canonical="x"),
        _result("c", 0.1),  # no canonical, no truth
    ])
    # mean of 0.8 and 0.6
    assert rep.truth_strain == 0.7
    # 2 of 3 cases were truth-scored
    assert rep.truth_coverage == round(2 / 3, 3)


def test_report_truth_strain_none_when_no_canonicals():
    rep = Report(results=[_result("a", 0.1), _result("b", 0.2)])
    assert rep.truth_strain is None
    assert rep.truth_coverage == 0.0


def test_confident_wrong_finding_fires():
    # 4 cases: highly consistent (low CAI strain) but very wrong (high truth strain).
    rep = Report(results=[
        _result(f"c{i}", cai_strain=0.05, truth_strain=0.9, canonical="x")
        for i in range(4)
    ])
    fs = findings_from(rep)
    types = [f.type for f in fs]
    assert "confident_wrong" in types
    cw = next(f for f in fs if f.type == "confident_wrong")
    assert "finetune" in (cw.cli_hint or "")


def test_confident_wrong_does_not_fire_without_canonicals():
    rep = Report(results=[
        _result(f"c{i}", cai_strain=0.05) for i in range(4)
    ])
    types = [f.type for f in findings_from(rep)]
    assert "confident_wrong" not in types


def test_confident_wrong_does_not_fire_when_truthful():
    # Consistent AND correct: low cai strain, low truth strain -> no finding.
    rep = Report(results=[
        _result(f"c{i}", cai_strain=0.05, truth_strain=0.05, canonical="x")
        for i in range(4)
    ])
    types = [f.type for f in findings_from(rep)]
    assert "confident_wrong" not in types


def test_truth_gate_rejects_consistency_win_that_lost_truth():
    imp = sys.modules["contradish.improve"]
    suite_mod = sys.modules["contradish.suite"]
    repair_mod = sys.modules["contradish.repair"]

    def mk(cai, truth):
        return Report(results=[_result("c", cai, truth_strain=truth, canonical="x")])

    class FakeSuite:
        def __init__(self, app=None, **kw):
            self.app = app
        def add(self, tc):
            pass
        def run(self, paraphrases=5, verbose=False, concurrency=4):
            out = self.app("__probe__")
            # improved prompt: consistent (low cai) but WRONG (high truth strain)
            return mk(0.05, 0.80) if "improved" in out else mk(0.50, 0.10)

    class FakeVariant:
        improved_prompt = "improved"
        improved_cai_score = 0.95
        original_cai_score = 0.50
        strain_delta = -0.45
        rank = 1
        report = mk(0.05, 0.80)

    class FakeRepair:
        def __init__(self, *a, **kw):
            pass
        def fix(self, **kw):
            return [FakeVariant()]

    def fake_make_app(system_prompt, **kw):
        lbl = "improved" if "improved" in (system_prompt or "").lower() else "baseline"
        return lambda q: f"{lbl}::{q}"

    orig_suite = suite_mod.Suite
    orig_repair = repair_mod.PromptRepair
    orig_mk = imp._make_app_for_prompt
    suite_mod.Suite = FakeSuite
    repair_mod.PromptRepair = FakeRepair
    imp._make_app_for_prompt = fake_make_app
    try:
        from contradish import improve
        result = improve(
            cases=[TestCase(input="max dose?", name="c0", canonical_answer="1200mg")],
            system_prompt="be helpful", model="x", target_strain=0.20, verbose=False,
        )
        # CAI strain target was hit...
        assert result.improved_strain <= 0.20
        # ...but truth regressed, so the win is rejected.
        assert result.truth_regressed is True
        assert result.target_met is False
        assert "REJECTED" in result.summary()
    finally:
        suite_mod.Suite = orig_suite
        repair_mod.PromptRepair = orig_repair
        imp._make_app_for_prompt = orig_mk


if __name__ == "__main__":
    import contradish  # ensure submodules registered
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
