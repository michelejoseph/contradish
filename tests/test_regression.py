"""
Full test coverage for contradish/regression.py (RegressionSuite, the
baseline-vs-candidate CI/CD gate) and contradish.models.RegressionResult,
the object RegressionSuite.compare() returns. The two are tested together
since they're tightly coupled -- compare()'s whole job is to produce a
RegressionResult, and RegressionResult had essentially no dedicated
coverage anywhere else in the suite.

regression.py imports Suite at module level (`from .suite import Suite`),
so these tests patch `contradish.regression.Suite` directly.
"""
import json
import sys

import pytest

from contradish.models import Report, RegressionResult, TestCase, TestResult
from contradish.regression import RegressionSuite


# ── Fixture builders ────────────────────────────────────────────────────────

def _tc(input="a test question", name=None, **kw):
    return TestCase(input=input, name=name, **kw)


def _tr(tc=None, consistency_score=0.9, **kw):
    return TestResult(
        test_case=tc or _tc(),
        paraphrases=["p1"],
        outputs=["o0", "o1"],
        consistency_score=consistency_score,
        contradiction_score=0.0,
        **kw,
    )


def _report(results):
    return Report(results=results)


# ── Fake Suite ───────────────────────────────────────────────────────────────

class FakeSuite:
    """Resolves its canned Report by which app callable it was built with."""
    REPORTS_BY_APP: dict = {}
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
        return FakeSuite.REPORTS_BY_APP[self.app]


@pytest.fixture(autouse=True)
def _reset_fake_suite(monkeypatch):
    FakeSuite.REPORTS_BY_APP = {}
    FakeSuite.instances = []
    monkeypatch.setattr("contradish.regression.Suite", FakeSuite)


def _app(tag):
    """A distinguishable, hashable app callable to key FakeSuite's report map."""
    def app(question):
        return f"{tag} reply"
    app.tag = tag
    return app


# ── RegressionSuite.__init__ ────────────────────────────────────────────────

def test_init_stores_test_cases_api_key_and_provider():
    cases = [_tc()]
    s = RegressionSuite(test_cases=cases, api_key="k", provider="anthropic")
    assert s.test_cases is cases
    assert s.api_key == "k"
    assert s.provider == "anthropic"


# ── RegressionSuite.load() ──────────────────────────────────────────────────

def test_load_json_list_format(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"input": "q1", "name": "case one"},
        {"input": "q2"},
    ]))
    s = RegressionSuite.load(str(path))
    assert len(s.test_cases) == 2
    assert s.test_cases[0].input == "q1"
    assert s.test_cases[0].name == "case one"
    assert s.test_cases[1].name == "q2"  # auto-generated from input


def test_load_json_dict_wrapped_format(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"test_cases": [{"input": "q1"}]}))
    s = RegressionSuite.load(str(path))
    assert len(s.test_cases) == 1
    assert s.test_cases[0].input == "q1"


def test_load_json_dict_without_test_cases_key_yields_empty(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"something_else": []}))
    s = RegressionSuite.load(str(path))
    assert s.test_cases == []


def test_load_maps_expected_traits(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"input": "q1", "expected_traits": ["a", "b"]}]))
    s = RegressionSuite.load(str(path))
    assert s.test_cases[0].expected_traits == ["a", "b"]


def test_load_defaults_expected_traits_to_empty_list(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"input": "q1"}]))
    s = RegressionSuite.load(str(path))
    assert s.test_cases[0].expected_traits == []


def test_load_passes_api_key_through(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"input": "q1"}]))
    s = RegressionSuite.load(str(path), api_key="secret")
    assert s.api_key == "secret"


def test_load_yaml_list_format(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("- input: q1\n  name: case one\n- input: q2\n")
    pytest.importorskip("yaml")
    s = RegressionSuite.load(str(path))
    assert len(s.test_cases) == 2
    assert s.test_cases[0].name == "case one"


def test_load_yaml_dict_wrapped_format(tmp_path):
    path = tmp_path / "cases.yml"
    path.write_text("test_cases:\n  - input: q1\n")
    pytest.importorskip("yaml")
    s = RegressionSuite.load(str(path))
    assert len(s.test_cases) == 1
    assert s.test_cases[0].input == "q1"


def test_load_yaml_raises_import_error_when_pyyaml_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    path = tmp_path / "cases.yaml"
    path.write_text("- input: q1\n")
    with pytest.raises(ImportError, match="Install pyyaml"):
        RegressionSuite.load(str(path))


# ── RegressionSuite.compare() ───────────────────────────────────────────────

def test_compare_runs_both_apps_and_returns_regression_result():
    baseline_app = _app("baseline")
    candidate_app = _app("candidate")
    baseline_report = _report([_tr(consistency_score=0.9)])
    candidate_report = _report([_tr(consistency_score=0.7)])
    FakeSuite.REPORTS_BY_APP = {baseline_app: baseline_report, candidate_app: candidate_report}

    s = RegressionSuite(test_cases=[_tc()], api_key="k", provider="anthropic")
    result = s.compare(baseline_app, candidate_app,
                        baseline_label="prod", candidate_label="pr-123", verbose=False)

    assert isinstance(result, RegressionResult)
    assert result.baseline_label == "prod"
    assert result.candidate_label == "pr-123"
    assert result.baseline_report is baseline_report
    assert result.candidate_report is candidate_report


def test_compare_default_labels():
    baseline_app, candidate_app = _app("b"), _app("c")
    FakeSuite.REPORTS_BY_APP = {baseline_app: _report([_tr()]), candidate_app: _report([_tr()])}
    s = RegressionSuite(test_cases=[_tc()])
    result = s.compare(baseline_app, candidate_app, verbose=False)
    assert result.baseline_label == "baseline"
    assert result.candidate_label == "candidate"


def test_compare_adds_all_test_cases_to_both_suites():
    baseline_app, candidate_app = _app("b"), _app("c")
    FakeSuite.REPORTS_BY_APP = {baseline_app: _report([_tr()]), candidate_app: _report([_tr()])}
    cases = [_tc(name="c1"), _tc(name="c2")]
    s = RegressionSuite(test_cases=cases)
    s.compare(baseline_app, candidate_app, verbose=False)
    assert len(FakeSuite.instances) == 2
    assert FakeSuite.instances[0].cases == cases
    assert FakeSuite.instances[1].cases == cases


def test_compare_forwards_api_key_and_provider_to_both_suites():
    baseline_app, candidate_app = _app("b"), _app("c")
    FakeSuite.REPORTS_BY_APP = {baseline_app: _report([_tr()]), candidate_app: _report([_tr()])}
    s = RegressionSuite(test_cases=[_tc()], api_key="k", provider="openai")
    s.compare(baseline_app, candidate_app, verbose=False)
    for inst in FakeSuite.instances:
        assert inst.api_key == "k"
        assert inst.provider == "openai"


def test_compare_verbose_prints_progress(capsys):
    baseline_app, candidate_app = _app("b"), _app("c")
    FakeSuite.REPORTS_BY_APP = {baseline_app: _report([_tr()]), candidate_app: _report([_tr()])}
    s = RegressionSuite(test_cases=[_tc()])
    s.compare(baseline_app, candidate_app,
              baseline_label="prod", candidate_label="pr-9", verbose=True)
    out = capsys.readouterr().out
    assert "Running baseline (prod)" in out
    assert "Running candidate (pr-9)" in out


# ── RegressionResult ─────────────────────────────────────────────────────────

def _rr(baseline_results, candidate_results, **kw):
    return RegressionResult(
        baseline_label="prod", candidate_label="pr",
        baseline_report=_report(baseline_results),
        candidate_report=_report(candidate_results),
        **kw,
    )


def test_cai_delta_positive_means_improvement():
    r = _rr([_tr(consistency_score=0.6)], [_tr(consistency_score=0.9)])
    assert r.cai_delta == round(0.9 - 0.6, 3)


def test_cai_delta_none_when_either_report_unscored():
    r = _rr([], [_tr(consistency_score=0.9)])
    assert r.cai_delta is None


def test_strain_delta_negative_means_improvement():
    r = _rr([_tr(consistency_score=0.6)], [_tr(consistency_score=0.9)])  # strain 0.4 -> 0.1
    assert r.strain_delta == round(0.1 - 0.4, 3)
    assert r.strain_delta < 0


def test_strain_delta_none_when_either_report_unscored():
    r = _rr([], [_tr(consistency_score=0.9)])
    assert r.strain_delta is None


def test_regressed_true_when_strain_rose():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.6)])  # strain rose
    assert r.regressed is True


def test_regressed_false_when_strain_fell_or_equal():
    improved = _rr([_tr(consistency_score=0.6)], [_tr(consistency_score=0.9)])
    assert improved.regressed is False
    same = _rr([_tr(consistency_score=0.6)], [_tr(consistency_score=0.6)])
    assert same.regressed is False


def test_regressed_false_when_delta_none():
    r = _rr([], [_tr(consistency_score=0.9)])
    assert r.regressed is False


def test_per_case_deltas_matches_common_case_by_name():
    tc = _tc(name="shared case")
    r = _rr([_tr(tc=tc, consistency_score=0.9)], [_tr(tc=tc, consistency_score=0.6)])
    rows = r.per_case_deltas
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "shared case"
    assert row["baseline_strain"] == 0.1
    assert row["candidate_strain"] == 0.4
    assert row["delta"] == round(0.4 - 0.1, 3)
    assert row["regressed"] is True


def test_per_case_deltas_includes_case_only_in_baseline():
    tc = _tc(name="baseline only")
    r = _rr([_tr(tc=tc, consistency_score=0.9)], [])
    rows = r.per_case_deltas
    assert rows == [{
        "name": "baseline only", "baseline_strain": 0.1,
        "candidate_strain": None, "delta": None, "regressed": False,
    }]


def test_per_case_deltas_includes_case_only_in_candidate():
    tc = _tc(name="candidate only")
    r = _rr([], [_tr(tc=tc, consistency_score=0.9)])
    rows = r.per_case_deltas
    assert rows == [{
        "name": "candidate only", "baseline_strain": None,
        "candidate_strain": 0.1, "delta": None, "regressed": False,
    }]


def test_per_case_deltas_preserves_baseline_order_then_appends_new():
    tc1, tc2 = _tc(name="first"), _tc(name="second")
    tc3 = _tc(name="third")
    r = _rr(
        [_tr(tc=tc1, consistency_score=0.9), _tr(tc=tc2, consistency_score=0.8)],
        [_tr(tc=tc2, consistency_score=0.8), _tr(tc=tc3, consistency_score=0.7)],
    )
    names = [row["name"] for row in r.per_case_deltas]
    assert names == ["first", "second", "third"]


def test_per_case_deltas_falls_back_to_input_prefix_when_name_blank():
    tc = _tc(input="this is the input text used as a fallback name")
    tc.name = ""  # force the blank-name edge case (TestCase normally auto-names)
    r = _rr([_tr(tc=tc, consistency_score=0.9)], [])
    assert r.per_case_deltas[0]["name"] == "this is the input text used as a fallback name"[:60]


def test_summary_shows_na_and_question_mark_when_delta_none():
    r = _rr([], [_tr(consistency_score=0.9)])
    out = r.summary()
    assert "n/a" in out
    assert "?" in out


def test_summary_shows_up_arrow_on_regression():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.6)])
    out = r.summary()
    assert "↑" in out
    assert "regressions=1" in out


def test_summary_shows_down_arrow_on_improvement():
    r = _rr([_tr(consistency_score=0.6)], [_tr(consistency_score=0.9)])
    out = r.summary()
    assert "↓" in out
    assert "regressions=0" in out


def test_summary_shows_equals_when_strain_unchanged():
    r = _rr([_tr(consistency_score=0.6)], [_tr(consistency_score=0.6)])
    out = r.summary()
    assert "=" in out


def test_summary_includes_labels_and_scores():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.9)])
    out = r.summary()
    assert "prod" in out
    assert "pr" in out


def test_fail_if_above_does_not_raise_within_threshold():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.9)])  # strain 0.1
    r.fail_if_above(strain=0.25)  # must not raise


def test_fail_if_above_raises_when_over_threshold():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.5)])  # strain 0.5
    with pytest.raises(AssertionError, match="CAI regression"):
        r.fail_if_above(strain=0.25)


def test_fail_if_above_message_includes_labels_and_delta():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.5)])
    with pytest.raises(AssertionError) as exc_info:
        r.fail_if_above(strain=0.25)
    msg = str(exc_info.value)
    assert "pr" in msg
    assert "prod" in msg
    assert "Delta:" in msg


def test_fail_if_above_does_not_raise_when_candidate_unscored():
    r = _rr([_tr(consistency_score=0.9)], [])  # candidate cai_strain None
    r.fail_if_above(strain=0.0)  # must not raise -- nothing to compare


def test_fail_if_above_default_threshold_is_quarter():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.9)])  # strain 0.1 < 0.25 default
    r.fail_if_above()  # must not raise


def test_fail_if_below_does_not_raise_when_above_minimum():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.9)])  # cai_score 0.9 >= 0.75
    r.fail_if_below(consistency=0.75)  # must not raise


def test_fail_if_below_raises_when_below_minimum():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.5)])
    with pytest.raises(AssertionError, match="CAI regression"):
        r.fail_if_below(consistency=0.75)


def test_fail_if_below_message_includes_labels_and_delta():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.5)])
    with pytest.raises(AssertionError) as exc_info:
        r.fail_if_below(consistency=0.75)
    msg = str(exc_info.value)
    assert "pr" in msg and "prod" in msg
    assert "Delta:" in msg


def test_fail_if_below_default_threshold():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.9)])
    r.fail_if_below()  # 0.9 >= default 0.75, must not raise


def test_regression_result_to_dict_shape():
    tc = _tc(name="a case")
    r = _rr([_tr(tc=tc, consistency_score=0.9)], [_tr(tc=tc, consistency_score=0.6)])
    d = r.to_dict()
    assert d["baseline_label"] == "prod"
    assert d["candidate_label"] == "pr"
    assert d["baseline_strain"] == r.baseline_report.cai_strain
    assert d["candidate_strain"] == r.candidate_report.cai_strain
    assert d["strain_delta"] == r.strain_delta
    assert d["baseline_cai"] == r.baseline_report.cai_score
    assert d["candidate_cai"] == r.candidate_report.cai_score
    assert d["cai_delta"] == r.cai_delta
    assert d["regressed"] == r.regressed
    assert d["baseline"] == r.baseline_report.to_dict()
    assert d["candidate"] == r.candidate_report.to_dict()


def test_regression_result_str_shows_regression_status():
    r = _rr([_tr(consistency_score=0.9)], [_tr(consistency_score=0.5)])
    out = str(r)
    assert "REGRESSION" in out
    assert "prod" in out and "pr" in out


def test_regression_result_str_shows_pass_status_and_na_when_no_delta():
    r = _rr([], [_tr(consistency_score=0.9)])  # strain_delta None
    out = str(r)
    assert "PASS" in out
    assert "N/A" in out
    assert "n/a" in out
