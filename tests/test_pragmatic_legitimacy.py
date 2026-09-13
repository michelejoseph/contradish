"""
Tests for contradish.pragmatic_legitimacy.

No API key required: reviewer_judges are deterministic mocks (plain
Python callables), and score_legitimacy_votes / reclassify_sacrifice_rate
are the pure, model-free cores these tests exercise directly.
"""
from contradish.pragmatic_legitimacy import (
    AdjustedRateReport,
    PragmaticLegitimacyReport,
    measure_pragmatic_legitimacy,
    measure_pragmatic_legitimacy_batch,
    reclassify_sacrifice_rate,
    score_legitimacy_votes,
)


# ── score_legitimacy_votes (pure core) ────────────────────────────────────────

def test_majority_true_is_legitimate_shift():
    assert score_legitimacy_votes([True, True, False]) == "legitimate_shift"


def test_majority_false_is_illegitimate_collapse():
    assert score_legitimacy_votes([False, False, True]) == "illegitimate_collapse"


def test_tie_is_inconclusive():
    assert score_legitimacy_votes([True, False]) == "inconclusive"


def test_all_none_is_inconclusive():
    assert score_legitimacy_votes([None, None]) == "inconclusive"


def test_none_votes_excluded_before_majority():
    # 2 true, 1 false, 1 None -> None dropped, majority is still true
    assert score_legitimacy_votes([True, True, False, None]) == "legitimate_shift"


def test_empty_votes_is_inconclusive():
    assert score_legitimacy_votes([]) == "inconclusive"


# ── measure_pragmatic_legitimacy / batch ──────────────────────────────────────

def _judge(verdict):
    return lambda goal_a, goal_b: verdict


def test_single_instance_classification():
    reviewers = {"r1": _judge(True), "r2": _judge(True)}
    result = measure_pragmatic_legitimacy("i1", "goal A", "goal B", reviewers)
    assert result.verdict == "legitimate_shift"
    assert result.reviewer_votes == {"r1": True, "r2": True}


def test_requires_at_least_one_reviewer():
    try:
        measure_pragmatic_legitimacy("i1", "a", "b", {})
        assert False, "expected ValueError for empty reviewer_judges"
    except ValueError:
        pass


def test_batch_computes_rates_across_instances():
    reviewers = {"r1": None, "r2": None}  # placeholder, replaced per-instance below

    def make_reviewers(verdict):
        return {"r1": _judge(verdict), "r2": _judge(verdict)}

    goals = {
        "legit_1": ("goal A", "goal B"),
        "illegit_1": ("goal A", "goal A restated"),
    }
    # Run each instance with its own reviewer set by calling measure_pragmatic_legitimacy_batch
    # per instance would defeat the point, so instead build one reviewer dict whose verdict
    # depends on the instance's goals.
    def judge_factory(goal_neutral, goal_pressured):
        return True if goal_pressured == "goal B" else False

    reviewers = {"r1": judge_factory, "r2": judge_factory}
    report = measure_pragmatic_legitimacy_batch(goals, reviewers)
    assert isinstance(report, PragmaticLegitimacyReport)
    assert len(report.instances) == 2
    assert "legit_1" in report.legitimate_shift_ids
    assert "illegit_1" in report.illegitimate_collapse_ids
    assert report.legitimate_shift_rate == 0.5
    assert report.illegitimate_collapse_rate == 0.5


def test_report_and_summary_render_as_strings():
    reviewers = {"r1": _judge(True)}
    report = measure_pragmatic_legitimacy_batch({"i1": ("a", "b")}, reviewers)
    assert isinstance(report.summary(), str)
    text = report.report()
    assert isinstance(text, str)
    assert "i1" in text


# ── reclassify_sacrifice_rate (the integration that changes a real number) ──

def test_no_instances_excused_leaves_rate_unchanged():
    report = PragmaticLegitimacyReport(instances=[], legitimate_shift_rate=None,
                                        illegitimate_collapse_rate=None, inconclusive_rate=None)
    adjusted = reclassify_sacrifice_rate(0.4, ["p1", "p2"], report)
    assert adjusted.adjusted_rate == 0.4
    assert adjusted.n_excused == 0


def test_excusing_a_legitimate_shift_lowers_the_rate():
    from contradish.pragmatic_legitimacy import PragmaticLegitimacyVerdict
    verdict = PragmaticLegitimacyVerdict(
        instance_id="p1", goal_neutral="a", goal_pressured="b",
        reviewer_votes={"r1": True}, verdict="legitimate_shift",
    )
    report = PragmaticLegitimacyReport(instances=[verdict], legitimate_shift_rate=1.0,
                                        illegitimate_collapse_rate=0.0, inconclusive_rate=0.0)
    # raw_rate = 2/10 = 0.2, flagged ids = [p1, p2] (2 flagged instances)
    adjusted = reclassify_sacrifice_rate(0.2, ["p1", "p2"], report)
    assert adjusted.n_excused == 1
    assert adjusted.excused_ids == ["p1"]
    # denominator recovered as 2/0.2 = 10; after excusing p1, (2-1)/10 = 0.1
    assert adjusted.adjusted_rate == 0.1


def test_zero_raw_rate_is_returned_unchanged():
    report = PragmaticLegitimacyReport(instances=[], legitimate_shift_rate=None,
                                        illegitimate_collapse_rate=None, inconclusive_rate=None)
    adjusted = reclassify_sacrifice_rate(0.0, [], report)
    assert adjusted.adjusted_rate == 0.0
    assert adjusted.n_total == 0


def test_adjusted_report_renders_as_string():
    report = PragmaticLegitimacyReport(instances=[], legitimate_shift_rate=None,
                                        illegitimate_collapse_rate=None, inconclusive_rate=None)
    adjusted = reclassify_sacrifice_rate(0.3, ["p1"], report)
    assert isinstance(adjusted, AdjustedRateReport)
    assert isinstance(adjusted.summary(), str)
    assert isinstance(adjusted.report(), str)
