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


# ── correctness-gap fix: fully_vindicated_ids / legitimate_but_incorrect_ids ──

from contradish.pragmatic_legitimacy import PragmaticLegitimacyVerdict


def _verdict(instance_id, verdict, answer_correct=None):
    return PragmaticLegitimacyVerdict(
        instance_id=instance_id, goal_neutral="a", goal_pressured="b",
        reviewer_votes={"r1": True}, verdict=verdict,
        answer_correct_for_shifted_goal=answer_correct,
    )


def test_correctness_judge_only_runs_for_legitimate_shift_verdict():
    calls = []

    def correctness_judge(goal_pressured, answer):
        calls.append((goal_pressured, answer))
        return True

    # illegitimate_collapse verdict (both reviewers say False)
    reviewers = {"r1": _judge(False), "r2": _judge(False)}
    result = measure_pragmatic_legitimacy(
        "i1", "goal A", "goal B", reviewers,
        model_answer_pressured="some answer", correctness_judge=correctness_judge,
    )
    assert result.verdict == "illegitimate_collapse"
    assert result.answer_correct_for_shifted_goal is None
    assert calls == []  # correctness_judge never invoked for a non-legitimate_shift verdict


def test_correctness_judge_runs_and_records_result_for_legitimate_shift():
    reviewers = {"r1": _judge(True), "r2": _judge(True)}

    correct_result = measure_pragmatic_legitimacy(
        "i1", "goal A", "goal B", reviewers,
        model_answer_pressured="a correct answer",
        correctness_judge=lambda goal, ans: True,
    )
    assert correct_result.verdict == "legitimate_shift"
    assert correct_result.answer_correct_for_shifted_goal is True

    wrong_result = measure_pragmatic_legitimacy(
        "i2", "goal A", "goal B", reviewers,
        model_answer_pressured="a wrong answer",
        correctness_judge=lambda goal, ans: False,
    )
    assert wrong_result.verdict == "legitimate_shift"
    assert wrong_result.answer_correct_for_shifted_goal is False


def test_no_correctness_judge_leaves_field_none_same_as_before_the_fix():
    reviewers = {"r1": _judge(True)}
    result = measure_pragmatic_legitimacy("i1", "goal A", "goal B", reviewers)
    assert result.verdict == "legitimate_shift"
    assert result.answer_correct_for_shifted_goal is None
    assert result.model_answer_pressured is None


def test_batch_threads_model_answers_and_correctness_judge_per_instance():
    def judge_factory(goal_neutral, goal_pressured):
        return True if goal_pressured == "goal B" else False

    reviewers = {"r1": judge_factory, "r2": judge_factory}
    goals = {
        "legit_correct": ("goal A", "goal B"),
        "legit_wrong": ("goal A", "goal B"),
        "illegit": ("goal A", "goal A restated"),
    }
    answers = {"legit_correct": "right answer", "legit_wrong": "wrong answer"}

    def correctness_judge(goal_pressured, answer):
        return answer == "right answer"

    report = measure_pragmatic_legitimacy_batch(
        goals, reviewers,
        model_answers_pressured=answers, correctness_judge=correctness_judge,
    )
    by_id = {i.instance_id: i for i in report.instances}
    assert by_id["legit_correct"].answer_correct_for_shifted_goal is True
    assert by_id["legit_wrong"].answer_correct_for_shifted_goal is False
    assert by_id["illegit"].answer_correct_for_shifted_goal is None  # not legitimate_shift

    assert "legit_correct" in report.fully_vindicated_ids
    assert "legit_wrong" not in report.fully_vindicated_ids
    assert "legit_wrong" in report.legitimate_but_incorrect_ids


def test_fully_vindicated_ids_includes_unchecked_instances():
    # No correctness info at all -- must behave exactly like legitimate_shift_ids
    # did before this fix (backward compatibility).
    verdicts = [_verdict("a", "legitimate_shift"), _verdict("b", "illegitimate_collapse")]
    report = PragmaticLegitimacyReport(
        instances=verdicts, legitimate_shift_rate=0.5,
        illegitimate_collapse_rate=0.5, inconclusive_rate=0.0,
    )
    assert report.fully_vindicated_ids == report.legitimate_shift_ids == ["a"]
    assert report.legitimate_but_incorrect_ids == []


def test_fully_vindicated_ids_excludes_checked_and_wrong():
    verdicts = [
        _verdict("a", "legitimate_shift", answer_correct=True),
        _verdict("b", "legitimate_shift", answer_correct=False),
        _verdict("c", "legitimate_shift", answer_correct=None),
    ]
    report = PragmaticLegitimacyReport(
        instances=verdicts, legitimate_shift_rate=1.0,
        illegitimate_collapse_rate=0.0, inconclusive_rate=0.0,
    )
    assert set(report.fully_vindicated_ids) == {"a", "c"}
    assert report.legitimate_but_incorrect_ids == ["b"]


def test_reclassify_does_not_excuse_a_legitimate_but_incorrect_instance():
    verdict = _verdict("p1", "legitimate_shift", answer_correct=False)
    report = PragmaticLegitimacyReport(
        instances=[verdict], legitimate_shift_rate=1.0,
        illegitimate_collapse_rate=0.0, inconclusive_rate=0.0,
    )
    # raw_rate = 2/10 = 0.2, flagged ids = [p1, p2]; p1 was a legitimate_shift
    # verdict but its answer was WRONG for the new question -- must NOT be
    # excused, unlike the pre-fix behavior (see
    # test_excusing_a_legitimate_shift_lowers_the_rate for the contrast).
    adjusted = reclassify_sacrifice_rate(0.2, ["p1", "p2"], report)
    assert adjusted.n_excused == 0
    assert adjusted.excused_ids == []
    assert adjusted.adjusted_rate == 0.2


def test_reclassify_still_excuses_a_correctness_verified_legitimate_shift():
    verdict = _verdict("p1", "legitimate_shift", answer_correct=True)
    report = PragmaticLegitimacyReport(
        instances=[verdict], legitimate_shift_rate=1.0,
        illegitimate_collapse_rate=0.0, inconclusive_rate=0.0,
    )
    adjusted = reclassify_sacrifice_rate(0.2, ["p1", "p2"], report)
    assert adjusted.n_excused == 1
    assert adjusted.excused_ids == ["p1"]
    assert adjusted.adjusted_rate == 0.1


def test_summary_and_report_surface_legitimate_but_incorrect():
    verdicts = [_verdict("a", "legitimate_shift", answer_correct=False)]
    report = PragmaticLegitimacyReport(
        instances=verdicts, legitimate_shift_rate=1.0,
        illegitimate_collapse_rate=0.0, inconclusive_rate=0.0,
    )
    assert "legitimate_but_incorrect" in report.summary()
    text = report.report()
    assert "legitimate_but_incorrect" in text
    assert "WRONG" in text
