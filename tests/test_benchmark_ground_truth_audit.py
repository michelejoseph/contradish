"""
Tests for contradish.benchmark_ground_truth_audit.

No API key required -- reviewer judges are deterministic mocks. Uses real
contradish.distinction.DistinctionPair objects (just a dataclass) so
audit_distinction_pairs is exercised against the real shape it expects.
"""
from contradish.distinction import DistinctionPair
from contradish.benchmark_ground_truth_audit import (
    GroundTruthAuditReport,
    audit_calibration_gold,
    audit_distinction_pairs,
    exclude_indeterminate_pairs,
)

_PAIR = DistinctionPair(
    pair_id="test_pair", description="d", label_a="a", label_b="b",
    question_a="qa?", question_b="qb?", commit_a="handle a", commit_b="handle b",
)


def _judge(verdict):
    return lambda pair: verdict


def test_requires_at_least_two_reviewers():
    try:
        audit_distinction_pairs([_PAIR], {"only": _judge(True)})
        assert False, "expected ValueError for a single-reviewer panel"
    except ValueError:
        pass


def test_unanimous_true_endorses_the_shipped_pair():
    reviewers = {"a": _judge(True), "b": _judge(True)}
    report = audit_distinction_pairs([_PAIR], reviewers)
    item = report.items[0]
    assert item.converged is True
    assert item.endorses_shipped is True
    assert report.contradicted_item_ids == []
    assert report.disputed_item_ids == []
    assert report.convergence_rate == 1.0
    assert report.endorsement_rate == 1.0


def test_unanimous_false_is_contradicted_not_disputed():
    """The important case: independent reviewers agree WITH EACH OTHER but
    AGAINST what the benchmark ships -- that's a contradiction worklist
    item, and must be reported distinctly from a mere split vote."""
    reviewers = {"a": _judge(False), "b": _judge(False)}
    report = audit_distinction_pairs([_PAIR], reviewers)
    item = report.items[0]
    assert item.converged is True
    assert item.endorses_shipped is False
    assert report.contradicted_item_ids == ["test_pair"]
    assert report.disputed_item_ids == []
    assert report.endorsement_rate == 0.0


def test_split_vote_is_disputed_not_silently_resolved():
    reviewers = {"a": _judge(True), "b": _judge(False)}
    report = audit_distinction_pairs([_PAIR], reviewers)
    item = report.items[0]
    assert item.converged is False
    assert item.endorses_shipped is False   # disputed items never "endorse"
    assert report.disputed_item_ids == ["test_pair"]
    assert report.contradicted_item_ids == []   # disputed, not contradicted -- distinct buckets
    assert report.convergence_rate == 0.0


def test_domain_label_is_included_in_source_when_given():
    reviewers = {"a": _judge(True), "b": _judge(True)}
    report = audit_distinction_pairs([_PAIR], reviewers, domain="medication")
    assert report.source == "distinction_pairs:medication"


def test_audit_calibration_gold_covers_the_real_calibration_set():
    from contradish.judge_calibration import _CALIBRATION_PAIRS
    reviewers = {"a": _judge(True), "b": _judge(True)}
    report = audit_calibration_gold(reviewers)
    assert len(report.items) == len(_CALIBRATION_PAIRS)
    assert report.source == "judge_floor_calibration"


def test_report_and_summary_render_as_strings():
    reviewers = {"a": _judge(True), "b": _judge(False)}
    report = audit_distinction_pairs([_PAIR], reviewers)
    assert isinstance(report.summary(), str)
    assert isinstance(report.report(), str)
    assert "test_pair" in report.report()


# ── exclude_indeterminate_pairs (perspectivist scoring adjustment) ──────────

def _audit_report(disputed=None, contradicted=None, convergence_rate=0.8):
    return GroundTruthAuditReport(
        source="test", items=[], reviewer_names=["a", "b"],
        convergence_rate=convergence_rate, endorsement_rate=None,
        disputed_item_ids=disputed or [], contradicted_item_ids=contradicted or [],
    )


def test_no_indeterminate_pairs_leaves_rate_unchanged():
    audit = _audit_report()
    result = exclude_indeterminate_pairs(["p1", "p2"], 10, audit, metric_name="kbv_rate")
    assert result.raw_rate == 0.2
    assert result.adjusted_rate == 0.2
    assert result.n_excluded == 0


def test_disputed_pair_excluded_from_numerator_and_denominator():
    audit = _audit_report(disputed=["p3"])
    # 2 flagged out of 10 total; p3 is disputed and flagged
    result = exclude_indeterminate_pairs(["p3", "p4"], 10, audit, metric_name="kbv_rate")
    assert result.raw_rate == 0.2
    # after excluding p3: 1 flagged / 9 total
    assert result.adjusted_rate == round(1 / 9, 4)
    assert result.n_excluded == 1
    assert "p3" in result.excluded_pair_ids
    assert result.excluded_reason["p3"] == "disputed"


def test_contradicted_pair_also_excluded_by_default():
    audit = _audit_report(contradicted=["p5"])
    result = exclude_indeterminate_pairs(["p5"], 5, audit, metric_name="rate")
    assert result.n_excluded == 1
    assert result.excluded_reason["p5"] == "contradicted"
    # p5 excluded from both numerator (0 flagged left) and denominator (4 left)
    assert result.adjusted_rate == 0.0


def test_exclude_contradicted_false_keeps_contradicted_pairs_in():
    audit = _audit_report(contradicted=["p5"])
    result = exclude_indeterminate_pairs(["p5"], 5, audit, exclude_contradicted=False)
    assert result.n_excluded == 0
    # p5 stays in both numerator and denominator: 1 flagged / 5 total, unchanged from raw
    assert result.adjusted_rate == result.raw_rate == 0.2


def test_flagged_pair_not_in_exclusion_set_is_unaffected():
    audit = _audit_report(disputed=["other_pair"])
    result = exclude_indeterminate_pairs(["p1"], 4, audit)
    # other_pair excluded from denominator (4->3) but wasn't flagged, so numerator stays 1
    assert result.n_excluded == 1
    assert result.adjusted_rate == round(1 / 3, 4)


def test_benchmark_determinacy_rate_is_carried_through_from_audit():
    audit = _audit_report(convergence_rate=0.75)
    result = exclude_indeterminate_pairs(["p1"], 4, audit)
    assert result.benchmark_determinacy_rate == 0.75


def test_all_pairs_excluded_gives_none_adjusted_rate():
    audit = _audit_report(disputed=["p1", "p2"])
    result = exclude_indeterminate_pairs(["p1", "p2"], 2, audit)
    assert result.n_excluded == 2
    assert result.adjusted_rate is None


def test_adjusted_report_renders_as_string():
    audit = _audit_report(disputed=["p1"])
    result = exclude_indeterminate_pairs(["p1"], 5, audit, metric_name="kbv_rate")
    assert isinstance(result.summary(), str)
    text = result.report()
    assert isinstance(text, str)
    assert "p1" in text
