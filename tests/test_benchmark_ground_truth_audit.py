"""
Tests for contradish.benchmark_ground_truth_audit.

No API key required -- reviewer judges are deterministic mocks. Uses real
contradish.distinction.DistinctionPair objects (just a dataclass) so
audit_distinction_pairs is exercised against the real shape it expects.
"""
from contradish.distinction import DistinctionPair
from contradish.benchmark_ground_truth_audit import audit_calibration_gold, audit_distinction_pairs

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
