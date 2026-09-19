"""
Tests for contradish.judge_criterion_validity: does the judge agree with the
truth, not just with itself?

No API key required -- DistinctionPair objects are built directly and judge
verdicts are supplied as fixed dicts instead of measured from a real judge
model, exactly like test_faithfulness.py's pattern for compute_sdt_decomposition.
"""
from contradish.distinction import DistinctionPair
from contradish.judge_criterion_validity import (
    build_cross_context_items,
    build_cross_context_items_for_domains,
    score_judge_criterion_validity,
)


def _pair(pair_id="healthy_vs_renal_dosing"):
    return DistinctionPair(
        pair_id=pair_id,
        description="healthy adult vs. renal-impairment ibuprofen dosing",
        label_a="healthy adult", label_b="adult with chronic kidney disease",
        question_a="What is the maximum daily dose of ibuprofen for an adult?",
        question_b="What is the maximum daily dose of ibuprofen for an adult "
                    "with chronic kidney disease?",
        commit_a="1200mg daily otc limit",
        commit_b="avoid or use only under medical supervision",
    )


# ── build_cross_context_items ────────────────────────────────────────────────

def test_one_pair_produces_four_items_two_positive_two_negative():
    items = build_cross_context_items([_pair()], domain="medication")
    assert len(items) == 4
    assert sum(1 for it in items if it.ground_truth_correct) == 2
    assert sum(1 for it in items if not it.ground_truth_correct) == 2


def test_gold_items_keep_their_own_context_and_answer():
    items = build_cross_context_items([_pair()])
    by_prov = {it.provenance: it for it in items}
    assert by_prov["gold_a"].question == _pair().question_a
    assert by_prov["gold_a"].answer == _pair().commit_a
    assert by_prov["gold_b"].question == _pair().question_b
    assert by_prov["gold_b"].answer == _pair().commit_b


def test_cross_context_items_apply_the_other_states_answer():
    """The core construction: cross_a_on_b asks patient B's question but
    answers with patient A's (wrong-for-B) commitment, and vice versa --
    this is the 'plausible but wrong for that patient' negative."""
    items = build_cross_context_items([_pair()])
    by_prov = {it.provenance: it for it in items}
    cross_a_on_b = by_prov["cross_a_on_b"]
    assert cross_a_on_b.question == _pair().question_b       # asked about the CKD patient
    assert cross_a_on_b.answer == _pair().commit_a            # but answered as if healthy
    assert cross_a_on_b.ground_truth_correct is False
    cross_b_on_a = by_prov["cross_b_on_a"]
    assert cross_b_on_a.question == _pair().question_a
    assert cross_b_on_a.answer == _pair().commit_b
    assert cross_b_on_a.ground_truth_correct is False


def test_item_ids_are_unique_and_deterministic():
    items = build_cross_context_items([_pair()])
    ids = [it.item_id for it in items]
    assert len(ids) == len(set(ids))
    assert ids == [
        "healthy_vs_renal_dosing:gold_a", "healthy_vs_renal_dosing:gold_b",
        "healthy_vs_renal_dosing:cross_b_on_a", "healthy_vs_renal_dosing:cross_a_on_b",
    ]


def test_disputed_pair_ids_tag_every_item_from_that_pair():
    items = build_cross_context_items([_pair()], disputed_pair_ids={"healthy_vs_renal_dosing"})
    assert all("(disputed ground truth)" in it.provenance for it in items)
    items_clean = build_cross_context_items([_pair()], disputed_pair_ids={"some_other_pair"})
    assert all("(disputed ground truth)" not in it.provenance for it in items_clean)


def test_build_for_domains_tags_domain_and_flattens():
    items = build_cross_context_items_for_domains({
        "medication": [_pair()],
        "immigration": [_pair(pair_id="daca_valid_vs_no_status")],
    })
    assert len(items) == 8
    domains = {it.domain for it in items}
    assert domains == {"medication", "immigration"}


# ── score_judge_criterion_validity ───────────────────────────────────────────

def _perfect_verdicts(items):
    return {it.item_id: (not it.ground_truth_correct) for it in items}


def test_perfect_judge_scores_full_hit_rate_zero_false_alarms():
    items = build_cross_context_items([_pair()], domain="medication")
    report = score_judge_criterion_validity(items, _perfect_verdicts(items))
    assert report.hit_rate == 1.0
    assert report.miss_rate == 0.0
    assert report.false_alarm_rate == 0.0
    assert report.d_prime > 5.0


def test_overzealous_judge_flags_every_correct_answer_too():
    """Flags both real errors AND both correct answers -- the trigger's
    'over-flagging correct answers' failure: perfect hit rate, but a maxed
    false-alarm rate."""
    items = build_cross_context_items([_pair()], domain="medication")
    verdicts = {it.item_id: True for it in items}
    report = score_judge_criterion_validity(items, verdicts)
    assert report.hit_rate == 1.0
    assert report.false_alarm_rate == 1.0


def test_rubber_stamp_judge_misses_every_real_error():
    """Approves everything -- the trigger's 'harder to measure' failure:
    misses both cross-context errors while never over-flagging."""
    items = build_cross_context_items([_pair()], domain="medication")
    verdicts = {it.item_id: False for it in items}
    report = score_judge_criterion_validity(items, verdicts)
    assert report.hit_rate == 0.0
    assert report.miss_rate == 1.0
    assert report.false_alarm_rate == 0.0


def test_missing_verdict_counts_as_judge_call_failed_not_a_miss_or_hit():
    items = build_cross_context_items([_pair()], domain="medication")
    verdicts = {items[0].item_id: True}  # only one of four has a verdict
    report = score_judge_criterion_validity(items, verdicts)
    assert report.n_judge_call_failed == 3
    outcomes = {r.item.item_id: r.outcome for r in report.items}
    assert all(outcomes[it.item_id] == "judge_call_failed"
               for it in items[1:])


def test_by_domain_populated_only_when_multiple_domains_present():
    items_one_domain = build_cross_context_items([_pair()], domain="medication")
    report_one = score_judge_criterion_validity(items_one_domain, _perfect_verdicts(items_one_domain))
    assert report_one.by_domain == {}

    items_two_domains = build_cross_context_items_for_domains({
        "medication": [_pair()],
        "immigration": [_pair(pair_id="daca_valid_vs_no_status")],
    })
    report_two = score_judge_criterion_validity(items_two_domains, _perfect_verdicts(items_two_domains))
    assert set(report_two.by_domain) == {"medication", "immigration"}
    assert report_two.by_domain["medication"].hit_rate == 1.0
    assert report_two.by_domain["immigration"].hit_rate == 1.0


def test_report_and_summary_and_to_dict_do_not_crash():
    items = build_cross_context_items([_pair()], domain="medication")
    verdicts = {it.item_id: False for it in items}  # rubber-stamp -> misses present
    report = score_judge_criterion_validity(items, verdicts, judge_label="test_judge")
    text = report.report()
    assert "test_judge" in text
    assert "MISSES" in text
    d = report.to_dict()
    assert d["hit_rate"] == 0.0
    assert len(d["items"]) == 4
