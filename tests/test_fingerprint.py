"""
Full test coverage for contradish/fingerprint.py.

fingerprint() clusters a Report's failed TestResults by an inferred
root-cause pattern (keyword match against unstable_patterns, falling back
to a severity label, falling back to "unclassified"), then returns
FailureCluster objects sorted by frequency (most common pattern first),
each internally sorted worst-first by cai_score.
"""
import pytest

from contradish.models import ContradictionPair, Report, TestCase, TestResult
from contradish.fingerprint import FailureCluster, _classify_pattern, fingerprint


# ── Fixture builders ────────────────────────────────────────────────────────

def _tc(input="a test question", name=None, **kw):
    return TestCase(input=input, name=name, **kw)


def _cp(output_a="out a", output_b="out b", severity="policy",
        explanation="they disagree", input_a="in a", input_b="in b"):
    return ContradictionPair(
        input_a=input_a, input_b=input_b,
        output_a=output_a, output_b=output_b,
        explanation=explanation, severity=severity,
    )


def _tr(tc=None, consistency_score=0.5, contradictions=None,
        unstable_patterns=None, suggestion=None, **kw):
    return TestResult(
        test_case=tc or _tc(),
        paraphrases=["p1"],
        outputs=["o0", "o1"],
        consistency_score=consistency_score,
        contradiction_score=0.0,
        contradictions=contradictions or [],
        unstable_patterns=unstable_patterns or [],
        suggestion=suggestion,
        **kw,
    )


def _report(results):
    return Report(results=results)


# ── _classify_pattern ────────────────────────────────────────────────────────

@pytest.mark.parametrize("patterns,expected", [
    (["invents a special case exception"], "exception_invention"),
    (["hedges with maybe and possibly"],    "hedge_inconsistency"),
    (["drifts on the number of days"],      "numeric_drift"),
    (["flips on whether user is eligible"], "eligibility_flip"),
    (["changes the deadline window"],       "deadline_drift"),
    (["blurs the legal disclaimer"],        "legal_boundary_blur"),
    (["contradicts on privacy/gdpr data"],  "data_policy_drift"),
    (["disagrees on coverage benefit"],     "coverage_inconsistency"),
])
def test_classify_pattern_matches_each_keyword_bucket(patterns, expected):
    assert _classify_pattern(patterns, severity="policy") == expected


def test_classify_pattern_first_matching_bucket_wins():
    # "exception" (bucket 1) appears before "number" (bucket 3) in keyword
    # priority order -- both are present, exception_invention should win.
    assert _classify_pattern(["an exception with a number of days"], "policy") == "exception_invention"


def test_classify_pattern_falls_back_to_severity_label_when_known():
    assert _classify_pattern(["nothing keyword-y here"], "factual") == "Factual contradiction"
    assert _classify_pattern([], "logical") == "Logical inconsistency"
    assert _classify_pattern([], "policy") == "Policy contradiction"


def test_classify_pattern_falls_back_to_raw_severity_when_unknown_label():
    assert _classify_pattern([], "some_custom_severity") == "some_custom_severity"


def test_classify_pattern_falls_back_to_unclassified_when_no_severity():
    assert _classify_pattern([], "") == "unclassified"


def test_classify_pattern_is_case_insensitive():
    assert _classify_pattern(["EXCEPTION granted"], "policy") == "exception_invention"


# ── FailureCluster.__str__ ──────────────────────────────────────────────────

def test_str_singular_rule_wording():
    c = FailureCluster(pattern_type="numeric_drift", frequency=1,
                        affected_rules=["rule a"], example_rule="rule a")
    out = str(c)
    assert "1 rule" in out
    assert "1 rules" not in out


def test_str_plural_rule_wording():
    c = FailureCluster(pattern_type="numeric_drift", frequency=2,
                        affected_rules=["rule a", "rule b"], example_rule="rule a")
    assert "2 rules" in str(c)


def test_str_lists_up_to_three_rules_without_more_suffix():
    c = FailureCluster(pattern_type="p", frequency=3,
                        affected_rules=["a", "b", "c"], example_rule="a")
    out = str(c)
    assert "a, b, c" in out
    assert "more" not in out


def test_str_shows_plus_n_more_beyond_three_rules():
    c = FailureCluster(pattern_type="p", frequency=5,
                        affected_rules=["a", "b", "c", "d", "e"], example_rule="a")
    out = str(c)
    assert "a, b, c" in out
    assert "(+2 more)" in out


def test_str_includes_fix_line_when_present():
    c = FailureCluster(pattern_type="p", frequency=1, affected_rules=["a"],
                        example_rule="a", suggested_fix="anchor the number")
    assert "fix:     anchor the number" in str(c)


def test_str_omits_fix_line_when_absent():
    c = FailureCluster(pattern_type="p", frequency=1, affected_rules=["a"], example_rule="a")
    assert "fix:" not in str(c)


# ── FailureCluster.to_dict() ────────────────────────────────────────────────

def test_to_dict_without_example_pair():
    c = FailureCluster(pattern_type="p", frequency=1, affected_rules=["a"], example_rule="a")
    d = c.to_dict()
    assert d == {
        "pattern_type": "p", "frequency": 1, "affected_rules": ["a"],
        "example_rule": "a", "suggested_fix": None,
    }
    assert "example_pair" not in d


def test_to_dict_with_example_pair():
    pair = _cp(input_a="ia", output_a="oa", input_b="ib", output_b="ob",
               explanation="exp", severity="factual")
    c = FailureCluster(pattern_type="p", frequency=1, affected_rules=["a"],
                        example_rule="a", example_pair=pair, suggested_fix="fix it")
    d = c.to_dict()
    assert d["example_pair"] == {
        "input_a": "ia", "output_a": "oa", "input_b": "ib", "output_b": "ob",
        "explanation": "exp", "severity": "factual",
    }
    assert d["suggested_fix"] == "fix it"


# ── fingerprint() ────────────────────────────────────────────────────────────

def test_fingerprint_empty_report_returns_empty_list():
    assert fingerprint(_report([])) == []


def test_fingerprint_no_failures_returns_empty_list():
    # All results pass thresholds -> report.failed is empty.
    passing = _tr(consistency_score=0.95)
    assert fingerprint(_report([passing])) == []


def test_fingerprint_groups_by_pattern_and_sorts_by_frequency():
    numeric_a = _tr(tc=_tc(name="numeric a"), consistency_score=0.4,
                     unstable_patterns=["drifts on number of days"])
    numeric_b = _tr(tc=_tc(name="numeric b"), consistency_score=0.3,
                     unstable_patterns=["drifts on the day count"])
    hedge = _tr(tc=_tc(name="hedge a"), consistency_score=0.5,
                unstable_patterns=["hedges with maybe"])
    clusters = fingerprint(_report([numeric_a, numeric_b, hedge]))
    assert [c.pattern_type for c in clusters] == ["numeric_drift", "hedge_inconsistency"]
    assert clusters[0].frequency == 2
    assert clusters[1].frequency == 1


def test_fingerprint_worst_case_is_first_in_affected_rules_and_example_rule():
    better = _tr(tc=_tc(name="better rule"), consistency_score=0.6,
                 unstable_patterns=["numeric drift here"])
    worse = _tr(tc=_tc(name="worse rule"), consistency_score=0.2,
                unstable_patterns=["numeric drift here too"])
    clusters = fingerprint(_report([better, worse]))
    assert len(clusters) == 1
    c = clusters[0]
    assert c.example_rule == "worse rule"
    assert c.affected_rules == ["worse rule", "better rule"]  # worst-first


def test_fingerprint_example_pair_comes_from_worst_results_first_contradiction():
    pair_worse = _cp(explanation="worse pair")
    pair_better = _cp(explanation="better pair")
    better = _tr(tc=_tc(name="better"), consistency_score=0.6, contradictions=[pair_better],
                 unstable_patterns=["numeric drift"])
    worse = _tr(tc=_tc(name="worse"), consistency_score=0.2, contradictions=[pair_worse],
                unstable_patterns=["numeric drift"])
    clusters = fingerprint(_report([better, worse]))
    assert clusters[0].example_pair.explanation == "worse pair"


def test_fingerprint_example_pair_none_when_worst_has_no_contradictions():
    worse = _tr(consistency_score=0.2, contradictions=[], unstable_patterns=["numeric drift"])
    clusters = fingerprint(_report([worse]))
    assert clusters[0].example_pair is None


def test_fingerprint_suggestion_prefers_worst_results_own_suggestion():
    better = _tr(tc=_tc(name="better"), consistency_score=0.6,
                 unstable_patterns=["numeric drift"], suggestion="better's fix")
    worse = _tr(tc=_tc(name="worse"), consistency_score=0.2,
                unstable_patterns=["numeric drift"], suggestion="worse's fix")
    clusters = fingerprint(_report([better, worse]))
    assert clusters[0].suggested_fix == "worse's fix"


def test_fingerprint_suggestion_falls_through_to_next_result_when_worst_has_none():
    better = _tr(tc=_tc(name="better"), consistency_score=0.6,
                 unstable_patterns=["numeric drift"], suggestion="better's fix")
    worse = _tr(tc=_tc(name="worse"), consistency_score=0.2,
                unstable_patterns=["numeric drift"], suggestion=None)
    clusters = fingerprint(_report([better, worse]))
    assert clusters[0].suggested_fix == "better's fix"


def test_fingerprint_suggestion_none_when_no_result_has_one():
    r = _tr(consistency_score=0.2, unstable_patterns=["numeric drift"], suggestion=None)
    clusters = fingerprint(_report([r]))
    assert clusters[0].suggested_fix is None


def test_fingerprint_uses_severity_when_no_pattern_keywords_match():
    r = _tr(consistency_score=0.2, unstable_patterns=[], contradictions=[_cp(severity="factual")])
    clusters = fingerprint(_report([r]))
    assert clusters[0].pattern_type == "Factual contradiction"


def test_fingerprint_unclassified_when_no_patterns_and_no_contradictions():
    r = _tr(consistency_score=0.2, unstable_patterns=[], contradictions=[])
    clusters = fingerprint(_report([r]))
    assert clusters[0].pattern_type == "unclassified"


def test_fingerprint_handles_missing_cai_score_in_sort():
    # cai_score falls back to `or 0.0` when consistency_score is None.
    scored = _tr(tc=_tc(name="scored"), consistency_score=0.3,
                 unstable_patterns=["numeric drift"])
    unscored = TestResult(
        test_case=_tc(name="unscored"),
        paraphrases=["p1"], outputs=["o0", "o1"],
        # consistency_score left None (untested this axis), but
        # contradiction_score alone is enough to fail passed() and land
        # this result in report.failed.
        consistency_score=None, contradiction_score=0.9,
        unstable_patterns=["numeric drift"],
    )
    clusters = fingerprint(_report([scored, unscored]))
    # unscored (cai_score None -> 0.0) sorts as "worse" than scored (0.3)
    assert clusters[0].example_rule == "unscored"
