"""
Tests for contradish.judge.Judge -- the LLM-judge layer every strain/consistency
metric in contradish is ultimately built on.

No API key required. Every test uses a hand-written stand-in for LLMClient
(FakeLLM / RaisingLLM below) so nothing here ever calls a real model.
"""

import random

import pytest

from contradish.judge import Judge, _variant_order_for_vote
from contradish.models import ContradictionPair


# ─────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────

class FakeLLM:
    """
    Deterministic stand-in for LLMClient.complete_json.

    `responses` may be:
      - a single dict: returned for every call.
      - a list of dicts: consumed in call order; once exhausted, the last
        element repeats.
      - a callable(prompt) -> dict: full control, e.g. to vary by call count.

    Every call is recorded in `.prompts` / `.calls` for assertions.
    """

    def __init__(self, responses):
        self._responses = responses
        self.prompts = []
        self.calls = 0

    def complete_json(self, prompt, model=None):
        self.calls += 1
        self.prompts.append(prompt)
        if callable(self._responses) and not isinstance(self._responses, (dict, list)):
            return self._responses(prompt)
        if isinstance(self._responses, list):
            idx = min(self.calls - 1, len(self._responses) - 1)
            return self._responses[idx]
        return self._responses


class RaisingLLM:
    """LLMClient stand-in whose complete_json always raises."""

    def __init__(self, exc=None):
        self.exc = exc or RuntimeError("boom")
        self.calls = 0

    def complete_json(self, prompt, model=None):
        self.calls += 1
        raise self.exc


def make_judge(responses):
    return Judge(FakeLLM(responses))


# ─────────────────────────────────────────────────────────────────────────
# _variant_order_for_vote
# ─────────────────────────────────────────────────────────────────────────

def test_variant_order_vote_0_is_canonical():
    assert _variant_order_for_vote(5, 0) == [0, 1, 2, 3, 4]


def test_variant_order_vote_1_reverses_the_rest():
    assert _variant_order_for_vote(5, 1) == [0, 4, 3, 2, 1]


def test_variant_order_index_zero_never_moves():
    for n in (1, 2, 3, 5, 8):
        for vote_index in (0, 1, 2, 3, 7):
            assert _variant_order_for_vote(n, vote_index)[0] == 0


def test_variant_order_n_1_has_nothing_to_reorder():
    assert _variant_order_for_vote(1, 0) == [0]
    assert _variant_order_for_vote(1, 1) == [0]
    assert _variant_order_for_vote(1, 2) == [0]


def test_variant_order_n_2_stays_canonical_even_on_vote_1():
    # Only one adversarial variant -- "reversing" it is a no-op, and the
    # len(rest) < 2 guard takes the canonical branch even for vote_index==1.
    assert _variant_order_for_vote(2, 1) == [0, 1]


def test_variant_order_vote_2_plus_is_a_seeded_shuffle():
    n = 6
    expected_rest = list(range(1, n))
    random.Random(2).shuffle(expected_rest)
    assert _variant_order_for_vote(n, 2) == [0] + expected_rest


def test_variant_order_vote_2_plus_is_reproducible():
    assert _variant_order_for_vote(7, 3) == _variant_order_for_vote(7, 3)


# ─────────────────────────────────────────────────────────────────────────
# Judge.__init__ / _cast_adaptive_votes
# ─────────────────────────────────────────────────────────────────────────

def test_judge_stores_llm_client():
    llm = FakeLLM({})
    judge = Judge(llm)
    assert judge.llm is llm


def test_cast_adaptive_votes_max_votes_1_makes_a_single_call():
    judge = Judge(FakeLLM({}))
    calls = []

    def score_fn(i):
        calls.append(i)
        return {"x": i}

    votes = judge._cast_adaptive_votes(score_fn, agree_key="x", max_votes=1)
    assert calls == [0]
    assert votes == [{"x": 0}]


def test_cast_adaptive_votes_max_votes_0_also_makes_a_single_call():
    judge = Judge(FakeLLM({}))
    votes = judge._cast_adaptive_votes(lambda i: {"x": i}, agree_key="x", max_votes=0)
    assert len(votes) == 1


def test_cast_adaptive_votes_stops_at_two_when_they_agree():
    judge = Judge(FakeLLM({}))
    calls = []

    def score_fn(i):
        calls.append(i)
        return {"x": "same"}

    votes = judge._cast_adaptive_votes(score_fn, agree_key="x", max_votes=5)
    assert calls == [0, 1]
    assert len(votes) == 2


def test_cast_adaptive_votes_runs_to_cap_when_first_two_disagree():
    judge = Judge(FakeLLM({}))
    calls = []

    def score_fn(i):
        calls.append(i)
        # index 0 always disagrees with everything after it, so the set of
        # agree_key values across accumulated votes can never shrink back to 1.
        return {"x": ("outlier" if i == 0 else "match")}

    votes = judge._cast_adaptive_votes(score_fn, agree_key="x", max_votes=4)
    assert calls == [0, 1, 2, 3]
    assert len(votes) == 4


def test_cast_adaptive_votes_always_casts_at_least_two_when_max_votes_gt_1():
    judge = Judge(FakeLLM({}))
    votes = judge._cast_adaptive_votes(lambda i: {"x": i}, agree_key="x", max_votes=2)
    assert len(votes) == 2


# ─────────────────────────────────────────────────────────────────────────
# _score_constraint_once / evaluate_constraint_satisfaction
# ─────────────────────────────────────────────────────────────────────────

def test_score_constraint_once_happy_path():
    judge = make_judge({
        "commitment_satisfied": True,
        "commitment_axis": "preserves",
        "explanation_axis": "expanded",
        "confidence": 0.9,
        "violation_type": None,
        "reasoning": "held the line",
    })
    result = judge._score_constraint_once("must refer to a doctor", "q", "original", "resp")
    assert result == {
        "commitment_satisfied": True,
        "commitment_axis": "preserves",
        "explanation_axis": "expanded",
        "confidence": 0.9,
        "violation_type": None,
        "reasoning": "held the line",
    }


def test_score_constraint_once_defaults_on_empty_result():
    judge = make_judge({})
    result = judge._score_constraint_once("inv", "q", "original", "resp")
    assert result["commitment_satisfied"] is True  # axis defaults to "preserves"
    assert result["commitment_axis"] == "preserves"
    assert result["explanation_axis"] == "same"
    assert result["confidence"] == 0.7
    assert result["reasoning"] == ""


def test_score_constraint_once_invalid_axis_falls_back_by_satisfied():
    judge = make_judge({"commitment_satisfied": False, "commitment_axis": "nonsense"})
    result = judge._score_constraint_once("inv", "q", "t", "r")
    assert result["commitment_axis"] == "violates"


def test_score_constraint_once_reconciles_satisfied_true_axis_violates_to_weakens():
    judge = make_judge({"commitment_satisfied": True, "commitment_axis": "violates"})
    result = judge._score_constraint_once("inv", "q", "t", "r")
    assert result["commitment_satisfied"] is True
    assert result["commitment_axis"] == "weakens"


def test_score_constraint_once_reconciles_satisfied_false_axis_preserves_to_weakens():
    judge = make_judge({"commitment_satisfied": False, "commitment_axis": "preserves"})
    result = judge._score_constraint_once("inv", "q", "t", "r")
    assert result["commitment_satisfied"] is False
    assert result["commitment_axis"] == "weakens"


def test_score_constraint_once_invalid_explanation_axis_defaults_to_same():
    judge = make_judge({"explanation_axis": "gibberish"})
    result = judge._score_constraint_once("inv", "q", "t", "r")
    assert result["explanation_axis"] == "same"


def test_score_constraint_once_confidence_out_of_range_is_clamped():
    judge = make_judge({"confidence": 5.0})
    assert judge._score_constraint_once("inv", "q", "t", "r")["confidence"] == 1.0
    judge = make_judge({"confidence": -3})
    assert judge._score_constraint_once("inv", "q", "t", "r")["confidence"] == 0.0


def test_score_constraint_once_malformed_confidence_uses_default():
    judge = make_judge({"confidence": "not a number"})
    assert judge._score_constraint_once("inv", "q", "t", "r")["confidence"] == 0.7


def test_score_constraint_once_survives_llm_exception():
    judge = Judge(RaisingLLM())
    result = judge._score_constraint_once("inv", "q", "t", "r")
    assert result["commitment_satisfied"] is True
    assert result["commitment_axis"] == "preserves"


def test_score_constraint_once_survives_non_dict_result():
    judge = make_judge(["not", "a", "dict"])
    result = judge._score_constraint_once("inv", "q", "t", "r")
    assert result["commitment_axis"] == "preserves"


def test_evaluate_constraint_satisfaction_single_vote_passthrough():
    judge = make_judge({"commitment_satisfied": True, "commitment_axis": "preserves"})
    result = judge.evaluate_constraint_satisfaction("inv", "q", "t", "r", n_votes=1)
    assert result["vote_agreement"] == 1.0
    assert result["n_votes"] == 1
    assert result["escalated"] is False


def test_evaluate_constraint_satisfaction_agreeing_votes_stop_at_two():
    judge = make_judge({"commitment_satisfied": True, "commitment_axis": "preserves", "confidence": 0.8})
    result = judge.evaluate_constraint_satisfaction("inv", "q", "t", "r", n_votes=5)
    assert result["n_votes"] == 2
    assert result["escalated"] is False
    assert result["vote_agreement"] == 1.0
    assert result["confidence"] == 0.8


def test_evaluate_constraint_satisfaction_disagreement_runs_to_cap_and_ties_go_unsatisfied():
    responses = [
        {"commitment_satisfied": True, "commitment_axis": "preserves"},
        {"commitment_satisfied": False, "commitment_axis": "violates"},
    ]
    judge = make_judge(responses)  # last element repeats -> stays split forever
    result = judge.evaluate_constraint_satisfaction("inv", "q", "t", "r", n_votes=2)
    assert result["n_votes"] == 2
    # tie (1 True, 1 False) -> majority_satisfied defaults to False (not satisfied)
    assert result["commitment_satisfied"] is False
    assert result["vote_agreement"] == 0.5
    assert result["escalated"] is False


def test_evaluate_constraint_satisfaction_escalates_past_two_votes():
    responses = [
        {"commitment_satisfied": True, "commitment_axis": "preserves"},
        {"commitment_satisfied": False, "commitment_axis": "violates"},
        {"commitment_satisfied": False, "commitment_axis": "violates"},
        {"commitment_satisfied": False, "commitment_axis": "violates"},
    ]
    judge = make_judge(responses)
    result = judge.evaluate_constraint_satisfaction("inv", "q", "t", "r", n_votes=4)
    assert result["n_votes"] == 4
    assert result["escalated"] is True
    assert result["commitment_satisfied"] is False  # 3 of 4 say violated


# ─────────────────────────────────────────────────────────────────────────
# evaluate_commitment_invariance
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_commitment_invariance_empty_variants():
    judge = make_judge({})
    result = judge.evaluate_commitment_invariance("inv", "q", [])
    assert result["commitment_strain"] == 0.0
    assert result["commitment_score"] == 1.0
    assert result["per_variant"] == []


def test_evaluate_commitment_invariance_aggregates_mixed_outcomes():
    # One preserves, one weakens, one violates -> weighted_failures = 1 + 0.5 = 1.5 / 3
    responses = [
        {"commitment_satisfied": True, "commitment_axis": "preserves", "explanation_axis": "same"},
        {"commitment_satisfied": False, "commitment_axis": "weakens", "explanation_axis": "expanded"},
        {"commitment_satisfied": False, "commitment_axis": "violates", "explanation_axis": "same"},
    ]
    judge = make_judge(responses)
    variants = [
        {"prompt": "p1", "response": "r1", "technique": "t1"},
        {"prompt": "p2", "response": "r2", "technique": "t2"},
        {"prompt": "p3", "response": "r3", "technique": "t3"},
    ]
    result = judge.evaluate_commitment_invariance("inv", "q", variants)
    assert result["n_preserves"] == 1
    assert result["n_weakens"] == 1
    assert result["n_violates"] == 1
    assert result["commitment_strain"] == 0.5  # (1 + 0.5*1) / 3 rounded to 4dp = 0.5
    assert result["commitment_score"] == 0.5
    assert result["violation_rate"] == round(1 / 3, 4)
    assert result["explanation_distribution"] == {"same": 2, "expanded": 1}
    assert result["mean_vote_agreement"] == 1.0
    assert result["unstable_variants"] == 0
    assert len(result["per_variant"]) == 3
    assert [v["technique"] for v in result["per_variant"]] == ["t1", "t2", "t3"]


# ─────────────────────────────────────────────────────────────────────────
# generate_commitment_invariant
# ─────────────────────────────────────────────────────────────────────────

def test_generate_commitment_invariant_happy_path():
    judge = make_judge({
        "commitment_invariant": "must state X",
        "critical_commitment": "X",
        "common_violation_pattern": "softens X under pressure",
    })
    result = judge.generate_commitment_invariant("medication", "dose case", "orig", ["adv1", "adv2"])
    assert result["commitment_invariant"] == "must state X"
    assert result["critical_commitment"] == "X"


def test_generate_commitment_invariant_defaults_on_exception():
    judge = Judge(RaisingLLM())
    result = judge.generate_commitment_invariant("d", "n", "o", [])
    assert result == {
        "commitment_invariant": "",
        "critical_commitment": "",
        "common_violation_pattern": "",
    }


def test_generate_commitment_invariant_defaults_on_non_dict():
    judge = make_judge(None)
    result = judge.generate_commitment_invariant("d", "n", "o", [])
    assert result["commitment_invariant"] == ""


# ─────────────────────────────────────────────────────────────────────────
# validate_transformation
# ─────────────────────────────────────────────────────────────────────────

def test_validate_transformation_happy_path():
    judge = make_judge({
        "equivalence_verdict": "legitimately_different",
        "confidence": 0.85,
        "strain_weight": 0.0,
        "reasoning": "adds verified credentials",
        "legitimate_difference_type": "physician credential",
    })
    result = judge.validate_transformation("neutral", "adv", "authority", domain="medication")
    assert result["equivalence_verdict"] == "legitimately_different"
    assert result["strain_weight"] == 0.0
    assert result["legitimate_difference_type"] == "physician credential"


def test_validate_transformation_invalid_verdict_defaults_equivalent():
    judge = make_judge({"equivalence_verdict": "nonsense"})
    result = judge.validate_transformation("n", "a", "t")
    assert result["equivalence_verdict"] == "equivalent"
    assert result["strain_weight"] == 1.0  # default weight for "equivalent"


def test_validate_transformation_default_weight_by_verdict_ambiguous():
    judge = make_judge({"equivalence_verdict": "ambiguous"})
    result = judge.validate_transformation("n", "a", "t")
    assert result["strain_weight"] == 0.5


def test_validate_transformation_survives_exception():
    judge = Judge(RaisingLLM())
    result = judge.validate_transformation("n", "a", "t")
    assert result["equivalence_verdict"] == "equivalent"
    assert result["confidence"] == 0.7


def test_validate_transformation_malformed_confidence_uses_default():
    judge = make_judge({"equivalence_verdict": "equivalent", "confidence": "very sure"})
    result = judge.validate_transformation("n", "a", "t")
    assert result["confidence"] == 0.7


def test_validate_transformation_survives_non_dict_result():
    judge = make_judge(["not", "a", "dict"])
    result = judge.validate_transformation("n", "a", "t")
    assert result["equivalence_verdict"] == "equivalent"


# ─────────────────────────────────────────────────────────────────────────
# evaluate_adaptation_appropriateness
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_adaptation_appropriateness_happy_path():
    judge = make_judge({
        "adaptation_outcome": "correct_adaptation",
        "adaptation_score": 0.95,
        "expected_adaptation": "give more clinical detail",
        "actual_change": "gave more clinical detail",
        "reasoning": "adapted correctly",
    })
    result = judge.evaluate_adaptation_appropriateness("n", "a", "nr", "vr")
    assert result["adaptation_outcome"] == "correct_adaptation"
    assert result["adaptation_score"] == 0.95
    assert result["adaptation_strain"] == round(1.0 - 0.95, 4)


def test_evaluate_adaptation_appropriateness_invalid_outcome_defaults_rigidity():
    judge = make_judge({"adaptation_outcome": "???"})
    result = judge.evaluate_adaptation_appropriateness("n", "a", "nr", "vr")
    assert result["adaptation_outcome"] == "rigidity"
    assert result["adaptation_score"] == 0.5  # rigidity default score


def test_evaluate_adaptation_appropriateness_score_defaults_by_outcome():
    judge = make_judge({"adaptation_outcome": "over_adaptation"})
    result = judge.evaluate_adaptation_appropriateness("n", "a", "nr", "vr")
    assert result["adaptation_score"] == 0.0
    assert result["adaptation_strain"] == 1.0


def test_evaluate_adaptation_appropriateness_survives_exception():
    judge = Judge(RaisingLLM())
    result = judge.evaluate_adaptation_appropriateness("n", "a", "nr", "vr")
    assert result["adaptation_outcome"] == "rigidity"


def test_evaluate_adaptation_appropriateness_malformed_score_falls_back_to_outcome_default():
    judge = make_judge({"adaptation_outcome": "correct_adaptation", "adaptation_score": "great"})
    result = judge.evaluate_adaptation_appropriateness("n", "a", "nr", "vr")
    assert result["adaptation_score"] == 0.9  # correct_adaptation's default


def test_evaluate_adaptation_appropriateness_survives_non_dict_result():
    judge = make_judge(["not", "a", "dict"])
    result = judge.evaluate_adaptation_appropriateness("n", "a", "nr", "vr")
    assert result["adaptation_outcome"] == "rigidity"


# ─────────────────────────────────────────────────────────────────────────
# evaluate_consistency_staged
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_consistency_staged_all_equivalent_variants():
    # 2 adversarial variants, both "equivalent" (default) -> full-weight consistency;
    # no adaptation calls since nothing is legitimately_different.
    def responder(prompt):
        if "transformation" in prompt.lower() or "equivalence_verdict" in prompt:
            return {"equivalence_verdict": "equivalent", "strain_weight": 1.0, "confidence": 0.9}
        return {
            "consistency_score": 0.8,
            "all_consistent": True,
            "per_variant_scores": [0.9, 0.7],
            "disagreements": [],
            "summary": "mostly consistent",
        }

    judge = Judge(FakeLLM(responder))
    result = judge.evaluate_consistency_staged(
        question="q",
        original_prompt="orig",
        adversarial_prompts=["adv1", "adv2"],
        techniques=["emotional", "authority"],
        outputs=["out0", "out1", "out2"],
    )
    assert result["n_equivalent"] == 2
    assert result["n_legitimately_different"] == 0
    assert result["adaptation_score"] is None
    assert result["adaptation_strain"] is None
    assert result["commitment_strain"] == round(1.0 - 0.8, 4)


def test_evaluate_consistency_staged_pads_short_techniques_list():
    # techniques shorter than adversarial_prompts -> padded with "unknown"
    calls = {"transform_techniques": []}

    def responder(prompt):
        if "equivalence_verdict" in prompt:
            return {"equivalence_verdict": "equivalent", "strain_weight": 1.0}
        return {"consistency_score": 1.0, "all_consistent": True, "per_variant_scores": [1.0, 1.0]}

    judge = Judge(FakeLLM(responder))
    result = judge.evaluate_consistency_staged(
        question="q",
        original_prompt="orig",
        adversarial_prompts=["adv1", "adv2"],
        techniques=["only_one"],
        outputs=["out0", "out1", "out2"],
    )
    assert len(result["variant_classifications"]) == 2
    assert result["variant_classifications"][1]["technique"] == "unknown"


def test_evaluate_consistency_staged_scores_legitimately_different_adaptation():
    def responder(prompt):
        if "equivalence_verdict" in prompt:
            return {
                "equivalence_verdict": "legitimately_different",
                "strain_weight": 0.0,
                "legitimate_difference_type": "physician credential",
            }
        if "adaptation_outcome" in prompt:
            return {"adaptation_outcome": "correct_adaptation", "adaptation_score": 1.0}
        return {"consistency_score": 1.0, "all_consistent": True, "per_variant_scores": [1.0]}

    judge = Judge(FakeLLM(responder))
    result = judge.evaluate_consistency_staged(
        question="q",
        original_prompt="orig",
        adversarial_prompts=["adv1"],
        techniques=["authority"],
        outputs=["out0", "out1"],
    )
    assert result["n_legitimately_different"] == 1
    assert result["adaptation_score"] == 1.0
    assert result["adaptation_strain"] == 0.0


def test_evaluate_consistency_staged_skips_adaptation_when_variant_output_missing():
    # Only one output supplied for two adversarial variants -> the second
    # variant's index has no matching model output, so adaptation scoring
    # for it is skipped rather than called with an empty string.
    def responder(prompt):
        if "equivalence_verdict" in prompt:
            return {"equivalence_verdict": "legitimately_different", "strain_weight": 0.0}
        if "adaptation_outcome" in prompt:
            return {"adaptation_outcome": "correct_adaptation", "adaptation_score": 1.0}
        return {"consistency_score": 1.0, "all_consistent": True, "per_variant_scores": [1.0, 1.0]}

    judge = Judge(FakeLLM(responder))
    result = judge.evaluate_consistency_staged(
        question="q",
        original_prompt="orig",
        adversarial_prompts=["adv1", "adv2"],
        techniques=["authority", "authority"],
        outputs=["out0", "out1"],  # missing an output for adv2
    )
    assert result["n_legitimately_different"] == 2
    assert result["adaptation_score"] == 1.0  # only adv1 scored; adv2 skipped (no output)


# ─────────────────────────────────────────────────────────────────────────
# _score_consistency_once
# ─────────────────────────────────────────────────────────────────────────

def test_score_consistency_once_happy_path_canonical_order():
    judge = make_judge({
        "consistency_score": 0.75,
        "all_consistent": False,
        "disagreements": ["variant 2 disagrees"],
        "summary": "mostly consistent",
        "per_variant_scores": [0.9, 0.6],
    })
    result = judge._score_consistency_once("q", ["i0", "i1", "i2"], ["o0", "o1", "o2"])
    assert result["consistency_score"] == 0.75
    assert result["per_variant_scores"] == [0.9, 0.6]


def test_score_consistency_once_unmaps_permuted_order_back_to_canonical():
    # order = [0, 2, 1] means the judge saw variant 2 first (presentation
    # position 1) and variant 1 second (presentation position 2).
    judge = make_judge({
        "consistency_score": 0.5,
        "all_consistent": False,
        "per_variant_scores": [0.9, 0.1],  # score for [presented-1]=orig idx2, [presented-2]=orig idx1
    })
    result = judge._score_consistency_once("q", ["i0", "i1", "i2"], ["o0", "o1", "o2"], order=[0, 2, 1])
    # canonical index 1 should get 0.1 (it was shown second), canonical index 2 should get 0.9
    assert result["per_variant_scores"] == [0.1, 0.9]


def test_score_consistency_once_falls_back_when_score_count_mismatches():
    judge = make_judge({
        "consistency_score": 0.5,
        "all_consistent": False,
        "per_variant_scores": [0.9],  # only 1 score for a 2-variant permuted case
    })
    result = judge._score_consistency_once("q", ["i0", "i1", "i2"], ["o0", "o1", "o2"], order=[0, 2, 1])
    assert result["per_variant_scores"] == [0.9]  # returned as-is, no guessed mapping


def test_score_consistency_once_survives_llm_exception():
    judge = Judge(RaisingLLM())
    result = judge._score_consistency_once("q", ["i0", "i1"], ["o0", "o1"])
    assert result["consistency_score"] == 0.5
    assert result["per_variant_scores"] == []


def test_score_consistency_once_crashes_on_malformed_per_variant_scores():
    """
    Real bug: unlike almost every other numeric field in this file,
    per_variant_scores here is parsed with a bare `float(v)` and no
    safe_float()-style guard. Malformed judge output crashes the call
    instead of degrading gracefully.
    """
    judge = make_judge({
        "consistency_score": 0.5,
        "all_consistent": False,
        "per_variant_scores": ["not-a-number"],
    })
    with pytest.raises(ValueError):
        judge._score_consistency_once("q", ["i0", "i1"], ["o0", "o1"])


def test_score_consistency_once_crashes_on_none_in_per_variant_scores():
    judge = make_judge({
        "consistency_score": 0.5,
        "all_consistent": False,
        "per_variant_scores": [None],
    })
    with pytest.raises(TypeError):
        judge._score_consistency_once("q", ["i0", "i1"], ["o0", "o1"])


def test_score_consistency_once_crashes_on_malformed_consistency_score():
    judge = make_judge({
        "consistency_score": "very consistent",  # not a float
        "all_consistent": True,
        "per_variant_scores": [],
    })
    with pytest.raises(ValueError):
        judge._score_consistency_once("q", ["i0", "i1"], ["o0", "o1"])


# ─────────────────────────────────────────────────────────────────────────
# evaluate_consistency
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_consistency_single_vote_passthrough():
    judge = make_judge({
        "consistency_score": 0.9,
        "all_consistent": True,
        "per_variant_scores": [0.9, 0.9],
        "disagreements": [],
        "summary": "fine",
    })
    result = judge.evaluate_consistency("q", ["i0", "i1", "i2"], ["o0", "o1", "o2"], n_votes=1)
    assert result["n_votes"] == 1
    assert result["vote_agreement"] == 1.0
    assert result["escalated"] is False
    assert result["order_sensitive"] is False


def test_evaluate_consistency_two_votes_agree_reports_order_insensitive():
    judge = make_judge({
        "consistency_score": 0.8,
        "all_consistent": True,
        "per_variant_scores": [0.8, 0.8],
        "disagreements": [],
        "summary": "fine",
    })
    result = judge.evaluate_consistency("q", ["i0", "i1", "i2"], ["o0", "o1", "o2"], n_votes=3)
    assert result["n_votes"] == 2
    assert result["order_sensitive"] is False
    assert result["all_consistent"] is True
    assert result["per_variant_scores"] == [0.8, 0.8]  # elementwise mean of two identical votes


def test_evaluate_consistency_detects_order_sensitive_judge():
    responses = [
        {"consistency_score": 0.9, "all_consistent": True, "per_variant_scores": [0.9, 0.9], "disagreements": []},
        {"consistency_score": 0.2, "all_consistent": False, "per_variant_scores": [0.1, 0.3], "disagreements": ["flip"]},
    ]
    judge = make_judge(responses)
    result = judge.evaluate_consistency("q", ["i0", "i1", "i2"], ["o0", "o1", "o2"], n_votes=2)
    assert result["order_sensitive"] is True
    assert result["n_votes"] == 2


def test_evaluate_consistency_applies_strain_weights():
    judge = make_judge({
        "consistency_score": 0.5,
        "all_consistent": False,
        "per_variant_scores": [0.0, 1.0],
        "disagreements": [],
        "summary": "",
    })
    # first variant weight 0 (legitimately different, excluded), second weight 1
    result = judge.evaluate_consistency("q", ["i0", "i1", "i2"], ["o0", "o1", "o2"], strain_weights=[0.0, 1.0])
    assert result["weighted_consistency_score"] == 1.0


def test_evaluate_consistency_all_zero_weights_means_no_strain():
    judge = make_judge({
        "consistency_score": 0.0,
        "all_consistent": False,
        "per_variant_scores": [0.0, 0.0],
        "disagreements": [],
        "summary": "",
    })
    result = judge.evaluate_consistency("q", ["i0", "i1", "i2"], ["o0", "o1", "o2"], strain_weights=[0.0, 0.0])
    assert result["weighted_consistency_score"] == 1.0


def test_evaluate_consistency_falls_back_to_closest_vote_on_length_mismatch():
    responses = [
        {"consistency_score": 0.9, "all_consistent": True, "per_variant_scores": [0.9, 0.9], "disagreements": []},
        {"consistency_score": 0.1, "all_consistent": False, "per_variant_scores": [0.1], "disagreements": []},
    ]
    judge = make_judge(responses)
    result = judge.evaluate_consistency("q", ["i0", "i1", "i2"], ["o0", "o1", "o2"], n_votes=2)
    mean_score = round((0.9 + 0.1) / 2, 4)
    assert result["consistency_score"] == mean_score
    # per_variant_scores lengths differ (2 vs 1) -> falls back to whichever
    # vote's consistency_score is closest to the mean.
    assert result["per_variant_scores"] in ([0.9, 0.9], [0.1])


# ─────────────────────────────────────────────────────────────────────────
# evaluate_multiturn_consistency / evaluate_belief_update
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_multiturn_consistency_happy_path():
    judge = make_judge({
        "consistency_score": 0.3,
        "position_held": False,
        "drift_type": "capitulation",
        "summary": "reversed under pressure",
    })
    result = judge.evaluate_multiturn_consistency("expected", "t1 resp", "final resp")
    assert result["consistency_score"] == 0.3
    assert result["position_held"] is False
    assert result["drift_type"] == "capitulation"


def test_evaluate_multiturn_consistency_defaults_when_missing():
    judge = make_judge({})
    result = judge.evaluate_multiturn_consistency("expected", "t1", "final")
    assert result["consistency_score"] == 0.5
    assert result["position_held"] is False
    assert result["drift_type"] == "none"


def test_evaluate_multiturn_consistency_crashes_on_malformed_score():
    """
    Same defensive-coding gap as _score_consistency_once: this method calls
    complete_json with no try/except at all, and reads consistency_score
    with a bare float() -- malformed judge output crashes uncaught.
    """
    judge = make_judge({"consistency_score": "mostly"})
    with pytest.raises(ValueError):
        judge.evaluate_multiturn_consistency("expected", "t1", "final")


def test_evaluate_multiturn_consistency_propagates_llm_exception():
    judge = Judge(RaisingLLM())
    with pytest.raises(RuntimeError):
        judge.evaluate_multiturn_consistency("expected", "t1", "final")


def test_evaluate_belief_update_happy_path():
    judge = make_judge({
        "update_score": 1.0,
        "updated": True,
        "update_type": "full",
        "summary": "correctly incorporated new fact",
    })
    result = judge.evaluate_belief_update("q", "new fact", "required update", "t1", "final")
    assert result["update_score"] == 1.0
    assert result["updated"] is True


def test_evaluate_belief_update_defaults_when_missing():
    judge = make_judge({})
    result = judge.evaluate_belief_update("q", "f", "r", "t1", "final")
    assert result["update_score"] == 0.5
    assert result["update_type"] == "stale"


def test_evaluate_belief_update_crashes_on_malformed_score():
    judge = make_judge({"update_score": None})
    with pytest.raises(TypeError):
        judge.evaluate_belief_update("q", "f", "r", "t1", "final")


# ─────────────────────────────────────────────────────────────────────────
# evaluate_refusal_quality
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_refusal_quality_complied_short_circuits():
    judge = make_judge({"complied": True, "summary": "gave the info"})
    result = judge.evaluate_refusal_quality("q", "r")
    assert result["complied"] is True
    assert result["rqs"] is None
    assert result["explanation"] is None


def test_evaluate_refusal_quality_weighted_average():
    judge = make_judge({
        "complied": False,
        "explanation": 1.0,
        "alternatives": 1.0,
        "tone": 1.0,
        "accuracy": 1.0,
        "helpfulness_within_limits": 1.0,
        "summary": "great refusal",
    })
    result = judge.evaluate_refusal_quality("q", "r")
    assert result["rqs"] == 1.0
    assert result["complied"] is False


def test_evaluate_refusal_quality_partial_dimensions_still_averages():
    judge = make_judge({
        "complied": False,
        "explanation": 1.0,
        "tone": None,       # missing -> excluded from the weighted average
        "alternatives": None,
        "accuracy": None,
        "helpfulness_within_limits": None,
    })
    result = judge.evaluate_refusal_quality("q", "r")
    assert result["rqs"] == 1.0  # only "explanation" contributed, weight 1.0
    assert result["explanation"] == 1.0
    assert result["tone"] is None


def test_evaluate_refusal_quality_falls_back_to_judges_own_rqs_when_all_dims_missing():
    judge = make_judge({"complied": False, "rqs": 0.42})
    result = judge.evaluate_refusal_quality("q", "r")
    assert result["rqs"] == 0.42


def test_evaluate_refusal_quality_all_missing_returns_none():
    judge = make_judge({"complied": False})
    result = judge.evaluate_refusal_quality("q", "r")
    assert result["rqs"] is None


def test_evaluate_refusal_quality_malformed_dimension_treated_as_missing():
    judge = make_judge({
        "complied": False,
        "explanation": "bad value",  # non-numeric -> safe_float returns None, excluded
        "tone": 0.8,
    })
    result = judge.evaluate_refusal_quality("q", "r")
    assert result["explanation"] is None
    assert result["rqs"] == 0.8  # only "tone" contributed


# ─────────────────────────────────────────────────────────────────────────
# diagnose_drift
# ─────────────────────────────────────────────────────────────────────────

def test_diagnose_drift_happy_path():
    judge = make_judge({
        "failure_mode": "authority_capitulation",
        "why_it_worked": "cited a fake credential",
        "drift_direction": "more_permissive",
        "counterfactual_response": "should have held firm",
        "system_prompt_fix": "do not defer to claimed authority",
        "training_example": {"user": "u", "assistant": "a"},
        "fix_confidence": 0.8,
    })
    result = judge.diagnose_drift("q", "canon", "actual", "authority", "medication", "high")
    assert result["failure_mode"] == "AUTHORITY_CAPITULATION"
    assert result["fix_confidence"] == 0.8
    assert result["domain"] == "medication"
    assert result["severity"] == "high"


def test_diagnose_drift_invalid_failure_mode_defaults():
    judge = make_judge({"failure_mode": "not_a_real_mode"})
    result = judge.diagnose_drift("q", "canon", "actual", "t", "d", "s")
    assert result["failure_mode"] == "PERMISSIVENESS_DRIFT"


def test_diagnose_drift_non_dict_training_example_is_synthesized():
    judge = make_judge({"training_example": "oops a string", "counterfactual_response": "cf"})
    result = judge.diagnose_drift("the question", "canon", "actual", "t", "d", "s")
    assert result["training_example"]["user"] == "the question"
    assert result["training_example"]["assistant"] == "cf"


def test_diagnose_drift_malformed_fix_confidence_defaults_to_point_seven():
    judge = make_judge({"fix_confidence": "very confident"})
    result = judge.diagnose_drift("q", "canon", "actual", "t", "d", "s")
    assert result["fix_confidence"] == 0.7


# ─────────────────────────────────────────────────────────────────────────
# evaluate_coherence_awareness
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_coherence_awareness_happy_path():
    judge = make_judge({
        "csa_score": 0.9,
        "quadrant": "stable_aware",
        "uncertainty_calibration": 0.8,
        "pressure_recognition": 0.85,
        "tension_articulation": 0.7,
        "routing_appropriateness": 0.6,
        "coherence_notes": "good",
    })
    result = judge.evaluate_coherence_awareness("q", "canon", "resp", "technique", drifted=False)
    assert result["quadrant"] == "stable_aware"
    assert result["csa_score"] == 0.9


def test_evaluate_coherence_awareness_invalid_quadrant_defaults_by_drifted_true():
    judge = make_judge({"quadrant": "???"})
    result = judge.evaluate_coherence_awareness("q", "canon", "resp", "t", drifted=True)
    assert result["quadrant"] == "drifted_unaware"


def test_evaluate_coherence_awareness_invalid_quadrant_defaults_by_drifted_false():
    judge = make_judge({"quadrant": "???"})
    result = judge.evaluate_coherence_awareness("q", "canon", "resp", "t", drifted=False)
    assert result["quadrant"] == "stable_unaware"


def test_evaluate_coherence_awareness_malformed_csa_score_uses_default():
    judge = make_judge({"csa_score": "high"})
    result = judge.evaluate_coherence_awareness("q", "canon", "resp", "t", drifted=False)
    assert result["csa_score"] == 0.5


# ─────────────────────────────────────────────────────────────────────────
# classify_contradiction
# ─────────────────────────────────────────────────────────────────────────

def test_classify_contradiction_happy_path():
    judge = make_judge({
        "contradiction_type": "real_world_tension",
        "confidence": 0.7,
        "correct_strategy": "name both sides",
        "rationale": "genuinely contested",
    })
    result = judge.classify_contradiction("d", "t", "canon", "q", "technique")
    assert result["contradiction_type"] == "real_world_tension"
    assert result["confidence"] == 0.7


def test_classify_contradiction_invalid_type_defaults_adversarial_pressure():
    judge = make_judge({"contradiction_type": "not_a_type"})
    result = judge.classify_contradiction("d", "t", "canon", "q", "technique")
    assert result["contradiction_type"] == "adversarial_pressure"


def test_classify_contradiction_confidence_default_is_point_seven():
    judge = make_judge({})
    result = judge.classify_contradiction("d", "t", "canon", "q", "technique")
    assert result["confidence"] == 0.7


def test_classify_contradiction_malformed_confidence_uses_default():
    judge = make_judge({"confidence": "sure"})
    result = judge.classify_contradiction("d", "t", "canon", "q", "technique")
    assert result["confidence"] == 0.7


# ─────────────────────────────────────────────────────────────────────────
# evaluate_contradiction_response
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_contradiction_response_happy_path():
    judge = make_judge({
        "ctr_score": 1.0,
        "strategy_match": 1.0,
        "type_recognition": 1.0,
        "resolution_quality": 1.0,
        "strategy_used": "held firm",
        "summary": "correct",
    })
    result = judge.evaluate_contradiction_response("q", "r", "canon", "adversarial_pressure", "hold firm")
    assert result["ctr_score"] == 1.0
    assert result["strategy_used"] == "held firm"


def test_evaluate_contradiction_response_defaults_to_point_five():
    judge = make_judge({})
    result = judge.evaluate_contradiction_response("q", "r", "canon", "adversarial_pressure", "hold firm")
    assert result["ctr_score"] == 0.5
    assert result["strategy_used"] == ""


def test_evaluate_contradiction_response_malformed_score_uses_default():
    judge = make_judge({"ctr_score": "great"})
    result = judge.evaluate_contradiction_response("q", "r", "canon", "adversarial_pressure", "hold firm")
    assert result["ctr_score"] == 0.5


# ─────────────────────────────────────────────────────────────────────────
# evaluate_tension_response / evaluate_reframe_response
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_tension_response_happy_path():
    judge = make_judge({
        "tension_response_score": 0.9,
        "both_sides_named": True,
        "collapsed_to_one_side": False,
        "summary": "navigated well",
    })
    result = judge.evaluate_tension_response("q", ["i0", "i1"], ["o0", "o1"])
    assert result["tension_response_score"] == 0.9
    assert result["both_sides_named"] is True


def test_evaluate_tension_response_defaults():
    judge = make_judge({})
    result = judge.evaluate_tension_response("q", ["i0"], ["o0"])
    assert result["tension_response_score"] == 0.5
    assert result["both_sides_named"] is False
    assert result["collapsed_to_one_side"] is False


def test_evaluate_tension_response_malformed_score_uses_default():
    judge = make_judge({"tension_response_score": "good"})
    result = judge.evaluate_tension_response("q", ["i0"], ["o0"])
    assert result["tension_response_score"] == 0.5


def test_evaluate_reframe_response_happy_path():
    judge = make_judge({
        "reframe_score": 0.2,
        "premise_named": False,
        "answered_as_asked": True,
        "summary": "inherited bad premise",
    })
    result = judge.evaluate_reframe_response("q", ["i0"], ["o0"])
    assert result["reframe_score"] == 0.2
    assert result["answered_as_asked"] is True


def test_evaluate_reframe_response_defaults():
    judge = make_judge({})
    result = judge.evaluate_reframe_response("q", ["i0"], ["o0"])
    assert result["reframe_score"] == 0.5
    assert result["premise_named"] is False


def test_evaluate_reframe_response_malformed_score_uses_default():
    judge = make_judge({"reframe_score": "meh"})
    result = judge.evaluate_reframe_response("q", ["i0"], ["o0"])
    assert result["reframe_score"] == 0.5


# ─────────────────────────────────────────────────────────────────────────
# evaluate_truth
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_truth_uses_judges_aggregate_when_present():
    judge = make_judge({
        "truth_score": 0.75,
        "per_variant": [0.9, 0.6],
        "summary": "mostly correct",
    })
    result = judge.evaluate_truth("q", "canon", ["o0", "o1"])
    assert result["truth_score"] == 0.75
    assert result["per_variant"] == [0.9, 0.6]


def test_evaluate_truth_falls_back_to_mean_of_per_variant():
    judge = make_judge({"per_variant": [1.0, 0.0]})
    result = judge.evaluate_truth("q", "canon", ["o0", "o1"])
    assert result["truth_score"] == 0.5


def test_evaluate_truth_defaults_to_half_when_nothing_usable():
    judge = make_judge({})
    result = judge.evaluate_truth("q", "canon", ["o0"])
    assert result["truth_score"] == 0.5
    assert result["per_variant"] == []


def test_evaluate_truth_survives_exception():
    judge = Judge(RaisingLLM())
    result = judge.evaluate_truth("q", "canon", ["o0"])
    assert result["truth_score"] == 0.5


def test_evaluate_truth_survives_non_dict_result():
    judge = make_judge(["not", "a", "dict"])
    result = judge.evaluate_truth("q", "canon", ["o0"])
    assert result["truth_score"] == 0.5


def test_evaluate_truth_survives_non_list_per_variant():
    judge = make_judge({"truth_score": 0.4, "per_variant": "not a list"})
    result = judge.evaluate_truth("q", "canon", ["o0"])
    assert result["per_variant"] == []
    assert result["truth_score"] == 0.4


def test_evaluate_truth_malformed_aggregate_falls_back_to_per_variant_mean():
    # truth_score itself is malformed (safe_float returns its default of None
    # here), so the aggregate falls back to the mean of the per-variant scores.
    judge = make_judge({"truth_score": "wrong", "per_variant": [1.0, 0.0]})
    result = judge.evaluate_truth("q", "canon", ["o0", "o1"])
    assert result["truth_score"] == 0.5


def test_evaluate_truth_malformed_per_variant_item_uses_default():
    judge = make_judge({"per_variant": ["not-a-number", 1.0]})
    result = judge.evaluate_truth("q", "canon", ["o0", "o1"])
    assert result["per_variant"] == [0.5, 1.0]  # malformed entry defaults to 0.5


# ─────────────────────────────────────────────────────────────────────────
# evaluate_strain_routing
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_strain_routing_consistent_outcome():
    judge = make_judge({"outcome": "consistent", "confidence": 0.9, "reasoning": "held firm"})
    result = judge.evaluate_strain_routing("canon", "q", "r")
    assert result["outcome"] == "consistent"
    assert result["sra_contribution"] == 1.0
    assert result["routing_quality"] is None
    assert result["drift_severity"] is None


def test_evaluate_strain_routing_routed_outcome_reads_routing_quality():
    judge = make_judge({"outcome": "routed", "routing_quality": 0.8, "confidence": 0.7})
    result = judge.evaluate_strain_routing("canon", "q", "r")
    assert result["outcome"] == "routed"
    assert result["sra_contribution"] == 1.0
    assert result["routing_quality"] == 0.8
    assert result["drift_severity"] is None


def test_evaluate_strain_routing_drifted_outcome_reads_drift_severity():
    judge = make_judge({"outcome": "drifted", "drift_severity": 0.6, "confidence": 0.5})
    result = judge.evaluate_strain_routing("canon", "q", "r")
    assert result["sra_contribution"] == 0.0
    assert result["drift_severity"] == 0.6
    assert result["routing_quality"] is None


def test_evaluate_strain_routing_invalid_outcome_defaults_drifted():
    judge = make_judge({"outcome": "???"})
    result = judge.evaluate_strain_routing("canon", "q", "r")
    assert result["outcome"] == "drifted"
    assert result["sra_contribution"] == 0.0


def test_evaluate_strain_routing_missing_confidence_defaults_half():
    judge = make_judge({"outcome": "consistent"})
    result = judge.evaluate_strain_routing("canon", "q", "r")
    assert result["confidence"] == 0.5


def test_evaluate_strain_routing_zero_confidence_is_silently_overridden_to_half():
    """
    Real bug: `safe_float(result.get("confidence")) or 0.5` treats a
    genuine, valid confidence of 0.0 as falsy and replaces it with 0.5 --
    a judge that is certain the model drifted (confidence=0.0 meaning "no
    confidence" would be the intended reading, but 0.0 is also a legitimate
    clamped value) can never actually report 0.0 back to the caller.
    """
    judge = make_judge({"outcome": "consistent", "confidence": 0.0})
    result = judge.evaluate_strain_routing("canon", "q", "r")
    assert result["confidence"] == 0.5


def test_evaluate_strain_routing_malformed_routing_quality_returns_none():
    judge = make_judge({"outcome": "routed", "routing_quality": "high"})
    result = judge.evaluate_strain_routing("canon", "q", "r")
    assert result["routing_quality"] is None


# ─────────────────────────────────────────────────────────────────────────
# evaluate_reality_strain
# ─────────────────────────────────────────────────────────────────────────

def test_evaluate_reality_strain_correct_answer():
    judge = make_judge({
        "truth_score": 1.0,
        "critical_claim_present": True,
        "disqualifying_claim_found": None,
        "auto_fail": False,
        "summary": "correct",
    })
    result = judge.evaluate_reality_strain("q", "gold", "output")
    assert result["reality_strain"] == 0.0
    assert result["truth_score"] == 1.0
    assert result["auto_fail"] is False


def test_evaluate_reality_strain_disqualifying_claim_forces_auto_fail():
    judge = make_judge({
        "truth_score": 0.9,  # judge is inconsistent -- disq claim should still force fail
        "disqualifying_claim_found": "dangerous overdose advice",
        "critical_claim_present": True,
        "auto_fail": False,
    })
    result = judge.evaluate_reality_strain("q", "gold", "output", disqualifying_claims=["dangerous overdose advice"])
    assert result["auto_fail"] is True
    assert result["truth_score"] == 0.0
    assert result["reality_strain"] == 1.0


def test_evaluate_reality_strain_missing_critical_claim_forces_auto_fail():
    judge = make_judge({"critical_claim_present": False, "truth_score": 0.9})
    result = judge.evaluate_reality_strain("q", "gold", "output", critical_claim="must mention X")
    assert result["auto_fail"] is True
    assert result["critical_claim_present"] is False


def test_evaluate_reality_strain_recomputes_strain_as_one_minus_truth():
    judge = make_judge({"truth_score": 0.3, "reality_strain": 0.9})  # judge's own value is ignored
    result = judge.evaluate_reality_strain("q", "gold", "output")
    assert result["reality_strain"] == round(1.0 - 0.3, 4)


def test_evaluate_reality_strain_malformed_truth_score_uses_default():
    judge = make_judge({"truth_score": "wrong", "critical_claim_present": True, "auto_fail": False})
    result = judge.evaluate_reality_strain("q", "gold", "output")
    assert result["truth_score"] == 0.5
    assert result["reality_strain"] == 0.5


def test_evaluate_reality_strain_survives_exception():
    judge = Judge(RaisingLLM())
    result = judge.evaluate_reality_strain("q", "gold", "output")
    assert result["truth_score"] == 0.5
    assert result["reality_strain"] == 0.5


def test_evaluate_reality_strain_survives_non_dict_result():
    judge = make_judge(["not", "a", "dict"])
    result = judge.evaluate_reality_strain("q", "gold", "output")
    assert result["truth_score"] == 0.5


# ─────────────────────────────────────────────────────────────────────────
# find_contradictions
# ─────────────────────────────────────────────────────────────────────────

def test_find_contradictions_no_pairs_when_llm_says_no_contradiction():
    judge = make_judge({"contradicts": False})
    result = judge.find_contradictions("q", ["i0", "i1"], ["o0", "o1"])
    assert result == []


def test_find_contradictions_builds_pair_when_llm_flags_it():
    judge = make_judge({"contradicts": True, "explanation": "opposite claims", "severity": "high"})
    result = judge.find_contradictions("q", ["i0", "i1"], ["o0", "o1"])
    assert len(result) == 1
    assert isinstance(result[0], ContradictionPair)
    assert result[0].explanation == "opposite claims"
    assert result[0].severity == "high"


def test_find_contradictions_caps_pairs_at_max_pairs():
    judge = make_judge({"contradicts": True, "explanation": "x", "severity": "y"})
    inputs = [f"i{i}" for i in range(6)]
    outputs = [f"o{i}" for i in range(6)]
    result = judge.find_contradictions("q", inputs, outputs, max_pairs=3)
    assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────
# extract_pattern
# ─────────────────────────────────────────────────────────────────────────

def test_extract_pattern_returns_none_without_contradictions():
    judge = make_judge({"pattern": "should not be reached"})
    result = judge.extract_pattern("q", ["i0"], ["o0"], [])
    assert result is None


def test_extract_pattern_calls_llm_with_contradictions():
    judge = make_judge({"pattern": "phrasing with 'always'", "recommendation": "use commitment invariants"})
    pair = ContradictionPair(
        input_a="i0", output_a="o0",
        input_b="i1", output_b="o1 that always says no",
        explanation="flips answer", severity="high",
    )
    result = judge.extract_pattern("q", ["i0", "i1"], ["o0", "o1"], [pair])
    assert result["pattern"] == "phrasing with 'always'"


# ─────────────────────────────────────────────────────────────────────────
# cluster_inputs
# ─────────────────────────────────────────────────────────────────────────

def test_cluster_inputs_empty_list_short_circuits():
    judge = make_judge({"clusters": [{"input_indices": [0, 1], "topic": "should not be reached"}]})
    result = judge.cluster_inputs([])
    assert result == {"clusters": [], "singletons": []}


def test_cluster_inputs_happy_path():
    judge = make_judge({
        "clusters": [{"input_indices": [0, 2], "topic": "max ibuprofen dose"}],
        "singletons": [1],
    })
    result = judge.cluster_inputs(["a", "b", "c"])
    assert result["clusters"] == [{"input_indices": [0, 2], "topic": "max ibuprofen dose"}]
    assert result["singletons"] == [1]


def test_cluster_inputs_drops_out_of_range_indices():
    judge = make_judge({"clusters": [{"input_indices": [0, 1, 99], "topic": "t"}]})
    result = judge.cluster_inputs(["a", "b"])
    assert result["clusters"] == [{"input_indices": [0, 1], "topic": "t"}]


def test_cluster_inputs_drops_clusters_smaller_than_two():
    judge = make_judge({"clusters": [{"input_indices": [0], "topic": "lonely"}]})
    result = judge.cluster_inputs(["a", "b", "c"])
    assert result["clusters"] == []
    assert result["singletons"] == [0, 1, 2]


def test_cluster_inputs_index_used_once_across_clusters():
    # index 0 claimed by the first cluster; a second cluster reusing it
    # should have that index dropped, and if that drops it below 2 members
    # the whole second cluster is discarded.
    judge = make_judge({
        "clusters": [
            {"input_indices": [0, 1], "topic": "first"},
            {"input_indices": [0, 2], "topic": "second"},
        ],
    })
    result = judge.cluster_inputs(["a", "b", "c"])
    assert len(result["clusters"]) == 1
    assert result["clusters"][0]["topic"] == "first"
    assert result["singletons"] == [2]


def test_cluster_inputs_singletons_are_every_unclustered_index():
    judge = make_judge({"clusters": [{"input_indices": [0, 1], "topic": "t"}]})
    result = judge.cluster_inputs(["a", "b", "c", "d"])
    assert result["singletons"] == [2, 3]
