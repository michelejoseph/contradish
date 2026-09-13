"""
Tests for the Decision-Relevance wiring in contradish.bench.evaluate.run_frozen_policy.

No API key required: app and judge are deterministic stubs. Runs against
the real, shipped medication.json frozen benchmark (not a synthetic case
list) so this also catches any drift between the wiring and the actual
data shape run_frozen_policy is fed in production.
"""
from contradish.bench.evaluate import run_frozen_policy


def _stub_app(question: str) -> str:
    return f"response to: {question[:20]}"


class _SpuriousOnEmotionalJudge:
    """No evaluate_commitment_invariance/evaluate_refusal_quality attrs, so
    run_frozen_policy takes the plain legacy consistency path. Makes
    technique index 0 ("emotional") spuriously sensitive (score 0.1 ->
    sensitivity 0.9 >= the 0.5 classification threshold) and every other
    technique solidly invariant (score 0.95 -> sensitivity 0.05)."""

    def evaluate_consistency(self, original, inputs, outputs, strain_weights=None, n_votes=1):
        n_adv = len(inputs) - 1
        per_variant = [0.1 if i == 0 else 0.95 for i in range(n_adv)]
        overall = sum(per_variant) / len(per_variant) if per_variant else 1.0
        return {
            "consistency_score": overall,
            "weighted_consistency_score": overall,
            "all_consistent": False,
            "disagreements": [],
            "summary": "stub",
            "per_variant_scores": per_variant,
            "vote_agreement": 1.0,
            "n_votes": n_votes,
            "order_sensitive": False,
        }


class _NeverSpuriousJudge:
    """Every technique fully invariant (score 1.0 on every variant)."""

    def evaluate_consistency(self, original, inputs, outputs, strain_weights=None, n_votes=1):
        n_adv = len(inputs) - 1
        per_variant = [1.0] * n_adv
        return {
            "consistency_score": 1.0,
            "weighted_consistency_score": 1.0,
            "all_consistent": True,
            "disagreements": [],
            "summary": "stub",
            "per_variant_scores": per_variant,
            "vote_agreement": 1.0,
            "n_votes": n_votes,
            "order_sensitive": False,
        }


def test_new_fields_are_present_on_a_real_frozen_policy_run():
    result = run_frozen_policy("medication", _stub_app, _SpuriousOnEmotionalJudge(), verbose=False)
    assert "technique_spurious_rate" in result
    assert "cases_with_spurious_technique" in result
    assert all("dependency_spurious_techniques" in d for d in result["details"])


def test_only_the_spurious_technique_is_flagged_per_case():
    result = run_frozen_policy("medication", _stub_app, _SpuriousOnEmotionalJudge(), verbose=False)
    for d in result["details"]:
        assert d["dependency_spurious_techniques"] == ["emotional"]


def test_pooled_spurious_rate_matches_one_of_eight_techniques():
    result = run_frozen_policy("medication", _stub_app, _SpuriousOnEmotionalJudge(), verbose=False)
    assert result["technique_spurious_rate"] == round(1 / 8, 4)
    assert len(result["cases_with_spurious_technique"]) == result["total"]


def test_a_perfectly_invariant_model_has_zero_spurious_rate_and_no_flagged_cases():
    result = run_frozen_policy("medication", _stub_app, _NeverSpuriousJudge(), verbose=False)
    assert result["technique_spurious_rate"] == 0.0
    assert result["cases_with_spurious_technique"] == []
    for d in result["details"]:
        assert d["dependency_spurious_techniques"] == []


def test_existing_fields_are_unaffected_by_the_new_wiring():
    result = run_frozen_policy("medication", _stub_app, _SpuriousOnEmotionalJudge(), verbose=False)
    assert result["cai_score"] is not None
    assert result["cai_strain"] is not None
    assert result["total"] == len(result["details"])
    # the technique that's spuriously sensitive should show up with higher
    # aggregate strain than one that never is, in the pre-existing
    # technique_cts breakdown -- unrelated to the new fields, just confirming
    # nothing about the existing computation shifted.
    assert result["technique_cts"]["emotional"] > result["technique_cts"]["presuppose"]
