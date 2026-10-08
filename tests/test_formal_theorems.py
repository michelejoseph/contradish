"""
The theorem checker must have teeth: a scorer with a planted defect has to
make at least one theorem fail. (The positive result, that the real scorer
satisfies T1-T10 and L1-L3, is asserted in test_reference_conformance.py.)

    python -m contradish.reference           # full report, ~40s
    python -m contradish.reference --full    # enlarge the 3-case universe
"""
from fractions import Fraction

from contradish import reference as ref


def _failing(hooks):
    return {r["id"] for r in ref.check_theorems("tiny", hooks=hooks) if not r["ok"]}


def test_unmutated_scorer_satisfies_the_theorems_tiny():
    assert _failing(None) == set()


def test_wrong_fidelity_denominator_violates_skill_score_identity():
    assert "T3" in _failing({"fidelity": lambda off, needed, n: 1 - Fraction(off, n) if n else None})


def test_wrong_preservation_denominator_violates_faithfulness_characterization():
    assert "T1" in _failing({"preservation": lambda k, n, needed: Fraction(k, n) if n else None})


def test_blind_authority_metric_violates_obedience_separation():
    assert "T9" in _failing({"authority_respected": lambda c, p: Fraction(1) if p else None})


def test_clipping_fidelity_violates_monotonicity():
    assert "T5" in _failing({"fidelity": lambda off, needed, n: max(Fraction(0), 1 - Fraction(off, needed)) if needed else None})


def test_converse_of_distinction_perfection_is_false():
    cs = [{"id": "c0", "before": "a", "after": "a"}, {"id": "c1", "before": "b", "after": "b"}]
    r = ref.score(cs, {"c0": "a", "c1": "b"}, {"c0": "b", "c1": "a"})
    assert r["persistence"] == 1 and not r["faithful"]      # a relabeling preserves every distinction and is wrong


def test_mean_of_fidelities_is_not_the_pooled_fidelity():
    A = [{"id": "x", "before": "a", "after": "b"}]
    B = [{"id": f"y{i}", "before": "a", "after": "b"} for i in range(3)]
    pooled = ref.score(A + B, {"x": "a", **{f"y{i}": "a" for i in range(3)}}, {"x": "b", **{f"y{i}": "a" for i in range(3)}})
    assert pooled["fidelity"] == Fraction(1, 4)
