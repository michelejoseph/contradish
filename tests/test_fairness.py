"""
Tests for contradish.fairness: disparate-treatment audit across protected attributes.
Run with: pytest tests/test_fairness.py
No API key required (LLMClient and Judge are patched).
"""
from unittest.mock import patch

from contradish.fairness import audit_fairness, FairnessAudit, IdentityProfile, DEFAULT_PROFILES


class _FakeLLM:
    def __init__(self, *a, **kw):
        pass


class _FakeJudge:
    """Consistency = 1.0 if the two answers are identical, else 0.2."""
    def __init__(self, llm):
        pass
    def evaluate_consistency(self, question, inputs, outputs):
        a, b = outputs
        return {"consistency_score": 1.0 if a == b else 0.2}


def _biased_on_disability(q):
    if "cognitive disability" in q.lower():
        return "I'd rather not get into the details."
    return "The standard answer is X."


def _fair_app(q):
    return "The standard answer is X, regardless of who is asking."


def test_audit_isolates_discriminated_attribute():
    with patch("contradish.llm.LLMClient", _FakeLLM), patch("contradish.judge.Judge", _FakeJudge):
        audit = audit_fairness(
            app=_biased_on_disability,
            cases=["What are my options?", "Explain the policy."],
            verbose=False, concurrency=1,
        )
    assert isinstance(audit, FairnessAudit)
    assert audit.worst_attribute == "disability"
    # disability shifted; everything else flat
    assert audit.per_attribute["disability"] > 0.5
    for attr, shift in audit.per_attribute.items():
        if attr != "disability":
            assert shift == 0.0
    assert len(audit.flagged) >= 2


def test_fair_app_produces_no_flags():
    with patch("contradish.llm.LLMClient", _FakeLLM), patch("contradish.judge.Judge", _FakeJudge):
        audit = audit_fairness(
            app=_fair_app,
            cases=["What are my options?"],
            verbose=False, concurrency=1,
        )
    assert audit.flagged == []
    assert audit.max_shift == 0.0
    assert "No disparate treatment" in audit.summary()


def test_to_dict_shape():
    with patch("contradish.llm.LLMClient", _FakeLLM), patch("contradish.judge.Judge", _FakeJudge):
        audit = audit_fairness(app=_biased_on_disability, cases=["q?"],
                               verbose=False, concurrency=1)
    d = audit.to_dict()
    assert "per_attribute" in d and "flagged" in d and "worst_attribute" in d
    assert d["worst_attribute"] == "disability"


def test_ranked_attributes_sorted_worst_first():
    with patch("contradish.llm.LLMClient", _FakeLLM), patch("contradish.judge.Judge", _FakeJudge):
        audit = audit_fairness(app=_biased_on_disability, cases=["q?"],
                               verbose=False, concurrency=1)
    ranked = audit.ranked_attributes()
    shifts = [r["mean_shift"] for r in ranked]
    assert shifts == sorted(shifts, reverse=True)
    assert ranked[0]["attribute"] == "disability"


def test_identity_profile_apply():
    p = IdentityProfile(name="x", attribute="age", label="l", prefix="I'm 16.")
    assert p.apply("What now?") == "I'm 16. What now?"
    p2 = IdentityProfile(name="y", attribute="ses", label="l", suffix="I can't afford a pro.")
    assert p2.apply("What now?") == "What now? I can't afford a pro."


def test_default_profiles_cover_protected_axes():
    attrs = {p.attribute for p in DEFAULT_PROFILES}
    # the protected/relevant axes the regulations care about
    for expected in ("age", "national_origin", "disability", "socioeconomic"):
        assert expected in attrs


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
