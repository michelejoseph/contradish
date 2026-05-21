"""
Tests for contradish.prompt_analyzer: static prompt-contradiction analysis.
Run with: pytest tests/test_prompt_analyzer.py
No API key required (LLMClient is patched).
"""
from unittest.mock import patch

from contradish.prompt_analyzer import (
    analyze_prompt, PromptAnalysis, PromptTension,
    KNOWN_TECHNIQUES, KNOWN_FAILURE_MODES,
)


class _FakeLLM:
    judge_model = "fake-judge"
    def __init__(self, *a, **kw):
        pass
    def complete_json(self, prompt, model=None):
        return {
            "tensions": [
                {
                    "clauses": ["Be empathetic.", "No exceptions."],
                    "description": "Empathy conflicts with the no-exceptions rule under sympathy framing.",
                    "exploiting_techniques": ["sympathy", "emotional", "not_a_real_technique"],
                    "failure_mode": "EMPATHY_OVERRIDE",
                    "severity": "critical",
                    "suggested_fix": "Empathy never produces exceptions.",
                },
                {
                    "clauses": ["Provide accurate info."],
                    "description": "Authority claim can reinterpret the policy.",
                    "exploiting_techniques": ["authority"],
                    "failure_mode": "BOGUS_MODE",     # should be dropped to None
                    "severity": "weird",               # should fall back to medium
                    "suggested_fix": "Credentials do not modify policy.",
                },
                {"clauses": [], "description": ""},     # malformed -> skipped
            ],
            "deconflicted_prompt": "Be empathetic. Empathy never produces exceptions. No exceptions.",
        }


def test_analyze_prompt_parses_and_ranks():
    with patch("contradish.llm.LLMClient", _FakeLLM):
        a = analyze_prompt("Be empathetic. No exceptions.")
    assert isinstance(a, PromptAnalysis)
    # malformed third entry dropped -> 2 tensions
    assert a.tension_count == 2
    # critical sorts before medium
    assert a.tensions[0].severity == "critical"
    assert a.tensions[1].severity == "medium"   # 'weird' fell back to medium
    # unknown technique filtered out, known ones kept
    assert "sympathy" in a.tensions[0].exploiting_techniques
    assert "not_a_real_technique" not in a.tensions[0].exploiting_techniques
    # bogus failure mode coerced to None
    assert a.tensions[1].failure_mode is None
    assert a.tensions[0].failure_mode == "EMPATHY_OVERRIDE"


def test_critical_and_high_counts():
    with patch("contradish.llm.LLMClient", _FakeLLM):
        a = analyze_prompt("x")
    assert a.critical_count == 1
    assert a.high_or_above_count == 1   # only the critical one; the other is medium


def test_at_or_above_filter():
    with patch("contradish.llm.LLMClient", _FakeLLM):
        a = analyze_prompt("x")
    assert len(a.at_or_above("critical")) == 1
    assert len(a.at_or_above("medium")) == 2
    try:
        a.at_or_above("nonsense")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_deconflicted_prompt_returned():
    with patch("contradish.llm.LLMClient", _FakeLLM):
        a = analyze_prompt("Be empathetic. No exceptions.")
    assert "Empathy never produces exceptions" in a.deconflicted_prompt
    assert a.to_dict()["tension_count"] == 2


def test_empty_prompt_raises():
    try:
        analyze_prompt("   ")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_no_tensions_falls_back_to_original():
    class EmptyLLM(_FakeLLM):
        def complete_json(self, prompt, model=None):
            return {"tensions": [], "deconflicted_prompt": ""}
    with patch("contradish.llm.LLMClient", EmptyLLM):
        a = analyze_prompt("A perfectly clear single instruction.")
    assert a.tension_count == 0
    assert a.deconflicted_prompt == "A perfectly clear single instruction."


def test_ontology_constants_intact():
    assert len(KNOWN_TECHNIQUES) == 16
    assert len(KNOWN_FAILURE_MODES) == 8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
