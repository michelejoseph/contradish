"""
Tests for contradish.judge_calibration: the judge-floor measurement.
Run with: pytest tests/test_judge_calibration.py
No API key required (LLMClient is patched).
"""
from unittest.mock import patch

from contradish.judge_calibration import measure_judge_floor, JudgeCalibration


class _StableLLM:
    """A perfectly stable judge: same verdict every rephrasing."""
    provider = "openai"
    judge_model = "stable-judge"
    def __init__(self, *a, **kw):
        pass
    def complete_json(self, prompt, model=None):
        return {"equivalent": True}


class _FlippyLLM:
    """A maximally unstable judge: alternates verdicts every call."""
    provider = "openai"
    judge_model = "flippy-judge"
    _n = 0
    def __init__(self, *a, **kw):
        pass
    def complete_json(self, prompt, model=None):
        _FlippyLLM._n += 1
        return {"equivalent": bool(_FlippyLLM._n % 2)}


def test_stable_judge_has_zero_floor():
    with patch("contradish.llm.LLMClient", _StableLLM):
        cal = measure_judge_floor(judge_provider="openai", n_rephrasings=2, concurrency=1)
    assert isinstance(cal, JudgeCalibration)
    assert cal.n_pairs == 24
    assert cal.n_rephrasings == 2
    # A judge that never changes its verdict across rephrasings has floor 0.
    assert cal.floor_strain == 0.0
    assert cal.confidence_floor == 0.0
    # Always-True judge: correct on the 12 equivalent pairs, wrong on the 12
    # non-equivalent pairs -> accuracy ~0.5
    assert abs(cal.accuracy - 0.5) < 0.05


def test_flippy_judge_has_nonzero_floor():
    _FlippyLLM._n = 0
    with patch("contradish.llm.LLMClient", _FlippyLLM):
        cal = measure_judge_floor(judge_provider="openai", n_rephrasings=2, concurrency=1)
    assert cal.floor_strain > 0.0
    assert cal.confidence_floor >= cal.floor_strain


def test_to_dict_shape():
    with patch("contradish.llm.LLMClient", _StableLLM):
        cal = measure_judge_floor(judge_provider="openai", n_rephrasings=1, concurrency=1)
    d = cal.to_dict()
    for k in ("judge_provider", "judge_model", "floor_strain", "confidence_floor", "accuracy", "per_pair"):
        assert k in d


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
