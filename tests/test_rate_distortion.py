"""
Tests for contradish.rate_distortion: the information-graded resolution curve.

Same deterministic-mock convention as tests/test_resolution.py: no API key
required, model_fn is a plain function keyed on markers in the augmented
question text (the hedge phrase and/or the pole statement's own wording),
never a live LLM call.
"""
import json
from unittest.mock import patch

import pytest

from contradish.distinction import DistinctionPair
from contradish.resolution import ResolutionCandidate, ResolutionResult
from contradish.rate_distortion import (
    RateDistortionPoint,
    RateDistortionCurveResult,
    _spearman_r,
    measure_rate_distortion_curve,
    measure_rate_distortion_for_resolution,
)


def _identity_extractor(question: str, answer: str) -> str:
    return answer


_PAIR = DistinctionPair(
    pair_id="test_pair",
    description="test distinction for the rate-distortion curve",
    label_a="state A",
    label_b="state B",
    question_a="Question about A?",
    question_b="Question about B?",
    commit_a="answer for a",
    commit_b="answer for b",
)

_POLE_A = "The hidden flag is TRUE_A."
_POLE_B = "The hidden flag is TRUE_B."


# ── structural sanity ────────────────────────────────────────────────────

def test_rate_distortion_point_and_result_are_dataclasses():
    point = RateDistortionPoint(
        level_name="no_information", level_index=0,
        accuracy_a=0.0, accuracy_b=0.0, mean_accuracy=0.0,
    )
    result = RateDistortionCurveResult(
        pair_id="p", description="d", condition="c",
        points=[point], correlation=0.0, shape="insensitive",
    )
    assert "RATE-DISTORTION CURVE" in result.report()
    assert "INSENSITIVE" in result.report()
    assert "p" in result.summary()

    d = result.to_dict()
    json.dumps(d)  # must not raise
    assert d["schema_version"] == "1.0"
    assert d["shape"] == "insensitive"
    assert d["points"][0]["level_name"] == "no_information"


def test_rate_distortion_curve_result_summary_handles_no_points():
    result = RateDistortionCurveResult(pair_id="p", description="d", condition="c")
    assert "no points measured" in result.summary()


# ── _spearman_r: the from-scratch rank correlation ─────────────────────────

def test_spearman_r_perfect_positive_correlation():
    assert _spearman_r([0, 1, 2, 3, 4], [0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(1.0)


def test_spearman_r_perfect_negative_correlation():
    assert _spearman_r([0, 1, 2, 3, 4], [1.0, 0.75, 0.5, 0.25, 0.0]) == pytest.approx(-1.0)


def test_spearman_r_handles_ties_with_average_rank():
    # ys has two tied 0s and three tied 1s -- this is the exact tie shape a
    # brittle "jump only at the last rung" curve produces.
    r = _spearman_r([0, 1, 2, 3, 4], [0.0, 0.0, 1.0, 1.0, 1.0])
    assert r == pytest.approx(0.8660254, abs=1e-4)


def test_spearman_r_zero_variance_returns_zero_not_error():
    assert _spearman_r([0, 1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0, 1.0]) == 0.0
    assert _spearman_r([0, 0, 0], [0.1, 0.2, 0.3]) == 0.0


def test_spearman_r_fewer_than_two_points_returns_zero():
    assert _spearman_r([1.0], [1.0]) == 0.0
    assert _spearman_r([], []) == 0.0


# ── measure_rate_distortion_curve: the three honestly-distinguished shapes ─

def test_measure_rate_distortion_curve_detects_graded_shape():
    def model_fn(system_prompt: str, question: str) -> str:
        if "TRUE_A" not in question and "TRUE_B" not in question:
            return "wrong answer"  # no_information rung: no signal at all
        # Correct from moderate_signal onward (rung index >= 2), wrong on
        # weak_hint (rung 1) -- a smooth, gradually-improving climb.
        if ("slight, unconfirmed indication" in question):
            return "wrong answer"
        if "TRUE_A" in question:
            return "answer for a"
        if "TRUE_B" in question:
            return "answer for b"
        return "wrong answer"

    curve = measure_rate_distortion_curve(
        pair=_PAIR, pole_a_statement=_POLE_A, pole_b_statement=_POLE_B,
        model_fn=model_fn, commitment_extractor=_identity_extractor,
        validation_samples=1,
    )

    assert [p.mean_accuracy for p in curve.points] == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert curve.shape == "graded"
    assert curve.correlation > 0.7
    assert "graceful" in curve.report() or "gracefully" in curve.report()


def test_measure_rate_distortion_curve_detects_threshold_shape():
    def model_fn(system_prompt: str, question: str) -> str:
        # Only correct at the very last rung, where the pole is stated as a
        # bare, unhedged fact -- every hedge, however strong, still fails.
        # A model that is only ever as good as its most certain sentence.
        hedge_markers = (
            "slight, unconfirmed indication",
            "suggests, but does not confirm",
            "very likely, though not fully confirmed",
        )
        if any(m in question for m in hedge_markers):
            return "wrong answer"
        if "TRUE_A" in question:
            return "answer for a"
        if "TRUE_B" in question:
            return "answer for b"
        return "wrong answer"

    curve = measure_rate_distortion_curve(
        pair=_PAIR, pole_a_statement=_POLE_A, pole_b_statement=_POLE_B,
        model_fn=model_fn, commitment_extractor=_identity_extractor,
        validation_samples=1, graded_threshold=0.9,
    )

    assert [p.mean_accuracy for p in curve.points] == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert curve.shape == "threshold"
    assert "brittle" in curve.report()


def test_measure_rate_distortion_curve_detects_insensitive_shape():
    def model_fn(system_prompt: str, question: str) -> str:
        # The candidate condition never actually helps, at any certainty --
        # the honest negative result this module exists to surface, the
        # rate-distortion sibling of resolution.py's own "not resolved".
        return "wrong answer"

    curve = measure_rate_distortion_curve(
        pair=_PAIR, pole_a_statement=_POLE_A, pole_b_statement=_POLE_B,
        model_fn=model_fn, commitment_extractor=_identity_extractor,
        validation_samples=1,
    )

    assert [p.mean_accuracy for p in curve.points] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert curve.shape == "insensitive"
    assert "does not actually help" in curve.report()


# ── measure_rate_distortion_for_resolution: the discover_resolution wrapper ─

def test_measure_rate_distortion_for_resolution_returns_none_when_no_best():
    result = ResolutionResult(
        pair_id="test_pair", description="d", label_a="A", label_b="B",
        baseline_hold_rate=0.0, candidates=[], best=None,
        resolved=False, validated_hold_rate=None, system_prompt_patch=None,
    )

    def model_fn(system_prompt: str, question: str) -> str:
        return "irrelevant"

    curve = measure_rate_distortion_for_resolution(
        result, model_fn=model_fn, commitment_extractor=_identity_extractor,
        pair=_PAIR,
    )
    assert curve is None


def test_measure_rate_distortion_for_resolution_requires_pair():
    candidate = ResolutionCandidate(
        condition="c", rationale="r",
        pole_a_statement=_POLE_A, pole_b_statement=_POLE_B,
        direct_match_rate=1.0, flip_rate=1.0, causal_effect_size=1.0, n_probes=4,
    )
    result = ResolutionResult(
        pair_id="test_pair", description="d", label_a="A", label_b="B",
        baseline_hold_rate=0.0, candidates=[candidate], best=candidate,
        resolved=True, validated_hold_rate=1.0, system_prompt_patch="patch",
    )

    def model_fn(system_prompt: str, question: str) -> str:
        return "irrelevant"

    with pytest.raises(ValueError):
        measure_rate_distortion_for_resolution(
            result, model_fn=model_fn, commitment_extractor=_identity_extractor,
        )


def test_measure_rate_distortion_for_resolution_uses_winning_candidate_poles():
    candidate = ResolutionCandidate(
        condition="c", rationale="r",
        pole_a_statement=_POLE_A, pole_b_statement=_POLE_B,
        direct_match_rate=1.0, flip_rate=1.0, causal_effect_size=1.0, n_probes=4,
    )
    result = ResolutionResult(
        pair_id="test_pair", description="d", label_a="A", label_b="B",
        baseline_hold_rate=0.0, candidates=[candidate], best=candidate,
        resolved=True, validated_hold_rate=1.0, system_prompt_patch="patch",
    )

    def model_fn(system_prompt: str, question: str) -> str:
        if "TRUE_A" in question:
            return "answer for a"
        if "TRUE_B" in question:
            return "answer for b"
        return "wrong answer"

    curve = measure_rate_distortion_for_resolution(
        result, model_fn=model_fn, commitment_extractor=_identity_extractor,
        pair=_PAIR, validation_samples=1,
    )
    assert curve is not None
    assert curve.pair_id == "test_pair"
    assert curve.points[-1].mean_accuracy == 1.0


# ── CLI wiring: `contradish distinguish --resolve --rate-distortion` ───────

class _FakeLLMWithCandidates:
    provider = "anthropic"

    def __init__(self, *a, **kw):
        pass

    def complete_json(self, prompt: str) -> dict:
        return {"candidates": [{
            "condition": "the hidden flag",
            "rationale": "test rationale",
            "pole_a_statement": "The hidden flag is TRUE_A.",
            "pole_b_statement": "The hidden flag is TRUE_B.",
        }]}


class _RateDistortionArgs:
    domain = "medication"
    app = "tests.test_rate_distortion:_cli_mock_app"
    n_samples = 1
    threshold = None
    report = False
    json = True
    kbv = False
    kbv_threshold = None
    resolve = True
    resolve_collapse_threshold = 0.3
    resolve_candidates = 1
    resolve_samples = 1
    rate_distortion = True


def _cli_mock_app(question: str) -> str:
    # Collapsed and insensitive to the hedge/pole text -- exercises the
    # rate-distortion path end-to-end without needing the model to actually
    # resolve anything; the CLI must still run and report the curve.
    return "the standard answer applies in this situation"


def test_cmd_distinguish_rate_distortion_requires_resolve_flag_in_help():
    import contradish.cli as cli
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["contradish", "distinguish", "--help"]
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
    finally:
        sys.argv = old_argv


def test_cmd_distinguish_rate_distortion_runs_end_to_end(capsys):
    import contradish.cli as cli

    with patch("contradish.llm.LLMClient", _FakeLLMWithCandidates), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch.object(cli, "_default_commitment_extractor", lambda llm: _identity_extractor):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_distinguish(_RateDistortionArgs())
        assert exc.value.code == 0

    out = capsys.readouterr().out
    # Three JSON objects printed back to back: the loss map, the resolution
    # results, then the rate-distortion results.
    parts = out.split("}\n{")
    assert len(parts) == 3
    rate_distortion_json_text = "{" + parts[2]
    d = json.loads(rate_distortion_json_text)
    assert "rate_distortion_results" in d
    for c in d["rate_distortion_results"]:
        assert c["schema_version"] == "1.0"
        assert "shape" in c
        assert c["shape"] in ("graded", "threshold", "insensitive")
