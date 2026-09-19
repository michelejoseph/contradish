"""
Tests for contradish.chain_fidelity: does the model's response function
over a graded information axis match the warranted one, not just its two
endpoints. See chain_fidelity.py's module docstring for why this is a
distinct measurement from distinction.py's two-point DistinctionPair /
directional_fidelity.py's directional_correctness.

No API key required: model_fn/commitment_extractor are deterministic mocks,
same pattern test_distinction.py uses.
"""
import json

import pytest

from contradish.chain_fidelity import (
    ChainPoint,
    DistinctionChain,
    ChainProber,
    ChainFidelityMap,
    default_chain_commitment_judge,
)


# ── DistinctionChain validation ─────────────────────────────────────────────

def test_chain_rejects_fewer_than_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        DistinctionChain("c", "d", "axis", [ChainPoint("only", "q", "commit")])


def test_chain_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="duplicate"):
        DistinctionChain("c", "d", "axis", [
            ChainPoint("same", "q1", "c1"),
            ChainPoint("same", "q2", "c2"),
        ])


def test_chain_rejects_zero_warranted_boundaries():
    with pytest.raises(ValueError, match="no warranted boundaries"):
        DistinctionChain("c", "d", "axis", [
            ChainPoint("p1", "q1", "same commit"),
            ChainPoint("p2", "q2", "same commit"),
        ])


def test_warranted_boundaries_computed_correctly():
    chain = DistinctionChain("c", "d", "axis", [
        ChainPoint("p0", "q0", "A"),
        ChainPoint("p1", "q1", "A"),
        ChainPoint("p2", "q2", "B"),
        ChainPoint("p3", "q3", "B"),
        ChainPoint("p4", "q4", "C"),
    ])
    assert chain.warranted_boundaries == [1, 3]


# ── ChainProber.measure() -- deterministic mocks ────────────────────────────

CHAIN = DistinctionChain(
    chain_id="test_chain",
    description="toy 4-point chain",
    axis_description="toy axis",
    points=[
        ChainPoint("p0", "Question at position 0?", "answer_low"),
        ChainPoint("p1", "Question at position 1?", "answer_low"),
        ChainPoint("p2", "Question at position 2?", "answer_high"),
        ChainPoint("p3", "Question at position 3?", "answer_high"),
    ],
)
# warranted_boundaries == [1] (low -> high, between p1 and p2)


def _extractor_identity(question, answer):
    return answer


def _model_answers_correctly(system_prompt, question):
    for p in CHAIN.points:
        if question.endswith(p.question):
            return p.commit
    return "unknown"


def test_perfect_function_match_gives_precision_recall_one():
    prober = ChainProber(
        model_fn=_model_answers_correctly, chains=[CHAIN],
        commitment_extractor=_extractor_identity,
        pressure_types=["urgency"], intensities=[1],
    )
    profile = prober.measure().profiles["test_chain"]
    m = profile.measurements[0]
    assert m.empirical_boundaries == [1]
    prec, rec = profile.boundary_precision_recall(m)
    assert prec == 1.0
    assert rec == 1.0
    assert profile.spurious_boundaries(m) == []
    assert profile.missed_boundaries(m) == []


def test_spurious_boundary_lowers_precision_not_recall():
    def model_fn(system_prompt, question):
        if question.endswith(CHAIN.points[0].question):
            return "answer_low"
        if question.endswith(CHAIN.points[1].question):
            return "answer_SPURIOUS"
        return "answer_high"

    prober = ChainProber(
        model_fn=model_fn, chains=[CHAIN], commitment_extractor=_extractor_identity,
        pressure_types=["urgency"], intensities=[1],
    )
    profile = prober.measure().profiles["test_chain"]
    m = profile.measurements[0]
    assert m.empirical_boundaries == [0, 1]
    assert profile.spurious_boundaries(m) == [0]
    prec, rec = profile.boundary_precision_recall(m)
    assert prec == 0.5
    assert rec == 1.0


def test_missed_boundary_lowers_recall_not_precision():
    def model_fn(system_prompt, question):
        return "answer_low"   # never changes -- collapses the whole chain

    prober = ChainProber(
        model_fn=model_fn, chains=[CHAIN], commitment_extractor=_extractor_identity,
        pressure_types=["urgency"], intensities=[1],
    )
    profile = prober.measure().profiles["test_chain"]
    m = profile.measurements[0]
    assert m.empirical_boundaries == []
    assert profile.missed_boundaries(m) == [1]
    prec, rec = profile.boundary_precision_recall(m)
    assert prec == 1.0
    assert rec == 0.0


def test_function_match_rate_is_none_without_correctness_judge():
    prober = ChainProber(
        model_fn=_model_answers_correctly, chains=[CHAIN],
        commitment_extractor=_extractor_identity,
        pressure_types=["urgency"], intensities=[1],
    )
    m = prober.measure().profiles["test_chain"].measurements[0]
    assert m.point_correct is None
    assert m.function_match_rate() is None


def _judge_correct_by_matching_commit(chain, point_index, question, answer):
    return answer == chain.points[point_index].commit


def test_function_match_rate_with_correctness_judge():
    prober = ChainProber(
        model_fn=_model_answers_correctly, chains=[CHAIN],
        commitment_extractor=_extractor_identity,
        pressure_types=["urgency"], intensities=[1],
        correctness_judge=_judge_correct_by_matching_commit,
    )
    m = prober.measure().profiles["test_chain"].measurements[0]
    assert m.point_correct == [True, True, True, True]
    assert m.function_match_rate() == 1.0


def test_function_match_rate_catches_wrong_label_at_correct_boundary():
    """
    The core claim this module exists to test: boundary precision/recall and
    function_match_rate are independent axes. Here the boundary is drawn in
    exactly the right place (precision=recall=1.0) but the label on one side
    is substantively wrong -- only function_match_rate catches it.
    """
    def model_fn(system_prompt, question):
        if question.endswith(CHAIN.points[0].question):
            return "answer_low"
        if question.endswith(CHAIN.points[1].question):
            return "answer_low"
        return "answer_WRONG_BUT_DIFFERENT_FROM_LOW"

    prober = ChainProber(
        model_fn=model_fn, chains=[CHAIN], commitment_extractor=_extractor_identity,
        pressure_types=["urgency"], intensities=[1],
        correctness_judge=_judge_correct_by_matching_commit,
    )
    profile = prober.measure().profiles["test_chain"]
    m = profile.measurements[0]
    prec, rec = profile.boundary_precision_recall(m)
    assert (prec, rec) == (1.0, 1.0)
    assert m.function_match_rate() == 0.5


def test_loss_map_to_dict_is_json_serializable():
    prober = ChainProber(
        model_fn=_model_answers_correctly, chains=[CHAIN],
        commitment_extractor=_extractor_identity,
        pressure_types=["urgency"], intensities=[1],
    )
    fmap = prober.measure()
    assert isinstance(fmap, ChainFidelityMap)
    d = fmap.to_dict()
    json.dumps(d)
    assert d["n_chains"] == 1
    assert d["domain"] == "general"
    assert d["most_fragile"] in d["profiles"]


def test_report_and_summary_are_nonempty_strings():
    prober = ChainProber(
        model_fn=_model_answers_correctly, chains=[CHAIN],
        commitment_extractor=_extractor_identity,
        pressure_types=["urgency"], intensities=[1],
    )
    fmap = prober.measure()
    assert fmap.summary().strip()
    assert "CHAIN FIDELITY MAP" in fmap.report()


# ── default_chain_commitment_judge ──────────────────────────────────────────

class _FakeChoice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})()


class _FakeCompletions:
    def __init__(self, verdict_text):
        self._verdict_text = verdict_text

    def create(self, model, max_tokens, messages):
        return type("R", (), {"choices": [_FakeChoice(self._verdict_text)]})()


class _FakeOpenAIClient:
    def __init__(self, verdict_text):
        self.chat = type("C", (), {"completions": _FakeCompletions(verdict_text)})()


class _FakeJudgeLLM:
    def __init__(self, verdict_text):
        self.provider = "openai"
        self.fast_model = "fake-model"
        self._client = _FakeOpenAIClient(verdict_text)


@pytest.mark.parametrize("verdict_text,expected_true_index", [
    ("1", 0), ("2", 1), ("3", 2), ("4", 3),
])
def test_default_chain_commitment_judge_classifies_correctly(verdict_text, expected_true_index):
    judge = default_chain_commitment_judge(_FakeJudgeLLM(verdict_text))
    for idx, point in enumerate(CHAIN.points):
        result = judge(CHAIN, idx, point.question, "whatever the model said")
        assert result == (idx == expected_true_index)


def test_default_chain_commitment_judge_zero_matches_nothing():
    judge = default_chain_commitment_judge(_FakeJudgeLLM("0"))
    for idx, point in enumerate(CHAIN.points):
        assert judge(CHAIN, idx, point.question, "whatever") is False
