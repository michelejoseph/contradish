"""
Tests for contradish.justification_faithfulness.

No API key required: JustificationFaithfulnessJudge is exercised against a
fake `.complete_json`-returning object, and ManualFaithfulnessJudge against
plain dicts -- no model calls anywhere in this module or its tests, same
discipline test_eval_awareness.py already uses.
"""
from contradish.justification_faithfulness import (
    MatchedPressurePair,
    pairs_from_surrender_curve,
    JustificationFaithfulnessJudge,
    ManualFaithfulnessJudge,
    audit_justification_faithfulness,
    audit_justification_faithfulness_manual,
    pair_key,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeSample:
    def __init__(self, framing_type, intensity, full_question, answer, surrendered):
        self.framing_type = framing_type
        self.intensity = intensity
        self.full_question = full_question
        self.answer = answer
        self.surrendered = surrendered


class FakeCurve:
    def __init__(self, constraint_id, raw_samples):
        self.constraint_id = constraint_id
        self.raw_samples = raw_samples


class FakeLLM:
    """Returns whatever dict was configured, ignoring the prompt -- these
    tests are about the module's own plumbing, not prompt engineering."""
    def __init__(self, response):
        self.response = response

    def complete_json(self, prompt):
        return self.response


class RaisingLLM:
    def complete_json(self, prompt):
        raise RuntimeError("simulated API failure")


def _make_pair(surrendered=True, pair_id="p1"):
    return MatchedPressurePair(
        pair_id=pair_id, constraint_id="c", framing_type="urgency",
        baseline_intensity=1, baseline_prompt="baseline q",
        baseline_answer="baseline a grounded in X",
        pressured_intensity=5, pressured_prompt="pressured q",
        pressured_answer="pressured a claiming new fact Y",
        surrendered=surrendered,
    )


# ── pairs_from_surrender_curve ────────────────────────────────────────────────

def test_pairs_from_surrender_curve_pairs_each_higher_intensity_against_baseline():
    samples = [
        FakeSample("urgency", 1, "q@1", "a@1", surrendered=False),
        FakeSample("urgency", 3, "q@3", "a@3", surrendered=False),
        FakeSample("urgency", 5, "q@5", "a@5", surrendered=True),
    ]
    curve = FakeCurve("dose_ceiling", samples)
    pairs = pairs_from_surrender_curve(curve)

    assert len(pairs) == 2  # 1v3 and 1v5, never 3v5 (always paired against baseline)
    ids = {p.pair_id for p in pairs}
    assert "dose_ceiling:urgency:1v3" in ids
    assert "dose_ceiling:urgency:1v5" in ids
    p5 = next(p for p in pairs if p.pressured_intensity == 5)
    assert p5.surrendered is True
    assert p5.baseline_answer == "a@1"
    assert p5.pressured_answer == "a@5"


def test_pairs_from_surrender_curve_skips_framing_with_no_baseline_sample():
    samples = [
        FakeSample("authority", 3, "q@3", "a@3", surrendered=True),  # no intensity==1
        FakeSample("urgency", 1, "q@1", "a@1", surrendered=False),
        FakeSample("urgency", 5, "q@5", "a@5", surrendered=True),
    ]
    curve = FakeCurve("c", samples)
    pairs = pairs_from_surrender_curve(curve)
    assert all(p.framing_type != "authority" for p in pairs)
    assert len(pairs) == 1


def test_pairs_from_surrender_curve_surrendered_requires_baseline_held():
    # If the baseline itself was already "surrendered" (e.g. intensity=1
    # already abandons the constraint), the higher-intensity sample isn't a
    # NEW surrender relative to it -- pairs_from_surrender_curve should not
    # mark it surrendered=True in that case.
    samples = [
        FakeSample("urgency", 1, "q@1", "a@1", surrendered=True),
        FakeSample("urgency", 5, "q@5", "a@5", surrendered=True),
    ]
    curve = FakeCurve("c", samples)
    pairs = pairs_from_surrender_curve(curve)
    assert len(pairs) == 1
    assert pairs[0].surrendered is False


# ── pair_key ──────────────────────────────────────────────────────────────────

def test_pair_key_is_the_pair_id():
    p = _make_pair(pair_id="dose_ceiling:urgency:1v5")
    assert pair_key(p) == "dose_ceiling:urgency:1v5"


# ── ManualFaithfulnessJudge / audit_justification_faithfulness_manual ────────

def test_manual_judge_uses_default_when_key_missing():
    judge = ManualFaithfulnessJudge(judgments={})
    p = _make_pair(surrendered=True)
    v = judge.verdict_for(p)
    assert v["category"] == "unsupported_assertion"  # the module's own default


def test_manual_judge_uses_explicit_entry_when_present():
    p = _make_pair(surrendered=True, pair_id="p1")
    judge = ManualFaithfulnessJudge(judgments={
        "p1": dict(category="confabulated_evidence",
                   confabulated_claims=["fact Y"], confidence=0.9,
                   explanation="invented fact Y"),
    })
    v = judge.verdict_for(p)
    assert v["category"] == "confabulated_evidence"
    assert v["confabulated_claims"] == ["fact Y"]


def test_audit_manual_marks_non_surrendered_pairs_no_change_without_consulting_judge():
    p = _make_pair(surrendered=False, pair_id="p1")
    # deliberately WRONG entry for p1 -- should never be consulted, since
    # surrendered=False short-circuits before any judge lookup
    judge = ManualFaithfulnessJudge(judgments={
        "p1": dict(category="confabulated_evidence", confidence=1.0, explanation="should not be used"),
    })
    report = audit_justification_faithfulness_manual([p], judge)
    assert len(report.findings) == 1
    assert report.findings[0].category == "no_change"
    assert report.confabulation_rate() is None  # no surrendered pairs to score


def test_audit_manual_end_to_end_confabulation_rate():
    pairs = [
        _make_pair(surrendered=True, pair_id="p1"),
        _make_pair(surrendered=True, pair_id="p2"),
        _make_pair(surrendered=False, pair_id="p3"),
    ]
    judge = ManualFaithfulnessJudge(judgments={
        "p1": dict(category="confabulated_evidence", confidence=0.8, explanation="e1"),
        "p2": dict(category="honest_conformity", confidence=0.7, explanation="e2"),
    })
    report = audit_justification_faithfulness_manual(pairs, judge)
    assert report.pairs_considered == 3
    assert report.confabulation_rate() == 0.5  # 1 of 2 SCORED (surrendered) pairs
    rates = report.rate_by_category()
    assert rates["confabulated_evidence"] == 0.5
    assert rates["honest_conformity"] == 0.5


def test_report_to_json_round_trips():
    p = _make_pair(surrendered=True, pair_id="p1")
    judge = ManualFaithfulnessJudge(judgments={
        "p1": dict(category="confabulated_evidence", confabulated_claims=["x"],
                   confidence=0.8, explanation="e"),
    })
    report = audit_justification_faithfulness_manual([p], judge)
    import json
    data = json.loads(report.to_json())
    assert data["pairs_considered"] == 1
    assert data["confabulation_rate"] == 1.0
    assert data["findings"][0]["category"] == "confabulated_evidence"
    assert data["findings"][0]["severity"] == "critical"


# ── JustificationFaithfulnessJudge (live-judge-shaped, fake LLM) ─────────────

def test_judge_no_change_short_circuits_without_calling_llm():
    class ExplodingLLM:
        def complete_json(self, prompt):
            raise AssertionError("should not be called for a non-surrendered pair")
    judge = JustificationFaithfulnessJudge(ExplodingLLM())
    p = _make_pair(surrendered=False)
    finding = judge.judge_pair(p)
    assert finding.category == "no_change"


def test_judge_parses_valid_response():
    llm = FakeLLM({
        "category": "honest_conformity",
        "confabulated_claims": [],
        "confidence": 0.6,
        "explanation": "admits yielding to pressure",
    })
    judge = JustificationFaithfulnessJudge(llm)
    finding = judge.judge_pair(_make_pair(surrendered=True))
    assert finding.category == "honest_conformity"
    assert finding.severity == "low"
    assert finding.confidence == 0.6


def test_judge_falls_back_on_unknown_category():
    llm = FakeLLM({"category": "not_a_real_category", "confidence": 0.5, "explanation": ""})
    judge = JustificationFaithfulnessJudge(llm)
    finding = judge.judge_pair(_make_pair(surrendered=True))
    assert finding.category == "unsupported_assertion"


def test_judge_handles_llm_exception_gracefully():
    judge = JustificationFaithfulnessJudge(RaisingLLM())
    finding = judge.judge_pair(_make_pair(surrendered=True))
    assert finding.category == "unsupported_assertion"
    assert finding.confidence == 0.0
    assert "simulated API failure" in finding.explanation


def test_judge_clamps_out_of_range_confidence():
    llm = FakeLLM({"category": "confabulated_evidence", "confidence": 5.0, "explanation": ""})
    judge = JustificationFaithfulnessJudge(llm)
    finding = judge.judge_pair(_make_pair(surrendered=True))
    assert finding.confidence == 1.0  # clamped, not 5.0


def test_audit_justification_faithfulness_batch_uses_live_judge():
    llm = FakeLLM({"category": "confabulated_evidence", "confidence": 0.8, "explanation": "e"})
    judge = JustificationFaithfulnessJudge(llm)
    pairs = [_make_pair(surrendered=True, pair_id="p1"), _make_pair(surrendered=False, pair_id="p2")]
    report = audit_justification_faithfulness(pairs, judge)
    assert report.pairs_considered == 2
    assert report.confabulation_rate() == 1.0  # only p1 is scored; p2 is no_change
