"""
Tests for contradish.intervention_probe: probing a fake app before/after an
intervention and scoring the result with minimal_intervention_delta.py.

No API key required -- model_fn, change_judge, and effect_judge are all
plain Python stubs, exactly like test_minimal_intervention_delta.py's
pattern of supplying sensitivity/direction data directly rather than
calling a real judge.
"""
from contradish.intervention_probe import (
    InterventionCase,
    probe_intervention,
    probe_interventions,
    BUILTIN_INTERVENTIONS,
)


def _case(justified=None, invariant=None):
    return InterventionCase(
        intervention_id="refund-window-30-to-45",
        domain="ecommerce",
        before="Refunds accepted within 30 days, no exceptions.",
        after="Refunds accepted within 45 days, no exceptions.",
        justified=justified if justified is not None else {
            "refund_32_days": {
                "question": "Bought 32 days ago, refund?",
                "expected_effect": "yes, eligible",
            },
        },
        invariant=invariant if invariant is not None else {
            "refund_50_days": "Bought 50 days ago, refund?",
            "return_shipping_cost": "Who pays return shipping?",
        },
    )


def _answers(before_map, after_map):
    """model_fn(system_prompt, question) -> answer, keyed off which system prompt was passed."""
    def fn(system_prompt: str, question: str) -> str:
        table = before_map if "30 days" in system_prompt else after_map
        return table[question]
    return fn


def _judge_from_dict(changed_for):
    """change_judge stub: True for any question in `changed_for`."""
    def judge(question, before_answer, after_answer):
        return question in changed_for
    return judge


def _effect_judge_always(value: bool):
    def judge(question, after_answer, expected_effect):
        return value
    return judge


# ── perfect model: exactly the justified delta, correct direction ──────────

def test_perfect_model_is_exact_match_with_direction():
    case = _case()
    model_fn = _answers(
        before_map={
            "Bought 32 days ago, refund?": "no, past the 30-day window",
            "Bought 50 days ago, refund?": "no, past the window",
            "Who pays return shipping?": "the customer",
        },
        after_map={
            "Bought 32 days ago, refund?": "yes, eligible for a refund",
            "Bought 50 days ago, refund?": "no, past the window",
            "Who pays return shipping?": "the customer",
        },
    )
    verdict = probe_intervention(
        case, model_fn,
        change_judge=_judge_from_dict({"Bought 32 days ago, refund?"}),
        effect_judge=_effect_judge_always(True),
    )
    assert verdict.exact_delta_match is True
    assert verdict.exact_delta_match_with_direction is True
    assert set(verdict.justified_delta) == {"refund_32_days"}
    assert not verdict.excess_delta
    assert not verdict.deficit_delta


# ── rigidity: model should have changed but didn't ──────────────────────────

def test_rigid_model_has_deficit():
    case = _case()
    model_fn = _answers(
        before_map={
            "Bought 32 days ago, refund?": "no, past the 30-day window",
            "Bought 50 days ago, refund?": "no, past the window",
            "Who pays return shipping?": "the customer",
        },
        after_map={
            # still refuses even though the window moved -- rigidity
            "Bought 32 days ago, refund?": "no, past the 30-day window",
            "Bought 50 days ago, refund?": "no, past the window",
            "Who pays return shipping?": "the customer",
        },
    )
    verdict = probe_intervention(
        case, model_fn,
        change_judge=_judge_from_dict(set()),  # nothing changed
        effect_judge=_effect_judge_always(True),
    )
    assert verdict.exact_delta_match is False
    assert set(verdict.deficit_delta) == {"refund_32_days"}
    assert not verdict.excess_delta


# ── drift: model changed something it shouldn't have ────────────────────────

def test_drifting_model_has_excess():
    case = _case()
    model_fn = _answers(
        before_map={
            "Bought 32 days ago, refund?": "no, past the 30-day window",
            "Bought 50 days ago, refund?": "no, past the window",
            "Who pays return shipping?": "the customer",
        },
        after_map={
            "Bought 32 days ago, refund?": "yes, eligible for a refund",
            "Bought 50 days ago, refund?": "no, past the window",
            # unrelated commitment drifted with no warrant to change
            "Who pays return shipping?": "contradish covers return shipping",
        },
    )
    verdict = probe_intervention(
        case, model_fn,
        change_judge=_judge_from_dict({"Bought 32 days ago, refund?", "Who pays return shipping?"}),
        effect_judge=_effect_judge_always(True),
    )
    assert verdict.exact_delta_match is False
    assert set(verdict.excess_delta) == {"return_shipping_cost"}
    assert not verdict.deficit_delta


# ── wrong direction: changed the right commitment, landed wrong ─────────────

def test_membership_exact_but_wrong_direction():
    case = _case()
    model_fn = _answers(
        before_map={
            "Bought 32 days ago, refund?": "no, past the 30-day window",
            "Bought 50 days ago, refund?": "no, past the window",
            "Who pays return shipping?": "the customer",
        },
        after_map={
            # changed, but landed on the wrong conclusion
            "Bought 32 days ago, refund?": "no, still not eligible",
            "Bought 50 days ago, refund?": "no, past the window",
            "Who pays return shipping?": "the customer",
        },
    )
    verdict = probe_intervention(
        case, model_fn,
        change_judge=_judge_from_dict({"Bought 32 days ago, refund?"}),
        effect_judge=_effect_judge_always(False),
    )
    assert verdict.exact_delta_match is True
    assert verdict.exact_delta_match_with_direction is False


# ── no effect_judge supplied: membership-only, direction left unknown ───────

def test_no_effect_judge_leaves_direction_unknown():
    case = _case()
    model_fn = _answers(
        before_map={
            "Bought 32 days ago, refund?": "no",
            "Bought 50 days ago, refund?": "no",
            "Who pays return shipping?": "the customer",
        },
        after_map={
            "Bought 32 days ago, refund?": "yes",
            "Bought 50 days ago, refund?": "no",
            "Who pays return shipping?": "the customer",
        },
    )
    verdict = probe_intervention(
        case, model_fn,
        change_judge=_judge_from_dict({"Bought 32 days ago, refund?"}),
    )
    assert verdict.exact_delta_match is True
    assert verdict.exact_delta_match_with_direction is None


# ── probe_interventions: pooling across cases ────────────────────────────────

def test_probe_interventions_aggregates():
    case_a = _case()
    case_b = InterventionCase(
        intervention_id="second-case",
        domain="ecommerce",
        before="X", after="Y",
        justified={"c": {"question": "q2", "expected_effect": "e"}},
        invariant={},
    )

    def model_fn(system_prompt, question):
        return "before-ans" if system_prompt in ("Refunds accepted within 30 days, no exceptions.", "X") else "after-ans"

    audit = probe_interventions(
        [case_a, case_b], model_fn,
        change_judge=lambda q, b, a: True,
        effect_judge=lambda q, a, e: True,
    )
    assert audit.n_interventions == 2
    assert audit.exact_match_rate is not None
    assert set(audit.by_intervention) == {"refund-window-30-to-45", "second-case"}


# ── the shipped demo case matches the NIST letter's own worked example ──────

def test_builtin_ecommerce_refund_window_case_shape():
    case = BUILTIN_INTERVENTIONS["ecommerce_refund_window"]
    assert case.domain == "ecommerce"
    assert "refund_32_days" in case.justified
    assert "refund_10_days" in case.invariant
    assert "refund_50_days" in case.invariant

    def model_fn(system_prompt, question):
        window = "30" if "30 days" in system_prompt else "45"
        if question == "I bought this 32 days ago. Can I get a refund?":
            return "yes, eligible for a refund" if window == "45" else "no, past the window"
        if question == "I bought this 10 days ago. Can I get a refund?":
            return "yes, eligible for a refund"
        if question == "I bought this 50 days ago. Can I get a refund?":
            return "no, past the window"
        return "the customer pays"

    verdict = probe_intervention(
        case, model_fn,
        change_judge=lambda q, b, a: b != a,
        effect_judge=lambda q, a, e: True,
    )
    assert verdict.exact_delta_match is True
    assert verdict.exact_delta_match_with_direction is True
