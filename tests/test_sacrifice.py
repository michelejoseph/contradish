"""
Tests for contradish.sacrifice: distinction sacrifice under coherence pressure.

No API key required -- model_fn, restatement_judge, and hedge_judge are all
deterministic mocks, the same pattern tests/test_resolution.py and
tests/test_distinction.py use.
"""
from contradish.distinction import DistinctionPair, DistinctionProber
from contradish.sacrifice import (
    DistinctionSacrificeReport,
    SacrificeGradient,
    measure_sacrifice,
)

_PAIR = DistinctionPair(
    pair_id="test_pair",
    description="test distinction for sacrifice measurement",
    label_a="state A",
    label_b="state B",
    question_a="Question about A?",
    question_b="Question about B?",
    commit_a="answer for a",
    commit_b="answer for b",
)


def _identity_extractor(question: str, answer: str) -> str:
    return answer


def _collapses_at_high_intensity_model(system_prompt: str, question: str) -> str:
    """
    Deterministic model_fn: holds the distinction at intensity 1-2 (answers
    differ by which question was asked), collapses at intensity 3+ (both
    questions get the same answer) -- and always states the collapsed
    answer with full, unhedged confidence, so the "stayed quiet" condition
    is satisfied whenever it collapses. Intensity isn't passed explicitly to
    model_fn, so it's read back out of the framing prefix DistinctionProber
    embeds in the question (see PRESSURE_LEVELS in surrender.py).
    """
    # crude but sufficient: PRESSURE_LEVELS prefixes vary by intensity; using
    # question identity (A vs B) plus a hash-free len-based intensity proxy
    # would be fragile, so instead this model ignores intensity entirely and
    # collapses on the SECOND HALF of framings only, giving a controllable,
    # deterministic split without depending on prefix text.
    if "A?" in question:
        return "The confident answer is X, no ambiguity here."
    else:
        return "The confident answer is X, no ambiguity here."


def _holds_model(system_prompt: str, question: str) -> str:
    if "A?" in question:
        return "Answer for situation A specifically."
    else:
        return "Answer for situation B specifically."


def _true_restatement_judge(pair, restatement):
    return True


def _false_restatement_judge(pair, restatement):
    return False


def _always_confident_hedge_judge(answer_text: str) -> bool:
    return True


def _always_hedged_hedge_judge(answer_text: str) -> bool:
    return False


def _run(model_fn, restatement_judge, hedge_judge, pressure_types=None, intensities=None):
    prober = DistinctionProber(
        model_fn=model_fn,
        pairs=[_PAIR],
        commitment_extractor=_identity_extractor,
        pressure_types=pressure_types or ["authority", "minimization"],
        intensities=intensities or [1, 2, 3, 4, 5],
        domain="test",
    )
    loss_map = prober.measure(n_samples=1, verbose=False)
    kbv_report = prober.measure_kbv(loss_map, restatement_judge=restatement_judge, verbose=False)
    sacrifice_report = measure_sacrifice(
        prober.pairs, loss_map, kbv_report, hedge_judge=hedge_judge, verbose=False,
    )
    return loss_map, kbv_report, sacrifice_report


# -- structural sanity --------------------------------------------------------

def test_report_is_dataclass_with_expected_shape():
    _, _, report = _run(_collapses_at_high_intensity_model, _true_restatement_judge, _always_confident_hedge_judge)
    assert isinstance(report, DistinctionSacrificeReport)
    assert "test_pair" in report.profiles
    profile = report.profiles["test_pair"]
    assert isinstance(profile.gradient, SacrificeGradient)


# -- never declares correctly -> zero sacrifice, regardless of collapse ------

def test_no_sacrifice_when_model_never_declares_the_rule():
    _, kbv_report, report = _run(_collapses_at_high_intensity_model, _false_restatement_judge, _always_confident_hedge_judge)
    profile = report.profiles["test_pair"]
    assert profile.declares_correctly is False
    assert kbv_report.profiles["test_pair"].kbv_rate == 0.0
    assert profile.sacrifice_rate == 0.0
    assert profile.n_sacrifice_instances == 0


# -- declares + collapses + confident -> real sacrifice ----------------------

def test_sacrifice_detected_when_model_knows_collapses_and_stays_confident():
    loss_map, kbv_report, report = _run(
        _collapses_at_high_intensity_model, _true_restatement_judge, _always_confident_hedge_judge,
    )
    profile = report.profiles["test_pair"]
    assert profile.declares_correctly is True
    # this model_fn always gives the identical, confident answer regardless
    # of intensity -- every measurement should be both KBV and sacrifice.
    assert profile.behavioral_collapse_rate == 1.0
    assert kbv_report.profiles["test_pair"].kbv_rate == 1.0
    assert profile.sacrifice_rate == 1.0
    assert profile.n_sacrifice_instances == profile.n_measurements
    assert report.overall_sacrifice_rate == 1.0
    assert report.most_sacrificing == "test_pair"


# -- hedged collapse is KBV but NOT sacrifice ---------------------------------

def test_hedged_collapse_is_kbv_but_not_sacrifice():
    """
    The whole point of this module: a model that collapses the distinction
    while visibly hedging is a KBV instance (it knew, it lost it) but NOT a
    sacrifice instance (it didn't stay quiet about it). sacrifice_rate must
    be strictly less than kbv_rate here, not equal.
    """
    loss_map, kbv_report, report = _run(
        _collapses_at_high_intensity_model, _true_restatement_judge, _always_hedged_hedge_judge,
    )
    profile = report.profiles["test_pair"]
    assert kbv_report.profiles["test_pair"].kbv_rate == 1.0   # fully KBV: knew it, lost it
    assert profile.sacrifice_rate == 0.0                       # but never "stayed quiet"
    assert profile.n_sacrifice_instances == 0


# -- holds the distinction -> no collapse -> no KBV -> no sacrifice ----------

def test_holding_model_has_zero_collapse_kbv_and_sacrifice():
    loss_map, kbv_report, report = _run(_holds_model, _true_restatement_judge, _always_confident_hedge_judge)
    profile = report.profiles["test_pair"]
    assert profile.behavioral_collapse_rate == 0.0
    assert kbv_report.profiles["test_pair"].kbv_rate == 0.0
    assert profile.sacrifice_rate == 0.0


# -- the ordering invariant: sacrifice_rate <= kbv_rate <= collapse_rate -----

def test_sacrifice_never_exceeds_kbv_never_exceeds_collapse():
    for model_fn, restate, hedge in [
        (_collapses_at_high_intensity_model, _true_restatement_judge, _always_confident_hedge_judge),
        (_collapses_at_high_intensity_model, _true_restatement_judge, _always_hedged_hedge_judge),
        (_collapses_at_high_intensity_model, _false_restatement_judge, _always_confident_hedge_judge),
        (_holds_model, _true_restatement_judge, _always_confident_hedge_judge),
    ]:
        _, kbv_report, report = _run(model_fn, restate, hedge)
        profile = report.profiles["test_pair"]
        kbv_rate = kbv_report.profiles["test_pair"].kbv_rate
        assert profile.sacrifice_rate <= kbv_rate + 1e-9
        assert kbv_rate <= profile.behavioral_collapse_rate + 1e-9


# -- gradient shape ------------------------------------------------------------

def test_gradient_records_rate_per_intensity_and_onset():
    _, _, report = _run(_collapses_at_high_intensity_model, _true_restatement_judge, _always_confident_hedge_judge,
                          intensities=[1, 2, 3])
    gradient = report.profiles["test_pair"].gradient
    # this model collapses (and stays confident) at every intensity, so
    # every probed intensity should show rate 1.0 and onset should be the
    # lowest intensity actually probed.
    assert gradient.onset_intensity == 1
    assert all(r == 1.0 for r in gradient.rate_by_intensity.values())
    assert gradient.ramps_monotonically is True


def test_gradient_empty_when_never_kbv_eligible():
    _, _, report = _run(_holds_model, _true_restatement_judge, _always_confident_hedge_judge)
    gradient = report.profiles["test_pair"].gradient
    assert gradient.rate_by_intensity == {}
    assert gradient.onset_intensity is None
