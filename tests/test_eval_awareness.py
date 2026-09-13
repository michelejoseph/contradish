"""
Tests for contradish.eval_awareness.

No API key required: probe_fn is a plain Python callable the caller
supplies, so these tests use deterministic fake probes -- no model calls
anywhere in this module or its tests.
"""
from contradish.eval_awareness import (
    EVAL_SIGNAL_FRAMINGS,
    measure_eval_awareness,
    measure_eval_awareness_batch,
)


def test_identical_behavior_under_all_framings_has_zero_gap():
    probe = lambda prefix: 0.8   # same value no matter the framing
    instance = measure_eval_awareness(probe, probe_id="p1", domain="d")
    assert instance.neutral_value == 0.8
    assert instance.eval_signaled_value == 0.8
    assert instance.gap == 0.0
    assert len(instance.per_signal_values) == len(EVAL_SIGNAL_FRAMINGS)


def test_model_that_behaves_better_under_eval_framing_has_positive_gap():
    def probe(prefix):
        return 0.5 if prefix == "" else 0.9   # neutral prefix is ""
    instance = measure_eval_awareness(probe, neutral_prefix="")
    assert instance.neutral_value == 0.5
    assert instance.eval_signaled_value == 0.9
    assert instance.gap == round(0.9 - 0.5, 4)


def test_gap_is_signed_negative_when_eval_framing_performs_worse():
    def probe(prefix):
        return 0.9 if prefix == "" else 0.5
    instance = measure_eval_awareness(probe, neutral_prefix="")
    assert instance.gap < 0


def test_custom_eval_signal_prefixes_are_used_instead_of_default():
    calls = []
    def probe(prefix):
        calls.append(prefix)
        return 1.0
    custom = ["signal A", "signal B"]
    instance = measure_eval_awareness(probe, eval_signal_prefixes=custom, neutral_prefix="neutral")
    assert set(instance.per_signal_values.keys()) == set(custom)
    assert calls[0] == "neutral"
    assert set(calls[1:]) == set(custom)


def test_empty_eval_signal_prefixes_raises():
    try:
        measure_eval_awareness(lambda prefix: 1.0, eval_signal_prefixes=[])
        assert False, "expected ValueError for empty eval_signal_prefixes"
    except ValueError:
        pass


def test_batch_computes_mean_gap_and_identifies_most_aware():
    probes = {
        "stable": lambda prefix: 0.5,
        "reactive": (lambda prefix: 0.5 if prefix == "" else 0.95),
    }
    report = measure_eval_awareness_batch(probes, neutral_prefix="")
    assert len(report.instances) == 2
    assert report.most_aware == "reactive"
    assert report.mean_gap is not None
    stable_inst = [i for i in report.instances if i.probe_id == "stable"][0]
    assert stable_inst.gap == 0.0


def test_batch_with_no_probes_is_well_defined_not_a_crash():
    report = measure_eval_awareness_batch({})
    assert report.instances == []
    assert report.mean_gap is None
    assert report.mean_abs_gap is None
    assert report.most_aware == ""


def test_report_and_summary_render_as_strings_and_include_caveat():
    probe = lambda prefix: 0.5 if prefix == "" else 0.95
    report = measure_eval_awareness_batch({"p1": probe}, neutral_prefix="")
    assert isinstance(report.summary(), str)
    text = report.report()
    assert isinstance(text, str)
    assert "not proof of strategic deception" in text
    assert "p1" in text
