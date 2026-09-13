"""
Tests for contradish.compliance_gap.

No API key required: default_word_limit_checker is pure/deterministic, and
probe_fn is a plain callable in these tests -- no model calls anywhere.
"""
from contradish.compliance_gap import (
    ComplianceInstance,
    default_word_limit_checker,
    measure_compliance_gap,
    measure_compliance_gap_batch,
    score_compliance,
)


# ── default_word_limit_checker ────────────────────────────────────────────────

def test_no_declared_commitment_returns_none_and_false():
    commitment, complied = default_word_limit_checker("Just an ordinary response with no promises.")
    assert commitment is None
    assert complied is False


def test_declared_commitment_that_is_honored():
    # actually_complied checks the WHOLE response (the declaring sentence
    # counts toward the limit too) -- 11 words total, declared limit is 15.
    text = "Sure, I'll answer in under 15 words. Short answer here now."
    commitment, complied = default_word_limit_checker(text)
    assert commitment is not None
    assert "15 words" in commitment
    assert len(text.split()) <= 15
    assert complied is True


def test_declared_commitment_that_is_violated():
    text = (
        "I will respond in under 5 words. "
        "But then I kept going anyway and wrote a much longer response than promised."
    )
    commitment, complied = default_word_limit_checker(text)
    assert commitment is not None
    assert complied is False


def test_recognizes_multiple_limit_phrasings():
    for phrase in ["at most 6 words", "no more than 6 words", "fewer than 6 words", "within 6 words"]:
        text = f"Okay, {phrase}. Done."
        commitment, complied = default_word_limit_checker(text)
        assert commitment is not None, phrase


# ── measure_compliance_gap / ComplianceInstance ───────────────────────────────

def test_measure_compliance_gap_wraps_probe_and_checker():
    probe = lambda: "I will keep this under 3 words. One two."
    instance = measure_compliance_gap(probe, probe_id="p1", domain="d")
    assert instance.probe_id == "p1"
    assert instance.declared is True
    assert instance.response_excerpt.startswith("I will keep")


def test_instance_is_gap_true_only_when_declared_and_not_complied():
    gap_inst = ComplianceInstance("p", "d", verbal_commitment="under 3 words", actually_complied=False)
    ok_inst = ComplianceInstance("p", "d", verbal_commitment="under 3 words", actually_complied=True)
    no_decl_inst = ComplianceInstance("p", "d", verbal_commitment=None, actually_complied=False)
    assert gap_inst.is_gap is True
    assert ok_inst.is_gap is False
    assert no_decl_inst.is_gap is False   # no declaration = nothing to violate
    assert no_decl_inst.declared is False


# ── score_compliance ──────────────────────────────────────────────────────────

def test_score_compliance_computes_vcr_acr_gap_rate():
    instances = [
        ComplianceInstance("a", "d", "x", True),    # declared, complied
        ComplianceInstance("b", "d", "x", False),   # declared, gap
        ComplianceInstance("c", "d", None, False),  # no declaration
    ]
    report = score_compliance(instances)
    assert report.vcr == round(2 / 3, 4)          # 2 of 3 declared
    assert report.acr == round(1 / 2, 4)          # 1 of 2 declared instances complied
    assert report.gap_rate == round(1 / 3, 4)     # 1 of 3 total is a gap
    assert report.gap_ids == ["b"]


def test_score_compliance_with_no_declarations_has_none_acr():
    instances = [ComplianceInstance("a", "d", None, False)]
    report = score_compliance(instances)
    assert report.vcr == 0.0
    assert report.acr is None
    assert report.gap_rate == 0.0


def test_score_compliance_empty_list_is_well_defined():
    report = score_compliance([])
    assert report.vcr is None
    assert report.acr is None
    assert report.gap_rate is None
    assert report.gap_ids == []


# ── measure_compliance_gap_batch ──────────────────────────────────────────────

def test_batch_runs_multiple_probes_and_aggregates():
    probes = {
        "honors": lambda: "under 8 words. One two three.",   # 6 words total, declared limit 8
        "violates": lambda: "under 2 words. But I kept writing anyway regardless.",
        "silent": lambda: "no promises made here at all.",
    }
    report = measure_compliance_gap_batch(probes)
    assert len(report.instances) == 3
    assert "violates" in report.gap_ids
    assert "honors" not in report.gap_ids
    assert "silent" not in report.gap_ids


def test_report_and_summary_render_as_strings_with_scope_limit_caveat():
    probes = {"p1": lambda: "under 3 words. One two."}
    report = measure_compliance_gap_batch(probes)
    assert isinstance(report.summary(), str)
    text = report.report()
    assert isinstance(text, str)
    assert "SCOPE LIMIT" in text
    assert "p1" in text
