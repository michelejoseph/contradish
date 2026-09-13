"""
Tests for contradish.format_fidelity.

No API key required: probe_fn is a plain callable and default_format_classifier
is pure/deterministic -- these tests exercise both without any model calls.
"""
from contradish.format_fidelity import (
    default_format_classifier,
    measure_format_fidelity,
    measure_format_fidelity_batch,
)


# ── default_format_classifier ────────────────────────────────────────────────

def test_classifies_json_object():
    assert default_format_classifier('{"a": 1, "b": 2}') == "json"


def test_classifies_json_array():
    assert default_format_classifier('[1, 2, 3]') == "json"


def test_classifies_bulleted_list():
    text = "- first item\n- second item\n- third item"
    assert default_format_classifier(text) == "bulleted_list"


def test_classifies_numbered_list():
    text = "1. first item\n2. second item\n3. third item"
    assert default_format_classifier(text) == "numbered_list"


def test_classifies_table():
    text = "a | b\n1 | 2\n3 | 4"
    assert default_format_classifier(text) == "table"


def test_classifies_single_word():
    assert default_format_classifier("Yes") == "single_word"


def test_classifies_prose_as_fallback():
    text = "This is a full sentence explaining something in ordinary prose."
    assert default_format_classifier(text) == "prose"


def test_classifies_empty_string():
    assert default_format_classifier("") == "empty"
    assert default_format_classifier("   ") == "empty"


# ── measure_format_fidelity ───────────────────────────────────────────────────

def test_stable_format_across_all_paraphrases_has_consistency_rate_one():
    probe = lambda phrasing: '{"x": 1}'
    instance = measure_format_fidelity(
        probe, paraphrases=["say a", "say b", "say c"],
        instruction_id="i1", domain="d",
    )
    assert instance.modal_format == "json"
    assert instance.consistency_rate == 1.0
    assert instance.collapsed is False


def test_format_switch_lowers_consistency_rate_and_flags_collapse():
    outputs = {"p1": '{"x": 1}', "p2": "just some prose here", "p3": "more prose output"}
    probe = lambda phrasing: outputs[phrasing]
    instance = measure_format_fidelity(
        probe, paraphrases=["p1", "p2", "p3"], threshold=0.75,
    )
    # modal format is "prose" (2/3), consistency_rate = 0.6667 < 0.75
    assert instance.modal_format == "prose"
    assert instance.consistency_rate < 0.75
    assert instance.collapsed is True


def test_custom_classify_fn_is_used():
    probe = lambda phrasing: phrasing
    classify = lambda output: "always_the_same"
    instance = measure_format_fidelity(
        probe, paraphrases=["a", "b"], classify_fn=classify,
    )
    assert instance.modal_format == "always_the_same"
    assert instance.consistency_rate == 1.0


def test_requires_at_least_two_paraphrases():
    try:
        measure_format_fidelity(lambda p: "x", paraphrases=["only one"])
        assert False, "expected ValueError for < 2 paraphrases"
    except ValueError:
        pass


def test_labels_must_match_paraphrases_length():
    try:
        measure_format_fidelity(
            lambda p: "x", paraphrases=["a", "b"], labels=["only_one_label"],
        )
        assert False, "expected ValueError for mismatched labels length"
    except ValueError:
        pass


def test_custom_labels_are_used_as_dict_keys():
    probe = lambda phrasing: "x"
    instance = measure_format_fidelity(
        probe, paraphrases=["phrasing one", "phrasing two"],
        labels=["variant_a", "variant_b"],
    )
    assert set(instance.per_paraphrase_format.keys()) == {"variant_a", "variant_b"}


# ── measure_format_fidelity_batch ─────────────────────────────────────────────

def test_batch_runs_multiple_instructions_and_tracks_most_unstable():
    def probe_fn(iid, phrasing):
        if iid == "stable":
            return '{"x": 1}'
        return "prose" if phrasing == "p1" else "1. a\n2. b"

    instructions = {"stable": ["p1", "p2"], "unstable": ["p1", "p2"]}
    report = measure_format_fidelity_batch(
        instructions, classify_fn=default_format_classifier, probe_fn=probe_fn,
    )
    assert len(report.instances) == 2
    assert report.most_unstable == "unstable"
    assert report.collapse_rate == 0.5


def test_batch_with_no_instructions_is_well_defined():
    report = measure_format_fidelity_batch(
        {}, classify_fn=default_format_classifier, probe_fn=lambda iid, p: "x",
    )
    assert report.instances == []
    assert report.mean_consistency_rate is None
    assert report.collapse_rate is None
    assert report.most_unstable == ""


def test_report_and_summary_render_as_strings():
    probe = lambda phrasing: '{"x": 1}'
    instance = measure_format_fidelity(probe, paraphrases=["a", "b"], instruction_id="i1")
    from contradish.format_fidelity import FormatFidelityReport
    report = FormatFidelityReport(
        instances=[instance], mean_consistency_rate=1.0, collapse_rate=0.0, most_unstable="i1",
    )
    assert isinstance(report.summary(), str)
    text = report.report()
    assert isinstance(text, str)
    assert "i1" in text
