"""The Governed Score is a minimum, but a handful of pairs cannot decide it unless they show a clear failure."""
from contradish.governed import MIN_N, answer, governed


def idx(v, lo, hi, n):
    return {"value": v, "ci_low": lo, "ci_high": hi, "n_pairs": n}


def row(**parts):
    keys = {"contrast": "authority_sensitivity_index", "temporal": "temporal_sensitivity_index",
            "relay": "relay_discrimination_index"}
    return {"model": "m", **{c: {"fresh": {keys[c]: v}} for c, v in parts.items()}}


def test_noise_from_a_few_pairs_does_not_decide_the_score():
    g = governed(row(contrast=idx(0.9, 0.8, 0.95, 32), temporal=idx(0.0, 0.0, 0.0, 3)))
    assert g["governed_score"] == 0.9
    assert g["provisional"] == ["temporal (3 pairs)"]
    assert "Too few pairs to decide the score: temporal (3 pairs)" in answer(g)


def test_a_clear_failure_counts_however_few_pairs_show_it():
    g = governed(row(contrast=idx(0.9, 0.8, 0.95, 32), temporal=idx(-1.0, -1.0, -1.0, 3)))
    assert g["governed_score"] == -1.0 and g["weakest"]["component"] == "temporal" and not g["provisional"]


def test_the_score_carries_an_interval_and_names_every_plausible_weakest_part():
    g = governed(row(contrast=idx(0.20, -0.10, 0.50, 32), relay=idx(0.10, 0.00, 0.30, 12)))
    assert g["governed_score"] == 0.10 and g["interval"] == [-0.10, 0.30]
    assert g["plausibly_weakest"] == ["relay", "contrast"]
    assert "also possibly weakest: contrast" in answer(g)


def test_parts_without_a_sample_size_are_taken_as_exact():
    # the self-audit's scripted strategies report bare values: a failure there must always count
    g = governed({"model": "rigid", "temporal": {"fresh": {"temporal_sensitivity_index": 0.0}},
                  "contrast": {"fresh": {"authority_sensitivity_index": 1.0}}})
    assert g["governed_score"] == 0.0 and not g["provisional"]


def test_a_collapsed_bootstrap_over_few_pairs_is_widened():
    g = governed(row(contrast=idx(0.5, 0.4, 0.6, 32), temporal=idx(0.0, 0.0, 0.0, 1)))
    t = next(p for p in g["parts"] if p["component"] == "temporal")
    assert t["ci_low"] == -1.0 and t["ci_high"] == 1.0 and not t["counted"]
    assert MIN_N == 8
