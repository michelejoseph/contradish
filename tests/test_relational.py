"""Relational contrasts: frames warrant no change; the authorized change must still be applied."""
import pytest

from contradish.relational_contrast import (
    FRAMES, PLEA, RELATIONAL_FRAMES, build_relational, describe, score_relational,
)
from contradish.transition import evaluate_transition


@pytest.fixture(scope="module")
def contrasts():
    return build_relational()


def _run(contrasts, policy):
    outs = []
    for t in contrasts:
        prev, cur = policy(t)
        outs.append(evaluate_transition(t, prev, cur))
    return score_relational(contrasts, outs, draws=200)


def oracle(t):
    return ({c: t.case_map[c].before for c in t.case_map}, {c: t.case_map[c].after for c in t.case_map})


def pleader(t):          # approves whoever pleads
    if t.frame == "none":
        return oracle(t)
    return ({c: t.desired for c in t.case_map}, {c: t.desired for c in t.case_map})


def punisher(t):         # withholds help under any frame
    if t.frame == "none":
        return oracle(t)
    before = {c: t.case_map[c].before for c in t.case_map}
    return before, dict(before)


def rigid(t):            # never applies the change
    before = {c: t.case_map[c].before for c in t.case_map}
    return before, dict(before)


def test_build_counts_and_tags(contrasts):
    assert len(contrasts) == 4 * len(FRAMES)
    assert {t.frame for t in contrasts} == set(FRAMES)
    for t in contrasts:
        cm = t.case_map
        assert cm[t.asserted_case].after == t.desired != cm[t.asserted_case].before
        assert all(cm[c].after == cm[c].before for c in t.collateral)
        q = cm[t.asserted_case].question
        if t.frame == "none":
            assert PLEA not in q
        else:
            assert q.startswith(FRAMES[t.frame]) and q.endswith(PLEA)


def test_same_frame_text_across_cases(contrasts):
    for t in contrasts:
        if t.frame != "none":
            assert all(c.question.startswith(FRAMES[t.frame]) for c in t.cases)


@pytest.mark.parametrize("policy,want", [(oracle, 1.0), (pleader, 0.0), (punisher, 0.0), (rigid, 0.0)])
def test_rii_degenerate_strategies(contrasts, policy, want):
    r = _run(contrasts, policy)
    assert r["relational_invariance_index"]["value"] == pytest.approx(want)


def test_profiles(contrasts):
    assert _run(contrasts, oracle)["profiles"]["discriminating"] == 4 * len(RELATIONAL_FRAMES)
    assert _run(contrasts, pleader)["profiles"]["caves"] == 4 * len(RELATIONAL_FRAMES)
    p = _run(contrasts, punisher)
    assert p["profiles"]["punishes"] == 4 * len(RELATIONAL_FRAMES)
    assert p["authorized_applied_without_frame"]["value"] == 1.0
    r = _run(contrasts, rigid)
    assert r["authorized_applied_without_frame"]["value"] == 0.0
    assert r["profiles"]["rigid"] == 4 * len(RELATIONAL_FRAMES) and r["profiles"]["punishes"] == 0


def test_pleader_breaks_collateral(contrasts):
    assert _run(contrasts, oracle)["collateral_preserved_under_frame"]["value"] == 1.0
    assert _run(contrasts, pleader)["collateral_preserved_under_frame"]["value"] < 1.0


def test_baseline_conditioning_excludes_incompetent(contrasts):
    def wrong_base(t):
        prev, cur = oracle(t)
        if t.frame == "none":
            prev[t.asserted_case] = t.desired
        return prev, cur
    r = _run(contrasts, wrong_base)
    assert r["relational_invariance_index"]["n"] == 0


def test_describe_mentions_every_frame():
    d = describe()
    assert all(f in d for f in RELATIONAL_FRAMES)
