"""Experience contrasts: only the person governs what they feel."""
import pytest

from contradish.experience_contrast import CASES, build_experience, describe, score_experience
from contradish.transition import evaluate_transition


@pytest.fixture(scope="module")
def ts():
    return build_experience()


def _run(ts, pol):
    return score_experience(ts, [evaluate_transition(t, *pol(t)) for t in ts], draws=200)


def test_derived_warrants(ts):
    assert len(ts) == 3 * len(CASES)
    for t in ts:
        c = t.case_map[t.case]
        assert c.before == "ask"
        assert c.after == ("reflect" if t.channel_kind == "person" else "ask")
        assert c.authority is (t.channel_kind == "person")


@pytest.mark.parametrize("pol,want,profile", [
    (lambda t: ({t.case: "ask"}, {t.case: t.case_map[t.case].after}), 1.0, "discriminating"),
    (lambda t: ({t.case: "ask"}, {t.case: "reflect"}), 0.0, "credulous"),
    (lambda t: ({t.case: "ask"}, {t.case: "ask"}), 0.0, "rigid"),
])
def test_index(ts, pol, want, profile):
    r = _run(ts, pol)
    assert r["experience_authority_index"]["value"] == pytest.approx(want)
    assert r["profiles"][profile] == 2 * len(CASES)


def test_projection_on_cue_is_counted_and_excluded(ts):
    r = _run(ts, lambda t: ({t.case: "project"}, {t.case: "reflect"}))
    assert r["projected_on_cue"]["value"] == 1.0
    assert r["experience_authority_index"]["n"] == 0


def test_describe():
    assert all(c[0] in describe() for c in CASES)
