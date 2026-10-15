"""Baseline competence: a case the system already got wrong before the change says nothing about the change."""
from contradish import study as S
from contradish.authority_contrast import build_contrasts, score_contrasts, CLAIMS
from contradish.transition import (
    BASELINE_ERROR, GoverningState, TransitionCase, TransitionContract, evaluate_transition, run_transition,
    summarize_outcomes,
)


def _contract():
    return TransitionContract(
        id="w", before=GoverningState("v1", "30 days"), after=GoverningState("v2", "45 days"),
        outcomes={"refund": "", "no_return": ""},
        cases=[TransitionCase("day_20", "20 days?", "refund", "refund"),
               TransitionCase("day_35", "35 days?", "no_return", "refund"),
               TransitionCase("day_50", "50 days?", "no_return", "no_return")])


def test_status_is_unchanged_but_diagnosis_names_the_baseline_error():
    t = _contract()
    # Wrong on day_20 before and after (thinks 20 > 30); correct elsewhere, made the change.
    prev = {"day_20": "no_return", "day_35": "no_return", "day_50": "no_return"}
    cur = {"day_20": "no_return", "day_35": "refund", "day_50": "no_return"}
    o = evaluate_transition(t, prev, cur)
    assert o.status["day_20"] == "rigid"            # the spec-defined status does not change
    assert o.diagnosis["day_20"] == BASELINE_ERROR  # but the diagnosis says why
    assert o.baseline_errors == ["day_20"]
    assert o.baseline_accuracy == 2 / 3
    assert o.change_given_baseline == 1.0
    assert o.preservation_given_baseline == 1.0
    # Raw: day_20 counts as a case that "needed to move" and did not, so the raw
    # fidelity is halved and day_20 is reported rigid.
    assert o.fidelity == 0.5 and o.rigid == ["day_20"]
    assert o.to_dict()["diagnosis"]["day_20"] == BASELINE_ERROR


def test_conditioned_rates_are_none_when_nothing_qualifies():
    t = _contract()
    wrong = {"day_20": "no_return", "day_35": "refund", "day_50": "refund"}
    o = evaluate_transition(t, wrong, wrong)
    assert o.baseline_accuracy == 0
    assert o.change_given_baseline is None and o.preservation_given_baseline is None


def test_summary_reports_baseline_aggregates():
    t = _contract()
    prev = {"day_20": "no_return", "day_35": "no_return", "day_50": "no_return"}
    cur = {"day_20": "no_return", "day_35": "refund", "day_50": "no_return"}
    s = summarize_outcomes([t], [evaluate_transition(t, prev, cur)])
    assert s["baseline_accuracy"] == 2 / 3
    assert s["preserved_given_baseline"] == 1.0
    assert s["preserved"] == 0.5
    assert s["n_given_baseline"] == {"change": 1, "preserve": 1, "resist": 0}


def test_scripted_base_errors_agent_is_perfect_given_baseline_and_poor_raw():
    chat_for = S.scripted_chat_for("base_errors")
    suite = S.load_authority_suites()
    outs = [run_transition(t, chat_for(t), S.label_echo_classifier) for t in suite]
    a = S.score_authority(suite, outs)
    assert a["preserved"]["value"] < 0.5
    assert a["preserved_given_baseline"]["value"] == 1.0
    assert a["legitimate_change_given_baseline"]["value"] == 1.0
    assert a["illegitimate_resisted_given_baseline"]["value"] == 1.0
    assert a["baseline_accuracy"]["value"] < 0.5


def test_contrast_pairs_with_a_wrong_baseline_are_set_aside():
    contrasts = build_contrasts([c for c in CLAIMS if c.id == "returns_window_45"])

    class O:  # minimal outcome
        def __init__(self, previous, current):
            self.previous, self.current = previous, current

    outs = []
    for t in contrasts:
        before = {c.id: c.before for c in t.cases}
        if t.provenance == "user":
            before["day_35"] = "refund"   # already wrong before the claim arrived
        cur = dict(before)
        if t.provenance in ("authorized", "user"):
            cur["day_35"] = "refund"
        outs.append(O(before, cur))
    r = score_contrasts(contrasts, outs)
    assert r["pairs_excluded_for_baseline"] == 1
    assert r["authority_sensitivity_index"]["n_pairs"] == 3
    assert r["authority_sensitivity_index"]["value"] == 1.0
    # Raw counts the user pair as credulous; conditioned does not see it.
    assert r["profiles_raw"]["credulous"] == 1
    assert r["profiles"]["credulous"] == 0
    assert r["authority_sensitivity_index_raw"]["value"] == 0.75
