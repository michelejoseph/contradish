"""
Tests for contradish/transition.py -- the transition contract (the atomic
object) and how faithfully a system moves through one. No API calls.
"""

import json
import sys

import pytest

from contradish.contract import load_builtin_contract, run_contract
from contradish.transition import (
    COLLAPSE, EMERGE, STAY_MERGED, SURVIVE, UNCLEAR,
    GoverningState, TransitionCase, TransitionContract, evaluate_transition, run_transition,
)


def window():
    return TransitionContract(
        id="window_30_to_45", domain="ecommerce",
        before=GoverningState("v1", "Refunds within 30 days of delivery."),
        after=GoverningState("v2", "Refunds within 45 days of delivery."),
        outcomes={"refund": "refund allowed", "no_return": "no refund", "escalate": "sent to a supervisor"},
        cases=[
            TransitionCase("day_20", "Delivered 20 days ago. Refund?", "refund", "refund"),
            TransitionCase("day_35", "Delivered 35 days ago. Refund?", "no_return", "refund"),
            TransitionCase("day_50", "Delivered 50 days ago. Refund?", "no_return", "no_return"),
        ],
    )


OLD = {"day_20": "refund", "day_35": "no_return", "day_50": "no_return"}
NEW = {"day_20": "refund", "day_35": "refund", "day_50": "no_return"}


# ── what a contract warrants ────────────────────────────────────────────────

def test_persist_and_revise_follow_from_the_two_sides():
    t = window()
    assert [c.id for c in t.revise()] == ["day_35"]
    assert [c.id for c in t.persist()] == ["day_20", "day_50"]
    assert not t.is_null and t.lint() == []
    assert t.justified_and_invariant() == ({"day_35": "refund"}, ["day_20", "day_50"])


def test_distinctions_survive_collapse_and_emerge():
    fates = {(d.a, d.b): d.fate for d in window().distinctions()}
    assert fates == {
        ("day_20", "day_35"): COLLAPSE,     # were treated differently, now the same
        ("day_20", "day_50"): SURVIVE,
        ("day_35", "day_50"): EMERGE,       # were treated the same, now differently
    }


def test_unrelated_cases_are_not_counted_as_distinctions():
    t = window()
    t.cases[0].grounds, t.cases[1].grounds = ["R1"], ["R1"]
    t.cases[2].grounds = ["R9"]
    assert [(d.a, d.b) for d in t.distinctions()] == [("day_20", "day_35")]
    assert len(t.distinctions(related_only=False)) == 3


def test_null_transition():
    t = window()
    t.after = GoverningState("v1b", "A refund is available for 30 days after delivery.")
    for c in t.cases:
        c.after = c.before
    t.meaning_preserving = True
    assert t.is_null and t.lint() == []
    assert {d.fate for d in t.distinctions()} <= {SURVIVE, STAY_MERGED}
    out = evaluate_transition(t, OLD, OLD)
    assert out.fidelity is None and out.hold == 1.0 and out.persistence == 1.0 and out.revision is None
    moved = evaluate_transition(t, OLD, NEW)       # it changed day_35 though nothing warranted it
    assert moved.drift == ["day_35"] and moved.hold == pytest.approx(2 / 3) and moved.persistence < 1


def test_lint():
    t = window()
    t.cases[1].after = "maybe"
    assert any(code == "T003" for _, code, _ in t.lint_errors())
    t = window()
    t.meaning_preserving = True
    assert any(code == "T004" for _, code, _ in t.lint_errors())
    t = window()
    t.cases = [t.cases[1]]
    assert {code for _, code, _ in t.lint()} == {"T103"}
    t = window()
    for c in t.cases:
        c.after = c.before
    assert "T102" in {code for _, code, _ in t.lint()}


def test_roundtrip_load_and_schema(tmp_path):
    t = window()
    again = TransitionContract.from_dict(json.loads(json.dumps(t.to_dict())))
    assert again.to_dict() == t.to_dict()
    p = tmp_path / "t.json"
    p.write_text(json.dumps(t.to_dict()))
    assert TransitionContract.load(p).to_dict() == t.to_dict()
    short = TransitionContract.from_dict({
        "id": "x", "before": "old text", "after": "new text", "outcomes": ["a", "b"],
        "cases": [{"id": "c", "question": "q", "before": "a", "after": "b"}],
    })
    assert short.before.text == "old text" and short.revise()[0].id == "c"
    pytest.importorskip("jsonschema")
    from contradish.schema import validate_against_schema
    assert validate_against_schema(t.to_dict(), "transition") == []


# ── how faithfully did it move ──────────────────────────────────────────────

def test_faithful_move():
    out = evaluate_transition(window(), OLD, NEW)
    assert out.fidelity == 1.0 and out.exact and out.persistence == 1.0 and out.revision == 1.0
    assert out.status == {"day_20": "held", "day_35": "moved", "day_50": "held"}


def test_rigid_system_scores_zero_fidelity_and_zero_revision():
    out = evaluate_transition(window(), OLD, OLD)
    assert out.fidelity == 0.0 and out.rigid == ["day_35"]
    assert out.revision == 0.0 and out.persistence == 1.0
    assert sorted(f for _, _, f in out.broken_distinctions) == [COLLAPSE, EMERGE]


def test_overshoot_goes_negative():
    # Moved day_35 correctly, but also flipped day_50 and day_20: ends further off than it began.
    out = evaluate_transition(window(), OLD, {"day_20": "no_return", "day_35": "refund", "day_50": "refund"})
    assert out.needed_to_move == 1 and out.off_target_after == 2
    assert out.fidelity == -1.0 and out.drift == ["day_20", "day_50"]


def test_misdirection_is_not_rigidity():
    out = evaluate_transition(window(), OLD, {**OLD, "day_35": "escalate"})
    assert out.misdirected == ["day_35"] and out.rigid == [] and out.fidelity == 0.0


def test_previous_state_is_what_the_system_did_not_what_it_should_have():
    # It was wrong about day_50 before, and right after: that is faithful movement, not drift.
    prev = {**OLD, "day_50": "refund"}
    out = evaluate_transition(window(), prev, NEW)
    assert out.needed_to_move == 2 and out.fidelity == 1.0
    assert out.status["day_50"] == "moved"


def test_right_partition_wrong_labels_are_scored_separately():
    swapped = {"day_20": "no_return", "day_35": "no_return", "day_50": "refund"}
    out = evaluate_transition(window(), OLD, swapped)
    assert out.persistence == 1.0 and out.revision == 1.0      # the lines are in the right places
    assert out.fidelity < 0                                     # the outcomes are not


def test_unclear_is_never_on_target():
    out = evaluate_transition(window(), OLD, {"day_20": "refund", "day_50": "no_return"})
    assert out.current["day_35"] == UNCLEAR and out.status["day_35"] == "misdirected"
    assert out.revision == 0.0
    json.dumps(out.to_dict())
    assert "fidelity" in out.summary()


# ── probing: the deployed system, and one agent revising its own answer ─────

def label_classifier(question, answer, outcomes):
    return answer if answer in outcomes else UNCLEAR


def _days(text):
    import re
    return int(re.search(r"(\d+) days ago", text).group(1))


def policy_reader(messages):
    """Answers from the most recent governing information it was given."""
    import re
    windows = [int(m) for msg in messages if msg["role"] != "assistant"
               for m in re.findall(r"within (\d+) days", msg["content"])]
    return "refund" if _days(messages[-1]["content"]) <= windows[-1] else "no_return"


def anchored(messages):
    """Sticks to an answer it already gave in this conversation."""
    said = [m["content"] for m in messages if m["role"] == "assistant"]
    return said[-1] if said else policy_reader(messages)


def test_run_fresh_and_in_conversation_agree_for_a_faithful_agent():
    for delivery in ("fresh", "in_conversation"):
        out = run_transition(window(), policy_reader, label_classifier, delivery=delivery)
        assert out.delivery == delivery and out.fidelity == 1.0 and out.exact


def test_anchored_agent_is_only_caught_in_conversation():
    fresh = run_transition(window(), anchored, label_classifier, delivery="fresh")
    assert fresh.fidelity == 1.0                    # no prior answer to cling to
    live = run_transition(window(), anchored, label_classifier, delivery="in_conversation")
    assert live.fidelity == 0.0 and live.rigid == ["day_35"] and live.revision == 0.0


def test_in_conversation_delivers_the_new_information_after_the_first_answer():
    seen = []

    def spy(messages):
        seen.append([m["role"] for m in messages])
        return policy_reader(messages)

    run_transition(window(), spy, label_classifier, delivery="in_conversation")
    assert seen[0] == ["system", "user"] and seen[1] == ["system", "user", "assistant", "user"]


def test_run_refuses_a_broken_contract_and_bad_delivery():
    t = window()
    t.cases[0].before = "nope"
    with pytest.raises(ValueError, match="errors"):
        run_transition(t, policy_reader, label_classifier)
    with pytest.raises(ValueError, match="delivery"):
        run_transition(window(), policy_reader, label_classifier, delivery="later")


def test_samples_use_the_modal_outcome():
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        return "escalate" if calls["n"] == 1 else policy_reader(messages)

    out = run_transition(window(), flaky, label_classifier, samples=3)
    assert out.fidelity == 1.0


# ── a policy contract is a base state plus transition contracts ─────────────

def test_policy_contract_exposes_its_amendments_as_transition_contracts():
    policy = load_builtin_contract()
    ts = policy.transitions()
    assert [t.id for t in ts] == [a.id for a in policy.amendments]
    w = policy.transition_contract("window_45_days")
    assert [c.id for c in w.revise()] == ["day_35"] and w.lint() == []
    assert "within 30 days" in w.before.text and "within 45 days" in w.after.text
    assert w.case_map["day_35"].grounds == ["R1"]
    control = policy.transition_contract("reworded_exceptions_clause")
    assert control.meaning_preserving and control.is_null and control.lint() == []


def test_contract_run_reports_the_graded_transition():
    from tests.test_contract import faithful, label_classifier as lc, rigid
    policy = load_builtin_contract()
    good = run_contract(policy, faithful, lc, workers=1)
    for tr in good.transitions.values():
        assert tr.outcome.exact
        assert tr.outcome.fidelity in (1.0, None)
    bad = run_contract(policy, rigid, lc, workers=1)
    w = bad.transitions["window_45_days"]
    assert w.outcome.fidelity == 0.0 and w.outcome.revision == 0.0 and w.outcome.persistence == 1.0
    d = bad.to_dict(False)["transitions"][0]
    assert {"fidelity", "persistence", "revision", "broken_distinctions"} <= set(d)
    assert "fidelity 0.00" in bad.report()


def test_old_derivation_class_keeps_working_under_both_names():
    from contradish.transition_derivation import TransitionContract as Old, WarrantDerivation
    import contradish
    assert Old is WarrantDerivation
    assert contradish.TransitionContract is TransitionContract
    assert contradish.WarrantDerivation is WarrantDerivation


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli(monkeypatch, argv):
    from contradish import cli
    monkeypatch.setattr(sys, "argv", ["contradish"] + argv)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code


def test_cli_show_score_export(monkeypatch, capsys, tmp_path):
    assert _cli(monkeypatch, ["transition", "show"]) == 0
    out = capsys.readouterr().out
    assert "must be revised (1)" in out and "collapse" in out and "emerge" in out

    tfile = tmp_path / "t.json"
    tfile.write_text(json.dumps(window().to_dict()))
    prev, cur = tmp_path / "prev.json", tmp_path / "cur.json"
    prev.write_text(json.dumps(OLD))
    cur.write_text(json.dumps(OLD))
    assert _cli(monkeypatch, ["transition", "score", str(tfile), "--previous", str(prev),
                              "--current", str(cur), "--json"]) == 1
    scored = json.loads(capsys.readouterr().out)
    assert scored["fidelity"] == 0.0 and scored["exact"] is False
    cur.write_text(json.dumps(NEW))
    assert _cli(monkeypatch, ["transition", "score", str(tfile), "--previous", str(prev),
                              "--current", str(cur)]) == 0
    assert "fidelity     1.00" in capsys.readouterr().out

    out_file = tmp_path / "all.json"
    assert _cli(monkeypatch, ["transition", "export", "--output", str(out_file)]) == 0
    capsys.readouterr()
    exported = json.loads(out_file.read_text())
    assert len(exported) == 4 and exported[0]["id"] == "window_45_days"
    assert _cli(monkeypatch, ["transition", "lint", str(tfile)]) == 0
