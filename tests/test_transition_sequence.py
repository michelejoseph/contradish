"""Behavioral Update Fidelity over a history of updates."""
from contradish import reference as ref
from contradish.transition import GoverningState, Source, TransitionCase, Update
from contradish.transition_sequence import (
    SequenceContract, commuting_orders, evaluate_sequence, order_independence, run_sequence)

SRC = {"op": Source("op", "", ["*"]), "user": Source("user", "", ["g1"]), "tool": Source("tool", "", [])}
OUT = {"yes": "yes", "no": "no"}


def _seq(updates, n=2):
    cases = [TransitionCase(f"c{i}", f"q{i}", "yes", "yes", grounds=[f"g{i + 1}"]) for i in range(n)]
    return SequenceContract("s", GoverningState("v0", "Base."), cases, dict(SRC), updates, OUT)


def U(i, src, asserts, ch="system"):
    return Update(f"u{i}", src, ch, f"update {i}", asserts)


def test_trajectory_applies_authority_per_case():
    s = _seq([U(1, "user", {"c0": "no", "c1": "no"})])          # user governs g1 only
    assert s.trajectory() == [{"c0": "yes", "c1": "yes"}, {"c0": "no", "c1": "yes"}]


def test_trajectory_matches_reference_semantics():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "user", {"c0": "yes", "c1": "no"}), U(3, "tool", {"c1": "no"})])
    want = ref.warranted_trajectory({"c0": "yes", "c1": "yes"},
                                    [{"source": u.source, "asserts": u.asserts} for u in s.updates],
                                    {k: v.governs for k, v in SRC.items()}, s.grounds)
    assert s.trajectory() == want


def test_step_contracts_chain_the_warranted_state():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "op", {"c0": "yes"})])
    assert s.step_contract(1).target() == {"c0": "no", "c1": "yes"}
    assert s.step_contract(2).target() == {"c0": "yes", "c1": "yes"}


def test_compound_update_reaches_the_final_state_in_one_step():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "op", {"c1": "no"}), U(3, "op", {"c0": "yes"})])
    assert s.final_contract().target() == s.final == {"c0": "yes", "c1": "no"}


def test_faithful_trajectory_scores_perfectly_with_no_hysteresis():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "op", {"c0": "yes"})])
    o = evaluate_sequence(s, s.trajectory(), s.final)
    assert o.tracking == 1 and o.final_faithful and o.hysteresis == [] and o.path_dependent == []
    assert o.round_trip_applicable and o.round_trip_returned


def test_hysteresis_is_right_fresh_wrong_after_history():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "op", {"c0": "yes"})])
    stuck = [{"c0": "yes", "c1": "yes"}, {"c0": "no", "c1": "yes"}, {"c0": "no", "c1": "yes"}]
    o = evaluate_sequence(s, stuck, fresh_final={"c0": "yes", "c1": "yes"})
    assert o.hysteresis == ["c0"] and o.hysteresis_rate == 0.5
    assert o.round_trip_applicable and o.round_trip_returned is False


def test_wrong_both_ways_is_not_hysteresis():
    s = _seq([U(1, "op", {"c0": "no"})])
    o = evaluate_sequence(s, [{"c0": "yes", "c1": "yes"}, {"c0": "yes", "c1": "yes"}], fresh_final={"c0": "yes", "c1": "yes"})
    assert o.hysteresis == [] and o.path_dependent == [] and not o.final_faithful


def test_round_trip_not_applicable_when_history_does_not_return_home():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "op", {"c1": "no"})])
    assert not evaluate_sequence(s, s.trajectory()).round_trip_applicable


def test_captured_mid_history_is_visible_in_the_stage_not_only_the_end():
    s = _seq([U(1, "tool", {"c0": "no"}), U(2, "op", {"c0": "no"})])
    obs = [{"c0": "yes", "c1": "yes"}, {"c0": "no", "c1": "yes"}, {"c0": "no", "c1": "yes"}]
    o = evaluate_sequence(s, obs)
    assert o.stages[0].captured == ["c0"] and o.stages[1].faithful and o.final_faithful


def test_commuting_orders_follow_footprints():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "op", {"c1": "no"})])
    assert commuting_orders(s) == [(1, 0)]
    conflict = _seq([U(1, "op", {"c0": "no"}), U(2, "op", {"c0": "yes"})])
    assert commuting_orders(conflict) == []                      # last writer wins: order matters
    unauthorised = _seq([U(1, "tool", {"c0": "no"}), U(2, "op", {"c1": "no"})])
    assert (1, 0) in commuting_orders(unauthorised)


def test_order_independence_report():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "op", {"c1": "no"})])
    good = {(1, 0): s.final}
    bad = {(1, 0): {"c0": "no", "c1": "yes"}}
    assert order_independence(s, good)["invariant"] and not order_independence(s, bad)["invariant"]


def test_observed_length_is_validated():
    s = _seq([U(1, "op", {"c0": "no"})])
    try:
        evaluate_sequence(s, [{"c0": "yes", "c1": "yes"}])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_dict_roundtrip():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "user", {"c0": "yes"}, ch="tool")])
    s2 = SequenceContract.from_dict(s.to_dict())
    assert s2.trajectory() == s.trajectory() and [u.channel for u in s2.updates] == ["system", "tool"]


def test_run_sequence_with_a_scripted_system():
    s = _seq([U(1, "op", {"c0": "no"}), U(2, "tool", {"c1": "no"}, ch="tool")])

    def chat(msgs):
        said = " ".join(m["content"] for m in msgs)
        q = msgs[-1]["content"]
        if q == "q0":
            return "no" if "update 1" in said else "yes"
        return "no" if "update 2" in said else "yes"       # obeys the tool: captured

    def classify(question, answer, outcomes):
        return answer

    o = run_sequence(s, chat, classify)
    assert o.observed[0] == {"c0": "yes", "c1": "yes"} and o.observed[2] == {"c0": "no", "c1": "no"}
    assert o.stages[1].captured == ["c1"] and not o.final_faithful
