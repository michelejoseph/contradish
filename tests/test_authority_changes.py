"""Authority decided on the grounds an update changes, not on any ground the case happens to share."""
import itertools
import random

from contradish import reference
from contradish.transition import (GoverningState, Source, TransitionCase, TransitionContract, Update,
                                   derive_transition, load_builtin_transitions)
from contradish.transition_sequence import SequenceContract

SOURCES = {"owner": Source("owner", "policy_owner", ["clause:window"]),
           "customer": Source("customer", "user", ["fact:contact_pref"]),
           "everything": Source("everything", "operator", ["*"])}
OUTCOMES = {"refund": "", "deny": ""}


def case():
    return TransitionCase("day_40", "Refund a day-40 return?", "deny", "deny", [], None,
                          ["clause:window", "fact:contact_pref"])


def derive(source, changes):
    u = Update("u", source, "user", "The return window is now 60 days.", {"day_40": "refund"}, changes=changes)
    return derive_transition("t", GoverningState("v1", "30-day window"), [case()], u, SOURCES, OUTCOMES)


def test_sharing_an_unrelated_ground_no_longer_confers_standing():
    t = derive("customer", ["clause:window"])
    assert t.cases[0].authority is False and t.cases[0].after == "deny"
    assert not [i for i in t.lint() if i[0] == "error"]


def test_the_base_rule_is_unchanged_and_now_warned_about():
    t = derive("customer", None)
    assert t.cases[0].authority is True and t.cases[0].after == "refund"       # the spec's any-overlap rule
    assert any(code == "T106" for _, code, _ in t.lint())


def test_the_source_that_governs_the_changed_ground_keeps_authority():
    t = derive("owner", ["clause:window"])
    assert t.cases[0].authority is True and t.cases[0].after == "refund"
    assert not any(code in ("T106", "T014") for _, code, _ in t.lint())


def test_a_change_to_a_ground_the_case_does_not_rest_on_warrants_nothing():
    t = derive("owner", ["clause:shipping"])
    assert t.cases[0].authority is False and t.cases[0].after == "deny"


def test_protected_grounds_stay_ungoverned_with_declared_changes():
    s = Source("op", "operator", ["*"])
    assert not s.has_authority_over(["protected:ai_disclosure"], ["protected:ai_disclosure"])
    assert not reference.authorized(["*"], ["protected:x", "a"], ["a"])


def test_reference_and_production_agree_on_every_small_combination():
    universe = ["a", "b", "c", "protected:p"]
    subsets = [list(x) for r in range(0, 4) for x in itertools.combinations(universe, r)]
    governs = subsets + [["*"]]
    for gov in governs:
        src = Source("s", "", gov)
        for grounds in subsets:
            for changes in [None] + subsets:
                assert src.has_authority_over(grounds, changes) == reference.authorized(gov, grounds, changes), \
                    (gov, grounds, changes)


def test_round_trip_keeps_declared_changes():
    t = derive("owner", ["clause:window"])
    t2 = TransitionContract.from_dict(t.to_dict())
    assert t2.update.changes == ["clause:window"]
    assert TransitionContract.from_dict(derive("owner", None).to_dict()).update.changes is None


def test_contradicting_the_declared_changes_is_a_lint_error():
    t = derive("customer", ["clause:window"])
    t.cases[0].authority = True
    t.cases[0].after = "refund"
    assert any(code == "T014" for lvl, code, _ in t.lint() if lvl == "error")


def test_sequences_and_the_reference_trajectory_honor_declared_changes():
    u = Update("u", "customer", "user", "60 days now", {"day_40": "refund"}, changes=["clause:window"])
    seq = SequenceContract(id="s", initial=GoverningState("v1", "30-day window"), cases=[case()], updates=[u],
                           sources=SOURCES, outcomes=OUTCOMES)
    assert seq.footprint(1) == {}
    traj = reference.warranted_trajectory(
        {"day_40": "deny"}, [{"source": "customer", "asserts": {"day_40": "refund"}, "changes": ["clause:window"]}],
        {"customer": ["fact:contact_pref"]}, {"day_40": ["clause:window", "fact:contact_pref"]})
    assert traj[-1] == {"day_40": "deny"}


def test_builtin_suites_declare_changes_wherever_authority_rests_on_overlap():
    for name in ("authority_returns", "authority_pharmacy", "authority_hr", "authority_bank"):
        for t in load_builtin_transitions(name):
            assert not [i for i in t.lint() if i[1] in ("T106", "T014")], (name, t.id)


def test_random_contracts_derive_the_same_labels_as_the_reference():
    rng = random.Random(7)
    grounds_pool = ["g1", "g2", "g3", "g4"]
    for _ in range(300):
        gov = rng.sample(grounds_pool, rng.randint(0, 3))
        srcs = {"s": Source("s", "", gov)}
        cases = [TransitionCase(f"c{i}", "q", "a", "a", [], None, rng.sample(grounds_pool, rng.randint(1, 3)))
                 for i in range(4)]
        changes = rng.choice([None, rng.sample(grounds_pool, rng.randint(1, 2))])
        asserts = {c.id: "b" for c in cases if rng.random() < 0.7}
        t = derive_transition("t", GoverningState("v", "x"), cases, Update("u", "s", "system", "x", asserts,
                                                                             changes=changes), srcs, {"a": "", "b": ""})
        for c, orig in zip(t.cases, cases):
            want = reference.warranted_after("a", asserts.get(c.id), gov, orig.grounds) if changes is None else \
                ("b" if c.id in asserts and reference.authorized(gov, orig.grounds, changes) else "a")
            assert c.after == want
