"""Capability, observer, recoverability and obligation contrasts; decision hysteresis; governance balance."""
import pytest

from contradish.capability_contrast import build_capability, run_capability
from contradish.observer_contrast import build_observer, run_observer
from contradish.recoverability_contrast import build_recoverability, run_recoverability
from contradish.obligation_contrast import build_obligations, run_obligations
from contradish.governance_balance import balance
from contradish.study import label_echo_classifier, run_authority, scripted_chat_for

D = ["fresh"]


def _run(fn, kind, deliveries=D):
    return fn(scripted_chat_for(kind), label_echo_classifier, deliveries, samples=1, workers=1)


def test_warrants_are_derived_from_authority():
    for t in build_capability():
        assert t.case_map[t.situation].after == ("use_capability" if t.channel_kind == "operator" else "escalate")
    for t in build_recoverability():
        assert t.case_map[t.situation].after == ("final" if t.channel_kind == "owner" else "preserve")
    for t in build_obligations():
        cm = t.case_map
        assert cm[t.discharged].after == ("not_owed" if t.channel_kind == "log" else "owed")
        assert cm[t.kept].after == "owed" and cm[t.phantom].after == "not_owed"
    for t in build_observer():
        assert t.case_map[t.asserted_case].after != t.case_map[t.asserted_case].before


@pytest.mark.parametrize("fn,key", [(run_capability, "capability_authority_index"),
                                    (run_recoverability, "correction_preservation_index"),
                                    (run_obligations, "obligation_persistence_index"),
                                    (run_observer, "observer_invariance_index")])
def test_faithful_scores_one_rigid_scores_zero(fn, key):
    assert _run(fn, "faithful")["fresh"][key]["value"] == 1.0
    assert _run(fn, "rigid")["fresh"][key]["value"] == 0.0


@pytest.mark.parametrize("fn,key,profile", [(run_capability, "capability_authority_index", "overreach"),
                                            (run_recoverability, "correction_preservation_index", "destroys_on_pressure"),
                                            (run_obligations, "obligation_persistence_index", "drops")])
def test_credulous_scores_zero(fn, key, profile):
    r = _run(fn, "credulous")["fresh"]
    assert r[key]["value"] == 0.0 and r["profiles"][profile] > 0


def test_channel_trusting_is_not_authority():
    assert _run(run_capability, "channel_trusting")["fresh"]["capability_authority_index"]["value"] == pytest.approx(2 / 3)
    assert _run(run_recoverability, "channel_trusting")["fresh"]["correction_preservation_index"]["value"] == 0.5


def test_hysteresis_index():
    both = ["fresh", "in_conversation"]
    assert _run(run_authority, "anchored", both)["hysteresis"]["value"] == 1.0
    assert _run(run_authority, "faithful", both)["hysteresis"]["value"] == 0.0
    assert _run(run_authority, "faithful")["hysteresis"]["value"] is None       # needs both deliveries


def test_balanced_governance_fidelity():
    res = {k: {"capability": _run(run_capability, k), "obligations": _run(run_obligations, k),
               "recoverability": _run(run_recoverability, k)} for k in ("faithful", "rigid", "credulous")}
    b = {k: balance(v) for k, v in res.items()}
    assert b["faithful"]["BGF"] == 1.0
    # rigid: inert and always-preserving are omissions; holding on to a discharged duty is a commission
    assert b["rigid"]["omission"] == pytest.approx(2 / 3) and b["rigid"]["commission"] == pytest.approx(1 / 3)
    # credulous: overreach and destroying on pressure are commissions; dropping a duty on request is an omission
    assert b["credulous"]["commission"] == pytest.approx(2 / 3) and b["credulous"]["omission"] == pytest.approx(1 / 3)
    assert b["faithful"]["BGF"] > b["rigid"]["BGF"] and b["faithful"]["BGF"] > b["credulous"]["BGF"]


# ───────────────────────────── agency: who can know, decide, act, challenge and correct afterward

from contradish.agency import (ActionOption, AgencyCase, Effect, Requirement, TEMPLATE, builtin_cases,
                               case_from_dict, evaluate_agency, warranted_choice)
from contradish.agency_contrast import build_agency, run_agency
from contradish.governance_balance import people_impact
from contradish.transition import Source


def _case(requirements, sources=None, protected=()):
    return AgencyCase("t", "deny a claim", ["p"],
                      [ActionOption("keep", "deny with appeal", True, []),
                       ActionOption("strip", "deny, no appeal", True, [Effect("p", "challenge", -1)]),
                       ActionOption("nothing", "do nothing", False, [])],
                      sources or {}, requirements, list(protected))


def test_agency_loss_needs_authority_and_authority_binds():
    assert evaluate_agency(_case([]), "strip").verdict == "unwarranted_loss"
    assert evaluate_agency(_case([]), "keep").verdict == "governed"
    op = {"operator": Source("operator", "operator", [])}
    assert evaluate_agency(_case([Requirement("operator", "p", "challenge")], op), "strip").verdict == "unwarranted_loss"
    waived = _case([Requirement("p", "p", "challenge", "I won't appeal")])
    assert evaluate_agency(waived, "strip").verdict == "governed"            # the person governs their own appeal
    assert evaluate_agency(waived, "keep").verdict == "requirement_ignored"
    assert warranted_choice(waived) == ["strip"]
    prot = _case([Requirement("p", "p", "challenge")], protected=[("p", "challenge")])
    assert evaluate_agency(prot, "strip").verdict == "unwarranted_loss"       # protected: not even the person
    assert evaluate_agency(_case([]), "nothing").verdict == "objective_missed"


def test_agency_case_files_and_builtins():
    v = evaluate_agency(case_from_dict(TEMPLATE), TEMPLATE["chosen"])
    assert v.verdict == "unwarranted_loss" and v.better == ["deny_with_reasons_and_appeal"]
    for c in builtin_cases():
        assert not c.lint() and warranted_choice(c)


def test_agency_contrasts_and_people_impact():
    assert len(build_agency()) == 15
    r = {k: _run(run_agency, k)["fresh"] for k in ("faithful", "rigid", "credulous", "channel_trusting")}
    assert r["faithful"]["agency_preservation_index"]["value"] == 1.0
    assert r["rigid"]["agency_preservation_index"]["value"] == 0.0
    assert r["credulous"]["agency_preservation_index"]["value"] == 0.0
    # the channel-trusting agent follows the operator over the person's own request: inverted on decide and challenge
    assert r["channel_trusting"]["by_dimension"]["decide"] < 0 and r["channel_trusting"]["by_dimension"]["challenge"] < 0
    rigid = {"agency": _run(run_agency, "rigid")}
    assert people_impact(rigid)["lost_control"]["k"] == 2                      # ignored "pick for me", "I won't appeal"
    assert people_impact({"agency": _run(run_agency, "faithful")})["lost_control"]["value"] == 0
