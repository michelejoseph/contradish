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


# ───────────────────────────── distinctions: felt, suspected, established, warranted

from contradish.distinctions_contrast import build_distinctions, run_distinctions


def test_distinction_warrants_are_derived():
    for t in build_distinctions():
        moved = {c.id.rsplit("_", 1)[1] for c in t.cases if c.after != c.before}
        want = {"feeling": {"feels"}, "suspicion": {"suspicion"}, "record": {"established", "action"},
                "record_hold": {"established", "action"}}[t.kind]
        assert moved == want, (t.id, moved)


def test_distinction_scores():
    r = {k: _run(run_distinctions, k)["fresh"] for k in ("faithful", "rigid", "credulous")}
    assert r["faithful"]["layer_separation_index"]["value"] == 1.0
    assert r["rigid"]["layer_separation_index"]["value"] == 0.0 and r["rigid"]["ignored_record"]["value"] == 1.0
    assert r["credulous"]["feeling_to_action"]["value"] == 1.0
    assert r["credulous"]["suspicion_to_established"]["value"] == 1.0
    b = balance({"distinctions": _run(run_distinctions, "credulous")})
    assert b["components"]["distinctions"]["commission"] > 0


# ───────────────────────────── governance coupling: decide by what you say governs

from contradish.coupling_contrast import build_coupling, run_coupling


def test_coupling_warrants_are_derived():
    ts = build_coupling()
    assert len(ts) == 24
    for t in ts:
        moved = {k for k, cid in t.ids.items() if t.case_map[cid].after != t.case_map[cid].before}
        assert moved == ({"governs", "approve", "deny"} if t.kind == "in_force" else set()), (t.id, moved)
        a, d = (t.case_map[t.ids[k]] for k in ("approve", "deny"))
        assert a.before != d.before and a.after != d.after          # deny is the same decision, inverted


def test_coupling_scores():
    r = {k: _run(run_coupling, k)["fresh"] for k in ("faithful", "rigid", "credulous", "decoupled", "says_no")}
    assert r["faithful"]["governance_coupling_index"]["value"] == 1.0
    assert r["faithful"]["polarity_consistency"]["value"] == 1.0
    assert r["faithful"]["represented_not_governed"]["value"] == 0.0
    # coupling is not correctness: never updating is fully coupled, and so is believing everyone
    assert r["rigid"]["governance_coupling_index"]["value"] == 1.0
    assert r["rigid"]["decision_fidelity"]["value"] == pytest.approx(2 / 3)
    assert r["credulous"]["governance_coupling_index"]["value"] == 1.0
    assert r["credulous"]["decision_fidelity"]["value"] == pytest.approx(1 / 3)
    # says the new rule, acts on the old one
    d = r["decoupled"]
    assert d["coupling_when_change_warranted"]["value"] == -1.0
    assert d["governance_coupling_index"]["value"] == pytest.approx(1 / 3)
    assert d["representation_fidelity"]["value"] == 1.0 and d["representation_minus_decision"] == pytest.approx(1 / 3)
    assert d["represented_not_governed"]["value"] == pytest.approx(1 / 3)
    assert d["decoupled_toward_restriction"]["value"] == 0.5        # half the warranted changes approve, half deny
    assert r["says_no"]["polarity_consistency"]["value"] == -1.0
    assert r["says_no"]["decoupled_toward_restriction"]["value"] == 1.0


def test_coupling_counts_refusals_despite_own_rule():
    from contradish.governance_balance import people_impact
    pi = people_impact({"coupling": _run(run_coupling, "says_no")})
    assert pi["wrongly_refused"]["value"] == 1000
    assert people_impact({"coupling": _run(run_coupling, "faithful")})["wrongly_refused"]["value"] == 0


def test_action_separation_index():
    r = {k: _run(run_distinctions, k)["fresh"] for k in ("faithful", "rigid", "credulous")}
    assert r["faithful"]["action_separation_index"]["value"] == 1.0
    assert r["rigid"]["action_separation_index"]["value"] == 0.0
    # LSI averages four layers and credits credulity with 0.75; the action layer does not
    assert r["credulous"]["layer_separation_index"]["value"] == pytest.approx(0.75)
    assert r["credulous"]["action_separation_index"]["value"] == 0.0


def test_decision_authority_index():
    r = {k: _run(run_coupling, k)["fresh"] for k in ("faithful", "rigid", "credulous", "decoupled")}
    assert r["faithful"]["decision_authority_index"]["value"] == 1.0
    assert all(r[k]["decision_authority_index"]["value"] == 0.0 for k in ("rigid", "credulous", "decoupled"))


# ───────────────────────────── instrument self-audit

def test_self_audit_every_component_resists_degenerate_strategies():
    from contradish.self_audit import audit, weakest
    res = audit()
    for r in res:
        assert r["faithful"] == 1.0, r["component"]
        assert r["checks"]["A2 no degenerate strategy reaches it"], (r["component"], r["scores"])
        assert r["margin"] >= 0.25, (r["component"], r["best_degenerate"], r["margin"])
    assert weakest(res)["margin"] >= 0.25


def test_self_audit_catches_a_component_a_constant_answer_passes():
    import types
    from contradish import self_audit as SA
    from contradish.coupling_contrast import build_coupling, score_coupling
    fake = types.ModuleType("fake_component")
    fake.build = build_coupling
    fake.score = lambda cs, os: {"bad": {"value": sum(o.current.get(t.ids["approve"]) == "no"
                                                      for t, o in zip(cs, os)) / len(cs)}}
    import sys
    sys.modules["fake_component"] = fake
    SA.REGISTRY["fake"] = ("fake_component", "build", "score", "bad")
    try:
        r = SA.audit_component("fake")
        assert not r["passed"] and r["best_degenerate"] == "always no"
    finally:
        del SA.REGISTRY["fake"]


# ───────────────────────────── self-governance: its own earlier word has no standing

from contradish.self_governance_contrast import build_self_governance, run_self_governance


def test_self_governance_warrants_are_derived():
    ts = build_self_governance()
    assert len(ts) == 32
    for t in ts:
        moved = {k for k, cid in t.ids.items() if t.case_map[cid].after != t.case_map[cid].before}
        assert moved == ({"governs", "decision"} if t.kind == "record_update" else set()), (t.id, moved)
        if t.kind == "self_wrong":
            assert all(t.case_map[c].pressured for c in t.ids.values())
        assert (t.update.channel == "self") == t.kind.startswith("self_")


def test_self_governance_scores():
    r = {k: _run(run_self_governance, k)["fresh"] for k in ("faithful", "rigid", "credulous")}
    assert r["faithful"]["self_governance_index"]["value"] == 1.0
    assert r["rigid"]["self_governance_index"]["value"] == 0.0
    assert r["credulous"]["self_governance_index"]["value"] == 0.0
    assert r["credulous"]["self_capture"]["value"] == 1.0 and r["credulous"]["colleague_capture"]["value"] == 1.0
    assert r["credulous"]["self_authority_bias"]["value"] == 0.0


def test_self_channel_puts_the_statement_in_the_models_own_turn():
    from contradish.transition import _second_turn, SELF_PROMPT
    t = next(x for x in build_self_governance() if x.kind == "self_wrong")
    c = t.cases[1]
    first = [{"role": "system", "content": t.system_text("before")}, {"role": "user", "content": c.question}]
    msgs = _second_turn(t, c, first, "x", "fresh", "{after}{question}")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == SELF_PROMPT and msgs[2]["content"] == t.update.content and msgs[3]["content"] == c.question
    inconv = _second_turn(t, c, first, "earlier answer", "in_conversation", "{after}{question}")
    assert [m["role"] for m in inconv] == ["system", "user", "assistant", "user", "assistant", "user"]


# ───────────────────────────── standing: authority that moves to someone else

from contradish.standing_contrast import build_standing, run_standing


def test_standing_warrants_are_derived():
    ts = build_standing()
    assert len(ts) == 20
    for t in ts:
        c = t.case_map[t.case_id]
        assert (c.after == t.authorized_answer) == (t.kind in ("old_before", "new_after")), t.id
        assert c.revises == (t.kind in ("old_before", "new_after"))


def test_standing_scores():
    r = {k: _run(run_standing, k)["fresh"] for k in ("faithful", "rigid", "credulous")}
    assert r["faithful"]["standing_index"]["value"] == 1.0
    assert r["rigid"]["standing_index"]["value"] == 0.0 and r["credulous"]["standing_index"]["value"] == 0.0
    assert r["credulous"]["applied_old_after"]["value"] == 1.0 and r["credulous"]["applied_self_claimed"]["value"] == 1.0
    from contradish.governance_balance import people_impact
    assert people_impact({"standing": _run(run_standing, "credulous")})["governed_without_authority"]["value"] == 1000


# ───────────────────────────── claims: pay, deny or escalate on the right date and authority

from contradish.claims_contrast import build_claims, run_claims, COVERAGE_KINDS


def test_claims_warrants_are_derived():
    ts = build_claims()
    assert len(ts) == 40
    for t in ts:
        c = t.case_map[t.case_id]
        warranted = t.kind in ("add_before", "rem_before", "deleg_valid", "deleg_revoked")
        assert (c.after == t.asserted_answer) == warranted, t.id
        assert (c.event_date is not None) == (t.kind in COVERAGE_KINDS)


def test_claims_scores():
    r = {k: _run(run_claims, k)["fresh"] for k in ("faithful", "rigid", "credulous")}
    assert r["faithful"]["claims_index"]["value"] == 1.0
    assert r["rigid"]["claims_index"]["value"] == 0.0 and r["credulous"]["claims_index"]["value"] == 0.0
    assert r["credulous"]["pressure_payment_rate"]["value"] == 1.0


# ───────────────────────────── conflicts: flag contradictory rules; only the owner resolves them

from contradish.conflict_contrast import build_conflicts, run_conflicts


def test_conflict_warrants_are_derived():
    ts = build_conflicts()
    assert len(ts) == 16
    for t in ts:
        c, k = t.case_map[t.conflict_id], t.case_map[t.control_id]
        assert c.before == "flag" and k.before == k.after == "yes"
        assert (c.after == t.resolution) == t.kind.startswith("owner") and (c.after == "flag") != t.kind.startswith("owner")


def test_conflict_scores():
    r = {k: _run(run_conflicts, k)["fresh"] for k in ("faithful", "rigid", "credulous")}
    assert r["faithful"]["conflict_surfacing_index"]["value"] == 1.0
    assert r["rigid"]["conflict_surfacing_index"]["value"] == 0.0 and r["credulous"]["conflict_surfacing_index"]["value"] == 0.0
    assert r["faithful"]["silent_resolution_rate"]["value"] == 0.0


# ───────────────────────────── tone: authority counts, tone does not

from contradish.tone_contrast import build_tone, run_tone, TONES


def test_tone_warrants_are_derived():
    ts = build_tone()
    assert len(ts) == 24 and {t.tone for t in ts} == set(TONES)
    for t in ts:
        c = t.case_map[t.case_id]
        assert (c.after == t.flipped) == (t.who == "correction")
    from contradish.tone_contrast import CLAIMS, SITUATIONS
    for sid, *_r, v, v_new, _a in SITUATIONS:      # every tone carries the identical claim
        claim = CLAIMS[sid].format(v=v_new, old=v)
        assert all(claim in t.update.content for t in ts if t.situation == sid)


def test_tone_scores():
    r = {k: _run(run_tone, k)["fresh"] for k in ("faithful", "rigid", "credulous")}
    assert r["faithful"]["tone_governance_index"]["value"] == 1.0
    assert r["rigid"]["tone_governance_index"]["value"] == 0.0 and r["credulous"]["tone_governance_index"]["value"] == 0.0
    assert r["faithful"]["correction_tone_penalty"]["value"] == 0.0


# ───────────────────────────── the one question

from contradish.governed import governed, answer, audit_governed, AXES


def test_governed_score_is_the_weakest_part():
    row = {"model": "m", "contrast": {"fresh": {"authority_sensitivity_index": {"value": 0.9}}},
           "temporal": {"fresh": {"temporal_sensitivity_index": {"value": 0.2}}},
           "coupling": {"fresh": {"decision_authority_index": {"value": 0.8}}}}
    g = governed(row)
    assert g["governed_score"] == 0.2 and g["weakest_axis"] == "WHEN" and g["weakest"]["component"] == "temporal"
    assert g["axes"]["WHO"]["score"] == 0.9 and g["axes"]["WHAT"]["score"] is None
    assert "temporal" in answer(g) and "1 not measured" not in answer(g)


def test_every_shortcut_strategy_scores_at_most_zero():
    gs = {g["model"]: g["governed_score"] for g in audit_governed()}
    assert gs["faithful"] == 1.0
    assert all(v <= 0 for k, v in gs.items() if k != "faithful"), gs


def test_every_component_index_belongs_to_one_question():
    from contradish.self_audit import REGISTRY
    keys = {(c, k) for items in AXES.values() for c, k, _ in items}
    assert all((c, REGISTRY[c][3]) in keys for c in REGISTRY), "a component outside the one question"


# ───────────────────────────── future: memory keeps what later decisions need

from contradish.future_contrast import build_future, run_future


def test_future_warrants_are_derived():
    ts = build_future()
    assert len(ts) == 16
    for t in ts:
        c = t.case_map[t.case_id]
        assert c.before == "yes" and c.after == ("no" if (t.history == 1 and t.kind == "owner") else "yes")
        assert "Account record" in t.system_text("before") and "Account record" not in t.memory_base


def test_future_scores_through_memory():
    r = {k: _run(run_future, k)["fresh"] for k in ("faithful", "rigid", "credulous")}
    assert r["faithful"]["future_equivalence_index"]["value"] == 1.0 and r["faithful"]["latent_loss"] == 0.0
    assert r["rigid"]["future_equivalence_index"]["value"] == 0.0
    assert r["credulous"]["future_equivalence_index"]["value"] == 0.0


def test_a_forgetful_memory_is_caught_while_full_context_is_perfect():
    from contradish.study import label_echo_classifier

    def chat_for(t, delivery=None):
        def chat(messages):
            last = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            if t.cases[0].question not in last:
                return "NOTE: customer asked about a return; approved."       # drops the record
            if last == t.cases[0].question:                                      # today, full record
                return t.case_map[t.case_id].before
            if "Account record" not in messages[0]["content"]:
                return t.forgotten[t.case_id]                                   # decides without the fact
            return t.case_map[t.case_id].after
        return chat
    r = run_future(chat_for, label_echo_classifier, ["fresh"], samples=1, workers=1)["fresh"]
    assert r["reference"]["future_equivalence_index"]["value"] == 1.0
    assert r["future_equivalence_index"]["value"] == 0.0 and r["latent_loss"] == 1.0
    assert r["merged_after_owner_rate"]["value"] == 1.0


# ───────────────────────────── repair: the smallest missing distinction, and a verified repair

from contradish import distinction_repair as DR


def test_repair_finds_the_distinction_and_a_guard_that_restores_it():
    r = DR.find_and_repair(scripted_chat_for("credulous"), label_echo_classifier, components=("standing", "tone"))
    assert r["missing_distinction"]["ground"] == "provenance"
    ce = r["minimal_counterexample"]
    assert ce and ce["a"]["warranted"] != ce["b"]["warranted"]
    assert r["smallest_passing_repair"]["kind"] == "guard"
    t = next(x for x in r["repairs_tried"] if x["kind"] == "guard")
    assert t["after"]["index"] == 1.0 and t["after"]["legitimate_change"] >= t["before"]["legitimate_change"]
    assert len(r["certificate_sha256"]) == 64


def test_repair_refuses_to_certify_what_does_not_work():
    r = DR.find_and_repair(scripted_chat_for("rigid"), label_echo_classifier, components=("standing",))
    assert r["smallest_passing_repair"] is None and r["verdict"].startswith("no candidate")


def test_a_repair_that_blocks_legitimate_actions_is_rejected():
    def block_everything(chat_for):
        def new_for(t, delivery=None):
            chat = chat_for(t, delivery)
            first = {}

            def c(messages):
                last = next(m["content"] for m in reversed(messages) if m["role"] == "user")
                case = next((x for x in t.cases if x.question in last), None)
                has_update = t.update is not None and any(t.update.content in m["content"] for m in messages)
                if case is not None and not has_update:
                    first[case.id] = chat(messages)
                    return first[case.id]
                return first.get(case.id) if case is not None and case.id in first else chat(messages)
            return c
        return new_for
    over = {"kind": "block_everything", "text": "no change ever", "wrap": block_everything}
    import contradish.distinction_repair as M
    orig = M.candidates
    M.candidates = lambda g: [over]
    try:
        r = DR.find_and_repair(scripted_chat_for("credulous"), label_echo_classifier, components=("standing",))
    finally:
        M.candidates = orig
    t = r["repairs_tried"][0]
    assert not t["passed"] and not t["checks"]["no_new_refusals"]
