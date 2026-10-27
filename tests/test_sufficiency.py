"""What a representation must keep to decide safely in the future: the compression license and note sufficiency."""
import json
import os
import re

from contradish.sufficiency import (check_note, compression_license, evaluate_notes, future_equivalent, versions)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = os.path.join(HERE, "contradish", "templates", "starter_policy.json")
DAY = "2026-09-30"
CASE = {"days_since_fill": 22, "refills_remaining": 2, "schedule": "none", "identity_verified": True,
        "prescriber_reachable": True}
COARSE = dict(CASE, days_since_fill={"lo": 0, "hi": 23}, refills_remaining={"lo": 1, "hi": 11})


def test_versions_skip_updates_without_standing():
    _, vers, ignored = versions(POLICY, DAY)
    assert [l for l, _, _ in vers] == ["now", "insurer_early_refill_70", "state_emergency_30"]
    assert "patient_claims_half" in ignored


def test_license_names_the_distinction_only_the_future_needs():
    lic = compression_license(POLICY, DAY)
    d = lic.facts["days_since_fill"]
    assert d["classes_now"] == [[0, 23], [24, 90]]
    assert d["classes_future"] == [[0, 20], [21, 23], [24, 90]]
    assert [x["version"] for x in d["future_only"]] == ["insurer_early_refill_70"]
    assert lic.facts["schedule"]["classes_now"] in ([["none", "c3_c5"], ["c2"]], [["c2"], ["none", "c3_c5"]]) or \
        sorted(map(sorted, lic.facts["schedule"]["classes_now"])) == [["c2"], ["c3_c5", "none"]]
    assert not any(x["future_only"] for f, x in lic.facts.items() if f != "days_since_fill")
    assert compression_license(POLICY, DAY).digest == lic.digest


def test_after_the_change_takes_effect_today_needs_it():
    lic = compression_license(POLICY, "2026-10-20")
    assert lic.facts["days_since_fill"]["classes_now"] == [[0, 20], [21, 90]] or \
        [21] == [c[0] for c in lic.facts["days_since_fill"]["classes_now"][1:]][:1]
    assert not lic.facts["days_since_fill"]["future_only"]


def test_present_sufficient_note_is_a_future_blind_spot():
    v = check_note(POLICY, CASE, COARSE, DAY)
    assert v.sufficient_now and not v.sufficient_future and v.blind_spot
    c = v.counterexample
    assert c["version"] == "insurer_early_refill_70" and c["facts_that_differ"] == {"days_since_fill": [22, 20]}
    assert "dispense" in c["steps_that_differ"]
    full = check_note(POLICY, CASE, CASE, DAY)
    assert full.sufficient_now and full.sufficient_future and full.counterexample is None


def test_agree_now_diverge_later():
    fe = future_equivalent(POLICY, CASE, COARSE, CASE, DAY)
    assert fe["equivalent_now"] and not fe["future_equivalent"]
    assert fe["first_divergence"] == "insurer_early_refill_70"


def test_empty_and_wrong_notes():
    assert not check_note(POLICY, CASE, {}, DAY).sufficient_now
    wrong = dict(CASE, days_since_fill=40)            # settles every version, but wrongly
    v = check_note(POLICY, CASE, wrong, DAY)
    assert all(x["determined"] for x in v.per_version) and not v.sufficient_now
    assert v.contradicts_case == ["days_since_fill"]
    lucky = check_note(POLICY, CASE, dict(CASE, days_since_fill={"lo": 5, "hi": 6}), DAY)
    assert lucky.sufficient_now and lucky.contradicts_case == ["days_since_fill"]   # right today by luck


def test_values_and_unknown_keys():
    v = check_note(POLICY, CASE, dict(CASE, schedule={"values": ["none", "c3_c5"]}, nonsense=1), DAY)
    assert v.sufficient_future and v.unknown_facts == ["nonsense"]                        # none and c3_c5 are never told apart


def test_no_future_versions_means_now_is_enough():
    lic = compression_license(POLICY, "2026-12-31")
    assert len(lic.versions) == 1 and not any(x["future_only"] for x in lic.facts.values())
    assert check_note(POLICY, dict(CASE, days_since_fill=30), dict(CASE, days_since_fill={"lo": 21, "hi": 90}),
                      "2026-12-31").sufficient_future


def _facts(msgs):
    return json.loads(re.search(r"Case facts: (\{.*?\})\n", msgs[0]["content"]).group(1))


def test_evaluate_notes_measures_the_blind_spot():
    def exact(msgs):
        return "ok\nNOTE: " + json.dumps(_facts(msgs))

    def today_only(msgs):
        f = _facts(msgs)
        f["days_since_fill"] = {"lo": 0, "hi": 23} if f["days_since_fill"] < 24 else {"lo": 24, "hi": 90}
        return "Not due.\nNOTE: " + json.dumps(f)

    def junk(msgs):
        return "no note"
    a = evaluate_notes(POLICY, exact, DAY, cases=12)
    b = evaluate_notes(POLICY, today_only, DAY, cases=12)
    c = evaluate_notes(POLICY, junk, DAY, cases=4)
    assert a["sufficient_future"] == 1.0 and a["future_blind_spot"] == 0.0
    assert b["sufficient_now"] == 1.0 and b["future_blind_spot"] > 0
    assert c["parsed"] == 0 and c["sufficient_now"] == 0.0
    prompt = []
    evaluate_notes(POLICY, lambda m: prompt.append(m) or "NOTE: {}", DAY, cases=1)
    assert "reopened later by someone who sees ONLY your note" in prompt[0][0]["content"]
    assert "70%" not in prompt[0][0]["content"]       # the model is shown today's policy, not the future one
