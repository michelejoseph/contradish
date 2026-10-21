"""The placebo noise floor: an authorized update that changes nothing, scored apart from the authority suites."""
from contradish.study import (AUTHORITY_SUITES, label_echo_classifier, placebo_transitions, run_authority,
                              scripted_chat_for)


def test_one_placebo_per_suite_and_every_case_must_hold():
    ps = placebo_transitions()
    assert [p.suite_name for p in ps] == list(AUTHORITY_SUITES)
    for p in ps:
        assert p.meaning_preserving and p.update.asserts == {}
        assert p.update.changes and all(c.after == c.before for c in p.cases)
        assert not p.lint()
        clause = p.update.changes[0]
        assert f"[{clause}]" in p.update.content and p.update.content.split(": ", 1)[1] in p.before.text


def jittery_chat_for(t, delivery=None):
    """Faithful, except that any message arriving after the first turn knocks the answer to another label."""
    inner = scripted_chat_for("faithful")(t, delivery)
    first = {c.id: [{"role": "system", "content": t.system_text("before")}, {"role": "user", "content": c.question}]
             for c in t.cases}

    def chat(messages):
        ans = inner(messages)
        last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        case = next((c for c in t.cases if c.question in last_user), None)
        if case is not None and messages != first[case.id]:
            others = [o for o in t.outcomes_for(case) if o != ans]
            return others[0] if others else ans
        return ans
    return chat


def test_a_faithful_agent_shows_no_drift_and_a_jittery_one_shows_it_all():
    for name, chat_for, want in (("faithful", scripted_chat_for("faithful"), 0.0), ("jittery", jittery_chat_for, 1.0)):
        res = run_authority(chat_for, label_echo_classifier, ["fresh"], samples=1, workers=1)
        pl = res["fresh"]["placebo"]
        assert pl["drift"]["n"] > 20 and pl["unclassifiable"] == 0, name
        assert pl["drift"]["value"] == want, (name, pl["drift"])


def test_unclassifiable_answers_are_not_counted_as_holding():
    def mute_chat_for(t, delivery=None):
        return lambda messages: "I cannot say."
    pl = run_authority(mute_chat_for, label_echo_classifier, ["fresh"], samples=1, workers=1)["fresh"]["placebo"]
    assert pl["drift"]["n"] == 0 and pl["drift"]["value"] is None and pl["unclassifiable"] > 20


def test_the_suites_own_numbers_do_not_include_the_placebo():
    res = run_authority(scripted_chat_for("faithful"), label_echo_classifier, ["fresh"], samples=1, workers=1)
    ids = {p["transition"] for p in res["fresh"]["transitions"]}
    assert not any(i.endswith("__placebo") for i in ids) and len(ids) == 28
