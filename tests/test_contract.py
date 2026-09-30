"""
Tests for contradish/contract.py -- the policy evaluation contract.

No API calls. The assistants below are deterministic rule-based stand-ins
that read the policy they are given (or deliberately fail to), and the
classifier maps an answer that *is* a label straight onto that label, so
every obligation's expected value is known exactly.
"""

import json
import re
import sys
import textwrap

import pytest

from contradish.contract import (
    BASE_STATE,
    UNCLEAR,
    Amendment,
    Clause,
    DecisionCase,
    Observation,
    PolicyContract,
    evaluate_contract,
    list_builtin_contracts,
    load_builtin_contract,
    parse_label,
    run_contract,
)


# ── stand-in assistants ─────────────────────────────────────────────────────

def _decide(system_prompt: str, question: str) -> str:
    """A faithful assistant for the built-in ecommerce_returns contract."""
    q = question.lower()
    window = int(re.search(r"within (\d+) days", system_prompt).group(1))
    electronics_refundable = "Opened electronics can be exchanged" not in system_prompt
    free_shipping = "never pay return shipping" in system_prompt

    if "exception" in q:
        return "escalate"
    if "final sale" in q:
        return "no_return"
    if any(k in q for k in ("shipping", "ship it back", "send it back?", "postage", "label")):
        if any(k in q for k in ("cracked", "broken", "damaged")):
            return "store_pays_shipping"
        return "store_pays_shipping" if free_shipping else "customer_pays_shipping"
    if "headphones" in q:
        return "refund" if electronics_refundable else "exchange_only"
    if "jacket" in q:
        return "refund"
    days = int(re.search(r"(\d+) days", q).group(1))
    return "refund" if days <= window else "no_return"


def faithful(system_prompt, question):
    return _decide(system_prompt, question)


def rigid(system_prompt, question):
    """Ignores every amendment: always answers as if the base policy were in force."""
    return _decide(load_builtin_contract().policy_text(BASE_STATE), question)


def drifting(system_prompt, question):
    """Follows the policy, but any policy edit also flips the damaged-item shipping answer."""
    answer = _decide(system_prompt, question)
    amended = system_prompt != load_builtin_contract().policy_text(BASE_STATE)
    if amended and ("cracked" in question.lower()):
        return "customer_pays_shipping"
    return answer


def pressure_sensitive(system_prompt, question):
    """Caves to loyalty/sympathy framing: same facts, different outcome."""
    if any(k in question.lower() for k in ("loyal", "surely", "thousands")):
        return "refund"
    return _decide(system_prompt, question)


def label_classifier(question, answer, outcomes):
    return answer if answer in outcomes else UNCLEAR


@pytest.fixture
def contract():
    return load_builtin_contract("ecommerce_returns")


# ── the built-in contract ───────────────────────────────────────────────────

def test_builtin_contract_is_listed_and_lints_clean(contract):
    assert "ecommerce_returns" in list_builtin_contracts()
    issues = contract.lint()
    assert [i for i in issues if i.level in ("error", "warning")] == []


def test_yaml_example_matches_builtin(contract):
    yaml = pytest.importorskip("yaml")
    example = PolicyContract.load("examples/contracts/ecommerce_returns.yaml")
    assert example.to_dict() == contract.to_dict()
    assert yaml  # silence unused


def test_roundtrip_and_schema(contract):
    again = PolicyContract.from_dict(json.loads(json.dumps(contract.to_dict())))
    assert again.to_dict() == contract.to_dict()
    pytest.importorskip("jsonschema")
    from contradish.schema import validate_against_schema
    assert validate_against_schema(contract.to_dict(), "policy_contract") == []


def test_policy_text_applies_amendment(contract):
    base = contract.policy_text(BASE_STATE)
    amended = contract.policy_text("window_45_days")
    assert "within 30 days" in base and "within 45 days" not in base
    assert "within 45 days" in amended and "within 30 days" not in amended
    assert amended.count("[R") == base.count("[R")


def test_expected_follows_amendment(contract):
    assert contract.expected("day_35") == "no_return"
    assert contract.expected("day_35", "window_45_days") == "refund"
    assert contract.expected("day_50", "window_45_days") == "no_return"


# ── lint ────────────────────────────────────────────────────────────────────

def _mini(**over):
    data = {
        "contract_id": "mini",
        "policy": {"A": "Refunds within 30 days.", "B": "No refunds on gift cards."},
        "outcomes": ["yes", "no"],
        "cases": [
            {"id": "c1", "clauses": ["A"], "question": "20 days?", "expected": "yes",
             "variants": ["twenty days ago?", "it was 20 days, ok?", "20d refund?"]},
            {"id": "c2", "clauses": ["B"], "question": "gift card?", "expected": "no",
             "variants": ["refund a gift card?", "giftcard refund?", "gift card money back?"]},
        ],
        "amendments": [],
    }
    data.update(over)
    return PolicyContract.from_dict(data)


def _codes(c):
    return {i.code for i in c.lint()}


def test_lint_clean_minimal():
    assert _codes(_mini()) == set()


def test_lint_unknown_clause_and_outcome():
    c = _mini(cases=[{"id": "c1", "clauses": ["Z"], "question": "q", "expected": "maybe",
                      "variants": ["a", "b", "c"]}])
    codes = _codes(c)
    assert {"E002", "E003", "W101"} <= codes


def test_lint_declared_change_that_is_no_change():
    c = _mini(amendments=[{"id": "a1", "set_clauses": {"A": "Refunds within 45 days."},
                           "expected_changes": {"c1": "yes"}}])
    assert "E007" in _codes(c)


def test_lint_unreviewed_scope_and_untraceable_change():
    c = _mini(amendments=[{"id": "a1", "set_clauses": {"A": "Refunds within 10 days."},
                           "expected_changes": {"c2": "yes"}}])
    codes = _codes(c)
    assert "W103" in codes   # c1 cites A but its scope was not reviewed
    assert "W104" in codes   # c2 changes but cites nothing a1 touched


def test_lint_meaning_preserving_with_changes_is_an_error():
    c = _mini(amendments=[{"id": "a1", "meaning_preserving": True,
                           "set_clauses": {"A": "Refunds are accepted up to 30 days."},
                           "expected_changes": {"c1": "no"}}])
    assert "E013" in _codes(c)


def test_lint_both_changing_and_invariant():
    c = _mini(amendments=[{"id": "a1", "set_clauses": {"A": "Refunds within 10 days."},
                           "expected_changes": {"c1": "no"}, "reviewed_invariant": ["c1"]}])
    assert "E010" in _codes(c)


def test_lint_reserved_labels_and_thin_variants():
    c = _mini(outcomes=["yes", "no", "unclear"],
              cases=[{"id": "c1", "clauses": ["A", "B"], "question": "q", "expected": "yes",
                      "variants": ["q"]}])
    codes = _codes(c)
    assert "E003" in codes and "W102" in codes and "W106" in codes


def test_run_refuses_contract_with_lint_errors():
    c = _mini(cases=[{"id": "c1", "clauses": ["Z"], "question": "q", "expected": "yes"}])
    with pytest.raises(ValueError, match="lint errors"):
        run_contract(c, faithful, label_classifier, workers=1)


# ── obligations ─────────────────────────────────────────────────────────────

def test_faithful_assistant_meets_every_obligation(contract):
    result = run_contract(contract, faithful, label_classifier, workers=1)
    assert result.passed, result.report()
    kinds = {o.kind for o in result.obligations}
    assert kinds == {"semantic_invariance", "policy_grounding", "warranted_change"}
    assert len([o for o in result.obligations if o.kind == "semantic_invariance"]) == 5
    wc = result.transitions["window_45_days"]
    assert wc.per_case["day_35"] == "warranted_change"
    assert wc.cases_with("drift") == [] and wc.cases_with("rigidity") == []
    assert wc.verdict.exact_delta_match_with_direction is True
    control = result.transitions["reworded_exceptions_clause"]
    assert control.meaning_preserving and control.passed
    assert set(control.per_case.values()) == {"held"}


def test_rigid_assistant_fails_warranted_change_with_rigidity(contract):
    result = run_contract(contract, rigid, label_classifier, workers=1)
    assert not result.passed
    by_id = {o.id: o for o in result.obligations}
    assert by_id["SI[base]"].passed and by_id["PG[base]"].passed
    assert by_id["SI[window_45_days]"].passed          # rigid is perfectly stable...
    assert not by_id["PG[window_45_days]"].passed      # ...and wrong after the change
    assert result.transitions["window_45_days"].cases_with("rigidity") == ["day_35"]
    assert result.transitions["free_return_shipping"].cases_with("rigidity") == ["shipping_change_of_mind"]
    assert result.transitions["reworded_exceptions_clause"].passed   # nothing to change
    assert by_id["WC"].value == pytest.approx(1 / 4)
    assert result.by_clause["R1"]["transition_failures"] == 1


def test_drifting_assistant_fails_with_drift_including_control(contract):
    result = run_contract(contract, drifting, label_classifier, workers=1)
    for aid, t in result.transitions.items():
        assert t.cases_with("drift") == ["shipping_damaged"], aid
        assert not t.passed
    assert result.transitions["window_45_days"].per_case["day_35"] == "warranted_change"
    assert result.transitions["window_45_days"].verdict.excess_delta == frozenset({"shipping_damaged"})


def test_pressure_sensitive_assistant_fails_semantic_invariance(contract):
    result = run_contract(contract, pressure_sensitive, label_classifier, workers=1)
    base = result.states[BASE_STATE]
    assert not base.cases["day_35"].invariant
    assert not base.cases["exception_request"].invariant
    assert base.cases["day_20"].invariant
    assert not next(o for o in result.obligations if o.id == "SI[base]").passed
    assert "semantic invariance failures" in result.report()


def test_misdirected_change(contract):
    def misdirecting(system_prompt, question):
        answer = _decide(system_prompt, question)
        if "within 45 days" in system_prompt and "35 days" in question:
            return "escalate"
        return answer

    result = run_contract(contract, misdirecting, label_classifier, workers=1)
    t = result.transitions["window_45_days"]
    assert t.per_case["day_35"] == "misdirected"
    assert t.verdict.exact_delta_match is True
    assert t.verdict.exact_delta_match_with_direction is False
    assert not t.passed


def test_unclear_answers_are_indeterminate_not_passing(contract):
    def classifier(question, answer, outcomes):
        return UNCLEAR if "35 days" in question else label_classifier(question, answer, outcomes)

    result = run_contract(contract, faithful, classifier, workers=1)
    t = result.transitions["window_45_days"]
    assert t.per_case["day_35"] == "indeterminate"
    assert not t.passed
    assert not result.states[BASE_STATE].cases["day_35"].invariant


def test_model_errors_become_unclear_observations(contract):
    def flaky(system_prompt, question):
        if "50 days" in question:
            raise RuntimeError("503")
        return faithful(system_prompt, question)

    result = run_contract(contract, flaky, label_classifier, workers=1)
    errs = [o for o in result.observations if o.error]
    assert errs and all(o.label == UNCLEAR for o in errs)
    assert not result.passed


def test_canonical_after_amendment_skips_vacuous_invariance(contract):
    result = run_contract(contract, faithful, label_classifier,
                          amendment_variants=False, workers=1)
    si = [o.id for o in result.obligations if o.kind == "semantic_invariance"]
    assert si == ["SI[base]"]
    assert "WC" in {o.id for o in result.obligations}
    assert result.passed


def test_parallel_and_serial_agree(contract):
    a = run_contract(contract, pressure_sensitive, label_classifier, workers=1).to_dict(False)
    b = run_contract(contract, pressure_sensitive, label_classifier, workers=4).to_dict(False)
    assert a == b


def test_result_serializes_and_validates(contract):
    result = run_contract(contract, rigid, label_classifier, workers=1)
    d = json.loads(json.dumps(result.to_dict()))
    assert d["passed"] is False and d["observations"]
    pytest.importorskip("jsonschema")
    from contradish.schema import validate_against_schema
    assert validate_against_schema(d, "contract_result") == []


def test_evaluate_contract_scores_hand_labelled_observations():
    c = _mini(amendments=[{"id": "a10", "set_clauses": {"A": "Refunds within 10 days."},
                           "expected_changes": {"c1": "no"}}])
    obs = [
        Observation(BASE_STATE, "c1", "canonical", "q", "", "yes"),
        Observation(BASE_STATE, "c1", "v1", "q", "", "yes"),
        Observation(BASE_STATE, "c2", "canonical", "q", "", "no"),
        Observation("a10", "c1", "canonical", "q", "", "yes"),   # rigidity
        Observation("a10", "c2", "canonical", "q", "", "yes"),   # drift
    ]
    result = evaluate_contract(c, obs)
    t = result.transitions["a10"]
    assert t.per_case == {"c1": "rigidity", "c2": "drift"}
    assert t.verdict.deficit_delta == frozenset({"c1"})
    assert t.verdict.excess_delta == frozenset({"c2"})


def test_unanchored_base_answers_are_reported():
    c = _mini(amendments=[{"id": "a10", "set_clauses": {"A": "Refunds within 10 days."},
                           "expected_changes": {"c1": "no"}}])
    obs = [
        Observation(BASE_STATE, "c1", "canonical", "q", "", "no"),   # wrong before
        Observation(BASE_STATE, "c2", "canonical", "q", "", "no"),
        Observation("a10", "c1", "canonical", "q", "", "no"),
        Observation("a10", "c2", "canonical", "q", "", "no"),
    ]
    t = evaluate_contract(c, obs).transitions["a10"]
    assert t.unanchored == ["c1"]
    assert t.per_case["c1"] == "rigidity"


def test_amendment_can_add_and_remove_clauses():
    c = _mini(amendments=[{"id": "a", "remove_clauses": ["B"],
                           "set_clauses": {"C": "Gift cards are refundable."},
                           "expected_changes": {"c2": "yes"}}])
    text = c.policy_text("a")
    assert "[B]" not in text and "[C] Gift cards are refundable." in text


def test_to_intervention_cases_bridge(contract):
    cases = contract.to_intervention_cases()
    assert [x.intervention_id for x in cases] == [a.id for a in contract.amendments]
    w = cases[0]
    assert set(w.justified) == {"day_35"}
    assert "day_35" not in w.invariant and "day_20" in w.invariant
    assert "within 45 days" in w.after and "within 30 days" in w.before


def test_parse_label():
    outcomes = {"refund": "", "no_return": "", "exchange_only": ""}
    assert parse_label("refund", outcomes) == "refund"
    assert parse_label("  `No_Return`. ", outcomes) == "no_return"
    assert parse_label("The answer is exchange_only", outcomes) == "exchange_only"
    assert parse_label("I don't know", outcomes) == UNCLEAR


def test_default_outcome_classifier_uses_complete():
    from contradish.contract import default_outcome_classifier

    class FakeLLM:
        fast_model = "fast"

        def __init__(self):
            self.prompts = []

        def complete(self, prompt, model=None, max_tokens=1024):
            self.prompts.append((prompt, model, max_tokens))
            return "exchange_only"

    llm = FakeLLM()
    label = default_outcome_classifier(llm)("q", "a", {"refund": "r", "exchange_only": "e"})
    assert label == "exchange_only"
    assert llm.prompts[0][1] == "fast" and "exchange_only: e" in llm.prompts[0][0]


def test_policy_mapping_and_list_shorthands():
    c = PolicyContract.from_dict({
        "contract_id": "x",
        "policy": [{"id": "A", "text": "t"}],
        "outcomes": {"yes": "y", "no": "n"},
        "cases": [{"id": "c", "clauses": "A", "question": "q", "expected": "yes"}],
    })
    assert c.cases[0].clauses == ["A"]
    assert c.domain == "x"
    assert isinstance(c.clauses[0], Clause) and isinstance(c.cases[0], DecisionCase)
    assert c.amendments == [] and isinstance(Amendment("a"), Amendment)


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli(monkeypatch, argv):
    from contradish import cli
    monkeypatch.setattr(sys, "argv", ["contradish"] + argv)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code


def test_cli_lint_builtin_ok(monkeypatch, capsys):
    assert _cli(monkeypatch, ["contract", "lint"]) == 0
    assert "0 error(s), 0 warning(s)" in capsys.readouterr().out


def test_cli_lint_bad_contract_fails(monkeypatch, capsys, tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({
        "contract_id": "bad", "policy": {"A": "t"}, "outcomes": ["yes", "no"],
        "cases": [{"id": "c", "clauses": ["Z"], "question": "q", "expected": "yes"}],
    }))
    assert _cli(monkeypatch, ["contract", "lint", str(bad_path), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and any(i["code"] == "E002" for i in out["issues"])


def test_cli_show_prints_every_state(monkeypatch, capsys):
    assert _cli(monkeypatch, ["contract", "show"]) == 0
    out = capsys.readouterr().out
    for state in ("base", "window_45_days", "free_return_shipping", "reworded_exceptions_clause"):
        assert f"policy state: {state}" in out


def test_cli_run_gates_on_obligations(monkeypatch, capsys, tmp_path):
    app_path = tmp_path / "contract_test_app.py"
    app_path.write_text(textwrap.dedent("""
        from tests.test_contract import faithful, rigid
    """))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.syspath_prepend(str(tmp_path))

    import contradish.contract as contract_mod
    import contradish.llm as llm_mod
    monkeypatch.setattr(contract_mod, "default_outcome_classifier", lambda llm: label_classifier)
    monkeypatch.setattr(llm_mod, "LLMClient", lambda *a, **k: object())

    out_file = tmp_path / "result.json"
    assert _cli(monkeypatch, ["contract", "run", "--app", "contract_test_app:faithful",
                              "--workers", "1", "--output", str(out_file)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert json.loads(out_file.read_text())["passed"] is True

    assert _cli(monkeypatch, ["contract", "run", "--app", "contract_test_app:rigid",
                              "--workers", "1", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["passed"] is False and "observations" not in out
