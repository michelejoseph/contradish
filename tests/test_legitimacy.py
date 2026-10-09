"""Legitimacy of changes to the governing specification: provenance, procedure, authority by effect, invariants."""

import copy
import json

import pytest

from contradish.charter import (
    DEMO_CASES, approve_pin, check_invariants, demo_transition, derive_effects, legitimacy_certificate,
    validate_charter,
)
from contradish.evidence import seal
from contradish.evidence_check import check_legitimacy, check_run
from contradish.policy_program import PolicyProgram, load_builtin_program
from contradish.versions import (
    VerificationScope, fixture_key, pin, run_certificate, sign_pin, version_witness,
)

EXPECTED = {
    "restocking_fee_20": {"provenance": True, "procedure": True, "authority_by_effect": True,
                          "invariants": True, "charter_amendment": True},
    "pricing_prepaid_cut": {"provenance": True, "procedure": False, "authority_by_effect": False,
                            "invariants": False, "charter_amendment": True},
    "window_45_unapproved": {"provenance": True, "procedure": False, "authority_by_effect": True,
                             "invariants": True, "charter_amendment": True},
    "window_45": {"provenance": True, "procedure": True, "authority_by_effect": True,
                  "invariants": True, "charter_amendment": True},
    "defective_restocking": {"provenance": True, "procedure": True, "authority_by_effect": True,
                             "invariants": False, "charter_amendment": True},
    "charter_amendment": {"provenance": True, "procedure": True, "authority_by_effect": True,
                          "invariants": True, "charter_amendment": False},
}


@pytest.fixture(scope="module")
def certs():
    out = {}
    for case in DEMO_CASES:
        v1, v2, anchors, _ = demo_transition(case)
        out[case] = legitimacy_certificate(v1, v2, anchors)
    return out


def test_builtin_charter_is_well_formed_and_holds_on_v1():
    p, _ = load_builtin_program()
    assert validate_charter(p) == []
    v1, v2, anchors, _ = demo_transition("restocking_fee_20")
    c = legitimacy_certificate(v1, v2, anchors)
    assert all(c["baseline_invariants_hold"].values())


@pytest.mark.parametrize("case", list(EXPECTED))
def test_demo_verdicts_and_independent_check(certs, case):
    c = certs[case]
    assert c["checks"] == EXPECTED[case]
    assert c["legitimate"] == all(EXPECTED[case].values())
    v = check_legitimacy(json.loads(json.dumps(c)))
    assert v["verdict"] == "VERIFIED", v["problems"]


def test_effect_authority_sees_what_clause_authority_cannot(certs):
    c = certs["pricing_prepaid_cut"]
    assert c["clause_level_authority"]["status"] == "authorized"       # it owns clause R5
    outside = {(x["step"], tuple(x["args"])) for x in c["effects"]["unauthorized"]}
    assert ("label", ("prepaid",)) in outside and ("refund", ("amount",)) in outside
    assert "clause-level authority would pass" in c["statement"]


def test_invariant_witness_is_a_real_violation(certs):
    c = certs["defective_restocking"]
    inv = next(i for i in c["invariants"] if i["id"] == "INV-defective-full")
    sit = inv["violations"][0]["situation"]
    v1, v2, _, _ = demo_transition("defective_restocking")
    call = v2.program.run_norm("refund", v2.program.complete_state(sit))[0].call
    assert dict(call.args)["amount"] < sit["price"] - 0.005


def test_bound_invariants_are_exact_at_the_grid_edge():
    """A violation that exists only at the single largest price is found, not missed."""
    p, _ = load_builtin_program()
    spec = p.to_dict()
    spec["clauses"]["R6"]["define"]["refund_amount"] = [
        "if", [">", "price", 4999.98], ["+", "price", 0.01],
        ["max", 0, ["-", "price", "fee_after_discount", "shipping_charge"]]]
    q = PolicyProgram({k: v for k, v in spec.items() if k != "updates"})
    ch = copy.deepcopy(spec["charter"])
    eff = derive_effects(p, q, ch)
    inv = {i["id"]: i for i in check_invariants(q, ch, eff["cells"])}
    assert not inv["INV-refund-cap"]["holds"]
    assert inv["INV-refund-cap"]["violations"][0]["situation"]["price"] == 4999.99
    assert inv["INV-final-sale"]["holds"]


def test_charter_amendment_rules():
    p, _ = load_builtin_program()
    keys = {r: fixture_key(r) for r in ("policy_owner", "legal", "consumer_protection")}
    v1 = sign_pin(pin(p, "v1", "t", "f", effective_at="2026-10-01"), keys["policy_owner"])
    spec = p.to_dict()
    spec.pop("updates", None)
    spec["charter"]["invariants"] = [i for i in spec["charter"]["invariants"] if i["id"] != "INV-notice"]
    q = PolicyProgram(spec)
    v2 = pin(q, "v2", "t", "f", issued_by="policy_owner", supersedes=v1.digest,
             proposed_at="2026-10-01", effective_at="2026-11-05")
    sign_pin(v2, keys["policy_owner"])
    c = legitimacy_certificate(v1, v2, [keys["policy_owner"]["public"]])
    assert not c["checks"]["charter_amendment"]                          # needs all three roles
    approve_pin(v2, keys["legal"], "legal")
    approve_pin(v2, keys["consumer_protection"], "consumer_protection")
    c = legitimacy_certificate(v1, v2, [keys["policy_owner"]["public"]])
    assert c["checks"]["charter_amendment"] and c["legitimate"]          # non-entrenched, full procedure
    assert check_legitimacy(c)["verdict"] == "VERIFIED"


def test_tampering_is_rejected(certs):
    c = certs["pricing_prepaid_cut"]
    bad = copy.deepcopy(c)
    bad["legitimate"] = True
    assert check_legitimacy(seal(bad))["verdict"] == "REJECTED"
    bad = copy.deepcopy(c)
    bad["effects"]["unauthorized"] = bad["effects"]["unauthorized"][1:]  # hide an out-of-grant effect
    assert check_legitimacy(seal(bad))["verdict"] == "REJECTED"
    bad = copy.deepcopy(c)
    for i in bad["invariants"]:
        i["holds"] = True
    assert check_legitimacy(seal(bad))["verdict"] == "REJECTED"
    bad = copy.deepcopy(certs["window_45"])
    bad["after"]["effective_at"] = "2026-10-03"                          # shorten the review after signing
    assert check_legitimacy(seal(bad))["verdict"] == "REJECTED"
    bad = copy.deepcopy(certs["window_45"])
    bad["after"]["approvals"][0]["key"] = fixture_key("stranger")["public"]
    assert check_legitimacy(seal(bad))["verdict"] == "REJECTED"


def test_run_certificate_chains_legitimacy(certs):
    v1, v2, anchors, _ = demo_transition("pricing_prepaid_cut")
    scope = VerificationScope(situations={"price": {"min": 0, "max": 50}}, actions=["label"])
    c = run_certificate(v1, v2, version_witness("faithful"), "witness:faithful", scope=scope,
                        trust_anchors=anchors, legitimacy=certs["pricing_prepaid_cut"])
    assert c["claim"]["proved"] and c["claim"]["legitimate"] is False and not c["claim"]["verified"]
    assert check_run(c)["verdict"] == "VERIFIED"
    with pytest.raises(Exception):
        run_certificate(v1, v2, version_witness("faithful"), "witness:faithful", scope=scope,
                        trust_anchors=anchors, legitimacy=certs["window_45"])
