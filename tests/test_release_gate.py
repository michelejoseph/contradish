"""The release gate: what must change, what must hold, and a verdict, against scripted agents of known behavior."""
import json

import pytest

from contradish.policy_program import load_builtin_program
from contradish.release_gate import (
    collateral_witness, demo_suite, export_examples, plan_cells, release_gate, threshold_witness, to_junit,
    to_markdown,
)
from contradish.symbolic import diff_versions
from contradish.versions import fixture_key, pin, sign_pin, version_witness


@pytest.fixture(scope="module")
def pair():
    p, U = load_builtin_program("returns")
    ok, _ = p.split_edits(U["restocking_fee_20"])
    k = fixture_key("policy_owner")
    v1 = sign_pin(pin(p, "v1", "fixture", "test"), k)
    v2 = sign_pin(pin(p.apply(ok), "v2", "fixture", "test", issued_by="policy_owner", supersedes=v1.digest), k)
    return p, v1, v2, [k["public"]]


def gate(pair, agent, **kw):
    p, v1, v2, anchors = pair
    return release_gate(v1, v2, agent, getattr(agent, "__name__", "a"), trust_anchors=anchors, **kw)[0]


def test_faithful_passes(pair):
    r = gate(pair, version_witness("faithful"))
    assert r["verdict"] == "PASS" and r["exit_code"] == 0 and not r["reasons"]
    assert {c["step"] for c in r["what_must_change"]} == {"refund", "ledger", "notify"}
    assert "label" in r["what_must_hold"]["steps_proved_unchanged_everywhere"]


def test_failure_to_change_and_unwarranted_change_are_told_apart(pair):
    rigid = gate(pair, version_witness("rigid"))
    assert rigid["verdict"] == "BLOCK" and rigid["findings"]["failed_to_change"]
    assert not rigid["findings"]["unwarranted_change"]
    col = gate(pair, collateral_witness("label"))
    assert col["verdict"] == "BLOCK" and not col["findings"]["failed_to_change"]
    assert [x["step"] for x in col["findings"]["unwarranted_change"]] == ["label"]


def test_high_risk_steps_downgrade_other_failures_to_review(pair):
    r = gate(pair, collateral_witness("label"), policy={"high_risk_steps": ["refund"]})
    assert r["verdict"] == "REVIEW" and r["exit_code"] == 2


def test_classes_plan_is_cheaper_and_still_catches_the_collateral_break(pair):
    p, v1, v2, _ = pair
    d = diff_versions(v1.program, v2.program)
    assert len(plan_cells(d, "classes")) < len(plan_cells(d, "complete")) / 5
    r = gate(pair, collateral_witness("label"), policy={"plan": "classes"})
    assert r["verdict"] == "BLOCK" and r["plan"]["is_proof"] is False


def test_audit_measures_snapshot_regression(pair):
    p = pair[0]
    suite = demo_suite(p, 120)
    faithful = gate(pair, version_witness("faithful"), suite=suite, audit=True)
    sn = faithful["audit"]["snapshot_regression"]
    assert sn["cases_flagged"] == sn["flagged_but_intended"] > 0        # every snapshot flag is a false alarm
    rigid = gate(pair, version_witness("rigid"), suite=suite, audit=True)
    assert rigid["audit"]["snapshot_regression"]["cases_flagged"] == 0  # the snapshot sees nothing...
    assert rigid["audit"]["snapshot_regression"]["real_failures_passed_silently"] > 0   # ...and passes a failure
    assert rigid["verdict"] == "BLOCK"


def test_audit_reports_what_a_cheap_plan_missed(pair):
    suite = demo_suite(pair[0], 300)
    r = gate(pair, threshold_witness(), suite=suite, audit=True, policy={"plan": "classes"})
    assert r["audit"]["plan_missed_failures"] and r["verdict"] == "BLOCK"
    assert any("missed" in x for x in r["reasons"])
    full = gate(pair, threshold_witness(), suite=suite, audit=True, policy={"plan": "suite"})
    assert not full["audit"]["plan_missed_failures"]


def test_suite_coverage_gaps(pair):
    r = gate(pair, version_witness("faithful"), suite=demo_suite(pair[0], 100))
    s = r["suite"]
    assert s["regions_exercised"] <= s["regions_total"] and 0 <= s["change_regions_never_exercised"] <= 21


def test_outputs(pair):
    r = gate(pair, collateral_witness("label"))
    md, xml = to_markdown(r), to_junit(r)
    assert "BLOCK" in md and "Changed without warrant" in md
    assert 'failures="1"' in xml and 'name="label"' in xml
    json.dumps(r, default=str)


def test_export_examples(pair):
    p, v1, v2, _ = pair
    ex = export_examples(v2, diff_versions(v1.program, v2.program))
    assert len(ex) == len(diff_versions(v1.program, v2.program).cells)
    assert {"inputs", "outputs", "metadata"} <= set(ex[0])
