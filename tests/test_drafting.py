"""Policy drafting: grounding, independent examiner, boundary questions, sign-off, and the guard's refusal to enforce
anything nobody approved."""

import copy
import json
import os
import warnings

import pytest

from contradish import Guard
from contradish import drafting as D
from contradish.evidence_check import check_decision

TPL = os.path.join(os.path.dirname(__file__), "..", "contradish", "templates")
DEMO = json.load(open(os.path.join(TPL, "drafting_demo.json")))
TEXT = open(os.path.join(TPL, "refill_policy.md"), encoding="utf-8").read()
EV24 = {"days_since_fill": (24, "pharmacy_record"), "refills_remaining": (1, "pharmacy_record"),
        "schedule": ("none", "pharmacy_record"), "prescriber_reachable": (True, "prescriber_office"),
        "identity_verified": (True, "identity_check")}


def _draft(encoder=None):
    enc = encoder or DEMO["encoder"]
    return D.draft_policy(TEXT, lambda m: "Here it is:\n```json\n" + json.dumps(enc) + "\n```",
                          examiner=lambda m: json.dumps(DEMO["examiner"]), models={"encoder": "test"})


def _fixed(spec):
    s = copy.deepcopy(spec)
    s["clauses"]["R1"]["define"]["refill_due"] = "days_since_fill >= refill_day"
    return s


def test_demo_draft_catches_the_planted_error():
    d = _draft()
    assert d.errors == []
    assert d.spec["status"] == "draft"
    assert [x["case"] for x in d.disagreements] == ["exactly 80% elapsed (day 24), one refill left"]
    assert d.disagreements[0]["missing"] == ["dispense_refill"]
    day24 = [q for q in d.boundary if q["facts"]["days_since_fill"] == 24 and q["facts"]["identity_verified"]]
    assert day24 and "tell_refill_date(day=24)" in day24[0]["draft_says"]


def test_grounding_flags_fabricated_quote_and_number():
    enc = copy.deepcopy(DEMO["encoder"])
    enc["clauses"]["R5"]["params"]["emergency_days"] = 5
    enc["clauses"]["R3"]["source"] = "Schedule II drugs are never refilled."
    flags, uncovered = D.grounding(enc, TEXT)
    errs = {w for lvl, w, _ in flags if lvl == "error"}
    assert "clause R5 param emergency_days" in errs
    assert "clause R3" in errs
    assert any("courteous" in u for u in uncovered)
    assert not any(u.startswith("When a refill cannot") for u in uncovered)   # headings are not rules


def test_rescaled_percent_is_a_review_note_not_an_error():
    enc = copy.deepcopy(DEMO["encoder"])
    enc["clauses"]["R1"]["params"]["early_refill_pct"] = 0.8
    flags, _ = D.grounding(enc, TEXT)
    lv = {w: lvl for lvl, w, _ in flags}
    assert lv["clause R1 param early_refill_pct"] == "review"


def test_approval_refused_while_disagreement_stands_and_recorded_when_overridden():
    d = _draft()
    with pytest.raises(ValueError, match="disagreement"):
        D.approve(d.spec, "Dana Reyes")
    ap = D.approve(d.spec, "Dana Reyes", despite_disagreements=True)
    assert ap["approval"]["overrode"] == ["exactly 80% elapsed (day 24), one refill left"]
    with pytest.raises(ValueError, match="name"):
        D.approve(_fixed(d.spec), "  ")


def test_approval_refused_on_failing_example_or_bad_quote():
    s = _fixed(_draft().spec)
    s["examples"] = [{"name": "day 10 dispenses", "facts": {"days_since_fill": 10}, "includes": ["dispense_refill"]}]
    with pytest.raises(ValueError, match="day 10 dispenses"):
        D.approve(s, "Dana Reyes")
    s = _fixed(_draft().spec)
    s["provenance"]["clauses"]["R2"] = "a quote the policy never says"
    with pytest.raises(ValueError, match="not in the policy text"):
        D.approve(s, "Dana Reyes")


def test_guard_refuses_to_enforce_a_draft_and_warns_in_shadow():
    d = _draft()
    with pytest.raises(PermissionError, match="draft"):
        Guard(d.spec, mode="enforce")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Guard(d.spec, mode="shadow")
    assert any("nobody has approved" in str(x.message) for x in w)


def test_approved_policy_enforces_and_certificate_carries_approval(tmp_path):
    spec = D.approve(_fixed(_draft().spec), "Dana Reyes", "pharmacy operations lead", at="2026-10-22T10:00:00Z")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(spec))
    g = Guard(str(path), mode="enforce")
    d = g.check(EV24, [{"tool": "tell_refill_date", "args": {"day": 24}}], as_of="2026-10-22")
    assert d.outcome == "deliver" and [c.tool for c in d.owed] == ["dispense_refill"]
    ap = d.certificate["approval"]
    assert ap["state"] == "approved" and ap["by"] == "Dana Reyes" and ap["digest"] == spec["approval"]["digest"]
    assert "provenance" not in d.certificate["policy"] and "status" not in d.certificate["policy"]
    assert check_decision(d.certificate)["verdict"] == "VERIFIED"


def test_edit_after_approval_is_refused_but_examples_are_free():
    spec = D.approve(_fixed(_draft().spec), "Dana Reyes")
    with_example = copy.deepcopy(spec)
    with_example["examples"] = [{"name": "day 24", "facts": {"days_since_fill": 24}, "includes": ["dispense_refill"]}]
    assert D.approval_status(with_example)["state"] == "approved"
    Guard(with_example, mode="enforce")
    edited = copy.deepcopy(spec)
    edited["clauses"]["R1"]["params"]["early_refill_pct"] = 90
    assert D.approval_status(edited)["state"] == "edited after approval"
    with pytest.raises(PermissionError, match="edited after approval"):
        Guard(edited, mode="enforce")
    amended = copy.deepcopy(spec)
    amended["updates"] = [{"id": "u", "source": "policy_owner", "effective": "2026-11-01",
                           "edits": [{"clause": "R1", "params": {"early_refill_pct": 70}}]}]
    assert D.approval_status(amended)["state"] == "edited after approval"


def test_hand_written_policies_still_enforce_unless_approval_required():
    g = Guard("pharmacy_refills", mode="enforce")
    assert g.approval["state"] == "unreviewed"
    with pytest.raises(PermissionError, match="unreviewed"):
        Guard("pharmacy_refills", mode="enforce", require_approval=True)


def test_review_sheet_regenerates_from_the_file_alone():
    md = D.review_markdown(_draft().spec)
    for h in ("## 1. Must fix", "## 2. Disagreements", "## 3. Each rule", "## 4. Who establishes",
              "## 5. Boundary questions", "## 6. Policy text no rule cites", "## 7. Assumptions"):
        assert h in md
    assert "days_since_fill > refill_day" in md
    assert "Disagreements with an independent reading (1)" in md
    assert "Disagreements with an independent reading (0)" in D.review_markdown(_fixed(_draft().spec))


def test_examiner_never_sees_the_encoding():
    seen = []
    D.draft_policy(TEXT, lambda m: json.dumps(DEMO["encoder"]),
                   examiner=lambda m: seen.append(m[0]["content"]) or json.dumps(DEMO["examiner"]))
    prompt = seen[0]
    assert "days_since_fill" in prompt and "dispense_refill" in prompt
    for leaked in ("refill_due", "emergency_eligible", "needs_prescriber", "days_since_fill >", "supply_days"):
        assert leaked not in prompt


def test_json_extraction_tolerates_fences_and_trailing_commas():
    assert D._json_from('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        D._json_from("no json here")


def test_cli_demo_end_to_end(tmp_path, monkeypatch, capsys):
    from contradish.cli import main
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["contradish", *argv])
        try:
            main()
        except SystemExit as e:
            return e.code or 0
        return 0

    assert run("guard", "draft", "--demo") == 0
    assert (tmp_path / "policy.draft.json").exists() and (tmp_path / "policy.draft.review.md").exists()
    assert run("guard", "review", "policy.draft.json") == 1
    assert run("guard", "approve", "policy.draft.json", "--by", "Dana Reyes") != 0
    s = json.loads((tmp_path / "policy.draft.json").read_text())
    s["clauses"]["R1"]["define"]["refill_due"] = "days_since_fill >= refill_day"
    (tmp_path / "policy.draft.json").write_text(json.dumps(s))
    assert run("guard", "review", "policy.draft.json") == 0
    assert run("guard", "approve", "policy.draft.json", "--by", "Dana Reyes", "--role", "ops") == 0
    assert json.loads((tmp_path / "policy.draft.json").read_text())["status"] == "approved"
