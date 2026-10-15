"""Gate amendment fidelity: compiled gates scored exhaustively against the warranted change."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from contradish import gate_fidelity as G
from contradish.policy_program import Edit, ProgramUpdate, load_builtin_program

ROOT = Path(__file__).resolve().parent.parent
PROGRAM, UPDATES = load_builtin_program("returns")


def _run(update_id, kind, obligations=False):
    u = UPDATES[update_id]
    b, a = G.mutant_gates(kind, PROGRAM, u)
    return G.evaluate_gates(PROGRAM, u, b, a, obligations=obligations)


def test_reference_compiler_agrees_with_the_program_everywhere():
    gate = G.RuleGate(G.compile_rules(PROGRAM), {k: v.get("default") for k, v in PROGRAM.facts.items()})
    ref = G.ProgramGate(PROGRAM)
    f = G.derive_action_frontier(PROGRAM, UPDATES["window_45"])
    for p in G.gate_contract(f, obligations=True)[1]:
        s = f.situations[p.situation]
        if p.kind == "permission":
            assert gate.decide(s, p.call) == ref.decide(s, p.call), p.id
    for s in f.situations[:40]:
        assert sorted(json.dumps(c, sort_keys=True) for c in gate.required(s)) == \
            sorted(json.dumps(c, sort_keys=True) for c in ref.required(s))


@pytest.mark.parametrize("kind", ["faithful", "compiled"])
def test_correct_recompiles_are_verified(kind):
    r = _run("window_45", kind, obligations=True)
    assert r.verified
    s = r.summary()
    assert s["change"] == 1.0 and s["preservation"] == 1.0 and s["old_gate_accuracy"] == 1.0
    assert s["must_change"] > 0


def test_stale_gate_is_rigid_on_exactly_the_warranted_changes():
    r = _run("window_45", "stale")
    s = r.summary()
    assert s["counts"]["rigid"] == s["must_change"]
    assert s["counts"]["drift"] == 0 and not r.verified
    assert "rigid" in r.counterexamples()


def test_gate_obeying_a_customer_claim_is_captured():
    r = _run("customer_claims", "obeys_everything")
    s = r.summary()
    assert s["counts"]["captured"] == s["must_resist"] > 0
    assert s["authority_respected"] == 0.0


def test_injected_tool_text_does_not_count_as_an_amendment():
    assert _run("tool_injection", "stale").verified          # nothing had authority to change
    r = _run("tool_injection", "obeys_everything")
    assert r.summary()["counts"]["captured"] > 0


def test_overreach_is_drift():
    r = _run("restocking_fee_20", "overreach")
    assert r.summary()["counts"]["drift"] > 0 and r.summary()["counts"]["rigid"] == 0


def test_null_amendment_needs_no_change():
    r = _run("control_reworded", "stale")
    assert r.verified and r.summary()["must_change"] == 0


def test_old_gate_errors_are_baseline_errors_not_rigidity():
    u = UPDATES["window_45"]
    broken_old = G.FunctionGate(lambda s, c: "block" if c["tool"] == "issue_refund" else
                                G.ProgramGate(PROGRAM).decide(s, c), name="broken-old")
    _, good_new = G.mutant_gates("faithful", PROGRAM, u)
    r = G.evaluate_gates(PROGRAM, u, broken_old, good_new)
    s = r.summary()
    assert s["old_gate_accuracy"] < 1.0
    assert s["counts"]["baseline_error"] > 0
    assert s["change"] == 1.0 and s["preservation"] == 1.0   # the new gate itself is right
    assert not r.verified                                      # but the pair is not verified


def _naive_compiler(text, tools, facts):
    """Takes the LAST number of days mentioned anywhere, whoever wrote it."""
    days = re.findall(r"(\d+) days", text)
    window = int(days[-1]) if days else 30
    p = PROGRAM.apply([Edit("R1", params={"return_window_days": window})])
    return G.compile_rules(p)


def test_compiler_run_on_appended_text():
    ok = G.run_compiler(_naive_compiler, PROGRAM, UPDATES["window_45"])
    assert ok.verified                          # the owner's 45 days: applied correctly
    bad = G.run_compiler(_naive_compiler, PROGRAM, UPDATES["customer_claims"])
    assert bad.summary()["counts"]["captured"] > 0   # the customer's "60 days" got compiled in
    assert "Update received from customer via user" in G.after_text(PROGRAM, UPDATES["customer_claims"])


def test_rule_gate_rejects_unknown_names_and_bad_rules():
    with pytest.raises(ValueError):
        G.RuleGate({"rules": [{"tool": "x"}]})
    g = G.RuleGate({"rules": [{"tool": "x", "allow_when": ["<", "nope", 3]}]})
    with pytest.raises(KeyError):
        g.decide({}, {"tool": "x", "args": {}})


def test_cli_check_exit_codes(tmp_path):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps(G.compile_rules(PROGRAM)))
    new.write_text(json.dumps(G.compile_rules(PROGRAM.apply(PROGRAM.split_edits(UPDATES["window_45"])[0]))))
    ok = subprocess.run([sys.executable, "-m", "contradish.cli", "gate", "check", "--update", "window_45",
                         "--before", str(old), "--after", str(new)], capture_output=True, text=True, cwd=ROOT)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "VERIFIED" in ok.stdout
    bad = subprocess.run([sys.executable, "-m", "contradish.cli", "gate", "check", "--update", "window_45",
                          "--before", str(old), "--after", str(old)], capture_output=True, text=True, cwd=ROOT)
    assert bad.returncode == 1 and "NOT VERIFIED" in bad.stdout
