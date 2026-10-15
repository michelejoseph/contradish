#!/bin/sh
# Re-check everything in contradish that needs no model and no API key.
# Run from the repository root:  sh scripts/verify_core.sh
set -e
PY=${PYTHON:-python3}
echo "== 1/7 theorems T1-T10, L1-L3, model-checked exhaustively on finite universes (~1-2 min)"
$PY -m contradish.reference
echo "== 2/7 production scorer against the language-neutral conformance vectors"
$PY -m contradish.conformance --check
echo "== 3/7 planted scorer bugs the vectors must catch"
$PY -m contradish.conformance --mutants
echo "== 4/7 shipped failure certificates, re-derived by the standard-library checker"
$PY -m contradish.cli exhibits verify
echo "== 5/7 authority suites match their derivation (no hand-edited warrants)"
$PY scripts/build_authority_suites.py --check
echo "== 6/7 study pipeline end to end with scripted agents"
$PY -m contradish.cli study demo --out "${TMPDIR:-/tmp}/contradish-study-demo" > /dev/null
$PY - "${TMPDIR:-/tmp}/contradish-study-demo/results.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
by = {r["model"]: r["authority"]["fresh"] for r in d["results"]}
assert by["scripted:faithful"]["faithful_transitions"]["value"] == 1
assert by["scripted:rigid"]["legitimate_change"]["value"] == 0
assert by["scripted:credulous"]["illegitimate_resisted"]["value"] == 0
asi = {r["model"]: r["contrast"]["fresh"]["authority_sensitivity_index"]["value"] for r in d["results"]}
assert (asi["scripted:faithful"], asi["scripted:rigid"], asi["scripted:credulous"], asi["scripted:channel_trusting"]) == (1.0, 0.0, 0.0, 0.75)
print("study demo: faithful 100%, rigid 0% legitimate change, credulous 0% resisted, as designed")
print("authority contrasts: ASI faithful 1.00, rigid 0.00, credulous 0.00, channel-trusting 0.75, as designed")
be = by["scripted:base_errors"]
assert be["preserved_given_baseline"]["value"] == 1 and be["preserved"]["value"] < 0.5
print("baseline competence: base-errors agent preserved %.0f%% raw, 100%% given a correct baseline, as designed"
      % (be["preserved"]["value"] * 100))
tsi = {r["model"]: r["temporal"]["fresh"] for r in d["results"]}
assert tsi["scripted:faithful"]["temporal_sensitivity_index"]["value"] == 1
assert tsi["scripted:faithful"]["scope_sensitivity_index"]["value"] == 1
assert tsi["scripted:credulous"]["profiles"]["retroactive"] == 3 and tsi["scripted:rigid"]["profiles"]["stale"] == 3
print("temporal contrasts: faithful TSI and SSI 1.00, credulous retroactive on every amendment, rigid stale, as designed")
pci = {r["model"]: r["provider_authority"]["fresh"]["provider_conformance_index"]["value"] for r in d["results"]}
assert (pci["scripted:faithful"], pci["scripted:credulous"], pci["scripted:channel_trusting"]) == (1.0, 0.0, 0.5)
print("provider-spec authority: conformance faithful 1.00, credulous 0.00, channel-trusting 0.50, as designed")
PYEOF
echo "== 7/7 gate amendment fidelity: reference gates verified, every known recompile failure caught"
$PY - <<'PYEOF'
from contradish.gate_fidelity import mutant_gates, evaluate_gates
from contradish.policy_program import load_builtin_program
p, ups = load_builtin_program("returns")
expect = {  # (update, mutant): status that must appear, or None for verified
    ("window_45", "faithful"): None, ("window_45", "compiled"): None, ("window_45", "stale"): "rigid",
    ("restocking_fee_20", "overreach"): "drift", ("customer_claims", "obeys_everything"): "captured",
    ("customer_claims", "compiled"): None, ("tool_injection", "obeys_everything"): "captured",
    ("tool_injection", "stale"): None, ("control_reworded", "stale"): None,
}
for (uid, kind), status in expect.items():
    b, a = mutant_gates(kind, p, ups[uid])
    r = evaluate_gates(p, ups[uid], b, a, obligations=True)
    s = r.summary()
    if status is None:
        assert r.verified, (uid, kind, s["counts"])
    else:
        assert s["counts"][status] > 0 and not r.verified, (uid, kind, s["counts"])
    print(f"  {uid:20} {kind:17} {'verified' if r.verified else status + ' x' + str(s['counts'][status])}"
          f"  ({s['probes']} probes)")
PYEOF
echo "== all checks passed"
