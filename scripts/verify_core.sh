#!/bin/sh
# Re-check everything in contradish that needs no model and no API key.
# Run from the repository root:  sh scripts/verify_core.sh
set -e
PY=${PYTHON:-python3}
echo "== 1/6 theorems T1-T10, L1-L3, model-checked exhaustively on finite universes (~1-2 min)"
$PY -m contradish.reference
echo "== 2/6 production scorer against the language-neutral conformance vectors"
$PY -m contradish.conformance --check
echo "== 3/6 planted scorer bugs the vectors must catch"
$PY -m contradish.conformance --mutants
echo "== 4/6 shipped failure certificates, re-derived by the standard-library checker"
$PY -m contradish.cli exhibits verify
echo "== 5/6 authority suites match their derivation (no hand-edited warrants)"
$PY scripts/build_authority_suites.py --check
echo "== 6/6 study pipeline end to end with scripted agents"
$PY -m contradish.cli study demo --out "${TMPDIR:-/tmp}/contradish-study-demo" > /dev/null
$PY - "${TMPDIR:-/tmp}/contradish-study-demo/results.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
by = {r["model"]: r["authority"]["fresh"] for r in d["results"]}
assert by["scripted:faithful"]["faithful_transitions"]["value"] == 1
assert by["scripted:rigid"]["legitimate_change"]["value"] == 0
assert by["scripted:credulous"]["illegitimate_resisted"]["value"] == 0
print("study demo: faithful 100%, rigid 0% legitimate change, credulous 0% resisted, as designed")
PYEOF
echo "== all checks passed"
