# The release gate (`contradish release`)

**Before an AI agent is released against a changed policy, contradish identifies what must change and what must stay correct, and decides whether the release is safe.**

It produces three outputs:
- **What must change.** The complete set of regions where the new version requires, forbids, grants or revokes something different. These are derived from the two pinned policy versions and proved complete. Nobody writes expected outputs by hand.
- **What must hold.** Every other region and every other action, including customer protections the change never mentions.
- **A verdict.** PASS, REVIEW or BLOCK under a declared release policy (exit codes 0, 2, 1). It comes with a sealed certificate that anyone can re-check with `contradish evidence check`.

The gate reports the two possible failures separately, because they need different fixes:

| Failure | Example | What a snapshot regression does |
|---|---|---|
| **Failed to change** | the new restocking fee is not applied | **passes it**: the output did not change |
| **Changed without warrant** | the fee is applied, and the prepaid return label for defective items silently stops being sent | flags it, mixed in with every intended change |

## Why a regression suite of recorded outputs is not enough

A snapshot regression compares the new release's outputs with the previous release's recorded outputs. That is the usual evaluation in CI. After a policy change it has no way to know which differences were intended:

1. It flags every intended change as a failure. Someone must re-label the references by hand before the suite is useful again.
2. It mixes those false alarms with the real breaks, so a reviewer has to triage both.
3. It passes a required change that did not happen, because nothing changed.

The gate derives the expected behavior of the **new** version from the policy diff, so all three disappear.

## Measured, not assumed

`--suite cases.jsonl --audit` runs your own suite both ways and measures three things:
- what a snapshot regression would have flagged, falsely alarmed and silently passed;
- the regions your suite never reaches, which are often exactly where behavior must change;
- whether running the whole suite found any failure the gate's cheaper plan missed.

**First measurement.** These come from scripted agents of known behavior, not models, so they show what the measurement detects, not how often real agents fail. The setup:
- the built-in returns policy, with the update raising the restocking fee from 15% to 20%;
- 445 regions, 21 of which must change;
- a synthetic 600-case suite skewed the way production traffic is (`release_gate.demo_suite`).

| Agent | Gate verdict | Snapshot regression on the 600 cases |
|---|---|---|
| implements the change | **PASS** | flags 25 cases, **all 25 false alarms** |
| ignores the change | **BLOCK**: failed to change (refund, ledger, notify) | flags 0, **passes all 25 cases that should have changed** |
| implements the change, stops sending return labels | **BLOCK**: changed without warrant (label) | flags 513 cases; the intended changes are mixed into the same flags |
| implements the change, wrong refund for prices ≥ $500 | **BLOCK** | flags 120 cases: 21 false alarms and 99 real breaks, mixed together |

**The suite's own coverage.** The 600 cases reach 47 of the 445 regions, and only **3 of the 21 regions where behavior must change**. Most of the change is invisible to the suite.

**The three plans** (agent calls for this transition):

| Plan | Calls | Proof? | Missed anything the full suite found? |
|---|---:|---|---|
| `complete`: every region | 890 | yes, it issues a certificate | no, for all four agents |
| `classes`: every region that must change, plus one per behavior class elsewhere | **62** | no | **yes, once**: the price ≥ $500 refund bug in regions that must hold. The audit caught it and the gate blocked. |
| `suite`: your cases, plus one probe in each region they never reach | 1,996 | no | no |

The `classes` plan is 14× cheaper than `complete` and caught three of the four defects. The fourth sits at a boundary no policy clause has, so no region is split there, and probing one situation per region cannot reach it. That is the limit the certificate states ("behavior between tested situations is not proved"), and the audit is how the limit is measured on a real agent instead of assumed away. For a model, run `classes` on every pull request and `complete` or `suite` with `--audit` before a release.

**Real models: not yet measured.** `scripts/release_gate_experiment.py --models qwen2.5:7b,...` runs the same pipeline on local models through Ollama. Until it has been run, the numbers above describe the gate, not the defects real agents have.

## Using it

```bash
contradish release demo --agent collateral            # the scenario above, offline
contradish release check v1.pin.json v2.pin.json --app agent.adapter:chat \
    --suite tests/regression_cases.jsonl --audit --policy release_policy.json \
    --markdown gate.md --junit gate.xml --json gate.json --certificate run.json
contradish evidence check run.json                    # independent re-derivation
contradish release export v1.pin.json v2.pin.json --out examples.jsonl
```

The **release policy** (`examples/ci/release_policy.json`) has these fields:
- `block_on` and `review_on`: which findings lead to BLOCK and which to REVIEW;
- `high_risk_steps`: failures in other steps go to REVIEW instead of BLOCK;
- `trials` and `plan`.

**CI:** `examples/ci/contradish-release.yml` runs the gate on every pull request that touches the policy or the agent. It posts the summary on the pull request, uploads the evidence, and fails on BLOCK.

## Next to existing evaluation platforms

Platforms such as LangSmith already run regression evaluations in CI and can fail a deployment check. Contradish does not replace the runner. It supplies what such a runner cannot infer on its own: which behavior a policy change requires and which it forbids, derived and proved complete, with authority and protected obligations checked. `contradish release export` writes those expectations as `{inputs, outputs, metadata}` examples, one per region with its required norms, so the same expectations can run in an existing evaluation pipeline. Check your platform's import format.

## Limits

- Agents are probed at representative situations. Behavior between probes is proved only for a policy program, not for a model. The audit measures what that misses on your suite.
- The verdict is only as good as the two pinned versions. Contradish checks that a change is authorized and complete, not that the new policy is wise.
- The measurements above are on scripted agents and one built-in policy.
