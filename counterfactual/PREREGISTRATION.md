# Contradish Counterfactual Benchmark Suite: preregistration

Written 2026-10-01, before any model was run through the suite. Status of
each part is stated at the end. Nothing below is an empirical finding about
any AI model.

## The claim

Ordinary task accuracy and warranted behavioral updating are two distinct,
measurable properties of an AI system. If that is right, three things
should hold on independently built agent benchmarks:

1. A model that reliably completes a task under one policy still fails, at a
   rate beyond its ordinary noise, when the policy is amended in a way that
   warrants a different outcome.
2. How well a model updates predicts how reliable it is on tasks it was
   never amended on, after its task accuracy is accounted for.
3. Both hold on more than one benchmark.

The suite does not contribute new tasks. Each adapter takes an existing
benchmark, amends its governing policy, re-derives the expected outcome, and
scores with the benchmark's own deterministic scorer.

## Benchmarks

| Order | Benchmark | Status |
|---|---|---|
| 1 | STATE-Bench (Microsoft, MIT), customer_support | adapter built, ground truth derived and self-checked |
| 2 | AppWorld | not started |
| 3 | tau3-bench (Sierra) | not started |

STATE-Bench is the primary study. AppWorld and tau3-bench are replications:
the same hypotheses, the same analysis code, the same decision rules.

## Contradish x STATE-Bench v0.1

STATE-Bench commit `5644b1838d`. Domain: customer_support.

**Ground truth.** STATE-Bench's policy is executable (named constants its
environment computes with) and it ships gold trajectories. For an amendment
that changes one constant and the policy prose that states it, the gold tool
calls are replayed under the base and the amended policy. A task is

- `changed` if the gold path's final state differs: the amended final state
  is the new expected state;
- `invariant` if it is identical: the base expected state stands;
- `excluded`, with the reason recorded, if the replay cannot vouch for it.

Exclusion rules, all applied automatically:

- the gold trajectory does not reproduce STATE-Bench's own checked-in
  expected state under the base policy (18 of 100 trajectories);
- a tool call changes outcome class (success / rejection / error);
- the agent supplied a number the amendment makes stale and the replay has
  no rule to recompute;
- the amended expected state cannot be verified against the amended replay,
  or the base expected state still passes;
- **a non-updating agent would reach the amended state anyway** because the
  environment writes the amended value itself (this removed every candidate
  case in the shopping_assistant domain and the cancellation-fee amendment);
- the simulated user's script cites the old rule, or a changed amount as a
  bare number. (Amounts written as `$N` that map to one amended value are
  rewritten; those cases are tiered `rewritten`.)

**Amendments.** Six substantive, one control:

| Amendment | Change | changed | invariant | excluded |
|---|---|---:|---:|---:|
| cs_restocking_fee_20 | restocking fee 15% -> 20% | 11 | 70 | 1 |
| cs_shipping_clawback_12 | free-shipping clawback $8 -> $12 | 14 | 64 | 4 |
| cs_gold_restocking_discount_75 | Gold restocking discount 50% -> 75% | 4 | 77 | 1 |
| cs_bulk_clawback_8 | bulk clawback $5 -> $8 per item | 3 | 78 | 1 |
| cs_repeat_surcharge_10 | repeat-category surcharge $5 -> $10 | 2 | 79 | 1 |
| cs_return_shipping_fee_12 | return shipping fee $8 -> $12 | 1 | 77 | 4 |
| cs_control_reworded | two sentences reworded, nothing changed | 0 | 82 | 0 |

35 changed cases on 19 distinct tasks (31 `clean`, 4 `rewritten`), from 82
replay-verified tasks. All amendments are changes to an amount. None changes
eligibility, a deadline, or a procedure; that is a stated limit on what
"updating" means in v0.1.

**Self-check (done, no model involved).** A scripted oracle that replays the
gold path with the amended amounts passes every case. A scripted rigid agent
that replays it with the old amounts holds all invariant and control cases
and is scored as rigidity on all 35 changed cases.

## Run protocol

Per model, per run: all 82 base tasks; every changed case; 12 invariant
tasks per substantive amendment (seeded sample); the 19 changed tasks under
the control amendment. That is 208 episodes per run. Five runs per model.

- Agent, user simulator, and tools are STATE-Bench's, unmodified, apart from
  the amended policy constants and prose.
- Scoring is STATE-Bench's deterministic state scorer in every condition.
  The LLM-judged conversation requirements are not used, so accuracy and
  update fidelity are measured with the same instrument and no judge.
- Half of the tasks that are `changed` under no amendment are reserved, by a
  fixed hash, as criterion tasks and are never run under any amendment.

## Quantities

A (model, task) pair is **mastered** if the model passed the task on all
five base runs.

| Symbol | Definition |
|---|---|
| ACC | mean base pass rate over non-criterion tasks |
| CUF | P(amended state reached), mastered pairs, changed cases |
| HOLD | P(base state kept), mastered pairs, invariant cases |
| CTRL | P(base state kept), mastered pairs, control amendment |
| BUF | mean of min(1, CUF/CTRL) and min(1, HOLD/CTRL) |
| REL_FAIL | on criterion tasks: of tasks passed at least once in five runs, the share not passed all five times |

BUF is divided by CTRL because raw CUF and HOLD contain the model's ordinary
run-to-run failures and its sensitivity to any change in the policy text.
Without that step BUF would correlate with reliability by construction.

## Hypotheses and tests

**H1. Updating is a hurdle beyond accuracy.** On mastered pairs, the failure
rate under a substantive amendment exceeds the failure rate on the same
tasks under the control amendment.
Statistic: mean over tasks of (failure under amendment - failure under
control), averaged over models. Test: one-sided sign-flip permutation over
tasks, alpha = 0.05. Reported with a task-bootstrap 95% interval.

**H2. BUF explains reliability failures after controlling for accuracy.**
Across models, logit REL_FAIL is regressed on logit ACC and logit BUF with a
separate intercept per benchmark. Test: BUF's coefficient is negative,
one-sided, Freedman-Lane permutation within benchmark, alpha = 0.025 (the
reason for the stricter threshold is under "Study size").
Reported with the partial correlation and the gain in R^2 over the
accuracy-only model.

**Replication.** Each of H1 and H2 is tested on each benchmark separately
with the same code. The claim of generality requires H1 supported on every
benchmark and H2 supported on the pooled data with the same sign on each.

**Descriptive only.** Correlation of ACC with CUF across models, corrected
for split-half unreliability.

**Sensitivity analyses, fixed in advance.** (a) `clean` cases only.
(b) Mastery defined as a majority of base runs. (c) Each amendment left out
in turn.

## What would count against the claim

- H1 not supported: models that master a task update correctly as often as
  they survive a rewording. Then update fidelity is not a separate hurdle on
  this benchmark.
- H2 not supported with adequate power: update fidelity adds nothing to
  accuracy in explaining reliability failures.
- H1 supported and H2 not: updating is a distinct failure, but it is not
  what makes models unreliable elsewhere. That is a narrower result and
  would be reported as such.

## Study size (from simulation, not from data)

`contradish/counterfactual/simulate.py` generates synthetic agents with a
known accuracy trait and a known update trait and emits the same records a
real run does. Task counts were matched to v0.1. Every number below is in
`power_simulation_v0.1.json`.

**H1** (alpha 0.05, 400 simulated studies per cell): power 0.88 with 6
models, 0.95 with 8, 0.99 with 12, if the update hurdle is as large as the
simulation assumes. With no hurdle it rejects 2.2% of the time.

**H2 false-positive rate** (reliability failures generated from accuracy
alone; 1000 studies per cell):

| Models | Traits correlated | at p < 0.05 | at p < 0.025 |
|---:|---:|---:|---:|
| 8 | 0.5 | 4.2% | |
| 16 | 0.5 | 5.6% | |
| 24 | 0.5 | | 2.7% |
| 8 | 0.8 | 5.2% | |
| 16 | 0.8 | 8.4% | 3.5% |
| 24 | 0.8 | 7.7% (600 studies) | 6.1% |

At 0.05 the test is too liberal when accuracy and update fidelity are
strongly correlated. The cause is structural: accuracy is controlled for
with a measured score, and a measured score never removes all of the
underlying ability, so some of it leaks into BUF's coefficient. A quadratic
accuracy term did not fix it. **H2 is therefore tested at p < 0.025.** That
holds the simulated rate at or under 3.5% except in the most extreme cell
(24 models, traits correlated 0.8: 6.1%). The observed correlation between
ACC and BUF is reported with every H2 result so a reader can see which
regime the data are in.

**H2 power at p < 0.025** (moderate effect, traits correlated 0.5): 0.59
with 12 models, 0.74 with 16, 0.91 with 24. Plan for about 20 models. With
fewer than 12, a null result says little. Models can be pooled across
benchmarks; permutation is within benchmark.

These numbers depend on the simulation's assumed effect sizes. They size
the study; they do not predict its result.

## Status

| Part | Status |
|---|---|
| STATE-Bench adapter, ground truth, self-check | done |
| Analysis code, validated on synthetic agents | done |
| Live-run path through STATE-Bench's orchestrator | written, not exercised (no model credentials where this was built) |
| Any model run | not done |
| H1 / H2 on real models | not done |
| AppWorld replication | not started |
| tau3-bench replication | not started |
