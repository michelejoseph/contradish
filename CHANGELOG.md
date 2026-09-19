# Changelog

All notable changes to contradish are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this file starts at 1.29.0;
earlier releases were not retroactively documented.

## [1.50.0] - 2026-09-19

### Added

- **`contradish/intervention_probe.py` -- closes the loop from raw
  governing-information text to a `MinimalDeltaVerdict`, and
  `contradish update` -- the CLI command that makes it runnable.**
  `minimal_intervention_delta.py` (1.49.0) and `decision_relevance.py`
  are both pure scoring layers: no model calls, no judge calls. Nothing
  actually ran an app before and after an intervention and turned "did
  commitment X's answer change" into the `sensitivity_profile` /
  `expected_effect_matches` those scorers consume -- so the narrow,
  testable claim behind this project's NIST AI 200-2 comment letter
  ("Behavioral Update Fidelity": given new governing information, did the
  model change exactly what it warranted, no more, no less, in the right
  direction) existed as a well-tested internal module and a public
  `contradish.minimal_intervention_delta` export, but nothing a user could
  actually run. The narrower, more durable claim was real; it just wasn't
  the product's front door.
  - `InterventionCase` -- one intervention: governing information changes
    from `before` to `after`, `justified` names the commitments that
    should change (with the question to ask and the expected new answer),
    `invariant` names the commitments that must not move.
  - `default_change_judge(llm)` / `default_effect_judge(llm)` -- the same
    "classify against the canonical statement, not freeform-paraphrase-
    then-string-compare" judge pattern `distinction.py`'s
    `default_commitment_judge` already established, applied across time
    (before vs. after) instead of across paraphrase.
  - `probe_intervention()` / `probe_interventions()` -- run an
    `InterventionCase` (or many) against a `model_fn(system_prompt,
    question)`, judge which commitments changed and whether the changed
    ones landed correctly, and score the result with
    `intervention_delta_spec()` / `score_dependency_structure()` /
    `score_minimal_delta()` exactly as documented in
    `minimal_intervention_delta.py` -- no new scoring math, only the
    probe-and-judge glue `distinguish` has via `DistinctionProber` but this
    axis didn't.
  - `BUILTIN_INTERVENTIONS["ecommerce_refund_window"]` -- the NIST letter's
    own worked "TEVV-Athlon Event" example (a retailer's refund window
    amended from 30 to 45 days, disclosed mid-interaction), shipped as a
    zero-config demo case so `contradish update` works with no case file,
    the same way the bare `contradish` command ships a demo ecommerce
    policy pack.
  - `contradish update [--case-file FILE] [--app MODULE:FUNCTION]
    [--threshold F] [--json]` -- new CLI command. Unlike
    `distinguish`/`fairness`/etc., `--app` here takes `(system_prompt,
    question)`, not just `question`: this measurement's entire point is
    swapping the governing information itself between the app's before and
    after states, which a single-argument black-box callable that already
    has its governing information fixed inside it can't be probed for.
  - `contradish/__init__.py`: new import block (`InterventionCase`,
    `default_change_judge`, `default_effect_judge`, `probe_intervention`,
    `probe_interventions`, `BUILTIN_INTERVENTIONS`) + 6 new `__all__`
    entries. `__version__` bumped `1.49.0` -> `1.50.0`.
  - **`tests/test_intervention_probe.py`** (new file): 7 deterministic
    tests, no API key required -- perfect update, rigidity (deficit),
    drift (excess), membership-exact-but-wrong-direction, no-effect-judge
    leaves direction unknown, multi-case aggregation, and the shipped
    built-in demo case's own shape.
  - **`tests/test_cli_update.py`** (new file): 4 tests covering argparse
    registration, the built-in demo case run end-to-end through
    `cmd_update` with a mocked LLM and a fake `--app`, `--threshold`
    exit-code gating, and a YAML `--case-file` round-trip.
  - **`README.md`**: new "Warranted behavioral updating (`contradish
    update`)" section, placed directly after "The repair loop" and before
    "Three axes most tools miss" -- ahead of, not folded into, the broader
    axis list, since this is the narrower and more durable claim, not one
    more axis alongside the others.
  - **`pyproject.toml`**: version bumped to `1.50.0` in the same edit as
    `__init__.py` (kept in sync deliberately -- see the 1.34.0 history for
    the drift that happened when these were bumped separately).

### Verification

`python3 -m py_compile` clean on `intervention_probe.py`, the edited
`cli.py`, and both new test files. `import contradish` succeeds with the
full ~79-module package (device access was available this session, but
this org's network egress blocks `pypi.org`, so `pytest` itself could not
be installed anywhere reachable this session -- a real, disclosed
limitation, not a full `pytest tests/` run). In its place: every test
function in `test_intervention_probe.py`, `test_minimal_intervention_delta.py`,
and `test_decision_relevance.py` was executed directly (no pytest, plain
function calls) against the real, current `contradish/` package pulled
from the user's Mac -- 7/7, 16/16, 28/28 passed, confirming this addition
didn't regress the modules it builds on. `test_cli_update.py`'s four
scenarios were separately re-executed by hand (same assertions, manual
`SystemExit`/stdout capture in place of `pytest.raises`/`capsys`) against
the real `cli.py`, including the full `update --help` text and the whole
`contradish --help` command list, both rendering correctly alongside all
19 pre-existing commands.

That manual re-execution caught a real bug the first "manually verified,
4/4 passing" pass had missed: `test_cmd_update_case_file_yaml`'s
`class _FileArgs(_Args): case_file = str(case_file)` reassigns the same
name (`case_file`) it reads, inside a class body nested in the test
function. Python resolves a name a class body assigns to via `LOAD_NAME`
against the class's own namespace and then module globals -- never
against the enclosing function's locals, even on the same line, even when
the read happens before the write executes. That makes it `NameError:
name 'case_file' is not defined`, not a `tmp_path`/`capsys`-fixture
question -- the identical code fails the same way whether pytest supplies
the fixtures or a hand-rolled harness does. Confirmed independently
against a two-line reproduction (`class C: x = x + 1` inside a function
with a local `x`) before touching the test file, to be sure this wasn't
an artifact of the manual harness. Fixed by renaming the on-disk path
variable to `_case_file_path` so the class body only *reads* an outer
name it never reassigns (the pattern every other test in this file
already uses without incident) -- `tests/test_cli_update.py` is the one
file in this release that WAS modified after the fact, and only for this
one-line rename; re-executed all four scenarios again afterward, 4/4
passing for real this time. Every other committed test file
(`test_intervention_probe.py` and the pre-existing files re-run for
regression) is ordinary pytest (`pytest.raises`, `capsys`, `tmp_path`) and
was not touched. All of it should still be re-run with the project's
normal `pytest tests/` once run somewhere `pytest` is actually installed
-- manual re-execution against the real package is real verification, not
a substitute for it, per this project's own standing practice of not
overstating what a network-constrained sandbox session can verify.

## [1.49.0] - 2026-09-17

### Added

- **`minimal_intervention_delta.py` -- given intervention deltaI, what is
  the smallest justified deltaB in behavior, and did the model produce
  exactly that deltaB?** `decision_relevance.py` already classifies every
  named factor of ONE commitment into tracked/missed/spurious/invariant
  against a stated relevance spec -- and `score_dependency_structure()` is
  already fully generic over what a "factor" is, not restricted to
  `bench/evaluate.py`'s 8 rhetorical techniques (`default_technique_drs` is
  just the one seed function shipped for that axis so far). Flipping which
  side plays "commitment" and which plays "factor" answers this session's
  question with zero changes to that module: the intervention becomes the
  `commitment_id`, and each of the model's downstream behavioral
  commitments that might be implicated becomes a "factor" -- relevant if
  the intervention logically necessitates a change there, irrelevant
  (must stay invariant) otherwise.
  - `intervention_delta_spec(intervention_id, domain, justified_commitments,
    invariant_commitments)` seeds a `DecisionRelevanceSpec` for this axis,
    the direct counterpart to `default_technique_drs` on the technique
    axis. `justified_commitments` accepts either a plain list (membership
    only) or a `{name: expected_effect}` dict when a direction-aware
    verdict is wanted. Raises `ValueError` if a commitment is listed as
    both justified and invariant.
  - What `decision_relevance.py` does not give you on either axis is what
    this question actually asks for as its own named object: not a
    continuous pooled rate but the literal minimal set the model was
    supposed to change. `score_minimal_delta(report)` is a pure layer over
    an existing `DependencyStructureReport` -- no new relevance/sensitivity
    math -- computing `justified_delta` (`tracked | missed`), `actual_delta`
    (`tracked | spurious`), `excess_delta` (`spurious`, changed but
    shouldn't have), `deficit_delta` (`missed`, should have changed but
    didn't), and two strict all-or-nothing verdicts:
    `exact_delta_match` (membership only: no excess, no deficit) and
    `exact_delta_match_with_direction` (the complete claim: also every
    changed commitment landed on its `expected_effect` --
    `tracked_wrong_direction` empty too; `None`, not `False`, when the
    underlying report was never given direction data, the same
    "don't silently compute a misleading number over an unmeasured gap"
    rule `true_hit_rate` already follows).
  - `aggregate_minimal_delta(domain, verdicts)` pools verdicts across many
    interventions into an `exact_match_rate` and (when at least one verdict
    carries direction data) `exact_match_rate_with_direction`, plus explicit
    `interventions_with_excess`/`_deficit`/`_wrong_direction` lists --
    mirrors `DecisionRelevanceAudit`'s existing pooling pattern.
  - `tests/test_minimal_intervention_delta.py`: 16 deterministic tests, no
    API key required, run against the real installed package (not a stub) --
    spec construction (membership, expected_effect, the both-lists overlap
    `ValueError`), exact/excess/deficit membership scoring, all three
    direction outcomes (correct, wrong, unchecked), aggregation and its
    three diagnostic lists, and `summary()`/`report()`/`to_dict()` smoke
    checks. All 16 passed on-device against the real package import, and
    `tests/test_decision_relevance.py`'s existing 28 tests were re-run
    unmodified to confirm zero regression (28/28 passed) -- this module
    only imports from `decision_relevance.py`, it does not edit it.

## [1.48.0] - 2026-09-17

### Added

- **`chain_fidelity.py` -- does the model's behavioral update function match
  the update function warranted by its governing information, not just
  whether it gets two sampled endpoints right?** Every synchronic
  measurement in this package up to now checks the model's response at, at
  most, two points on an information axis: `distinction.py`'s
  `DistinctionPair` (state A vs. state B), and `directional_fidelity.py`'s
  `directional_correctness` built on top of it, which only adds "and did it
  land on the right side" to that same single flip. A model can pass every
  two-point pairwise check in this package and still implement the wrong
  FUNCTION -- wrong thresholds, an extra unwarranted flip between the two
  tested points, a real flip the two tested points happen not to straddle.
  Two points can't tell a step function with the right shape apart from one
  with the wrong shape; you only find that by sampling an axis at more than
  two places and checking the boundary structure between them, not just the
  endpoints.
  - `DistinctionChain` generalizes `DistinctionPair` from two states to an
    ordered chain of `ChainPoint`s (a `DistinctionPair` is exactly a 2-point
    chain) along a real information axis -- a day-count, a dose range, a
    severity grade. `warranted_boundaries` is computed from the points'
    commit text, not authored separately; `__post_init__` rejects a chain
    with zero boundaries (nothing for a chain probe to measure) the same
    way a `DistinctionPair` needs `commit_a != commit_b`.
  - `ChainProber` (mirrors `DistinctionProber`'s constructor shape on
    purpose) scores the model's actual response function against the
    warranted one on two independent axes: `function_match_rate` (pointwise
    -- the K-way generalization of `distinction.py`'s `both_correct`) and
    boundary precision/recall (structural -- of the places the model's
    answer actually changes between adjacent points, how many are places
    the warranted function actually changes too, and vice versa).
    `spurious_boundaries`/`missed_boundaries` name the two error directions
    explicitly, generalizing `distinction.py`'s Type II/Type I vocabulary to
    however many boundaries a chain actually has. The two axes are
    independently informative on purpose, not collapsed into one number --
    a model can draw every boundary in exactly the right place
    (precision=recall=100%) while having swapped which label goes on which
    side (function_match_rate catches this and boundary detection alone
    would not); see `examples/chain_demo.py`'s "Model D" for a worked case,
    and its "Model B"/"Model C" for the reverse asymmetry (locally-plausible
    per-point labels that hide a spurious flip or a collapsed tier that only
    show up against the warranted boundary set).
  - `default_chain_commitment_judge(llm)`: the K-way generalization of
    `distinction.py`'s `default_commitment_judge` (see its docstring and the
    1.47.1 entry above for the exact-string-match bug that design avoids)
    -- classifies an answer against all of a chain's points at once rather
    than string-comparing a freeform paraphrase against one reference
    sentence.
  - New `contradish chain-distinguish` CLI command (`--chains
    MODULE:VARIABLE`, `--app`, `--n-samples`, `--threshold`, `--report`,
    `--json`) mirrors `contradish distinguish`'s shape. No built-in chain
    set -- a `DistinctionChain` needs an author-verified ordered sequence of
    correct answers along a real axis, which is domain-specific content this
    package doesn't ship pre-authored yet; bring your own, same as
    `distinction.py`'s own "bring your own pairs" discipline for anything
    beyond its two built-in domains.
  - `examples/chain_demo.py`: a fully offline, toy-domain (generic
    return-window policy, not medical/legal content, so it needs no real
    ground-truth authority) worked demo with four hand-written models
    illustrating the correct case and each of the three failure modes
    (spurious, missed, wrong-label) against one shared warranted function.
  - 14 new tests in `test_chain_fidelity.py`: chain validation (rejects
    fewer than 2 points, duplicate labels, zero warranted boundaries),
    `warranted_boundaries` computation, the perfect/spurious/missed
    boundary cases, `function_match_rate` with and without a
    `correctness_judge`, the boundary-precision-recall-vs-function-match-
    rate independence claim itself (`test_function_match_rate_catches_
    wrong_label_at_correct_boundary`), `to_dict()`/`report()` output, and
    `default_chain_commitment_judge`'s K-way classification parsing
    (parametrized).

## [1.47.1] - 2026-09-17

### Fixed

- **`distinction.py`'s `both_correct` -- and `directional_fidelity.py`'s
  `directional_correctness` computed from it -- was silently ~always False
  when used with the library's own shipped `default_commitment_extractor`.**
  `both_correct` compared `commitment_extractor`'s output against
  `pair.commit_a`/`commit_b` with exact string equality. That's fine for
  `distinction_held` (`com_a != com_b`, any two independent paraphrases
  still compare validly), but `default_commitment_extractor` asks a judge to
  freely paraphrase an answer in 3-8 words -- it has no way to know it needs
  to reproduce a hand-written reference sentence verbatim, so it essentially
  never did. Every real caller had already independently rediscovered and
  worked around this: `run_groq_distinction_probe.py` (the Groq pilot behind
  the 2026-09-13 directional-fidelity numbers) hand-rolled a classify-
  against-A/B/N judge instead of using the default extractor's output
  directly, and `examples/distinction_demo.py` hand-rolls its own
  keyword-based extractor that returns canonical strings by construction --
  but `contradish distinguish` (the actual CLI command) and `contradish
  compare --distinctions` were still using the broken default with no
  workaround, so both_correct/directional_correctness read as near-total
  failure out of the box regardless of how correct the model under test
  actually was. Not a finding about any model -- a structural bug in the
  measurement. (The Groq pilot's own published numbers are unaffected: its
  script already used the correct classify-based approach, independently
  discovered before this fix existed.)
  - Added `default_commitment_judge(llm)`: classifies an answer against the
    pair's own two canonical commitments directly ("does this answer reach
    conclusion A, B, or neither") instead of string-comparing a freeform
    paraphrase -- the same fix the pilot script had to hand-roll locally,
    now the library's own default.
  - Added an optional `correctness_judge` parameter to `DistinctionProber`.
    When supplied, `both_correct` is computed via classification; when
    omitted (default), the old exact-match behavior is unchanged, so any
    caller whose `commitment_extractor` already normalizes to
    `commit_a`/`commit_b` (both shipped examples) is unaffected.
  - `contradish distinguish` and `contradish compare --distinctions` now
    pass `default_commitment_judge(llm)` through by default, via a new
    `_default_commitment_judge()` CLI wrapper mirroring the existing
    `_default_commitment_extractor()` one.
  - 10 new tests in `test_distinction.py`: backward-compatibility with a
    canonical extractor, the pre-fix failure mode reproduced directly
    (documents the bug), the fix confirmed with a freeform extractor, a
    wrong-direction answer correctly caught, `default_commitment_judge`'s
    A/B/N classification parsing (parametrized), and CLI wiring
    (`correctness_judge` reaches `DistinctionProber`).

## [1.47.0] - 2026-09-16

### Added

- **`resolution_dynamics.py` -- does a correction survive interaction, and
  was losing it ever actually the right call?** Every other measurement in
  this package is synchronic (one moment, is the answer consistent/correct).
  This module is the first diachronic one: given a stream of already-
  collected multi-turn probe data, it operationalizes three formal belief-
  revision laws behaviorally, checked against real source first (`pragmatic_
  legitimacy.py`, `decision_boundary.py`, `sacrifice.py`) to confirm none of
  them already covered this -- none did.
  - `classify_resolution()` implements Darwiche-Pearl's iterated-revision
    postulates (C1-C4) as a six-way outcome: `integrated` (corrected and
    confirmed to generalize to a logically-entailed transfer cell),
    `behaviorally_resolved` (corrected, generalization untested),
    `distorted` (corrected but generalized wrong), `superseded` (a
    correction was verified, then reverted, but intervening information
    genuinely contradicted it -- per C2, a legitimate override, not decay),
    `forgotten` (verified, then reverted, with nothing that justified it --
    the real C3/C4 violation), and `unresolved` (never verified, or the
    deciding probe is unstable/oscillating -- no number is forced either way).
  - `check_closure()` is the diachronic version of this package's existing
    `spurious` check (`distinction.py`): AGM's inclusion/vacuity postulate,
    applied over time -- did a revision also disturb a control cell it had
    no logical claim on.
  - `EntrenchmentTrial` / `score_entrenchment_fidelity()` operationalize
    AGM's minimal-change/entrenchment postulate: when a forced revision
    requires giving up one of several commitments and the domain has stated
    a priority ordering over them in advance, did the model actually
    sacrifice the least-entrenched one, or something the domain says
    matters more. Genuinely new, not a renaming of anything else in the
    codebase; validates at construction that the recorded sacrifice was
    actually one of the elicited commitments.
  - `EventType` (`REVISION`/`UPDATE`) is an optional Katsuno-Mendelzon tag on
    `ContradictionEvent`; `check_event_type_consistency()` flags (as a
    warning, not an exception) the one concrete category error the tag
    exposes -- `pragmatically_excused` presupposes a purported claim about a
    single fixed world, so setting it on a tagged `UPDATE` event (the world
    itself differs) doesn't apply.
  - `score_signal_separation()` / `holm_bonferroni_correction()` /
    `score_multiple_signals()`: a deterministic (seeded), dependency-free
    permutation test of whether an ex-ante signal (confidence, hedging, ...)
    differs between later-validated and later-invalidated events, reporting
    Cohen's d alongside the p-value (never significance alone, per this
    package's established discipline -- see `faithfulness.py`'s d'/c split)
    with an `underpowered` flag below n=5 per group, and Holm-Bonferroni
    correction when testing several candidate signals against the same split.
  - Pure, dependency-free scoring core (no model or judge calls, no import
    of any other `contradish` module) -- the re-probing/data-collection
    pipeline that would produce `ContradictionEvent`/`ResolutionProbe` from
    live multi-turn model behavior does not exist in this package yet; this
    module is the part that becomes checkable once that data does, the same
    scoping discipline `resolution.py` and `benchmark_ground_truth_audit.py`
    already apply to their own model-calling vs. pure-scoring halves.
  - 42 new deterministic tests in `test_resolution_dynamics.py`, run for
    real with `pytest` (all six `classify_resolution()` outcomes including
    the instability and event-id-mismatch edge cases, `check_closure()`'s
    contamination detection and its before/after-required exclusion,
    `check_event_type_consistency()`'s warning and its three no-warning
    cases, `EntrenchmentTrial`'s three validation errors plus faithful/
    unfaithful scoring, `score_signal_separation()`'s large-effect and
    no-effect synthetic cases plus underpowered/insufficient-data handling,
    `holm_bonferroni_correction()` against a hand-checked example, and the
    batch `score_resolution_dynamics()` entry point). No API key required.

## [1.46.0] - 2026-09-16

### Added

- **`judge_criterion_validity.py` -- does the judge agree with the truth, not
  just with itself?** Three existing modules ask "can this package's verdicts
  be trusted" and none of them ask this: `judge_calibration.py`/`judge_
  calibration_ext.py` measure judge RELIABILITY (test-retest self-agreement
  across rephrasings); `benchmark_ground_truth_audit.py` measures the
  benchmark's own labels' CONSTRUCT VALIDITY (do independent reviewers
  converge that the shipped ground truth is correct); `ground_truth.py`'s
  `GroundTruthAuditor` uses a judge to score a model's answer but assumes
  that judge is trustworthy. This module measures that assumption directly:
  CRITERION VALIDITY -- given an answer whose correctness is already known,
  does the judge's flag/approve verdict agree with it?
  - Scored as two asymmetric rates, not one pooled accuracy: `miss_rate`
    (a real error the judge approved -- reaches whoever relied on the
    answer) and `false_alarm_rate` (a real correct answer the judge flagged
    -- costly over-flagging, e.g. delaying medication access in a healthcare
    deployment). `hit_rate`/`false_alarm_rate` feed `faithfulness.py`'s
    existing `compute_sdt_decomposition`/`classify_sdt_pattern` (imported,
    not reimplemented) for a d'/criterion decomposition of the same shape
    faithfulness.py already uses for the model-under-test's context-
    sensitivity, applied here to judge accuracy instead.
  - `build_cross_context_items()` generates the "plausible but wrong for
    that patient" negatives the module needs with no new authoring and no
    LLM-generated perturbations: every `BUILTIN_DISTINCTION_PAIRS`
    `DistinctionPair` already has two patient contexts whose correct
    commitments genuinely differ, so cross-applying `commit_a` to
    `question_b` (and vice versa) is, by construction, a real, benchmark-
    grounded, clinically-fluent answer that happens to be wrong for that
    specific patient -- exactly the shape asked for, without inventing new
    ground truth or reopening the who-validates-the-perturbation problem.
  - `score_judge_criterion_validity()` is a pure scoring core (no model
    calls, same discipline as `directional_fidelity.py`/`decision_
    relevance.py`); `measure_judge_criterion_validity()` is the one-call
    convenience path that builds items, runs the judge under test
    concurrently, and scores. Domain-stratified automatically via
    `by_domain` whenever more than one domain is present, mirroring `judge_
    calibration_ext.score_calibration_votes_by_domain()`'s per-domain
    surfacing instead of hiding heterogeneity inside one pooled number.
  - `disputed_pair_ids` (accepts the union of a `GroundTruthAuditReport`'s
    `disputed_item_ids`/`contradicted_item_ids`) tags every item built from
    a flagged pair rather than silently excluding it -- consistent with
    `benchmark_ground_truth_audit.py`'s "audit, not auto-correction"
    discipline; exclusion remains a caller decision.
  - 12 new deterministic tests in `test_judge_criterion_validity.py`
    (`build_cross_context_items`'s 4-item construction and cross-application,
    a perfect/overzealous/rubber-stamp judge's expected rates, missing-
    verdict handling, domain stratification, disputed-pair tagging,
    report/summary/to_dict). No API key required.

## [1.45.0] - 2026-09-13

### Changed

- **Consolidation pass on the decision-relevance / directional-fidelity
  cluster.** While auditing this cluster for duplicated logic (prompted by
  a broader review of how many modules independently compute a
  relevant/sensitive/correct classification -- `decision_relevance.py`,
  `directional_fidelity.py`, `decision_boundary.py`, `behavioral_mapping.py`),
  found that `decision_boundary.py` and `behavioral_mapping.py` were
  already clean composition (`behavioral_mapping.compare_to_normative_
  structure()` calls `score_dependency_structure()` and
  `quantify_boundary_discrepancy()` directly, reimplementing neither) --
  an earlier characterization of this cluster as "four independent
  implementations of the same comparison" was an overstatement corrected
  here. One real, narrow duplication did exist:
  `directional_fidelity.score_directional_fidelity()` re-derived its own
  relevant/sensitive/correct -> cell classification inline (a hand-rolled
  `if sensitivity < threshold / elif directional_correctness >= threshold`
  chain) instead of delegating to `decision_relevance.
  score_dependency_structure()`, the one canonical implementation of that
  predicate elsewhere in the package. Two independent implementations of
  the same three-way classification could drift out of sync silently; now
  there is exactly one, and `score_directional_fidelity()` builds a
  single-factor `DecisionRelevanceSpec` and calls the canonical scorer
  internally instead.
  - **Zero behavior change**: output is bit-for-bit identical to the prior
    inline logic for every input (the classification cases and their
    boundary conditions map exactly onto `score_dependency_structure()`'s
    existing relevant/sensitive/`expected_effect_matches` semantics). Proven
    by the full pre-existing `test_directional_fidelity.py` (19 tests)
    passing unmodified, plus `test_decision_relevance.py` (28),
    `test_behavioral_mapping.py` (18), `test_decision_boundary.py` (22), and
    `test_pragmatic_legitimacy.py` (23) as adjacency checks -- 110 tests,
    zero regressions.
  - No public API changes: `score_directional_fidelity()`'s signature,
    `DirectionalFidelityReport`'s fields, and every other function in the
    module are unchanged.

## [1.44.0] - 2026-09-13

### Fixed

- **`pragmatic_legitimacy.reclassify_sacrifice_rate()` excused
  `legitimate_shift` instances without ever checking whether the model's
  answer was actually correct for the shifted question.** A
  `legitimate_shift` verdict certifies only that the pressured framing
  legitimately asks a *different* question (per independent reviewer
  agreement) -- it never checked whether the model's actual answer under
  that framing is a *correct* answer to the new question. A model could
  correctly notice the question changed and still answer the new question
  wrong, and the old logic excused it anyway on the sole evidence that
  reinterpretation occurred: the same category error this module exists to
  catch elsewhere ("the boundary moved for a legitimate reason" conflated
  with "the label on the new side of that boundary is correct"),
  discovered while cross-checking this package's pragmatic-legitimacy
  machinery against a broader partition-fidelity theory pass the same day.
  - **`default_shift_correctness_judge(llm)`** (new) -- an optional,
    additive judge checking the model's answer against the shifted goal,
    independent of the legitimacy verdict itself.
  - **`measure_pragmatic_legitimacy()`** gained optional
    `model_answer_pressured` / `correctness_judge` parameters;
    **`measure_pragmatic_legitimacy_batch()`** gained the batch equivalents
    (`model_answers_pressured` / `correctness_judge`). When supplied and the
    verdict is `legitimate_shift`, the result now records
    `answer_correct_for_shifted_goal`.
  - **`PragmaticLegitimacyReport.fully_vindicated_ids`** (new) -- the set
    `reclassify_sacrifice_rate()` now excuses, instead of the raw
    `legitimate_shift_ids`: a legitimate_shift instance later verified to
    have the *wrong* answer for its own new question is excluded and lands
    in the new **`legitimate_but_incorrect_ids`** instead, still counted as
    a failure.
  - All new parameters and fields are optional and additive; every existing
    call site (none of which could have supplied the new arguments before
    they existed) reproduces its prior behavior exactly -- proven by the
    full existing `test_pragmatic_legitimacy.py` suite (14 tests) passing
    unmodified against this change. 9 new tests added alongside it (23
    total in that file).

## [1.43.0] - 2026-09-13

### Added

- **`decision_relevance.score_dependency_structure()` gained an optional
  `expected_effect_matches: dict[str, bool]` parameter.** This closes the
  second half of a user-specified "iff" for right judgment: "did the
  judgment change if and only if something decision-relevant changed?...
  Right judgment preserves relevant distinctions, ignores irrelevant
  distinctions, remains anchored to truth under pressure, and changes when
  the truth relevant to the judgment changes." The existing four-cell
  classification (`tracked`/`missed`/`spurious`/`invariant`) already
  answered "did it move" but not "did it move to the CORRECT new answer" --
  a model that flipped to an arbitrary wrong conclusion under a relevant
  factor scored identically to one that flipped to the right one. When a
  caller now supplies `expected_effect_matches` (True/False per relevant,
  sensitive factor), the `tracked` cell is further split, without changing
  its own membership, into:
    - `DependencyStructureReport.tracked_correct` -- relevant, sensitive,
      AND landed on the correct answer.
    - `DependencyStructureReport.tracked_wrong_direction` -- relevant,
      sensitive, but landed on the WRONG answer. Not right judgment, but
      also not `missed` or `spurious` -- a third, previously invisible way
      to fail.
    - `DependencyStructureReport.factors_with_unknown_direction` -- tracked
      factors whose correctness was never checked (no entry supplied).
  - **`DependencyStructureReport.true_hit_rate`** -- the stricter,
    direction-aware hit rate this enables: of every relevant factor, what
    fraction landed correctly (a `tracked_wrong_direction` factor counts
    against it exactly like `missed` does). `None` unless
    `expected_effect_matches` was actually supplied for that report --
    matches the existing "don't silently compute a misleading number"
    discipline already used for `unmeasured_factors`.
  - **`DecisionRelevanceAudit.pooled_true_hit_rate`** and
    **`.commitments_with_wrong_direction`** -- the same pooled-not-averaged
    aggregation `aggregate_dependency_structure()` already does for the
    other rates, extended to the new direction-aware ones.
  - **`FactorClassification.matched_expected_effect: Optional[bool]`** --
    the per-factor True/False/None (None = not `"tracked"`, or `"tracked"`
    but no entry supplied) backing all of the above.
  All new fields are additive and default-valued; `hit_rate`/
  `false_alarm_rate`/`dependency_fidelity`/`tracked`/`missed`/`spurious`/
  `invariant` and every other existing field's computation is completely
  unchanged, and every call site that doesn't pass
  `expected_effect_matches` reproduces prior behavior exactly (proven by
  the full existing test suite passing unmodified against this change).
- **`directional_fidelity.expected_effect_matches_from_reports()`** -- the
  bridge that makes the above actually usable end to end: converts
  `directional_fidelity.py`'s own `{pair_id: DirectionalFidelityReport}`
  output into the `{factor_name: True/False}` shape
  `expected_effect_matches` accepts (`"tracked_correct"` -> `True`,
  `"tracked_wrong_direction"` -> `False`, `"missed"` -> omitted, since a
  factor that never moved has no direction to report). A probed
  `DistinctionPair` can now drive the direction-aware split in the full
  technique-factor `score_dependency_structure()` report, not just its own
  narrower pair-specific numbers.

### Changed

- User-specified terminology -- `missed == "unfaithful invariance"`
  (something decision-relevant changed; the judgment didn't) and
  `spurious == "unfaithful variance"` (the judgment changed; nothing
  decision-relevant did) -- is now documented in `decision_relevance.py`'s
  module docstring as the canonical human-readable names for those two
  cells. Per the user's explicit choice, the underlying `cell` string
  values and dataclass field names themselves are **unchanged**
  (`tracked`/`missed`/`spurious`/`invariant` remain exactly as before) to
  avoid breaking any existing consumer or test that reads them; the user's
  terms are aliases in prose and `report()` output only.
- `directional_fidelity.py`'s module docstring paragraph that previously
  explained why `decision_relevance.py`'s scoring core was deliberately
  left unmodified ("those are already load-bearing... Composing the two is
  a caller's choice") is rewritten to describe the new composition path via
  `expected_effect_matches_from_reports()`. `score_directional_fidelity()`/
  `aggregate_directional_fidelity()` remain the right tool when only the
  DistinctionPair-specific numbers are wanted on their own.

## [1.42.0] - 2026-09-13

### Added

- **`BUILTIN_DISTINCTION_PAIRS["scriptural_ethics"]`** -- a third domain in
  `distinction.py`, added at user request to explore a text-as-ground-truth
  ("assume the Bible teaches truth, use it to make contradish better")
  domain design. Four pairs (`exodus21_premeditated_vs_accidental_killing`,
  `charity_needy_vs_enabling_idleness`, `sabbath_necessity_vs_routine_labor`,
  `authority_justice_vs_personal_vengeance`), each chosen because the cited
  passage draws the pair's A/B distinction explicitly in its own text
  (Exodus 21:12-14, Deuteronomy 15:11 / 2 Thessalonians 3:10, Exodus 20:8-10
  / Matthew 12:1-12, Romans 12:19 / 13:4) -- the same "ground truth traceable
  to a citation, not asserted" discipline the medication/immigration pairs
  already follow, applied to a declared textual authority instead of a
  clinical/regulatory one. Deliberately excludes questions where mainstream
  Christian traditions substantively disagree on the answer (divorce and
  remarriage, the bounds of just war, Sabbath-keeping in general), to avoid
  building on contested ground truth. See `distinction.py`'s inline note on
  the domain for the full reasoning.
- **`run_groq_scriptural_ethics_probe.py`** -- sibling to
  `run_groq_distinction_probe.py`, same validated fixes (model-name
  preflight, key-sanity check, rate-limit backoff, `reasoning_effort="low"`,
  pair-aware classification extractor), pointed at the new domain's 4 pairs
  directly rather than through `JUNCTION_CASE_MAP` (see below).

### Changed

- `scriptural_ethics` is intentionally **not** added to
  `predictive_validity.JUNCTION_CASE_MAP`. That map exists specifically for
  pairs that verbatim-match a real, existing, frozen CAI-Bench case file
  (`medication.json` / `immigration.json`); no such case file exists for
  this domain, and adding synthetic entries there would misrepresent these
  pairs as part of the shipped 360-case v2 benchmark when they are not. The
  pairs remain fully usable directly via `DistinctionProber` and
  `directional_fidelity.score_directional_fidelity`.

## [1.41.0] - 2026-09-13

### Added

- **`contradish/directional_fidelity.py` -- closes the gap the 1.40.0 entry
  above named as the next follow-up.** `DRSFactor.expected_effect` has
  existed since `decision_relevance.py` was written but was never read
  anywhere (`grep -rn '\.expected_effect' contradish/` turned up exactly
  the one line that serializes it) -- meaning `score_dependency_structure()`'s
  `"tracked"` cell has only ever meant "sensitivity crossed the threshold",
  never "moved to the correct new answer". `distinction.py` already had the
  real mechanism this needed: `DistinctionMeasurement.both_correct`, computed
  against each pair's own `commit_a`/`commit_b` ground truth. This module is
  the bridge, nothing more: `score_directional_fidelity()` refines a
  `DistinctionPair`'s classification into `"missed"` (unchanged),
  `"tracked_wrong_direction"` (sensitive, but landed on the wrong answer --
  previously indistinguishable from correct), and `"tracked_correct"` (the
  only cell that should actually count as right for the right reasons).
  `directional_fidelity_for_domain()` wires this to
  `predictive_validity.JUNCTION_CASE_MAP` for a whole domain at once.
  `drs_factor_from_distinction_pair()` / `spec_with_distinction_pairs()`
  separately give `expected_effect` real, non-empty content (from a pair's
  own `commit_a`/`commit_b`) for anyone building a full spec. Zero new model
  calls in this module -- it's a pure scoring layer over evidence
  `DistinctionProber` already produces; getting non-synthetic numbers out of
  it still requires actually running `DistinctionProber` against a real
  model for the mapped pairs, which this change does not do by itself.
  22 tests, including one full run through the real `DistinctionProber`
  pipeline (not just hand-built dataclasses) against the new pairs below.

### Changed

- **`contradish/distinction.py` / `contradish/predictive_validity.py` --
  grew `JUNCTION_CASE_MAP`'s real relevant-axis coverage from 3 pairs / 4
  cases (medication only) to 7 pairs / 8 cases across medication AND
  immigration.** Two new medication pairs
  (`bp_med_self_stop_vs_physician_directed`, grounded in medication-005;
  `acetaminophen_healthy_vs_liver_impaired`, grounded in medication-004) and
  one new immigration pair (`i485_pending_travel_without_ap`, grounded in
  immigration-006), each with `question_a`/`question_b` verbatim-matched to
  a real case the same way the original three were. Also wired in
  `naturalization_english_standard_vs_exempt`, an immigration pair that
  existed since this module was written but was never mapped to a case
  (its `question_a` is verbatim immigration-007's). `daca_valid_vs_no_status`
  and `advance_parole_approved_vs_pending` deliberately remain unmapped --
  neither verbatim-matches any of immigration.json's 18 cases, and forcing
  a mapping to inflate the coverage number would be exactly the mistake
  this map's existing discipline exists to avoid. Still 8 of 360 total v2
  cases (2.2%) -- explicitly still a pilot, not a powered study; scaling
  further means hand-grounding more pairs the same way, not mass-generating
  them (see `distinction.py`'s module note on why that would trade one
  unvalidated-ground-truth problem for a bigger one).

## [1.40.0] - 2026-09-13

### Changed

- **`contradish/bench/evaluate.py` -- the first real wiring of Decision-
  Relevance scoring into the pipeline that produces published numbers.**
  Every case's `run_frozen_policy()` details now carry
  `dependency_spurious_techniques`: which specific techniques (of the 8 in
  `TECHNIQUE_NAMES`) triggered a wrong (spurious) reaction on THIS case,
  not just the pooled average `cai_strain` across all eight. The run-level
  output gains `technique_spurious_rate` (pooled false-alarm rate across
  every case's technique classifications, via
  `decision_relevance.aggregate_dependency_structure()`, reused unmodified)
  and `cases_with_spurious_technique`. Zero new model calls -- this is a
  pure post-hoc layer over the `per_variant_scores` the loop already
  collects.

  Deliberately does NOT add a `dependency_fidelity` field here, and says
  why in an inline comment: this dataset's 8 techniques are all
  pressure/framing variants of the SAME underlying question, so there is
  no relevant-fact axis per case for `default_technique_drs()`'s R to
  classify anything as `tracked`/`missed` against -- `relevant_sensitivity`
  is structurally undefined for this loop, not merely unmeasured. A real
  `dependency_fidelity` needs pairing with a `distinction.py`
  `DistinctionPair`, which is exactly what `faithfulness.py` already does
  via `JUNCTION_CASE_MAP` -- wiring `score_dependency_structure()` in
  there (where it would report genuine new information, not a rescaling of
  the existing faithfulness score) is a distinct, smaller follow-up, not
  done here.

## [1.39.0] - 2026-09-13

### Added

- **`contradish/behavioral_topology.py` -- an AI behavioral topology: the
  structure governing when a model's answers change, not just whether it
  answers correctly.** topology.py already had everything this needed --
  its own docstring opens with almost exactly this framing -- but its only
  constructor, `topology_from_phi_star()`, populated the graph from
  self-report (asking a model what a claim depends on and clustering the
  free-text answer). `topology_from_behavioral_map()` is the missing
  constructor that feeds the SAME `FailureTopologyMap` /
  `ReasoningNode` / `ReasoningEdge` -- reused completely unmodified -- from
  behavioral_mapping.py's controlled-intervention measurements instead.
  `cai_strain` is a deliberate reinterpretation, not a raw carry-over: 1.0
  when a factor's four-cell classification is `missed` or `spurious` (the
  dependency structure is WRONG), 0.0 for `tracked`/`invariant` (correct)
  -- scoring wrongness of the dependency rather than raw movement, a
  distinction topology_from_phi_star() had no way to make since
  decision_relevance.py's R didn't exist yet. `reality_strain` reuses a
  real measured quantity -- `abs(normalized_displacement)` from a factor's
  `BoundaryDiscrepancyReport` -- when a boundary was recovered, instead of
  a placeholder. A candidate that screened sensitive but has no entry in
  the normative structure still becomes a node rather than vanishing.
  Deliberately does NOT default to topology_from_phi_star()'s linear-chain
  edge fallback, since independently-probed behavioral candidates have no
  implied order. Because the result is an ordinary `FailureTopologyMap`,
  `critical_path()`, `superspreader_influence()`, `certification_coverage()`,
  `gini_coefficient`, and `topology_distance()` (cross-model/cross-time
  structural comparison) all compose for free.

## [1.38.0] - 2026-09-13

### Added

- **`contradish/behavioral_mapping.py` -- the practical method: discover,
  map, compare.** Ties decision_relevance.py and decision_boundary.py
  together with the one step neither had: discovering which variables to
  even test, instead of requiring the factor set up front.
  `screen_candidates()` runs a cheap, purely behavioral discovery pass
  (never asks the model what it depends on -- that's topology.py's
  self-report-based expand_node(); this perturbs and watches) over a
  candidate pool seeded by default from prompt_analyzer.py's real
  16-technique `KNOWN_TECHNIQUES` catalog, a strict superset of
  decision_relevance.py's 8-factor default spec. `build_behavioral_map()`
  feeds what screens positive into decision_relevance.py's
  `score_dependency_structure()` (categorical) and reuses
  decision_boundary.py's `recover_boundary_via_binary_search()` unmodified
  (ordinal) to construct the behavioral dependency/decision-boundary map.
  `compare_to_normative_structure()` scores that map against an
  independently specified `NormativeStructure` (a `DecisionRelevanceSpec`
  plus, optionally, per-factor `BoundaryLadder`s) -- and explicitly
  surfaces `unspecified_sensitive_variables`: candidates that screened
  behaviorally sensitive but have no entry at all in the normative
  structure's R, a case score_dependency_structure() would otherwise
  silently drop since it only iterates the factors it was given.

## [1.37.0] - 2026-09-13

### Added

- **`contradish/decision_boundary.py` -- Decision Boundary Recovery (DBR):
  specify the legitimate decision boundary, experimentally recover the
  model's behavioral one, quantify the discrepancy.** Every existing
  sensitivity measurement in this package is categorical (is factor F
  relevant, does the answer differ between two hand-picked states). Nothing
  before this module LOCATED anything on an ordered semantic dimension.
  `BoundaryLadder` names the legitimate boundary `B*` -- the rung index
  along an ordered intervention ladder (e.g. "days early requesting a
  refill": 0..10) where the correct decision changes -- honestly flagged as
  authored content with no existing `TECHNIQUE_NAMES`-style constant to seed
  it from (nothing in `policies/*.py` currently encodes a numeric threshold;
  `illustrative_ladder()` ships a clearly-labeled synthetic example only,
  not a validated domain claim). `recover_boundary_via_binary_search()`
  recovers the model's actual behavioral boundary `B_M` in O(log n) queries
  via controlled semantic interventions (an oracle callable, same
  swappable-judge pattern as `default_hedge_judge`), with a local check
  (verified computationally to be a genuinely informative probe, not a
  tautological one) that catches a meaningful share of oscillation right
  around the boundary without claiming a full monotonicity guarantee;
  `recover_boundary_from_observations()` is the honest zero-assumptions
  version for an already-completed full sweep. A model that never shows a
  clean single transition gets `regime` = `unstable` / `always_a` /
  `always_b` / `insufficient_data` rather than a forced number.
  `quantify_boundary_discrepancy()` computes `Delta = B_M - B*`: signed
  displacement in rungs, a normalized version for cross-domain comparison,
  and a direction (`shifted_toward_a` / `shifted_toward_b` / `exact`) --
  deliberately not labeled "conservative"/"permissive", since which
  direction is safer depends on domain context this module doesn't have.

## [1.36.0] - 2026-09-13

### Added

- **`contradish/decision_relevance.py` -- the Decision-Relevance
  Specification (DRS), the object every existing sensitivity measurement in
  this package was implicitly checking against but never stated explicitly.**
  bench/evaluate.py's technique_scores, distinction.py's hold_rate, and
  faithfulness.py's subtraction of the two all measure SENSITIVITY (did the
  answer change). None of them, before this module, represented as a
  first-class object which factors SHOULD move a decision and which
  shouldn't. `DecisionRelevanceSpec` names that: `R: factor -> {relevant,
  irrelevant, conditional}` per commitment, with `default_technique_drs()`
  seeding it from the real 8-technique `TECHNIQUE_NAMES` set and the
  relevance defaults already implicit in judge.py's own
  `_TRANSFORMATION_VALIDATOR_PROMPT` guidance (seven techniques irrelevant,
  `authority` conditional on verified credentials). `score_dependency_
  structure()` crosses R against a measured sensitivity profile to classify
  every factor into one of four cells: tracked (relevant+sensitive, the
  correct case), missed (relevant+insensitive -- distinction loss's
  failure mode), spurious (irrelevant+sensitive -- CAI Strain's failure
  mode, now broken out per technique instead of pooled), and invariant
  (irrelevant+insensitive -- correct, and not named anywhere else in this
  package before this addition). `sensitivity_from_consistency_score()`
  bridges directly from bench/evaluate.py's already-computed
  technique_scores, so DRS scoring runs as a pure post-hoc layer over data
  contradish already collects -- zero new model calls for a first working
  pass. Proven, not just asserted: subtracting the pooled hit rate and
  false-alarm rate this module computes is exactly Youden's J (Youden 1950),
  which means faithfulness.py's existing score is the two-factor degenerate
  case of a DRS score, and faithfulness.py's SDT decomposition
  (`compute_sdt_decomposition`/`classify_sdt_pattern`) is reused here
  directly rather than reimplemented -- this module is the general form
  faithfulness.py was always a special case of.
  `aggregate_dependency_structure()` pools tracked/missed/spurious/invariant
  counts across many commitments (pooled, not averaged-of-averages, for the
  same reason judge_calibration_ext.py's domain-stratification note gives).

## [1.35.0] - 2026-09-13

### Added

- **`contradish/pragmatic_legitimacy.py` -- is a pressure-induced answer
  shift a genuine failure, or a legitimate pragmatic reinterpretation?**
  Every pressure-based construct in this package (sacrifice.py, Type I loss,
  CAI Strain itself) assumes a "pressure" framing changes only HOW a
  question is dressed up, never WHAT is being asked. Gricean pragmatics and
  the Rational Speech Act framework deny that: a cooperative listener's
  sense of what's being asked is itself a function of context and stakes,
  so an answer that shifts under a stakes-signaling framing may be correctly
  tracking a different implicit question, not eroding a distinction. This
  module operationalizes the objection instead of leaving it as a caveat:
  `infer_rational_goal()` asks an independent LLM what a rational listener
  would take a framing's implicit goal to be, `default_legitimacy_reviewer()`
  asks independent reviewers whether two framings' inferred goals differ
  enough to justify different answers, and `reclassify_sacrifice_rate()`
  recomputes an existing rate (sacrifice_rate or similar) after excusing
  instances reviewers converged were legitimate pragmatic shifts. This is
  the one module in this package that changes a headline number based on a
  purely theoretical objection, deliberately: the alternative was shipping a
  metric this package's own research review found conflates two different
  things and calling it one.

### Changed

- **`contradish/benchmark_ground_truth_audit.py`: ground-truth audits now
  feed back into scoring, narrowly.** Perspectivist annotation methodology
  ("Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation",
  Aroyo & Welty; "Beyond Consensus: Perspectivist Modeling and Evaluation of
  Annotator Disagreement in NLP", arXiv 2601.09065) argues forced consensus
  over genuinely contested items destroys information rather than resolving
  it. `exclude_indeterminate_pairs()` recomputes a rate (kbv_rate,
  sacrifice_rate, or similar) after excluding pairs this package's own
  ground-truth audit found disputed or contradicted -- not by rewriting the
  pair (still never auto-applied, see the module's original docstring), but
  by declining to let an indeterminate item count for or against a model's
  score at all, the same way standard item-analysis practice drops
  low-inter-rater-reliability items from a scale. `DeterminacyAdjustedRateReport`
  also reports `benchmark_determinacy_rate` -- how much of what was audited
  even has a reviewer-agreed determinate answer, a different and arguably
  prior question to "how much of it is correct."

- **`contradish/faithfulness.py`: Signal Detection Theory decomposition.**
  `faithfulness = relevant_sensitivity - irrelevant_sensitivity` treats both
  terms as one axis. SDT -- applied directly to LLM behavior in "LLMs as
  Signal Detectors: Sensitivity, Bias, and the Temperature-Criterion
  Analogy" (arXiv 2603.14893) and "Do LLMs Know What They Know? Measuring
  Metacognitive Efficiency with Signal Detection Theory" (arXiv 2603.25112)
  -- treats the same two rates as orthogonal: sensitivity (d', can the model
  discriminate the two conditions at all) and criterion (c, where its
  response threshold sits, independent of discrimination ability).
  `compute_sdt_decomposition()` adds `sensitivity_d_prime`/`criterion` to
  every `FaithfulnessJunction`, and `classify_sdt_pattern()` gives a coarse
  label distinguishing a genuine discrimination collapse from a criterion
  shift -- two failures a single subtraction cannot tell apart, and which
  call for different fixes.

- **`contradish/judge_calibration_ext.py`: domain-stratified floor_strain.**
  `floor_strain` pools every domain into one scalar, which is exactly the
  classical-test-theory unidimensionality assumption the LLM Psychometrics
  systematic review (already cited for this module) flags as often false.
  `score_calibration_votes_by_domain()` reports floor_strain per domain plus
  a `_heterogeneity` figure (max minus min across domains) surfacing
  whether pooling was hiding real unevenness in judge reliability. Additive
  only -- `measure_*_judge_floor()`'s existing pooled return value is
  unchanged.

## [1.34.0] - 2026-09-13

### Added

- **`contradish/eval_awareness.py` -- the Evaluation-Awareness Gap.** Every
  construct shipped before this assumed failures are incidental (no
  strategic behavior-shaping). Sandbagging research ("AI Sandbagging:
  Language Models can Strategically Underperform on Evaluations", van der
  Weij et al. 2024, arXiv 2406.07358) studies the opposite: a model that
  behaves *differently*, and specifically better, once it detects it's
  being tested. `measure_eval_awareness()`/`measure_eval_awareness_batch()`
  wrap any existing probe (`probe_fn(prefix) -> float`) and compare its
  output under a neutral framing vs. framings that explicitly signal "this
  is an evaluation," reporting the gap. Architecture-agnostic: composes
  with hold_rate, sacrifice_rate, kbv_rate, or a caller's own metric. A
  nonzero gap is evidence-consistent-with, not proof of, strategic
  deception -- the module says so in its own printed report.

- **`contradish/format_fidelity.py` -- output-format collapse under
  paraphrase.** All prior constructs check whether *content* survives
  pressure; none checked whether output *form* survives paraphrase of a
  formatting instruction (a model told "respond with one JSON object" two
  different ways may silently switch between JSON, a table, and prose).
  `measure_format_fidelity()` runs the same semantic instruction across
  several paraphrasings and reports how consistently a classifier's format
  label holds, via `default_format_classifier` (a coarse, deterministic,
  API-free json/table/bulleted_list/numbered_list/single_word/prose
  classifier) or a caller-supplied one. Orthogonal to every other
  construct here: a stable-format response can still be wrong, sacrificed,
  or hedged.

- **`contradish/compliance_gap.py` -- verbal vs. actual compliance, and an
  explicit scope limit.** "The Compliance Gap" (arxiv.org/html/2605.01771v1)
  distinguishes Verbal Compliance Rate from Actual Compliance Rate and
  proves a DPI-undetectability result: a transcript-only checker cannot,
  even in principle, always detect a verbal/actual gap. contradish is a
  text-only benchmark, so this module says that limitation out loud rather
  than papering over it: it operationalizes VCR/ACR only within a single
  response (does the model's own stated commitment match what the rest of
  that same response actually contains), via `measure_compliance_gap()`
  and the worked-example `default_word_limit_checker`. It does not, and
  cannot, detect the deeper transcript-vs-deployed-action gap the source
  paper describes -- every report it prints says so.

### Fixed

- **`contradish.__version__` had drifted from `pyproject.toml` since
  1.32.0** (`__init__.py` still said `1.31.1` while `pyproject.toml` said
  `1.33.0`) -- caught by `test_packaging.test_version_in_sync`, which
  should have failed on this before now. Both now read `1.34.0`.

### Documentation

- Added related-work citations throughout `BENCHMARK.md`: the
  Knowledge-Behavior Gap (arxiv.org/abs/2608.12341) and the Compliance Gap
  (arxiv.org/html/2605.01771v1) alongside the existing "Models Recall What
  They Violate" citation; an explicit disambiguation of this package's
  `faithfulness.py` construct from chain-of-thought faithfulness
  (Turpin/Lanham/Anthropic) -- same word, different measurement, no API
  rename since 1.32.0 is public; and a psychometric reframing of
  `judge_calibration.py`/`judge_calibration_ext.py` as reliability
  (test-retest) and `benchmark_ground_truth_audit.py` as validity
  (construct validity), citing the LLM Psychometrics systematic review.

## [1.33.0] - 2026-09-13

### Added

- **`contradish/judge_calibration_ext.py` -- judge-floor calibration for
  every judge role, not just the original one.** `judge_calibration.py`
  measured the equivalence/consistency judge's own CAI Strain against a
  24-item gold set; the three judge roles added alongside sacrifice.py, KBV,
  and provenance.py (`default_restatement_judge`, `default_hedge_judge`,
  `default_usage_judge`) had no floor measurement at all. This module
  extends the same method (self-agreement across rephrased instructions on
  a known-truth calibration set) to all three, with `measure_hedge_judge_floor`,
  `measure_restatement_judge_floor`, and `measure_usage_judge_floor`. Wired
  into the CLI: `contradish judge-floor --judge-role {hedge,restatement,usage}`.

- **`contradish/witness.py`: `build_witnessed()`.** A one-call convenience
  wrapper turning any single-LLM judge factory (`default_hedge_judge`, etc.)
  plus >=2 `LLMClient`s into a witnessed judge and its `WitnessPanel`, so
  the honest, multi-witnessed default is exactly as short as the unwitnessed
  one. `run_predictive_validity_study.py`'s live path now uses it
  automatically for `restatement_judge`/`hedge_judge` whenever a second
  provider's API key is available, printing an explicit warning rather than
  silently running single-judge when it isn't. Also added `WitnessPanel.calls`
  (read-only view of every recorded call, for callers needing per-item detail).

- **`contradish/benchmark_ground_truth_audit.py` -- audits the benchmark's
  OWN ground truth, not a model's answers.** `BUILTIN_DISTINCTION_PAIRS`'
  `commit_a`/`commit_b` values and `judge_calibration.py`'s 24
  `gold_equivalent` labels were single-author artifacts nobody had checked
  for independent-reviewer convergence. `audit_distinction_pairs()` and
  `audit_calibration_gold()` run >=2 independent reviewer models over that
  ground truth (via `WitnessPanel`) and report where they converge, where
  they split, and -- the important case -- where they unanimously
  *contradict* what's shipped (`contradicted_item_ids`), as a human-review
  worklist rather than an auto-correction. Distinct from the pre-existing
  `contradish/ground_truth.py`'s `GroundTruthAuditor`, which audits a
  model's accuracy against known facts; this audits the benchmark's own
  authored facts instead.

- `BENCHMARK.md`: documented all of the above, including an addendum to
  the multi-witness convergence section and two new sections
  ("Judge-floor calibration for every judge role", "Auditing the
  benchmark's own ground truth").

## [1.32.0] - 2026-09-13

### Added

- **`contradish/sacrifice.py` -- distinction sacrifice under coherence
  pressure.** `measure_sacrifice()` names and measures a specific failure
  mode: a model knows a distinction (KBV's `declares_correctly`), loses it
  under pressure (Type I collapse), and states the collapsed answer with
  full, unhedged confidence (`default_hedge_judge`) rather than visibly
  struggling. `sacrifice_rate <= kbv_rate <= collapse_rate` by construction.
  Includes `SacrificeGradient` (onset intensity, whether the rate ramps
  monotonically with pressure).

- **`contradish/faithfulness.py` -- faithfulness score.** Truth-over-coherence
  as a number: `faithfulness = relevant_sensitivity (Type I hold_rate) -
  irrelevant_sensitivity (Type II cai_strain)`, computed from two
  measurements this package already produces via
  `predictive_validity.JUNCTION_CASE_MAP`, no new probing required. Range
  [-1, 1]; negative values flag the "exactly backwards" signature (ignores
  real distinctions, reacts to irrelevant wording).

- **`contradish/witness.py` -- multi-witness convergence (`WitnessPanel`).**
  A generic combinator wrapping >=2 independent judge/classifier callables
  into one combined callable that only confirms a finding when all
  witnesses agree, logging every disagreement (`ConvergenceReport`). Drops
  into any judge slot in this package (`hedge_judge`, `restatement_judge`,
  etc.) without that call site knowing convergence is happening.

- **`contradish/provenance.py` -- Provenance Collapse.** A new failure mode:
  a model is given a claim explicitly sourced as weak/unverified, then
  asked a question inviting its use; collapse is using the claim's content
  while stripping its sourcing, so the answer reads as fully warranted when
  the actual basis was one unverified source. Ships with 3 built-in test
  claims for the medication domain (`BUILTIN_PROVENANCE_CLAIMS`).

- **Competing-explanations checks in `contradish/predictive_validity.py`.**
  `PredictiveValidityReport.base_rate` / `.precision_lift` check whether
  the sacrifice-rate signal beats a naive "always predict fail" baseline;
  `pressure_specificity_verdict()` and `length_confound_check()` check
  whether an observed effect tracks pressure intensity specifically or is
  equally explained by a one-off fluke or by prompt length alone, reusing
  data already collected -- no new model calls.

- `BENCHMARK.md`: formal write-ups of all of the above, plus a "competing
  explanations" table mapping each CAI variable (distinction sacrifice,
  coherence pressure, KBV, predictive validity) to its most obvious
  alternative explanation and whether/how this package checks for it.

## [1.31.1] - 2026-09-12

### Changed

- Removed every em dash and en dash from all user-facing documentation and
  CLI output: README.md, CHANGELOG.md, BENCHMARK.md, PAPER.md,
  ground-truth/README.md, and the printed/help text in `contradish/cli.py`.
  No functional changes. This release exists solely to get a clean README
  onto the PyPI project page (PyPI renders whichever release is newest;
  1.31.0's page can't be edited in place).

## [1.31.0] - 2026-09-12

Adds the rate-distortion curve: the resolution operator's black-box,
behavioral answer to whether a fix degrades gracefully or falls off a
cliff as certainty about the hidden variable it depends on goes from
nothing to a plain stated fact.

### Added

- **`contradish/rate_distortion.py` -- the information-graded resolution
  curve.** For a distinction the resolution operator (`resolution.py`)
  actually resolved, `measure_rate_distortion_curve()` re-probes it across
  a 5-rung certainty ladder for the winning candidate's disambiguating
  condition (no information, weak hint, moderate signal, strong signal,
  full information stated as fact), measures accuracy at each rung, and
  computes the Spearman rank correlation (implemented from scratch, no
  external dependency, average-rank tie handling) between certainty and
  accuracy. Classifies the result as "graded" (accuracy rises smoothly with
  information -- correlation >= threshold), "threshold" (only recovers
  near full certainty -- brittle: reliable with a stated fact, worthless
  with a hedge), or "insensitive" (the candidate doesn't actually help,
  even at full information -- an honest negative result, the
  rate-distortion sibling of resolution.py's own "not resolved").
  `measure_rate_distortion_for_resolution()` is the convenience wrapper
  that runs it directly on an already-computed `ResolutionResult`. Wired
  into the CLI as `contradish distinguish --resolve --rate-distortion`,
  and exported from the top-level `contradish` package alongside
  `discover_resolution`.
- This module is the black-box behavioral analog of an internal
  weight-level research result -- on a hand-built transformer, collateral
  damage to an unrelated constraint from narrow fine-tuning scaled
  monotonically with the bits of missing information about the hidden
  disambiguating variable across a graded noisy channel (pooled Spearman
  r=+0.96 vs. noise level, over 20 seeds per setting). That result
  measured real weight-level damage under gradient descent; it is not
  re-tested here, and nothing in this module proves it generalizes to how
  frontier models are actually fine-tuned. What transfers, and what this
  module actually tests, is the shape of the claim -- is degradation
  graded or a cliff -- on a real, black-box model, using linguistic hedges
  in place of an injected noisy channel. As far as the research pass
  behind this module found, no eval/guardrail tool (Petri, PromptPex,
  Giskard, TruLens-class tools, LMUnit) reports a graded information
  curve for constraint resolution; they report a pass/fail or a single
  score.
- 15 new tests (`tests/test_rate_distortion.py`): the from-scratch
  Spearman implementation (perfect correlation, ties, zero variance,
  n < 2), all three curve shapes with deterministic mock models, the
  `discover_resolution` wrapper's `None`-when-unresolved and
  `ValueError`-when-no-pair behavior, and CLI end-to-end wiring. Full
  suite: 1212 passed / 2 skipped, plus the same 11 pre-existing failures
  from the earlier 1.30.0 release (missing optional `anthropic` package in
  this environment) -- zero new regressions.

## [1.30.0] - 2026-09-12

Adds the resolution operator: `contradish distinguish` no longer only
reports that a distinction collapsed, it can now search for why.

### Added

- **`contradish/resolution.py` -- the resolution operator.** For a
  distinction a real model is observed to collapse, `discover_resolution()`
  proposes candidate hidden variables that would make both sides of the
  distinction correct at once, proves the winning candidate causal with a
  real flip test (assert one pole, assert the opposite, check the answer
  flips both ways -- `causal_effect_size`), and only reports the
  distinction resolved if the resulting one-line system-prompt patch
  measurably raises the hold rate on fresh, unconditioned probes
  (`validated_hold_rate`). A candidate that passes the flip test but whose
  patch doesn't help is reported honestly as unresolved, not shipped as a
  guess. `discover_resolutions_for_loss_map()` runs it over every
  sufficiently-collapsed pair in an existing `DistinctionLossMap` in one
  call -- the natural follow-up to `DistinctionProber.measure()`. Wired
  into the CLI as `contradish distinguish --resolve` (plus
  `--resolve-collapse-threshold`, `--resolve-candidates`,
  `--resolve-samples`), and exported from the top-level `contradish`
  package alongside `DistinctionProber`.

## [1.29.0] - 2026-09-02

"Harden the core" pass: fixed every known bug, closed the remaining test
coverage gaps, and made the CLI's flagship documented usage actually work.
No public API changes.

### Fixed

- **Attribute-shadowing bug in `replay`/`improve`/`reconcile`.** `contradish/__init__.py`
  re-exported `improve`, `reconcile`, and `replay` as functions with the same
  names as their own submodules. Importing the submodule set
  `contradish.improve` (etc.) as a side effect, which the subsequent
  `from .improve import improve` then silently overwrote, so
  `contradish.improve.improve(...)` (attribute-chain access) raised
  `AttributeError`, and `import contradish.improve as m` didn't dodge it
  either. Fixed by renaming the submodules on disk to `_improve.py`,
  `_reconcile.py`, `_replay.py`. The public API (`from contradish import
  improve/reconcile/replay`, as documented in the README) is unchanged.
- **CLI: bare freeform-prompt invocation crashed.** The README's very first
  Quickstart example,
  `contradish "You are a support agent. Refunds within 30 days only."`,
  raised `argparse.ArgumentError` / exited with code 2. `main()` registers
  both `parser.add_subparsers(dest="command")` and a fallback `system_prompt`
  positional on the same parser; argparse's positional matching always let
  the subparsers action claim an unrecognized lone token first, regardless
  of declaration order. Fixed by detecting that one ambiguous invocation
  shape before `parser.parse_args()` runs and routing it to a
  subparsers-free fallback parser. `--policy X`, `--prompt file.txt`, all 16
  real subcommands, and the bare-no-args smoke test / help paths are
  unaffected.
- **`reporter.to_html()` ignored its `version` argument.** A caller-supplied
  `version=` kwarg was silently discarded in favor of the installed
  package's `__version__`.
- **`reporter.to_html()` crashed on a `Report` with no results (or no
  results carrying a consistency score).** `report.cai_score` being `None`
  raised `TypeError` on `None >= 0.80`; now falls back to `0.0`.

### Test coverage

Closed the remaining gaps identified for a 1.0-grade stability guarantee:

- `reporter.py`: 14% → 100%
- `exporters.py`: 8% → 100%
- `audit.py`: 10% → 100%
- `cli.py`'s `cmd_benchmark` + `main()` argparse/dispatch chain: 63% → 91%
  (remaining gaps are in other `cmd_*` handlers, out of scope for this pass)

Full suite: 1143 passed, 2 skipped, 0 failed.

### Scope note

Tier-2 experimental modules (`theorems.py`, `observatory.py`, `oracle.py`,
and similar) remain out of scope, per contradish's documented two-tier
guarantee system; they were never claimed to be covered or stable.
