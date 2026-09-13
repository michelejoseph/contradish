# Changelog

All notable changes to contradish are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this file starts at 1.29.0;
earlier releases were not retroactively documented.

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
