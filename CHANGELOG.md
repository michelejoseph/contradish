# Changelog

All notable changes to contradish are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this file starts at 1.29.0;
earlier releases were not retroactively documented.

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
