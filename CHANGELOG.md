# Changelog

All notable changes to contradish are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this file starts at 1.29.0 —
earlier releases were not retroactively documented.

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

## [1.29.0] — 2026-09-02

"Harden the core" pass: fixed every known bug, closed the remaining test
coverage gaps, and made the CLI's flagship documented usage actually work.
No public API changes.

### Fixed

- **Attribute-shadowing bug in `replay`/`improve`/`reconcile`.** `contradish/__init__.py`
  re-exported `improve`, `reconcile`, and `replay` as functions with the same
  names as their own submodules. Importing the submodule set
  `contradish.improve` (etc.) as a side effect, which the subsequent
  `from .improve import improve` then silently overwrote — so
  `contradish.improve.improve(...)` (attribute-chain access) raised
  `AttributeError`, and `import contradish.improve as m` didn't dodge it
  either. Fixed by renaming the submodules on disk to `_improve.py`,
  `_reconcile.py`, `_replay.py`. The public API (`from contradish import
  improve/reconcile/replay`, as documented in the README) is unchanged.
- **CLI: bare freeform-prompt invocation crashed.** The README's very first
  Quickstart example —
  `contradish "You are a support agent. Refunds within 30 days only."` —
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
guarantee system — they were never claimed to be covered or stable.
