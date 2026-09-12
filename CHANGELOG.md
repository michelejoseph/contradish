# Changelog

All notable changes to contradish are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this file starts at 1.29.0 —
earlier releases were not retroactively documented.

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
