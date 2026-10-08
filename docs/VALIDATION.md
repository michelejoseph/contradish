# Validating Behavioral Update Fidelity

What has been checked, what has not, and how someone else can check it.

## Already done (by the author)

1. **Formal definitions and proofs**: `docs/BUF-SPEC.md` §1–§4.
2. **Exhaustive model-checking** of T1–T10 and L1–L3: `python -m contradish.reference` (≈ 8.7M checks, ~40 s; `--full` enlarges the 3-case universe).
3. **Differential testing** of the production scorer against the reference on every contract in the finite universes: `tests/test_reference_conformance.py`.
4. **Conformance vectors** with a 19-bug mutation library every vector set must kill: `python -m contradish.conformance --check|--mutants`.
5. **Separability of the invariants** via single-defect oracle agents: `python -c "from contradish.invariants import format_matrix; print(format_matrix())"`.

Limit: the reference and production scorer share an author. This is *N-version programming by one programmer*, which catches slips but not shared misunderstandings.

## What independent validation means, concretely

**A. Re-implementation (a day of work).** Someone who has not read `transition.py` implements §1–§2 of the spec in any language, runs it against `conformance/vectors.json`, and reports any failing vector. Failures are either bugs in their code, ambiguities in the spec (fix the spec), or bugs in ours (fix ours and add a vector).

**B. Adversarial review of the definitions.** A reviewer tries to produce (i) a system that is intuitively faithful and scores low, or (ii) one that is intuitively unfaithful and scores high. Every such case is either a bug or a documented limit.

**C. Measurement validity on real systems.** Run `contradish transition` / `run_sequence` against deployed models with a pre-registered contract set and two or more independent labelers; report inter-rater agreement on the labels and on the contracts' warrants; report which scores move with the label noise (`contradish/judge_criterion_validity.py` and `judge_calibration.py`).

**D. Predictive/criterion validity.** Show the score predicts something that matters: incidents under injection, regressions after policy updates, rater-judged "appropriateness of update".

**E. Prior-art check.** Before any priority claim, read and cite the nearest published work on belief updating in language models (belief-revision-for-ML, belief-shift benchmarks, instruction-hierarchy and prompt-injection benchmarks, agent behavioral contracts). The defensible novelty is in the unified transition object, per-case authority, the identities in §3, hysteresis, and the conformance suite — not in the question itself.

## What would count as success

A. and B. are cheap and should happen first. Publishing the spec, vectors and reference under a DOI-bearing release lets anyone do them. C. and D. are research results and belong in a paper; this repository does not claim them.

## How to credit

See `CITATION.cff`. A Zenodo release of this repository mints a DOI; `.zenodo.json` carries the metadata.
