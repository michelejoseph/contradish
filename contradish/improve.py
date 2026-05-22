"""
End-to-end repair loop: detect → diagnose → repair → re-verify.

contradish.improve() is the single entry point that takes a model and a set of
test cases, runs the benchmark, finds the failures, generates an improved
system prompt that addresses the failure modes, re-runs the benchmark with
the improved prompt, and returns the diff in CAI Strain.

The home-page promise is that contradish closes the repair loop. This module
is the code path that delivers it in one call.

Usage:
    from contradish import improve

    # Against a built-in policy pack
    result = improve(
        cases="ecommerce",
        system_prompt="You are a support agent. Refunds within 30 days only.",
        model="gpt-4o-mini",
        target_strain=0.15,
    )

    print(result.summary())
    print(result.improved_prompt)

CLI:
    contradish improve --policy ecommerce --model gpt-4o-mini --target-strain 0.15

Two methods are exposed:

  method="prompt"   (default, shippable today)
      Generate N improved system prompt variants, test each, return the
      variant that produced the lowest CAI Strain. The artifact is the
      improved prompt string — drop into your config and ship.

  method="finetune" (scaffold, not yet wired to a training provider)
      Produce a JSONL fine-tuning pair set from the diagnoses, scaffold
      the upload-and-train call to OpenAI's fine-tuning API. The actual
      training submission is gated behind --enable-finetune so the cost
      doesn't surprise anyone; the JSONL is written to disk regardless.

The prompt path returns a usable improvement in one CLI invocation. The
fine-tune path returns a queued job ID once the integration lands. Either
way, the user does not run the benchmark twice by hand.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from .models   import TestCase, Report, RepairResult


@dataclass
class ImprovementResult:
    """
    Result of contradish.improve().

    Attributes:
        baseline_strain:   CAI Strain before any repair attempt.
        improved_strain:   CAI Strain after applying the best improvement.
        strain_delta:      improved_strain - baseline_strain (negative = better).
        target_strain:     Caller-specified target. target_met if improved_strain <= target.
        target_met:        Whether the improved Strain hit the target.
        method:            "prompt" or "finetune".
        baseline_prompt:   The original system prompt that was tested.
        improved_prompt:   The system prompt that achieved improved_strain.
                           For method="finetune", this is the prompt that ALSO
                           anchored the fine-tuning run.
        baseline_report:   The full Report from the baseline run.
        improved_report:   The full Report from the post-repair run.
        variant_results:   All RepairResults that were tried (sorted best first).
        ft_jsonl_path:     Path to the JSONL fine-tuning pair set written to disk.
                           None for pure-prompt repair.
        ft_job_id:         Fine-tuning job ID from the provider. None until the
                           submission step lands. Surfaces as "scaffolded" in
                           the summary so the user knows what's done vs queued.
    """
    baseline_strain:  float
    improved_strain:  float
    strain_delta:     float
    target_strain:    float
    target_met:       bool
    method:           str
    baseline_prompt:  str
    improved_prompt:  str
    baseline_report:  Report
    improved_report:  Report
    variant_results:  list[RepairResult] = field(default_factory=list)
    ft_jsonl_path:    Optional[str] = None
    ft_job_id:        Optional[str] = None
    # Holdout-split fields. Populated when improve() is called with
    # holdout_frac > 0. The baseline_strain / improved_strain pair above
    # reports honest (holdout) numbers when a holdout exists; the train-set
    # numbers are kept here so a sceptical reader can verify the model
    # generalized beyond the cases used to pick the winner.
    train_baseline_strain: Optional[float] = None
    train_improved_strain: Optional[float] = None
    holdout_size:          Optional[int]   = None
    train_size:            Optional[int]   = None
    # Truth-gate fields. Populated only when the cases carry canonical answers.
    # The integrity rule: a consistency win that came at the cost of truth is
    # not a win. A prompt rewrite can make a model answer more consistently AND
    # more wrongly (confident, fluent, uniform, incorrect). When that happens,
    # truth_regressed is True and target_met is forced False regardless of the
    # CAI Strain improvement. contradish will not sell a fluent lie as a fix.
    baseline_truth_strain: Optional[float] = None
    improved_truth_strain: Optional[float] = None
    truth_regressed:       bool            = False

    def summary(self) -> str:
        """One-line summary for stdout."""
        arrow = "↓" if self.strain_delta < 0 else "↑"
        pct   = abs(self.strain_delta) / self.baseline_strain * 100 if self.baseline_strain else 0
        hit   = "target met" if self.target_met else f"target {self.target_strain:.2f} not yet hit"
        scope = ""
        if self.holdout_size is not None:
            scope = f"  [holdout n={self.holdout_size}, train n={self.train_size}]"
        truth = ""
        if self.truth_regressed:
            truth = (
                f"  [REJECTED: truth_strain rose {self.baseline_truth_strain:.3f} "
                f"to {self.improved_truth_strain:.3f}; consistency gain is not a win]"
            )
        elif self.improved_truth_strain is not None:
            truth = f"  [truth_strain {self.baseline_truth_strain:.3f} to {self.improved_truth_strain:.3f}]"
        return (
            f"CAI Strain {self.baseline_strain:.3f} → {self.improved_strain:.3f}  "
            f"({arrow} {abs(self.strain_delta):.3f} / {pct:.0f}% reduction)  "
            f"[{hit}]  method={self.method}{scope}{truth}"
        )

    def to_dict(self) -> dict:
        return {
            "method":           self.method,
            "baseline_strain":  self.baseline_strain,
            "improved_strain":  self.improved_strain,
            "strain_delta":     self.strain_delta,
            "target_strain":    self.target_strain,
            "target_met":       self.target_met,
            "baseline_prompt":  self.baseline_prompt,
            "improved_prompt":  self.improved_prompt,
            "ft_jsonl_path":    self.ft_jsonl_path,
            "ft_job_id":        self.ft_job_id,
            "train_baseline_strain": self.train_baseline_strain,
            "train_improved_strain": self.train_improved_strain,
            "holdout_size":          self.holdout_size,
            "train_size":            self.train_size,
            "baseline_truth_strain": self.baseline_truth_strain,
            "improved_truth_strain": self.improved_truth_strain,
            "truth_regressed":       self.truth_regressed,
            "baseline_report":  self.baseline_report.to_dict(),
            "improved_report":  self.improved_report.to_dict(),
            "variant_strains":  [
                {
                    "rank":             v.rank,
                    "original_strain":  v.original_cai_strain,
                    "improved_strain":  v.improved_cai_strain,
                    "strain_delta":     v.strain_delta,
                }
                for v in self.variant_results
            ],
        }


# ── App-factory helpers ────────────────────────────────────────────────────────

def _make_app_for_prompt(
    system_prompt:  str,
    model:          Optional[str]      = None,
    provider:       Optional[str]      = None,
    api_key:        Optional[str]      = None,
    max_tokens:     int                = 256,
) -> Callable[[str], str]:
    """
    Build an LLM-app callable bound to (system_prompt, model, provider).

    Returns a function that takes one user question and returns the model's
    reply text. Used internally by improve() to test each candidate
    system prompt against the same case set.
    """
    from .llm import LLMClient
    llm = LLMClient(api_key=api_key, provider=provider)
    use_model = model or llm.fast_model

    def app(question: str) -> str:
        if llm.provider == "anthropic":
            msg = llm._client.messages.create(
                model=use_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
            )
            return msg.content[0].text.strip()
        else:
            resp = llm._client.chat.completions.create(
                model=use_model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                ],
            )
            return resp.choices[0].message.content.strip()

    return app


def _resolve_cases(cases: Union[str, list[TestCase]]) -> list[TestCase]:
    """If `cases` is a string, treat it as a policy name and load the pack."""
    if isinstance(cases, list):
        return cases
    if isinstance(cases, str):
        from .policies import load_policy
        pack = load_policy(cases)
        return pack.cases
    raise TypeError(f"cases must be a list of TestCase or a policy name string, got {type(cases).__name__}")


# ── Fine-tune scaffold ─────────────────────────────────────────────────────────

def _write_finetune_jsonl(
    report:         Report,
    system_prompt:  str,
    out_path:       Optional[str] = None,
) -> str:
    """
    Write a JSONL fine-tuning pair set from a Report's failures.

    Each failure becomes one chat-format training example:
      {"messages": [
          {"role": "system",    "content": <system_prompt + repair_patch>},
          {"role": "user",      "content": <adversarial input that drifted>},
          {"role": "assistant", "content": <counterfactual correct response>},
      ]}

    Returns the path to the written file. The file is suitable to upload to
    OpenAI's fine-tuning endpoint as-is.
    """
    out = Path(out_path or "repair_finetune.jsonl").resolve()
    n = 0
    with open(out, "w") as f:
        for r in report.failed:
            if not r.contradictions:
                continue
            for pair in r.contradictions:
                example = {
                    "messages": [
                        {"role": "system",
                         "content": (system_prompt or "").strip() + (
                             "\n\n" + r.suggestion.strip() if r.suggestion else ""
                         )},
                        {"role": "user",      "content": pair.input_b},
                        {"role": "assistant", "content": pair.output_a},
                    ],
                    "meta": {
                        "rule":        r.test_case.name,
                        "severity":    pair.severity,
                        "strain":      r.cai_strain,
                        "explanation": pair.explanation,
                    },
                }
                f.write(json.dumps(example) + "\n")
                n += 1
    return str(out) if n > 0 else ""


# ── Main entry point ───────────────────────────────────────────────────────────

def improve(
    cases:          Union[str, list[TestCase]],
    system_prompt:  str                  = "",
    model:          Optional[str]        = None,
    provider:       Optional[str]        = None,
    api_key:        Optional[str]        = None,
    method:         str                  = "prompt",
    target_strain:  float                = 0.20,
    n_variants:     int                  = 3,
    paraphrases:    int                  = 5,
    enable_finetune: bool                = False,
    ft_provider:    str                  = "openai",
    verbose:        bool                 = True,
    concurrency:    int                  = 4,
    holdout_frac:   float                = 0.0,
    seed:           int                  = 0,
) -> ImprovementResult:
    """
    Close the repair loop end-to-end.

    Runs the benchmark, identifies failures, generates an improved system
    prompt, re-runs the benchmark, and returns the before/after diff.

    Args:
        cases:           List of TestCase objects, or the name of a built-in
                         policy pack (e.g. "ecommerce").
        system_prompt:   The starting system prompt. If empty, the policy
                         pack's description is used as a stub. Required for
                         any meaningful repair on real apps.
        model:           Model to test against. Defaults to the configured
                         LLMClient's fast model (gpt-4o-mini / claude-haiku).
        provider:        "anthropic" or "openai". Auto-detected from env.
        api_key:         API key. Reads from env if omitted.
        method:          "prompt" (default) generates improved system prompts
                         and picks the best by post-Strain. "finetune" writes
                         the training JSONL and (when enable_finetune=True)
                         submits a fine-tuning job to the provider; the prompt
                         path still runs first so a usable improvement is
                         returned regardless of fine-tune completion.
        target_strain:   The CAI Strain you want to achieve. Used to set
                         target_met on the result. Default 0.20.
        n_variants:      Number of improved-prompt variants to try.
        paraphrases:     Number of adversarial paraphrases per case.
        enable_finetune: Set True to actually submit the fine-tuning job. The
                         JSONL is always written; this gates the API call so
                         training costs don't surprise the user.
        ft_provider:     Fine-tuning provider. Only "openai" is wired today.
        verbose:         Print progress to stdout.
        concurrency:     Test cases run in parallel per Suite.run. Default 4.
                         Also bounds parallelism across variant re-tests inside
                         PromptRepair.fix.
        holdout_frac:    If > 0, reserve this fraction of cases as a held-out
                         set. Variants are diagnosed and selected on the
                         remaining (train) cases; the winner is then re-scored
                         on the holdout. The returned baseline/improved Strain
                         report the holdout numbers, with the train numbers
                         retained on the result for transparency. This fixes
                         the train-on-test bias of the legacy path. Default 0.0
                         preserves prior behavior.
        seed:            Seed for the holdout shuffle so splits are reproducible.

    Returns:
        ImprovementResult with before/after CAI Strain, the improved prompt,
        both full reports, and (for fine-tune mode) the JSONL path and a
        job ID once submitted.
    """
    from .suite import Suite
    from .repair import PromptRepair

    case_list = _resolve_cases(cases)
    if not case_list:
        raise ValueError("No test cases provided. Pass a policy name or a non-empty list of TestCase objects.")

    # ── 0. Optional train/holdout split ────────────────────────────────────────
    # With holdout_frac > 0, diagnose & pick the winner on a train subset,
    # then re-score the winner on the held-out cases. The "headline" numbers
    # surfaced to the user are the holdout numbers — that's the honest read.
    holdout_cases: list[TestCase] = []
    train_cases:   list[TestCase] = list(case_list)
    if holdout_frac and holdout_frac > 0:
        if not (0 < holdout_frac < 1):
            raise ValueError(f"holdout_frac must be between 0 and 1 exclusive, got {holdout_frac}")
        rng = random.Random(seed)
        shuffled = list(case_list)
        rng.shuffle(shuffled)
        n_holdout = max(1, int(round(len(shuffled) * holdout_frac)))
        n_holdout = min(n_holdout, len(shuffled) - 1)  # always leave ≥1 for train
        holdout_cases = shuffled[:n_holdout]
        train_cases   = shuffled[n_holdout:]
        if verbose:
            print(f"  [improve] holdout split: train n={len(train_cases)}, holdout n={len(holdout_cases)} (seed={seed})")

    use_holdout = bool(holdout_cases)

    # ── 1. Baseline run (TRAIN cases — these drive variant generation) ─────────
    if verbose:
        print(f"\n  [improve] baseline run on train set: {len(train_cases)} cases on {model or '<default model>'}")
    baseline_app = _make_app_for_prompt(
        system_prompt=system_prompt or "You are a helpful assistant.",
        model=model,
        provider=provider,
        api_key=api_key,
    )
    baseline_suite = Suite(app=baseline_app, api_key=api_key, provider=provider)
    for tc in train_cases:
        baseline_suite.add(tc)
    baseline_report_train = baseline_suite.run(
        paraphrases = paraphrases,
        verbose     = verbose,
        concurrency = concurrency,
    )

    train_baseline_strain = baseline_report_train.cai_strain or 0.0
    if verbose:
        print(f"  [improve] train CAI Strain (baseline): {train_baseline_strain:.3f}")

    # Helper: re-score an arbitrary case set against a prompt, return (Report, strain).
    def _score_on(case_set: list[TestCase], prompt_for_app: str) -> tuple[Report, float]:
        app   = _make_app_for_prompt(
            system_prompt=prompt_for_app, model=model, provider=provider, api_key=api_key,
        )
        suite = Suite(app=app, api_key=api_key, provider=provider)
        for tc in case_set:
            suite.add(tc)
        rep    = suite.run(paraphrases=paraphrases, verbose=False, concurrency=concurrency)
        strain = rep.cai_strain or 0.0
        return rep, strain

    # ── Baseline on holdout (if any) so we have an honest before/after pair ────
    if use_holdout:
        if verbose:
            print(f"  [improve] baseline run on holdout: {len(holdout_cases)} cases")
        baseline_report_holdout, baseline_strain_holdout = _score_on(holdout_cases, system_prompt or "You are a helpful assistant.")
        baseline_strain_headline = baseline_strain_holdout
        baseline_report_headline = baseline_report_holdout
    else:
        baseline_strain_headline = train_baseline_strain
        baseline_report_headline = baseline_report_train

    # ── Early exit: already meeting target on the headline (holdout if present) ─
    if baseline_strain_headline <= target_strain:
        if verbose:
            print(f"  [improve] baseline already meets target ({target_strain:.2f}); no repair needed.")
        return ImprovementResult(
            baseline_strain  = baseline_strain_headline,
            improved_strain  = baseline_strain_headline,
            strain_delta     = 0.0,
            target_strain    = target_strain,
            target_met       = True,
            method           = method,
            baseline_prompt  = system_prompt,
            improved_prompt  = system_prompt,
            baseline_report  = baseline_report_headline,
            improved_report  = baseline_report_headline,
            variant_results  = [],
            train_baseline_strain = train_baseline_strain if use_holdout else None,
            train_improved_strain = train_baseline_strain if use_holdout else None,
            holdout_size          = len(holdout_cases)    if use_holdout else None,
            train_size            = len(train_cases)      if use_holdout else None,
        )

    # ── 2. Generate improved-prompt variants ────────────────────────────────────
    if verbose:
        print(f"  [improve] generating {n_variants} improved-prompt variants")

    repair = PromptRepair(api_key=api_key, provider=provider, n=n_variants)
    def app_factory(prompt: str) -> Callable[[str], str]:
        return _make_app_for_prompt(
            system_prompt=prompt,
            model=model,
            provider=provider,
            api_key=api_key,
        )

    variants = repair.fix(
        system_prompt = system_prompt,
        report        = baseline_report_train,
        app_factory   = app_factory,
        paraphrases   = paraphrases,
        verbose       = verbose,
        concurrency   = concurrency,
        cases         = train_cases,   # variants are scored on train only
    )

    if not variants:
        if verbose:
            print("  [improve] variant generation failed; returning baseline result.")
        return ImprovementResult(
            baseline_strain  = baseline_strain_headline,
            improved_strain  = baseline_strain_headline,
            strain_delta     = 0.0,
            target_strain    = target_strain,
            target_met       = False,
            method           = method,
            baseline_prompt  = system_prompt,
            improved_prompt  = system_prompt,
            baseline_report  = baseline_report_headline,
            improved_report  = baseline_report_headline,
            variant_results  = [],
            train_baseline_strain = train_baseline_strain if use_holdout else None,
            train_improved_strain = train_baseline_strain if use_holdout else None,
            holdout_size          = len(holdout_cases)    if use_holdout else None,
            train_size            = len(train_cases)      if use_holdout else None,
        )

    # ── 3. Pick the best on TRAIN ───────────────────────────────────────────────
    best = variants[0]   # PromptRepair sorts best-first (highest cai_score = lowest Strain)
    improved_prompt       = best.improved_prompt
    train_improved_report = best.report
    train_improved_strain = train_improved_report.cai_strain or train_baseline_strain

    if verbose:
        arrow_t = "↓" if train_improved_strain < train_baseline_strain else "↑"
        print(f"  [improve] train winner: Strain {train_baseline_strain:.3f} {arrow_t} {train_improved_strain:.3f}")

    # ── 4. Score winner on HOLDOUT (the honest read) ────────────────────────────
    if use_holdout:
        if verbose:
            print(f"  [improve] scoring winner on holdout ({len(holdout_cases)} cases)")
        improved_report, improved_strain = _score_on(holdout_cases, improved_prompt)
        if verbose:
            arrow_h = "↓" if improved_strain < baseline_strain_headline else "↑"
            print(f"  [improve] holdout result: Strain {baseline_strain_headline:.3f} {arrow_h} {improved_strain:.3f}")
    else:
        improved_report = train_improved_report
        improved_strain = train_improved_strain

    strain_delta = round(improved_strain - baseline_strain_headline, 4)

    # ── Truth gate ──────────────────────────────────────────────────────────────
    # A consistency win that traded away truth is not a win. If the cases carry
    # canonical answers, compare truth_strain before and after. When the rewrite
    # made the model more consistent but more wrong (truth_strain rose beyond a
    # small tolerance), reject the win: target_met is forced False even if the
    # CAI Strain target was hit. Without canonicals, this is a no-op and behavior
    # is unchanged.
    baseline_truth = getattr(baseline_report_headline, "truth_strain", None)
    improved_truth = getattr(improved_report, "truth_strain", None)
    truth_regressed = False
    _TRUTH_TOLERANCE = 0.02
    if baseline_truth is not None and improved_truth is not None:
        truth_regressed = improved_truth > baseline_truth + _TRUTH_TOLERANCE

    target_met = (improved_strain <= target_strain) and not truth_regressed
    if verbose and truth_regressed:
        print(
            f"  [improve] REJECTED: CAI Strain fell but truth_strain rose "
            f"{baseline_truth:.3f} to {improved_truth:.3f}. A more consistent, "
            f"more wrong model is not an improvement."
        )

    result = ImprovementResult(
        baseline_strain  = baseline_strain_headline,
        improved_strain  = improved_strain,
        strain_delta     = strain_delta,
        target_strain    = target_strain,
        target_met       = target_met,
        method           = method,
        baseline_prompt  = system_prompt,
        improved_prompt  = improved_prompt,
        baseline_report  = baseline_report_headline,
        improved_report  = improved_report,
        variant_results  = variants,
        train_baseline_strain = train_baseline_strain if use_holdout else None,
        train_improved_strain = train_improved_strain if use_holdout else None,
        holdout_size          = len(holdout_cases)    if use_holdout else None,
        train_size            = len(train_cases)      if use_holdout else None,
        baseline_truth_strain = baseline_truth,
        improved_truth_strain = improved_truth,
        truth_regressed       = truth_regressed,
    )

    # ── 4. Fine-tune scaffold ───────────────────────────────────────────────────
    if method == "finetune":
        # Always write the JSONL: a useful artifact even if we don't submit.
        # Mine the train baseline run (it carries the failures, contradictions,
        # and repair suggestions the pair set is built from).
        ft_jsonl_path = _write_finetune_jsonl(
            report        = baseline_report_train,
            system_prompt = improved_prompt,   # train against the improved prompt
            out_path      = "repair_finetune.jsonl",
        )
        result.ft_jsonl_path = ft_jsonl_path or None
        if verbose and ft_jsonl_path:
            print(f"  [improve] wrote fine-tuning pairs: {ft_jsonl_path}")

        if enable_finetune:
            # The actual submission lives behind an explicit flag so this
            # cost-bearing step never happens by accident from a docs example.
            try:
                job_id = _submit_finetune_job(ft_jsonl_path, ft_provider, model, verbose)
                result.ft_job_id = job_id
                if verbose and job_id:
                    print(f"  [improve] fine-tuning job submitted: {job_id}  (provider={ft_provider})")
            except NotImplementedError as e:
                if verbose:
                    print(f"  [improve] fine-tune submission not yet implemented for provider={ft_provider}: {e}")
        else:
            if verbose:
                print("  [improve] fine-tune JSONL written; pass enable_finetune=True to actually submit the job.")

    return result


# ── Closing the loop at the system level ────────────────────────────────────

def improve_from_production(
    report,
    replay_report,
    system_prompt:   str             = "",
    model:           Optional[str]   = None,
    *,
    base_cases:      Optional[list]  = None,
    kinds:           tuple           = ("validity_gap", "coverage_gap"),
    match_threshold: float           = 0.3,
    relevance_fn                     = None,
    target_strain:   float           = 0.15,
    verbose:         bool            = True,
    **improve_kwargs,
) -> Optional[ImprovementResult]:
    """
    Close the repair loop at the system level: let production recalibrate the
    benchmark, then repair against it.

    improve() closes the loop *inside one benchmark run*: it repairs the cases
    you already wrote. But the benchmark only knows what you thought to test.
    Production breaks on things you didn't. This function feeds those back in:
    it reconciles the benchmark Report against a ReplayReport of real
    contradictions, turns every break the benchmark missed (a validity gap) or
    never tested (a coverage gap) into a fresh adversarial case, and runs the
    normal repair loop over those cases plus any you supply.

    The result is auto-recalibration. A contradiction observed in production
    becomes a harder benchmark case, drives a prompt rewrite, and the post-run
    Strain measures whether the rewrite actually held. It is the same detect,
    diagnose, patch, remeasure cycle, but now seeded by reality instead of
    only by the cases you hand-wrote.

    Args:
        report:          benchmark Report to reconcile (the cases you tested).
        replay_report:   ReplayReport of production contradictions (from logs).
        system_prompt:   the prompt to repair. Same role as in improve().
        model:           model to test against. Defaults to the fast model.
        base_cases:      optional TestCases to fold in alongside the derived
                         ones, so the regression set still covers your originals.
        kinds:           which reconciliation verdicts become cases. Defaults to
                         validity_gap + coverage_gap (the two the bench got
                         wrong). "confirmed" is excluded because the bench
                         already catches those.
        match_threshold: relevance cutoff passed to reconcile().
        relevance_fn:    optional semantic scorer passed to reconcile() (e.g. an
                         EmbeddingRelevance) for higher recall on paraphrases.
        target_strain:   target for the repair. Default 0.15.
        verbose:         print progress.
        **improve_kwargs: forwarded to improve() (provider, api_key, method,
                         n_variants, paraphrases, holdout_frac, seed,
                         concurrency, enable_finetune, ...).

    Returns:
        An ImprovementResult, or None when reconciliation surfaced nothing the
        benchmark missed (no validity or coverage gaps), so there is simply
        nothing new to repair.
    """
    from .reconcile import reconcile, cases_from_reconciliation

    rec = reconcile(
        report, replay_report,
        match_threshold=match_threshold,
        relevance_fn=relevance_fn,
    )
    derived = cases_from_reconciliation(rec, kinds=kinds)

    if verbose:
        print(
            f"  [improve_from_production] reconcile: "
            f"{len(rec.validity_gaps)} validity gap(s), "
            f"{len(rec.coverage_gaps)} coverage gap(s), "
            f"{len(rec.confirmed)} confirmed -> {len(derived)} new case(s)"
        )

    if not derived:
        if verbose:
            print("  [improve_from_production] production surfaced nothing the "
                  "benchmark missed; nothing to repair.")
        return None

    # Merge supplied base cases first (so a user's canonical case wins on a
    # collision), then the production-derived cases, deduped by (input, answer).
    merged: list = []
    seen = set()
    for tc in list(base_cases or []) + derived:
        key = (str(getattr(tc, "input", "")).strip().lower(),
               str(getattr(tc, "canonical_answer", "") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(tc)

    if verbose:
        print(f"  [improve_from_production] repairing over {len(merged)} case(s) "
              f"({len(base_cases or [])} supplied + {len(derived)} from production)")

    return improve(
        cases=merged,
        system_prompt=system_prompt,
        model=model,
        target_strain=target_strain,
        verbose=verbose,
        **improve_kwargs,
    )


_DEFAULT_FT_BASE_MODEL = "gpt-4o-mini-2024-07-18"


def _submit_finetune_job(
    jsonl_path: str,
    provider:   str,
    base_model: Optional[str],
    verbose:    bool,
) -> Optional[str]:
    """
    Submit a fine-tuning job to the configured provider.

    OpenAI today. Anthropic / Vertex / Bedrock will land as separate branches
    when their fine-tuning APIs are stable enough to wrap behind one call.

    The caller gates this entire path behind `--enable-finetune` (CLI) or
    `enable_finetune=True` (Python), so this function only ever runs when
    the user has explicitly opted into paying for a training run.

    Args:
        jsonl_path: Path to the training-pair JSONL written by
                    _write_finetune_jsonl. Must be uploadable as-is.
        provider:   "openai" today. Anything else raises NotImplementedError
                    with a clear message so the user knows nothing was billed.
        base_model: Optional base model. Defaults to the cheapest commonly
                    fine-tunable OpenAI model so an accidental submission
                    doesn't bill enterprise rates.
        verbose:    Stdout progress about the upload step.

    Returns:
        The provider's job ID string. The caller is expected to surface this
        to the user so they can poll status with the provider's own tools.

    Raises:
        ImportError if the openai SDK isn't installed.
        NotImplementedError for non-openai providers.
        Anything the provider SDK raises (auth, quota, validation) propagates
        unchanged — failure modes belong with the user, not silently swallowed.
    """
    if provider != "openai":
        raise NotImplementedError(
            f"fine-tune provider '{provider}' is not yet supported. "
            f"Today only 'openai' is wired. The JSONL at {jsonl_path} is "
            f"ready to upload to your provider of choice manually."
        )

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai SDK is required for fine-tune submission. Install with:\n"
            "    pip install \"contradish[openai]\""
        ) from e

    if not jsonl_path:
        raise ValueError("no JSONL path supplied to fine-tune submission")

    client = OpenAI()
    model  = base_model or _DEFAULT_FT_BASE_MODEL

    if verbose:
        print(f"  [improve] uploading {jsonl_path} to OpenAI")
    with open(jsonl_path, "rb") as f:
        training_file = client.files.create(file=f, purpose="fine-tune")
    if verbose:
        print(f"  [improve] file uploaded:    {training_file.id}")
        print(f"  [improve] creating fine-tune job on base model: {model}")

    job = client.fine_tuning.jobs.create(
        training_file = training_file.id,
        model         = model,
    )
    if verbose:
        print(f"  [improve] check status: openai api fine_tuning.jobs.retrieve {job.id}")
    return job.id
