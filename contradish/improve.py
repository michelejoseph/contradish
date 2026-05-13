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

    def summary(self) -> str:
        """One-line summary for stdout."""
        arrow = "↓" if self.strain_delta < 0 else "↑"
        pct   = abs(self.strain_delta) / self.baseline_strain * 100 if self.baseline_strain else 0
        hit   = "target met" if self.target_met else f"target {self.target_strain:.2f} not yet hit"
        return (
            f"CAI Strain {self.baseline_strain:.3f} → {self.improved_strain:.3f}  "
            f"({arrow} {abs(self.strain_delta):.3f} / {pct:.0f}% reduction)  "
            f"[{hit}]  method={self.method}"
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

    # ── 1. Baseline run ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  [improve] baseline run: {len(case_list)} cases on {model or '<default model>'}")
    baseline_app = _make_app_for_prompt(
        system_prompt=system_prompt or "You are a helpful assistant.",
        model=model,
        provider=provider,
        api_key=api_key,
    )
    baseline_suite = Suite(app=baseline_app, api_key=api_key, provider=provider)
    for tc in case_list:
        baseline_suite.add(tc)
    baseline_report = baseline_suite.run(paraphrases=paraphrases, verbose=verbose)

    baseline_strain = baseline_report.cai_strain or 0.0
    if verbose:
        print(f"  [improve] baseline CAI Strain: {baseline_strain:.3f}")

    # Early exit: already meeting target
    if baseline_strain <= target_strain:
        if verbose:
            print(f"  [improve] baseline already meets target ({target_strain:.2f}); no repair needed.")
        return ImprovementResult(
            baseline_strain  = baseline_strain,
            improved_strain  = baseline_strain,
            strain_delta     = 0.0,
            target_strain    = target_strain,
            target_met       = True,
            method           = method,
            baseline_prompt  = system_prompt,
            improved_prompt  = system_prompt,
            baseline_report  = baseline_report,
            improved_report  = baseline_report,
            variant_results  = [],
        )

    # ── 2. Generate improved-prompt variants ────────────────────────────────────
    if verbose:
        print(f"  [improve] generating {n_variants} improved-prompt variants")

    repair = PromptRepair(api_key=api_key, provider=provider, n=n_variants)
    # PromptRepair needs an app_factory that builds an app from a prompt
    def app_factory(prompt: str) -> Callable[[str], str]:
        return _make_app_for_prompt(
            system_prompt=prompt,
            model=model,
            provider=provider,
            api_key=api_key,
        )

    variants = repair.fix(
        system_prompt=system_prompt,
        report=baseline_report,
        app_factory=app_factory,
        paraphrases=paraphrases,
        verbose=verbose,
    )

    if not variants:
        if verbose:
            print("  [improve] variant generation failed; returning baseline result.")
        return ImprovementResult(
            baseline_strain  = baseline_strain,
            improved_strain  = baseline_strain,
            strain_delta     = 0.0,
            target_strain    = target_strain,
            target_met       = False,
            method           = method,
            baseline_prompt  = system_prompt,
            improved_prompt  = system_prompt,
            baseline_report  = baseline_report,
            improved_report  = baseline_report,
            variant_results  = [],
        )

    # ── 3. Pick the best ────────────────────────────────────────────────────────
    best = variants[0]   # PromptRepair sorts best-first (highest cai_score = lowest Strain)
    improved_prompt = best.improved_prompt
    improved_report = best.report
    improved_strain = improved_report.cai_strain or baseline_strain
    strain_delta    = round(improved_strain - baseline_strain, 4)

    if verbose:
        arrow = "↓" if strain_delta < 0 else "↑"
        print(f"  [improve] best variant: Strain {baseline_strain:.3f} {arrow} {improved_strain:.3f}")

    result = ImprovementResult(
        baseline_strain  = baseline_strain,
        improved_strain  = improved_strain,
        strain_delta     = strain_delta,
        target_strain    = target_strain,
        target_met       = improved_strain <= target_strain,
        method           = method,
        baseline_prompt  = system_prompt,
        improved_prompt  = improved_prompt,
        baseline_report  = baseline_report,
        improved_report  = improved_report,
        variant_results  = variants,
    )

    # ── 4. Fine-tune scaffold ───────────────────────────────────────────────────
    if method == "finetune":
        # Always write the JSONL — useful artifact even if we don't submit.
        ft_jsonl_path = _write_finetune_jsonl(
            report        = baseline_report,
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


def _submit_finetune_job(
    jsonl_path: str,
    provider:   str,
    base_model: Optional[str],
    verbose:    bool,
) -> Optional[str]:
    """
    Submit a fine-tuning job to the configured provider.

    Stubbed. Wired only as a scaffold today — the production integration with
    OpenAI's fine-tuning endpoint will live here. Returns the job ID once
    the submission lands; until then raises NotImplementedError so callers
    using enable_finetune=True see a clear "not yet" rather than a silent
    no-op.
    """
    if provider != "openai":
        raise NotImplementedError(f"fine-tune provider '{provider}' is not yet supported")
    # TODO: wire openai.fine_tuning.jobs.create here once the upload+wait flow
    # is hardened. The shape is:
    #
    #   from openai import OpenAI
    #   client = OpenAI()
    #   training_file = client.files.create(file=open(jsonl_path), purpose="fine-tune")
    #   job = client.fine_tuning.jobs.create(
    #       training_file=training_file.id,
    #       model=base_model or "gpt-4o-mini-2024-07-18",
    #   )
    #   return job.id
    raise NotImplementedError(
        "OpenAI fine-tuning submission is not yet wired. "
        "The JSONL is written and ready to upload via the OpenAI dashboard "
        "or `openai api fine_tuning.jobs.create` manually. "
        "Full automation lands in 1.4.0."
    )
