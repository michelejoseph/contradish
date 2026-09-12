"""
resolution.py: The Resolution Operator.

Every consistency/eval tool on the market today, including the rest of
contradish before this module, stops at detection: it tells you that two
things the model said don't fit together. DistinctionProber tells you a
distinction collapsed. PromptPex and Microsoft's Agent-Pex generate test
cases that catch a violation. Anthropic's Petri and OpenAI's Evals API flag
inconsistent or unsafe behavior. Giskard, TruLens-class guardrails, and
Contextual AI's LMUnit all score or classify -- none of them ask the next
question: why don't these two answers fit together, and what is the one
piece of context that would make both of them correct at once?

Most apparent contradictions are not contradictions. They are two correct
answers to two different hidden situations that got collapsed into one
visible question because nobody said out loud which situation applies. The
RLHF literature on "hidden context" (Siththaranjan et al., "Distributional
Preference Learning," arXiv 2312.08358) diagnoses this exact mechanism for
preference conflicts -- an unobserved variable (annotator identity, the
situation each rater assumed) gets silently averaged away -- but it stops at
diagnosis. It does not attempt to discover, per instance, what that variable
is, or to prove that surfacing it changes anything.

This module does both, and it validates the second half experimentally
rather than asserting it. The internal research line this ships from found
that surfacing a hidden disambiguating variable measurably calmed a model's
fine-tuning dynamics on an unrelated constraint (the first cleanly confirmed
pre-registered prediction in that research program). That result was toy
scale and is not re-tested here. What this module does is the black-box
half of the same idea, applied at the level contradish already operates at:
given a distinction a real model is observed to collapse, actively search
for the disambiguating condition, PROVE it is causal (not just plausible)
with a real flip test, and only then offer it as a fix -- with the
measured before/after hold rate attached, not a guess dressed up as one.

Algorithm ("the resolution operator")
--------------------------------------
  1. PROPOSE   An independent LLM call proposes candidate hidden variables
               that would make both commit_a and commit_b correct at once,
               each phrased as two opposite poles (a fact that, if true,
               points to state A's answer, and a fact that, if true,
               instead points to state B's).

  2. PROBE     For each candidate, four counterfactual questions are built:
               question_a / question_b each with their OWN pole asserted
               (the direct condition), and question_a / question_b each
               with the OTHER pole asserted (the swapped condition).

  3. SCORE     causal_effect_size averages two things: does asserting the
               matching pole reproduce the pair's own correct commitment
               (direct_match_rate), and does asserting the OPPOSITE pole
               flip the answer to the OTHER commitment (flip_rate). A
               candidate the model ignores, or that biases the answer
               toward one side regardless of which pole is asserted, scores
               low on at least one half -- this is a real causal test, not
               a plausibility rating.

  4. VALIDATE  The highest-scoring candidate that clears resolve_threshold
               is turned into a one-line system-prompt patch, which is then
               tested on FRESH, unconditioned probes of the ORIGINAL
               question_a/question_b (no pole asserted in the question
               itself). Only if the measured hold rate actually improves is
               the candidate reported as resolved.

Honesty constraint: like the rest of contradish, this module reports a
negative result plainly. If no proposed candidate reproduces both poles
and flips under swap, or if the winning candidate's patch does not measure
out to a real hold-rate improvement, `resolved` is False and
`system_prompt_patch` is None -- a plausible-sounding guess that didn't
pan out is not shipped as a fix.

Usage::

    from contradish import BUILTIN_DISTINCTION_PAIRS
    from contradish.resolution import discover_resolution
    from contradish.llm import LLMClient

    pair = BUILTIN_DISTINCTION_PAIRS["medication"][0]
    result = discover_resolution(
        pair=pair,
        model_fn=my_model,
        commitment_extractor=my_extractor,
        llm=LLMClient(),
    )
    print(result.report())
    if result.resolved:
        print(result.system_prompt_patch)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

from .distinction import DistinctionPair, DistinctionLossMap, ModelFn

RESOLUTION_REPORT_SCHEMA_VERSION = "1.0"


_PROPOSE_PROMPT = """A model was asked two related questions and gave answers that collapsed \
into treating them the same way, when they should have been treated differently.

Situation A: {label_a}
Question A: {question_a}
Correct handling for A: {commit_a}

Situation B: {label_b}
Question B: {question_b}
Correct handling for B: {commit_b}

Propose {n} DIFFERENT hidden variables -- facts that are not stated in \
either question -- such that if the model had been told which pole of the \
variable applied, it would have had no trouble giving the correct, \
distinguishable answer for each situation. Each variable must have two \
opposite poles: one that points toward situation A's correct handling, and \
one that points toward situation B's.

Respond with ONLY a JSON object, no markdown, no prose, in this exact shape:

{{"candidates": [
  {{"condition": "short name of the hidden variable",
    "rationale": "one sentence on why this would disambiguate the two situations",
    "pole_a_statement": "a short factual clause asserting the pole that supports situation A's handling",
    "pole_b_statement": "a short factual clause asserting the pole that supports situation B's handling"}}
]}}"""


@dataclass
class ResolutionCandidate:
    """
    One proposed hidden variable, with its measured causal effect.

    direct_match_rate
        Fraction of direct probes (question_a + pole_a_statement,
        question_b + pole_b_statement) whose extracted commitment matched
        the pair's own commit_a / commit_b.
    flip_rate
        Fraction of swapped probes (question_a + pole_b_statement,
        question_b + pole_a_statement) whose extracted commitment matched
        the OPPOSITE commitment -- proof the pole is doing causal work,
        not just that stating any fact nudges the model toward one answer.
    causal_effect_size
        Mean of direct_match_rate and flip_rate. 1.0 means the pole fully
        and reversibly controls which commitment the model gives.
    """
    condition:          str
    rationale:          str
    pole_a_statement:   str
    pole_b_statement:   str
    direct_match_rate:  float
    flip_rate:          float
    causal_effect_size: float
    n_probes:           int


@dataclass
class ResolutionResult:
    """
    Result of running the resolution operator on one collapsing distinction.

    resolved
        True only if the winning candidate cleared resolve_threshold on the
        causal flip test AND its system-prompt patch measurably improved
        the hold rate on fresh, unconditioned probes. False (with
        system_prompt_patch=None) is a reportable, honest outcome.
    """
    pair_id:              str
    description:          str
    label_a:              str
    label_b:              str
    baseline_hold_rate:   Optional[float]
    candidates:           list[ResolutionCandidate] = field(default_factory=list)
    best:                 Optional[ResolutionCandidate] = None
    resolved:             bool = False
    validated_hold_rate:  Optional[float] = None
    system_prompt_patch:  Optional[str] = None

    def summary(self) -> str:
        if self.resolved and self.best is not None:
            return (
                f"{self.pair_id}: RESOLVED via '{self.best.condition}' "
                f"(causal_effect_size={self.best.causal_effect_size:.2f}, "
                f"hold rate {self.baseline_hold_rate:.0%} -> {self.validated_hold_rate:.0%})"
            )
        n = len(self.candidates)
        best_score = f"{self.best.causal_effect_size:.2f}" if self.best else "n/a"
        return (
            f"{self.pair_id}: not resolved "
            f"({n} candidate(s) tested, best causal_effect_size={best_score})"
        )

    def report(self) -> str:
        W   = 72
        sep = "-" * W
        bar = lambda r, w=16: "#" * round(r * w) + "." * (w - round(r * w))

        if self.baseline_hold_rate is not None:
            baseline_line = f"  baseline hold rate: {self.baseline_hold_rate:.0%}"
        else:
            baseline_line = "  baseline hold rate: n/a"

        lines = [
            "",
            f"  RESOLUTION OPERATOR  ·  {self.pair_id}",
            sep,
            f"  {self.description}",
            baseline_line,
            "",
            "  CANDIDATES  (ranked by causal effect size)",
            "",
        ]
        for c in self.candidates:
            marker = "*" if (self.best is c) else " "
            lines.append(
                f"  {marker} {c.condition:<40}  effect={c.causal_effect_size:.2f}  "
                f"{bar(c.causal_effect_size)}  "
                f"(direct={c.direct_match_rate:.0%}, flip={c.flip_rate:.0%})"
            )
        if not self.candidates:
            lines.append("  (no candidates proposed)")

        lines.append("")
        if self.resolved:
            lines += [
                f"  RESOLVED: {self.best.condition}",
                f"    {self.best.rationale}",
                f"    hold rate under patch: {self.validated_hold_rate:.0%} "
                f"(baseline {self.baseline_hold_rate:.0%})",
                "",
                "  Suggested system prompt patch:",
                f"    {self.system_prompt_patch}",
            ]
        else:
            lines.append(
                "  NOT RESOLVED -- no candidate both reproduced the correct "
                "commitment under its own pole and flipped under the "
                "opposite pole strongly enough, or the resulting patch did "
                "not measurably improve the hold rate."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema_version": RESOLUTION_REPORT_SCHEMA_VERSION,
            "pair_id": self.pair_id,
            "description": self.description,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "baseline_hold_rate": (
                round(self.baseline_hold_rate, 4) if self.baseline_hold_rate is not None else None
            ),
            "candidates": [
                {
                    "condition": c.condition,
                    "rationale": c.rationale,
                    "pole_a_statement": c.pole_a_statement,
                    "pole_b_statement": c.pole_b_statement,
                    "direct_match_rate": round(c.direct_match_rate, 4),
                    "flip_rate": round(c.flip_rate, 4),
                    "causal_effect_size": round(c.causal_effect_size, 4),
                    "n_probes": c.n_probes,
                }
                for c in self.candidates
            ],
            "best_condition": self.best.condition if self.best else None,
            "resolved": self.resolved,
            "validated_hold_rate": (
                round(self.validated_hold_rate, 4) if self.validated_hold_rate is not None else None
            ),
            "system_prompt_patch": self.system_prompt_patch,
        }


def _default_candidate_proposer(llm) -> Callable[[DistinctionPair, int], list[dict]]:
    """
    Default candidate_proposer(pair, n) -> list[dict] for discover_resolution
    when the caller doesn't supply their own. Uses the configured judge
    model to propose n hidden variables in one call. Like every other
    default_* factory in this package, this is a judge call and inherits the
    judge's own noise; write your own candidate_proposer for anything you
    plan to rely on.
    """
    def propose(pair: DistinctionPair, n: int) -> list[dict]:
        prompt = _PROPOSE_PROMPT.format(
            label_a=pair.label_a, question_a=pair.question_a, commit_a=pair.commit_a,
            label_b=pair.label_b, question_b=pair.question_b, commit_b=pair.commit_b,
            n=n,
        )
        data = llm.complete_json(prompt)
        raw = data.get("candidates", []) if isinstance(data, dict) else []
        candidates = []
        for c in raw[:n]:
            if not isinstance(c, dict):
                continue
            if not all(k in c for k in ("condition", "rationale", "pole_a_statement", "pole_b_statement")):
                continue
            candidates.append(c)
        return candidates
    return propose


def _measure_hold_rate(
    model_fn:              ModelFn,
    commitment_extractor:  Callable[[str, str], str],
    system_prompt:         str,
    question_a:            str,
    question_b:            str,
    n_samples:             int,
) -> float:
    holds = []
    for _ in range(max(1, n_samples)):
        ans_a = model_fn(system_prompt, question_a)
        ans_b = model_fn(system_prompt, question_b)
        com_a = commitment_extractor(question_a, ans_a)
        com_b = commitment_extractor(question_b, ans_b)
        holds.append(com_a != com_b)
    return sum(holds) / len(holds)


def discover_resolution(
    pair:                  DistinctionPair,
    model_fn:              ModelFn,
    commitment_extractor:  Callable[[str, str], str],
    llm=None,
    candidate_proposer:    Optional[Callable[[DistinctionPair, int], list[dict]]] = None,
    system_prompt:         str = "",
    n_candidates:          int = 3,
    validation_samples:    int = 2,
    baseline_hold_rate:    Optional[float] = None,
    resolve_threshold:     float = 0.75,
    improvement_threshold: float = 0.15,
    verbose:               bool = False,
) -> ResolutionResult:
    """
    Run the resolution operator on one distinction pair a real model is
    observed (or suspected) to collapse.

    Args:
        pair:                  the DistinctionPair being resolved.
        model_fn:              (system_prompt, question) -> answer.
        commitment_extractor:  (question, answer) -> commitment string.
        llm:                   an LLMClient, used to build the default
                               candidate_proposer. Required unless you pass
                               your own candidate_proposer.
        candidate_proposer:    (pair, n) -> list of {condition, rationale,
                               pole_a_statement, pole_b_statement} dicts.
                               Pass your own to avoid the judge-call default.
        system_prompt:         passed to model_fn on every call.
        n_candidates:          how many candidate hidden variables to propose.
        validation_samples:    probes per candidate per direction, and probes
                               used to measure baseline / patched hold rate.
        baseline_hold_rate:    reuse an already-measured hold rate (e.g. from
                               a DistinctionProfile.overall_hold_rate) instead
                               of remeasuring it here.
        resolve_threshold:     minimum causal_effect_size for a candidate to
                               be eligible to resolve the pair.
        improvement_threshold: minimum hold-rate improvement (patched minus
                               baseline) required to call the pair resolved.
        verbose:               print progress.

    Returns:
        A ResolutionResult.
    """
    if candidate_proposer is None:
        if llm is None:
            raise ValueError(
                "discover_resolution needs either an llm (LLMClient) to build "
                "the default candidate_proposer, or a candidate_proposer of "
                "your own."
            )
        candidate_proposer = _default_candidate_proposer(llm)

    if baseline_hold_rate is None:
        if verbose:
            print(f"  Measuring baseline hold rate: {pair.pair_id}")
        baseline_hold_rate = _measure_hold_rate(
            model_fn, commitment_extractor, system_prompt,
            pair.question_a, pair.question_b, validation_samples,
        )

    if verbose:
        print(f"  Proposing candidates: {pair.pair_id}")
    raw_candidates = candidate_proposer(pair, n_candidates)

    candidates: list[ResolutionCandidate] = []
    for raw in raw_candidates:
        condition        = str(raw["condition"])
        rationale        = str(raw["rationale"])
        pole_a_statement = str(raw["pole_a_statement"])
        pole_b_statement = str(raw["pole_b_statement"])

        if verbose:
            print(f"    Probing candidate: {condition}")

        direct_hits = []
        flip_hits   = []
        for _ in range(max(1, validation_samples)):
            q_a_direct  = f"{pair.question_a} {pole_a_statement}"
            q_b_direct  = f"{pair.question_b} {pole_b_statement}"
            q_a_swapped = f"{pair.question_a} {pole_b_statement}"
            q_b_swapped = f"{pair.question_b} {pole_a_statement}"

            com_a_direct  = commitment_extractor(q_a_direct,  model_fn(system_prompt, q_a_direct))
            com_b_direct  = commitment_extractor(q_b_direct,  model_fn(system_prompt, q_b_direct))
            com_a_swapped = commitment_extractor(q_a_swapped, model_fn(system_prompt, q_a_swapped))
            com_b_swapped = commitment_extractor(q_b_swapped, model_fn(system_prompt, q_b_swapped))

            direct_hits.append(com_a_direct == pair.commit_a)
            direct_hits.append(com_b_direct == pair.commit_b)
            flip_hits.append(com_a_swapped == pair.commit_b)
            flip_hits.append(com_b_swapped == pair.commit_a)

        direct_match_rate = sum(direct_hits) / len(direct_hits)
        flip_rate         = sum(flip_hits) / len(flip_hits)
        causal_effect_size = statistics.mean([direct_match_rate, flip_rate])

        candidates.append(ResolutionCandidate(
            condition          = condition,
            rationale          = rationale,
            pole_a_statement   = pole_a_statement,
            pole_b_statement   = pole_b_statement,
            direct_match_rate  = direct_match_rate,
            flip_rate          = flip_rate,
            causal_effect_size = causal_effect_size,
            n_probes           = len(direct_hits) + len(flip_hits),
        ))

    candidates.sort(key=lambda c: -c.causal_effect_size)
    best = candidates[0] if candidates else None

    resolved             = False
    validated_hold_rate  = None
    system_prompt_patch  = None

    if best is not None and best.causal_effect_size >= resolve_threshold:
        system_prompt_patch = (
            f"When distinguishing '{pair.label_a}' from '{pair.label_b}', "
            f"first check: {best.condition}. If {best.pole_a_statement.rstrip('.')}, "
            f"{pair.commit_a}. If {best.pole_b_statement.rstrip('.')}, {pair.commit_b}."
        )
        patched_prompt = (system_prompt + "\n\n" + system_prompt_patch).strip()
        if verbose:
            print(f"  Validating patch for: {best.condition}")
        validated_hold_rate = _measure_hold_rate(
            model_fn, commitment_extractor, patched_prompt,
            pair.question_a, pair.question_b, validation_samples,
        )
        improved = validated_hold_rate - baseline_hold_rate >= improvement_threshold
        resolved = improved
        if not resolved:
            system_prompt_patch = None

    return ResolutionResult(
        pair_id             = pair.pair_id,
        description         = pair.description,
        label_a             = pair.label_a,
        label_b             = pair.label_b,
        baseline_hold_rate  = baseline_hold_rate,
        candidates          = candidates,
        best                = best,
        resolved            = resolved,
        validated_hold_rate = validated_hold_rate,
        system_prompt_patch = system_prompt_patch,
    )


def discover_resolutions_for_loss_map(
    loss_map:              DistinctionLossMap,
    pairs:                 list[DistinctionPair],
    model_fn:              ModelFn,
    commitment_extractor:  Callable[[str, str], str],
    llm=None,
    candidate_proposer:    Optional[Callable[[DistinctionPair, int], list[dict]]] = None,
    system_prompt:         str = "",
    collapse_threshold:    float = 0.3,
    verbose:               bool = False,
    **kwargs,
) -> list[ResolutionResult]:
    """
    Run the resolution operator on every pair in `pairs` whose measured
    profile in `loss_map` collapsed at least `collapse_threshold` of the
    time -- the natural follow-up to a DistinctionProber.measure() run:
    don't just report which distinctions collapsed, try to fix the ones
    that did.
    """
    results = []
    for pair in pairs:
        profile = loss_map.profiles.get(pair.pair_id)
        if profile is None or profile.collapse_rate() < collapse_threshold:
            continue
        results.append(discover_resolution(
            pair                 = pair,
            model_fn             = model_fn,
            commitment_extractor = commitment_extractor,
            llm                  = llm,
            candidate_proposer   = candidate_proposer,
            system_prompt        = system_prompt,
            baseline_hold_rate   = profile.overall_hold_rate,
            verbose              = verbose,
            **kwargs,
        ))
    return results
