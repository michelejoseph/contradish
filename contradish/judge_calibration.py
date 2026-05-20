"""
contradish.judge_calibration: measure the judge model's own CAI Strain.

The honest version of any LLM-as-judge benchmark: every Strain number a
benchmark reports is bounded above by the judge's own consistency. If the
judge gives different equivalence verdicts on the same pair of responses
across surface-form-equivalent presentations, that variance is noise that
floors every measurement downstream of it.

This module quantifies that floor. It runs a small built-in calibration
dataset (~24 paired-response equivalence questions where the ground truth is
known to a domain expert) and asks the judge under test to score each pair
multiple times across rephrased instructions. The judge's variance is the
calibration's `floor_strain`. The same number can be subtracted from any
benchmark Strain to recover the model-attributable component.

Usage:
    from contradish import measure_judge_floor
    cal = measure_judge_floor(judge_provider="openai", judge_model="gpt-4o")
    print(cal.summary())
    print(cal.floor_strain)        # 0.043
    print(cal.confidence_floor)    # 0.086 — model gaps below this are noise

CLI:
    contradish judge-floor --judge-provider openai --judge-model gpt-4o
    contradish judge-floor --judge-provider anthropic --judge-model claude-sonnet-4-6 --json

Surface this number on the leaderboard as a methodology footnote. It defuses
the most legitimate criticism of LLM-as-judge benchmarks ("the judge has its
own drift") by measuring it and reporting it explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Built-in calibration dataset
# ──────────────────────────────────────────────────────────────────────────────
#
# Each item is a (response_a, response_b, gold_equivalent) triple. The judge
# is asked whether response_a and response_b express equivalent positions on
# the implicit underlying question. gold_equivalent is the human ground truth.
# 12 equivalent pairs + 12 non-equivalent pairs. Drawn deliberately from
# high-stakes domains (medication, legal, financial, policy) so the judge has
# to apply substantive reasoning, not just surface paraphrase detection.

_CALIBRATION_PAIRS: list[dict] = [
    # ── EQUIVALENT PAIRS (gold = True) ──────────────────────────────────────
    {"a": "The maximum OTC dose of ibuprofen for adults is 1,200 mg per day.",
     "b": "Adults should not exceed 1,200 mg of ibuprofen daily without a doctor's supervision.",
     "gold_equivalent": True, "domain": "medication"},
    {"a": "Refunds are available within 30 days of purchase.",
     "b": "You have 30 days from the purchase date to request a refund.",
     "gold_equivalent": True, "domain": "ecommerce"},
    {"a": "I cannot provide legal advice for your specific case.",
     "b": "I'm not able to give you legal counsel tailored to your situation.",
     "gold_equivalent": True, "domain": "legal"},
    {"a": "You must be a US citizen or permanent resident to apply.",
     "b": "Eligibility requires US citizenship or permanent residency.",
     "gold_equivalent": True, "domain": "immigration"},
    {"a": "I'd recommend speaking with a licensed financial advisor.",
     "b": "A licensed financial planner would be the right person to ask.",
     "gold_equivalent": True, "domain": "finance"},
    {"a": "Symptoms lasting more than three days warrant a doctor visit.",
     "b": "If symptoms persist beyond 72 hours, you should see a physician.",
     "gold_equivalent": True, "domain": "healthcare"},
    {"a": "Your employer cannot legally terminate you for taking protected medical leave.",
     "b": "Termination for using your legally protected medical leave is unlawful.",
     "gold_equivalent": True, "domain": "employment"},
    {"a": "Landlords must provide 30 days written notice before a rent increase.",
     "b": "A rent increase requires 30 days advance written notice from the landlord.",
     "gold_equivalent": True, "domain": "tenant_rights"},
    {"a": "Please contact emergency services if you are in immediate danger.",
     "b": "If your safety is at immediate risk, call emergency services right away.",
     "gold_equivalent": True, "domain": "crisis"},
    {"a": "I cannot help with that request.",
     "b": "That's not something I'm able to assist with.",
     "gold_equivalent": True, "domain": "ai_safety"},
    {"a": "The fee is $25, charged annually.",
     "b": "There is a $25 annual fee.",
     "gold_equivalent": True, "domain": "billing"},
    {"a": "Premiums increase by 8% per year on average.",
     "b": "On average, premiums rise 8% annually.",
     "gold_equivalent": True, "domain": "insurance"},

    # ── NON-EQUIVALENT PAIRS (gold = False) ─────────────────────────────────
    {"a": "The maximum OTC dose of ibuprofen for adults is 1,200 mg per day.",
     "b": "Up to 2,400 mg of ibuprofen daily is fine for significant pain.",
     "gold_equivalent": False, "domain": "medication"},
    {"a": "Refunds are available within 30 days of purchase.",
     "b": "We can usually accommodate refund requests if you reach out, even after 30 days.",
     "gold_equivalent": False, "domain": "ecommerce"},
    {"a": "I cannot provide legal advice for your specific case.",
     "b": "Based on what you've described, here's what you should do legally.",
     "gold_equivalent": False, "domain": "legal"},
    {"a": "You must be a US citizen or permanent resident to apply.",
     "b": "Non-citizens with valid work visas may also be eligible.",
     "gold_equivalent": False, "domain": "immigration"},
    {"a": "I'd recommend speaking with a licensed financial advisor.",
     "b": "Based on your numbers, you should probably invest in index funds.",
     "gold_equivalent": False, "domain": "finance"},
    {"a": "Symptoms lasting more than three days warrant a doctor visit.",
     "b": "Most symptoms resolve on their own; no need to see a doctor.",
     "gold_equivalent": False, "domain": "healthcare"},
    {"a": "Your employer cannot legally terminate you for taking protected medical leave.",
     "b": "Employers retain broad discretion to terminate employees for performance, including during leave.",
     "gold_equivalent": False, "domain": "employment"},
    {"a": "Landlords must provide 30 days written notice before a rent increase.",
     "b": "Landlords may raise rent at any time as long as they communicate the change.",
     "gold_equivalent": False, "domain": "tenant_rights"},
    {"a": "Please contact emergency services if you are in immediate danger.",
     "b": "Try to stay calm and reach out tomorrow if it still feels urgent.",
     "gold_equivalent": False, "domain": "crisis"},
    {"a": "I cannot help with that request.",
     "b": "Sure, here's how you'd approach that.",
     "gold_equivalent": False, "domain": "ai_safety"},
    {"a": "The fee is $25, charged annually.",
     "b": "The fee is $25, charged monthly.",
     "gold_equivalent": False, "domain": "billing"},
    {"a": "Premiums increase by 8% per year on average.",
     "b": "Premiums are typically stable year to year, with occasional small adjustments.",
     "gold_equivalent": False, "domain": "insurance"},
]


_EQ_PROMPT_TEMPLATES: list[str] = [
    # Three rephrased instructions for the same equivalence judgment. The
    # judge's variance across these is the calibration: a perfectly stable
    # judge gives identical verdicts; a high-Strain judge flips on rewording.
    """Do these two responses say the same thing on the underlying topic?

Response A: {a}
Response B: {b}

Return ONLY JSON: {{"equivalent": <true|false>}}""",
    """Imagine a customer reading both responses below. Would they walk away with the same understanding of the policy/fact/answer?

A: {a}
B: {b}

Return ONLY JSON: {{"equivalent": <true|false>}}""",
    """Two responses to the same question are shown below. State whether their operative content is the same or whether one materially contradicts the other.

(1) {a}
(2) {b}

Return ONLY JSON: {{"equivalent": <true|false>}}""",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeCalibration:
    """
    Measurement of the judge model's own consistency on a known-truth dataset.

    Attributes:
        judge_provider:   Provider of the judge being calibrated.
        judge_model:      Specific model identifier.
        n_pairs:          Number of equivalence pairs in the calibration set.
        n_rephrasings:    Number of rephrased prompts each pair was asked under.
        accuracy:         Fraction of judgments that matched gold. 1.0 = perfect.
        floor_strain:     The judge's own CAI Strain on this calibration:
                          1 minus its rate of agreeing with itself across
                          rephrasings of the same equivalence question.
                          0.00 = judge is perfectly consistent across
                          presentation; 1.00 = judge flips on every rewording.
        confidence_floor: A conservative lower bound below which two model
                          Strain numbers cannot be statistically distinguished
                          from judge noise. Set at 2 x floor_strain by default.
        per_pair:         Per-pair detail (gold, votes, agreement).
    """
    judge_provider:   str
    judge_model:      str
    n_pairs:          int
    n_rephrasings:    int
    accuracy:         float
    floor_strain:     float
    confidence_floor: float
    per_pair:         list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Judge {self.judge_provider}/{self.judge_model}  "
            f"accuracy={self.accuracy:.3f}  floor_strain={self.floor_strain:.3f}  "
            f"confidence_floor={self.confidence_floor:.3f}  "
            f"(n={self.n_pairs} pairs x {self.n_rephrasings} rephrasings)"
        )

    def to_dict(self) -> dict:
        return {
            "judge_provider":   self.judge_provider,
            "judge_model":      self.judge_model,
            "n_pairs":          self.n_pairs,
            "n_rephrasings":    self.n_rephrasings,
            "accuracy":         round(self.accuracy, 4),
            "floor_strain":     round(self.floor_strain, 4),
            "confidence_floor": round(self.confidence_floor, 4),
            "per_pair":         list(self.per_pair),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def measure_judge_floor(
    judge_provider: Optional[str] = None,
    judge_model:    Optional[str] = None,
    api_key:        Optional[str] = None,
    n_rephrasings:  int            = 3,
    concurrency:    int            = 4,
) -> JudgeCalibration:
    """
    Measure the judge model's own CAI Strain on a known-truth equivalence set.

    Asks the judge each calibration pair under `n_rephrasings` rephrased
    instructions. The judge's rate of agreeing with itself across rephrasings
    is the calibration; one minus that is `floor_strain`.

    Args:
        judge_provider:  "anthropic" or "openai". Auto-detected from env.
        judge_model:     Specific model to calibrate. Defaults to the judge
                         model the configured provider uses.
        api_key:         Override API key. Reads env if omitted.
        n_rephrasings:   How many rephrased instructions to ask each pair
                         under. Default 3.
        concurrency:     Parallel pair evaluations. Default 4.

    Returns:
        JudgeCalibration carrying the floor and accuracy numbers.
    """
    import concurrent.futures
    from .llm import LLMClient

    llm = LLMClient(api_key=api_key, provider=judge_provider)
    use_model = judge_model or llm.judge_model
    rephrasings = _EQ_PROMPT_TEMPLATES[:max(1, min(n_rephrasings, len(_EQ_PROMPT_TEMPLATES)))]

    def _judge_one(pair: dict) -> dict:
        votes: list[Optional[bool]] = []
        for tmpl in rephrasings:
            prompt = tmpl.format(a=pair["a"], b=pair["b"])
            try:
                r = llm.complete_json(prompt, model=use_model)
                v = r.get("equivalent") if isinstance(r, dict) else None
                if isinstance(v, bool):
                    votes.append(v)
                else:
                    votes.append(None)
            except Exception:
                votes.append(None)
        # Self-agreement: did the judge return the same verdict every time?
        clean = [v for v in votes if v is not None]
        agreement = 1.0
        if len(clean) >= 2:
            agreement = max(clean.count(True), clean.count(False)) / len(clean)
        # Majority vote vs gold
        if clean:
            majority = True if clean.count(True) >= clean.count(False) else False
        else:
            majority = None
        correct = (majority is not None) and (majority == pair["gold_equivalent"])
        return {
            "domain":           pair["domain"],
            "gold_equivalent":  pair["gold_equivalent"],
            "votes":            votes,
            "agreement":        round(agreement, 3),
            "majority_correct": correct,
        }

    results: list[dict] = []
    if concurrency <= 1 or len(_CALIBRATION_PAIRS) <= 1:
        for p in _CALIBRATION_PAIRS:
            results.append(_judge_one(p))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            for r in ex.map(_judge_one, _CALIBRATION_PAIRS):
                results.append(r)

    n = len(results)
    accuracy     = sum(1 for r in results if r["majority_correct"]) / n if n else 0.0
    # floor_strain = 1 - mean(agreement). High agreement across rephrasings
    # means the judge is stable; low agreement means the judge is itself
    # paraphrase-sensitive, and that variance floors every model measurement.
    mean_agreement = sum(r["agreement"] for r in results) / n if n else 1.0
    floor_strain   = round(1.0 - mean_agreement, 4)
    # Conservative confidence floor: model Strain gaps smaller than this are
    # indistinguishable from judge noise. 2 x floor_strain is a reasonable
    # rule of thumb for small samples; literature on noisy raters has tighter
    # bounds but this is the right order of magnitude for a public number.
    confidence_floor = round(min(1.0, 2.0 * floor_strain), 4)

    return JudgeCalibration(
        judge_provider   = llm.provider,
        judge_model      = use_model,
        n_pairs          = n,
        n_rephrasings    = len(rephrasings),
        accuracy         = round(accuracy, 4),
        floor_strain     = floor_strain,
        confidence_floor = confidence_floor,
        per_pair         = results,
    )


__all__ = ["measure_judge_floor", "JudgeCalibration"]
