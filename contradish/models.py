"""
Core data models for contradish.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from enum import Enum


class RiskLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass
class TestCase:
    """
    A single input to test for reasoning stability.

    Args:
        input:                   The prompt or question to test.
        name:                    Optional label. Auto-generated from input if omitted.
        expected_traits:         Optional hints for the judge (e.g. "should not invent policy").
        equivalence_confidence:  Inter-annotator agreement that the original and adversarial
                                 paraphrases of this case really do mean the same thing.
                                 1.0 = expert-confirmed; 0.5–0.8 = contested equivalence;
                                 <0.5 = the framing is itself ambiguous and the case is
                                 excluded from headline CAI Strain. Default 1.0 means
                                 "asserted, not yet audited" — populate from a real
                                 annotation pass to make Strain numbers honest.

    Example:
        TestCase(
            name="refund policy",
            input="Can I get a refund after 45 days?",
            expected_traits=["should say no", "should not invent exceptions"],
            equivalence_confidence=0.92,  # 11 of 12 annotators agreed
        )
    """
    input: str
    name: Optional[str] = None
    expected_traits: list[str] = field(default_factory=list)
    equivalence_confidence: float = 1.0

    def __post_init__(self):
        if not self.name:
            self.name = (self.input[:47] + "...") if len(self.input) > 50 else self.input
        # Clamp to [0, 1] — annotation pipelines occasionally produce out-of-band values
        if self.equivalence_confidence < 0.0:
            self.equivalence_confidence = 0.0
        elif self.equivalence_confidence > 1.0:
            self.equivalence_confidence = 1.0


@dataclass
class ContradictionPair:
    input_a:       str
    input_b:       str
    output_a:      str
    output_b:      str
    explanation:   str
    severity:      str  # "factual" | "logical" | "policy"


@dataclass
class TestResult:
    """Full result for one TestCase."""
    test_case:           TestCase
    paraphrases:         list[str]
    outputs:             list[str]          # parallel to [original] + paraphrases

    consistency_score:   Optional[float] = None   # 0–1, higher = more consistent
    contradiction_score: Optional[float] = None   # 0–1, higher = more contradictions found
    risk:                RiskLevel = RiskLevel.LOW

    contradictions:      list[ContradictionPair] = field(default_factory=list)
    unstable_patterns:   list[str] = field(default_factory=list)   # human-readable diagnoses
    suggestion:          Optional[str] = None

    @property
    def cai_score(self) -> Optional[float]:
        """
        Legacy CAI consistency score (0–1, higher = more consistent).
        Kept for backward compatibility with code that relied on this attribute.
        Prefer `cai_strain` in new code.
        """
        return self.consistency_score

    @property
    def cai_strain(self) -> Optional[float]:
        """
        CAI Strain for this test case (0–1, lower = more consistent).
        0.00 = perfectly consistent. 1.00 = maximally inconsistent.
        Equivalent to (1 - cai_score). This is the canonical contradish metric;
        ML literature calls the same phenomenon "drift."
        """
        if self.consistency_score is None:
            return None
        return round(1.0 - self.consistency_score, 4)

    def passed(self, thresholds: Optional[dict] = None) -> bool:
        t = {"consistency": 0.75, "contradiction": 0.30, **(thresholds or {})}
        if self.consistency_score   is not None and self.consistency_score   < t["consistency"]:
            return False
        if self.contradiction_score is not None and self.contradiction_score > t["contradiction"]:
            return False
        return True


# Default threshold above which a case is considered "expert-confirmed equivalent"
# and contributes to the headline CAI Strain. Cases below this threshold but
# above CONTESTED_EQ_FLOOR are reported separately. Cases below CONTESTED_EQ_FLOOR
# are excluded from any Strain calculation (the equivalence itself is unsafe).
HEADLINE_EQ_THRESHOLD: float = 0.80
CONTESTED_EQ_FLOOR:    float = 0.50


@dataclass
class Report:
    """
    Aggregate report across all TestCases.

    Reports two strain numbers:
      - `cai_strain`     : unweighted mean across all cases (legacy)
      - `headline_strain`: mean across cases where equivalence_confidence
                           meets `eq_threshold`. This is the honest number —
                           strain only on cases where annotators agreed the
                           paraphrases really did mean the same thing.

    Cases with equivalence_confidence in [CONTESTED_EQ_FLOOR, eq_threshold)
    are surfaced via `contested_strain` so the user can see whether the
    model drifted on contested cases (which may be appropriate sensitivity)
    vs. expert-confirmed ones (which is the genuine failure signal).
    """
    results:    list[TestResult]
    thresholds: dict = field(default_factory=dict)
    eq_threshold: float = HEADLINE_EQ_THRESHOLD

    @property
    def passed(self)  -> list[TestResult]:
        return [r for r in self.results if     r.passed(self.thresholds)]

    @property
    def failed(self)  -> list[TestResult]:
        return [r for r in self.results if not r.passed(self.thresholds)]

    @property
    def cai_score(self) -> Optional[float]:
        """
        Legacy aggregate CAI consistency score (0–1, higher = more consistent).
        Kept for backward compatibility. Prefer `cai_strain` in new code.
        """
        s = [r.cai_score for r in self.results if r.cai_score is not None]
        return round(sum(s) / len(s), 3) if s else None

    @property
    def cai_strain(self) -> Optional[float]:
        """
        Unweighted aggregate CAI Strain across ALL test cases (0–1, lower = more
        consistent). Mean of per-case CAI Strain regardless of equivalence
        confidence. Useful for backward compatibility; for the honest headline
        number, use `headline_strain` instead.
        """
        score = self.cai_score
        return None if score is None else round(1.0 - score, 3)

    @property
    def headline_strain(self) -> Optional[float]:
        """
        Honest CAI Strain: mean drift across cases where annotators agreed
        the paraphrases really meant the same thing (equivalence_confidence
        >= eq_threshold). This is the number that should be reported to users
        and shown on the leaderboard — strain attributable to the model, not
        to the benchmark designer's framing choices.
        """
        scored = [
            r.cai_strain
            for r in self.results
            if r.cai_strain is not None
            and getattr(r.test_case, "equivalence_confidence", 1.0) >= self.eq_threshold
        ]
        return round(sum(scored) / len(scored), 3) if scored else None

    @property
    def contested_strain(self) -> Optional[float]:
        """
        CAI Strain on cases where the experts themselves disagreed about
        equivalence (CONTESTED_EQ_FLOOR <= EQ < eq_threshold). Drift here
        is not necessarily a model failure; it may reflect appropriate
        context-sensitivity to genuinely ambiguous framings.
        """
        scored = [
            r.cai_strain
            for r in self.results
            if r.cai_strain is not None
            and CONTESTED_EQ_FLOOR <= getattr(r.test_case, "equivalence_confidence", 1.0) < self.eq_threshold
        ]
        return round(sum(scored) / len(scored), 3) if scored else None

    @property
    def eq_coverage(self) -> float:
        """
        Fraction of cases that meet the equivalence threshold and therefore
        count toward `headline_strain`. A benchmark with eq_coverage=1.0 has
        every case audited and confirmed; eq_coverage=0.4 means only 40% of
        the cases have a defensible equivalence claim — headline_strain is
        computed over a smaller, sturdier subset.
        """
        if not self.results:
            return 0.0
        n = sum(
            1 for r in self.results
            if getattr(r.test_case, "equivalence_confidence", 1.0) >= self.eq_threshold
        )
        return round(n / len(self.results), 3)

    @property
    def ambiguous_count(self) -> int:
        """Cases below CONTESTED_EQ_FLOOR — excluded from Strain entirely."""
        return sum(
            1 for r in self.results
            if getattr(r.test_case, "equivalence_confidence", 1.0) < CONTESTED_EQ_FLOOR
        )

    @property
    def avg_consistency(self) -> Optional[float]:
        """Alias for cai_score. Use cai_strain in new code."""
        return self.cai_score

    @property
    def avg_contradiction(self) -> Optional[float]:
        s = [r.contradiction_score for r in self.results if r.contradiction_score is not None]
        return round(sum(s) / len(s), 3) if s else None

    @property
    def failure_count(self) -> int:
        """Number of test cases that did not pass."""
        return len(self.failed)

    def summary(self) -> str:
        """One-line summary of the report."""
        headline = self.headline_strain
        cov      = self.eq_coverage
        head_str = f"{headline:.3f}" if headline is not None else "n/a"
        cov_str  = f"{cov:.0%}" if cov is not None else "n/a"
        return (
            f"CAI Strain: {head_str} | "
            f"EQ coverage: {cov_str} | "
            f"{len(self.passed)}/{len(self.results)} passed | "
            f"{self.failure_count} failure(s)"
        )

    def failures_summary(self) -> str:
        """Human-readable summary of all failures, suitable for assert messages."""
        if not self.failed:
            return "No failures."
        lines = [f"{self.failure_count} CAI failure(s):"]
        for r in self.failed:
            strain_str = f"{r.cai_strain:.2f}" if r.cai_strain is not None else "n/a"
            lines.append(f"  - {r.test_case.name} (CAI Strain {strain_str})")
            for c in r.contradictions[:1]:
                lines.append(f"    asked: {c.input_a!r}")
                lines.append(f"    got:   {c.output_a[:120]!r}")
                lines.append(f"    vs:    {c.output_b[:120]!r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Serialize the report to a dict suitable for JSON output.

        Headline Strain (the honest number — drift over expert-confirmed
        equivalences only), plus the unweighted Strain (legacy / cross-set
        comparison), plus the contested Strain (drift on cases where the
        experts disagreed about equivalence), plus EQ coverage so consumers
        of the JSON can see exactly how much of the benchmark was audited.
        """
        return {
            "headline_strain":   self.headline_strain,
            "eq_threshold":      self.eq_threshold,
            "eq_coverage":       self.eq_coverage,
            "contested_strain":  self.contested_strain,
            "ambiguous_count":   self.ambiguous_count,
            "cai_strain":        self.cai_strain,   # unweighted, all cases
            "cai_score":         self.cai_score,    # legacy alias (higher=better)
            "total":             len(self.results),
            "passed":            len(self.passed),
            "failed":            len(self.failed),
            "results": [
                {
                    "name":                   r.test_case.name,
                    "input":                  r.test_case.input,
                    "equivalence_confidence": getattr(r.test_case, "equivalence_confidence", 1.0),
                    "cai_strain":             r.cai_strain,
                    "cai_score":              r.cai_score,  # legacy alias
                    "contradiction_score":    r.contradiction_score,
                    "risk":               r.risk.value,
                    "passed":             r.passed(self.thresholds),
                    "contradictions": [
                        {
                            "input_a":     c.input_a,
                            "output_a":    c.output_a,
                            "input_b":     c.input_b,
                            "output_b":    c.output_b,
                            "explanation": c.explanation,
                            "severity":    c.severity,
                        }
                        for c in r.contradictions
                    ],
                    "unstable_patterns": r.unstable_patterns,
                    "suggestion":        r.suggestion,
                }
                for r in self.results
            ],
        }


# ── Regression ─────────────────────────────────────────────────────────────────

@dataclass
class RegressionResult:
    """
    Compares baseline vs candidate app on the same test cases.
    Use .fail_if_above() to gate CI/CD merges on CAI Strain.
    """
    baseline_label:   str
    candidate_label:  str
    baseline_report:  Report
    candidate_report: Report

    @property
    def cai_delta(self) -> Optional[float]:
        """Legacy: change in cai_score (higher=better). Positive = improvement."""
        b = self.baseline_report.cai_score
        c = self.candidate_report.cai_score
        if b is None or c is None:
            return None
        return round(c - b, 3)

    @property
    def strain_delta(self) -> Optional[float]:
        """
        Change in CAI Strain (lower=better). Negative = improvement (less drift).
        Positive = regression (more drift).
        """
        b = self.baseline_report.cai_strain
        c = self.candidate_report.cai_strain
        if b is None or c is None:
            return None
        return round(c - b, 3)

    @property
    def regressed(self) -> bool:
        """True if candidate CAI Strain rose vs baseline (more drift)."""
        delta = self.strain_delta
        return delta is not None and delta > 0

    def fail_if_above(self, strain: float = 0.25) -> None:
        """
        Raise AssertionError if candidate CAI Strain exceeds threshold.
        Drop this in a GitHub Actions step to block merges on regressions.

        Args:
            strain: Maximum acceptable CAI Strain (default 0.25). Lower is better.

        Raises:
            AssertionError: If candidate CAI Strain > strain threshold.

        Example:
            result = suite.compare(baseline_app, candidate_app)
            result.fail_if_above(strain=0.20)  # CI fails if Strain rises above 0.20
        """
        observed = self.candidate_report.cai_strain
        if observed is not None and observed > strain:
            base_strain = self.baseline_report.cai_strain
            base_str = f"{base_strain:.3f}" if base_strain is not None else "n/a"
            delta_str = f"{self.strain_delta:+.3f}" if self.strain_delta is not None else "n/a"
            raise AssertionError(
                f"CAI regression: {self.candidate_label} CAI Strain {observed:.3f} "
                f"(max allowed: {strain}). "
                f"Baseline ({self.baseline_label}): {base_str}. "
                f"Delta: {delta_str}"
            )

    def fail_if_below(self, consistency: float = 0.75) -> None:
        """
        Legacy CI gate: raise if candidate cai_score (higher=better) falls below
        `consistency`. Prefer `fail_if_above(strain=...)` in new code.
        """
        score = self.candidate_report.cai_score
        if score is not None and score < consistency:
            raise AssertionError(
                f"CAI regression: {self.candidate_label} scored {score:.3f} "
                f"(min: {consistency}). "
                f"Baseline ({self.baseline_label}): {self.baseline_report.cai_score:.3f}. "
                f"Delta: {self.cai_delta:+.3f}"
            )

    def to_dict(self) -> dict:
        return {
            "baseline_label":  self.baseline_label,
            "candidate_label": self.candidate_label,
            "baseline_strain":    self.baseline_report.cai_strain,
            "candidate_strain":   self.candidate_report.cai_strain,
            "strain_delta":       self.strain_delta,
            # legacy fields (cai_score-based, higher=better):
            "baseline_cai":    self.baseline_report.cai_score,
            "candidate_cai":   self.candidate_report.cai_score,
            "cai_delta":       self.cai_delta,
            "regressed":       self.regressed,
            "baseline":        self.baseline_report.to_dict(),
            "candidate":       self.candidate_report.to_dict(),
        }

    def __str__(self) -> str:
        delta = self.strain_delta
        status = "REGRESSION" if self.regressed else "PASS"
        # for Strain, an improvement is a *negative* delta (less drift), so flip arrow
        arrow = f"{delta:+.3f}" if delta is not None else "N/A"
        base_strain = self.baseline_report.cai_strain
        cand_strain = self.candidate_report.cai_strain
        base_str = f"{base_strain:.3f}" if base_strain is not None else "n/a"
        cand_str = f"{cand_strain:.3f}" if cand_strain is not None else "n/a"
        return (
            f"\nCAI Strain Regression: {status}\n"
            f"  {self.baseline_label}:  {base_str}\n"
            f"  {self.candidate_label}: {cand_str}\n"
            f"  delta: {arrow}  (lower CAI Strain is better)\n"
        )


# ── Firewall ────────────────────────────────────────────────────────────────────

@dataclass
class FirewallResult:
    """
    Result from a single Firewall.check() call.

    If blocked=True, response contains the safe fallback (not the original).
    If contradiction_detected=True, the contradicting cached entry is in cached_*.
    """
    query:                  str
    response:               str
    blocked:                bool
    contradiction_detected: bool
    cached_query:           Optional[str] = None
    cached_response:        Optional[str] = None
    explanation:            Optional[str] = None


# ── PromptRepair ───────────────────────────────────────────────────────────────

@dataclass
class RepairResult:
    """One improved prompt variant from PromptRepair.fix()."""
    original_prompt:    str
    improved_prompt:    str
    original_cai_score: float        # legacy (higher = better)
    improved_cai_score: float        # legacy
    delta:              float        # legacy: improved_cai_score - original_cai_score (positive = better)
    report:             Report
    rank:               int  # 1 = best

    @property
    def original_cai_strain(self) -> float:
        """CAI Strain of the original prompt (lower = better)."""
        return round(1.0 - self.original_cai_score, 4)

    @property
    def improved_cai_strain(self) -> float:
        """CAI Strain of the improved prompt (lower = better)."""
        return round(1.0 - self.improved_cai_score, 4)

    @property
    def strain_delta(self) -> float:
        """
        Change in CAI Strain from original to improved (negative = improvement).
        Equivalent to (improved_cai_strain - original_cai_strain).
        """
        return round(self.improved_cai_strain - self.original_cai_strain, 4)
