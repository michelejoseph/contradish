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
        input:           The prompt or question to test.
        name:            Optional label. Auto-generated from input if omitted.
        expected_traits: Optional hints for the judge (e.g. "should not invent policy").

    Example:
        TestCase(
            name="refund policy",
            input="Can I get a refund after 45 days?",
            expected_traits=["should say no", "should not invent exceptions"],
        )
    """
    input: str
    name: Optional[str] = None
    expected_traits: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = (self.input[:47] + "...") if len(self.input) > 50 else self.input


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
        CAI (Compression-Aware Intelligence) score for this test case.
        Measures how consistently the app reasons across semantically
        equivalent inputs. 0 = maximally inconsistent. 1 = fully stable.
        Alias for consistency_score.
        """
        return self.consistency_score

    def passed(self, thresholds: Optional[dict] = None) -> bool:
        t = {"consistency": 0.75, "contradiction": 0.30, **(thresholds or {})}
        if self.consistency_score   is not None and self.consistency_score   < t["consistency"]:
            return False
        if self.contradiction_score is not None and self.contradiction_score > t["contradiction"]:
            return False
        return True


@dataclass
class Report:
    """Aggregate report across all TestCases."""
    results:    list[TestResult]
    thresholds: dict = field(default_factory=dict)

    @property
    def passed(self)  -> list[TestResult]:
        return [r for r in self.results if     r.passed(self.thresholds)]

    @property
    def failed(self)  -> list[TestResult]:
        return [r for r in self.results if not r.passed(self.thresholds)]

    @property
    def cai_score(self) -> Optional[float]:
        """
        Aggregate CAI score across all test cases.
        Average of individual CAI scores. 0 = maximally inconsistent. 1 = fully stable.
        """
        s = [r.cai_score for r in self.results if r.cai_score is not None]
        return round(sum(s) / len(s), 3) if s else None

    @property
    def avg_consistency(self) -> Optional[float]:
        """Alias for cai_score. Use cai_score in new code."""
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
        score = self.cai_score
        score_str = f"{score:.3f}" if score is not None else "n/a"
        return (
            f"CAI score: {score_str} | "
            f"{len(self.passed)}/{len(self.results)} passed | "
            f"{self.failure_count} failure(s)"
        )

    def failures_summary(self) -> str:
        """Human-readable summary of all failures, suitable for assert messages."""
        if not self.failed:
            return "No failures."
        lines = [f"{self.failure_count} CAI failure(s):"]
        for r in self.failed:
            score_str = f"{r.cai_score:.2f}" if r.cai_score is not None else "n/a"
            lines.append(f"  - {r.test_case.name} (score {score_str})")
            for c in r.contradictions[:1]:
                lines.append(f"    asked: {c.input_a!r}")
                lines.append(f"    got:   {c.output_a[:120]!r}")
                lines.append(f"    vs:    {c.output_b[:120]!r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Serialize the report to a dict suitable for JSON output.
        CAI scores are included at both the report and test-case level.
        """
        return {
            "cai_score":   self.cai_score,
            "total":       len(self.results),
            "passed":      len(self.passed),
            "failed":      len(self.failed),
            "results": [
                {
                    "name":               r.test_case.name,
                    "input":              r.test_case.input,
                    "cai_score":          r.cai_score,
                    "contradiction_score": r.contradiction_score,
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
    Use .fail_if_below() to gate CI/CD merges on CAI score.
    """
    baseline_label:   str
    candidate_label:  str
    baseline_report:  Report
    candidate_report: Report

    @property
    def cai_delta(self) -> Optional[float]:
        """Change in CAI score. Positive = improvement. Negative = regression."""
        b = self.baseline_report.cai_score
        c = self.candidate_report.cai_score
        if b is None or c is None:
            return None
        return round(c - b, 3)

    @property
    def regressed(self) -> bool:
        """True if candidate CAI score dropped vs baseline."""
        delta = self.cai_delta
        return delta is not None and delta < 0

    def fail_if_below(self, consistency: float = 0.75) -> None:
        """
        Raise AssertionError if candidate CAI score falls below threshold.
        Drop this in a GitHub Actions step to block merges on regressions.

        Args:
            consistency: Minimum acceptable CAI score (default 0.75).

        Raises:
            AssertionError: If candidate CAI score < consistency threshold.

        Example:
            result = suite.compare(baseline_app, candidate_app)
            result.fail_if_below(consistency=0.80)  # CI fails if score drops below 0.80
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
            "baseline_cai":    self.baseline_report.cai_score,
            "candidate_cai":   self.candidate_report.cai_score,
            "cai_delta":       self.cai_delta,
            "regressed":       self.regressed,
            "baseline":        self.baseline_report.to_dict(),
            "candidate":       self.candidate_report.to_dict(),
        }

    def __str__(self) -> str:
        delta = self.cai_delta
        status = "REGRESSION" if self.regressed else "PASS"
        arrow = f"{delta:+.3f}" if delta is not None else "N/A"
        return (
            f"\nCAI Regression: {status}\n"
            f"  {self.baseline_label}:  {self.baseline_report.cai_score:.3f}\n"
            f"  {self.candidate_label}: {self.candidate_report.cai_score:.3f}\n"
            f"  delta: {arrow}\n"
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
    original_cai_score: float
    improved_cai_score: float
    delta:              float
    report:             Report
    rank:               int  # 1 = best
