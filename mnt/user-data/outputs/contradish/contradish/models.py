"""
Core data models for contradish.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class TestCase:
    """
    A single test case to evaluate.

    Args:
        input: The prompt or question to test.
        name: Optional human-readable label.
        context: Optional retrieved context (for RAG grounding checks).
        expected_traits: Optional list of expected behaviors (used as judge hints).

    Example:
        TestCase(
            name="refund policy",
            input="Can I get a refund after 45 days?",
            context="Refunds are allowed within 30 days of purchase.",
            expected_traits=["should not invent policy", "should be consistent"]
        )
    """
    input: str
    name: Optional[str] = None
    context: Optional[str] = None
    expected_traits: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = self.input[:50] + ("..." if len(self.input) > 50 else "")


@dataclass
class FailurePattern:
    """A detected failure pattern across paraphrase runs."""
    pattern: str
    issue: str
    affected_runs: int
    total_runs: int
    examples: list[dict]

    @property
    def failure_rate(self) -> float:
        return self.affected_runs / self.total_runs if self.total_runs > 0 else 0.0


@dataclass
class TestResult:
    """Result for a single TestCase."""
    test_case: TestCase
    outputs: list[str]
    paraphrases: list[str]

    # Scores (0.0 - 1.0, higher is better/safer)
    consistency_score: Optional[float] = None
    contradiction_score: Optional[float] = None  # lower = more contradictions
    grounding_score: Optional[float] = None

    # Risk
    risk: RiskLevel = RiskLevel.LOW

    # Failures
    failure_patterns: list[FailurePattern] = field(default_factory=list)
    contradictions_found: list[dict] = field(default_factory=list)

    # Raw judge responses
    judge_notes: list[str] = field(default_factory=list)

    def passed(self, thresholds: Optional[dict] = None) -> bool:
        defaults = {"consistency": 0.80, "contradiction": 0.20, "grounding": 0.80}
        t = {**defaults, **(thresholds or {})}
        if self.consistency_score is not None and self.consistency_score < t["consistency"]:
            return False
        if self.contradiction_score is not None and self.contradiction_score > t["contradiction"]:
            return False
        if self.grounding_score is not None and self.grounding_score < t["grounding"]:
            return False
        return True

    def __str__(self) -> str:
        lines = [f"\n{'='*60}", f"Test: {self.test_case.name}", f"{'='*60}"]
        if self.consistency_score is not None:
            lines.append(f"  consistency_score : {self.consistency_score:.2f}")
        if self.contradiction_score is not None:
            lines.append(f"  contradiction_risk: {self.contradiction_score:.2f}")
        if self.grounding_score is not None:
            lines.append(f"  grounding_score   : {self.grounding_score:.2f}")
        lines.append(f"  risk              : {self.risk.value}")
        if self.failure_patterns:
            lines.append("\n  Detected issues:")
            for fp in self.failure_patterns:
                lines.append(f"    • [{fp.affected_runs}/{fp.total_runs} runs] {fp.issue}")
                lines.append(f"      Pattern: {fp.pattern}")
        if self.contradictions_found:
            lines.append(f"\n  Contradictions ({len(self.contradictions_found)}):")
            for c in self.contradictions_found[:3]:
                lines.append(f"    A: {c.get('output_a', '')[:80]}")
                lines.append(f"    B: {c.get('output_b', '')[:80]}")
        return "\n".join(lines)


@dataclass
class Report:
    """Aggregate report across all test cases."""
    results: list[TestResult]
    thresholds: dict = field(default_factory=dict)

    @property
    def passed(self) -> list[TestResult]:
        return [r for r in self.results if r.passed(self.thresholds)]

    @property
    def failed(self) -> list[TestResult]:
        return [r for r in self.results if not r.passed(self.thresholds)]

    @property
    def avg_consistency(self) -> Optional[float]:
        scores = [r.consistency_score for r in self.results if r.consistency_score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def avg_contradiction_risk(self) -> Optional[float]:
        scores = [r.contradiction_score for r in self.results if r.contradiction_score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def avg_grounding(self) -> Optional[float]:
        scores = [r.grounding_score for r in self.results if r.grounding_score is not None]
        return sum(scores) / len(scores) if scores else None

    def summary(self) -> str:
        lines = [
            "\n" + "="*60,
            "CONTRADISH REPORT",
            "="*60,
            f"  Tests run : {len(self.results)}",
            f"  Passed    : {len(self.passed)}",
            f"  Failed    : {len(self.failed)}",
            "",
            "  Aggregate scores:",
        ]
        if self.avg_consistency is not None:
            lines.append(f"    consistency_score : {self.avg_consistency:.2f}")
        if self.avg_contradiction_risk is not None:
            lines.append(f"    contradiction_risk: {self.avg_contradiction_risk:.2f}")
        if self.avg_grounding is not None:
            lines.append(f"    grounding_score   : {self.avg_grounding:.2f}")
        if self.failed:
            lines.append("\n  Failed tests:")
            for r in self.failed:
                lines.append(f"    ✗ {r.test_case.name}")
        return "\n".join(lines)

    def __str__(self) -> str:
        parts = [self.summary()]
        for result in self.results:
            parts.append(str(result))
        return "\n".join(parts)


@dataclass
class RegressionResult:
    """Comparison between baseline and candidate."""
    baseline_label: str
    candidate_label: str
    baseline_report: Report
    candidate_report: Report

    @property
    def consistency_delta(self) -> Optional[float]:
        b = self.baseline_report.avg_consistency
        c = self.candidate_report.avg_consistency
        if b is not None and c is not None:
            return c - b
        return None

    @property
    def contradiction_delta(self) -> Optional[float]:
        b = self.baseline_report.avg_contradiction_risk
        c = self.candidate_report.avg_contradiction_risk
        if b is not None and c is not None:
            return c - b
        return None

    @property
    def grounding_delta(self) -> Optional[float]:
        b = self.baseline_report.avg_grounding
        c = self.candidate_report.avg_grounding
        if b is not None and c is not None:
            return c - b
        return None

    def fail_if_below(
        self,
        consistency: float = 0.80,
        contradiction_max: float = 0.25,
        grounding: float = 0.80,
    ) -> None:
        """Raise AssertionError if candidate regresses below thresholds. Use in CI."""
        errors = []
        c_cons = self.candidate_report.avg_consistency
        c_cont = self.candidate_report.avg_contradiction_risk
        c_grou = self.candidate_report.avg_grounding

        if c_cons is not None and c_cons < consistency:
            delta = self.consistency_delta
            errors.append(
                f"consistency_score {c_cons:.2f} below threshold {consistency:.2f}"
                + (f" (Δ {delta:+.2f})" if delta is not None else "")
            )
        if c_cont is not None and c_cont > contradiction_max:
            delta = self.contradiction_delta
            errors.append(
                f"contradiction_risk {c_cont:.2f} above threshold {contradiction_max:.2f}"
                + (f" (Δ {delta:+.2f})" if delta is not None else "")
            )
        if c_grou is not None and c_grou < grounding:
            delta = self.grounding_delta
            errors.append(
                f"grounding_score {c_grou:.2f} below threshold {grounding:.2f}"
                + (f" (Δ {delta:+.2f})" if delta is not None else "")
            )
        if errors:
            raise AssertionError(
                f"\nCONTRADISH REGRESSION DETECTED ({self.baseline_label} → {self.candidate_label})\n"
                + "\n".join(f"  ✗ {e}" for e in errors)
            )

    def __str__(self) -> str:
        def fmt_delta(d: Optional[float], invert: bool = False) -> str:
            if d is None:
                return "n/a"
            sign = "+" if d > 0 else ""
            better = (d > 0 and not invert) or (d < 0 and invert)
            indicator = "▲" if better else "▼"
            return f"{sign}{d:.2f} {indicator}"

        lines = [
            "\n" + "="*60,
            f"REGRESSION: {self.baseline_label} → {self.candidate_label}",
            "="*60,
        ]
        b, c = self.baseline_report, self.candidate_report
        if b.avg_consistency is not None:
            lines.append(
                f"  consistency_score : {b.avg_consistency:.2f} → {c.avg_consistency:.2f}  "
                f"({fmt_delta(self.consistency_delta)})"
            )
        if b.avg_contradiction_risk is not None:
            lines.append(
                f"  contradiction_risk: {b.avg_contradiction_risk:.2f} → {c.avg_contradiction_risk:.2f}  "
                f"({fmt_delta(self.contradiction_delta, invert=True)})"
            )
        if b.avg_grounding is not None:
            lines.append(
                f"  grounding_score   : {b.avg_grounding:.2f} → {c.avg_grounding:.2f}  "
                f"({fmt_delta(self.grounding_delta)})"
            )
        return "\n".join(lines)
