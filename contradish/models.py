"""
Core data models for contradish.
"""

from dataclasses import dataclass, field
from typing import Optional
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
    def avg_consistency(self) -> Optional[float]:
        s = [r.consistency_score for r in self.results if r.consistency_score is not None]
        return round(sum(s) / len(s), 3) if s else None

    @property
    def avg_contradiction(self) -> Optional[float]:
        s = [r.contradiction_score for r in self.results if r.contradiction_score is not None]
        return round(sum(s) / len(s), 3) if s else None
