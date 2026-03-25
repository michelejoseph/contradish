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
