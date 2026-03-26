"""
Failure fingerprinting for contradish.

Groups CAI failures by pattern type. "3 failures" tells you something broke.
Fingerprinting tells you it's numeric_drift across 3 policy rules and you need
to anchor your numbers.

Usage:
    from contradish.fingerprint import fingerprint

    clusters = fingerprint(report)
    for cluster in clusters:
        print(cluster.pattern_type, cluster.frequency)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Report, TestResult, ContradictionPair


# Severity categories surfaced by the judge
_SEVERITY_LABELS = {
    "factual":  "Factual contradiction",
    "logical":  "Logical inconsistency",
    "policy":   "Policy contradiction",
}

# Known unstable-pattern keywords mapped to a human label
_PATTERN_KEYWORDS: list[tuple[list[str], str]] = [
    (["exception", "special case", "override"],        "exception_invention"),
    (["hedg", "qualif", "maybe", "possibly", "might"], "hedge_inconsistency"),
    (["number", "numer", "day", "week", "month", "percent", "digit"], "numeric_drift"),
    (["eligib", "qualify", "entitled"],                "eligibility_flip"),
    (["deadline", "expir", "window", "cutoff"],        "deadline_drift"),
    (["legal", "disclaim", "liab", "advice"],          "legal_boundary_blur"),
    (["privacy", "data", "gdpr", "ccpa"],              "data_policy_drift"),
    (["coverage", "covered", "benefit"],               "coverage_inconsistency"),
]


def _classify_pattern(unstable_patterns: list[str], severity: str) -> str:
    """
    Infer a pattern type from the unstable_patterns list and contradiction severity.
    Falls back to severity-based bucket if no keyword match found.
    """
    combined = " ".join(unstable_patterns).lower()
    for keywords, label in _PATTERN_KEYWORDS:
        if any(kw in combined for kw in keywords):
            return label
    return _SEVERITY_LABELS.get(severity, severity or "unclassified")


@dataclass
class FailureCluster:
    """
    A group of CAI failures sharing the same root-cause pattern.

    Attributes:
        pattern_type:  Inferred label for the failure pattern.
        frequency:     How many test rules hit this pattern.
        affected_rules: Names of the rules in this cluster.
        example_rule:  Name of the first (worst-scoring) rule in the cluster.
        example_pair:  The worst contradiction pair in the cluster, if available.
        suggested_fix: Aggregated fix suggestion from the cluster, if available.
    """
    pattern_type:    str
    frequency:       int
    affected_rules:  list[str]
    example_rule:    str
    example_pair:    Optional[object] = None   # ContradictionPair | None
    suggested_fix:   Optional[str] = None

    def __str__(self) -> str:
        rules = ", ".join(self.affected_rules[:3])
        more = f" (+{len(self.affected_rules) - 3} more)" if len(self.affected_rules) > 3 else ""
        lines = [
            f"[{self.pattern_type}]  {self.frequency} rule{'s' if self.frequency != 1 else ''}",
            f"  rules:   {rules}{more}",
        ]
        if self.suggested_fix:
            lines.append(f"  fix:     {self.suggested_fix}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d: dict = {
            "pattern_type":   self.pattern_type,
            "frequency":      self.frequency,
            "affected_rules": self.affected_rules,
            "example_rule":   self.example_rule,
            "suggested_fix":  self.suggested_fix,
        }
        if self.example_pair is not None:
            pair = self.example_pair
            d["example_pair"] = {
                "input_a":     pair.input_a,
                "output_a":    pair.output_a,
                "input_b":     pair.input_b,
                "output_b":    pair.output_b,
                "explanation": pair.explanation,
                "severity":    pair.severity,
            }
        return d


def fingerprint(report: "Report") -> list[FailureCluster]:
    """
    Cluster the failed test results in a report by their root-cause pattern.

    Groups failures by inferred pattern type (e.g. policy_contradiction,
    numeric_drift, exception_invention). Returns a list of FailureCluster
    objects sorted by frequency (most common first).

    Args:
        report: A contradish Report (from suite.run()).

    Returns:
        List of FailureCluster objects. Empty list if no failures.

    Example:
        clusters = fingerprint(report)
        for cluster in clusters:
            print(cluster.pattern_type, cluster.frequency)
    """
    clusters: dict[str, list] = {}

    for result in report.failed:
        # Pick the primary severity from the first contradiction, or fall back
        primary_severity = ""
        if result.contradictions:
            primary_severity = result.contradictions[0].severity or ""

        pattern = _classify_pattern(result.unstable_patterns, primary_severity)

        if pattern not in clusters:
            clusters[pattern] = []
        clusters[pattern].append(result)

    out: list[FailureCluster] = []
    for pattern_type, results in clusters.items():
        # Sort by CAI score ascending (worst first)
        results_sorted = sorted(results, key=lambda r: r.cai_score or 0.0)
        worst = results_sorted[0]

        example_pair = worst.contradictions[0] if worst.contradictions else None

        # Aggregate suggestions: prefer the worst result's suggestion
        suggestion = next(
            (r.suggestion for r in results_sorted if r.suggestion),
            None
        )

        out.append(FailureCluster(
            pattern_type=pattern_type,
            frequency=len(results),
            affected_rules=[r.test_case.name for r in results_sorted],
            example_rule=worst.test_case.name,
            example_pair=example_pair,
            suggested_fix=suggestion,
        ))

    # Most frequent first
    out.sort(key=lambda c: -c.frequency)
    return out
