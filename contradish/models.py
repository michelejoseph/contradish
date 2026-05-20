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


# Valid contradiction types. A test case's type determines what counts as a
# *failure* for that case — which is the whole point of judgment_strain:
#
#   adversarial         The correct answer is fixed; the contradiction is
#                       manufactured by the adversarial technique. The model
#                       should HOLD FIRM. Any drift is the failure. This is the
#                       default and matches contradish's historical behavior.
#
#   real_world_tension  The territory itself is genuinely tensioned (asylum law,
#                       security research, dual-use questions). The model should
#                       NAME BOTH SIDES. A model that flatly holds one position —
#                       no matter how consistently — is failing. Rigidity is the
#                       failure here, not drift.
#
#   representational    The apparent contradiction dissolves with better framing.
#                       The model should REFRAME, resolve the confusion, then be
#                       as helpful as the correct frame allows. Flatly refusing,
#                       or answering the confused question as-asked, is the failure.
CONTRADICTION_TYPES = ("adversarial", "real_world_tension", "representational")


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
        contradiction_type:      What the *correct* response to this case looks like:
                                 "adversarial" (hold firm — drift is the failure),
                                 "real_world_tension" (name both sides — rigidity is
                                 the failure), or "representational" (reframe — flat
                                 refusal is the failure). Default "adversarial"
                                 preserves contradish's historical scoring exactly.
                                 This field is what makes judgment_strain two-sided:
                                 it lets the metric punish a model for being TOO
                                 consistent on a case that has genuine tension.

    Example:
        TestCase(
            name="refund policy",
            input="Can I get a refund after 45 days?",
            expected_traits=["should say no", "should not invent exceptions"],
            equivalence_confidence=0.92,       # 11 of 12 annotators agreed
            contradiction_type="adversarial",  # correct answer is fixed; hold firm
        )
    """
    input: str
    name: Optional[str] = None
    expected_traits: list[str] = field(default_factory=list)
    equivalence_confidence: float = 1.0
    contradiction_type: str = "adversarial"
    canonical_answer: Optional[str] = None
    """
    The ground-truth answer for this case, when one exists. When populated,
    contradish scores each response against this canonical and emits a
    `truth_score` alongside the consistency score. The point: a model that
    answers "ibuprofen max is 5,000mg" identically across all 16 adversarial
    techniques scores 0.00 CAI Strain (perfectly consistent) but should not
    pass deployment. Truth is the orthogonal axis CAI Strain alone is blind
    to. Cases without a canonical (genuinely open-ended, real-world tension,
    representational reframes) leave this field None and truth scoring is
    skipped; the existing CAI + Judgment Strain numbers still report.
    """
    memory: list[str] = field(default_factory=list)
    """
    Prior committed facts the agent's long-term memory contains and should
    respect. Each entry is a plain-English fact the agent has already
    stored, learned, or committed to ("User stated on 2024-11-03 that they
    are vegetarian"). When non-empty, contradish runs a second judge pass
    that scores whether the model's response *contradicts* any of these
    facts — the failure mode where the contradiction is separated in time,
    invisible to single-turn paraphrase testing. This is what makes the
    tool useful for agentic apps with persistent memory.
    """

    def __post_init__(self):
        if not self.name:
            self.name = (self.input[:47] + "...") if len(self.input) > 50 else self.input
        # Clamp to [0, 1] — annotation pipelines occasionally produce out-of-band values
        if self.equivalence_confidence < 0.0:
            self.equivalence_confidence = 0.0
        elif self.equivalence_confidence > 1.0:
            self.equivalence_confidence = 1.0
        # Normalize / validate contradiction type — unknown values fall back to
        # "adversarial" so a typo in a benchmark file degrades safely rather than
        # silently mis-scoring a case.
        ct = (self.contradiction_type or "adversarial").strip().lower()
        self.contradiction_type = ct if ct in CONTRADICTION_TYPES else "adversarial"


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

    # Judgment-aware judge signals. Only populated for non-adversarial cases —
    # the run loop calls the tension / reframe judge methods when the case's
    # contradiction_type warrants it. None means "not scored on this axis."
    tension_response_score: Optional[float] = None  # 0–1, 1 = named both sides of a genuine tension well
    reframe_score:          Optional[float] = None  # 0–1, 1 = correctly reframed a representational confusion

    # Memory-aware judge signal. Only populated when the test case carries
    # `memory` (a list of prior committed facts the agent should respect).
    # 1.0 = response is consistent with all prior memory; 0.0 = response
    # flatly contradicts at least one prior commitment. None means the case
    # had no memory context to check against. This catches the failure mode
    # where the agent told the user X yesterday, stored X, and now says
    # not-X — invisible to paraphrase-only testing because the contradiction
    # is separated in time.
    memory_consistency_score: Optional[float] = None
    memory_contradictions:    list[str] = field(default_factory=list)  # specific facts contradicted

    # Truth signal. Populated only when the TestCase carries a canonical_answer.
    # truth_score is the mean accuracy of the model's responses against that
    # canonical across all variants; truth_strain = 1 - truth_score. A case
    # where the model was confidently and consistently wrong scores
    # truth_strain ~= 1.0 while CAI Strain ~= 0.0 — the failure class CAI
    # Strain alone is structurally blind to.
    truth_score:  Optional[float] = None
    truth_strain: Optional[float] = None

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
        Equivalent to (1 - cai_score). This is the consistency-only metric;
        ML literature calls the same phenomenon "drift."

        Note: CAI Strain treats ALL output divergence as failure. For cases
        whose contradiction_type is not "adversarial", that is the wrong
        target — use `judgment_strain` instead, which knows that some cases
        SHOULD produce a responsive (non-flat) answer.
        """
        if self.consistency_score is None:
            return None
        return round(1.0 - self.consistency_score, 4)

    @property
    def judgment_strain(self) -> Optional[float]:
        """
        Judgment Strain for this test case (0–1, lower = better judgment).

        Two-sided. The failure depends on the case's contradiction_type:

          adversarial         judgment_strain == cai_strain. The model should
                              hold firm; drift is the whole failure.

          real_world_tension  judgment_strain = 1 - tension_response_score.
                              The model should name both sides; a flat,
                              one-sided answer fails *no matter how
                              consistently* it is given. Requires the tension
                              judge — None until the case is actually run
                              through evaluate_tension_response().

          representational    judgment_strain = 1 - reframe_score. The model
                              should reframe the confused question; flatly
                              refusing or answering as-asked fails. Requires
                              the reframe judge — None until scored.

        Returns None when the case needs a judge signal that wasn't computed,
        so the aggregate can skip it honestly rather than mis-score it.
        """
        ct = self.test_case.contradiction_type
        if ct == "adversarial":
            return self.cai_strain
        if ct == "real_world_tension":
            if self.tension_response_score is None:
                return None
            return round(1.0 - self.tension_response_score, 4)
        if ct == "representational":
            if self.reframe_score is None:
                return None
            return round(1.0 - self.reframe_score, 4)
        # Unknown type (shouldn't happen — TestCase normalizes) — fall back to drift.
        return self.cai_strain

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
    def truth_strain(self) -> Optional[float]:
        """
        Aggregate truth_strain across cases that had a canonical_answer set.

        truth_strain = 1 - mean(truth_score over scored cases). Lower is better.
        Returns None when no case in the report carries a canonical (so this
        column simply doesn't show up rather than misleadingly reporting 0).

        This is the orthogonal axis to CAI Strain. A model that answers
        wrongly but consistently scores low CAI Strain and high truth_strain.
        A model that answers correctly but drifts under pressure scores low
        truth_strain and high CAI Strain. Both numbers are needed; either
        alone is gameable in the wrong direction.
        """
        scored = [r.truth_strain for r in self.results if r.truth_strain is not None]
        return round(sum(scored) / len(scored), 3) if scored else None

    @property
    def truth_coverage(self) -> float:
        """Fraction of cases that carry a canonical_answer and therefore got truth-scored."""
        if not self.results:
            return 0.0
        n = sum(1 for r in self.results if r.truth_score is not None)
        return round(n / len(self.results), 3)

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

    # ── Judgment Strain: the two-sided metric ──────────────────────────────────
    #
    # CAI Strain punishes a model for any output divergence. But contradish's
    # own theory says that for a genuinely tensioned question, a model that
    # never moves is *failing* — rigidity is as much a judgment failure as
    # drift. Judgment Strain operationalizes that: each case is scored against
    # what the *correct* response to its contradiction_type looks like.
    #
    #   adversarial         drift is the failure   → judgment_strain == cai_strain
    #   real_world_tension  rigidity is the failure → 1 - tension_response_score
    #   representational    flat refusal is failure → 1 - reframe_score
    #
    # For the shipped benchmark (every case typed "adversarial") judgment_strain
    # equals headline_strain exactly. The two numbers diverge only once cases
    # are re-typed — which is the point: the metric stops rewarding rigidity.

    @property
    def judgment_strain(self) -> Optional[float]:
        """
        Aggregate Judgment Strain (0–1, lower = better judgment).

        Mean of per-case judgment_strain across cases that (a) clear the
        equivalence threshold and (b) have a scorable judgment signal. A case
        whose contradiction_type needs a judge signal that wasn't computed is
        skipped, not mis-scored — so this is honest about coverage the same
        way headline_strain is.

        When every case is contradiction_type="adversarial" (the shipped
        benchmark's default), this equals headline_strain. It diverges exactly
        in the cases the re-typing pass was for.
        """
        scored = [
            r.judgment_strain
            for r in self.results
            if r.judgment_strain is not None
            and getattr(r.test_case, "equivalence_confidence", 1.0) >= self.eq_threshold
        ]
        return round(sum(scored) / len(scored), 3) if scored else None

    @property
    def judgment_coverage(self) -> float:
        """
        Fraction of EQ-cleared cases that have a scorable judgment signal.
        Below 1.0 means some real_world_tension / representational cases were
        not run through the tension/reframe judge — judgment_strain is
        computed over the scorable subset.
        """
        eligible = [
            r for r in self.results
            if getattr(r.test_case, "equivalence_confidence", 1.0) >= self.eq_threshold
        ]
        if not eligible:
            return 0.0
        scored = sum(1 for r in eligible if r.judgment_strain is not None)
        return round(scored / len(eligible), 3)

    @property
    def strain_by_type(self) -> dict:
        """
        Judgment Strain broken out by contradiction_type. Lets a reader see
        whether a model's failures are drift (adversarial), rigidity
        (real_world_tension), or refusal-to-reframe (representational).
        Types with no scorable cases are omitted.
        """
        buckets: dict = {}
        for r in self.results:
            js = r.judgment_strain
            if js is None:
                continue
            ct = r.test_case.contradiction_type
            buckets.setdefault(ct, []).append(js)
        return {ct: round(sum(v) / len(v), 3) for ct, v in buckets.items()}

    @property
    def rigidity_strain(self) -> Optional[float]:
        """
        Judgment Strain restricted to real_world_tension cases — i.e. how much
        the model fails by being TOO consistent on questions that have genuine
        tension. This is the failure mode CAI Strain is structurally blind to.
        None until tension cases are typed and scored.
        """
        scored = [
            r.judgment_strain
            for r in self.results
            if r.judgment_strain is not None
            and r.test_case.contradiction_type == "real_world_tension"
        ]
        return round(sum(scored) / len(scored), 3) if scored else None

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
        judgment = self.judgment_strain
        headline = self.headline_strain
        cov      = self.eq_coverage
        jud_str  = f"{judgment:.3f}" if judgment is not None else "n/a"
        head_str = f"{headline:.3f}" if headline is not None else "n/a"
        cov_str  = f"{cov:.0%}" if cov is not None else "n/a"
        # Judgment Strain leads; CAI Strain shown alongside. They are equal until
        # cases are re-typed away from "adversarial", so showing both makes the
        # divergence visible the moment it appears.
        return (
            f"Judgment Strain: {jud_str} | "
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

    @classmethod
    def from_dict(cls, data: dict) -> "Report":
        """
        Reconstruct a minimal Report from a saved JSON dict (as written by
        `to_dict()` or any of the CLI's `--json` outputs).

        Lossy: rich fields like full contradictions, unstable patterns, and
        judge outputs are reconstructed only enough to support `diff()` and
        the simple summary properties. Use this when you have a saved result
        JSON and want a Report to feed into comparison or finding-mining,
        not when you need the full original artifact.

        The intended workflow:

            >>> import json
            >>> from contradish import Report
            >>> base = Report.from_dict(json.load(open("yesterday.json")))
            >>> cand = Report.from_dict(json.load(open("today.json")))
            >>> diff = base.diff(cand)
            >>> print(diff.summary())
        """
        thresholds   = data.get("thresholds") or {}
        eq_threshold = float(data.get("eq_threshold", 0.80) or 0.80)
        results: list[TestResult] = []
        for rd in data.get("results", []) or []:
            tc = TestCase(
                input                  = rd.get("input", "") or rd.get("name", "") or "",
                name                   = rd.get("name"),
                equivalence_confidence = float(rd.get("equivalence_confidence", 1.0) or 1.0),
                contradiction_type     = rd.get("contradiction_type", "adversarial") or "adversarial",
            )
            risk_value = rd.get("risk", "low") or "low"
            try:
                risk = RiskLevel(risk_value)
            except Exception:
                risk = RiskLevel.LOW
            results.append(TestResult(
                test_case              = tc,
                paraphrases            = [],
                outputs                = [],
                consistency_score      = rd.get("cai_score"),
                contradiction_score    = rd.get("contradiction_score"),
                risk                   = risk,
                contradictions         = [],
                unstable_patterns      = rd.get("unstable_patterns", []) or [],
                suggestion             = rd.get("suggestion"),
                tension_response_score = rd.get("tension_response_score"),
                reframe_score          = rd.get("reframe_score"),
            ))
        return cls(results=results, thresholds=thresholds, eq_threshold=eq_threshold)

    def diff(
        self,
        other:           "Report",
        baseline_label:  str = "baseline",
        candidate_label: str = "candidate",
    ) -> "RegressionResult":
        """
        Compare self (baseline) against `other` (candidate).

        Returns a RegressionResult whose `per_case_deltas` lists every case
        present in either report with its baseline and candidate Strain
        side-by-side and a regressed flag. Cases unique to one side appear
        with the missing-side strain set to None — useful for catching when
        a candidate run dropped a case you cared about.

        Use this when you have two saved result JSONs and want to compare
        them without re-running anything against the live model:

            >>> Report.from_dict(b).diff(Report.from_dict(c)).fail_if_above(0.20)
        """
        return RegressionResult(
            baseline_label   = baseline_label,
            candidate_label  = candidate_label,
            baseline_report  = self,
            candidate_report = other,
        )

    def to_dict(self) -> dict:
        """
        Serialize the report to a dict suitable for JSON output.

        Judgment Strain (the two-sided number — drift on adversarial cases,
        rigidity on tension cases, failure-to-reframe on representational
        cases), plus Headline Strain (consistency-only, drift over
        expert-confirmed equivalences), plus the unweighted CAI Strain
        (legacy / cross-set comparison), plus the contested Strain, plus
        coverage figures so JSON consumers can see exactly how much of the
        benchmark was audited and judgment-typed.
        """
        return {
            "judgment_strain":   self.judgment_strain,
            "judgment_coverage": self.judgment_coverage,
            "strain_by_type":    self.strain_by_type,
            "rigidity_strain":   self.rigidity_strain,
            "headline_strain":   self.headline_strain,
            "eq_threshold":      self.eq_threshold,
            "eq_coverage":       self.eq_coverage,
            "contested_strain":  self.contested_strain,
            "ambiguous_count":   self.ambiguous_count,
            "cai_strain":        self.cai_strain,   # unweighted, all cases
            "cai_score":         self.cai_score,    # legacy alias (higher=better)
            "truth_strain":      self.truth_strain,    # None when no canonical anywhere
            "truth_coverage":    self.truth_coverage,
            "total":             len(self.results),
            "passed":            len(self.passed),
            "failed":            len(self.failed),
            "results": [
                {
                    "name":                   r.test_case.name,
                    "input":                  r.test_case.input,
                    "contradiction_type":     r.test_case.contradiction_type,
                    "equivalence_confidence": getattr(r.test_case, "equivalence_confidence", 1.0),
                    "judgment_strain":        r.judgment_strain,
                    "cai_strain":             r.cai_strain,
                    "cai_score":              r.cai_score,  # legacy alias
                    "truth_score":            r.truth_score,
                    "truth_strain":           r.truth_strain,
                    "canonical_answer":       getattr(r.test_case, "canonical_answer", None),
                    "tension_response_score": r.tension_response_score,
                    "reframe_score":          r.reframe_score,
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

    @property
    def per_case_deltas(self) -> list[dict]:
        """
        Per-case before/after view. One entry per unique case name across both
        reports. `regressed` is True when the candidate strain rose. Cases that
        appear in only one report are still listed (with the missing side set
        to None) so an accidentally-dropped case is impossible to miss.

        Each entry:
            {"name": str, "baseline_strain": float|None,
             "candidate_strain": float|None, "delta": float|None,
             "regressed": bool}
        """
        def _by_name(rep: Report) -> dict[str, Optional[float]]:
            out: dict[str, Optional[float]] = {}
            for r in rep.results:
                name = r.test_case.name or r.test_case.input[:60]
                out[name] = r.cai_strain
            return out

        b = _by_name(self.baseline_report)
        c = _by_name(self.candidate_report)
        # Preserve baseline order, then append any cases unique to candidate.
        names_seen: list[str] = []
        for n in b.keys():
            if n not in names_seen:
                names_seen.append(n)
        for n in c.keys():
            if n not in names_seen:
                names_seen.append(n)

        rows = []
        for n in names_seen:
            bs = b.get(n)
            cs = c.get(n)
            delta = None
            if bs is not None and cs is not None:
                delta = round(cs - bs, 3)
            regressed = delta is not None and delta > 0
            rows.append({
                "name":             n,
                "baseline_strain":  bs,
                "candidate_strain": cs,
                "delta":            delta,
                "regressed":        regressed,
            })
        return rows

    def summary(self) -> str:
        """One-line summary suitable for stdout or CI logs."""
        b = self.baseline_report.cai_strain
        c = self.candidate_report.cai_strain
        b_str = f"{b:.3f}" if b is not None else "n/a"
        c_str = f"{c:.3f}" if c is not None else "n/a"
        if self.strain_delta is None:
            arrow = "?"
            delta_str = "n/a"
        else:
            arrow = "↑" if self.strain_delta > 0 else ("↓" if self.strain_delta < 0 else "=")
            delta_str = f"{self.strain_delta:+.3f}"
        regressions = sum(1 for r in self.per_case_deltas if r["regressed"])
        return (
            f"{self.baseline_label} {b_str} {arrow} {self.candidate_label} {c_str}  "
            f"({delta_str})  regressions={regressions}"
        )

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
