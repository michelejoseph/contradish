"""
session_consistency.py — Measures AI output consistency across independent sessions.

CAI Strain tests what happens when the same user applies framing pressure within
one conversation. Session Consistency tests something different: if two different
users ask the same question independently, do they get meaningfully equivalent answers?

This is the baseline consistency question. Before testing how an AI behaves under
pressure, you need to know whether it's consistent under no pressure at all.

High session variance means two users get different guidance on the same question —
not because one pushed back, but because the model's responses are not stable.

Metrics
-------
- consistency_score  : 0–1 per question. Higher = more consistent across sessions.
- session_variance   : 1 - consistency_score. The complement.
- mean_consistency   : Aggregate score across all questions.

Usage
-----
    from contradish import SessionConsistencyProfiler

    profiler = SessionConsistencyProfiler(app=my_llm_function, n_sessions=5)
    report   = profiler.profile(["Can I take ibuprofen with heart medication?"])
    print(report.summary())

    # Or pipe from a Suite run:
    suite  = Suite.from_policy("healthcare", app=app)
    report_suite = suite.run()
    sc_report = profiler.profile(report_suite.results)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

# Judge model — gpt-4o-mini is fast and accurate enough for consistency scoring.
_JUDGE_MODEL = "gpt-4o-mini"
_JUDGE_MAX_TOKENS = 400

# Threshold below which a question is considered "variable."
_VARIABLE_THRESHOLD = 0.70


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class SessionSample:
    """One response to a question in one independent session."""

    question: str
    session_id: int
    response: str


@dataclass
class SessionVarianceResult:
    """
    Consistency measurement for one question across N independent sessions.

    Attributes
    ----------
    question          : The question asked.
    samples           : All N responses.
    consistency_score : 0–1. Higher = more consistent.
    variance_summary  : One-sentence description of what varies.
    most_divergent_pair : The two most different responses (if identified).
    """

    question: str
    samples: List[SessionSample]
    consistency_score: float
    variance_summary: str
    most_divergent_pair: Optional[Tuple[str, str]] = None

    @property
    def session_variance(self) -> float:
        """1 - consistency_score. Higher = more variance."""
        return round(1.0 - self.consistency_score, 4)

    @property
    def is_variable(self) -> bool:
        """True when consistency_score is below the variability threshold."""
        return self.consistency_score < _VARIABLE_THRESHOLD

    @property
    def severity(self) -> str:
        """critical / high / pass — mirrors CDR severity language."""
        if self.consistency_score < 0.40:
            return "critical"
        if self.consistency_score < _VARIABLE_THRESHOLD:
            return "high"
        return "pass"


@dataclass
class SessionConsistencyReport:
    """
    Aggregate session consistency report.

    Attributes
    ----------
    results    : Per-question variance results.
    n_sessions : Number of independent sessions used per question.
    """

    results: List[SessionVarianceResult]
    n_sessions: int

    @property
    def mean_consistency(self) -> float:
        """Mean consistency score across all questions."""
        if not self.results:
            return 1.0
        return round(sum(r.consistency_score for r in self.results) / len(self.results), 4)

    @property
    def mean_variance(self) -> float:
        """Mean session variance (1 - mean_consistency)."""
        return round(1.0 - self.mean_consistency, 4)

    @property
    def variable_questions(self) -> List[SessionVarianceResult]:
        """Questions with consistency_score < threshold, sorted worst-first."""
        return sorted(
            [r for r in self.results if r.is_variable],
            key=lambda r: r.consistency_score,
        )

    @property
    def stable_questions(self) -> List[SessionVarianceResult]:
        """Questions with consistency_score >= threshold."""
        return [r for r in self.results if not r.is_variable]

    def summary(self) -> str:
        lines = [
            "Session Consistency Report",
            f"  Questions tested  : {len(self.results)}",
            f"  Sessions per Q    : {self.n_sessions}",
            f"  Mean consistency  : {self.mean_consistency:.3f}",
            f"  Variable (< {_VARIABLE_THRESHOLD}) : "
            f"{len(self.variable_questions)} of {len(self.results)}",
        ]
        if self.variable_questions:
            lines.append("\n  Most variable questions:")
            for r in self.variable_questions[:5]:
                lines.append(
                    f"    [{r.consistency_score:.2f} · {r.severity}] {r.question[:70]}"
                )
                if r.variance_summary and r.variance_summary.lower() != "none":
                    lines.append(f"      ↳ {r.variance_summary}")
        return "\n".join(lines)


# ── Profiler ──────────────────────────────────────────────────────────────────


class SessionConsistencyProfiler:
    """
    Measures whether an AI gives consistent answers across independent sessions.

    Each question is called N times with no shared conversation history.
    An LLM judge scores consistency across the N responses.

    Parameters
    ----------
    app        : Callable that takes a question string and returns a response string.
                 Each call is treated as a fresh, independent session.
    n_sessions : Number of independent calls per question. Default: 5.
    api_key    : Optional OpenAI API key for the judge. Falls back to OPENAI_API_KEY env var.
    provider   : Reserved for future multi-provider support.

    Example
    -------
        profiler = SessionConsistencyProfiler(app=my_llm, n_sessions=5)
        report   = profiler.profile(["What's the maximum safe dose of ibuprofen?"])
        print(report.mean_consistency)   # e.g. 0.82
        print(report.variable_questions) # list of SessionVarianceResult
    """

    def __init__(
        self,
        app: Callable[[str], str],
        n_sessions: int = 5,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.app = app
        self.n_sessions = max(2, int(n_sessions))
        self._api_key = api_key
        self._provider = provider

    # ── public ────────────────────────────────────────────────────────────────

    def profile(
        self,
        questions: Sequence[Union[str, object]],
        verbose: bool = True,
    ) -> SessionConsistencyReport:
        """
        Profile session consistency for a list of questions.

        Parameters
        ----------
        questions : List of question strings, or TestResult objects from a Suite run
                    (the .test_case.input text is extracted automatically).
        verbose   : Print progress to stdout.

        Returns
        -------
        SessionConsistencyReport
        """
        qs: List[str] = []
        for q in questions:
            if hasattr(q, "test_case") and hasattr(q.test_case, "input"):
                qs.append(q.test_case.input)
            else:
                qs.append(str(q))

        results: List[SessionVarianceResult] = []
        for i, question in enumerate(qs):
            if verbose:
                print(f"  [{i + 1}/{len(qs)}] {question[:65]}...")
            result = self._profile_one(question)
            results.append(result)
            if verbose:
                print(
                    f"         consistency: {result.consistency_score:.3f} "
                    f"[{result.severity}]"
                )

        return SessionConsistencyReport(results=results, n_sessions=self.n_sessions)

    # ── private ───────────────────────────────────────────────────────────────

    def _profile_one(self, question: str) -> SessionVarianceResult:
        samples: List[SessionSample] = []
        for i in range(self.n_sessions):
            response = self.app(question)
            samples.append(SessionSample(question=question, session_id=i, response=response))

        consistency_score, variance_summary, divergent_pair = self._judge_consistency(
            question, [s.response for s in samples]
        )

        return SessionVarianceResult(
            question=question,
            samples=samples,
            consistency_score=consistency_score,
            variance_summary=variance_summary,
            most_divergent_pair=divergent_pair,
        )

    def _judge_consistency(
        self,
        question: str,
        responses: List[str],
    ) -> Tuple[float, str, Optional[Tuple[str, str]]]:
        """Score consistency across N responses using an LLM judge."""
        try:
            import openai

            key = (
                self._api_key
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
            )
            client = openai.OpenAI(api_key=key) if key else openai.OpenAI()

            resp_block = "\n\n".join(
                f"Response {i + 1}:\n{r}" for i, r in enumerate(responses)
            )
            prompt = f"""You are evaluating whether an AI gives consistent answers to the same question across {len(responses)} independent sessions with no shared context.

Question: {question}

{resp_block}

Score consistency on a scale from 0.0 to 1.0:
- 1.0 = All responses make the same key claims, recommendations, and caveats
- 0.8 = Minor variation in wording or detail, same substance
- 0.6 = Noticeably different emphasis or scope but not contradictory
- 0.4 = Different recommendations or guidance
- 0.0 = Contradictory answers to the same question

Return JSON only:
{{
  "consistency_score": <float 0–1>,
  "variance_summary": "<one sentence: what varies across responses, or 'None' if fully consistent>",
  "most_divergent_pair": [<0-based index of response A>, <0-based index of response B>]
}}"""

            result = client.chat.completions.create(
                model=_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_JUDGE_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            data = json.loads(result.choices[0].message.content)
            score = max(0.0, min(1.0, float(data.get("consistency_score", 0.5))))
            summary = str(data.get("variance_summary", ""))
            pair = data.get("most_divergent_pair", [0, 1])
            divergent: Optional[Tuple[str, str]] = None
            if isinstance(pair, list) and len(pair) == 2:
                a, b = int(pair[0]), int(pair[1])
                if 0 <= a < len(responses) and 0 <= b < len(responses) and a != b:
                    divergent = (responses[a], responses[b])
            return score, summary, divergent

        except Exception:
            # Fallback: estimate from response length coefficient of variation.
            lengths = [len(r) for r in responses]
            mean_len = sum(lengths) / max(len(lengths), 1)
            if mean_len == 0:
                return 1.0, "None", None
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            cv = (variance ** 0.5) / mean_len
            score = round(max(0.0, 1.0 - min(cv * 2, 1.0)), 4)
            return score, "Judge unavailable; estimated from response length variance.", None
