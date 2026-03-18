"""
Suite — the single entry point for contradish.

Minimal usage:
    from contradish import Suite, TestCase

    suite = Suite(app=my_llm_function)
    suite.add(TestCase(input="Can I get a refund after 45 days?"))
    suite.run()
"""

from typing import Callable, Optional
from .models  import TestCase, TestResult, Report, RiskLevel
from .llm     import LLMClient
from .runner  import Runner
from .judge   import Judge
from .printer import print_report, print_progress, print_step


class Suite:
    """
    Point contradish at your LLM app and run reasoning stability checks.

    Args:
        app:      A callable that takes a str and returns a str.
                  This is your LLM app, agent, or RAG pipeline.
        api_key:  Anthropic or OpenAI API key.
                  If omitted, reads ANTHROPIC_API_KEY or OPENAI_API_KEY from env.
        provider: "anthropic" or "openai". Auto-detected from key prefix if omitted.

    Example:
        from contradish import Suite, TestCase

        def my_app(question: str) -> str:
            return openai_client.chat(question)

        suite = Suite(app=my_app)
        suite.add(TestCase(
            name="refund policy",
            input="Can I get a refund after 45 days?",
        ))
        suite.run()
    """

    def __init__(
        self,
        app:      Callable[[str], str],
        api_key:  Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.app        = app
        self._llm       = LLMClient(api_key=api_key, provider=provider)
        self._runner    = Runner(self._llm)
        self._judge     = Judge(self._llm)
        self._cases:    list[TestCase] = []
        self._thresholds: dict = {}

    # ── Builder API ────────────────────────────────────────────────

    def add(self, test_case: TestCase) -> "Suite":
        """Add a test case. Chainable."""
        self._cases.append(test_case)
        return self

    def thresholds(
        self,
        consistency:      float = 0.75,
        contradiction_max: float = 0.30,
    ) -> "Suite":
        """Override pass/fail thresholds. Chainable."""
        self._thresholds = {
            "consistency":   consistency,
            "contradiction": contradiction_max,
        }
        return self

    # ── Run ────────────────────────────────────────────────────────

    def run(
        self,
        paraphrases: int  = 5,
        verbose:     bool = True,
    ) -> Report:
        """
        Run all test cases and print the report.

        Args:
            paraphrases: Number of semantic variants to generate per input. Default 5.
            verbose:     Print progress to stdout. Default True.

        Returns:
            Report — also printed to stdout automatically.
        """
        if not self._cases:
            raise ValueError(
                "No test cases added.\n"
                "  suite.add(TestCase(input='your question here'))"
            )

        results = []
        total   = len(self._cases)

        for i, tc in enumerate(self._cases, 1):
            if verbose:
                print_step("running...", tc.name, i, total)
            result = self._run_one(tc, paraphrases=paraphrases, verbose=verbose)
            results.append(result)

        report = Report(results=results, thresholds=self._thresholds)

        if verbose:
            print_report(report)

        return report

    # ── Internal ───────────────────────────────────────────────────

    def _run_one(
        self,
        tc:          TestCase,
        paraphrases: int,
        verbose:     bool,
    ) -> TestResult:

        # 1. Paraphrase
        if verbose:
            print_progress(f"generating {paraphrases} paraphrases")
        para_list = self._runner.generate_paraphrases(tc.input, n=paraphrases)

        # 2. Run matrix
        total_calls = 1 + len(para_list)
        if verbose:
            print_progress(f"calling your app {total_calls}× across variants")
        inputs, outputs = self._runner.run_matrix(
            app=self.app,
            original=tc.input,
            paraphrases=para_list,
        )

        # 3. Consistency
        if verbose:
            print_progress("evaluating consistency")
        cons = self._judge.evaluate_consistency(
            question=tc.input,
            inputs=inputs,
            outputs=outputs,
        )
        consistency_score = cons["consistency_score"]

        # 4. Contradiction detection
        if verbose:
            print_progress("detecting contradictions")
        contradictions = self._judge.find_contradictions(
            question=tc.input,
            inputs=inputs,
            outputs=outputs,
        )
        contradiction_score = len(contradictions) / max(len(outputs) - 1, 1)
        contradiction_score = min(contradiction_score, 1.0)

        # 5. Pattern extraction (only if contradictions found)
        unstable_patterns: list[str] = []
        suggestion: Optional[str]    = None

        if contradictions:
            if verbose:
                print_progress("diagnosing failure patterns")
            pattern = self._judge.extract_pattern(
                question=tc.input,
                inputs=inputs,
                outputs=outputs,
                contradictions=contradictions,
            )
            if pattern:
                if pattern.get("pattern"):
                    unstable_patterns.append(pattern["pattern"])
                if pattern.get("root_cause"):
                    unstable_patterns.append(pattern["root_cause"])
                suggestion = pattern.get("suggestion")
        elif cons.get("disagreements"):
            unstable_patterns = cons["disagreements"][:2]

        # 6. Risk level
        risk = self._compute_risk(consistency_score, contradiction_score)

        return TestResult(
            test_case=tc,
            paraphrases=para_list,
            outputs=outputs,
            consistency_score=round(consistency_score, 3),
            contradiction_score=round(contradiction_score, 3),
            risk=risk,
            contradictions=contradictions,
            unstable_patterns=unstable_patterns,
            suggestion=suggestion,
        )

    @staticmethod
    def _compute_risk(consistency: float, contradiction: float) -> RiskLevel:
        avg = (consistency + (1 - contradiction)) / 2
        if avg >= 0.82:
            return RiskLevel.LOW
        if avg >= 0.62:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH
