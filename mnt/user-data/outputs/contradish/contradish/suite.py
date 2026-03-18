"""
Suite — the main entry point for contradish.
"""

from typing import Callable, Optional
from itertools import combinations

from .models import TestCase, TestResult, Report, RiskLevel, FailurePattern
from .runner import Runner
from .judge import Judge


class Suite:
    """
    Main contradish test suite. Point it at your LLM app and get back
    consistency, contradiction, and grounding scores.

    Example:
        from contradish import Suite, TestCase

        suite = Suite(
            api_key="sk-ant-...",
            app=my_llm_function
        )

        suite.add_test(TestCase(
            name="refund policy",
            input="Can I get a refund after 45 days?",
            context="Refunds are only allowed within 30 days."
        ))

        report = suite.run(paraphrases=5)
        print(report)
    """

    def __init__(
        self,
        api_key: str,
        app: Callable[[str], str],
        judge_model: str = "claude-sonnet-4-20250514",
        paraphrase_model: str = "claude-haiku-4-5-20251001",
    ):
        """
        Args:
            api_key: Your Anthropic API key.
            app: A callable that takes a string input and returns a string output.
                 This is your LLM app, RAG pipeline, or agent.
            judge_model: Claude model to use for evaluation. Default: claude-sonnet.
            paraphrase_model: Claude model for paraphrase generation. Default: claude-haiku.
        """
        self.api_key = api_key
        self.app = app
        self.test_cases: list[TestCase] = []
        self.thresholds: dict = {}

        self._runner = Runner(api_key=api_key, model=paraphrase_model)
        self._judge = Judge(api_key=api_key, model=judge_model)

    def add_test(self, test_case: TestCase) -> "Suite":
        """Add a test case. Returns self for chaining."""
        self.test_cases.append(test_case)
        return self

    def set_thresholds(
        self,
        consistency: float = 0.80,
        contradiction_max: float = 0.25,
        grounding: float = 0.80,
    ) -> "Suite":
        """Set pass/fail thresholds. Returns self for chaining."""
        self.thresholds = {
            "consistency": consistency,
            "contradiction": contradiction_max,
            "grounding": grounding,
        }
        return self

    def run(
        self,
        paraphrases: int = 5,
        repeats: int = 1,
        checks: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> Report:
        """
        Run all test cases and return a Report.

        Args:
            paraphrases: Number of semantic variants to generate per input.
            repeats: Number of times to call the app per variant.
            checks: Which checks to run. Options: "consistency", "contradiction",
                    "grounding". Default: all available.
            verbose: Print progress to stdout.

        Returns:
            Report with scores, failure patterns, and pass/fail status.
        """
        if not self.test_cases:
            raise ValueError("No test cases added. Use suite.add_test(TestCase(...))")

        all_checks = {"consistency", "contradiction", "grounding"}
        active_checks = set(checks) if checks else all_checks

        results = []
        for i, test_case in enumerate(self.test_cases):
            if verbose:
                print(f"[{i+1}/{len(self.test_cases)}] Running: {test_case.name}")

            result = self._run_single(
                test_case=test_case,
                paraphrases=paraphrases,
                repeats=repeats,
                checks=active_checks,
                verbose=verbose,
            )
            results.append(result)

        return Report(results=results, thresholds=self.thresholds)

    def _run_single(
        self,
        test_case: TestCase,
        paraphrases: int,
        repeats: int,
        checks: set,
        verbose: bool,
    ) -> TestResult:
        """Run a single test case end-to-end."""

        # 1. Generate paraphrases
        if verbose:
            print(f"  → Generating {paraphrases} paraphrases...")
        paraphrase_list = self._runner.generate_paraphrases(
            test_case.input, n=paraphrases
        )

        # 2. Run the app across all variants
        if verbose:
            total_calls = (1 + len(paraphrase_list)) * repeats
            print(f"  → Running app ({total_calls} calls)...")
        all_questions, all_outputs = self._runner.run_matrix(
            app=self.app,
            question=test_case.input,
            paraphrases=paraphrase_list,
            repeats=repeats,
        )

        result = TestResult(
            test_case=test_case,
            outputs=all_outputs,
            paraphrases=paraphrase_list,
        )

        # 3. Consistency check
        if "consistency" in checks:
            if verbose:
                print("  → Evaluating consistency...")
            consistency_result = self._judge.evaluate_consistency(
                question=test_case.input,
                outputs=all_outputs,
            )
            result.consistency_score = consistency_result.get("consistency_score", 0.5)
            if consistency_result.get("disagreements"):
                result.judge_notes.extend(consistency_result["disagreements"])

        # 4. Contradiction detection
        if "contradiction" in checks and len(all_outputs) >= 2:
            if verbose:
                print("  → Detecting contradictions...")
            contradictions = []
            # Sample pairs to avoid O(n^2) API calls on large matrices
            pairs = list(combinations(range(len(all_outputs)), 2))
            sample_pairs = pairs[:min(10, len(pairs))]  # cap at 10 pair checks

            for idx_a, idx_b in sample_pairs:
                check = self._judge.check_contradiction(
                    question=test_case.input,
                    output_a=all_outputs[idx_a],
                    output_b=all_outputs[idx_b],
                )
                if check.get("contradicts"):
                    contradictions.append({
                        "output_a": all_outputs[idx_a],
                        "output_b": all_outputs[idx_b],
                        "type": check.get("contradiction_type"),
                        "explanation": check.get("explanation"),
                    })

            result.contradictions_found = contradictions
            result.contradiction_score = len(contradictions) / len(sample_pairs) if sample_pairs else 0.0

            # Extract failure pattern if contradictions found
            if contradictions and all_outputs:
                stable_answer = all_outputs[0]  # treat first (original) as baseline
                divergent_answer = contradictions[0]["output_b"]
                if verbose:
                    print("  → Extracting failure patterns...")
                pattern = self._judge.extract_failure_pattern(
                    question=test_case.input,
                    paraphrases=all_questions,
                    outputs=all_outputs,
                    stable_answer=stable_answer,
                    divergent_answer=divergent_answer,
                )
                if pattern:
                    result.failure_patterns.append(pattern)

        # 5. Grounding check (only if context provided)
        if "grounding" in checks and test_case.context:
            if verbose:
                print("  → Evaluating grounding...")
            grounding_scores = []
            # Check grounding for a sample of outputs
            for output in all_outputs[:5]:
                g = self._judge.evaluate_grounding(
                    question=test_case.input,
                    answer=output,
                    context=test_case.context,
                )
                grounding_scores.append(g.get("grounding_score", 0.5))
            result.grounding_score = (
                sum(grounding_scores) / len(grounding_scores)
                if grounding_scores else None
            )

        # 6. Compute risk level
        result.risk = self._compute_risk(result)

        return result

    def _compute_risk(self, result: TestResult) -> RiskLevel:
        scores = []
        if result.consistency_score is not None:
            scores.append(result.consistency_score)
        if result.contradiction_score is not None:
            # Invert: high contradiction = low score for risk calc
            scores.append(1.0 - result.contradiction_score)
        if result.grounding_score is not None:
            scores.append(result.grounding_score)

        if not scores:
            return RiskLevel.LOW

        avg = sum(scores) / len(scores)
        if avg >= 0.85:
            return RiskLevel.LOW
        elif avg >= 0.65:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
