"""
RegressionSuite — compare baseline vs candidate for CI/CD regression detection.
"""

import json
import os
from typing import Callable, Optional

from .models import TestCase, Report, RegressionResult
from .suite import Suite


class RegressionSuite:
    """
    Compare two versions of your LLM app to detect regressions.

    Example:
        from contradish import RegressionSuite, TestCase

        suite = RegressionSuite(
            api_key="sk-ant-...",
            test_cases=[
                TestCase(input="Can I get a refund after 45 days?"),
                TestCase(input="What is your return policy?"),
            ]
        )

        result = suite.compare(
            baseline_app=old_app,
            baseline_label="prod-v12",
            candidate_app=new_app,
            candidate_label="branch-refactor",
        )

        print(result)
        result.fail_if_below(consistency=0.85)  # raises AssertionError in CI if regressed
    """

    def __init__(
        self,
        api_key: str,
        test_cases: list[TestCase],
        judge_model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key
        self.test_cases = test_cases
        self.judge_model = judge_model

    @classmethod
    def load(cls, path: str, api_key: Optional[str] = None) -> "RegressionSuite":
        """
        Load test cases from a YAML or JSON file.

        YAML format:
            test_cases:
              - input: "Can I get a refund after 45 days?"
                name: "refund policy"
                context: "Refunds allowed within 30 days."

        JSON format:
            [{"input": "...", "name": "...", "context": "..."}]
        """
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        with open(path) as f:
            content = f.read()

        if path.endswith(".yaml") or path.endswith(".yml"):
            try:
                import yaml
                data = yaml.safe_load(content)
                raw_cases = data.get("test_cases", data) if isinstance(data, dict) else data
            except ImportError:
                raise ImportError("Install pyyaml to load YAML files: pip install pyyaml")
        else:
            raw_cases = json.loads(content)
            if isinstance(raw_cases, dict):
                raw_cases = raw_cases.get("test_cases", [])

        test_cases = [
            TestCase(
                input=tc["input"],
                name=tc.get("name"),
                context=tc.get("context"),
                expected_traits=tc.get("expected_traits", []),
            )
            for tc in raw_cases
        ]
        return cls(api_key=resolved_key, test_cases=test_cases)

    def compare(
        self,
        baseline_app: Callable[[str], str],
        candidate_app: Callable[[str], str],
        baseline_label: str = "baseline",
        candidate_label: str = "candidate",
        paraphrases: int = 5,
        verbose: bool = True,
    ) -> RegressionResult:
        """
        Run both apps against the same test cases and return a RegressionResult.

        Args:
            baseline_app: The current production app callable.
            candidate_app: The new candidate app callable.
            baseline_label: Human-readable label for baseline (e.g. "prod-v12").
            candidate_label: Human-readable label for candidate (e.g. "branch-refactor").
            paraphrases: Number of paraphrases per test case.
            verbose: Print progress.

        Returns:
            RegressionResult with delta scores and pass/fail helpers.
        """
        if verbose:
            print(f"\nRunning baseline ({baseline_label})...")
        baseline_suite = Suite(
            api_key=self.api_key,
            app=baseline_app,
            judge_model=self.judge_model,
        )
        for tc in self.test_cases:
            baseline_suite.add_test(tc)
        baseline_report = baseline_suite.run(paraphrases=paraphrases, verbose=verbose)

        if verbose:
            print(f"\nRunning candidate ({candidate_label})...")
        candidate_suite = Suite(
            api_key=self.api_key,
            app=candidate_app,
            judge_model=self.judge_model,
        )
        for tc in self.test_cases:
            candidate_suite.add_test(tc)
        candidate_report = candidate_suite.run(paraphrases=paraphrases, verbose=verbose)

        return RegressionResult(
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
        )
