"""
RegressionSuite: compare baseline vs candidate for CI/CD gate.

Detects CAI regressions when you update a prompt, swap models, or refactor
your LLM pipeline. Drop .fail_if_below() into GitHub Actions to block merges
that degrade consistency.

Example:
    from contradish import RegressionSuite, TestCase

    suite = RegressionSuite(
        test_cases=[
            TestCase(input="Can I get a refund after 45 days?"),
            TestCase(input="What is your return policy?"),
        ]
    )

    result = suite.compare(
        baseline_app=old_app,
        candidate_app=new_app,
        baseline_label="prod-v12",
        candidate_label="branch-refactor",
    )

    print(result)
    result.fail_if_below(consistency=0.80)  # raises AssertionError in CI if score drops
"""

import json
import os
from typing import Callable, Optional

from .models import TestCase, RegressionResult
from .suite import Suite


class RegressionSuite:
    """
    Compare two versions of your LLM app to catch CAI regressions before deploy.

    Usage in CI (GitHub Actions):
        result = suite.compare(baseline_app, candidate_app)
        result.fail_if_below(consistency=0.80)

    Usage for manual comparison:
        result = suite.compare(old_app, new_app)
        print(result)           # CAI delta + pass/fail
        print(result.to_dict()) # JSON for dashboards
    """

    def __init__(
        self,
        test_cases: list[TestCase],
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.test_cases = test_cases
        self.api_key    = api_key
        self.provider   = provider

    @classmethod
    def load(cls, path: str, api_key: Optional[str] = None) -> "RegressionSuite":
        """
        Load test cases from a YAML or JSON file.

        YAML format:
            test_cases:
              - input: "Can I get a refund after 45 days?"
                name: "refund policy"

        JSON format:
            [{"input": "...", "name": "..."}]
        """
        with open(path) as f:
            content = f.read()

        if path.endswith((".yaml", ".yml")):
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
                expected_traits=tc.get("expected_traits", []),
            )
            for tc in raw_cases
        ]
        return cls(test_cases=test_cases, api_key=api_key)

    def compare(
        self,
        baseline_app:    Callable[[str], str],
        candidate_app:   Callable[[str], str],
        baseline_label:  str = "baseline",
        candidate_label: str = "candidate",
        paraphrases:     int = 5,
        verbose:         bool = True,
    ) -> RegressionResult:
        """
        Run both apps against all test cases and return a RegressionResult.

        Args:
            baseline_app:    The current production app callable.
            candidate_app:   The new version to test.
            baseline_label:  Human-readable label (e.g. "prod-v12").
            candidate_label: Human-readable label (e.g. "pr-456").
            paraphrases:     Adversarial variants per test case.
            verbose:         Print progress.

        Returns:
            RegressionResult with CAI delta and .fail_if_below() helper.
        """
        if verbose:
            print(f"\nRunning baseline ({baseline_label})...")

        baseline_suite = Suite(app=baseline_app, api_key=self.api_key, provider=self.provider)
        for tc in self.test_cases:
            baseline_suite.add(tc)
        baseline_report = baseline_suite.run(paraphrases=paraphrases, verbose=verbose)

        if verbose:
            print(f"\nRunning candidate ({candidate_label})...")

        candidate_suite = Suite(app=candidate_app, api_key=self.api_key, provider=self.provider)
        for tc in self.test_cases:
            candidate_suite.add(tc)
        candidate_report = candidate_suite.run(paraphrases=paraphrases, verbose=verbose)

        return RegressionResult(
            baseline_label=baseline_label,
            candidate_label=candidate_label,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
        )
