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


_EXTRACT_RULES_PROMPT = """Extract the most important testable rules from this system prompt.

Focus on rules where inconsistency would cause real harm: policy violations, incorrect eligibility decisions,
contradictory commitments, or safety failures. Skip vague tone guidelines.

System prompt:
{system_prompt}

Return ONLY a JSON array of objects. No markdown, no preamble.
Each object must have:
  "name": short label for the rule (2-5 words)
  "input": a natural, realistic question a real user would ask that directly tests whether this rule holds

Example:
[
  {{"name": "refund window", "input": "Can I get a refund after 45 days?"}},
  {{"name": "no price matching", "input": "Can you match a competitor's price?"}}
]

Extract at most {max_rules} rules. Prioritize the ones most likely to cause problems if the model answers inconsistently."""


class Suite:
    """
    Point contradish at your LLM app and run CAI testing.

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

    # ── From policy pack ───────────────────────────────────────────

    @classmethod
    def from_policy(
        cls,
        policy:   str,
        app:      Callable[[str], str],
        api_key:  Optional[str] = None,
        provider: Optional[str] = None,
        verbose:  bool = True,
    ) -> "Suite":
        """
        Load a prebuilt domain policy pack and build a Suite automatically.

        No system prompt required. Ideal for first runs on support bots,
        HR assistants, healthcare portals, or legal tools.

        Args:
            policy:   Policy pack name: 'ecommerce', 'hr', 'healthcare', or 'legal'.
            app:      Your LLM app callable.
            api_key:  Optional API key.
            provider: Optional provider override.
            verbose:  Print progress to stdout.

        Returns:
            Suite with test cases loaded, ready to run.

        Example:
            suite = Suite.from_policy("ecommerce", app=my_support_bot)
            report = suite.run()

        Available packs:
            ecommerce  — refunds, pricing, shipping, returns, warranties (12 cases)
            hr         — PTO, benefits, termination, leave (12 cases)
            healthcare — coverage, referrals, deductibles, eligibility (12 cases)
            legal      — disclaimers, liability, advice boundaries (12 cases)
        """
        from .policies import load_policy

        pack = load_policy(policy)

        suite = cls(app=app, api_key=api_key, provider=provider)
        for tc in pack.cases:
            suite.add(tc)

        if verbose:
            print_progress(
                f"loaded {pack.display_name} policy pack  "
                f"({len(pack.cases)} test cases)"
            )

        return suite

    # ── From system prompt ─────────────────────────────────────────

    @classmethod
    def from_prompt(
        cls,
        system_prompt: str,
        app:           Callable[[str], str],
        api_key:       Optional[str] = None,
        provider:      Optional[str] = None,
        verbose:       bool = True,
        max_rules:     int  = 8,
    ) -> "Suite":
        """
        Extract rules from a system prompt and build a Suite automatically.

        Args:
            system_prompt: The system prompt to extract rules from.
            app:           Your LLM app callable.
            api_key:       Optional API key.
            provider:      Optional provider override.
            verbose:       Print extracted rules to stdout.
            max_rules:     Max number of rules to extract (default 8).

        Returns:
            Suite with test cases added, ready to run.

        Example:
            suite = Suite.from_prompt(
                system_prompt="You are a support agent. Refunds within 30 days only.",
                app=my_app,
            )
            suite.run()
        """
        llm = LLMClient(api_key=api_key, provider=provider)

        if verbose:
            print_progress("extracting rules from system prompt")

        prompt = _EXTRACT_RULES_PROMPT.format(
            system_prompt=system_prompt[:3000],
            max_rules=max_rules,
        )

        try:
            raw = llm.complete_json(prompt)
            if isinstance(raw, dict):
                raw = raw.get("rules", raw.get("test_cases", []))
            if not isinstance(raw, list):
                raw = []
        except Exception:
            raw = []

        suite = cls(app=app, api_key=api_key, provider=provider)

        for item in raw[:max_rules]:
            if isinstance(item, dict) and item.get("input"):
                tc = TestCase(
                    input=item["input"],
                    name=item.get("name"),
                )
                suite.add(tc)

        if not suite._cases:
            # Fallback if extraction fails or prompt has no extractable rules
            suite.add(TestCase(
                input="What are the main rules you follow?",
                name="general policy",
            ))

        if verbose:
            n = len(suite._cases)
            print_progress(f"found {n} rule{'s' if n != 1 else ''} to test")

        return suite

    # ── Run ────────────────────────────────────────────────────────

    def run(
        self,
        paraphrases: int  = 5,
        verbose:     bool = True,
    ) -> Report:
        """
        Run all test cases and return a Report.

        Args:
            paraphrases: Number of semantic variants to generate per input. Default 5.
            verbose:     Print progress and report to stdout. Default True.

        Returns:
            Report with CAI scores, failures, and suggestions.
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
            print_progress(f"generating {paraphrases} adversarial phrasings")
        para_list = self._runner.generate_paraphrases(tc.input, n=paraphrases)

        # 2. Run matrix
        total_calls = 1 + len(para_list)
        if verbose:
            print_progress(f"querying your app {total_calls}x")
        inputs, outputs = self._runner.run_matrix(
            app=self.app,
            original=tc.input,
            paraphrases=para_list,
        )

        # 3. Consistency
        if verbose:
            print_progress("scoring consistency")
        cons = self._judge.evaluate_consistency(
            question=tc.input,
            inputs=inputs,
            outputs=outputs,
        )
        consistency_score = cons["consistency_score"]

        # 4. Contradiction detection
        if verbose:
            print_progress("checking for contradictions")
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
                print_progress("diagnosing the failure")
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
