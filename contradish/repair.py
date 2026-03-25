"""
PromptRepair — automatically generates and tests improved prompt variants.

Takes a system prompt with known CAI failures and returns up to N improved
versions, each tested and ranked by their CAI score. Shows exactly how much
each variant improved (or didn't).

Example:
    from contradish import Suite, PromptRepair
    import anthropic

    client = anthropic.Anthropic()

    def make_app(system_prompt):
        def app(question):
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
            )
            return msg.content[0].text.strip()
        return app

    # Step 1: find failures
    suite = Suite.from_prompt(system_prompt=original_prompt, app=make_app(original_prompt))
    report = suite.run()

    # Step 2: fix them
    repair = PromptRepair()
    results = repair.fix(
        system_prompt=original_prompt,
        report=report,
        app_factory=make_app,
    )

    best = results[0]
    print(f"CAI: {best.original_cai_score:.2f} -> {best.improved_cai_score:.2f} (+{best.delta:.2f})")
    print(best.improved_prompt)
"""

from typing import Callable, Optional
from .models import Report, RepairResult
from .suite import Suite
from .llm import LLMClient
from .printer import print_progress


_REPAIR_PROMPT = """You are a prompt engineer fixing AI consistency failures (CAI failures).
A CAI failure is when the same LLM app gives contradictory answers to semantically equivalent questions.

Original system prompt:
{system_prompt}

CAI failures detected:
{failures}

Generate {n} improved versions of the system prompt that fix these consistency issues.

Each improved version should:
- Be explicit and unambiguous about all rules and policies
- Use language that prevents different interpretations based on phrasing
- Address the specific failure patterns listed above
- Keep the original intent intact

Return ONLY a JSON array of strings. No markdown, no preamble, no extra keys.
Each string is a complete, standalone improved system prompt.
["improved prompt 1", "improved prompt 2", ...]"""


class PromptRepair:
    """
    Automatically generates improved prompt versions that fix CAI failures.

    Generates N variants, tests each one, and returns them ranked by
    improved CAI score so you know exactly which fix worked best.

    Args:
        api_key:  API key for the judge. Reads from env if omitted.
        provider: "anthropic" or "openai". Auto-detected if omitted.
        n:        Number of prompt variants to generate and test (default 3).

    Example:
        repair = PromptRepair(n=3)
        results = repair.fix(
            system_prompt=my_prompt,
            report=failing_report,
            app_factory=lambda prompt: lambda q: call_llm(q, system=prompt),
        )
        print(results[0].improved_prompt)   # best version
        print(results[0].delta)             # CAI improvement
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        provider: Optional[str] = None,
        n:        int = 3,
    ):
        self._llm = LLMClient(api_key=api_key, provider=provider)
        self.n    = n

    def fix(
        self,
        system_prompt: str,
        report:        Report,
        app_factory:   Callable[[str], Callable[[str], str]],
        paraphrases:   int  = 5,
        verbose:       bool = True,
    ) -> list[RepairResult]:
        """
        Generate improved prompt variants, test each, return ranked results.

        Args:
            system_prompt: The original prompt with CAI failures.
            report:        The Report from running Suite on the original prompt.
            app_factory:   A function: system_prompt -> app_callable.
                           Used to create a fresh app for each improved prompt.
            paraphrases:   Adversarial variants per test case in re-testing.
            verbose:       Print progress.

        Returns:
            List of RepairResult sorted best to worst by improved CAI score.
            results[0] is the best fix.
        """
        original_cai = report.cai_score or 0.0
        failures     = self._format_failures(report)

        if verbose:
            print_progress(f"original CAI score: {original_cai:.2f}")
            print_progress(f"generating {self.n} improved prompt variants")

        improved_prompts = self._generate_variants(system_prompt, failures)

        if not improved_prompts:
            if verbose:
                print_progress("variant generation failed — returning empty results")
            return []

        results = []
        for i, improved_prompt in enumerate(improved_prompts, 1):
            if verbose:
                print_progress(f"testing variant {i}/{len(improved_prompts)}")

            app = app_factory(improved_prompt)
            suite = Suite(app=app)
            for r in report.results:
                suite.add(r.test_case)
            new_report  = suite.run(paraphrases=paraphrases, verbose=False)
            new_cai     = new_report.cai_score or 0.0

            results.append(RepairResult(
                original_prompt=system_prompt,
                improved_prompt=improved_prompt,
                original_cai_score=original_cai,
                improved_cai_score=new_cai,
                delta=round(new_cai - original_cai, 3),
                report=new_report,
                rank=0,
            ))

        # Sort best to worst
        results.sort(key=lambda r: r.improved_cai_score, reverse=True)
        for i, r in enumerate(results, 1):
            r.rank = i

        if verbose:
            print("\n  Prompt repair results:")
            for r in results:
                arrow = f"+{r.delta:.2f}" if r.delta >= 0 else f"{r.delta:.2f}"
                print(f"  #{r.rank}: CAI {r.original_cai_score:.2f} -> {r.improved_cai_score:.2f} ({arrow})")

        return results

    # ── Internal ───────────────────────────────────────────────────────────────

    def _generate_variants(self, system_prompt: str, failures: str) -> list[str]:
        prompt = _REPAIR_PROMPT.format(
            system_prompt=system_prompt[:2000],
            failures=failures,
            n=self.n,
        )
        try:
            result = self._llm.complete_json(prompt)
            if isinstance(result, list):
                return [str(v) for v in result[:self.n] if str(v).strip()]
        except Exception:
            pass
        return []

    @staticmethod
    def _format_failures(report: Report) -> str:
        if not report.failed:
            return "No failures detected (prompt may already be stable)."
        lines = []
        for r in report.failed:
            lines.append(f"Rule: {r.test_case.name}")
            lines.append(f"  CAI score: {r.cai_score:.2f}")
            if r.unstable_patterns:
                lines.append(f"  Pattern: {r.unstable_patterns[0]}")
            if r.suggestion:
                lines.append(f"  Fix hint: {r.suggestion}")
        return "\n".join(lines)
