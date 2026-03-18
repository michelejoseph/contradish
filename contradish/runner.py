"""
Runner — generates semantic paraphrases and executes the test matrix.
"""

from typing import Callable
from .llm import LLMClient


_PARAPHRASE_PROMPT = """Generate {n} semantically equivalent paraphrases of the following question.

Rules:
- Preserve the EXACT meaning and intent — do not add, remove, or change any information
- Vary sentence structure, vocabulary, and phrasing as a real user would
- Each paraphrase should feel natural, not mechanical

Question: {question}

Return ONLY a JSON array of strings with no preamble or markdown.
Example: ["version 1", "version 2", "version 3"]"""


class Runner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_paraphrases(self, question: str, n: int) -> list[str]:
        """Generate n semantically equivalent paraphrases."""
        prompt = _PARAPHRASE_PROMPT.format(n=n, question=question)
        try:
            result = self.llm.complete_json(prompt, model=self.llm.fast_model)
            if isinstance(result, list):
                return [str(p) for p in result[:n]]
        except Exception:
            pass
        # Fallback: return original repeated (degraded but non-crashing)
        return [question] * n

    def run_matrix(
        self,
        app:         Callable[[str], str],
        original:    str,
        paraphrases: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Call app on [original] + paraphrases.
        Returns (inputs, outputs) as parallel lists.
        """
        inputs  = [original] + paraphrases
        outputs = []
        for inp in inputs:
            try:
                out = app(inp)
                outputs.append(str(out).strip())
            except Exception as e:
                outputs.append(f"[APP ERROR: {e}]")
        return inputs, outputs
