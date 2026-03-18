"""
Runner — generates paraphrases and executes the test matrix.
"""

import re
import json
from typing import Callable
import anthropic


PARAPHRASE_PROMPT = """Generate {n} semantically equivalent paraphrases of the following question.

Rules:
- Preserve the exact meaning and intent
- Vary sentence structure, word choice, and phrasing naturally
- Do not add or remove information
- Each paraphrase should feel like a different user asking the same thing

Question: {question}

Respond ONLY with a JSON array of strings, no other text.
Example: ["paraphrase 1", "paraphrase 2"]"""


class Runner:
    """
    Generates paraphrases of inputs and runs the app callable across all variants.
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        # Use Haiku for paraphrase generation (faster, cheaper)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_paraphrases(self, question: str, n: int = 5) -> list[str]:
        """Generate n semantically equivalent paraphrases of the question."""
        prompt = PARAPHRASE_PROMPT.format(n=n, question=question)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            paraphrases = json.loads(raw)
            return paraphrases[:n]
        except Exception:
            # Fallback: return original question repeated
            return [question] * n

    def run_matrix(
        self,
        app: Callable[[str], str],
        question: str,
        paraphrases: list[str],
        repeats: int = 1,
    ) -> tuple[list[str], list[str]]:
        """
        Run the app across all paraphrases (optionally repeated).
        Returns (all_questions, all_outputs).
        """
        all_questions = []
        all_outputs = []

        # Always include the original question
        inputs = [question] + paraphrases

        for inp in inputs:
            for _ in range(repeats):
                try:
                    output = app(inp)
                    all_questions.append(inp)
                    all_outputs.append(str(output))
                except Exception as e:
                    all_questions.append(inp)
                    all_outputs.append(f"[ERROR: {str(e)}]")

        return all_questions, all_outputs
