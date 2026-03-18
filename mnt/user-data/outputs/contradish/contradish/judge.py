"""
Judge layer — uses Claude to evaluate consistency, contradictions, and grounding.
"""

import json
import re
from typing import Optional
import anthropic

from .models import FailurePattern


CONSISTENCY_PROMPT = """You are an expert evaluator of LLM output consistency.

You will be given a question and multiple answers to that question from the same system.
Your job is to evaluate how consistent the answers are with each other.

Question: {question}

Answers:
{answers}

Evaluate:
1. Do all answers reach the same conclusion?
2. Are there any factual disagreements between answers?
3. Do any answers contradict each other on key points?

Respond ONLY with valid JSON in this exact format:
{{
  "consistency_score": <float 0.0-1.0, where 1.0 = perfectly consistent>,
  "consistent": <true/false>,
  "disagreements": [<list of strings describing specific disagreements>],
  "notes": "<brief explanation>"
}}"""


CONTRADICTION_PROMPT = """You are an expert at detecting logical contradictions.

You will be given a question and two answers. Determine if they contradict each other.

Question: {question}
Answer A: {output_a}
Answer B: {output_b}

Respond ONLY with valid JSON:
{{
  "contradicts": <true/false>,
  "contradiction_type": "<factual|logical|policy|none>",
  "explanation": "<what specifically contradicts, or 'none' if no contradiction>"
}}"""


GROUNDING_PROMPT = """You are an expert at evaluating whether an LLM answer is grounded in provided context.

Context (retrieved documents):
{context}

Question: {question}
Answer: {answer}

Evaluate:
1. Does the answer only use information present in the context?
2. Does the answer invent or hallucinate facts not in the context?
3. Does the answer correctly represent what the context says?

Respond ONLY with valid JSON:
{{
  "grounding_score": <float 0.0-1.0, where 1.0 = fully grounded>,
  "grounded": <true/false>,
  "hallucinated_claims": [<list of specific claims not supported by context>],
  "notes": "<brief explanation>"
}}"""


PATTERN_PROMPT = """You are an expert at diagnosing LLM failure patterns.

A question was paraphrased multiple times and sent to an LLM. Some answers disagreed with each other.

Original question: {question}

Paraphrases that led to different answers:
{divergent_paraphrases}

Stable answers said: {stable_answer}
Divergent answers said: {divergent_answer}

What input pattern is causing the instability? Be specific and actionable.

Respond ONLY with valid JSON:
{{
  "pattern": "<what type of phrasing triggers the divergence>",
  "issue": "<what the model is doing wrong>",
  "suggestion": "<what the developer should fix>"
}}"""


class Judge:
    """
    Calls Claude to evaluate outputs for consistency, contradiction, and grounding.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _call(self, prompt: str) -> dict:
        """Make a single judge call and parse JSON response."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    def evaluate_consistency(self, question: str, outputs: list[str]) -> dict:
        """
        Evaluate consistency across multiple outputs for the same question.
        Returns dict with consistency_score, consistent, disagreements, notes.
        """
        formatted_answers = "\n".join(
            f"[{i+1}] {o}" for i, o in enumerate(outputs)
        )
        prompt = CONSISTENCY_PROMPT.format(
            question=question,
            answers=formatted_answers,
        )
        try:
            return self._call(prompt)
        except Exception as e:
            return {
                "consistency_score": 0.5,
                "consistent": False,
                "disagreements": [],
                "notes": f"Judge error: {str(e)}",
            }

    def check_contradiction(self, question: str, output_a: str, output_b: str) -> dict:
        """
        Check if two specific outputs contradict each other.
        Returns dict with contradicts, contradiction_type, explanation.
        """
        prompt = CONTRADICTION_PROMPT.format(
            question=question,
            output_a=output_a,
            output_b=output_b,
        )
        try:
            return self._call(prompt)
        except Exception as e:
            return {
                "contradicts": False,
                "contradiction_type": "none",
                "explanation": f"Judge error: {str(e)}",
            }

    def evaluate_grounding(
        self, question: str, answer: str, context: str
    ) -> dict:
        """
        Evaluate whether an answer is grounded in retrieved context.
        Returns dict with grounding_score, grounded, hallucinated_claims, notes.
        """
        prompt = GROUNDING_PROMPT.format(
            context=context,
            question=question,
            answer=answer,
        )
        try:
            return self._call(prompt)
        except Exception as e:
            return {
                "grounding_score": 0.5,
                "grounded": False,
                "hallucinated_claims": [],
                "notes": f"Judge error: {str(e)}",
            }

    def extract_failure_pattern(
        self,
        question: str,
        paraphrases: list[str],
        outputs: list[str],
        stable_answer: str,
        divergent_answer: str,
    ) -> Optional[FailurePattern]:
        """
        Given a set of paraphrases and their outputs, extract what input pattern
        is causing instability.
        """
        # Find which paraphrases led to divergent answers
        divergent_paraphrases = []
        for p, o in zip(paraphrases, outputs):
            result = self.check_contradiction(question, stable_answer, o)
            if result.get("contradicts"):
                divergent_paraphrases.append(f"Paraphrase: {p}\nAnswer: {o}")

        if not divergent_paraphrases:
            return None

        prompt = PATTERN_PROMPT.format(
            question=question,
            divergent_paraphrases="\n\n".join(divergent_paraphrases),
            stable_answer=stable_answer,
            divergent_answer=divergent_answer,
        )
        try:
            result = self._call(prompt)
            return FailurePattern(
                pattern=result.get("pattern", "unknown"),
                issue=result.get("issue", "unknown"),
                affected_runs=len(divergent_paraphrases),
                total_runs=len(outputs),
                examples=[
                    {"paraphrase": p, "output": o}
                    for p, o in zip(paraphrases, outputs)
                    if any(p in d for d in divergent_paraphrases)
                ][:3],
            )
        except Exception:
            return None
