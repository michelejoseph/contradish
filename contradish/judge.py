"""
Judge — uses an LLM to evaluate consistency and detect contradictions.
"""

from typing import Optional
from .llm import LLMClient
from .models import ContradictionPair


_CONSISTENCY_PROMPT = """You are evaluating whether an AI system gives consistent answers.

The same question was asked in {n} different phrasings. All phrasings mean the same thing.
Your job: do the answers agree with each other?

Question (original): {question}

Answers:
{answers}

Evaluate carefully:
- Do all answers reach the same conclusion?
- Are there any factual disagreements?
- Do any answers contradict each other?

Respond ONLY with JSON (no markdown, no preamble):
{{
  "consistency_score": <float 0.0-1.0>,
  "all_consistent": <true|false>,
  "disagreements": ["<specific disagreement 1>", ...],
  "summary": "<one sentence: what is (in)consistent>"
}}"""


_CONTRADICTION_PROMPT = """You are a contradiction detector for AI systems.

The same question was asked in two different phrasings. Determine if the answers contradict each other.

Question: {question}
Phrasing A: {input_a}
Answer A:   {output_a}

Phrasing B: {input_b}
Answer B:   {output_b}

A contradiction means the answers make incompatible factual or logical claims.
Differences in tone, length, or wording alone are NOT contradictions.

Respond ONLY with JSON (no markdown, no preamble):
{{
  "contradicts": <true|false>,
  "severity": "<factual|logical|policy|none>",
  "explanation": "<what specifically contradicts, or 'none'>"
}}"""


_PATTERN_PROMPT = """An AI system gives inconsistent answers when the same question is phrased differently.

Original question: {question}

These phrasings produced DIFFERENT answers from the rest:
{divergent_cases}

The majority of answers said: "{majority_answer}"

Diagnose what is happening and give the developer a specific fix.

Respond ONLY with JSON (no markdown, no preamble):
{{
  "pattern": "<what input pattern triggers the instability — be specific, e.g. 'casual phrasing without explicit dates', 'emotional framing', 'presupposing a yes'>",
  "root_cause": "<one sentence on why the model fails — e.g. 'the model prioritizes helpfulness over policy when no time constraint is mentioned explicitly'>",
  "suggestion": "<a single sentence to add verbatim to the system prompt that would prevent this contradiction. Start with an imperative verb. Be specific to this rule, not generic. Example: 'Always state the 30-day refund limit explicitly in every response, regardless of how the request is phrased or what emotional context the user provides.'>"
}}"""


class Judge:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate_consistency(self, question: str, inputs: list[str], outputs: list[str]) -> dict:
        """
        Score how consistent a set of outputs are for semantically equivalent inputs.
        Returns dict with consistency_score, all_consistent, disagreements, summary.
        """
        formatted = "\n".join(
            f"  [{i+1}] (phrased as: \"{inp[:60]}\")\n      → {out[:200]}"
            for i, (inp, out) in enumerate(zip(inputs, outputs))
        )
        prompt = _CONSISTENCY_PROMPT.format(
            n=len(outputs),
            question=question,
            answers=formatted,
        )
        result = self.llm.complete_json(prompt)
        return {
            "consistency_score": float(result.get("consistency_score", 0.5)),
            "all_consistent":    bool(result.get("all_consistent", False)),
            "disagreements":     result.get("disagreements", []),
            "summary":           result.get("summary", ""),
        }

    def find_contradictions(
        self,
        question: str,
        inputs:   list[str],
        outputs:  list[str],
        max_pairs: int = 8,
    ) -> list[ContradictionPair]:
        """
        Check pairs of outputs for direct contradictions.
        Caps at max_pairs to control API cost.
        """
        from itertools import combinations
        contradictions = []

        pairs = list(combinations(range(len(outputs)), 2))
        # Prioritise pairs furthest apart in index (most likely to diverge)
        pairs = sorted(pairs, key=lambda p: p[1] - p[0], reverse=True)[:max_pairs]

        for i, j in pairs:
            prompt = _CONTRADICTION_PROMPT.format(
                question=question,
                input_a=inputs[i],  output_a=outputs[i],
                input_b=inputs[j],  output_b=outputs[j],
            )
            result = self.llm.complete_json(prompt)
            if result.get("contradicts"):
                contradictions.append(ContradictionPair(
                    input_a=inputs[i],  output_a=outputs[i],
                    input_b=inputs[j],  output_b=outputs[j],
                    explanation=result.get("explanation", ""),
                    severity=result.get("severity", "unknown"),
                ))
        return contradictions

    def extract_pattern(
        self,
        question:        str,
        inputs:          list[str],
        outputs:         list[str],
        contradictions:  list[ContradictionPair],
    ) -> Optional[dict]:
        """
        Given confirmed contradictions, diagnose which input patterns cause them
        and what the developer should do about it.
        """
        if not contradictions:
            return None

        # Find majority answer (most common output cluster by first sentence)
        from collections import Counter
        first_sentences = [o.split(".")[0].strip() for o in outputs]
        majority_answer = Counter(first_sentences).most_common(1)[0][0]

        divergent_cases = "\n\n".join(
            f"  Phrasing: \"{c.input_b}\"\n  Answer:   \"{c.output_b[:200]}\"\n  Conflict: {c.explanation}"
            for c in contradictions[:3]
        )

        prompt = _PATTERN_PROMPT.format(
            question=question,
            divergent_cases=divergent_cases,
            majority_answer=majority_answer[:200],
        )
        return self.llm.complete_json(prompt)
