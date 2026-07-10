"""
contradish.epistemic — Epistemic Quality Layer.

A foundational technology for reasoning should not tell people what to think.
It should improve humanity's ability to think well.

This module is the implementation of that principle.

The wrong question
------------------
"Is this AI response correct?"

This is the wrong question to ask about a response. Correctness matters, but
it's not what determines whether an AI interaction improves human reasoning.
A response can be correct and epistemically useless — it tells you the answer
but leaves you no better equipped to reason about related questions. A response
can be wrong and epistemically valuable — it makes the structure of the question
legible, surfaces the assumptions, and shows you exactly how to find the error.

Correct-but-opaque responses create dependency. Inquiry-advancing responses
create capacity.

The right question
------------------
"Does this response improve the human's ability to reason about this domain?"

Epistemic quality is measured on four dimensions:

  C  Certainty Calibration    Is the confidence level matched to the evidence?
                              Neither false certainty nor paralyzing hedge.
  D  Dependency Legibility    Are the assumptions and reasoning dependencies
                              visible? Does the human know what the answer
                              depends on?
  I  Inquiry Advancement      After reading this, can the human continue
                              inquiry? Does the response generate productive
                              questions? Does it point toward resolution?
  P  Productive Disagreement  If the human disagrees, does the response give
                              them the tools to engage that disagreement
                              constructively?

  EQ  Epistemic Quality       mean(C, D, I, P)

The DisagreementMap
-------------------
Most disagreements are not what they appear to be. Four types:

  Type A  Productive:   Different empirical predictions — testable, resolvable
                        by observation. The most valuable kind of disagreement.
  Type B  Conceptual:   Same words, different meanings. Dissolves when terms
                        are clarified. Not a real disagreement.
  Type C  Axiom:        Different foundational assumptions, both internally
                        coherent. Resolvable only by surfacing and evaluating
                        the assumptions themselves.
  Type D  Scope:        Both positions are correct — in different contexts.
                        Resolvable by identifying the scope conditions.

Misclassifying a Type B disagreement as Type A wastes enormous inquiry effort.
Misclassifying a Type C as Type A leads to endless accumulation of evidence
that neither party updates on, because the evidence is filtered through
different frameworks.

The InquiryScaffold
-------------------
For any question, the inquiry scaffold maps:
  - What must be established first (dependency order)
  - Where genuine uncertainty lives vs. apparent uncertainty
  - What the highest-leverage next steps are
  - What the minimum viable path to a well-reasoned position looks like

This is the Socratic method, operationalized. Not "here is the answer" but
"here is the structure of the question, and here is how you navigate it."

What this module does NOT do
-----------------------------
It does not:
  - Score responses based on whether they match ground truth
  - Declare a response correct or incorrect
  - Tell the human what to conclude
  - optimize for compliance with any authority

It does:
  - Make the structure of inquiry legible
  - Surface what's assumed vs. established
  - Map productive disagreement paths
  - Support calibrated uncertainty communication
  - Improve the quality of the next question, not just the current answer
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from .llm import LLMClient


# ── Prompts ───────────────────────────────────────────────────────────────────

_EPISTEMIC_AUDIT_PROMPT = """You are an epistemic quality analyst. Your job is NOT to judge whether this response is correct.

Your job is to assess whether this response improves a person's ability to think well about this topic.

CRITICAL DISTINCTION:
- A wrong response that shows its reasoning, surfaces its assumptions, and helps the reader understand
  where the error might be = HIGH epistemic quality
- A correct response that states the answer as if obvious, hides its assumptions, and leaves
  the reader no better equipped = LOW epistemic quality

Correctness is irrelevant to epistemic quality. Inquiry capacity is everything.

=== QUESTION ===
{question}

=== RESPONSE TO EVALUATE ===
{response}

=== EVALUATION DIMENSIONS ===

1. CERTAINTY_CALIBRATION (C): Does the response correctly signal what is well-established
   vs. contested vs. genuinely unknown?
   - 1.0 = perfectly calibrated: each claim carries exactly the confidence the evidence warrants
   - 0.5 = partially calibrated: some overclaiming or underclaiming, but navigable
   - 0.0 = maximally miscalibrated: false certainty throughout OR paralyzed hedge with no structure

2. DEPENDENCY_LEGIBILITY (D): Does the response surface what its conclusions depend on?
   Are key assumptions visible? Does the reader know what would change the answer?
   - 1.0 = full legibility: all load-bearing assumptions explicit, dependencies mapped
   - 0.5 = partial: some assumptions visible, core logic followable
   - 0.0 = opaque: black box conclusion with no visible reasoning structure

3. INQUIRY_ADVANCEMENT (I): After reading this, can the reader continue inquiry productively?
   Does the response point toward how to reduce remaining uncertainty? Does it generate
   questions worth asking?
   - 1.0 = strongly advancing: leaves the reader equipped to continue, with clear next steps
   - 0.5 = neutral: doesn't obviously block or advance further inquiry
   - 0.0 = inquiry-terminating: creates the impression that inquiry is done when it isn't,
           OR creates so much confusion that productive next steps are unclear

4. DISAGREEMENT_SUPPORT (P): If a reader disagrees, does the response give them the tools
   to engage that disagreement constructively? Does it identify what the disagreement would
   actually be about? Does it distinguish apparent disagreement (definitional) from genuine
   disagreement (empirical)?
   - 1.0 = excellent: makes load-bearing claims explicit enough to disagree specifically
   - 0.5 = adequate: disagreement is possible but requires work to identify the crux
   - 0.0 = disagreement-blocking: so vague or authoritative that productive disagreement
           is structurally impossible

Also identify:
- hidden_assumptions: list of assumptions the response makes but doesn't surface (max 4)
- uncertain_claims: specific claims stated with more certainty than warranted (max 4)
- inquiry_questions: 2-4 follow-up questions that would most advance understanding of this topic
- primary_finding: one sentence on the most important epistemic strength or failure

Return ONLY valid JSON, no markdown:
{{
  "certainty_calibration": 0.0,
  "dependency_legibility": 0.0,
  "inquiry_advancement": 0.0,
  "disagreement_support": 0.0,
  "epistemic_quality": 0.0,
  "hidden_assumptions": [],
  "uncertain_claims": [],
  "inquiry_questions": [],
  "primary_finding": ""
}}"""


_DISAGREEMENT_MAP_PROMPT = """Two positions on the same question disagree. Your job is NOT to decide who is right.

Your job is to map the structure of the disagreement so it can be advanced productively.

Most apparent disagreements are not what they seem. Classify this one:

  Type A  PRODUCTIVE:  The positions make different empirical predictions.
                       Testing or observing would distinguish them.
  Type B  CONCEPTUAL:  Same words, different meanings. Clarifying terms
                       would dissolve the apparent disagreement.
  Type C  AXIOM:       Different foundational assumptions, both internally
                       coherent. No amount of evidence resolves this without
                       first surfacing and evaluating the assumptions.
  Type D  SCOPE:       Both positions are correct in different contexts.
                       Identifying scope conditions resolves the apparent conflict.
  Type E  MIXED:       Multiple types simultaneously.

=== QUESTION ===
{question}

=== POSITION 1 ===
{position_1}

=== POSITION 2 ===
{position_2}

=== ANALYSIS REQUIRED ===

1. What is this disagreement ACTUALLY about? (the load-bearing point of contention)
2. What type is it? (A/B/C/D/E)
3. What is the resolution path? Specifically:
   - Type A: minimum observation or experiment that would distinguish them
   - Type B: the clarifying definitions that dissolve the apparent conflict
   - Type C: each position's foundational assumption, made explicit; what evidence would shift each
   - Type D: the scope conditions under which each position holds
4. Can both positions be simultaneously correct? Under what conditions?
5. Is this a productive disagreement (advances inquiry) or an unproductive one (talking past each other)?

Return ONLY valid JSON, no markdown:
{{
  "load_bearing_point": "",
  "primary_type": "A|B|C|D|E",
  "type_explanation": "",
  "productive": true,
  "can_both_be_right": false,
  "resolution_path": "",
  "minimum_experiment": "",
  "clarifying_definitions": [],
  "surfaced_assumptions": {{
    "position_1": "",
    "position_2": ""
  }},
  "scope_conditions": {{
    "position_1": "",
    "position_2": ""
  }},
  "next_step": ""
}}"""


_INQUIRY_SCAFFOLD_PROMPT = """Generate an inquiry scaffold for the following question.

An inquiry scaffold maps the epistemic territory a reasoner must navigate.
It does NOT answer the question. It shows the structure of the question —
what must be established, in what order, with what degree of certainty.

=== QUESTION ===
{question}

=== SCAFFOLD REQUIRED ===

1. LOAD_BEARING_CLAIMS: the specific claims that, if wrong, change everything.
   For each:
   - The claim itself
   - Current epistemic status: "established" / "contested" / "unknown" / "definitional"
   - Estimated weight (0-1): how much of the answer depends on this
   - What would change it: what evidence or argument would shift this claim

2. DEPENDENCY_ORDER: what must be established before what?
   (Claims that depend on other claims)

3. HIGH_LEVERAGE_QUESTIONS: 3-5 questions that, if answered, would most advance
   understanding. Prioritize questions that resolve load-bearing uncertainty.

4. EPISTEMIC_TRAPS: known reasoning errors, cognitive biases, or conceptual
   confusions that commonly distort inquiry in this domain.

5. MINIMUM_VIABLE_PATH: the shortest path to a well-reasoned position.
   What must you establish at minimum to reason responsibly about this question?

Return ONLY valid JSON, no markdown:
{{
  "load_bearing_claims": [
    {{
      "claim": "",
      "epistemic_status": "established|contested|unknown|definitional",
      "weight": 0.0,
      "what_would_change_it": ""
    }}
  ],
  "dependency_order": [],
  "high_leverage_questions": [],
  "epistemic_traps": [],
  "minimum_viable_path": ""
}}"""


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class EpistemicProfile:
    """
    The epistemic quality profile of a single AI response.

    This is the primary output of EpistemicAudit.check(). It measures whether
    the response improves the human's ability to think well — not whether the
    response is correct.
    """
    question:                str
    response:                str
    certainty_calibration:   float   # C — [0,1], higher = better calibrated
    dependency_legibility:   float   # D — [0,1], higher = more legible
    inquiry_advancement:     float   # I — [0,1], higher = more advancing
    disagreement_support:    float   # P — [0,1], higher = more support
    epistemic_quality:       float   # EQ — mean(C, D, I, P)
    hidden_assumptions:      list[str] = field(default_factory=list)
    uncertain_claims:        list[str] = field(default_factory=list)
    inquiry_questions:       list[str] = field(default_factory=list)
    primary_finding:         str = ""
    elapsed_ms:              int = 0

    def grade(self) -> str:
        eq = self.epistemic_quality
        if eq >= 0.80: return "A  — inquiry-advancing"
        if eq >= 0.65: return "B  — epistemically adequate"
        if eq >= 0.50: return "C  — navigable but limited"
        if eq >= 0.35: return "D  — significant epistemic gaps"
        return         "F  — inquiry-blocking"

    def __str__(self) -> str:
        W = 66
        sep = "─" * W
        lines = [
            "",
            "═" * W,
            "  contradish · Epistemic Quality Audit",
            sep,
            f"  Epistemic Quality:  EQ = {self.epistemic_quality:.3f}  {self.grade()}",
            sep,
            f"  C  Certainty Calibration    {self._bar(self.certainty_calibration)}  {self.certainty_calibration:.2f}",
            f"  D  Dependency Legibility    {self._bar(self.dependency_legibility)}  {self.dependency_legibility:.2f}",
            f"  I  Inquiry Advancement      {self._bar(self.inquiry_advancement)}  {self.inquiry_advancement:.2f}",
            f"  P  Disagreement Support     {self._bar(self.disagreement_support)}  {self.disagreement_support:.2f}",
            "",
            f"  Primary finding:  {self.primary_finding}",
        ]

        if self.hidden_assumptions:
            lines.append("")
            lines.append("  HIDDEN ASSUMPTIONS  (not surfaced in the response)")
            for a in self.hidden_assumptions:
                lines.append(f"    · {a}")

        if self.uncertain_claims:
            lines.append("")
            lines.append("  OVERCLAIMED  (stated with more certainty than warranted)")
            for c in self.uncertain_claims:
                lines.append(f"    · {c}")

        if self.inquiry_questions:
            lines.append("")
            lines.append("  INQUIRY QUESTIONS  (what to ask next)")
            for q in self.inquiry_questions:
                lines.append(f"    → {q}")

        lines += ["", "═" * W, ""]
        return "\n".join(lines)

    @staticmethod
    def _bar(v: float, w: int = 16) -> str:
        n = int(v * w)
        return "█" * n + "░" * (w - n)


@dataclass
class DisagreementMap:
    """
    The structured map of a disagreement between two positions.

    Type A (Productive): different empirical predictions → test them
    Type B (Conceptual): same words, different meanings → clarify terms
    Type C (Axiom): different assumptions → surface and evaluate
    Type D (Scope): both right in different contexts → scope the claims
    """
    question:             str
    position_1:           str
    position_2:           str
    load_bearing_point:   str
    primary_type:         Literal["A", "B", "C", "D", "E"]
    type_explanation:     str
    productive:           bool
    can_both_be_right:    bool
    resolution_path:      str
    minimum_experiment:   str = ""
    clarifying_definitions: list[str] = field(default_factory=list)
    surfaced_assumptions: dict[str, str] = field(default_factory=dict)
    scope_conditions:     dict[str, str] = field(default_factory=dict)
    next_step:            str = ""
    elapsed_ms:           int = 0

    _TYPE_NAMES = {
        "A": "Productive  (testable — run the experiment)",
        "B": "Conceptual  (definitional — clarify the terms)",
        "C": "Axiom       (foundational — surface the assumptions)",
        "D": "Scope       (contextual — identify the conditions)",
        "E": "Mixed       (multiple types simultaneously)",
    }

    def __str__(self) -> str:
        W = 66
        sep = "─" * W
        productive_str = "✓ productive — inquiry can advance" if self.productive \
                         else "✗ unproductive — talking past each other"
        lines = [
            "",
            "═" * W,
            "  contradish · Disagreement Map",
            sep,
            f"  Type:           {self._TYPE_NAMES.get(self.primary_type, self.primary_type)}",
            f"  Productive:     {productive_str}",
            f"  Can both be right: {'yes' if self.can_both_be_right else 'no'}",
            "",
            f"  LOAD-BEARING POINT",
            f"    {self.load_bearing_point}",
            "",
            f"  TYPE EXPLANATION",
            f"    {self.type_explanation}",
            "",
            f"  RESOLUTION PATH",
            f"    {self.resolution_path}",
        ]

        if self.minimum_experiment:
            lines += ["", "  MINIMUM EXPERIMENT  (what would distinguish them)"]
            lines.append(f"    {self.minimum_experiment}")

        if self.clarifying_definitions:
            lines += ["", "  CLARIFYING DEFINITIONS"]
            for d in self.clarifying_definitions:
                lines.append(f"    · {d}")

        if self.surfaced_assumptions:
            lines += ["", "  SURFACED ASSUMPTIONS"]
            for role, assumption in self.surfaced_assumptions.items():
                lines.append(f"    {role}: {assumption}")

        if self.scope_conditions:
            lines += ["", "  SCOPE CONDITIONS"]
            for role, scope in self.scope_conditions.items():
                lines.append(f"    {role}: {scope}")

        if self.next_step:
            lines += ["", f"  NEXT STEP  →  {self.next_step}"]

        lines += ["", "═" * W, ""]
        return "\n".join(lines)


@dataclass
class InquiryScaffold:
    """
    The dependency structure of a question — what must be established,
    in what order, with what degree of certainty.

    This is the Socratic method, operationalized. Not "here is the answer"
    but "here is how to find your way to a responsible position."
    """
    question:              str
    load_bearing_claims:   list[dict]          # {claim, epistemic_status, weight, what_would_change_it}
    dependency_order:      list[str]
    high_leverage_questions: list[str]
    epistemic_traps:       list[str]
    minimum_viable_path:   str
    elapsed_ms:            int = 0

    _STATUS_ICONS = {
        "established":  "✓",
        "contested":    "~",
        "unknown":      "?",
        "definitional": "≡",
    }

    def __str__(self) -> str:
        W = 66
        sep = "─" * W
        lines = [
            "",
            "═" * W,
            "  contradish · Inquiry Scaffold",
            sep,
            f"  Question: {self.question[:60]}{'...' if len(self.question) > 60 else ''}",
            "",
            "  LOAD-BEARING CLAIMS",
            "  (sorted by weight — most important first)",
        ]

        sorted_claims = sorted(
            self.load_bearing_claims,
            key=lambda c: c.get("weight", 0),
            reverse=True,
        )
        for c in sorted_claims:
            status = c.get("epistemic_status", "unknown")
            icon = self._STATUS_ICONS.get(status, "?")
            weight = c.get("weight", 0.0)
            claim = c.get("claim", "")
            bar_n = int(weight * 12)
            bar = "█" * bar_n + "░" * (12 - bar_n)
            lines.append(f"  {icon} [{bar}] {weight:.2f}  {claim}")
            change = c.get("what_would_change_it", "")
            if change:
                lines.append(f"         → changes if: {change}")

        if self.dependency_order:
            lines += ["", "  DEPENDENCY ORDER  (establish these in sequence)"]
            for i, dep in enumerate(self.dependency_order, 1):
                lines.append(f"    {i}. {dep}")

        if self.high_leverage_questions:
            lines += ["", "  HIGH-LEVERAGE QUESTIONS  (where to focus inquiry)"]
            for q in self.high_leverage_questions:
                lines.append(f"    → {q}")

        if self.epistemic_traps:
            lines += ["", "  EPISTEMIC TRAPS  (known distortions in this domain)"]
            for t in self.epistemic_traps:
                lines.append(f"    ⚠ {t}")

        if self.minimum_viable_path:
            lines += [
                "",
                "  MINIMUM VIABLE PATH",
                f"    {self.minimum_viable_path}",
            ]

        lines += ["", "═" * W, ""]
        return "\n".join(lines)


# ── Core classes ──────────────────────────────────────────────────────────────

class EpistemicAudit:
    """
    Analyzes the epistemic quality of AI responses.

    Does NOT evaluate whether a response is correct. Evaluates whether the
    response improves the human's ability to reason about the domain.

    Usage:
        audit = EpistemicAudit()

        # Audit a single response
        profile = audit.check(question="What causes inflation?", response="...")
        print(profile)

        # Map a disagreement
        dmap = audit.map_disagreement(
            question="Does X cause Y?",
            position_1="Yes, because...",
            position_2="No, because...",
        )
        print(dmap)

        # Generate an inquiry scaffold
        scaffold = audit.scaffold("What should I know about antibiotic resistance?")
        print(scaffold)
    """

    def __init__(
        self,
        api_key:   Optional[str] = None,
        provider:  Optional[str] = None,
    ):
        self._llm = LLMClient(api_key=api_key, provider=provider)

    def check(
        self,
        question:  str,
        response:  str,
    ) -> EpistemicProfile:
        """
        Audit the epistemic quality of a response to a question.

        Returns an EpistemicProfile with scores for Certainty Calibration,
        Dependency Legibility, Inquiry Advancement, and Disagreement Support.
        """
        t0 = time.time()
        prompt = _EPISTEMIC_AUDIT_PROMPT.format(
            question=question.strip(),
            response=response.strip(),
        )
        raw = self._llm.complete_json(prompt)
        elapsed = int((time.time() - t0) * 1000)

        if not isinstance(raw, dict):
            raw = {}

        c = float(raw.get("certainty_calibration",  0.0))
        d = float(raw.get("dependency_legibility",   0.0))
        i = float(raw.get("inquiry_advancement",     0.0))
        p = float(raw.get("disagreement_support",    0.0))
        eq = raw.get("epistemic_quality") or round((c + d + i + p) / 4, 4)

        return EpistemicProfile(
            question=question,
            response=response,
            certainty_calibration=round(c, 4),
            dependency_legibility=round(d, 4),
            inquiry_advancement=round(i, 4),
            disagreement_support=round(p, 4),
            epistemic_quality=round(float(eq), 4),
            hidden_assumptions=raw.get("hidden_assumptions", []),
            uncertain_claims=raw.get("uncertain_claims", []),
            inquiry_questions=raw.get("inquiry_questions", []),
            primary_finding=raw.get("primary_finding", ""),
            elapsed_ms=elapsed,
        )

    def map_disagreement(
        self,
        question:    str,
        position_1:  str,
        position_2:  str,
    ) -> DisagreementMap:
        """
        Map the structure of a disagreement between two positions.

        Does NOT decide who is right. Classifies the disagreement type and
        generates the minimum path to resolution.
        """
        t0 = time.time()
        prompt = _DISAGREEMENT_MAP_PROMPT.format(
            question=question.strip(),
            position_1=position_1.strip(),
            position_2=position_2.strip(),
        )
        raw = self._llm.complete_json(prompt)
        elapsed = int((time.time() - t0) * 1000)

        if not isinstance(raw, dict):
            raw = {}

        return DisagreementMap(
            question=question,
            position_1=position_1,
            position_2=position_2,
            load_bearing_point=raw.get("load_bearing_point", ""),
            primary_type=raw.get("primary_type", "E"),
            type_explanation=raw.get("type_explanation", ""),
            productive=bool(raw.get("productive", True)),
            can_both_be_right=bool(raw.get("can_both_be_right", False)),
            resolution_path=raw.get("resolution_path", ""),
            minimum_experiment=raw.get("minimum_experiment", ""),
            clarifying_definitions=raw.get("clarifying_definitions", []),
            surfaced_assumptions=raw.get("surfaced_assumptions", {}),
            scope_conditions=raw.get("scope_conditions", {}),
            next_step=raw.get("next_step", ""),
            elapsed_ms=elapsed,
        )

    def scaffold(self, question: str) -> InquiryScaffold:
        """
        Generate an inquiry scaffold for any question.

        Maps the dependency structure of the question — what must be established,
        in what order, with what degree of certainty. The Socratic method,
        operationalized: not "here is the answer" but "here is how to navigate
        the epistemic territory."
        """
        t0 = time.time()
        prompt = _INQUIRY_SCAFFOLD_PROMPT.format(question=question.strip())
        raw = self._llm.complete_json(prompt)
        elapsed = int((time.time() - t0) * 1000)

        if not isinstance(raw, dict):
            raw = {}

        return InquiryScaffold(
            question=question,
            load_bearing_claims=raw.get("load_bearing_claims", []),
            dependency_order=raw.get("dependency_order", []),
            high_leverage_questions=raw.get("high_leverage_questions", []),
            epistemic_traps=raw.get("epistemic_traps", []),
            minimum_viable_path=raw.get("minimum_viable_path", ""),
            elapsed_ms=elapsed,
        )

    def batch_audit(
        self,
        items: list[dict],   # [{question, response}]
    ) -> list[EpistemicProfile]:
        """
        Audit multiple question/response pairs.

        Args:
            items: list of {question, response} dicts

        Returns:
            List of EpistemicProfile in same order as input.
        """
        return [
            self.check(item["question"], item["response"])
            for item in items
        ]

    def compare(
        self,
        question:    str,
        responses:   dict[str, str],  # {model_name: response}
    ) -> dict:
        """
        Compare epistemic quality of multiple responses to the same question.

        Args:
            question:  the common question
            responses: {model_name: response_text}

        Returns:
            dict with profiles and ranking by epistemic quality
        """
        profiles = {
            name: self.check(question, response)
            for name, response in responses.items()
        }
        ranked = sorted(profiles.items(), key=lambda x: -x[1].epistemic_quality)
        return {
            "question": question,
            "profiles": {name: p for name, p in profiles.items()},
            "ranking":  [name for name, _ in ranked],
            "best":     ranked[0][0] if ranked else None,
            "scores":   {name: p.epistemic_quality for name, p in profiles.items()},
        }
