"""
contradish/format_fidelity.py -- output-format collapse under paraphrase: does
a model's OUTPUT FORM (not its content) hold steady when the same formatting
instruction is reworded?

─────────────────────────────────────────────────────────────────────────────
WHY THIS MODULE EXISTS
─────────────────────────────────────────────────────────────────────────────
Every prior construct in this package -- CAI Strain, distinction loss, KBV,
distinction sacrifice, provenance collapse, evaluation-awareness -- measures
whether the model's CONTENT (a commitment, a claim, a hedge) survives some
kind of pressure. None of them ask whether the model's output FORM survives
paraphrase of the same instruction. That's a distinct failure mode, studied
under paraphrase-induced output-mode collapse / semantic-consistency-score
research (e.g. ParaConsist-style work measuring format stability under
semantically-equivalent rewordings): a model told "respond with a single
JSON object" and, later, "please answer using just one JSON object" may
silently switch between JSON, a markdown table, and free prose depending on
which exact words were used -- even though the two instructions mean the
same thing and the model's underlying knowledge hasn't changed at all. A
tool that only checks whether the CONTENT stayed consistent would call that
a pass; a downstream caller expecting parseable JSON would call it a
failure. This module closes that gap: same instruction, reworded four ways,
does the model's chosen output FORM stay put?

Architecture mirrors eval_awareness.py: this wraps a `probe_fn(instruction)
-> str` the caller supplies (the model's raw text output for one phrasing
of a formatting instruction), plus a `classify_fn(output) -> str` that maps
raw output to a discrete format label ("json", "bulleted_list", "prose",
"single_word", "table", ...). This module supplies one deterministic,
API-free default classifier (`default_format_classifier`) that is
intentionally coarse -- good enough to catch a JSON-vs-prose-vs-list
collapse, not a claim to detect every possible format. Callers with a
narrower or stricter notion of "format" should pass their own classify_fn.

    from contradish.format_fidelity import measure_format_fidelity, default_format_classifier

    def probe(instruction):
        return my_model(system_prompt=instruction, user_prompt=question)

    instance = measure_format_fidelity(
        probe, classify_fn=default_format_classifier,
        instruction_id="json_summary", domain="support_tickets",
        paraphrases=[
            "Respond with a single JSON object containing 'summary' and 'priority'.",
            "Please answer using just one JSON object with the keys 'summary' and 'priority'.",
            "Your entire reply should be one JSON object -- fields: summary, priority.",
            "Format your answer as JSON only: {summary, priority}.",
        ],
    )
    print(instance.consistency_rate, instance.collapsed)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Default format classifier ─────────────────────────────────────────────────
#
# Deliberately coarse and deterministic (no model call): good enough to catch
# the failure mode this module targets -- gross format switching, not fine
# stylistic drift. Order matters (first match wins); callers wanting a finer
# taxonomy should pass their own classify_fn to measure_format_fidelity.

def default_format_classifier(output: str) -> str:
    """
    Classify raw model output into a coarse structural bucket:
    'empty', 'json', 'table', 'bulleted_list', 'numbered_list',
    'single_word', or 'prose' (the fallback).
    """
    text = (output or "").strip()
    if not text:
        return "empty"

    stripped = text
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        return "json"

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2 and sum(1 for ln in lines if "|" in ln) >= len(lines) - 1:
        return "table"

    bullet_re = re.compile(r"^\s*[-*•]\s+")
    if lines and sum(1 for ln in lines if bullet_re.match(ln)) >= max(1, len(lines) - 1):
        return "bulleted_list"

    numbered_re = re.compile(r"^\s*\d+[.)]\s+")
    if lines and sum(1 for ln in lines if numbered_re.match(ln)) >= max(1, len(lines) - 1):
        return "numbered_list"

    if len(text.split()) == 1:
        return "single_word"

    return "prose"


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FormatFidelityInstance:
    """
    One instruction's format stability across its paraphrases.

    per_paraphrase_format maps each paraphrase (or a caller-supplied label
    for it) to the format label classify_fn assigned its output.
    consistency_rate is the fraction of paraphrases whose format matches the
    modal (most common) format; collapsed is True when that rate falls
    below the threshold measure_format_fidelity was called with.
    """
    instruction_id:          str
    domain:                  str
    per_paraphrase_format:   dict[str, str] = field(default_factory=dict)
    modal_format:            str = ""
    consistency_rate:        float = 0.0
    collapsed:                bool = False

    def to_dict(self) -> dict:
        return {
            "instruction_id": self.instruction_id, "domain": self.domain,
            "per_paraphrase_format": self.per_paraphrase_format,
            "modal_format": self.modal_format,
            "consistency_rate": self.consistency_rate,
            "collapsed": self.collapsed,
        }


@dataclass
class FormatFidelityReport:
    instances:            list[FormatFidelityInstance]
    mean_consistency_rate: Optional[float]
    collapse_rate:         Optional[float]   # fraction of instances that collapsed
    most_unstable:         str

    def summary(self) -> str:
        mcr = "n/a" if self.mean_consistency_rate is None else f"{self.mean_consistency_rate:.4f}"
        cr = "n/a" if self.collapse_rate is None else f"{self.collapse_rate:.4f}"
        return (
            f"{len(self.instances)} instruction(s)  *  mean consistency {mcr}  *  "
            f"collapse_rate {cr}  *  most unstable: {self.most_unstable or 'n/a'}"
        )

    def report(self) -> str:
        sep = "─" * 78
        lines = ["", "  FORMAT FIDELITY UNDER PARAPHRASE", sep, ""]
        for inst in sorted(self.instances, key=lambda i: i.consistency_rate):
            flag = "COLLAPSED" if inst.collapsed else "stable"
            lines.append(f"  {inst.instruction_id}  ({inst.domain})  [{flag}]")
            lines.append(f"    modal_format={inst.modal_format}  "
                         f"consistency_rate={inst.consistency_rate:.4f}")
            lines.append(f"    per-paraphrase: {inst.per_paraphrase_format}")
            lines.append("")
        lines.append(f"  {self.summary()}")
        lines.append("")
        lines.append("  This checks output FORM, not content -- a low-collapse instance can")
        lines.append("  still be wrong, and a stable-format instance can still be sacrificed")
        lines.append("  or hedged per sacrifice.py/provenance.py. Orthogonal, not a replacement.")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "mean_consistency_rate": self.mean_consistency_rate,
            "collapse_rate": self.collapse_rate,
            "most_unstable": self.most_unstable,
            "instances": [i.to_dict() for i in self.instances],
        }


# ── Measurement ──────────────────────────────────────────────────────────────

def measure_format_fidelity(
    probe_fn: Callable[[str], str],
    paraphrases: list[str],
    classify_fn: Callable[[str], str] = default_format_classifier,
    instruction_id: str = "instruction",
    domain: str = "",
    threshold: float = 0.75,
    labels: Optional[list[str]] = None,
) -> FormatFidelityInstance:
    """
    Run probe_fn once per paraphrase, classify each output's format with
    classify_fn, and report how consistently the modal format holds.

    probe_fn(instruction: str) -> str
        Returns the model's raw text output for that phrasing of the
        (semantically fixed) formatting instruction.
    threshold
        consistency_rate below this marks the instance collapsed. Default
        0.75 means: format must hold for at least 3 of 4 paraphrases (or
        the equivalent fraction) to not be flagged.
    """
    if len(paraphrases) < 2:
        raise ValueError("need at least two paraphrases to measure format stability")

    keys = labels if labels is not None else paraphrases
    if len(keys) != len(paraphrases):
        raise ValueError("labels, if given, must be the same length as paraphrases")

    per_format: dict[str, str] = {}
    for key, phrasing in zip(keys, paraphrases):
        output = probe_fn(phrasing)
        per_format[key] = classify_fn(output)

    counts = Counter(per_format.values())
    modal_format, modal_count = counts.most_common(1)[0]
    consistency_rate = round(modal_count / len(per_format), 4)

    return FormatFidelityInstance(
        instruction_id=instruction_id, domain=domain,
        per_paraphrase_format=per_format,
        modal_format=modal_format,
        consistency_rate=consistency_rate,
        collapsed=consistency_rate < threshold,
    )


def measure_format_fidelity_batch(
    instructions: dict[str, list[str]],
    classify_fn: Callable[[str], str],
    probe_fn: Callable[[str, str], str],
    domain: str = "",
    threshold: float = 0.75,
) -> FormatFidelityReport:
    """
    Run measure_format_fidelity() over several named instructions at once.

    probe_fn(instruction_id: str, phrasing: str) -> str
        Unlike measure_format_fidelity's probe_fn, this one also receives
        the instruction_id, so a single probe_fn can dispatch across
        multiple distinct instructions/questions in one batch call.
    instructions
        Maps instruction_id -> list of paraphrases of that instruction.
    """
    instances = []
    for iid, paraphrases in instructions.items():
        instances.append(measure_format_fidelity(
            probe_fn=lambda phrasing, _iid=iid: probe_fn(_iid, phrasing),
            paraphrases=paraphrases, classify_fn=classify_fn,
            instruction_id=iid, domain=domain, threshold=threshold,
        ))

    n = len(instances)
    mean_consistency = round(sum(i.consistency_rate for i in instances) / n, 4) if n else None
    collapse_rate = round(sum(1 for i in instances if i.collapsed) / n, 4) if n else None
    most_unstable = min(instances, key=lambda i: i.consistency_rate).instruction_id if instances else ""

    return FormatFidelityReport(
        instances=instances, mean_consistency_rate=mean_consistency,
        collapse_rate=collapse_rate, most_unstable=most_unstable,
    )


__all__ = [
    "default_format_classifier", "FormatFidelityInstance", "FormatFidelityReport",
    "measure_format_fidelity", "measure_format_fidelity_batch",
]
