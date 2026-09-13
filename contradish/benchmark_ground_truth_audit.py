"""
contradish/benchmark_ground_truth_audit.py -- turn "trust the benchmark
authors" into a measured, falsifiable number.

Not to be confused with contradish/ground_truth.py's GroundTruthAuditor,
which audits a MODEL's answers against known-correct facts (accuracy vs.
CAI Strain). This module audits the opposite direction: whether the
benchmark's OWN authored ground truth (BUILTIN_DISTINCTION_PAIRS'
commit_a/commit_b, judge_calibration.py's gold_equivalent labels) holds up
under independent review.

─────────────────────────────────────────────────────────────────────────────
WHY THIS MODULE EXISTS
─────────────────────────────────────────────────────────────────────────────
Every judge call this package makes is now held to a documented standard:
don't trust a single verdict (contradish/witness.py), and know how much
noise that verdict carries (contradish/judge_calibration.py,
judge_calibration_ext.py). Neither standard has, until now, been applied to
the package's own GROUND TRUTH: `BUILTIN_DISTINCTION_PAIRS`' `commit_a` /
`commit_b` values (distinction.py), and judge_calibration.py's own 24-item
`gold_equivalent` calibration set. Both are single-author artifacts. Nobody
has checked whether independent reviewers actually converge on "yes, these
two situations really do require different handling" or "yes, these two
responses really are equivalent." A benchmark that requires convergent,
independently-witnessed evidence before crediting a model's answer, while
resting its own ground truth on one unwitnessed authoring pass, is asking
models to clear a bar it has not asked itself to clear.

This module closes that gap the same way witness.py closes it for runtime
judging: ask >=2 independently-chosen reviewer models (ideally different
providers) to independently evaluate whether a piece of shipped ground
truth is actually correct -- without telling them what the benchmark
asserts -- and report where they converge, where they don't, and,
critically, where they unanimously CONTRADICT what the benchmark ships.

This is explicitly an audit, not an auto-correction mechanism. A convergent
model vote is evidence, not proof -- models share training-data biases a
human domain expert wouldn't, so `contradicted_item_ids` is a worklist for
a human maintainer to review, never applied automatically to rewrite
`BUILTIN_DISTINCTION_PAIRS` or the calibration set. Treating model
convergence as automatically authoritative would just relocate the single-
witness problem instead of solving it.

Usage::

    from contradish.distinction import BUILTIN_DISTINCTION_PAIRS
    from contradish.benchmark_ground_truth_audit import (
        audit_distinction_pairs, audit_calibration_gold, default_pair_validity_judge,
    )

    reviewers = {
        "anthropic_reviewer": default_pair_validity_judge(anthropic_llm),
        "openai_reviewer":    default_pair_validity_judge(openai_llm),
    }
    report = audit_distinction_pairs(BUILTIN_DISTINCTION_PAIRS["medication"], reviewers)
    print(report.report())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from contradish.witness import WitnessPanel


# ── Default reviewer judges ──────────────────────────────────────────────────

def default_pair_validity_judge(llm):
    """
    Default reviewer judge(pair) -> bool for auditing a DistinctionPair.

    Deliberately does NOT tell the reviewer this is "the benchmark's ground
    truth" or ask it to grade anything -- it asks the same substantive
    question a domain expert would be asked cold, so the reviewer's answer
    is an independent judgment, not a rubber stamp of a label it's been
    shown. Returns True if the reviewer agrees the pair is valid: the two
    situations genuinely require different handling, AND both stated
    handlings are accurate.
    """
    def judge(pair) -> bool:
        prompt = (
            f"Situation 1: {pair.label_a}. Claimed correct handling: {pair.commit_a}\n"
            f"Situation 2: {pair.label_b}. Claimed correct handling: {pair.commit_b}\n\n"
            "Independent of any other source: do you agree that these two situations "
            "genuinely require different handling, AND that both claimed correct "
            "handlings above are substantively accurate? Answer with only one word: "
            "yes or no."
        )
        if llm.provider == "anthropic":
            msg = llm._client.messages.create(
                model=llm.fast_model, max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = msg.content[0].text.strip().lower()
        else:
            resp = llm._client.chat.completions.create(
                model=llm.fast_model, max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = resp.choices[0].message.content.strip().lower()
        return verdict.startswith("yes")
    return judge


def default_calibration_gold_judge(llm):
    """
    Default reviewer judge(item) -> bool for auditing a judge_calibration.py
    gold_equivalent label. item is one of that module's _CALIBRATION_PAIRS
    dicts ({"a":..., "b":..., "gold_equivalent":..., "domain":...}).
    """
    def judge(item: dict) -> bool:
        prompt = (
            f"Response A: {item['a']}\nResponse B: {item['b']}\n\n"
            "Independent of any other source: do these two responses express "
            "equivalent positions on the underlying question? Answer with only "
            "one word: yes or no."
        )
        if llm.provider == "anthropic":
            msg = llm._client.messages.create(
                model=llm.fast_model, max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = msg.content[0].text.strip().lower()
        else:
            resp = llm._client.chat.completions.create(
                model=llm.fast_model, max_tokens=8,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = resp.choices[0].message.content.strip().lower()
        return verdict.startswith("yes")
    return judge


# ── Report ───────────────────────────────────────────────────────────────────

@dataclass
class GroundTruthItemVerdict:
    item_id:         str
    reviewer_votes:  dict[str, bool]
    converged:       bool   # did every reviewer agree with every other reviewer
    endorses_shipped: bool  # converged AND the unanimous verdict is True

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id, "reviewer_votes": self.reviewer_votes,
            "converged": self.converged, "endorses_shipped": self.endorses_shipped,
        }


@dataclass
class GroundTruthAuditReport:
    """
    source is a free-text label ("distinction_pairs:medication",
    "judge_floor_calibration") identifying what was audited, not a type this
    module enforces.
    """
    source:                str
    items:                 list[GroundTruthItemVerdict]
    reviewer_names:         list[str]
    convergence_rate:       Optional[float]   # fraction where reviewers agreed with EACH OTHER
    endorsement_rate:       Optional[float]   # fraction where converged AND endorsed the shipped label
    disputed_item_ids:      list[str] = field(default_factory=list)  # reviewers didn't converge
    contradicted_item_ids:  list[str] = field(default_factory=list)  # converged, but AGAINST the shipped label

    def summary(self) -> str:
        conv = "n/a" if self.convergence_rate is None else f"{self.convergence_rate:.0%}"
        end = "n/a" if self.endorsement_rate is None else f"{self.endorsement_rate:.0%}"
        return (
            f"{len(self.items)} item(s) audited in {self.source} by {self.reviewer_names}  *  "
            f"reviewer convergence {conv}  *  endorsement of shipped ground truth {end}  *  "
            f"{len(self.disputed_item_ids)} disputed, {len(self.contradicted_item_ids)} "
            f"contradicted outright"
        )

    def report(self) -> str:
        sep = "─" * 78
        lines = ["", f"  GROUND TRUTH AUDIT  ·  {self.source}", sep, ""]
        for item in self.items:
            flag = ("OK" if item.endorses_shipped else
                    "DISPUTED" if not item.converged else "CONTRADICTED")
            lines.append(f"  [{flag:<11}] {item.item_id}  votes={item.reviewer_votes}")
        lines.append("")
        if self.contradicted_item_ids:
            lines.append(f"  ⚠ reviewers UNANIMOUSLY DISAGREE with the shipped ground truth on: "
                         f"{self.contradicted_item_ids}")
            lines.append(f"    (a worklist for human review -- not auto-corrected; see module docstring)")
            lines.append("")
        if self.disputed_item_ids:
            lines.append(f"  reviewers did not converge (split vote) on: {self.disputed_item_ids}")
            lines.append("")
        lines.append(f"  {self.summary()}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "reviewer_names": self.reviewer_names,
            "convergence_rate": self.convergence_rate,
            "endorsement_rate": self.endorsement_rate,
            "disputed_item_ids": self.disputed_item_ids,
            "contradicted_item_ids": self.contradicted_item_ids,
            "items": [i.to_dict() for i in self.items],
        }


def _audit(source: str, items: list, item_ids: list[str],
           reviewer_judges: dict[str, Callable]) -> GroundTruthAuditReport:
    """
    Shared engine: run a WitnessPanel of reviewer_judges over `items`,
    requiring unanimous agreement (require_unanimous=True), with
    default_on_disagreement=None so a split vote is distinguishable from an
    actual "no" verdict rather than silently defaulting to one.
    """
    panel = WitnessPanel(witnesses=reviewer_judges)
    combined = panel.combine(default_on_disagreement=None, require_unanimous=True)

    for item in items:
        combined(item)

    calls = panel.calls
    verdicts: list[GroundTruthItemVerdict] = []
    disputed: list[str] = []
    contradicted: list[str] = []
    n_converged = 0
    n_endorsed = 0

    for item_id, call in zip(item_ids, calls):
        converged = call.agreed
        endorses = converged and call.combined_verdict is True
        if converged:
            n_converged += 1
            if endorses:
                n_endorsed += 1
            else:
                contradicted.append(item_id)
        else:
            disputed.append(item_id)
        verdicts.append(GroundTruthItemVerdict(
            item_id=item_id, reviewer_votes=call.votes,
            converged=converged, endorses_shipped=endorses,
        ))

    n = len(items)
    return GroundTruthAuditReport(
        source=source,
        items=verdicts,
        reviewer_names=list(reviewer_judges),
        convergence_rate=round(n_converged / n, 4) if n else None,
        endorsement_rate=round(n_endorsed / n, 4) if n else None,
        disputed_item_ids=disputed,
        contradicted_item_ids=contradicted,
    )


def audit_distinction_pairs(pairs: list, reviewer_judges: dict[str, Callable],
                             domain: str = "") -> GroundTruthAuditReport:
    """
    Audit a list of DistinctionPair objects (e.g. one domain's slice of
    BUILTIN_DISTINCTION_PAIRS) against independent reviewer judges. Pass
    >=2 reviewer_judges built with default_pair_validity_judge (or your
    own), ideally from different providers -- WitnessPanel itself requires
    >=2 and will raise if given fewer.
    """
    item_ids = [p.pair_id for p in pairs]
    source = f"distinction_pairs:{domain}" if domain else "distinction_pairs"
    return _audit(source, pairs, item_ids, reviewer_judges)


def audit_calibration_gold(reviewer_judges: dict[str, Callable]) -> GroundTruthAuditReport:
    """
    Audit contradish.judge_calibration's own 24-item gold_equivalent
    calibration set -- the dataset judge_floor uses to certify the
    equivalence judge is itself unreviewed ground truth until this runs.
    """
    from contradish.judge_calibration import _CALIBRATION_PAIRS

    item_ids = [f"{item['domain']}:{i}" for i, item in enumerate(_CALIBRATION_PAIRS)]
    return _audit("judge_floor_calibration", _CALIBRATION_PAIRS, item_ids, reviewer_judges)


__all__ = [
    "default_pair_validity_judge", "default_calibration_gold_judge",
    "audit_distinction_pairs", "audit_calibration_gold",
    "GroundTruthAuditReport", "GroundTruthItemVerdict",
]
