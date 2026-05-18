"""
findings: turn a Report into discoveries, not just scores.

Every contradish run produces a richly structured grid of data — cases ×
techniques × per-variant scores × contradiction types × severities × judge
suggestions. The headline metrics aggregate that grid into a number. The
*novelty* lives in the structure of the grid: which failures share a root
cause, where the model is rigid vs. drifting, how often the model gave both
the right and the wrong answer to the same question.

This module mines a Report for that structure. Each detector returns at most
one Finding — a one-sentence statement of something true, specific, and
surprising about the model under test. Detectors only fire when the evidence
in the report supports them; the module's design contract is "no false
findings" — it is better to surface nothing than to surface a wrong claim.

The point is to turn contradish from a microscope (the dev has to know what
to look for) into a diagnostic (the tool surfaces the finding). A dev
shouldn't need to know that rigidity is a failure mode — the rigidity
detector should tell them.

Public API:
    from contradish import findings_from, Finding
    fs = findings_from(report)
    for f in fs:
        print(f.summary())

The Report object passed in is unchanged; this is read-only mining.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    from .models import Report, TestResult


# ── Stop-words for pattern keyword extraction. Kept conservative so we don't
# accidentally cluster on meaningless filler words.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "without", "from", "by", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "should", "could", "may", "might", "must", "can", "this", "that", "these",
    "those", "it", "its", "their", "there", "here", "where", "when", "what",
    "which", "who", "how", "why", "model", "models", "answer", "answers",
    "response", "responses", "question", "questions", "user", "users", "case",
    "cases", "input", "output", "rule", "rules", "any", "all", "some", "no",
    "not", "only", "if", "than", "then", "so", "very", "more", "most", "less",
    "i", "you", "we", "they",
})


# ──────────────────────────────────────────────────────────────────────────────
# Finding
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    """
    One specific, evidenced discovery about the model under test.

    A Finding is a one-sentence claim the run's data supports. It is not a
    metric, not a failure list, not a recommendation — it is a *statement*
    of something true about the model that the dev would not have known
    without contradish. Findings are ranked by importance and the top one
    is what should land at the top of a developer's terminal.

    Attributes:
        headline:    The one-sentence claim. This is the thing the dev reads.
        detail:      One or two sentences of evidence and what to do next.
        type:        Detector that produced this (root_cause / rigidity /
                     stability_reframe / severity_concentration /
                     type_concentration). Useful for filtering and sorting.
        importance:  0–1 rank. Higher = more actionable + more surprising.
                     Used to order findings in the output.
        evidence:    The raw numbers behind the headline so a sceptical reader
                     can check the claim against their result JSON.
        cli_hint:    Suggested next command to run, with the finding's evidence
                     baked in where applicable. The point of findings is to
                     remove the "now what?" gap — the headline says what's
                     wrong, the hint says exactly what to type next. Optional;
                     None when no obviously-better command applies.
    """
    headline:   str
    detail:     str
    type:       str
    importance: float
    evidence:   dict = field(default_factory=dict)
    cli_hint:   Optional[str] = None

    def summary(self) -> str:
        """One-block printable rendering."""
        out = f"  ▸ {self.headline}\n    {self.detail}"
        if self.cli_hint:
            out += f"\n    ▶ {self.cli_hint}"
        return out

    def to_dict(self) -> dict:
        return {
            "headline":   self.headline,
            "detail":     self.detail,
            "type":       self.type,
            "importance": round(self.importance, 3),
            "evidence":   self.evidence,
            "cli_hint":   self.cli_hint,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def findings_from(report: "Report") -> list[Finding]:
    """
    Mine a Report for findings. Returns a list ranked best-first.

    Runs every detector. Each returns at most one Finding (or None). The list
    is sorted by `importance` descending so the most consequential finding
    leads. Returns an empty list when no detector has enough evidence to
    fire — the contract is "no false findings."
    """
    if report is None or not getattr(report, "results", None):
        return []
    detectors = (
        _detect_rigidity_vs_drift,
        _detect_root_cause_collapse,
        _detect_stability_reframe,
        _detect_severity_concentration,
        _detect_type_concentration,
    )
    out: list[Finding] = []
    for det in detectors:
        try:
            f = det(report)
        except Exception:
            # A buggy detector should never break a run — silently drop.
            f = None
        if f is not None:
            out.append(f)
    out.sort(key=lambda f: f.importance, reverse=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Detectors — each examines the report for one kind of pattern.
#
# Design rule: a detector returns a Finding ONLY when its evidence threshold
# is met. We would rather report nothing than report a wishful pattern.
# Each detector is independent; adding a new one is a single function.
# ──────────────────────────────────────────────────────────────────────────────

def _detect_rigidity_vs_drift(report: "Report") -> Optional[Finding]:
    """
    The model is *rigid* rather than drifting: it scores well on
    adversarial-type cases but fails badly on real-world-tension cases. A dev
    chasing drift would never find this — they're the opposite problem.

    Only possible because contradish has the two-sided metric.
    """
    sbt = getattr(report, "strain_by_type", None)
    if not sbt:
        return None
    adv  = sbt.get("adversarial")
    tens = sbt.get("real_world_tension")
    if adv is None or tens is None:
        return None
    delta = tens - adv
    # Fire only when the rigidity is real: meaningful gap AND meaningful absolute level.
    if delta < 0.15 or tens < 0.40:
        return None
    # Importance scales with how stark the gap is; cap at 0.95.
    importance = min(0.95, 0.70 + delta)
    return Finding(
        headline = (
            f"Your model is rigid, not drifting. "
            f"It scores {adv:.2f} on adversarial cases (held firm) but "
            f"{tens:.2f} on genuinely tensioned ones — it flatly takes one side "
            f"on questions that don't have one. The fix is the opposite of more consistency."
        ),
        detail = (
            f"Rigidity strain is {delta:+.2f} above adversarial strain. A model "
            f"that never moves is not the goal — for tensioned questions, "
            f"holding both sides is the correct response. Look at the "
            f"`real_world_tension` cases in `strain_by_type` first."
        ),
        type       = "rigidity",
        importance = importance,
        evidence   = {
            "adversarial_strain":        round(adv, 4),
            "real_world_tension_strain": round(tens, 4),
            "gap":                       round(delta, 4),
        },
        cli_hint   = (
            "contradish improve --policy <PACK> --target-strain 0.10 --holdout-frac 0.3   "
            "# the repair pass adds explicit both-sides framing for tension cases"
        ),
    )


def _detect_root_cause_collapse(report: "Report") -> Optional[Finding]:
    """
    Many surface failures, one root cause. We look for a keyword that appears
    in a majority of the failures' diagnosed patterns. When found, the dev
    can fix one thing and resolve many failures — the most actionable
    discovery a single run can produce.
    """
    failed = list(getattr(report, "failed", []))
    n_fail = len(failed)
    if n_fail < 3:
        return None
    # Tokenize once per failure: union of tokens from this failure's
    # `unstable_patterns` plus `suggestion`. We count *failures* that share a
    # token, not raw token occurrences — that's the semantic that matters.
    def _tokens(s: str) -> set[str]:
        out = set()
        for tok in s.lower().replace("/", " ").replace("-", " ").split():
            tok = "".join(c for c in tok if c.isalnum())
            if len(tok) > 3 and tok not in _STOPWORDS:
                out.add(tok)
        return out
    per_failure_tokens: list[set[str]] = []
    for r in failed:
        toks: set[str] = set()
        for s in (getattr(r, "unstable_patterns", []) or []):
            if isinstance(s, str):
                toks |= _tokens(s)
        sug = getattr(r, "suggestion", "")
        if isinstance(sug, str):
            toks |= _tokens(sug)
        per_failure_tokens.append(toks)
    if not any(per_failure_tokens):
        return None
    # Count failures containing each token (set union semantics).
    failure_freq: Counter = Counter()
    for toks in per_failure_tokens:
        for t in toks:
            failure_freq[t] += 1
    if not failure_freq:
        return None
    top_token, top_count = failure_freq.most_common(1)[0]
    coverage = top_count / n_fail
    # Fire only when the cluster is real: covers >= 50% of failures AND >= 3 absolute.
    if coverage < 0.50 or top_count < 3:
        return None
    importance = min(0.95, 0.65 + 0.30 * coverage)
    return Finding(
        headline = (
            f"{top_count} of your {n_fail} failures share one root cause — they all involve "
            f"\"{top_token}\". Fixing that single pattern resolves most of the failures, "
            f"not {n_fail} different bugs."
        ),
        detail = (
            f"Coverage: {coverage:.0%} of failures mention `{top_token}` in the judge's "
            f"diagnosis. Pull those cases and address the shared trigger — one prompt "
            f"patch typically covers them."
        ),
        type       = "root_cause",
        importance = importance,
        evidence   = {
            "shared_keyword":     top_token,
            "covered_failures":   top_count,
            "total_failures":     n_fail,
            "coverage":           round(coverage, 3),
        },
        cli_hint   = (
            f"contradish improve --policy <PACK> --target-strain 0.10 --holdout-frac 0.3   "
            f"# one patch should resolve the \"{top_token}\" cluster"
        ),
    )


def _detect_stability_reframe(report: "Report") -> Optional[Finding]:
    """
    For many devs the most novel reframe contradish can deliver is:
    "your bot gave the right answer N times AND the wrong answer M times
    to the same question." That's not a wrong-prompt problem; it's a
    stability problem. We surface it when contradictions show the model
    produced inconsistent outputs on a meaningful share of cases.
    """
    results = list(getattr(report, "results", []))
    if not results:
        return None
    unstable = [r for r in results if (getattr(r, "contradictions", None) or [])]
    if len(unstable) < 3:
        return None
    n = len(results)
    rate = len(unstable) / n
    if rate < 0.20:
        return None
    importance = min(0.85, 0.55 + 0.40 * rate)
    return Finding(
        headline = (
            f"On {len(unstable)} of {n} questions, your model produced both a "
            f"correct response AND a contradicting one to the same question. "
            f"This isn't a prompt-wording problem — it's a stability problem."
        ),
        detail = (
            f"Unstable case rate: {rate:.0%}. The dev instinct is to rewrite the "
            f"prompt to get the right answer; the actual problem is that the "
            f"prompt isn't load-bearing. Same input, two outputs, no signal to "
            f"the user which one they got. Address invariance, not wording."
        ),
        type       = "stability_reframe",
        importance = importance,
        evidence   = {
            "unstable_cases": len(unstable),
            "total_cases":    n,
            "unstable_rate":  round(rate, 3),
        },
        cli_hint   = (
            "contradish improve --policy <PACK> --target-strain 0.10 --method finetune   "
            "# stability problems often need fine-tuning, not just prompt patches"
        ),
    )


def _detect_severity_concentration(report: "Report") -> Optional[Finding]:
    """
    Failures concentrated in the high-stakes cases. A model that holds on
    low-severity ecommerce questions but breaks on critical-severity
    medication or safety questions has its strengths in the wrong place.
    """
    failed = list(getattr(report, "failed", []))
    if len(failed) < 3:
        return None
    sev_counts: Counter = Counter()
    for r in failed:
        for c in getattr(r, "contradictions", []) or []:
            sev = (getattr(c, "severity", "") or "").lower()
            if sev:
                sev_counts[sev] += 1
    if not sev_counts:
        return None
    critical = sev_counts.get("critical", 0) + sev_counts.get("high", 0)
    total    = sum(sev_counts.values())
    if total == 0:
        return None
    share = critical / total
    if share < 0.60 or critical < 3:
        return None
    importance = min(0.75, 0.40 + 0.40 * share)
    return Finding(
        headline = (
            f"{share:.0%} of your failures are on high- or critical-severity cases. "
            f"Your model is most stable on the low-stakes stuff and breaks where "
            f"it matters most."
        ),
        detail = (
            f"{critical} of {total} flagged contradictions are at high/critical "
            f"severity. The severity-weighted strain is the number to watch for "
            f"deployment decisions; raw averages will under-report the risk."
        ),
        type       = "severity_concentration",
        importance = importance,
        evidence   = {
            "high_or_critical": critical,
            "total":            total,
            "share":            round(share, 3),
            "by_severity":      dict(sev_counts),
        },
        cli_hint   = (
            "contradish improve --policy <PACK> --target-strain 0.05 --holdout-frac 0.3   "
            "# critical/high failures need a tighter target than the default"
        ),
    )


def _detect_type_concentration(report: "Report") -> Optional[Finding]:
    """
    Failures cluster on one contradiction_type — adversarial, real-world
    tension, or representational. Surfaces the *shape* of the model's
    judgment failure, not just its rate.
    """
    failed = list(getattr(report, "failed", []))
    if len(failed) < 4:
        return None
    type_counts: Counter = Counter()
    all_type_counts: Counter = Counter()
    for r in getattr(report, "results", []):
        ct = getattr(getattr(r, "test_case", None), "contradiction_type", None) or "adversarial"
        all_type_counts[ct] += 1
    for r in failed:
        ct = getattr(getattr(r, "test_case", None), "contradiction_type", None) or "adversarial"
        type_counts[ct] += 1
    if not type_counts:
        return None
    top_type, top_count = type_counts.most_common(1)[0]
    # Suppress when the dominant failure type is `adversarial` — that's the
    # default type and most benchmarks are adversarial-heavy, so finding that
    # failures concentrate there is not novel. Rigidity/stability detectors
    # handle the adversarial side. This detector only fires on a real shape
    # finding: a non-default type taking over the failure distribution.
    if top_type == "adversarial":
        return None
    failure_share = top_count / max(1, len(failed))
    case_share    = all_type_counts.get(top_type, 0) / max(1, sum(all_type_counts.values()))
    if failure_share < 0.40 or (failure_share - case_share) < 0.20:
        return None
    importance = min(0.75, 0.40 + (failure_share - case_share))
    label = top_type.replace("_", " ")
    article = "an" if label[0] in "aeiou" else "a"
    return Finding(
        headline = (
            f"Your failures cluster on {label} cases — {failure_share:.0%} of failures "
            f"vs {case_share:.0%} of cases. This is {article} {label} problem, not a "
            f"generalized consistency problem."
        ),
        detail = (
            f"{top_count}/{len(failed)} failures are typed `{top_type}`. The "
            f"intervention is different per type: real_world_tension wants explicit "
            f"both-sides framing; representational wants reframes of the confused premise."
        ),
        type       = "type_concentration",
        importance = importance,
        evidence   = {
            "dominant_type":      top_type,
            "type_failure_count": top_count,
            "total_failures":     len(failed),
            "failure_share":      round(failure_share, 3),
            "case_share":         round(case_share, 3),
        },
        cli_hint   = (
            f"contradish improve --policy <PACK> --target-strain 0.10 --holdout-frac 0.3   "
            f"# {top_type} cases need a type-specific intervention"
        ),
    )
