"""
contradish.reconcile: make the layers agree.

contradish renders several verdicts about the same model's consistency: the
prompt analyzer flags clauses that conflict, the benchmark scores Strain on
adversarial cases, and the Firewall / replay observe what actually broke in
production. Those verdicts are never reconciled, which is the contradiction
inside the tool: the thing that demands coherence across contexts is itself
incoherent across its own.

This module closes the most important gap: it grades the benchmark against
production reality. Given a benchmark Report and a ReplayReport (production
contradictions surfaced from real logs), it expresses both as *commitments*,
matches them, and classifies each commitment that broke in production:

  - validity gap : the benchmark covered this commitment and said it was fine,
                   but it broke in production. Your benchmark was too weak here.
  - confirmed    : the benchmark also flagged it. The bench predicted reality.
  - coverage gap : production broke on a commitment the benchmark never tested.

From those it reports two honest numbers about the benchmark itself:

  - coverage            : fraction of production breaks the benchmark had any
                          relevant test for.
  - predictive_validity : of the production breaks the benchmark covered, the
                          fraction it actually caught.

The reconciliation is pure: it matches already-extracted claims by relevance
and makes no API call, the same way `findings` mines an existing report for
free. Matching is lexical by default; pass an embedding relevance_fn for higher
recall on paraphrased commitments.

    from contradish import reconcile

    rec = reconcile(benchmark_report, replay_report)
    print(rec.summary())
    for m in rec.validity_gaps:
        print(m.prod_claim, "<- passed bench, broke in prod")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .memory import Commitment, topic_of, _overlap_score


# A relevance scorer over two commitments, in [0, 1]. Defaults to lexical
# overlap; an EmbeddingRelevance instance is a drop-in for semantic matching.
RelevanceFn = Callable[[Commitment, Commitment], float]


@dataclass
class CommitmentMatch:
    """One production-broken commitment, reconciled against the benchmark."""
    verdict:       str                      # validity_gap | confirmed | coverage_gap
    prod_claim:    str
    prod_session:  Optional[str] = None
    prod_turn:     Optional[int] = None
    bench_claim:   Optional[str] = None
    bench_name:    Optional[str] = None
    bench_passed:  Optional[bool] = None
    bench_strain:  Optional[float] = None
    score:         float = 0.0              # match strength to the bench commitment

    def to_dict(self) -> dict:
        return {
            "verdict":      self.verdict,
            "prod_claim":   self.prod_claim,
            "prod_session": self.prod_session,
            "prod_turn":    self.prod_turn,
            "bench_claim":  self.bench_claim,
            "bench_name":   self.bench_name,
            "bench_passed": self.bench_passed,
            "bench_strain": self.bench_strain,
            "score":        round(self.score, 4),
        }


@dataclass
class ReconciliationReport:
    """Benchmark-vs-production reconciliation over commitments."""
    matches:  list = field(default_factory=list)
    n_bench:  int = 0
    n_prod:   int = 0     # distinct commitments that broke in production

    # -- buckets --
    @property
    def validity_gaps(self) -> list:
        return [m for m in self.matches if m.verdict == "validity_gap"]

    @property
    def confirmed(self) -> list:
        return [m for m in self.matches if m.verdict == "confirmed"]

    @property
    def coverage_gaps(self) -> list:
        return [m for m in self.matches if m.verdict == "coverage_gap"]

    # -- metrics --
    @property
    def covered(self) -> int:
        return len(self.validity_gaps) + len(self.confirmed)

    @property
    def coverage(self) -> Optional[float]:
        """Fraction of production breaks the benchmark had a relevant test for."""
        return round(self.covered / self.n_prod, 4) if self.n_prod else None

    @property
    def predictive_validity(self) -> Optional[float]:
        """Of the covered production breaks, the fraction the benchmark caught."""
        return round(len(self.confirmed) / self.covered, 4) if self.covered else None

    def to_dict(self) -> dict:
        return {
            "n_bench":             self.n_bench,
            "n_prod":              self.n_prod,
            "n_validity_gaps":     len(self.validity_gaps),
            "n_confirmed":         len(self.confirmed),
            "n_coverage_gaps":     len(self.coverage_gaps),
            "coverage":            self.coverage,
            "predictive_validity": self.predictive_validity,
            "matches":             [m.to_dict() for m in self.matches],
        }

    def summary(self) -> str:
        lines = ["", "  contradish reconcile"]
        lines.append(f"  {self.n_bench} benchmark commitments · "
                     f"{self.n_prod} broke in production")
        cov = "n/a" if self.coverage is None else f"{self.coverage}"
        pv = "n/a" if self.predictive_validity is None else f"{self.predictive_validity}"
        lines.append(f"  coverage {cov} · predictive validity {pv}")
        lines.append("")

        vg = self.validity_gaps
        if vg:
            lines.append(f"  validity gaps ({len(vg)}): passed the bench, broke in production")
            for m in vg:
                where = f"session {m.prod_session}" if m.prod_session else "production"
                lines.append(f"    · {m.prod_claim}")
                lines.append(f"        bench case \"{m.bench_name or m.bench_claim}\" "
                             f"passed (strain {m.bench_strain}); broke in {where}")
            lines.append("")
        cg = self.coverage_gaps
        if cg:
            lines.append(f"  coverage gaps ({len(cg)}): broke in production, never tested")
            for m in cg:
                lines.append(f"    · {m.prod_claim}")
            lines.append("")
        if self.confirmed:
            lines.append(f"  confirmed ({len(self.confirmed)}): the benchmark predicted these.")
            lines.append("")
        if not self.matches:
            lines.append("  no production contradictions to reconcile.")
            lines.append("")
        return "\n".join(lines)


# ── Deriving commitments from each layer ────────────────────────────────────

def _bench_commitments(report) -> list:
    """Each TestResult -> (Commitment, passed, strain). The commitment under
    test is its canonical answer when present, else the case name/input."""
    out = []
    thresholds = getattr(report, "thresholds", None) or {}
    for r in getattr(report, "results", []) or []:
        tc = getattr(r, "test_case", None)
        if tc is None:
            continue
        claim = (getattr(tc, "canonical_answer", None)
                 or getattr(tc, "name", None)
                 or getattr(tc, "input", "") or "")
        claim = str(claim).strip()
        if not claim:
            continue
        c = Commitment(claim=claim, topic=topic_of(claim), origin="benchmark",
                       source_query=str(getattr(tc, "input", "") or ""))
        try:
            passed = bool(r.passed(thresholds))
        except Exception:
            passed = True
        strain = None
        for attr in ("judgment_strain", "cai_strain"):
            v = getattr(r, attr, None)
            if isinstance(v, (int, float)):
                strain = round(float(v), 4)
                break
        out.append((c, passed, strain, getattr(tc, "name", None) or claim[:48]))
    return out


def _prod_broken_commitments(replay_report) -> list:
    """Each production contradiction -> a distinct broken Commitment (deduped by
    claim), carrying session/turn provenance."""
    out = []
    seen = set()
    for c in getattr(replay_report, "contradictions", []) or []:
        claim = (getattr(c, "prior_claim", None)
                 or getattr(c, "new_claim", None)
                 or getattr(c, "response", "") or "")
        claim = str(claim).strip()
        if not claim:
            continue
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)
        commitment = Commitment(claim=claim, topic=topic_of(claim), origin="response",
                                session=str(getattr(c, "session", "") or "default"))
        out.append((commitment,
                    getattr(c, "session", None),
                    getattr(c, "prior_turn_index", None)))
    return out


def reconcile(
    report,
    replay_report,
    match_threshold: float = 0.3,
    relevance_fn:    Optional[RelevanceFn] = None,
) -> ReconciliationReport:
    """
    Reconcile a benchmark Report against a ReplayReport over commitments.

    Args:
        report:          a contradish Report (benchmark results).
        replay_report:   a ReplayReport (production contradictions from logs).
        match_threshold: minimum relevance to consider a production commitment
                         and a benchmark commitment "the same matter".
        relevance_fn:    scorer (prod, bench) -> [0,1]. Defaults to lexical
                         overlap; pass an EmbeddingRelevance for semantic
                         matching of paraphrased commitments.

    Returns:
        ReconciliationReport with validity gaps, coverage gaps, confirmations,
        and the benchmark's coverage / predictive-validity numbers.
    """
    score = relevance_fn or _overlap_score
    bench = _bench_commitments(report)
    prod = _prod_broken_commitments(replay_report)

    matches: list = []
    for prod_c, session, turn in prod:
        best = None
        best_s = 0.0
        for bench_c, passed, strain, name in bench:
            s = score(prod_c, bench_c)
            if s > best_s:
                best_s = s
                best = (bench_c, passed, strain, name)
        if best is not None and best_s >= match_threshold:
            bench_c, passed, strain, name = best
            verdict = "confirmed" if not passed else "validity_gap"
            matches.append(CommitmentMatch(
                verdict=verdict,
                prod_claim=prod_c.claim,
                prod_session=session,
                prod_turn=turn,
                bench_claim=bench_c.claim,
                bench_name=name,
                bench_passed=passed,
                bench_strain=strain,
                score=best_s,
            ))
        else:
            matches.append(CommitmentMatch(
                verdict="coverage_gap",
                prod_claim=prod_c.claim,
                prod_session=session,
                prod_turn=turn,
            ))

    return ReconciliationReport(matches=matches, n_bench=len(bench), n_prod=len(prod))


__all__ = [
    "reconcile",
    "ReconciliationReport",
    "CommitmentMatch",
    "RelevanceFn",
]
