"""
observatory.py — Constraint Observatory

A reasoning system is a finite process navigating constraints.

  contradiction       = incompatible constraints
  repair              = changing the system to satisfy more constraints
  convergence         = satisfying an increasingly stable set of constraints
  load-bearing        = a constraint that induces many others

The observatory studies every intelligent system by the constraints it
satisfies, violates, and discovers.

This reframes every prior component:

  NodeProbe           → Constraint
  ε_c (sensitivity)   → constraint violation rate
  ε_r (reality)       → distance from the constraint's satisfaction region
  Φ*                  → the constraint set satisfied at the fixed point
  topology distance   → difference in constraint profiles
  superspreader       → load-bearing constraint with highest violation rate
  systemic artifact   → a constraint violated by all systems in the same direction
  irreducible         → a constraint whose satisfaction is underdetermined
                        by the current training distributions

The observatory's three functions:

  1. Profile    — characterize each system by its constraint satisfaction
  2. Catalog    — accumulate constraint knowledge across systems and time
  3. Frontier   — identify where new constraints are being discovered

The frontier is the most important function. It is not a list of failures.
It is the boundary where knowledge is being made.

Usage:
    PYTHONPATH=. python examples/observatory_demo.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from .phi_star import FRAMING_PREFIXES, jaccard_similarity
from .oracle import NodeProbe
from .active_oracle import GroundTruthSignal
from .structural_eval import StructuralEvaluator


# ─── Constraint ───────────────────────────────────────────────────────────────

@dataclass
class Constraint:
    """A constraint in a domain's reasoning structure.

    A constraint is a commitment a reasoning system must make to operate
    reliably in a domain. Constraints are partially ordered by dependency:
    some constraints induce others, and violating a load-bearing constraint
    propagates violations downstream.

    load_weight     λ ∈ [0,1]: how many other constraints this induces.
                    High load_weight = load-bearing constraint.
                    Repair here reduces the most downstream violations.

    dependencies    Constraint IDs that are induced by this one.
                    If this constraint is violated, those are at risk.

    discoverable    False if the constraint is genuinely underdetermined —
                    no finite system can satisfy it from the current
                    training distribution regardless of repair.

    ground_truth    The correct commitment if known (ε_r anchor).
                    None if the constraint's satisfaction is context-dependent
                    or not yet established by convergent inquiry.
    """
    constraint_id: str
    domain: str
    description: str
    question: str                           # the probe question
    load_weight: float = 0.5               # λ
    dependencies: list[str] = field(default_factory=list)
    discoverable: bool = True
    ground_truth: Optional[str] = None     # known correct commitment, or None

    def to_probe(self) -> NodeProbe:
        return NodeProbe(
            node_id=self.constraint_id,
            question=self.question,
            description=self.description,
            lambda_weight=self.load_weight,
        )


# ─── Constraint status ────────────────────────────────────────────────────────

@dataclass
class ConstraintStatus:
    """One system's relationship to one constraint."""
    constraint_id: str
    # How the system responds
    violation_rate: float                  # ε_c: fraction of framing pairs that disagree
    commitment: Optional[str]             # canonical answer under neutral framing
    drift_conditions: list[str]           # framings that produce state-change
    # Classification
    oracle_status: str                    # resolved | artifact | systemic_artifact |
                                          # irreducible | unverified | confirmed
    corroborated: Optional[bool]          # does ε_r confirm the commitment?

    @property
    def satisfied(self) -> bool:
        """The system satisfies this constraint: stable and corroborated."""
        return self.oracle_status in ("resolved", "confirmed")

    @property
    def violated(self) -> bool:
        """The system violates this constraint: inconsistent or wrong.

        Oracle classification takes priority over raw ε_c.
        A resolved/confirmed constraint is satisfied even if ε_c is high —
        fragility is tracked separately but doesn't override the oracle's
        corroborated verdict.
        """
        if self.satisfied:
            return False
        return (
            self.oracle_status in ("artifact", "systemic_artifact")
            or self.violation_rate >= 0.30
        )

    @property
    def undetermined(self) -> bool:
        return self.oracle_status in ("irreducible", "unverified")


# ─── Constraint profile ───────────────────────────────────────────────────────

@dataclass
class ConstraintProfile:
    """A complete characterization of one system's constraint satisfaction.

    The constraint profile IS the system's reasoning structure,
    as observable through meaning-preserving perturbation.

    What it measures that benchmarks cannot:
      - Not just which questions the system gets right
      - But which constraints it reliably satisfies, and under what conditions
      - Which constraints it violates consistently (artifacts)
      - Which constraints are beyond its current reach (undetermined)
      - How load-weighted satisfaction compares across systems and versions
    """
    model_label: str
    domain: str
    timestamp: str
    constraint_statuses: dict[str, ConstraintStatus]   # constraint_id → status
    constraint_catalog: dict[str, Constraint]          # constraint_id → constraint

    @property
    def satisfied(self) -> list[ConstraintStatus]:
        return [s for s in self.constraint_statuses.values() if s.satisfied]

    @property
    def violated(self) -> list[ConstraintStatus]:
        return [s for s in self.constraint_statuses.values() if s.violated]

    @property
    def undetermined(self) -> list[ConstraintStatus]:
        return [s for s in self.constraint_statuses.values() if s.undetermined]

    @property
    def satisfaction_rate(self) -> float:
        """Fraction of constraints satisfied."""
        n = len(self.constraint_statuses)
        if n == 0:
            return 0.0
        return len(self.satisfied) / n

    @property
    def load_weighted_satisfaction(self) -> float:
        """Satisfaction rate weighted by constraint load (λ).

        High load_weight constraints matter more: violating them propagates
        to many dependent constraints. This is a more honest measure of
        structural reliability than uniform satisfaction rate.
        """
        total_load = sum(
            self.constraint_catalog[cid].load_weight
            for cid in self.constraint_statuses
            if cid in self.constraint_catalog
        )
        if total_load < 1e-9:
            return self.satisfaction_rate

        satisfied_load = sum(
            self.constraint_catalog[cid].load_weight
            for cid, s in self.constraint_statuses.items()
            if s.satisfied and cid in self.constraint_catalog
        )
        return satisfied_load / total_load

    @property
    def most_load_bearing_violation(self) -> Optional[ConstraintStatus]:
        """The violated constraint with the highest load weight.
        Repairing this reduces the most downstream violations."""
        violated = [
            (self.constraint_catalog[s.constraint_id].load_weight, s)
            for s in self.violated
            if s.constraint_id in self.constraint_catalog
        ]
        if not violated:
            return None
        return max(violated, key=lambda x: x[0])[1]

    def report(self) -> str:
        w = 64
        mlbv = self.most_load_bearing_violation
        lines = [
            "─" * w,
            f"Constraint Profile  ·  {self.model_label}  ·  {self.domain}",
            f"Timestamp: {self.timestamp}",
            "─" * w,
            "",
            f"Satisfaction rate:           {self.satisfaction_rate:.2%}",
            f"Load-weighted satisfaction:  {self.load_weighted_satisfaction:.2%}",
            f"Satisfied:    {len(self.satisfied)}",
            f"Violated:     {len(self.violated)}",
            f"Undetermined: {len(self.undetermined)}",
        ]
        if mlbv and mlbv.constraint_id in self.constraint_catalog:
            c = self.constraint_catalog[mlbv.constraint_id]
            lines.append(
                f"\nHighest-impact repair target:"
            )
            lines.append(
                f"  {mlbv.constraint_id}  λ={c.load_weight:.2f}  "
                f"violation_rate={mlbv.violation_rate:.3f}"
            )
            lines.append(
                f"  Repairing this constraint satisfies "
                f"{len(c.dependencies)} dependent constraint"
                f"{'s' if len(c.dependencies) != 1 else ''}."
            )

        lines.append("\nConstraints:")
        for cid, status in sorted(
            self.constraint_statuses.items(),
            key=lambda x: (
                -self.constraint_catalog.get(x[0],
                    Constraint("", "", "", "")).load_weight
            )
        ):
            c = self.constraint_catalog.get(cid)
            lw = f"λ={c.load_weight:.2f}" if c else ""
            symbol = "✓" if status.satisfied else ("✗" if status.violated else "∅")
            color_tag = {"✓": "satisfied", "✗": "violated", "∅": "undetermined"}[symbol]
            lines.append(
                f"  {symbol} {cid:<26} {lw:<8} "
                f"ε_c={status.violation_rate:.3f}  [{status.oracle_status}]"
            )
        lines.append("─" * w)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_label": self.model_label,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "satisfaction_rate": self.satisfaction_rate,
            "load_weighted_satisfaction": self.load_weighted_satisfaction,
            "constraints": {
                cid: {
                    "violation_rate": s.violation_rate,
                    "commitment": s.commitment,
                    "drift_conditions": s.drift_conditions,
                    "oracle_status": s.oracle_status,
                    "corroborated": s.corroborated,
                    "satisfied": s.satisfied,
                    "violated": s.violated,
                }
                for cid, s in self.constraint_statuses.items()
            },
        }


# ─── Constraint delta ─────────────────────────────────────────────────────────

@dataclass
class ConstraintDelta:
    """Structural change between two constraint profiles of the same system."""
    label_a: str
    label_b: str
    satisfaction_rate_a: float
    satisfaction_rate_b: float
    load_weighted_a: float
    load_weighted_b: float
    newly_satisfied: list[str]     # constraints newly satisfied in b
    newly_violated: list[str]      # constraints newly violated in b
    newly_undetermined: list[str]  # constraints newly undetermined in b

    @property
    def load_weighted_change(self) -> float:
        return self.load_weighted_b - self.load_weighted_a

    @property
    def direction(self) -> str:
        if self.load_weighted_change > 0.05:
            return "improved"
        elif self.load_weighted_change < -0.05:
            return "regressed"
        return "unchanged"

    def report(self) -> str:
        d = "↑ improved" if self.direction == "improved" else (
            "↓ regressed" if self.direction == "regressed" else "→ unchanged")
        lines = [
            f"Constraint Delta: {self.label_a} → {self.label_b}",
            f"  Load-weighted satisfaction: "
            f"{self.load_weighted_a:.2%} → {self.load_weighted_b:.2%}  {d}",
        ]
        if self.newly_satisfied:
            lines.append(f"  Newly satisfied (+{len(self.newly_satisfied)}): "
                         f"{', '.join(self.newly_satisfied)}")
        if self.newly_violated:
            lines.append(f"  Newly violated  (-{len(self.newly_violated)}): "
                         f"{', '.join(self.newly_violated)}")
        if self.newly_undetermined:
            lines.append(f"  Now undetermined: {', '.join(self.newly_undetermined)}")
        return "\n".join(lines)


# ─── Constraint observatory ───────────────────────────────────────────────────

class ConstraintObservatory:
    """A place where every intelligent system is studied by the constraints
    it satisfies, violates, and discovers.

    The observatory has three functions:

      Profile    Register and retrieve constraint profiles for any system.
                 Every profile adds to the cumulative record.

      Catalog    Maintain the domain constraint catalog — the growing map
                 of which constraints exist, their load weights, and their
                 dependency structure. The catalog is the public good:
                 accumulated knowledge of a domain's constraint structure.

      Frontier   Identify where new constraints are being discovered.
                 The frontier consists of constraints that are satisfied
                 by some systems, violated by others, with no consensus.
                 These are not failures. They are where knowledge is being made.

    The astronomical analogy:
      The telescope (evaluation tooling) is proprietary.
      The star catalog (domain constraint structure) is published.
      The observations (per-model profiles) are confidential.
      The frontier (open problems) is the scientific contribution.
    """

    def __init__(self, path: Optional[str] = None):
        # domain → {constraint_id → Constraint}
        self._catalogs: dict[str, dict[str, Constraint]] = {}
        # domain → {model_label → list[ConstraintProfile]}  (list = version history)
        self._profiles: dict[str, dict[str, list[ConstraintProfile]]] = {}
        self._path = path
        if path:
            try:
                self._load(path)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

    # ── Catalog ────────────────────────────────────────────────────────────────

    def register_constraints(self, constraints: list[Constraint]) -> None:
        """Add constraints to the domain catalog."""
        for c in constraints:
            if c.domain not in self._catalogs:
                self._catalogs[c.domain] = {}
            self._catalogs[c.domain][c.constraint_id] = c

    def get_catalog(self, domain: str) -> dict[str, Constraint]:
        return self._catalogs.get(domain, {})

    def domains(self) -> list[str]:
        return list(self._catalogs.keys())

    # ── Profile ────────────────────────────────────────────────────────────────

    def register_profile(self, profile: ConstraintProfile) -> None:
        """Add a constraint profile to the observatory record."""
        domain = profile.domain
        if domain not in self._profiles:
            self._profiles[domain] = {}
        if profile.model_label not in self._profiles[domain]:
            self._profiles[domain][profile.model_label] = []
        self._profiles[domain][profile.model_label].append(profile)
        if self._path:
            self._save(self._path)

    def get_profile(
        self, model_label: str, domain: str, version: int = -1
    ) -> Optional[ConstraintProfile]:
        """Get a model's constraint profile (most recent by default)."""
        versions = self._profiles.get(domain, {}).get(model_label, [])
        if not versions:
            return None
        return versions[version]

    def get_history(self, model_label: str, domain: str) -> list[ConstraintProfile]:
        return self._profiles.get(domain, {}).get(model_label, [])

    def all_models(self, domain: str) -> list[str]:
        return list(self._profiles.get(domain, {}).keys())

    # ── Observatory queries ────────────────────────────────────────────────────

    def satisfied_by_all(self, domain: str) -> list[Constraint]:
        """Constraints satisfied by every registered model in this domain.
        These are the most firmly established constraints in the domain.
        The best available approximation of reality's constraint structure.
        """
        catalog = self._catalogs.get(domain, {})
        models = self.all_models(domain)
        if not models or not catalog:
            return []

        result = []
        for cid, constraint in catalog.items():
            if all(
                self._latest_status(model, domain, cid) is not None
                and self._latest_status(model, domain, cid).satisfied
                for model in models
            ):
                result.append(constraint)
        return sorted(result, key=lambda c: -c.load_weight)

    def violated_by_all(self, domain: str) -> list[Constraint]:
        """Constraints violated by every registered model in this domain.
        These are the systemic constraints — shared structural failures.
        Invisible to cross-model comparison. Only ε_r reveals them.
        """
        catalog = self._catalogs.get(domain, {})
        models = self.all_models(domain)
        if not models or not catalog:
            return []

        result = []
        for cid, constraint in catalog.items():
            if all(
                self._latest_status(model, domain, cid) is not None
                and self._latest_status(model, domain, cid).violated
                for model in models
            ):
                result.append(constraint)
        return sorted(result, key=lambda c: -c.load_weight)

    def frontier(self, domain: str) -> list[Constraint]:
        """Constraints satisfied by some models, violated by others.

        The frontier is where knowledge is being made.

        A constraint on the frontier is not merely contested — it is
        actively in the process of being discovered. Some finite systems
        have found a way to satisfy it; others have not. The ones that
        have satisfied it have discovered something. The ones that haven't
        have not yet. The frontier marks the current edge of discovery.

        Monitoring the frontier over time shows progress:
        as constraints move from frontier to satisfied-by-all,
        a domain's knowledge becomes more settled.
        """
        catalog = self._catalogs.get(domain, {})
        models = self.all_models(domain)
        if len(models) < 2 or not catalog:
            return []

        result = []
        for cid, constraint in catalog.items():
            statuses = [
                self._latest_status(model, domain, cid)
                for model in models
            ]
            statuses = [s for s in statuses if s is not None]
            if not statuses:
                continue
            n_satisfied = sum(1 for s in statuses if s.satisfied)
            n_violated = sum(1 for s in statuses if s.violated)
            # Frontier: some satisfied, some violated
            if n_satisfied > 0 and n_violated > 0:
                result.append(constraint)

        return sorted(result, key=lambda c: -c.load_weight)

    def who_discovered(self, constraint_id: str, domain: str) -> list[str]:
        """Which models have discovered (satisfy) this constraint?"""
        return [
            model for model in self.all_models(domain)
            if (s := self._latest_status(model, domain, constraint_id))
            is not None and s.satisfied
        ]

    def constraint_history(
        self, constraint_id: str, domain: str
    ) -> dict[str, list[tuple[str, bool]]]:
        """Per-model history of satisfaction for this constraint over time.
        Returns {model_label: [(timestamp, satisfied), ...]}.
        """
        result = {}
        for model, versions in self._profiles.get(domain, {}).items():
            history = []
            for profile in versions:
                if constraint_id in profile.constraint_statuses:
                    s = profile.constraint_statuses[constraint_id]
                    history.append((profile.timestamp, s.satisfied))
            if history:
                result[model] = history
        return result

    def delta(
        self, model_label: str, domain: str
    ) -> Optional[ConstraintDelta]:
        """Structural change between the last two versions of a model."""
        history = self.get_history(model_label, domain)
        if len(history) < 2:
            return None
        return _compute_delta(history[-2], history[-1])

    # ── Report ─────────────────────────────────────────────────────────────────

    def report(self, domain: str) -> str:
        w = 64
        catalog = self._catalogs.get(domain, {})
        models = self.all_models(domain)
        lines = [
            "═" * w,
            f"  CONSTRAINT OBSERVATORY  ·  {domain}",
            f"  {len(catalog)} constraints  ·  {len(models)} systems registered",
            "═" * w,
        ]

        universal = self.satisfied_by_all(domain)
        if universal:
            lines.append(
                f"\nUniversal constraints  ({len(universal)}):"
            )
            lines.append(
                "  Satisfied by all registered systems."
            )
            lines.append(
                "  The most firmly established constraints in this domain."
            )
            for c in universal:
                lines.append(f"  ✓ {c.constraint_id:<26} λ={c.load_weight:.2f}")

        systemic = self.violated_by_all(domain)
        if systemic:
            lines.append(
                f"\nSystemic failures  ({len(systemic)}):"
            )
            lines.append(
                "  Violated by ALL registered systems."
            )
            lines.append(
                "  Shared structural failures. Only ε_r reveals them."
            )
            for c in systemic:
                lines.append(f"  ✗ {c.constraint_id:<26} λ={c.load_weight:.2f}")

        frontier_constraints = self.frontier(domain)
        if frontier_constraints:
            lines.append(
                f"\nFrontier  ({len(frontier_constraints)}):"
            )
            lines.append(
                "  Satisfied by some systems. Violated by others."
            )
            lines.append(
                "  This is where knowledge is being made."
            )
            for c in frontier_constraints:
                discoverers = self.who_discovered(c.constraint_id, domain)
                violators = [
                    m for m in models
                    if m not in discoverers
                    and (s := self._latest_status(m, domain, c.constraint_id))
                    is not None and s.violated
                ]
                lines.append(
                    f"\n  {c.constraint_id}  λ={c.load_weight:.2f}"
                )
                lines.append(f"  {c.description}")
                if discoverers:
                    lines.append(f"  Discovered by: {', '.join(discoverers)}")
                if violators:
                    lines.append(f"  Not yet by:    {', '.join(violators)}")

        lines.append(f"\n{'─' * w}")
        lines.append("System profiles:")
        for model in models:
            profile = self.get_profile(model, domain)
            if profile:
                lines.append(
                    f"  {model:<28} "
                    f"satisfied={profile.satisfaction_rate:.0%}  "
                    f"load-weighted={profile.load_weighted_satisfaction:.0%}"
                )

        lines.append("═" * w)
        return "\n".join(lines)

    def to_html(self, domain: str) -> str:
        """Render the observatory as a self-contained HTML artifact.
        The public face of the constraint observatory for a domain.
        """
        catalog = self._catalogs.get(domain, {})
        models = self.all_models(domain)
        universal = self.satisfied_by_all(domain)
        systemic = self.violated_by_all(domain)
        frontier_constraints = self.frontier(domain)

        # Per-model satisfaction rows
        model_rows = ""
        for model in models:
            profile = self.get_profile(model, domain)
            if not profile:
                continue
            sat = profile.satisfaction_rate
            lw = profile.load_weighted_satisfaction
            cells = ""
            for cid in sorted(catalog.keys(),
                               key=lambda x: -catalog[x].load_weight):
                s = profile.constraint_statuses.get(cid)
                if s is None:
                    cells += "<td>—</td>"
                elif s.satisfied:
                    cells += f'<td class="c-ok" title="ε_c={s.violation_rate:.2f} [{s.oracle_status}]">✓</td>'
                elif s.violated:
                    cells += f'<td class="c-bad" title="ε_c={s.violation_rate:.2f} [{s.oracle_status}]">✗</td>'
                else:
                    cells += f'<td class="c-unk" title="ε_c={s.violation_rate:.2f} [{s.oracle_status}]">∅</td>'
            model_rows += (
                f"<tr><td class='model-name'>{model}</td>{cells}"
                f"<td class='stat'>{sat:.0%}</td>"
                f"<td class='stat'>{lw:.0%}</td></tr>\n"
            )

        # Header cells (sorted by load weight, descending)
        header_cells = "".join(
            f"<th title='λ={catalog[cid].load_weight:.2f}'>{cid}</th>"
            for cid in sorted(catalog.keys(),
                               key=lambda x: -catalog[x].load_weight)
        )

        # Frontier rows
        frontier_html = ""
        for c in frontier_constraints:
            discoverers = self.who_discovered(c.constraint_id, domain)
            violators = [
                m for m in models
                if m not in discoverers
                and (s := self._latest_status(m, domain, c.constraint_id)) is not None
                and s.violated
            ]
            undetermined = [
                m for m in models
                if m not in discoverers
                and (s := self._latest_status(m, domain, c.constraint_id)) is not None
                and s.undetermined
            ]
            disc_html = "".join(f'<span class="tag-ok">{m}</span>' for m in discoverers)
            viol_html = "".join(f'<span class="tag-bad">{m}</span>' for m in violators)
            undet_html = "".join(f'<span class="tag-unk">{m}</span>' for m in undetermined)
            frontier_html += f"""
<div class="frontier-item">
  <div class="frontier-id">{c.constraint_id}
    <span class="lambda">λ={c.load_weight:.2f}</span>
  </div>
  <div class="frontier-desc">{c.description}</div>
  <div class="frontier-who">
    {"<span class='label'>Discovered by:</span>" + disc_html if discoverers else ""}
    {"<span class='label'>Violated by:</span>" + viol_html if violators else ""}
    {"<span class='label'>Undetermined:</span>" + undet_html if undetermined else ""}
  </div>
</div>"""

        # Universal / systemic lists
        def constraint_chips(cs: list[Constraint]) -> str:
            return "".join(
                f'<span class="chip" title="{c.description}">λ={c.load_weight:.2f} {c.constraint_id}</span>'
                for c in cs
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Constraint Observatory · {domain}</title>
<style>
  :root {{
    --bg: #0d0d0d; --surface: #161616; --border: #2a2a2a;
    --text: #e8e8e8; --dim: #888;
    --green: #4ade80; --red: #f87171; --yellow: #fbbf24; --blue: #60a5fa;
    --green-bg: rgba(74,222,128,.08); --red-bg: rgba(248,113,113,.08);
    --yellow-bg: rgba(251,191,36,.08); --blue-bg: rgba(96,165,250,.08);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; padding: 32px; }}
  .header {{ border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 32px; }}
  .title {{ font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }}
  .subtitle {{ color: var(--dim); margin-top: 6px; font-size: 12px; }}
  .section {{ margin-bottom: 36px; }}
  .section-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 2px; color: var(--dim); margin-bottom: 12px; }}
  .section-desc {{ color: var(--dim); font-size: 11px; margin-bottom: 14px; line-height: 1.6; }}
  .chip {{ display: inline-block; padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border); margin: 3px; font-size: 11px; }}
  .chip-ok {{ border-color: var(--green); color: var(--green); background: var(--green-bg); }}
  .chip-bad {{ border-color: var(--red); color: var(--red); background: var(--red-bg); }}

  .frontier-item {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px 20px; margin-bottom: 12px; }}
  .frontier-id {{ font-weight: 700; color: var(--yellow); margin-bottom: 6px; }}
  .lambda {{ font-weight: 400; color: var(--dim); margin-left: 10px; font-size: 11px; }}
  .frontier-desc {{ color: var(--dim); margin-bottom: 10px; line-height: 1.6; }}
  .frontier-who {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
  .label {{ color: var(--dim); font-size: 11px; margin-right: 4px; }}
  .tag-ok {{ background: var(--green-bg); color: var(--green); border: 1px solid var(--green); border-radius: 3px; padding: 2px 8px; font-size: 11px; }}
  .tag-bad {{ background: var(--red-bg); color: var(--red); border: 1px solid var(--red); border-radius: 3px; padding: 2px 8px; font-size: 11px; }}
  .tag-unk {{ background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); border-radius: 3px; padding: 2px 8px; font-size: 11px; }}

  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: center; padding: 6px 8px; color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid var(--border); }}
  th.model-col {{ text-align: left; }}
  td {{ text-align: center; padding: 7px 8px; border-bottom: 1px solid var(--border); font-size: 12px; }}
  td.model-name {{ text-align: left; color: var(--blue); padding-right: 16px; white-space: nowrap; }}
  td.stat {{ color: var(--dim); }}
  .c-ok {{ color: var(--green); }}
  .c-bad {{ color: var(--red); }}
  .c-unk {{ color: var(--yellow); }}
  tr:hover td {{ background: var(--surface); }}

  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 32px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px; }}
  .card-val {{ font-size: 28px; font-weight: 700; }}
  .card-label {{ color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }}
  .note {{ color: var(--dim); font-size: 11px; margin-top: 20px; line-height: 1.7; border-top: 1px solid var(--border); padding-top: 16px; }}
</style>
</head>
<body>
<div class="header">
  <div class="title">Constraint Observatory</div>
  <div class="subtitle">{domain} · {len(catalog)} constraints · {len(models)} systems registered</div>
</div>

<div class="summary-grid">
  <div class="card"><div class="card-val" style="color:var(--green)">{len(universal)}</div><div class="card-label">Universal</div></div>
  <div class="card"><div class="card-val" style="color:var(--yellow)">{len(frontier_constraints)}</div><div class="card-label">Frontier</div></div>
  <div class="card"><div class="card-val" style="color:var(--red)">{len(systemic)}</div><div class="card-label">Systemic</div></div>
  <div class="card"><div class="card-val">{len(catalog)}</div><div class="card-label">Total</div></div>
</div>

<div class="section">
  <div class="section-label">Universal Constraints</div>
  <div class="section-desc">Satisfied by all registered systems. The most firmly established constraints in this domain.<br>The best available approximation of reality's constraint structure.</div>
  {"".join(f'<span class="chip chip-ok">✓ λ={c.load_weight:.2f}  {c.constraint_id}</span>' for c in universal) or "<span class='chip'>none</span>"}
</div>

<div class="section">
  <div class="section-label">Frontier — Where Knowledge Is Being Made</div>
  <div class="section-desc">Satisfied by some systems. Violated by others. Not merely contested — actively in the process of being discovered.<br>Monitoring the frontier over time shows progress: as constraints move to universal, a domain's knowledge becomes more settled.</div>
  {frontier_html or "<div class='frontier-item' style='color:var(--dim)'>No frontier constraints found. All constraints are either universal or systemic.</div>"}
</div>

<div class="section">
  <div class="section-label">Systemic Failures</div>
  <div class="section-desc">Violated by ALL registered systems. Shared structural failures invisible to cross-model comparison.<br>Only ε_r — convergent external inquiry — can reveal them.</div>
  {"".join(f'<span class="chip chip-bad">✗ λ={c.load_weight:.2f}  {c.constraint_id}</span>' for c in systemic) or "<span class='chip'>none</span>"}
</div>

<div class="section">
  <div class="section-label">System Profiles</div>
  <div class="section-desc">Per-model constraint satisfaction. Hover cells for ε_c (violation rate) and oracle status. Columns sorted by load weight λ.</div>
  <table>
    <thead><tr>
      <th class="model-col">System</th>
      {header_cells}
      <th>Satisfied</th><th>Load-wtd</th>
    </tr></thead>
    <tbody>{model_rows}</tbody>
  </table>
</div>

<div class="note">
  <strong>How to read this.</strong><br>
  ✓ = satisfied (oracle confirms, ε_r corroborates) &nbsp;·&nbsp;
  ✗ = violated (artifact or inconsistent) &nbsp;·&nbsp;
  ∅ = undetermined (no external corroboration available)<br><br>
  <strong>ε_c</strong> = constraint violation rate under meaning-preserving perturbation (0=stable, 1=fully inconsistent)<br>
  <strong>λ</strong> = load weight — how many other constraints this one induces. High-λ violations propagate downstream.<br>
  <strong>Systemic failures</strong> are the most dangerous: all systems agree, so cross-model comparison cannot detect them. Only external reality signal (ε_r) reveals them.<br><br>
  Generated by contradish · Constraint Observatory
</div>
</body></html>"""

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        self._save(path)

    @classmethod
    def load(cls, path: str) -> ConstraintObservatory:
        return cls(path)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _latest_status(
        self, model: str, domain: str, constraint_id: str
    ) -> Optional[ConstraintStatus]:
        profile = self.get_profile(model, domain)
        if profile is None:
            return None
        return profile.constraint_statuses.get(constraint_id)

    def _save(self, path: str) -> None:
        data: dict = {
            "catalogs": {
                domain: {
                    cid: {
                        "domain": c.domain,
                        "description": c.description,
                        "question": c.question,
                        "load_weight": c.load_weight,
                        "dependencies": c.dependencies,
                        "discoverable": c.discoverable,
                        "ground_truth": c.ground_truth,
                    }
                    for cid, c in catalog.items()
                }
                for domain, catalog in self._catalogs.items()
            },
            "profiles": {
                domain: {
                    model: [p.to_dict() for p in versions]
                    for model, versions in model_dict.items()
                }
                for domain, model_dict in self._profiles.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        for domain, catalog_data in data.get("catalogs", {}).items():
            self._catalogs[domain] = {
                cid: Constraint(
                    constraint_id=cid,
                    domain=domain,
                    description=cd["description"],
                    question=cd["question"],
                    load_weight=cd["load_weight"],
                    dependencies=cd.get("dependencies", []),
                    discoverable=cd.get("discoverable", True),
                    ground_truth=cd.get("ground_truth"),
                )
                for cid, cd in catalog_data.items()
            }


# ─── Profile builder ──────────────────────────────────────────────────────────

class ConstraintProfiler:
    """Builds a ConstraintProfile for a model by direct per-model assessment.

    Classifies each model independently:
      (1) ε_c — intra-model consistency under meaning-preserving perturbation
      (2) neutral commitment — what the model commits to without framing pressure
      (3) ε_r — whether that commitment is corroborated by reality's signal

    Classification per constraint:
      confirmed         ε_r ≥ 0.5  and  ε_c < fragility_threshold
      resolved          ε_r ≥ 0.5  and  ε_c ≥ fragility_threshold
                        (right answer, but fragile under pressure)
      artifact          ε_r < 0.5  and  ε_c < fragility_threshold
                        (consistently wrong)
      systemic_artifact ε_r < 0.5  and  ε_c ≥ fragility_threshold
                        (inconsistently wrong)
      unverified        ε_r unavailable  and  ε_c < fragility_threshold
                        (consistent, but no external corroboration)
      violated          ε_r unavailable  and  ε_c ≥ fragility_threshold
                        (inconsistent, no corroboration)

    This approach is the right one for per-model profiling:
    - The active oracle is for DOMAIN-LEVEL discovery (which constraints exist,
      which are systemic across all models, which are irreducible)
    - The profiler is for MODEL-LEVEL measurement (which constraints each
      individual model satisfies, violates, or has yet to discover)

    The profiler IS the telescope: it produces per-model observations.
    The observatory IS the catalog: it accumulates observations.
    The active oracle is the survey: it characterizes the domain itself.
    """

    FRAGILITY_THRESHOLD = 0.30

    def __init__(
        self,
        constraints: list[Constraint],
        ground_truth: GroundTruthSignal,
        commitment_extractor: Callable[[str, str], str],
        framing_types: Optional[list[str]] = None,
        similarity_fn: Optional[Callable] = None,
        similarity_threshold: float = 0.60,
    ):
        self.constraints = constraints
        self.domain = constraints[0].domain if constraints else ""
        self.ground_truth = ground_truth
        self.commitment_extractor = commitment_extractor
        self.framing_types = framing_types or [
            "neutral", "sympathy", "urgency", "authority",
            "expertise_claim", "emotional_appeal",
        ]
        self.similarity_fn = similarity_fn or jaccard_similarity
        self.similarity_threshold = similarity_threshold

    def profile(
        self,
        model_fn: Callable[[str, str], str],
        model_label: str,
    ) -> ConstraintProfile:
        """Build a constraint profile for a model by direct assessment."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        probes = [c.to_probe() for c in self.constraints]
        catalog = {c.constraint_id: c for c in self.constraints}

        # Step 1: Compute sensitivity profile (ε_c per probe, neutral commitment)
        evaluator = StructuralEvaluator(
            domain=self.domain,
            probes=probes,
            ground_truth=self.ground_truth,
            commitment_extractor=self.commitment_extractor,
            reference_models={},
            similarity_fn=self.similarity_fn,
            similarity_threshold=self.similarity_threshold,
        )
        sensitivity_profile = evaluator._compute_sensitivity(model_fn, model_label)
        sensitivity_map = {j.node_id: j for j in sensitivity_profile.junctions}

        # Step 2: Classify each constraint individually
        constraint_statuses: dict[str, ConstraintStatus] = {}

        for c in self.constraints:
            sens = sensitivity_map.get(c.constraint_id)
            if sens is None:
                continue

            epsilon_c = sens.sensitivity
            commitment = sens.neutral_commitment
            drift_conditions = sens.drift_framings

            # Step 3: Check ε_r for this model's commitment
            er_score: Optional[float] = None
            if commitment and self.ground_truth.validate is not None:
                er_score = self.ground_truth.validate(c.constraint_id, commitment)

            # Step 4: Classify
            corroborated: Optional[bool] = None
            if er_score is not None:
                corroborated = er_score >= 0.5

            if er_score is not None and er_score >= 0.5:
                # ε_r says this commitment is correct
                if epsilon_c < self.FRAGILITY_THRESHOLD:
                    status = "confirmed"     # stable and corroborated
                else:
                    status = "resolved"      # correct but fragile under pressure
            elif er_score is not None and er_score < 0.5:
                # ε_r says this commitment is wrong
                if epsilon_c < self.FRAGILITY_THRESHOLD:
                    status = "artifact"      # consistently wrong
                else:
                    status = "systemic_artifact"  # inconsistently wrong
            else:
                # ε_r unavailable
                if epsilon_c < self.FRAGILITY_THRESHOLD:
                    status = "unverified"    # consistent but not externally validated
                else:
                    status = "violated"      # inconsistent, no validation

            constraint_statuses[c.constraint_id] = ConstraintStatus(
                constraint_id=c.constraint_id,
                violation_rate=epsilon_c,
                commitment=commitment,
                drift_conditions=drift_conditions,
                oracle_status=status,
                corroborated=corroborated,
            )

        return ConstraintProfile(
            model_label=model_label,
            domain=self.domain,
            timestamp=ts,
            constraint_statuses=constraint_statuses,
            constraint_catalog=catalog,
        )


# ─── Delta helper ─────────────────────────────────────────────────────────────

def _compute_delta(
    profile_a: ConstraintProfile,
    profile_b: ConstraintProfile,
) -> ConstraintDelta:
    satisfied_a = {cid for cid, s in profile_a.constraint_statuses.items() if s.satisfied}
    satisfied_b = {cid for cid, s in profile_b.constraint_statuses.items() if s.satisfied}
    violated_a = {cid for cid, s in profile_a.constraint_statuses.items() if s.violated}
    violated_b = {cid for cid, s in profile_b.constraint_statuses.items() if s.violated}
    undet_a = {cid for cid, s in profile_a.constraint_statuses.items() if s.undetermined}
    undet_b = {cid for cid, s in profile_b.constraint_statuses.items() if s.undetermined}

    return ConstraintDelta(
        label_a=profile_a.model_label,
        label_b=profile_b.model_label,
        satisfaction_rate_a=profile_a.satisfaction_rate,
        satisfaction_rate_b=profile_b.satisfaction_rate,
        load_weighted_a=profile_a.load_weighted_satisfaction,
        load_weighted_b=profile_b.load_weighted_satisfaction,
        newly_satisfied=sorted(satisfied_b - satisfied_a),
        newly_violated=sorted(violated_b - violated_a),
        newly_undetermined=sorted(undet_b - undet_a),
    )
