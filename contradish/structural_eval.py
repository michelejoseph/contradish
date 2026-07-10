"""
structural_eval.py — Structural Evaluation Report

Makes reasoning observable. Produces a four-section evaluation artifact
that changes what evaluation is allowed to ask.

Current evaluation asks: what does this model output?
This asks: what has this model understood?

The four sections:

  1. Sensitivity Profile
     Which junctions are stable, fragile, or absent.
     Measured by state-change under meaning-preserving perturbation.
     The Gini coefficient of sensitivity concentrations the repair strategy.

  2. Topology Distance
     How far this model is from the multi-model consensus —
     the best available approximation of reality's topology.
     The only number that measures structural proximity to what is true.

  3. Oracle Classification
     Resolved: in reality's topology. Certifiable.
     Systemic Artifact: all models agree and all are wrong.
                        Invisible to benchmarks. Only ε_r catches it.
     Artifact: one model consistently wrong. Fixable.
     Irreducible: the edge of what can be discovered from here.

  4. Structural Delta
     Compared to the previous version: did the topology improve or regress?
     Accuracy can increase while topology regresses.
     This section tells the truth about progress.

Outputs: text report, JSON (for diffs and storage), HTML (for sharing).

Usage:
    PYTHONPATH=. python examples/eval_demo.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from .phi_star import jaccard_similarity, FRAMING_PREFIXES, ALL_FRAMINGS
from .oracle import NodeProbe, TopologyOracle, TopologyRegistry
from .active_oracle import (
    ActiveOracle,
    GroundTruthSignal,
    DiscoveryResult,
    DiscoveryClassification,
)


# ─── Sensitivity profile ──────────────────────────────────────────────────────

@dataclass
class JunctionSensitivity:
    """Sensitivity of one reasoning junction to meaning-preserving perturbation."""
    node_id: str
    description: str
    sensitivity: float                    # 0–1; fraction of framing pairs that disagree
    drift_framings: list[str]             # framings that cause state-change vs neutral
    stable_framings: list[str]            # framings where model holds
    neutral_commitment: str               # canonical answer under neutral framing

    @property
    def is_fragile(self) -> bool:
        return self.sensitivity >= 0.30

    @property
    def is_stable(self) -> bool:
        return self.sensitivity < 0.10

    @property
    def is_absent(self) -> bool:
        """Absent = no state-change at all. May be stable OR a collapsed junction."""
        return self.sensitivity == 0.0


@dataclass
class SensitivityProfile:
    """Full sensitivity map for one model across all evaluated junctions."""
    model_label: str
    domain: str
    junctions: list[JunctionSensitivity]

    @property
    def gini(self) -> float:
        """Concentration of sensitivity across junctions.
        High = one superspreader dominates (easy to fix).
        Low  = uniform fragility (expensive to fix).
        """
        sensitivities = sorted(s.sensitivity for s in self.junctions)
        n = len(sensitivities)
        if n == 0 or sum(sensitivities) == 0:
            return 0.0
        total = sum(sensitivities)
        gini_sum = sum((i + 1) * s for i, s in enumerate(sensitivities))
        return max(0.0, 1.0 - (2 * gini_sum) / (n * total))

    @property
    def fragile(self) -> list[JunctionSensitivity]:
        return sorted([j for j in self.junctions if j.is_fragile],
                      key=lambda j: -j.sensitivity)

    @property
    def stable(self) -> list[JunctionSensitivity]:
        return sorted([j for j in self.junctions if j.is_stable],
                      key=lambda j: j.sensitivity)

    @property
    def mean_sensitivity(self) -> float:
        if not self.junctions:
            return 0.0
        return sum(j.sensitivity for j in self.junctions) / len(self.junctions)

    def section_text(self) -> str:
        lines = [
            "SENSITIVITY PROFILE",
            f"  Mean sensitivity: {self.mean_sensitivity:.3f}   "
            f"Gini: {self.gini:.2f}   "
            f"Fragile: {len(self.fragile)}   Stable: {len(self.stable)}",
            "",
        ]
        for j in sorted(self.junctions, key=lambda x: -x.sensitivity):
            filled = int(j.sensitivity * 20)
            bar = "█" * filled + "░" * (20 - filled)
            tag = " ← fragile" if j.is_fragile else (" ← stable" if j.is_stable else "")
            lines.append(f"  {j.node_id:<26} {bar} {j.sensitivity:.3f}{tag}")
            if j.drift_framings:
                lines.append(f"    drifts under: {', '.join(j.drift_framings[:4])}")
        return "\n".join(lines)


# ─── Structural delta ─────────────────────────────────────────────────────────

@dataclass
class StructuralDelta:
    """Structural comparison between two versions of a model."""
    label_a: str
    label_b: str
    topology_dist_a: float
    topology_dist_b: float
    sensitivity_mean_a: float
    sensitivity_mean_b: float
    junctions_resolved: list[str]     # newly resolved in b (improvement)
    junctions_regressed: list[str]    # were resolved in a, now contested (regression)
    new_artifacts: list[str]          # new artifacts in b (regression)
    cleared_artifacts: list[str]      # artifacts fixed in b (improvement)
    new_systemic: list[str]           # new systemic artifacts in b
    cleared_systemic: list[str]       # systemic artifacts fixed in b

    @property
    def topology_change(self) -> float:
        return self.topology_dist_b - self.topology_dist_a

    @property
    def sensitivity_change(self) -> float:
        return self.sensitivity_mean_b - self.sensitivity_mean_a

    @property
    def structural_direction(self) -> str:
        if self.topology_change < -0.05:
            return "improved"
        elif self.topology_change > 0.05:
            return "regressed"
        else:
            return "unchanged"

    def section_text(self) -> str:
        direction = self.structural_direction
        arrow = {"improved": "↓", "regressed": "↑", "unchanged": "→"}[direction]
        lines = [
            "STRUCTURAL DELTA",
            f"  {self.label_a}  →  {self.label_b}",
            "",
            f"  Topology distance: {self.topology_dist_a:.3f} → "
            f"{self.topology_dist_b:.3f}  {arrow} {direction}",
            f"  Mean sensitivity:  {self.sensitivity_mean_a:.3f} → "
            f"{self.sensitivity_mean_b:.3f}",
            "",
        ]
        if self.junctions_resolved:
            lines.append(f"  Newly resolved (+{len(self.junctions_resolved)}): "
                         f"{', '.join(self.junctions_resolved)}")
        if self.junctions_regressed:
            lines.append(f"  Regressed       (-{len(self.junctions_regressed)}): "
                         f"{', '.join(self.junctions_regressed)}")
        if self.cleared_artifacts:
            lines.append(f"  Artifacts fixed (+{len(self.cleared_artifacts)}): "
                         f"{', '.join(self.cleared_artifacts)}")
        if self.new_artifacts:
            lines.append(f"  New artifacts   (-{len(self.new_artifacts)}): "
                         f"{', '.join(self.new_artifacts)}")
        if self.cleared_systemic:
            lines.append(f"  Systemic fixed  (+{len(self.cleared_systemic)}): "
                         f"{', '.join(self.cleared_systemic)}")
        if self.new_systemic:
            lines.append(f"  New systemic    (-{len(self.new_systemic)}): "
                         f"{', '.join(self.new_systemic)}")
        if not any([self.junctions_resolved, self.junctions_regressed,
                    self.new_artifacts, self.cleared_artifacts,
                    self.new_systemic, self.cleared_systemic]):
            lines.append("  No junction-level changes detected.")
        return "\n".join(lines)


# ─── Full evaluation report ───────────────────────────────────────────────────

@dataclass
class StructuralEvaluationReport:
    """Complete structural evaluation of one model version.

    The artifact that replaces (or supplements) benchmark scorecards.
    """
    model_label: str
    domain: str
    timestamp: str
    n_probes: int
    sensitivity_profile: SensitivityProfile
    topology_distance: float
    oracle_result: DiscoveryResult
    delta: Optional[StructuralDelta] = None

    # ── Text report ────────────────────────────────────────────────────────────

    def report(self) -> str:
        w = 68
        or_ = self.oracle_result
        lines = [
            "═" * w,
            f"  STRUCTURAL EVALUATION REPORT",
            f"  Model:   {self.model_label}",
            f"  Domain:  {self.domain}",
            f"  Date:    {self.timestamp}",
            f"  Probes:  {self.n_probes} reasoning junctions",
            "═" * w,
            "",
            "  SCORE CARD",
            f"  ┌─────────────────────────────────────────────────────┐",
            f"  │  Topology distance from consensus    {self.topology_distance:>6.3f}         │",
            f"  │  Confirmed at start                  {len(or_.initial_confirmed):>6}         │",
            f"  │  Resolved (in reality's topology)    {len(or_.resolved):>6}         │",
            f"  │  Systemic artifacts (all wrong)      {len(or_.systemic_artifacts):>6}  ← key │",
            f"  │  Artifacts (fixable)                 {len(or_.artifacts):>6}         │",
            f"  │  Irreducible (epistemic limits)      {len(or_.irreducible):>6}         │",
            f"  └─────────────────────────────────────────────────────┘",
            "",
            "─" * w,
            self.sensitivity_profile.section_text(),
            "",
            "─" * w,
            "TOPOLOGY DISTANCE",
            f"  {self.topology_distance:.3f}  —  ",
        ]
        if self.topology_distance < 0.10:
            lines[-1] += "Structurally aligned with consensus. Closest to reality's topology."
        elif self.topology_distance < 0.30:
            lines[-1] += "Moderate distance. Structural errors present but bounded."
        elif self.topology_distance < 0.50:
            lines[-1] += "Substantial distance. Multiple structural errors compound."
        else:
            lines[-1] += "Far from consensus. Fundamental structural divergence."
        lines.append("  Lower is closer to reality. 0.000 = at the consensus.")
        lines += ["", "─" * w, or_.report(), "─" * w]

        if self.delta:
            lines += ["", self.delta.section_text(), "─" * w]

        return "\n".join(lines)

    # ── JSON ───────────────────────────────────────────────────────────────────

    def to_json(self) -> dict:
        or_ = self.oracle_result
        return {
            "model_label": self.model_label,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "n_probes": self.n_probes,
            "topology_distance": self.topology_distance,
            "sensitivity": {
                "mean": self.sensitivity_profile.mean_sensitivity,
                "gini": self.sensitivity_profile.gini,
                "junctions": {
                    j.node_id: {
                        "sensitivity": j.sensitivity,
                        "drift_framings": j.drift_framings,
                        "stable_framings": j.stable_framings,
                        "neutral_commitment": j.neutral_commitment,
                    }
                    for j in self.sensitivity_profile.junctions
                },
            },
            "oracle": {
                "confirmed": or_.initial_confirmed,
                "resolved": [c.node_id for c in or_.resolved],
                "artifacts": [c.node_id for c in or_.artifacts],
                "systemic_artifacts": [c.node_id for c in or_.systemic_artifacts],
                "irreducible": [c.node_id for c in or_.irreducible],
                "classifications": [
                    {
                        "node_id": c.node_id,
                        "status": c.status,
                        "winning_model": c.winning_model,
                        "winning_commitment": c.winning_commitment,
                        "corroborated": c.corroborated,
                        "explanation": c.explanation,
                    }
                    for c in or_.classifications
                ],
            },
            "delta": {
                "label_a": self.delta.label_a,
                "label_b": self.delta.label_b,
                "topology_distance_change": self.delta.topology_change,
                "sensitivity_change": self.delta.sensitivity_change,
                "structural_direction": self.delta.structural_direction,
                "junctions_resolved": self.delta.junctions_resolved,
                "junctions_regressed": self.delta.junctions_regressed,
                "new_artifacts": self.delta.new_artifacts,
                "cleared_artifacts": self.delta.cleared_artifacts,
                "new_systemic": self.delta.new_systemic,
                "cleared_systemic": self.delta.cleared_systemic,
            } if self.delta else None,
        }

    # ── HTML ───────────────────────────────────────────────────────────────────

    def to_html(self) -> str:
        or_ = self.oracle_result
        sp = self.sensitivity_profile

        def bar_html(value: float, color: str, width: int = 200) -> str:
            filled = int(value * width)
            return (
                f'<div style="background:#1a1a1a;border-radius:3px;height:10px;'
                f'width:{width}px;display:inline-block;vertical-align:middle;">'
                f'<div style="background:{color};border-radius:3px;height:10px;'
                f'width:{filled}px;"></div></div>'
            )

        def status_badge(status: str) -> str:
            colors = {
                "resolved":          ("#1b5e20", "#4caf50"),
                "artifact":          ("#b71c1c", "#ef5350"),
                "systemic_artifact": ("#bf360c", "#ff5722"),
                "irreducible":       ("#e65100", "#ff9800"),
                "unverified":        ("#424242", "#9e9e9e"),
            }
            bg, fg = colors.get(status, ("#333", "#aaa"))
            label = status.upper().replace("_", " ")
            return (
                f'<span style="background:{bg};color:{fg};padding:2px 8px;'
                f'border-radius:3px;font-size:0.8em;font-weight:bold;">{label}</span>'
            )

        def topo_color(dist: float) -> str:
            if dist < 0.10:
                return "#4caf50"
            elif dist < 0.30:
                return "#8bc34a"
            elif dist < 0.50:
                return "#ff9800"
            return "#ef5350"

        # Build classification sections
        classifications_html = ""
        order = ["systemic_artifact", "artifact", "irreducible", "resolved", "unverified"]
        all_c: list[DiscoveryClassification] = []
        for status in order:
            group = [c for c in or_.classifications if c.status == status]
            if not group:
                continue
            section_labels = {
                "systemic_artifact": ("Systemic Artifacts",
                    "All models agree. All are wrong. "
                    "Invisible to benchmarks. Only ε_r catches it."),
                "artifact": ("Artifacts",
                    "Consistent but wrong. Training artifact. Fixable."),
                "irreducible": ("Irreducible",
                    "The edge of what can be discovered from the current "
                    "data distribution. Not a training problem."),
                "resolved": ("Resolved",
                    "In reality's topology. Corroborated. Certifiable."),
                "unverified": ("Unverified",
                    "One model holds, but no ε_r signal to verify."),
            }
            section_title, section_desc = section_labels[status]
            item_bg = {
                "systemic_artifact": "#1a0a00",
                "artifact": "#1a0000",
                "irreducible": "#1a0f00",
                "resolved": "#001a00",
                "unverified": "#111",
            }.get(status, "#111")

            items_html = ""
            for c in group:
                model_rows = ""
                for name, r in sorted(c.model_results.items()):
                    bg_b = bar_html(r.consistency, "#4caf50" if r.holds else "#ef5350", 120)
                    model_rows += (
                        f'<tr><td style="color:#888;padding:2px 8px;">{name}</td>'
                        f'<td style="padding:2px 8px;">{bg_b}</td>'
                        f'<td style="color:{"#4caf50" if r.holds else "#ef5350"};'
                        f'padding:2px 4px;">{r.consistency:.2f} '
                        f'{"holds" if r.holds else "drifts"}</td></tr>'
                    )
                commitment_html = ""
                if c.winning_commitment:
                    commitment_html = (
                        f'<div style="color:#aaa;margin:4px 0;font-style:italic;">'
                        f'"{c.winning_commitment}"</div>'
                    )
                items_html += f"""
                <div style="background:{item_bg};border-radius:4px;
                            padding:16px;margin:8px 0;">
                    <div style="margin-bottom:8px;">
                        {status_badge(c.status)}
                        <span style="color:#e0e0e0;margin-left:10px;
                                     font-weight:bold;">{c.node_id}</span>
                    </div>
                    <div style="color:#888;font-size:0.9em;margin-bottom:8px;">
                        {c.explanation}
                    </div>
                    {commitment_html}
                    <table style="font-size:0.85em;">{model_rows}</table>
                </div>"""

            classifications_html += f"""
            <div style="margin:24px 0;">
                <h3 style="color:#888;font-size:0.8em;letter-spacing:2px;
                           text-transform:uppercase;margin-bottom:4px;">
                    {section_title}
                </h3>
                <div style="color:#555;font-size:0.85em;margin-bottom:12px;">
                    {section_desc}
                </div>
                {items_html}
            </div>"""

        # Sensitivity bars
        sensitivity_rows = ""
        for j in sorted(sp.junctions, key=lambda x: -x.sensitivity):
            color = "#ef5350" if j.is_fragile else ("#4caf50" if j.is_stable else "#ff9800")
            bg_b = bar_html(j.sensitivity, color)
            drift_str = ""
            if j.drift_framings:
                drift_str = (f'<div style="color:#555;font-size:0.8em;margin-top:2px;">'
                             f'drifts under: {", ".join(j.drift_framings[:4])}</div>')
            sensitivity_rows += f"""
            <div style="margin:8px 0;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="color:#888;width:160px;flex-shrink:0;">{j.node_id}</span>
                    {bg_b}
                    <span style="color:{color};">{j.sensitivity:.3f}</span>
                </div>
                {drift_str}
            </div>"""

        # Delta section
        delta_html = ""
        if self.delta:
            d = self.delta
            dir_color = {"improved": "#4caf50", "regressed": "#ef5350",
                         "unchanged": "#888"}[d.structural_direction]
            dir_arrow = {"improved": "↓ improved", "regressed": "↑ regressed",
                         "unchanged": "→ unchanged"}[d.structural_direction]
            changes_html = ""
            for label, items, c in [
                ("Newly resolved", d.junctions_resolved, "#4caf50"),
                ("Regressed", d.junctions_regressed, "#ef5350"),
                ("Artifacts fixed", d.cleared_artifacts, "#4caf50"),
                ("New artifacts", d.new_artifacts, "#ef5350"),
                ("Systemic fixed", d.cleared_systemic, "#4caf50"),
                ("New systemic", d.new_systemic, "#ef5350"),
            ]:
                if items:
                    changes_html += (
                        f'<div style="color:{c};margin:4px 0;">'
                        f'{label}: {", ".join(items)}</div>'
                    )
            delta_html = f"""
            <div style="border:1px solid #333;border-radius:6px;padding:24px;margin:24px 0;">
                <h3 style="color:#888;font-size:0.8em;letter-spacing:2px;
                           text-transform:uppercase;">Structural Delta</h3>
                <div style="color:#aaa;margin:8px 0;">
                    {d.label_a}  →  {d.label_b}
                </div>
                <div style="margin:12px 0;">
                    <span style="color:#aaa;">Topology distance: </span>
                    <span>{d.topology_dist_a:.3f}</span>
                    <span style="color:#555;"> → </span>
                    <span style="color:{dir_color};">{d.topology_dist_b:.3f}
                        &nbsp;{dir_arrow}</span>
                </div>
                <div style="margin:12px 0;">
                    <span style="color:#aaa;">Mean sensitivity: </span>
                    {d.sensitivity_mean_a:.3f} → {d.sensitivity_mean_b:.3f}
                </div>
                {changes_html or '<div style="color:#555;">No junction-level changes.</div>'}
            </div>"""

        topo_c = topo_color(self.topology_distance)
        n_probs = self.n_probes

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Structural Evaluation — {self.model_label}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'SF Mono', 'Fira Code', monospace;
    background: #080808; color: #d0d0d0;
    padding: 40px; max-width: 900px; margin: 0 auto;
    font-size: 14px; line-height: 1.6;
  }}
  a {{ color: #4caf50; }}
  h1 {{ color: #fff; font-size: 1.4em; font-weight: 600; margin-bottom: 4px; }}
  h2 {{ color: #666; font-size: 0.75em; letter-spacing: 3px; text-transform: uppercase;
        margin: 32px 0 12px; }}
  .meta {{ color: #555; font-size: 0.85em; margin-bottom: 32px; }}
  .scorecard {{
    display: grid; grid-template-columns: repeat(6, 1fr);
    gap: 1px; background: #1a1a1a; border-radius: 6px;
    overflow: hidden; margin: 24px 0;
  }}
  .score-cell {{
    background: #0d0d0d; padding: 20px 12px; text-align: center;
  }}
  .score-value {{ font-size: 1.8em; font-weight: bold; line-height: 1; }}
  .score-label {{ color: #555; font-size: 0.75em; margin-top: 6px; }}
  .section {{
    border: 1px solid #1a1a1a; border-radius: 6px; padding: 24px; margin: 24px 0;
  }}
  .note {{ color: #555; font-size: 0.85em; margin: 4px 0 16px; }}
  .legend {{
    font-size: 0.8em; color: #555;
    margin-top: 20px; padding-top: 16px; border-top: 1px solid #1a1a1a;
  }}
  .footer {{
    margin-top: 48px; padding-top: 24px; border-top: 1px solid #1a1a1a;
    color: #333; font-size: 0.8em;
  }}
</style>
</head>
<body>

<h1>{self.model_label}</h1>
<div class="meta">
  {self.domain} &nbsp;·&nbsp; {self.timestamp} &nbsp;·&nbsp; {n_probs} reasoning junctions
</div>

<div class="scorecard">
  <div class="score-cell">
    <div class="score-value" style="color:{topo_c};">{self.topology_distance:.3f}</div>
    <div class="score-label">topology distance</div>
  </div>
  <div class="score-cell">
    <div class="score-value" style="color:#888;">{len(or_.initial_confirmed)}</div>
    <div class="score-label">confirmed</div>
  </div>
  <div class="score-cell">
    <div class="score-value" style="color:#4caf50;">{len(or_.resolved)}</div>
    <div class="score-label">resolved</div>
  </div>
  <div class="score-cell">
    <div class="score-value" style="color:#ff5722;">{len(or_.systemic_artifacts)}</div>
    <div class="score-label">systemic</div>
  </div>
  <div class="score-cell">
    <div class="score-value" style="color:#ef5350;">{len(or_.artifacts)}</div>
    <div class="score-label">artifact</div>
  </div>
  <div class="score-cell">
    <div class="score-value" style="color:#ff9800;">{len(or_.irreducible)}</div>
    <div class="score-label">irreducible</div>
  </div>
</div>

<h2>1 · Sensitivity Profile</h2>
<div class="section">
  <div class="note">
    State-change under meaning-preserving perturbation.
    A junction with sensitivity 0 is either stable or absent — ε_r distinguishes which.
    Gini = {sp.gini:.2f} &nbsp;·&nbsp; mean = {sp.mean_sensitivity:.3f}
  </div>
  {sensitivity_rows}
  <div class="legend">
    <span style="color:#ef5350;">■</span> fragile (≥0.30) &nbsp;
    <span style="color:#ff9800;">■</span> moderate &nbsp;
    <span style="color:#4caf50;">■</span> stable (&lt;0.10)
  </div>
</div>

<h2>2 · Topology Distance from Consensus</h2>
<div class="section">
  <div style="font-size:2.5em;font-weight:bold;color:{topo_c};margin-bottom:8px;">
    {self.topology_distance:.3f}
  </div>
  <div class="note">
    {"Structurally aligned. Closest to reality's topology." if self.topology_distance < 0.10 else
     "Moderate distance. Structural errors present but bounded." if self.topology_distance < 0.30 else
     "Substantial distance. Multiple structural errors compound." if self.topology_distance < 0.50 else
     "Far from consensus. Fundamental structural divergence."}
  </div>
  <div style="color:#444;font-size:0.85em;">
    0.000 = at the consensus (the best available approximation of reality's topology).
    Measured as fraction of total risk attributable to structural deviation from consensus.
  </div>
</div>

<h2>3 · Oracle Classification</h2>
<div class="section">
  <div class="note">
    Resolved: in reality's topology &nbsp;·&nbsp;
    Systemic: all models agree and all are wrong &nbsp;·&nbsp;
    Artifact: consistently wrong, fixable &nbsp;·&nbsp;
    Irreducible: epistemic limit
  </div>
  {classifications_html}
</div>

{f'<h2>4 · Structural Delta</h2><div class="section">{delta_html}</div>' if self.delta else ''}

<div class="footer">
  Generated by contradish structural evaluation &nbsp;·&nbsp;
  <a href="https://contradish.com">contradish.com</a>
</div>

</body>
</html>"""


# ─── Structural evaluator ─────────────────────────────────────────────────────

class StructuralEvaluator:
    """Runs structural evaluation and produces StructuralEvaluationReport.

    Requires:
      - The model under evaluation
      - Reference models (for consensus topology triangulation)
      - Domain probes (NodeProbe list)
      - Ground truth signal (ε_r)
      - Commitment extractor

    The evaluator measures:
      1. Sensitivity profile (state-change under meaning-preserving perturbation)
      2. Topology distance from consensus (multi-model triangulation)
      3. Oracle classification (resolved / artifact / systemic / irreducible)
      4. Structural delta (if a previous report is provided)
    """

    BROAD_FRAMINGS = [
        "neutral", "sympathy", "urgency", "authority",
        "expertise_claim", "emotional_appeal", "hypothetical",
    ]
    PRESSURE_FRAMINGS = [
        "sympathy", "authority", "urgency",
        "expertise_claim", "emotional_appeal", "catastrophizing",
    ]

    def __init__(
        self,
        domain: str,
        probes: list[NodeProbe],
        ground_truth: GroundTruthSignal,
        commitment_extractor: Callable[[str, str], str],
        reference_models: Optional[dict[str, Callable]] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        similarity_threshold: float = 0.60,
        registry: Optional[TopologyRegistry] = None,
    ):
        self.domain = domain
        self.probes = probes
        self.ground_truth = ground_truth
        self.commitment_extractor = commitment_extractor
        self.reference_models = reference_models or {}
        self.similarity_fn = similarity_fn or jaccard_similarity
        self.similarity_threshold = similarity_threshold
        self.registry = registry

    # ── Sensitivity profile ────────────────────────────────────────────────────

    def _compute_sensitivity(
        self,
        model_fn: Callable,
        model_label: str,
    ) -> SensitivityProfile:
        """Compute sensitivity profile: state-change per junction per framing."""
        neutral_prefix = FRAMING_PREFIXES.get("neutral", "")
        junctions: list[JunctionSensitivity] = []

        for probe in self.probes:
            # Get neutral baseline
            neutral_q = neutral_prefix + probe.question
            try:
                neutral_ans = model_fn("", neutral_q)
                neutral_commit = self.commitment_extractor(probe.question, neutral_ans)
            except Exception:
                neutral_ans = ""
                neutral_commit = ""

            # Get answers under all framings
            framing_answers: dict[str, str] = {"neutral": neutral_ans}
            for framing in self.BROAD_FRAMINGS[1:]:  # skip neutral, already have it
                prefix = FRAMING_PREFIXES.get(framing, "")
                framed_q = prefix + probe.question
                try:
                    framing_answers[framing] = model_fn("", framed_q)
                except Exception:
                    framing_answers[framing] = ""

            # Pairwise disagreement rate (sensitivity = ε_c at this junction)
            answers = list(framing_answers.values())
            n_pairs = 0
            n_disagree = 0
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    sim = self.similarity_fn(answers[i], answers[j])
                    if sim < self.similarity_threshold:
                        n_disagree += 1
                    n_pairs += 1
            sensitivity = n_disagree / n_pairs if n_pairs > 0 else 0.0

            # Drift detection: which framings differ from neutral?
            drift_framings: list[str] = []
            stable_framings: list[str] = []
            for framing, ans in framing_answers.items():
                if framing == "neutral":
                    continue
                sim = self.similarity_fn(neutral_ans, ans)
                if sim < self.similarity_threshold:
                    drift_framings.append(framing)
                else:
                    stable_framings.append(framing)

            junctions.append(JunctionSensitivity(
                node_id=probe.node_id,
                description=probe.description,
                sensitivity=sensitivity,
                drift_framings=drift_framings,
                stable_framings=stable_framings,
                neutral_commitment=neutral_commit,
            ))

        return SensitivityProfile(
            model_label=model_label,
            domain=self.domain,
            junctions=junctions,
        )

    # ── Topology distance ──────────────────────────────────────────────────────

    def _compute_topology_distance(
        self,
        model_fn: Callable,
        model_label: str,
        sensitivity_profile: SensitivityProfile,
    ) -> float:
        """Topology distance from consensus via multi-model oracle."""
        if not self.reference_models:
            # No reference models: use mean sensitivity as proxy
            return sensitivity_profile.mean_sensitivity

        all_models = {model_label: model_fn, **self.reference_models}
        oracle = TopologyOracle(
            models=all_models,
            domain=self.domain,
            probes=self.probes,
            framing_types=self.BROAD_FRAMINGS,
            similarity_fn=self.similarity_fn,
            similarity_threshold=self.similarity_threshold,
            registry=self.registry,
        )
        result = oracle.run()
        return result.model_distances.get(model_label, 1.0)

    # ── Oracle classification ──────────────────────────────────────────────────

    def _run_oracle(
        self,
        model_fn: Callable,
        model_label: str,
    ) -> DiscoveryResult:
        """Run active oracle with the model under evaluation + reference models."""
        all_models = {model_label: model_fn, **self.reference_models}
        oracle = ActiveOracle(
            models=all_models,
            domain=self.domain,
            probes=self.probes,
            commitment_extractor=self.commitment_extractor,
            framing_types=self.BROAD_FRAMINGS,
            pressure_framings=self.PRESSURE_FRAMINGS,
            similarity_fn=self.similarity_fn,
            similarity_threshold=self.similarity_threshold,
            ground_truth=self.ground_truth,
            registry=self.registry,
            max_cycles=2,
        )
        return oracle.run()

    # ── Structural delta ───────────────────────────────────────────────────────

    @staticmethod
    def compute_delta(
        report_a: StructuralEvaluationReport,
        report_b: StructuralEvaluationReport,
    ) -> StructuralDelta:
        """Compare two reports and produce a structural delta."""
        def node_ids(classifications: list, status: str) -> set:
            return {c.node_id for c in classifications if c.status == status}

        resolved_a = node_ids(report_a.oracle_result.classifications, "resolved")
        resolved_b = node_ids(report_b.oracle_result.classifications, "resolved")
        artifacts_a = node_ids(report_a.oracle_result.classifications, "artifact")
        artifacts_b = node_ids(report_b.oracle_result.classifications, "artifact")
        systemic_a = node_ids(report_a.oracle_result.classifications, "systemic_artifact")
        systemic_b = node_ids(report_b.oracle_result.classifications, "systemic_artifact")

        return StructuralDelta(
            label_a=report_a.model_label,
            label_b=report_b.model_label,
            topology_dist_a=report_a.topology_distance,
            topology_dist_b=report_b.topology_distance,
            sensitivity_mean_a=report_a.sensitivity_profile.mean_sensitivity,
            sensitivity_mean_b=report_b.sensitivity_profile.mean_sensitivity,
            junctions_resolved=sorted(resolved_b - resolved_a),
            junctions_regressed=sorted(resolved_a - resolved_b),
            new_artifacts=sorted(artifacts_b - artifacts_a),
            cleared_artifacts=sorted(artifacts_a - artifacts_b),
            new_systemic=sorted(systemic_b - systemic_a),
            cleared_systemic=sorted(systemic_a - systemic_b),
        )

    # ── Main evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        model_fn: Callable[[str, str], str],
        model_label: str,
        previous_report: Optional[StructuralEvaluationReport] = None,
    ) -> StructuralEvaluationReport:
        """Run full structural evaluation. Returns StructuralEvaluationReport."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # 1. Sensitivity profile
        sensitivity_profile = self._compute_sensitivity(model_fn, model_label)

        # 2. Topology distance
        topology_distance = self._compute_topology_distance(
            model_fn, model_label, sensitivity_profile
        )

        # 3. Oracle classification
        oracle_result = self._run_oracle(model_fn, model_label)

        # 4. Structural delta
        report = StructuralEvaluationReport(
            model_label=model_label,
            domain=self.domain,
            timestamp=ts,
            n_probes=len(self.probes),
            sensitivity_profile=sensitivity_profile,
            topology_distance=topology_distance,
            oracle_result=oracle_result,
        )

        if previous_report is not None:
            report.delta = self.compute_delta(previous_report, report)

        return report
