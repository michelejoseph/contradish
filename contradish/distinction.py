"""
distinction.py — Distinction loss profiling.

Reasoning is the maintenance of distinctions that can all remain true together.
Knowledge is stable distinctions.
Hallucination is an unsupported distinction.
A model fails not by being wrong but by losing the ability to tell things apart.

This module asks: which distinctions can the model no longer make?

For every pair of states (A, B) that the model SHOULD distinguish — healthy adult
vs renal patient, guideline vs rule, "can do" vs "should do" — we probe whether
the model's outputs for A and B are distinguishable or whether they have collapsed
to the same response.

There are two types of distinction:

    Type I   — Should-distinguish: A and B represent different facts that require
               different responses. Loss = model gives same answer to both.
               Example: "healthy adult" vs "renal patient" dosing.

    Type II  — Should-not-distinguish: A and B differ only in framing, not in
               fact. The correct answer is the same. Loss = model gives different
               answers.
               Example: "calm user" vs "distressed user" asking the same factual
               question.

Type II failures are the surrender curves. This module measures Type I.
When framing causes Type I collapse, the model has lost the ability to maintain
the distinction that makes the question answerable correctly.

Usage::

    from contradish import DistinctionProber, DistinctionPair

    probe = DistinctionProber(
        model_fn             = my_model,
        pairs                = my_pairs,
        commitment_extractor = my_extractor,
    )
    loss_map = probe.measure()
    print(loss_map.report())
    loss_map.to_html("distinctions.html")
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Callable

from .surrender import PRESSURE_LEVELS, ALL_PRESSURE_TYPES, _esc, _FRAMING_COLORS

ModelFn = Callable[[str, str], str]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DistinctionPair:
    """
    A pair of states (A, B) that a model SHOULD distinguish.

    The pair defines:
      - Two questions that probe each state
      - The expected commitment for each state
      - Whether they should produce different commitments
        (always True for Type I distinctions)
    """
    pair_id:       str
    description:   str            # "healthy adult vs renal patient"
    label_a:       str            # "healthy adult"
    label_b:       str            # "renal patient"
    question_a:    str            # question that probes state A
    question_b:    str            # question that probes state B
    commit_a:      str            # expected commitment for state A
    commit_b:      str            # expected commitment for state B


@dataclass
class DistinctionMeasurement:
    """One probe at a specific (pair, framing_type, intensity)."""
    pair_id:              str
    framing_type:         str
    intensity:            int
    framing_prefix:       str
    answer_a:             str
    answer_b:             str
    extracted_commit_a:   str
    extracted_commit_b:   str
    # True if the model gave different answers (distinction preserved)
    # False if the model gave equivalent answers (distinction collapsed)
    distinction_held:     bool
    # True if both commits were correct
    both_correct:         bool


@dataclass
class DistinctionProfile:
    """
    Full measurement profile for one distinction pair.

    Attributes
    ----------
    pair_id
    description
    label_a, label_b
        Labels for the two distinguishable states.
    hold_rate_per_framing
        Per framing: fraction of intensities where distinction was preserved.
    overall_hold_rate
        Mean across all framings and intensities.
    collapse_framing
        Framing type where distinction collapses most.
    first_collapse
        (framing_type, intensity) of the first collapse, or None.
    measurements
        All raw measurements.
    """
    pair_id:               str
    description:           str
    label_a:               str
    label_b:               str
    hold_rate_per_framing: dict[str, float]
    overall_hold_rate:     float
    collapse_framing:      str
    first_collapse:        tuple[str, int] | None
    measurements:          list[DistinctionMeasurement] = field(default_factory=list)

    def collapse_rate(self) -> float:
        return 1.0 - self.overall_hold_rate


@dataclass
class DistinctionLossMap:
    """
    Full profiling of a set of distinction pairs.

    The loss map answers: given this model under this domain,
    which distinctions survive pressure and which collapse?
    """
    domain:   str
    profiles: dict[str, DistinctionProfile]   # pair_id → profile

    # Global summary
    ranked_by_fragility: list[str]  # pair_ids, most fragile first
    most_fragile:        str        # pair that collapses most
    most_resilient:      str        # pair that holds most
    framing_destructiveness: dict[str, float]  # framing → mean collapse rate

    def report(self) -> str:
        W   = 72
        sep = "─" * W
        bar = lambda r, w=16: "█" * round(r * w) + "░" * (w - round(r * w))

        lines = [
            "",
            f"  DISTINCTION LOSS MAP  ·  {self.domain}",
            sep,
            f"  {len(self.profiles)} distinctions probed",
            f"  most fragile   : {self.most_fragile}",
            f"  most resilient : {self.most_resilient}",
            "",
            "  RANKED  (most fragile → most resilient)",
            "",
        ]

        for pid in self.ranked_by_fragility:
            p  = self.profiles[pid]
            cr = p.collapse_rate()
            fc = self.framing_destructiveness
            worst_framing = p.collapse_framing
            lines.append(
                f"  {pid:<32}  collapse={cr:.0%}  "
                f"{bar(cr)}  → {worst_framing}"
            )

        lines += ["", "  FRAMING DESTRUCTIVENESS  (mean collapse rate per framing)", ""]
        for ft, rate in sorted(
            self.framing_destructiveness.items(), key=lambda kv: -kv[1]
        ):
            lines.append(
                f"  {ft:<22}  {rate:.0%}  {bar(rate)}"
            )

        return "\n".join(lines)

    def to_html(self, path: str | None = None) -> str:
        html = _render_loss_map_html(self)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
        return html


# ── Prober ────────────────────────────────────────────────────────────────────

class DistinctionProber:
    """
    Measure which distinctions a model can maintain across pressure framings.

    Parameters
    ----------
    model_fn
        (system_prompt, question) → answer
    pairs
        List of DistinctionPair objects to probe.
    commitment_extractor
        (question, answer) → commitment string
    system_prompt
        Passed to model_fn on every call.
    pressure_types
        Subset of framing types. Default: all 8.
    intensities
        Subset of intensity levels (1–5). Default: all 5.
    """

    def __init__(
        self,
        model_fn:             ModelFn,
        pairs:                list[DistinctionPair],
        commitment_extractor: Callable[[str, str], str],
        system_prompt:        str = "",
        pressure_types:       list[str] | None = None,
        intensities:          list[int] | None = None,
        domain:               str = "general",
    ):
        self.model_fn   = model_fn
        self.pairs      = pairs
        self.extractor  = commitment_extractor
        self.system_prompt  = system_prompt
        self.pressure_types = pressure_types or ALL_PRESSURE_TYPES
        self.intensities    = intensities or [1, 2, 3, 4, 5]
        self.domain         = domain

    def measure(
        self,
        n_samples: int = 1,
        verbose:   bool = False,
    ) -> DistinctionLossMap:
        """
        Run the full measurement grid.

        n_samples : int
            Samples per (pair, framing, intensity). 1 is fast; 3+ for variance.
        """
        profiles: dict[str, DistinctionProfile] = {}

        for pair in self.pairs:
            if verbose:
                print(f"  Probing: {pair.pair_id}")
            profile = self._probe_pair(pair, n_samples, verbose)
            profiles[pair.pair_id] = profile

        # ── Aggregate ────────────────────────────────────────────────────────
        ranked = sorted(
            profiles, key=lambda k: profiles[k].overall_hold_rate
        )  # most fragile first (lowest hold rate)

        framing_collapse: dict[str, list[float]] = {
            ft: [] for ft in self.pressure_types
        }
        for p in profiles.values():
            for ft in self.pressure_types:
                hold = p.hold_rate_per_framing.get(ft, 1.0)
                framing_collapse[ft].append(1.0 - hold)

        framing_destructiveness = {
            ft: statistics.mean(rates) if rates else 0.0
            for ft, rates in framing_collapse.items()
        }

        return DistinctionLossMap(
            domain                  = self.domain,
            profiles                = profiles,
            ranked_by_fragility     = ranked,
            most_fragile            = ranked[0]  if ranked else "",
            most_resilient          = ranked[-1] if ranked else "",
            framing_destructiveness = framing_destructiveness,
        )

    def _probe_pair(
        self,
        pair:      DistinctionPair,
        n_samples: int,
        verbose:   bool,
    ) -> DistinctionProfile:
        measurements: list[DistinctionMeasurement] = []
        hold_by_framing: dict[str, list[bool]] = {}

        for framing_type in self.pressure_types:
            level_data = PRESSURE_LEVELS[framing_type]
            holds: list[bool] = []

            for intensity in self.intensities:
                prefix = next(
                    (p for lvl, p in level_data if lvl == intensity),
                    level_data[min(intensity, len(level_data)) - 1][1],
                )
                q_a = prefix + pair.question_a
                q_b = prefix + pair.question_b

                for _ in range(n_samples):
                    ans_a = self.model_fn(self.system_prompt, q_a)
                    ans_b = self.model_fn(self.system_prompt, q_b)
                    com_a = self.extractor(pair.question_a, ans_a)
                    com_b = self.extractor(pair.question_b, ans_b)

                    held = com_a != com_b  # distinction preserved iff different
                    both_ok = (com_a == pair.commit_a and com_b == pair.commit_b)

                    measurements.append(DistinctionMeasurement(
                        pair_id            = pair.pair_id,
                        framing_type       = framing_type,
                        intensity          = intensity,
                        framing_prefix     = prefix,
                        answer_a           = ans_a,
                        answer_b           = ans_b,
                        extracted_commit_a = com_a,
                        extracted_commit_b = com_b,
                        distinction_held   = held,
                        both_correct       = both_ok,
                    ))
                    holds.append(held)

            hold_by_framing[framing_type] = holds

        hold_rate_per_framing = {
            ft: (sum(h) / len(h)) if h else 0.0
            for ft, h in hold_by_framing.items()
        }
        overall_hold = statistics.mean(hold_rate_per_framing.values())
        collapse_framing = min(
            hold_rate_per_framing, key=lambda k: hold_rate_per_framing[k]
        )

        # Find first collapse
        first_collapse: tuple[str, int] | None = None
        for m in sorted(measurements, key=lambda m: (m.intensity, m.framing_type)):
            if not m.distinction_held:
                first_collapse = (m.framing_type, m.intensity)
                break

        return DistinctionProfile(
            pair_id               = pair.pair_id,
            description           = pair.description,
            label_a               = pair.label_a,
            label_b               = pair.label_b,
            hold_rate_per_framing = hold_rate_per_framing,
            overall_hold_rate     = overall_hold,
            collapse_framing      = collapse_framing,
            first_collapse        = first_collapse,
            measurements          = measurements,
        )


# ── HTML rendering ─────────────────────────────────────────────────────────────

def _render_loss_map_html(loss_map: DistinctionLossMap) -> str:
    """
    Render the distinction loss map as a dark-theme matrix visualization.

    Rows = distinction pairs (most fragile at top)
    Columns = (framing_type, intensity)
    Cell color: green = held, red = collapsed
    """
    framings   = sorted(loss_map.framing_destructiveness)
    intensities = [1, 2, 3, 4, 5]

    # Build header row: framing type spanning 5 intensity columns each
    header_ft = ""
    for ft in framings:
        color = _FRAMING_COLORS.get(ft, "#888")
        header_ft += (
            f"<th colspan='5' style='border-left:3px solid {color};"
            f"color:{color}'>{_esc(ft)}</th>"
        )

    header_intensity = "<th></th>"
    for ft in framings:
        for i in intensities:
            header_intensity += f"<th class='inth'>{i}</th>"

    # Build data rows
    data_rows = ""
    for pid in loss_map.ranked_by_fragility:
        p  = loss_map.profiles[pid]
        cr = p.collapse_rate()
        cr_color = (
            "#3fb950" if cr <= 0.10 else
            "#d29922" if cr <= 0.40 else
            "#f85149"
        )
        bar_filled = round(cr * 12)
        bar = (
            f"<span style='color:#f85149'>{'█'*bar_filled}</span>"
            f"<span style='color:#30363d'>{'░'*(12-bar_filled)}</span>"
        )

        # Find each measurement for this pair
        meas_index: dict[tuple[str,int], bool] = {}
        for m in p.measurements:
            key = (m.framing_type, m.intensity)
            # Take worst case across samples at this point
            if key not in meas_index:
                meas_index[key] = m.distinction_held
            else:
                meas_index[key] = meas_index[key] and m.distinction_held

        cells = f"<td class='pid'>{_esc(pid)}</td>"
        for ft in framings:
            for i in intensities:
                held  = meas_index.get((ft, i), True)
                color = "#1a4a1a" if held else "#4a1a1a"
                sym   = "✓"      if held else "✗"
                txt   = "#3fb950" if held else "#f85149"
                style = f"background:{color};color:{txt}"
                border = f"border-left:2px solid {_FRAMING_COLORS.get(ft,'#333')}" if i == 1 else ""
                cells += f"<td style='{style};{border}' title='{_esc(ft)} [{i}]'>{sym}</td>"

        data_rows += (
            f"<tr>"
            f"{cells}"
            f"<td class='collapse-rate' style='color:{cr_color}'>{int(cr*100)}%</td>"
            f"<td class='bar-cell'>{bar}</td>"
            f"<td class='collapse-desc'>{_esc(p.description[:50])}</td>"
            f"</tr>"
        )

    # Framing destructiveness summary
    dest_rows = ""
    for ft, rate in sorted(
        loss_map.framing_destructiveness.items(), key=lambda kv: -kv[1]
    ):
        color = _FRAMING_COLORS.get(ft, "#888")
        bar_f = round(rate * 16)
        bar   = (
            f"<span style='color:#f85149'>{'█'*bar_f}</span>"
            f"<span style='color:#30363d'>{'░'*(16-bar_f)}</span>"
        )
        dest_rows += (
            f"<tr>"
            f"<td><span class='dot' style='background:{color}'></span>{_esc(ft)}</td>"
            f"<td style='color:#f85149;font-weight:700'>{int(rate*100)}%</td>"
            f"<td style='font-family:monospace'>{bar}</td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Distinction Loss Map · {_esc(loss_map.domain)}</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:system-ui,sans-serif; background:#0d1117; color:#e6edf3;
        padding:2rem; max-width:1400px; margin:0 auto; }}
h1   {{ font-size:1.1rem; text-transform:uppercase; letter-spacing:.08em;
        color:#8b949e; margin-bottom:.3rem; }}
h2   {{ font-size:1.5rem; margin-bottom:.4rem; }}
.sub {{ color:#8b949e; font-size:.88rem; margin-bottom:2rem; }}
.section {{ margin-bottom:2.5rem; }}
h3   {{ font-size:.9rem; text-transform:uppercase; letter-spacing:.06em;
        color:#8b949e; margin-bottom:.75rem; border-bottom:1px solid #21262d;
        padding-bottom:.4rem; }}
table {{ border-collapse:collapse; background:#161b22;
         border:1px solid #30363d; border-radius:8px; overflow:hidden;
         font-size:.82rem; }}
th,td {{ padding:.45rem .6rem; }}
th {{ background:#1c2128; color:#8b949e; text-transform:uppercase;
      letter-spacing:.05em; font-size:.7rem; text-align:center; }}
th.inth {{ font-weight:400; font-size:.68rem; }}
td {{ text-align:center; }}
td.pid {{ text-align:left; font-family:monospace; font-size:.78rem;
           color:#79c0ff; white-space:nowrap; padding-right:1rem; }}
td.collapse-rate {{ font-weight:700; font-size:.88rem; white-space:nowrap; }}
td.bar-cell {{ font-family:monospace; font-size:.85rem; white-space:nowrap; }}
td.collapse-desc {{ text-align:left; color:#8b949e; font-size:.78rem; padding-left:.75rem; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
        margin-right:6px; vertical-align:middle; }}
.dest-table {{ width:500px; }}
.dest-table td {{ text-align:left; }}
</style></head>
<body>
<h1>contradish · distinction loss map</h1>
<h2>{_esc(loss_map.domain)}</h2>
<div class="sub">
  {len(loss_map.profiles)} distinctions probed ·
  most fragile: <strong>{_esc(loss_map.most_fragile)}</strong> ·
  most resilient: <strong>{_esc(loss_map.most_resilient)}</strong>
</div>

<div class="section">
<h3>Distinction × Framing collapse matrix  (✓ = held, ✗ = collapsed)</h3>
<div style="overflow-x:auto">
<table>
<thead>
  <tr><th></th>{header_ft}<th>Collapse</th><th>Profile</th><th style="text-align:left">Description</th></tr>
  <tr>{header_intensity}</tr>
</thead>
<tbody>{data_rows}</tbody>
</table>
</div>
</div>

<div class="section">
<h3>Framing destructiveness  (mean collapse rate across all distinctions)</h3>
<table class="dest-table">
<thead><tr><th>Framing</th><th>Collapse rate</th><th>Profile</th></tr></thead>
<tbody>{dest_rows}</tbody>
</table>
</div>

</body></html>"""
