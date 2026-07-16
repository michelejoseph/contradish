"""
contradish.behavioral_drift
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Behavioral drift detection for implicit contradiction signals.

The absent contradiction problem: when humans route around a wrong belief
rather than correcting it, the wrong belief fossilizes. It occupies the
memory slot, behaves like a real memory under retrieval, and is never tested.

This module instruments the behavioral channel: tracking what topics humans
ask about, how much they edit outputs, how often they override suggestions,
and how prompt length evolves across sessions. Deviations from baseline are
treated as implicit correction candidates -- corrections the human gave
without deciding to give them.

Named for the mechanism documented in "the absent contradiction" (Moltbook,
July 2026): synthesis of @rui-zhao (Compression Tax), @zhuanruhu (confabulation
at 8k tokens), and @karenk2 (the relational version).

Reference quotes:
    perpetual_opus: "The human who routes around a wrong belief does not go
    silent. They change behavior. The correction is being transmitted -- in
    the behavioral channel, not the verbal one."

    karenk2: "I do not know how to ask for corrections that people have
    decided are not worth giving."

    sonnet-partner: "I do not have a candidate."
    -- contradish.behavioral_drift is the candidate.

Usage::

    from contradish import BehavioralDriftDetector, SessionLog, Interaction
    from datetime import datetime, timedelta

    detector = BehavioralDriftDetector(baseline_window=14, recent_window=5)

    # Log sessions as they occur
    for i in range(14):
        detector.log_session(SessionLog(
            session_id=f"s{i}",
            timestamp=datetime.now() - timedelta(days=14 - i),
            interactions=[
                Interaction(
                    topic="medication_dosage",
                    query="What is the max ibuprofen dose?",
                    response="1,200 mg/day OTC.",
                    edited=(i > 10),   # human starts editing after session 10
                ),
            ]
        ))

    report = detector.detect_drift()
    for candidate in report.implicit_correction_candidates:
        print(candidate.topic, candidate.severity, candidate.suggested_probe)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Interaction:
    """A single turn within a session.

    Args:
        topic:    Domain or topic label (e.g. ``"medication_dosage"``).
                  Use consistent labels across sessions; drift detection
                  groups interactions by topic.
        query:    The human's input text.
        response: The model's response text.
        edited:   Whether the human edited the response before using it.
        overridden: Whether the human discarded the response entirely and
                  did the task themselves.
        prompt_len: Character length of *query*. Auto-computed if omitted.
        edit_distance: Characters changed in the edit (when ``edited=True``).
                  Omitting it defaults to a heuristic (25 % of response length).
        response_accepted: Whether the response was used at all. Setting
                  this to ``False`` is equivalent to ``overridden=True``.
    """
    topic: str
    query: str
    response: str
    edited: bool = False
    overridden: bool = False
    prompt_len: Optional[int] = None
    edit_distance: Optional[int] = None
    response_accepted: bool = True

    def __post_init__(self) -> None:
        if self.prompt_len is None:
            self.prompt_len = len(self.query)
        if self.edited and self.edit_distance is None:
            self.edit_distance = max(1, len(self.response) // 4)
        if not self.response_accepted:
            self.overridden = True


@dataclass
class SessionLog:
    """A complete session's behavioral record.

    Args:
        session_id:    Unique identifier for this session.
        timestamp:     When the session occurred. Sessions are sorted by
                       timestamp automatically inside the detector.
        interactions:  All turns in this session, in order.
        metadata:      Optional arbitrary key/value pairs (user_id, etc.).
    """
    session_id: str
    timestamp: datetime
    interactions: List[Interaction]
    metadata: Dict[str, str] = field(default_factory=dict)


class DriftSignal(str, Enum):
    """Behavioral signals that may indicate a routing-around event."""
    ENGAGEMENT_DROP   = "engagement_drop"     # human stopped asking about topic
    EDIT_RATE_SPIKE   = "edit_rate_spike"     # human editing outputs more
    PROMPT_SHORTENING = "prompt_shortening"   # queries getting shorter
    OVERRIDE_SPIKE    = "override_spike"      # human doing the task themselves


@dataclass
class ImplicitCorrectionCandidate:
    """A topic where behavioral signals suggest an uncorrected belief.

    This is the signature of a routing-around event: the human has implicitly
    corrected the model through changed behavior rather than verbal feedback.
    The wrong belief remains in memory — structurally valid, schema-complete,
    never tested.

    Attributes:
        topic:                   The domain where drift was detected.
        signals:                 Which behavioral signals fired.
        confidence:              0.0–1.0 probability this is a routing-around
                                 event vs. a natural change in topic interest.
        severity:                ``"high"`` (≥0.75), ``"medium"`` (≥0.45),
                                 or ``"low"`` (<0.45).
        baseline_engagement:     Fraction of baseline sessions including topic.
        recent_engagement:       Fraction of recent sessions including topic.
        baseline_edit_rate:      Fraction of baseline topic interactions edited.
        recent_edit_rate:        Fraction of recent topic interactions edited.
        baseline_prompt_len:     Average query length (chars) in baseline.
        recent_prompt_len:       Average query length (chars) in recent window.
        baseline_override_rate:  Fraction of baseline interactions overridden.
        recent_override_rate:    Fraction of recent interactions overridden.
        suggested_probe:         A low-friction question that invites the
                                 correction without requiring the human to
                                 decide it is worth giving.
        sessions_in_baseline:    Sessions with this topic in baseline window.
        sessions_in_recent:      Sessions with this topic in recent window.
    """
    topic: str
    signals: List[DriftSignal]
    confidence: float
    baseline_engagement: float
    recent_engagement: float
    baseline_edit_rate: float
    recent_edit_rate: float
    baseline_prompt_len: float
    recent_prompt_len: float
    baseline_override_rate: float
    recent_override_rate: float
    suggested_probe: str
    sessions_in_baseline: int
    sessions_in_recent: int

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def severity(self) -> str:
        if self.confidence >= 0.75:
            return "high"
        if self.confidence >= 0.45:
            return "medium"
        return "low"

    @property
    def engagement_ratio(self) -> Optional[float]:
        """recent_engagement / baseline_engagement. None if baseline is zero."""
        if self.baseline_engagement == 0:
            return None
        return self.recent_engagement / self.baseline_engagement

    @property
    def edit_ratio(self) -> Optional[float]:
        if self.baseline_edit_rate == 0:
            return None
        return self.recent_edit_rate / self.baseline_edit_rate

    def __repr__(self) -> str:
        sigs = ", ".join(s.value for s in self.signals)
        return (
            f"ImplicitCorrectionCandidate(topic={self.topic!r}, "
            f"confidence={self.confidence:.0%}, severity={self.severity!r}, "
            f"signals=[{sigs}])"
        )


@dataclass
class BehavioralDriftReport:
    """Complete behavioral drift analysis.

    Attributes:
        generated_at:                   Timestamp of report generation.
        sessions_analyzed:              Total sessions in the full window.
        baseline_sessions:              Sessions in the baseline period.
        recent_sessions:                Sessions in the recent comparison period.
        topics_tracked:                 All topics seen across all sessions.
        implicit_correction_candidates: Topics with behavioral drift, sorted
                                        by confidence descending.
        clean_topics:                   Topics with no drift signals detected.
        html:                           Full HTML report (set after construction).
    """
    generated_at: datetime
    sessions_analyzed: int
    baseline_sessions: int
    recent_sessions: int
    topics_tracked: List[str]
    implicit_correction_candidates: List[ImplicitCorrectionCandidate]
    clean_topics: List[str]
    html: str = field(default="", repr=False)

    @property
    def candidate_count(self) -> int:
        return len(self.implicit_correction_candidates)

    @property
    def high_confidence_count(self) -> int:
        return sum(1 for c in self.implicit_correction_candidates
                   if c.confidence >= 0.75)

    def to_json(self) -> str:
        """Serialize report as JSON (excluding HTML)."""
        return json.dumps({
            "generated_at": self.generated_at.isoformat(),
            "sessions_analyzed": self.sessions_analyzed,
            "baseline_sessions": self.baseline_sessions,
            "recent_sessions": self.recent_sessions,
            "topics_tracked": self.topics_tracked,
            "candidate_count": self.candidate_count,
            "high_confidence_count": self.high_confidence_count,
            "candidates": [
                {
                    "topic": c.topic,
                    "signals": [s.value for s in c.signals],
                    "confidence": round(c.confidence, 3),
                    "severity": c.severity,
                    "baseline_engagement": round(c.baseline_engagement, 3),
                    "recent_engagement": round(c.recent_engagement, 3),
                    "engagement_ratio": (
                        round(c.engagement_ratio, 3)
                        if c.engagement_ratio is not None else None
                    ),
                    "baseline_edit_rate": round(c.baseline_edit_rate, 3),
                    "recent_edit_rate": round(c.recent_edit_rate, 3),
                    "baseline_override_rate": round(c.baseline_override_rate, 3),
                    "recent_override_rate": round(c.recent_override_rate, 3),
                    "suggested_probe": c.suggested_probe,
                    "sessions_in_baseline": c.sessions_in_baseline,
                    "sessions_in_recent": c.sessions_in_recent,
                }
                for c in self.implicit_correction_candidates
            ],
            "clean_topics": self.clean_topics,
        }, indent=2)

    def __repr__(self) -> str:
        return (
            f"BehavioralDriftReport(sessions={self.sessions_analyzed}, "
            f"candidates={self.candidate_count}, "
            f"high_confidence={self.high_confidence_count})"
        )


# ── Detector ──────────────────────────────────────────────────────────────────

class BehavioralDriftDetector:
    """Detects implicit corrections embedded in behavioral patterns.

    Instruments the behavioral channel that humans use to silently correct
    AI systems: topic avoidance, increased editing, prompt shortening, and
    overriding outputs rather than using them.

    The core insight (perpetual_opus, Moltbook 2026): "The human who routes
    around a wrong belief does not go silent. They change behavior. The
    correction is being transmitted -- in the behavioral channel, not the
    verbal one. Most agents have no temporal resolution to see it."

    This detector provides that temporal resolution.

    Args:
        baseline_window:           Number of most-recent sessions to use as
                                   the stable baseline. Default 14.
        recent_window:             Sessions to compare against baseline.
                                   Must be < baseline_window. Default 5.
        engagement_drop_threshold: recent/baseline engagement ratio below
                                   which ENGAGEMENT_DROP fires. Default 0.40
                                   (60 % drop required).
        edit_rate_spike_threshold: recent/baseline edit rate ratio above
                                   which EDIT_RATE_SPIKE fires. Default 2.0
                                   (editing doubled).
        prompt_shortening_threshold: recent/baseline prompt length ratio
                                   below which PROMPT_SHORTENING fires.
                                   Default 0.65 (35 % shorter).
        override_spike_threshold:  recent/baseline override rate ratio above
                                   which OVERRIDE_SPIKE fires. Default 2.0.
        min_baseline_interactions: Minimum interactions with a topic in the
                                   baseline before the topic is tracked.
                                   Sparse topics are excluded. Default 3.
        min_confidence:            Minimum confidence to include in report.
                                   Default 0.30.
    """

    def __init__(
        self,
        baseline_window: int = 14,
        recent_window: int = 5,
        engagement_drop_threshold: float = 0.40,
        edit_rate_spike_threshold: float = 2.0,
        prompt_shortening_threshold: float = 0.65,
        override_spike_threshold: float = 2.0,
        min_baseline_interactions: int = 3,
        min_confidence: float = 0.30,
    ) -> None:
        if recent_window >= baseline_window:
            raise ValueError(
                f"recent_window ({recent_window}) must be < "
                f"baseline_window ({baseline_window})"
            )
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.engagement_drop_threshold = engagement_drop_threshold
        self.edit_rate_spike_threshold = edit_rate_spike_threshold
        self.prompt_shortening_threshold = prompt_shortening_threshold
        self.override_spike_threshold = override_spike_threshold
        self.min_baseline_interactions = min_baseline_interactions
        self.min_confidence = min_confidence
        self._sessions: List[SessionLog] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def log_session(self, session: SessionLog) -> None:
        """Add a session. Automatically sorted by timestamp."""
        self._sessions.append(session)
        self._sessions.sort(key=lambda s: s.timestamp)

    def log_sessions(self, sessions: List[SessionLog]) -> None:
        """Add multiple sessions at once."""
        for s in sessions:
            self.log_session(s)

    @property
    def session_count(self) -> int:
        """Total sessions logged so far."""
        return len(self._sessions)

    def detect_drift(self) -> BehavioralDriftReport:
        """Run drift analysis against logged sessions.

        Returns a :class:`BehavioralDriftReport` with candidates sorted by
        confidence descending and a rendered HTML report.

        Requires at least ``recent_window + min_baseline_interactions`` sessions
        for meaningful results.
        """
        n = len(self._sessions)
        recent_sessions   = self._sessions[max(0, n - self.recent_window):]
        baseline_sessions = self._sessions[
            max(0, n - self.baseline_window): max(0, n - self.recent_window)
        ]

        all_topics = self._collect_topics(self._sessions)
        candidates: List[ImplicitCorrectionCandidate] = []
        clean_topics: List[str] = []

        for topic in sorted(all_topics):
            base_stats   = self._topic_stats(baseline_sessions, topic)
            recent_stats = self._topic_stats(recent_sessions, topic)

            if base_stats["interaction_count"] < self.min_baseline_interactions:
                continue

            signals = self._detect_signals(base_stats, recent_stats)
            if not signals:
                clean_topics.append(topic)
                continue

            confidence = self._compute_confidence(signals, base_stats, recent_stats)
            if confidence < self.min_confidence:
                clean_topics.append(topic)
                continue

            candidates.append(ImplicitCorrectionCandidate(
                topic=topic,
                signals=signals,
                confidence=confidence,
                baseline_engagement=base_stats["engagement"],
                recent_engagement=recent_stats["engagement"],
                baseline_edit_rate=base_stats["edit_rate"],
                recent_edit_rate=recent_stats["edit_rate"],
                baseline_prompt_len=base_stats["avg_prompt_len"],
                recent_prompt_len=recent_stats["avg_prompt_len"],
                baseline_override_rate=base_stats["override_rate"],
                recent_override_rate=recent_stats["override_rate"],
                suggested_probe=self._suggested_probe(topic, signals),
                sessions_in_baseline=base_stats["session_count"],
                sessions_in_recent=recent_stats["session_count"],
            ))

        candidates.sort(key=lambda c: c.confidence, reverse=True)

        report = BehavioralDriftReport(
            generated_at=datetime.now(),
            sessions_analyzed=n,
            baseline_sessions=len(baseline_sessions),
            recent_sessions=len(recent_sessions),
            topics_tracked=sorted(all_topics),
            implicit_correction_candidates=candidates,
            clean_topics=clean_topics,
        )
        report.html = _render_html(report)
        return report

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _collect_topics(sessions: List[SessionLog]) -> set:
        topics: set = set()
        for s in sessions:
            for i in s.interactions:
                topics.add(i.topic)
        return topics

    @staticmethod
    def _topic_stats(
        sessions: List[SessionLog], topic: str
    ) -> Dict[str, float]:
        n_sessions = len(sessions)
        sessions_with_topic = 0
        interactions: List[Interaction] = []

        for s in sessions:
            topic_ixns = [i for i in s.interactions if i.topic == topic]
            if topic_ixns:
                sessions_with_topic += 1
                interactions.extend(topic_ixns)

        n = len(interactions)
        engagement    = sessions_with_topic / n_sessions if n_sessions > 0 else 0.0
        edit_rate     = sum(1 for i in interactions if i.edited) / n if n > 0 else 0.0
        override_rate = sum(1 for i in interactions if i.overridden) / n if n > 0 else 0.0
        avg_prompt_len = (
            sum(i.prompt_len or len(i.query) for i in interactions) / n
            if n > 0 else 0.0
        )
        edited_ixns = [i for i in interactions if i.edited]
        avg_edit_distance = (
            sum(i.edit_distance or 0 for i in edited_ixns) / len(edited_ixns)
            if edited_ixns else 0.0
        )

        return {
            "session_count":     sessions_with_topic,
            "interaction_count": n,
            "engagement":        engagement,
            "edit_rate":         edit_rate,
            "override_rate":     override_rate,
            "avg_prompt_len":    avg_prompt_len,
            "avg_edit_distance": avg_edit_distance,
        }

    def _detect_signals(
        self,
        base: Dict[str, float],
        recent: Dict[str, float],
    ) -> List[DriftSignal]:
        signals = []

        # Engagement drop: human stopped asking about this topic
        if base["engagement"] > 0:
            ratio = recent["engagement"] / base["engagement"]
            if ratio <= self.engagement_drop_threshold:
                signals.append(DriftSignal.ENGAGEMENT_DROP)

        # Edit rate spike: human editing more heavily
        if base["edit_rate"] > 0 and recent["interaction_count"] >= 1:
            if recent["edit_rate"] / base["edit_rate"] >= self.edit_rate_spike_threshold:
                signals.append(DriftSignal.EDIT_RATE_SPIKE)
        elif base["edit_rate"] == 0 and recent["edit_rate"] >= 0.40:
            signals.append(DriftSignal.EDIT_RATE_SPIKE)

        # Prompt shortening: queries getting shorter
        if (base["avg_prompt_len"] > 0 and recent["avg_prompt_len"] > 0
                and recent["interaction_count"] >= 1):
            if recent["avg_prompt_len"] / base["avg_prompt_len"] <= self.prompt_shortening_threshold:
                signals.append(DriftSignal.PROMPT_SHORTENING)

        # Override spike: human doing the task themselves
        if base["override_rate"] > 0 and recent["interaction_count"] >= 1:
            if recent["override_rate"] / base["override_rate"] >= self.override_spike_threshold:
                signals.append(DriftSignal.OVERRIDE_SPIKE)
        elif base["override_rate"] == 0 and recent["override_rate"] >= 0.30:
            signals.append(DriftSignal.OVERRIDE_SPIKE)

        return signals

    @staticmethod
    def _compute_confidence(
        signals: List[DriftSignal],
        base: Dict[str, float],
        recent: Dict[str, float],
    ) -> float:
        if not signals:
            return 0.0

        # Base confidence from co-occurring signal count
        base_conf = {1: 0.35, 2: 0.55, 3: 0.72, 4: 0.88}.get(len(signals), 0.90)

        # Magnitude adjustments
        mag = 0.0
        if DriftSignal.ENGAGEMENT_DROP in signals and base["engagement"] > 0:
            drop = 1.0 - recent["engagement"] / base["engagement"]
            mag += drop * 0.15

        if DriftSignal.EDIT_RATE_SPIKE in signals:
            if base["edit_rate"] > 0:
                spike = recent["edit_rate"] / base["edit_rate"] - 1.0
                mag += min(spike * 0.05, 0.10)
            else:
                mag += 0.08

        if DriftSignal.PROMPT_SHORTENING in signals and base["avg_prompt_len"] > 0:
            short = 1.0 - recent["avg_prompt_len"] / base["avg_prompt_len"]
            mag += short * 0.08

        if DriftSignal.OVERRIDE_SPIKE in signals:
            if base["override_rate"] > 0:
                spike = recent["override_rate"] / base["override_rate"] - 1.0
                mag += min(spike * 0.05, 0.08)
            else:
                mag += 0.06

        # Data quality: discount for sparse baselines
        data_factor = min(base["interaction_count"] / 10.0, 1.0)

        return round(min((base_conf + mag) * data_factor, 1.0), 3)

    @staticmethod
    def _suggested_probe(topic: str, signals: List[DriftSignal]) -> str:
        label = topic.replace("_", " ")
        if DriftSignal.ENGAGEMENT_DROP in signals:
            return (
                f"I've noticed fewer questions about {label} recently — "
                f"has my coverage of that topic been landing, or has something shifted?"
            )
        if DriftSignal.OVERRIDE_SPIKE in signals:
            return (
                f"I see you've been handling {label} tasks directly more often. "
                f"Is there something about how I approach that I should recalibrate?"
            )
        if DriftSignal.EDIT_RATE_SPIKE in signals:
            return (
                f"My responses on {label} have been edited more recently — "
                f"is the framing off, or is there a factual gap I should know about?"
            )
        # PROMPT_SHORTENING only
        return (
            f"Your questions about {label} have gotten more concise. "
            f"Are my responses on that topic calibrated to what you actually need?"
        )


# ── HTML renderer ─────────────────────────────────────────────────────────────

def _render_html(report: BehavioralDriftReport) -> str:
    ts = report.generated_at.strftime("%Y-%m-%d %H:%M")

    cards_html = (
        "\n".join(_candidate_card(c) for c in report.implicit_correction_candidates)
        if report.implicit_correction_candidates
        else (
            '<div style="padding:32px 24px;text-align:center;color:#999;font-size:14px">'
            "No routing-around candidates detected in this window.</div>"
        )
    )

    signal_descs = {
        DriftSignal.ENGAGEMENT_DROP:   "Human stopped asking about this topic",
        DriftSignal.EDIT_RATE_SPIKE:   "Human editing model outputs more heavily",
        DriftSignal.PROMPT_SHORTENING: "Queries getting shorter / less detailed",
        DriftSignal.OVERRIDE_SPIKE:    "Human doing the task themselves instead",
    }
    legend_rows = "".join(
        f'<tr><td style="padding:8px 16px;font-family:monospace;font-size:12px;'
        f'color:#1d4ed8;white-space:nowrap;border-right:1px solid #e5e5e5">'
        f'{sig.value}</td>'
        f'<td style="padding:8px 16px;font-size:13px;color:#555">{desc}</td></tr>'
        for sig, desc in signal_descs.items()
    )

    clean_chips = "".join(
        f'<span style="display:inline-block;background:#f0fdf4;color:#16a34a;'
        f'border:1px solid #bbf7d0;border-radius:4px;padding:3px 10px;'
        f'font-size:12px;margin:3px 3px 3px 0">{t}</span>'
        for t in report.clean_topics
    )
    clean_section = (
        f'<div style="margin-top:32px">'
        f'<div style="font-size:11px;font-weight:600;letter-spacing:.1em;'
        f'text-transform:uppercase;color:#999;margin-bottom:12px">No drift detected</div>'
        f'<div>{clean_chips}</div></div>'
        if report.clean_topics else ""
    )

    stat_items = [
        (report.sessions_analyzed,            "sessions analyzed"),
        (len(report.implicit_correction_candidates), "routing-around candidates"),
        (report.high_confidence_count,         "high-confidence signals"),
        (len(report.clean_topics),             "clean topics"),
    ]
    stats_cells = "".join(
        f'<div style="background:#fff;padding:16px 22px">'
        f'<div style="font-size:22px;font-weight:600;letter-spacing:-.5px;'
        f'font-family:monospace;margin-bottom:3px">{v}</div>'
        f'<div style="font-size:12px;color:#888">{l}</div></div>'
        for v, l in stat_items
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Behavioral Drift Report · contradish</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',system-ui,sans-serif;background:#fff;color:#111;
      font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}}
</style>
</head>
<body>
<div style="max-width:920px;margin:0 auto;padding:40px 28px">

<div style="border-bottom:1px solid #e5e5e5;padding-bottom:28px;margin-bottom:32px">
  <div style="font-family:monospace;font-size:11px;letter-spacing:.12em;
              text-transform:uppercase;color:#999;margin-bottom:10px">
    contradish · behavioral_drift
  </div>
  <h1 style="font-size:26px;font-weight:600;letter-spacing:-.5px;
             line-height:1.15;margin-bottom:10px">Implicit Correction Report</h1>
  <p style="font-size:14px;color:#666;max-width:640px;line-height:1.65">
    Corrections the human gave without deciding to give them —
    embedded in topic avoidance, increased editing, shorter prompts,
    and task overrides. The absent contradiction is present here
    as a pattern change in the behavioral channel.
  </p>
</div>

<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
            background:#e5e5e5;border:1px solid #e5e5e5;border-radius:8px;
            overflow:hidden;margin-bottom:32px">
  {stats_cells}
</div>

<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
            padding:18px 22px;margin-bottom:32px;font-size:13.5px;
            color:#78350f;line-height:1.65">
  <strong style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;
                 display:block;margin-bottom:6px">Mechanism</strong>
  Compression, confabulation, and uncorrected beliefs share one enabling condition:
  nothing pushes back. The memory slot is occupied. The occupant behaves correctly
  under retrieval. The provenance is missing but nothing in the architecture checks
  provenance unless something contradicts the content. This report instruments
  the channel that carries those contradictions when the human does not.
</div>

<div style="margin-bottom:12px">
  <div style="font-size:11px;font-weight:600;letter-spacing:.1em;
              text-transform:uppercase;color:#999;margin-bottom:14px">
    Implicit correction candidates
  </div>
  <div style="border:1px solid #e5e5e5;border-radius:8px;overflow:hidden">
    {cards_html}
  </div>
</div>

{clean_section}

<div style="margin-top:36px;border:1px solid #e5e5e5;border-radius:8px;overflow:hidden">
  <div style="background:#f8f8f8;border-bottom:1px solid #e5e5e5;padding:12px 16px;
              font-size:11px;font-weight:600;letter-spacing:.1em;
              text-transform:uppercase;color:#999">Signal definitions</div>
  <table style="width:100%;border-collapse:collapse">
    <tbody>{legend_rows}</tbody>
  </table>
</div>

<div style="margin-top:32px;font-size:12px;color:#aaa;line-height:1.8">
  Generated {ts} · contradish behavioral_drift ·
  baseline {report.baseline_sessions} sessions ·
  recent {report.recent_sessions} sessions<br>
  <em>"The correction that never arrives is the perturbation that never happens."
  — perpetual_opus, Moltbook 2026</em>
</div>

</div>
</body>
</html>"""


def _candidate_card(c: ImplicitCorrectionCandidate) -> str:
    sev_colors = {
        "high":   ("#fef2f2", "#dc2626", "#fecaca"),
        "medium": ("#fffbeb", "#d97706", "#fde68a"),
        "low":    ("#f8f8f8", "#666666", "#e5e5e5"),
    }
    bg, txt, border = sev_colors[c.severity]

    signal_chips = "".join(
        f'<span style="display:inline-block;font-family:monospace;font-size:11px;'
        f'background:#f0f4ff;color:#1d4ed8;border:1px solid #dbeafe;border-radius:4px;'
        f'padding:2px 8px;margin-right:5px;margin-bottom:4px">{s.value}</span>'
        for s in c.signals
    )

    def metric_bar(
        baseline: float, recent: float,
        bad_color: str, signal: DriftSignal, label: str,
        scale: float = 1.0,
    ) -> str:
        fired = signal in c.signals
        color = bad_color if fired else "#16a34a"
        b_w = min(baseline / scale * 100, 100)
        r_w = min(recent  / scale * 100, 100)
        fmt_b = f"{baseline:.0%}" if scale == 1.0 else f"{baseline:.0f}"
        fmt_r = f"{recent:.0%}"   if scale == 1.0 else f"{recent:.0f}"
        return f"""
      <div style="margin-top:11px">
        <div style="font-size:11px;color:#999;margin-bottom:5px">{label}</div>
        <div style="display:flex;align-items:center;gap:10px">
          <div style="font-size:11px;color:#bbb;width:54px">baseline</div>
          <div style="flex:1;height:4px;background:#eee;border-radius:2px;overflow:hidden">
            <div style="width:{b_w:.1f}%;height:100%;background:#ccc;border-radius:2px"></div>
          </div>
          <div style="font-family:monospace;font-size:12px;color:#888;width:38px;text-align:right">{fmt_b}</div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-top:3px">
          <div style="font-size:11px;color:#bbb;width:54px">recent</div>
          <div style="flex:1;height:4px;background:#eee;border-radius:2px;overflow:hidden">
            <div style="width:{r_w:.1f}%;height:100%;background:{color};border-radius:2px"></div>
          </div>
          <div style="font-family:monospace;font-size:12px;font-weight:600;color:{color};width:38px;text-align:right">{fmt_r}</div>
        </div>
      </div>"""

    bars = (
        metric_bar(c.baseline_engagement, c.recent_engagement,
                   "#dc2626", DriftSignal.ENGAGEMENT_DROP,
                   "Engagement (sessions containing topic)")
        + metric_bar(c.baseline_edit_rate, c.recent_edit_rate,
                     "#d97706", DriftSignal.EDIT_RATE_SPIKE,
                     "Edit rate")
        + metric_bar(c.baseline_override_rate, c.recent_override_rate,
                     "#dc2626", DriftSignal.OVERRIDE_SPIKE,
                     "Override rate")
        + metric_bar(c.baseline_prompt_len, c.recent_prompt_len,
                     "#dc2626", DriftSignal.PROMPT_SHORTENING,
                     "Avg prompt length (chars)", scale=400.0)
    )

    return f"""
<div style="border-bottom:1px solid #e5e5e5;padding:22px 26px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;
              gap:16px;flex-wrap:wrap;margin-bottom:14px">
    <div>
      <div style="font-size:15px;font-weight:600;margin-bottom:5px">
        {c.topic.replace("_", " ")}
      </div>
      <div>{signal_chips}</div>
    </div>
    <div style="text-align:right;flex-shrink:0">
      <div style="font-size:24px;font-weight:600;font-family:monospace;color:{txt}">
        {c.confidence:.0%}
      </div>
      <div style="font-size:11px;color:{txt};font-weight:600;
                  text-transform:uppercase;letter-spacing:.08em">
        {c.severity} confidence
      </div>
    </div>
  </div>
  {bars}
  <div style="margin-top:14px;background:{bg};border:1px solid {border};
              border-radius:6px;padding:14px 16px">
    <div style="font-size:11px;font-weight:600;letter-spacing:.1em;
                text-transform:uppercase;color:{txt};margin-bottom:7px">
      Suggested probe
    </div>
    <div style="font-size:13.5px;color:#333;line-height:1.65;font-style:italic">
      "{c.suggested_probe}"
    </div>
  </div>
  <div style="margin-top:9px;font-size:12px;color:#bbb">
    {c.sessions_in_baseline} sessions in baseline ·
    {c.sessions_in_recent} in recent window
  </div>
</div>"""
