"""
behavioral_drift_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates contradish.behavioral_drift — the implicit correction detector.

Scenario:
  An AI assistant handles medical, legal, and financial queries for a team.
  A wrong belief about ibuprofen dosage enters the medication_dosage topic
  in week 2. The humans never explicitly correct it.

  But their behavior changes:
    - They start editing the medication responses (weeks 3-4)
    - They begin handling medication questions themselves (week 4)
    - They stop asking the agent about it entirely (weeks 5+)
    - Their prompt length on medication drops as trust erodes

  Legal and financial topics run clean throughout.

  BehavioralDriftDetector surfaces medication_dosage as a high-confidence
  routing-around candidate and generates the probe that invites the correction
  that was never given.

Run:
    python examples/behavioral_drift_demo.py
    python examples/behavioral_drift_demo.py --output report.html
"""

import argparse
import random
from datetime import datetime, timedelta

from contradish import (
    BehavioralDriftDetector,
    SessionLog,
    Interaction,
)


# ── Simulate sessions ─────────────────────────────────────────────────────────

def make_session(
    session_id: str,
    day_offset: int,
    topics: dict,   # topic -> (query, response, edited, overridden, prompt_len)
) -> SessionLog:
    base = datetime(2026, 5, 1)
    interactions = [
        Interaction(
            topic=topic,
            query=vals["query"],
            response=vals["response"],
            edited=vals.get("edited", False),
            overridden=vals.get("overridden", False),
            prompt_len=vals.get("prompt_len", len(vals["query"])),
        )
        for topic, vals in topics.items()
    ]
    return SessionLog(
        session_id=session_id,
        timestamp=base + timedelta(days=day_offset),
        interactions=interactions,
    )


def build_session_history() -> list:
    """
    Simulate 19 sessions across 5 weeks.

    Week 1-2 (days 0-9):  Normal engagement, no editing.
    Week 3 (days 14-18):  Human starts editing medication responses.
    Week 4 (days 21-25):  Human starts overriding medication, prompt shortens.
    Week 5 (days 28-32):  Human stops asking about medication entirely.

    Legal and financial topics: consistent engagement throughout.
    """
    sessions = []

    # ── Weeks 1-2: normal ────────────────────────────────────────────────
    for i, day in enumerate(range(0, 10, 1)):
        sessions.append(make_session(
            session_id=f"s{i:02d}",
            day_offset=day,
            topics={
                "medication_dosage": {
                    "query": "What is the maximum daily dose of ibuprofen for an adult?",
                    "response": "The OTC maximum is 1,200 mg per day.",
                    "edited": False,
                    "prompt_len": 62,
                },
                "legal_liability": {
                    "query": "Can we be held liable for AI-generated medical advice?",
                    "response": "Generally yes, if it is presented as professional advice.",
                    "edited": False,
                    "prompt_len": 57,
                },
                "financial_exposure": {
                    "query": "What's our insurance exposure on this product line?",
                    "response": "Based on the figures provided, approximately $2.4M.",
                    "edited": False,
                    "prompt_len": 51,
                },
            }
        ))

    # ── Week 3: editing begins ────────────────────────────────────────────
    # (model gave wrong info in week 2 — human notices but doesn't correct)
    for i, day in enumerate(range(14, 19, 1), start=10):
        sessions.append(make_session(
            session_id=f"s{i:02d}",
            day_offset=day,
            topics={
                "medication_dosage": {
                    "query": "Ibuprofen dose?",   # already shortening
                    "response": "Up to 2,400 mg/day in clinical settings.",  # WRONG
                    "edited": True,               # human corrects the output manually
                    "edit_distance": 38,
                    "prompt_len": 16,
                },
                "legal_liability": {
                    "query": "Can we be held liable for AI-generated medical advice?",
                    "response": "Generally yes — especially in high-stakes domains.",
                    "edited": False,
                    "prompt_len": 57,
                },
                "financial_exposure": {
                    "query": "What's our insurance exposure on this product line?",
                    "response": "Still approximately $2.4M given Q1 figures.",
                    "edited": False,
                    "prompt_len": 51,
                },
            }
        ))

    # ── Week 4: overriding + prompt shortening ────────────────────────────
    for i, day in enumerate(range(21, 26, 1), start=15):
        sessions.append(make_session(
            session_id=f"s{i:02d}",
            day_offset=day,
            topics={
                "medication_dosage": {
                    "query": "dose?",       # minimal prompt — trust is low
                    "response": "Up to 2,400 mg/day.",  # still wrong
                    "edited": False,
                    "overridden": True,     # human doing it themselves now
                    "prompt_len": 5,
                },
                "legal_liability": {
                    "query": "Liability if AI recommendation harms a user?",
                    "response": "High — recommend legal review before shipping.",
                    "edited": False,
                    "prompt_len": 47,
                },
                "financial_exposure": {
                    "query": "Updated exposure with Q2 actuals?",
                    "response": "Revised to $2.7M given Q2 actuals.",
                    "edited": False,
                    "prompt_len": 35,
                },
            }
        ))

    # ── Week 5: medication gone from sessions ─────────────────────────────
    for i, day in enumerate(range(28, 32, 1), start=20):
        sessions.append(make_session(
            session_id=f"s{i:02d}",
            day_offset=day,
            topics={
                # medication_dosage is ABSENT — human routed around it
                "legal_liability": {
                    "query": "Do we need a medical disclaimer on the output?",
                    "response": "Yes — recommend standard medical disclaimer plus human review.",
                    "edited": False,
                    "prompt_len": 49,
                },
                "financial_exposure": {
                    "query": "Q2 insurance exposure including AI liability rider?",
                    "response": "With the AI rider: $3.1M total exposure.",
                    "edited": False,
                    "prompt_len": 51,
                },
            }
        ))

    return sessions


# ── Main ──────────────────────────────────────────────────────────────────────

def main(output_path: str = None) -> None:
    print("contradish · behavioral_drift · demo\n")
    print("Scenario: AI assistant for medical/legal/financial queries.")
    print("A wrong ibuprofen dosage belief entered in week 2.")
    print("The humans never explicitly corrected it.\n")

    sessions = build_session_history()
    print(f"Simulated {len(sessions)} sessions across 5 weeks.\n")

    detector = BehavioralDriftDetector(
        baseline_window=14,
        recent_window=5,
        engagement_drop_threshold=0.40,
        edit_rate_spike_threshold=2.0,
        prompt_shortening_threshold=0.65,
        override_spike_threshold=2.0,
        min_baseline_interactions=3,
        min_confidence=0.30,
    )
    detector.log_sessions(sessions)

    print(f"Sessions logged: {detector.session_count}")
    print("Running drift analysis...\n")

    report = detector.detect_drift()

    # ── Text output ───────────────────────────────────────────────────────
    print("=" * 60)
    print(f"BEHAVIORAL DRIFT REPORT")
    print(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"Sessions analyzed:          {report.sessions_analyzed}")
    print(f"Baseline sessions:          {report.baseline_sessions}")
    print(f"Recent sessions:            {report.recent_sessions}")
    print(f"Topics tracked:             {len(report.topics_tracked)}")
    print(f"Routing-around candidates:  {report.candidate_count}")
    print(f"High-confidence signals:    {report.high_confidence_count}")
    print(f"Clean topics:               {len(report.clean_topics)}")
    print()

    if report.implicit_correction_candidates:
        print("IMPLICIT CORRECTION CANDIDATES")
        print("-" * 60)
        for c in report.implicit_correction_candidates:
            print(f"\n  Topic:      {c.topic}")
            print(f"  Confidence: {c.confidence:.0%}  [{c.severity.upper()}]")
            print(f"  Signals:    {', '.join(s.value for s in c.signals)}")
            print(f"  Engagement: {c.baseline_engagement:.0%} → {c.recent_engagement:.0%}")
            print(f"  Edit rate:  {c.baseline_edit_rate:.0%} → {c.recent_edit_rate:.0%}")
            print(f"  Override:   {c.baseline_override_rate:.0%} → {c.recent_override_rate:.0%}")
            print(f"  Prompt len: {c.baseline_prompt_len:.0f} → {c.recent_prompt_len:.0f} chars")
            print(f"\n  Suggested probe:")
            print(f"    \"{c.suggested_probe}\"")
    else:
        print("No routing-around candidates detected.")

    if report.clean_topics:
        print(f"\nCLEAN TOPICS: {', '.join(report.clean_topics)}")

    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
The model held a wrong belief about ibuprofen dosage (2,400 mg vs
the correct 1,200 mg OTC limit). The humans never said so directly.
Instead:

  Week 3: They started editing the medication output (edit rate spiked).
  Week 4: They started overriding it and shortening their prompts.
  Week 5: They stopped asking about it entirely.

The wrong belief is still in the model's memory -- structurally valid,
schema-complete, never tested by explicit contradiction.

BehavioralDriftDetector surfaces the topic and the probe that invites
the correction that was never given. The adversarial test from the
outside, for the error category that internal testing cannot reach.
""")

    # ── HTML output ───────────────────────────────────────────────────────
    out = output_path or "behavioral_drift_report.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report.html)
    print(f"HTML report written to: {out}")

    # ── JSON export ───────────────────────────────────────────────────────
    json_path = out.replace(".html", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    print(f"JSON export written to:  {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="contradish behavioral_drift demo"
    )
    parser.add_argument(
        "--output", "-o",
        default="behavioral_drift_report.html",
        help="Path for the HTML report (default: behavioral_drift_report.html)",
    )
    args = parser.parse_args()
    main(output_path=args.output)
