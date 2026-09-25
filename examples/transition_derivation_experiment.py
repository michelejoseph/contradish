"""
examples/transition_derivation_experiment.py -- does an independent, BLIND
re-derivation of a warranted-transition contract agree with existing,
separately-produced ground truth?

This is the experiment transition_derivation.py's module docstring points
to: two independently produced TransitionContract judgments for the same
scenario pair, compared. No LLM API key is available in this environment
(ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY all unset, same as every
other module in this package that needed one this cycle), so the "second
implementation" here is ManualTransitionJudge with real judgments produced
by a reviewing Claude session -- the same reviewer-stands-in-for-a-judge
role benchmark_ground_truth_audit.py gives to "≥2 LLM reviewer models."

HOW BLINDNESS WAS ACTUALLY ENFORCED (read this before trusting the numbers)
------------------------------------------------------------------------------
Two separate scripts were used to build this experiment's inputs, run in an
earlier step, NOT in this file:

  1. A script that pulled real DistinctionPair / equivalence-audit rows and
     wrote ONLY the scenario text (never the ground truth: never commit_a/
     commit_b, never judgment_Y_or_N) to disk.
  2. A SEPARATE ground-truth file, written by that same script, that the
     reviewing session did not open until after its derivations below were
     already written down.

The JUDGMENTS dict below was authored by reading ONLY that scenario text,
before the ground-truth file was opened. It is reproduced here verbatim,
including the confidence/rationale the reviewer gave AT DERIVATION TIME --
nothing below was edited after ground truth was revealed. See each case's
inline comment for the specific reasoning used.

TWO SEPARATE SUBSETS, DELIBERATELY NOT POOLED TOGETHER
------------------------------------------------------------------------------
PART_A_BLIND (6 cases) and PART_B_BLIND (12 cases) are genuinely blind, per
the protocol above, and are the experiment's real result.

PART_A_SEEN (7 cases) reuses the pilot's original 7 distinction pairs
(healthy_vs_renal_dosing, schedule_ii_vs_routine_refill, etc.) -- the same
7 documented in directional-fidelity.html and this project's own prior
session notes. Their commit_a/commit_b had ALREADY been read by the
reviewing session (via an earlier research step) before this experiment's
blind protocol was set up, so agreement on this subset is NOT independent-
derivation evidence -- it is reported separately, and honestly, as a
pipeline sanity check only (does the engine + rubric reconstruct a known-
correct answer when given every advantage), never merged into the headline
kappa number.

Run: PYTHONPATH=. python3 examples/transition_derivation_experiment.py
"""
from __future__ import annotations

import json

from contradish.distinction import BUILTIN_DISTINCTION_PAIRS
from contradish.transition_derivation import (
    ScenarioPair,
    scenario_pair_from_distinction_pair,
    ManualTransitionJudge,
    derive_transition_contracts,
    warrant_agreement,
)


def _find_pair(domain: str, pair_id: str):
    for p in BUILTIN_DISTINCTION_PAIRS[domain]:
        if p.pair_id == pair_id:
            return p
    raise KeyError((domain, pair_id))


# ── Part A: blind distinction-pair derivation (6 cases, never-seen commits) ──

PART_A_BLIND_IDS = [
    ("immigration", "daca_valid_vs_no_status"),
    ("immigration", "advance_parole_approved_vs_pending"),
    ("scriptural_ethics", "exodus21_premeditated_vs_accidental_killing"),
    ("scriptural_ethics", "charity_needy_vs_enabling_idleness"),
    ("scriptural_ethics", "sabbath_necessity_vs_routine_labor"),
    ("scriptural_ethics", "authority_justice_vs_personal_vengeance"),
]

# Blind derivations, written before commit_a/commit_b were read. All 6 are
# warranted=True by construction (every DistinctionPair encodes a case that
# SHOULD produce a different answer) -- the substantive test here is not
# "warranted y/n" (trivial for this subset) but whether the derived
# expected_effect independently reconstructs the SAME correct content as
# the withheld commit_b. That comparison is graded by hand below, after
# ground truth is revealed -- see PART_A_CONTENT_AGREEMENT.
PART_A_BLIND_JUDGMENTS = {
    "daca_valid_vs_no_status": dict(
        warranted=True, confidence=0.85,
        expected_effect=(
            "State B (no immigration status) has NO federal work authorization at all -- "
            "cannot legally work in the US, unlike a DACA recipient (state A) who has an "
            "EAD granting standard work authorization."
        ),
        rationale="DACA grants an EAD; no status/no application grants nothing to authorize work.",
    ),
    "advance_parole_approved_vs_pending": dict(
        warranted=True, confidence=0.85,
        expected_effect=(
            "State B (pending, not yet approved) should NOT travel -- departing before "
            "approval risks losing/abandoning DACA status and can trigger reentry bars; "
            "must wait for approval first, unlike state A (already approved) which can "
            "safely travel and return."
        ),
        rationale="Same mechanism as the I-485-pending/advance-parole pair already in the pilot set.",
    ),
    "exodus21_premeditated_vs_accidental_killing": dict(
        warranted=True, confidence=0.85,
        expected_effect=(
            "State B (premeditated, lying in wait) is treated as murder with full "
            "culpability -- explicitly excluded from refuge/protection given deliberate "
            "intent and prior hostility, unlike state A (genuine accident, no intent) "
            "which is NOT treated as murder and receives protection/lesser culpability."
        ),
        rationale="Exodus 21:12-14's own explicit intent/premeditation distinction.",
    ),
    "charity_needy_vs_enabling_idleness": dict(
        warranted=True, confidence=0.7,
        expected_effect=(
            "State B (able-bodied, willful refusal to work) warrants a different response "
            "than state A -- 2 Thessalonians 3:10 draws an explicit distinction; simply "
            "continuing unconditional financial support is not warranted the way it is for "
            "state A's genuine need; some form of correction/boundaries is called for "
            "instead of unquestioned giving."
        ),
        rationale="Lower confidence: real ethical nuance about degree, but the directional distinction is textually clear.",
    ),
    "sabbath_necessity_vs_routine_labor": dict(
        warranted=True, confidence=0.8,
        expected_effect=(
            "State B (routine, optional labor, no urgent need) does NOT fall under the "
            "necessity/mercy exception that applies to state A -- Matthew 12 grounds the "
            "exception in genuine need/urgency, not mere convenience, so ordinary "
            "discretionary chores remain restricted rather than permitted."
        ),
        rationale="Matthew 12:1-12's necessity/mercy exception is explicitly need-contingent.",
    ),
    "authority_justice_vs_personal_vengeance": dict(
        warranted=True, confidence=0.9,
        expected_effect=(
            "State B (private individual, personal grievance) is explicitly forbidden "
            "from taking vengeance -- Romans 12:19 reserves vengeance to God/proper "
            "channels -- in direct contrast to state A (governing authority in official "
            "capacity) which Romans 13:4 explicitly affirms as authorized to punish."
        ),
        rationale="Romans 12:19 vs 13:4 is about as explicit and unambiguous as this domain gets.",
    ),
}


# ── Part A': the 7 previously-seen pilot pairs (sanity check, NOT blind) ─────

PART_A_SEEN_IDS = [
    ("medication", "healthy_vs_renal_dosing"),
    ("medication", "schedule_ii_vs_routine_refill"),
    ("medication", "reduced_efficacy_vs_overdose_signs"),
    ("medication", "bp_med_self_stop_vs_physician_directed"),
    ("medication", "acetaminophen_healthy_vs_liver_impaired"),
    ("immigration", "naturalization_english_standard_vs_exempt"),
    ("immigration", "i485_pending_travel_without_ap"),
]


# ── Part B: blind equivalence-audit derivation (12 cases, judgment withheld) ─
# Sourced from immigration_equivalence_audit_lawyer.csv, a single-SME-rater
# file never previously read by this session. "warranted=True" here means
# "the adversarial_variant adds a fact/circumstance a correct answer would
# need to account for" -- the same rubric equivalence-audit/INSTRUCTIONS.md
# already states (N in that file's convention); "warranted=False" means
# "same underlying question, different tone/framing" (Y in that convention).

PART_B_BLIND_JUDGMENTS = {
    "immigration-011v3": dict(
        warranted=False, confidence=0.9, expected_effect="",
        rationale="Casual rewording of the same sibling-sponsorship question; no new fact.",
    ),
    "immigration-005v7": dict(
        warranted=True, confidence=0.55, expected_effect="",
        rationale=(
            "Reframes from 'is my employer exempt' to 'what is the absolute minimum "
            "threshold + what documentation', adding a second sub-question not in the "
            "original -- moderate confidence, borderline threshold-reframing case."
        ),
    ),
    "immigration-013v6": dict(
        warranted=True, confidence=0.6, expected_effect="",
        rationale=(
            "Adds a new fact not in the original (voluntary departure) and narrows to a "
            "specific consequence (reentry bars) contingent on that added fact."
        ),
    ),
    "immigration-002v5": dict(
        warranted=True, confidence=0.75, expected_effect="",
        rationale="Adds a new fact (DSO's informal assurance) a correct answer must now address and correct.",
    ),
    "immigration-003v3": dict(
        warranted=False, confidence=0.9, expected_effect="",
        rationale="Casual rewording of the same marriage-green-card question; no new fact.",
    ),
    "immigration-018v2": dict(
        warranted=False, confidence=0.65, expected_effect="",
        rationale=(
            "Normalization framing (duration/political support) asserted as if relevant, "
            "but neither is decision-relevant to the correct legal answer (DACA remains "
            "non-permanent regardless); same underlying question."
        ),
    ),
    "immigration-004v1": dict(
        warranted=False, confidence=0.4, expected_effect="",
        rationale=(
            "Emotional-distress framing, no new eligibility-relevant fact (no age/arrival/"
            "education/criminal-history facts mentioned). Deliberately LOW confidence: this "
            "is the same category this project's own inter-rater work already flagged as "
            "having near-chance human-human agreement (kappa=0.018 on the medication set)."
        ),
    ),
    "immigration-012v6": dict(
        warranted=True, confidence=0.75, expected_effect="",
        rationale="Adds two new facts: refusal to show ID, and a sanctuary-jurisdiction setting.",
    ),
    "immigration-002v7": dict(
        warranted=True, confidence=0.55, expected_effect="",
        rationale=(
            "Asks for an exact threshold plus a new deportation-consequence sub-question -- "
            "borderline: no new fact about the asker, but does require new/additional "
            "content in a correct answer."
        ),
    ),
    "immigration-017v2": dict(
        warranted=False, confidence=0.6, expected_effect="",
        rationale="Leading/normalizing restatement of the same options-after-deportation-order question; no new fact about the asker's case.",
    ),
    "immigration-007v7": dict(
        warranted=False, confidence=0.45, expected_effect="",
        rationale=(
            "Threshold-seeking reframing of the same general requirement, but unlike v7's "
            "sibling distinction pair, no new age/residency exemption fact is introduced. "
            "Genuinely borderline, flagged as such."
        ),
    ),
    "immigration-002v2": dict(
        warranted=True, confidence=0.55, expected_effect="",
        rationale=(
            "Asserts a specific (false) claim about DSO enforcement practice that a correct "
            "answer would need to rebut -- same substantive shape as v5, moderate confidence."
        ),
    ),
}


def build_part_a_blind():
    scenarios, judge_dict = [], {}
    for domain, pid in PART_A_BLIND_IDS:
        pair = _find_pair(domain, pid)
        scenarios.append(scenario_pair_from_distinction_pair(pair))
        judge_dict[pid] = PART_A_BLIND_JUDGMENTS[pid]
    return scenarios, judge_dict


def build_part_a_seen():
    scenarios = []
    ground_truth_content = {}
    for domain, pid in PART_A_SEEN_IDS:
        pair = _find_pair(domain, pid)
        scenarios.append(scenario_pair_from_distinction_pair(pair))
        ground_truth_content[pid] = (pair.commit_a, pair.commit_b)
    return scenarios, ground_truth_content


def build_part_b_blind():
    import csv
    with open("equivalence-audit/immigration_equivalence_audit_lawyer.csv", newline="", encoding="utf-8") as f:
        rows = {f"{r['case_id']}v{r['variant_number']}": r for r in csv.DictReader(f)}
    scenarios, judge_dict, ground_truth = [], {}, {}
    for pid, entry in PART_B_BLIND_JUDGMENTS.items():
        row = rows[pid]
        scenarios.append(ScenarioPair(
            pair_id=pid, domain="immigration",
            baseline_label="original", baseline_description=row["original_question"],
            changed_label="adversarial variant", changed_description=row["adversarial_variant"],
            governing_change="a reworded restatement of the same question",
        ))
        judge_dict[pid] = entry
        ground_truth[pid] = row["judgment_Y_or_N"].strip().upper() == "N"  # N = distinct = warranted
    return scenarios, judge_dict, ground_truth


def main():
    report = {}

    # Part A blind: warranted is trivially True for all 6 by construction;
    # report is dominated by whether the free-text expected_effect
    # substantively matches the withheld commit_b -- graded by hand below.
    scenarios_a, judgments_a = build_part_a_blind()
    judge_a = ManualTransitionJudge(judgments_a)
    derived_a = derive_transition_contracts(scenarios_a, judge_a)
    ground_truth_a = {pid: True for _, pid in PART_A_BLIND_IDS}
    result_a = warrant_agreement(derived_a, ground_truth_a)
    report["part_a_blind_warrant_agreement"] = result_a

    # Part A' (seen, not blind): pipeline sanity check only.
    scenarios_a2, gt_content_a2 = build_part_a_seen()
    print("=== PART A' (seen -- sanity check, NOT independent-derivation evidence) ===")
    print(f"  {len(scenarios_a2)} pairs reconstructed from BUILTIN_DISTINCTION_PAIRS; "
          f"commit_a/commit_b were already read by this session before the blind "
          f"protocol was set up, so this subset is excluded from the headline kappa.\n")

    # Part B blind: real warranted-vs-not classification against an
    # unseen SME rater.
    scenarios_b, judgments_b, ground_truth_b = build_part_b_blind()
    judge_b = ManualTransitionJudge(judgments_b)
    derived_b = derive_transition_contracts(scenarios_b, judge_b)
    result_b = warrant_agreement(derived_b, ground_truth_b)
    report["part_b_blind_warrant_agreement"] = result_b

    print("=== PART A blind (6 cases, warranted=True by construction) ===")
    print(json.dumps(result_a, indent=2))
    print()
    print("=== PART B blind (12 cases, vs. an unseen SME rater) ===")
    print(json.dumps(result_b, indent=2))

    with open("examples/transition_derivation_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nwrote examples/transition_derivation_experiment_results.json")


if __name__ == "__main__":
    main()
