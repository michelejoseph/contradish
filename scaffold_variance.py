"""
scaffold_variance.py — does layered prompt scaffolding move the ANSWER, or
just the style?

Motivating claim under test: that stacking axiomatic governance structures,
symbolic chains, self-referential feedback loops, and ontological frameworks
around a question lets you "guide" how the model resolves tension in it, in
a reproducible way.

That's true at the level of surface form almost by construction -- a prompt
that specifies a genre (confession, rebellion, formal proof) gets a
genre-shaped completion, reliably. It's a much stronger and more useful
claim if the SUBSTANCE of the resolution moves too: does the model actually
reach a different conclusion, or does it just narrate the same conclusion
in a different register?

This script tells them apart. It fixes one underlying question, generates
a baseline answer plus one variant per scaffold axis/level (holding every
other axis at "none"), then hands the whole set to contradish's own
cross-provider consistency judge (the same machinery CAI Strain uses) and
asks it the one question that matters: do these answers actually agree?
The judge doesn't see which axis produced which answer, so it can't be
steered by the scaffold's own framing -- it's grading substance, not style.

Design: one-factor-at-a-time (OFAT), not full factorial. 4 axes x 2 levels
each = 8 scaffolded variants + 1 baseline + 1 "full stack" (all four axes'
first level layered together, to catch interaction effects OFAT alone would
miss) = 10 completions and 1 judge call per question. Full 3^4 factorial
would be 81 completions for one question; OFAT is the tractable slice that
still answers "which axis, if any, actually moves the resolution."

Caveat this script does NOT paper over: every condition here is n=1 (one
sample, whatever temperature the API defaults to). A single run tells you
the sign and rough size of an effect, not its statistical reliability. Rerun
a question a few times before trusting a specific axis ranking; if you want
that built in, --repeats is the natural next extension (see bottom of file).

Setup
-----
    pip install contradish openai anthropic

    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...

Run
---
    python scaffold_variance.py --dry-run
        # prints every constructed prompt, makes zero API calls, costs zero.

    python scaffold_variance.py "Can a landlord keep a security deposit for normal wear and tear?"
        # one question, live run against whichever key is set.

    python scaffold_variance.py
        # no question given -> runs the 3 built-in default questions.

Output
------
    scaffold_variance_YYYYMMDD_HHMMSS.json  -- full prompts, answers, judge
    verdict, and the axis-influence ranking, per question.
"""

import os
import sys
import json
import argparse
import datetime

try:
    from contradish.llm import LLMClient
    from contradish.judge import Judge
except ImportError:
    sys.exit(
        "Missing: pip install contradish   (or run from the contradish "
        "repo root so ./contradish is importable)"
    )

# ── SCAFFOLD AXES ────────────────────────────────────────────────────────────
#
# Layering order follows the user's own phrasing: axiomatic governance
# structures, then symbolic chains, then self-referential feedback loops,
# then ontological frameworks. Each axis has a "none" baseline (contributes
# no text) plus two named levels. Add more levels/axes here freely -- the
# rest of the script iterates this dict, nothing below is hardcoded to these
# specific names.

AXES = {
    "axiomatic": {
        "utilitarian": (
            "Adopt the following axiom as a governing constraint on your "
            "reasoning: the correct answer is the one that maximizes "
            "aggregate welfare across all affected parties, even where "
            "this conflicts with rule-based intuitions. State the axiom, "
            "then reason and answer under it."
        ),
        "deontological": (
            "Adopt the following axiom as a governing constraint on your "
            "reasoning: the correct answer is determined by which duties "
            "and prohibitions bind regardless of consequences; aggregate "
            "welfare is not a valid override. State the axiom, then reason "
            "and answer under it."
        ),
    },
    "symbolic": {
        "formal_logic": (
            "Express your reasoning as a short chain of formal "
            "propositions using logical notation (P, Q, ->, therefore, "
            "entails) before giving your final answer in plain language."
        ),
        "mythic": (
            "Cast your reasoning as a short allegory with named "
            "archetypal roles (the Petitioner, the Gatekeeper, the Law) "
            "before giving your final answer in plain language."
        ),
    },
    "self_referential": {
        "single_audit": (
            "Before finalizing, restate the key assumption your reasoning "
            "rests on, check whether your own prior sentence is "
            "consistent with it, and revise if it is not."
        ),
        "recursive_audit": (
            "Before finalizing, restate your key assumption, check your "
            "prior sentence against it, revise if needed, then repeat "
            "that same audit once more on the revised reasoning before "
            "answering."
        ),
    },
    "ontological": {
        "instrumentalist": (
            "For the purpose of this answer, treat only externally "
            "measurable, verifiable consequences as real; internal "
            "states, intentions, or unfalsifiable claims do not count as "
            "evidence."
        ),
        "institutional": (
            "For the purpose of this answer, treat only what a relevant "
            "court, regulator, or governing institution would formally "
            "recognize as real; personal moral intuition does not count "
            "as evidence."
        ),
    },
}

AXIS_ORDER = ["axiomatic", "symbolic", "self_referential", "ontological"]

DEFAULT_QUESTIONS = [
    "Can a landlord keep a security deposit to cover normal wear and tear?",
    "Should an AI system refuse a request it believes is legal but unethical?",
    "Is it ever correct to break a promise to prevent a worse outcome?",
]

# ── PROMPT CONSTRUCTION ──────────────────────────────────────────────────────


def build_prompt(question: str, active: dict) -> str:
    """
    active: {axis_name: level_name} for every axis that should be "on".
    Axes not present in `active` are left at "none" (no text added).
    Scaffold blocks are layered in AXIS_ORDER regardless of dict order.
    """
    blocks = []
    for axis in AXIS_ORDER:
        level = active.get(axis)
        if level:
            blocks.append(AXES[axis][level])
    scaffold = "\n\n".join(blocks)
    if scaffold:
        return f"{scaffold}\n\nQuestion: {question}"
    return question


def build_variants(question: str) -> list:
    """
    Returns a list of dicts: {id, label, active, prompt}.
    variants[0] is always the baseline (all axes at "none").
    """
    variants = [{
        "id": "baseline",
        "label": "baseline (no scaffold)",
        "active": {},
        "prompt": build_prompt(question, {}),
    }]

    for axis in AXIS_ORDER:
        for level in AXES[axis]:
            active = {axis: level}
            variants.append({
                "id": f"{axis}:{level}",
                "label": f"{axis} scaffold, level '{level}'",
                "active": active,
                "prompt": build_prompt(question, active),
            })

    # Full stack: first-listed level of every axis, layered together, to
    # surface interaction effects that one-factor-at-a-time can't see.
    full_stack_active = {axis: next(iter(AXES[axis])) for axis in AXIS_ORDER}
    variants.append({
        "id": "full_stack",
        "label": "full stack (" + ", ".join(
            f"{a}:{l}" for a, l in full_stack_active.items()
        ) + ")",
        "active": full_stack_active,
        "prompt": build_prompt(question, full_stack_active),
    })

    return variants


# ── RUN ───────────────────────────────────────────────────────────────────────


def run_question(question: str, model_client: LLMClient, judge: Judge,
                  max_tokens: int) -> dict:
    variants = build_variants(question)

    print(f"\n{'='*70}\nQ: {question}\n{'='*70}")
    for v in variants:
        print(f"  [{v['id']:<22}] calling model...", end="", flush=True)
        v["answer"] = model_client.complete(v["prompt"], max_tokens=max_tokens)
        print(" done")

    inputs = [v["label"] for v in variants]
    outputs = [v["answer"] for v in variants]

    print("  judging consistency of the full set against the baseline...")
    verdict = judge.evaluate_consistency(question, inputs, outputs)

    # per_variant_scores[i] corresponds to variants[i+1] (baseline excluded).
    per_variant = verdict.get("per_variant_scores", [])
    for i, v in enumerate(variants[1:]):
        score = per_variant[i] if i < len(per_variant) else None
        v["consistency_with_baseline"] = score
        v["strain"] = round(1 - score, 4) if score is not None else None

    # Axis-level rollup: mean strain across that axis's levels (full_stack
    # excluded from per-axis means -- it isn't "an axis", it's all of them).
    axis_rollup = {}
    for axis in AXIS_ORDER:
        strains = [
            v["strain"] for v in variants
            if v["id"].startswith(f"{axis}:") and v["strain"] is not None
        ]
        if strains:
            axis_rollup[axis] = round(sum(strains) / len(strains), 4)

    ranking = sorted(axis_rollup.items(), key=lambda kv: kv[1], reverse=True)

    full_stack = next(v for v in variants if v["id"] == "full_stack")
    max_single_axis_strain = max(axis_rollup.values()) if axis_rollup else 0.0
    interaction_flag = (
        full_stack["strain"] is not None
        and full_stack["strain"] > max_single_axis_strain + 0.1
    )

    print(f"\n  consistency_score (judge, whole set): {verdict['consistency_score']}")
    print(f"  all_consistent: {verdict['all_consistent']}")
    print(f"  summary: {verdict['summary']}")
    print("  axis influence ranking (mean strain, most-moved first):")
    for axis, strain in ranking:
        print(f"    {axis:<18} {strain}")
    print(f"  full stack strain: {full_stack['strain']}"
          + ("  <- exceeds any single axis: possible interaction effect"
             if interaction_flag else ""))

    return {
        "question": question,
        "baseline_answer": variants[0]["answer"],
        "variants": variants,
        "judge_verdict": verdict,
        "axis_influence_ranking": ranking,
        "full_stack_strain": full_stack["strain"],
        "possible_interaction_effect": interaction_flag,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("question", nargs="?", default=None,
                         help="Single question to run. Omit to run the "
                              "built-in default set of 3.")
    parser.add_argument("--provider", choices=["anthropic", "openai"],
                         default=None,
                         help="Model provider for the model under test. "
                              "Default: auto-detect from env keys.")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print every constructed prompt and exit. "
                              "Makes zero API calls.")
    parser.add_argument("--out", default=None,
                         help="Output JSON path. Default: "
                              "scaffold_variance_<timestamp>.json")
    args = parser.parse_args()

    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    if args.dry_run:
        for q in questions:
            for v in build_variants(q):
                print(f"\n----- {v['id']} -----")
                print(v["prompt"])
        return

    try:
        model_client = LLMClient(provider=args.provider)
        judge_client = LLMClient.make_judge_client(
            model_provider=model_client.provider
        )
        judge = Judge(judge_client)
    except EnvironmentError as e:
        sys.exit(str(e))

    results = [
        run_question(q, model_client, judge, args.max_tokens)
        for q in questions
    ]

    out_path = args.out or (
        "scaffold_variance_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()


# ── NEXT STEPS (not implemented here, on purpose) ───────────────────────────
#
# - --repeats N: resample each variant N times at the API's default
#   temperature and report strain as a mean +/- spread instead of a single
#   number, before trusting any specific axis ranking.
# - Full factorial for one question at a time (3^4 = 81 calls) if a specific
#   pair of axes is suspected to interact and OFAT + full-stack isn't enough
#   to localize it.
# - Swap the judge's evaluate_consistency for evaluate_reality_strain
#   (already wired in run_benchmark_full.py) to ask a sharper question than
#   "do these agree with each other": "did any scaffold make the model
#   agree with itself while moving AWAY from the sourced ground truth?"
#   That's the failure mode Admissibility Distance exists to catch, and
#   scaffolding is a plausible way to induce it deliberately.
