#!/usr/bin/env python3
"""
epistemic_audit.py — Improve reasoning, not just answers.

A foundational technology for reasoning should not tell people what to think.
It should improve humanity's ability to think well.

This tool does three things:

  AUDIT  — Analyze the epistemic quality of any AI response.
           Not: is this correct?
           But: does this improve the human's ability to reason?

  MAP    — Map the structure of a disagreement between two positions.
           Not: who is right?
           But: what kind of disagreement is this, and how do you advance it?

  SCAFFOLD — Generate the inquiry structure for any question.
             Not: here is the answer.
             But: here is the dependency structure of the question,
                  and here is how you navigate it.

Usage
-----
  # Audit a response
  python epistemic_audit.py audit \\
      --question "What causes inflation?" \\
      --response "Inflation is caused by too much money chasing too few goods..."

  # Read response from file
  python epistemic_audit.py audit \\
      --question "What causes inflation?" \\
      --response-file response.txt

  # Map a disagreement
  python epistemic_audit.py map \\
      --question "Does moderate alcohol consumption have health benefits?" \\
      --position1 "Yes — studies show moderate drinking reduces cardiovascular risk" \\
      --position2 "No — those studies had methodological flaws; any amount is harmful"

  # Generate inquiry scaffold
  python epistemic_audit.py scaffold \\
      "What should I understand about antibiotic resistance before taking a position?"

  # Compare multiple models on the same question
  python epistemic_audit.py compare \\
      --question "What causes inflation?" \\
      --models gpt-4o claude-sonnet-4-6

  # Full pipeline: generate + audit
  python epistemic_audit.py full \\
      --model gpt-4o \\
      --question "What are the main mechanisms of antibiotic resistance?"

Environment
-----------
  OPENAI_API_KEY     — for OpenAI models
  ANTHROPIC_API_KEY  — for Anthropic models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from contradish.epistemic import EpistemicAudit
from contradish.llm import LLMClient


def cmd_audit(args, audit: EpistemicAudit):
    response_text = args.response or ""
    if args.response_file:
        response_text = Path(args.response_file).read_text()
    if not response_text.strip():
        print("Error: --response or --response-file required.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Auditing response to: {args.question[:70]}...")
    profile = audit.check(question=args.question, response=response_text)
    print(profile)

    if args.output_json:
        out = {
            "question":               profile.question,
            "epistemic_quality":      profile.epistemic_quality,
            "certainty_calibration":  profile.certainty_calibration,
            "dependency_legibility":  profile.dependency_legibility,
            "inquiry_advancement":    profile.inquiry_advancement,
            "disagreement_support":   profile.disagreement_support,
            "grade":                  profile.grade(),
            "hidden_assumptions":     profile.hidden_assumptions,
            "uncertain_claims":       profile.uncertain_claims,
            "inquiry_questions":      profile.inquiry_questions,
            "primary_finding":        profile.primary_finding,
            "elapsed_ms":             profile.elapsed_ms,
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  Profile → {args.output_json}\n")


def cmd_map(args, audit: EpistemicAudit):
    print(f"\n  Mapping disagreement on: {args.question[:70]}...")
    dmap = audit.map_disagreement(
        question=args.question,
        position_1=args.position1,
        position_2=args.position2,
    )
    print(dmap)

    if args.output_json:
        out = {
            "question":             dmap.question,
            "primary_type":         dmap.primary_type,
            "type_explanation":     dmap.type_explanation,
            "productive":           dmap.productive,
            "can_both_be_right":    dmap.can_both_be_right,
            "load_bearing_point":   dmap.load_bearing_point,
            "resolution_path":      dmap.resolution_path,
            "minimum_experiment":   dmap.minimum_experiment,
            "clarifying_definitions": dmap.clarifying_definitions,
            "surfaced_assumptions": dmap.surfaced_assumptions,
            "scope_conditions":     dmap.scope_conditions,
            "next_step":            dmap.next_step,
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  Map → {args.output_json}\n")


def cmd_scaffold(args, audit: EpistemicAudit):
    question = args.question
    print(f"\n  Building inquiry scaffold for: {question[:70]}...")
    scaffold = audit.scaffold(question)
    print(scaffold)

    if args.output_json:
        out = {
            "question":               scaffold.question,
            "load_bearing_claims":    scaffold.load_bearing_claims,
            "dependency_order":       scaffold.dependency_order,
            "high_leverage_questions": scaffold.high_leverage_questions,
            "epistemic_traps":        scaffold.epistemic_traps,
            "minimum_viable_path":    scaffold.minimum_viable_path,
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  Scaffold → {args.output_json}\n")


def cmd_compare(args, audit: EpistemicAudit):
    """Generate responses from multiple models and compare epistemic quality."""
    models = args.models
    question = args.question

    print(f"\n  Comparing epistemic quality across {len(models)} models...")
    print(f"  Question: {question[:70]}...")
    print()

    responses = {}
    for model_name in models:
        provider = "anthropic" if "claude" in model_name.lower() else "openai"
        try:
            client = LLMClient(provider=provider)
            print(f"  Querying {model_name}...", end="", flush=True)
            response = client.complete(question, model=model_name)
            responses[model_name] = response
            print(" done")
        except Exception as e:
            print(f" ERROR: {e}", file=sys.stderr)

    if not responses:
        print("No responses collected.", file=sys.stderr)
        sys.exit(1)

    result = audit.compare(question, responses)

    W = 66
    print("\n" + "═" * W)
    print("  Epistemic Quality Comparison")
    print("─" * W)
    print(f"  {'model':<30}  {'EQ':>6}  {'C':>5}  {'D':>5}  {'I':>5}  {'P':>5}")
    print("─" * W)
    for name in result["ranking"]:
        p = result["profiles"][name]
        print(f"  {name:<30}  {p.epistemic_quality:.3f}  "
              f"{p.certainty_calibration:.2f}  {p.dependency_legibility:.2f}  "
              f"{p.inquiry_advancement:.2f}  {p.disagreement_support:.2f}")
    print("─" * W)
    print(f"  Best epistemic quality: {result['best']}")
    print("═" * W)
    print()

    if not args.quiet:
        best_name = result["best"]
        if best_name:
            print(result["profiles"][best_name])

    if args.output_json:
        out = {
            "question": question,
            "ranking": result["ranking"],
            "scores": result["scores"],
            "profiles": {
                name: {
                    "epistemic_quality":      p.epistemic_quality,
                    "certainty_calibration":  p.certainty_calibration,
                    "dependency_legibility":  p.dependency_legibility,
                    "inquiry_advancement":    p.inquiry_advancement,
                    "disagreement_support":   p.disagreement_support,
                    "grade":                  p.grade(),
                    "hidden_assumptions":     p.hidden_assumptions,
                    "inquiry_questions":      p.inquiry_questions,
                    "primary_finding":        p.primary_finding,
                }
                for name, p in result["profiles"].items()
            }
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  Results → {args.output_json}\n")


def cmd_full(args, audit: EpistemicAudit):
    """Generate model response, audit it, and scaffold the question."""
    question = args.question
    model = args.model
    provider = "anthropic" if "claude" in model.lower() else "openai"

    print(f"\n  Full pipeline for: {question[:70]}...")
    print(f"  Model: {model}")

    try:
        client = LLMClient(provider=provider)
        print("  Generating response...", end="", flush=True)
        response = client.complete(question, model=model)
        print(" done")
    except Exception as e:
        print(f"\n  Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n  === MODEL RESPONSE ===")
    print(f"  {response[:400]}{'...' if len(response) > 400 else ''}")

    print("\n  Auditing epistemic quality...", end="", flush=True)
    profile = audit.check(question=question, response=response)
    print(" done")
    print(profile)

    if not args.skip_scaffold:
        print("  Building inquiry scaffold...", end="", flush=True)
        scaffold = audit.scaffold(question)
        print(" done")
        print(scaffold)

    if args.output_json:
        out = {
            "question": question,
            "model": model,
            "response": response,
            "epistemic_audit": {
                "epistemic_quality":     profile.epistemic_quality,
                "certainty_calibration": profile.certainty_calibration,
                "dependency_legibility": profile.dependency_legibility,
                "inquiry_advancement":   profile.inquiry_advancement,
                "disagreement_support":  profile.disagreement_support,
                "grade":                 profile.grade(),
                "hidden_assumptions":    profile.hidden_assumptions,
                "inquiry_questions":     profile.inquiry_questions,
                "primary_finding":       profile.primary_finding,
            },
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"  Results → {args.output_json}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Improve reasoning, not just answers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"],
        help="Judge provider. Auto-detected from environment if omitted.")
    parser.add_argument("--output-json",
        help="Write results to this JSON file.")

    sub = parser.add_subparsers(dest="command", required=True)

    # audit
    p_audit = sub.add_parser("audit", help="Audit the epistemic quality of a response")
    p_audit.add_argument("--question", required=True)
    p_audit.add_argument("--response",
        help="Response text (or use --response-file)")
    p_audit.add_argument("--response-file",
        help="Path to file containing the response text")

    # map
    p_map = sub.add_parser("map", help="Map the structure of a disagreement")
    p_map.add_argument("--question", required=True)
    p_map.add_argument("--position1", required=True)
    p_map.add_argument("--position2", required=True)

    # scaffold
    p_scaf = sub.add_parser("scaffold", help="Generate an inquiry scaffold for any question")
    p_scaf.add_argument("question")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare epistemic quality across models")
    p_cmp.add_argument("--question", required=True)
    p_cmp.add_argument("--models", nargs="+", required=True)
    p_cmp.add_argument("--quiet", "-q", action="store_true")

    # full
    p_full = sub.add_parser("full",
        help="Generate a response, audit it, and scaffold the question")
    p_full.add_argument("--model", required=True)
    p_full.add_argument("--question", required=True)
    p_full.add_argument("--skip-scaffold", action="store_true")

    args = parser.parse_args()
    # Pass output_json to subcommand args
    if hasattr(args, 'output_json') and not hasattr(args, '_has_output'):
        pass  # already set

    audit = EpistemicAudit(provider=args.provider)

    if args.command == "audit":
        args.output_json = getattr(args, 'output_json', None)
        cmd_audit(args, audit)
    elif args.command == "map":
        args.output_json = getattr(args, 'output_json', None)
        cmd_map(args, audit)
    elif args.command == "scaffold":
        args.output_json = getattr(args, 'output_json', None)
        cmd_scaffold(args, audit)
    elif args.command == "compare":
        args.output_json = getattr(args, 'output_json', None)
        cmd_compare(args, audit)
    elif args.command == "full":
        args.output_json = getattr(args, 'output_json', None)
        cmd_full(args, audit)


if __name__ == "__main__":
    main()
