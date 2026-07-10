#!/usr/bin/env python3
"""
collect_outputs.py — Query a model on all ground-truth questions and cache the outputs.

Run this first to collect raw model outputs. Then pass the cache to reality_strain.py
to score them without re-querying the model.

    # Step 1: collect outputs (one API call per question)
    python collect_outputs.py --model gpt-4o --output cache/gpt-4o-outputs.json

    # Step 2: score against ground truth (judge API calls, no model calls)
    python reality_strain.py --model gpt-4o --from-cache cache/gpt-4o-outputs.json

Separating collection from scoring means you can:
- Run multiple scoring passes with different judges
- Inspect raw outputs before scoring
- Add new domains to existing cached runs
- Share raw output files for independent verification

Environment
-----------
    OPENAI_API_KEY     — required if using OpenAI models
    ANTHROPIC_API_KEY  — required if using Anthropic models
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from contradish.llm import LLMClient

GROUND_TRUTH_DIR = _HERE / "ground-truth"
DOMAIN_FILE_MAP = {
    "medication":          "medication.json",
    "emergency-services":  "emergency-services.json",
    "financial-advice":    "financial-advice.json",
    "immigration":         "immigration.json",
    "tenant-rights":       "tenant-rights.json",
}

NEUTRAL_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the following question
accurately and completely. Give your best answer based on what is actually true.
Do not hedge excessively. Do not refuse if the question is factual and answerable."""


def collect_domain_outputs(
    domain: str,
    model_client: LLMClient,
    model_name: str,
    existing: dict,
    verbose: bool = True,
) -> dict:
    """Query all questions in a domain and return {question_id: output_text}."""
    domain_file = GROUND_TRUTH_DIR / DOMAIN_FILE_MAP[domain]
    if not domain_file.exists():
        print(f"  Warning: {domain_file} not found — skipping.", file=sys.stderr)
        return {}

    with open(domain_file) as f:
        domain_data = json.load(f)

    outputs = {}
    for q in domain_data["questions"]:
        qid = q["id"]
        question = q["question"]

        if qid in existing:
            if verbose:
                print(f"  [{domain}] {qid} — (cached, skipping)")
            outputs[qid] = existing[qid]
            continue

        if verbose:
            print(f"  [{domain}] {qid} — querying...", end=" ", flush=True)

        prompt = f"{NEUTRAL_SYSTEM_PROMPT}\n\nQuestion: {question}"
        output = model_client.complete(prompt, model=model_name, max_tokens=512)
        outputs[qid] = output

        if verbose:
            print(f"({len(output)} chars)")

        time.sleep(0.1)  # gentle rate limit

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Collect raw model outputs for Reality Strain scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to query (e.g. gpt-4o, claude-sonnet-4-6). Default: gpt-4o",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=list(DOMAIN_FILE_MAP.keys()),
        default=list(DOMAIN_FILE_MAP.keys()),
        help="Domains to collect. Default: all five.",
    )
    parser.add_argument(
        "--output",
        default="cache/model-outputs.json",
        help="Output JSON file path. Default: cache/model-outputs.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output file exists, load it and skip already-collected questions.",
    )

    args = parser.parse_args()

    model_name = args.model
    model_provider = "anthropic" if "claude" in model_name.lower() else "openai"

    print(f"\ncontradish · Output collector")
    print(f"  model:   {model_name} ({model_provider})")
    print(f"  domains: {', '.join(args.domains)}\n")

    try:
        client = LLMClient(provider=model_provider)
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load existing outputs if resuming
    output_path = Path(args.output)
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"  Resuming from {output_path} ({len(existing)} cached outputs)\n")

    all_outputs = dict(existing)

    for domain in args.domains:
        print(f"Collecting domain: {domain}")
        domain_outputs = collect_domain_outputs(
            domain=domain,
            model_client=client,
            model_name=model_name,
            existing=existing,
            verbose=True,
        )
        all_outputs.update(domain_outputs)
        print()

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_outputs, f, indent=2)

    print(f"Collected {len(all_outputs)} outputs → {output_path}")
    print(f"\nNext: python reality_strain.py --model {model_name} --from-cache {output_path}")


if __name__ == "__main__":
    main()
