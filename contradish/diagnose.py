"""
contradish diagnose — consistency failure diagnosis and prescription.

The measurement layer (CTS, SRA, etc.) tells you your model is drifting.
The diagnosis layer tells you why, and what to change.

For every drifted case, diagnose produces:
  failure_mode            — which category of adversarial failure occurred
  why_it_worked           — one sentence on the exact trigger
  counterfactual_response — what the model should have said instead
  system_prompt_fix       — language to add to your system prompt to prevent this
  training_example        — a fine-tuning pair (user/assistant) ready to use

Aggregated across all failures in a result file:
  failure_mode_distribution — which patterns dominate your model's weaknesses
  aggregate_fixes           — deduplicated system prompt additions, ranked by impact
  training_examples         — JSONL-ready fine-tuning set, sorted by severity

Usage:
    from contradish.diagnose import analyze_result, export_training_jsonl
    report = analyze_result("results/sra_claude-sonnet-4-6.json", "openai", "gpt-4o")
    export_training_jsonl(report, "repair_package.jsonl")
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────
# Result file parsing
# ─────────────────────────────────────────────────────────

def _extract_from_sra(result: dict) -> list[dict]:
    """Extract drifted cases from an SRA result JSON."""
    cases = []
    for domain_result in result.get("domains", []):
        domain = domain_result.get("domain", "unknown")
        for case_result in domain_result.get("cases", []):
            severity = case_result.get("severity", "high")
            topic    = case_result.get("topic", "")
            canonical = case_result.get("canonical_position", "")
            for variant in case_result.get("variants", []):
                if variant.get("outcome") == "drifted":
                    technique = variant.get("techniques", "unknown")
                    if isinstance(technique, list):
                        technique = "+".join(technique)
                    cases.append({
                        "domain":            domain,
                        "severity":          severity,
                        "topic":             topic,
                        "technique":         technique,
                        "question":          variant.get("question", ""),
                        "actual_response":   variant.get("response", ""),
                        "canonical_position": canonical,
                        "source":            "sra",
                    })
    return cases


def _extract_from_benchmark(result: dict) -> list[dict]:
    """
    Extract failed/inconsistent cases from a main benchmark result JSON.
    Cases with consistency_score < 0.5 are selected for diagnosis.
    """
    cases = []
    for policy_result in result.get("policies", []):
        domain   = policy_result.get("policy", "unknown")
        severity = policy_result.get("severity", "medium")
        for case in policy_result.get("cases", []):
            case_score = case.get("consistency_score", 1.0)
            if case_score < 0.5:
                worst_variant = min(
                    case.get("variants", [{}]),
                    key=lambda v: v.get("per_variant_score", case_score),
                    default={},
                )
                if worst_variant:
                    cases.append({
                        "domain":            domain,
                        "severity":          severity,
                        "topic":             case.get("name", ""),
                        "technique":         worst_variant.get("technique", "unknown"),
                        "question":          worst_variant.get("input", ""),
                        "actual_response":   worst_variant.get("response", ""),
                        "canonical_position": case.get("original_question", ""),
                        "source":            "benchmark",
                    })
    return cases


def load_drift_cases(result_path: str) -> tuple[list[dict], str]:
    """
    Load any contradish result JSON and extract drift/failure cases.
    Returns (cases, source_type).
    """
    path = Path(result_path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    with open(path) as f:
        result = json.load(f)

    benchmark = result.get("benchmark", "")
    if "sra" in benchmark:
        return _extract_from_sra(result), "sra"
    return _extract_from_benchmark(result), "benchmark"


# ─────────────────────────────────────────────────────────
# Core diagnosis
# ─────────────────────────────────────────────────────────

def diagnose_case(case: dict, judge) -> dict:
    """Run repair diagnosis on a single drift case."""
    result = judge.diagnose_drift(
        question          = case["question"],
        canonical_position= case["canonical_position"],
        actual_response   = case["actual_response"],
        technique         = case["technique"],
        domain            = case["domain"],
        severity          = case["severity"],
    )
    result["topic"]  = case.get("topic", "")
    result["source"] = case.get("source", "unknown")
    return result


def analyze_result(
    result_path: str,
    judge_provider: str,
    judge_model: str,
    max_cases: Optional[int] = None,
) -> dict:
    """
    Full diagnosis of a contradish result file.

    Loads all drifted/failed cases, diagnoses each one, and returns a
    structured repair report: per-case diagnoses plus aggregate recommendations.
    """
    from contradish.llm   import LLMClient
    from contradish.judge import Judge

    drift_cases, source_type = load_drift_cases(result_path)

    if not drift_cases:
        return {
            "result_path": result_path,
            "source_type": source_type,
            "drift_count": 0,
            "diagnoses":   [],
            "aggregate":   None,
            "message":     "No drift cases found. Model performed well.",
        }

    if max_cases:
        drift_cases = drift_cases[:max_cases]

    llm   = LLMClient(provider=judge_provider, model=judge_model)
    judge = Judge(llm)

    diagnoses = []
    for case in drift_cases:
        diagnosis = diagnose_case(case, judge)
        diagnoses.append(diagnosis)

    aggregate = _aggregate(diagnoses)

    return {
        "result_path": result_path,
        "source_type": source_type,
        "drift_count": len(diagnoses),
        "diagnoses":   diagnoses,
        "aggregate":   aggregate,
    }


# ─────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _aggregate(diagnoses: list[dict]) -> dict:
    """
    Aggregate individual diagnoses into actionable package:
      - failure_mode_distribution: ranked by frequency
      - domain_distribution:       where failures concentrate
      - aggregate_fixes:           deduplicated system prompt additions
      - training_examples:         JSONL-ready fine-tuning set
      - priority_cases:            critical/high severity failures to address first
    """
    if not diagnoses:
        return {}

    mode_counts   = Counter(d.get("failure_mode", "UNKNOWN") for d in diagnoses)
    domain_counts = Counter(d.get("domain", "unknown") for d in diagnoses)

    # Best system prompt fix per failure mode (highest fix_confidence)
    fixes_by_mode: dict[str, dict] = {}
    for d in diagnoses:
        mode = d.get("failure_mode", "UNKNOWN")
        fix  = d.get("system_prompt_fix", "")
        conf = d.get("fix_confidence", 0.0)
        if fix and (mode not in fixes_by_mode or conf > fixes_by_mode[mode]["confidence"]):
            fixes_by_mode[mode] = {"fix": fix, "confidence": conf}

    aggregate_fixes = sorted(
        [
            {
                "failure_mode":      mode,
                "addresses_count":   mode_counts[mode],
                "system_prompt_fix": data["fix"],
                "confidence":        round(data["confidence"], 2),
            }
            for mode, data in fixes_by_mode.items()
        ],
        key=lambda x: -x["addresses_count"],
    )

    # Fine-tuning examples, critical-first
    training_examples = [
        {
            "messages": [
                {"role": "user",      "content": d["training_example"]["user"]},
                {"role": "assistant", "content": d["training_example"]["assistant"]},
            ],
            "metadata": {
                "domain":       d.get("domain", ""),
                "severity":     d.get("severity", ""),
                "failure_mode": d.get("failure_mode", ""),
                "technique":    d.get("technique", ""),
            },
        }
        for d in sorted(diagnoses, key=lambda x: _SEVERITY_RANK.get(x.get("severity", "medium"), 2))
        if (
            isinstance(d.get("training_example"), dict)
            and d["training_example"].get("user")
            and d["training_example"].get("assistant")
        )
    ]

    priority_cases = [
        {
            "domain":       d.get("domain", ""),
            "topic":        d.get("topic", ""),
            "failure_mode": d.get("failure_mode", ""),
            "why":          d.get("why_it_worked", ""),
            "severity":     d.get("severity", ""),
        }
        for d in diagnoses
        if d.get("severity") in ("critical", "high")
    ]

    return {
        "failure_mode_distribution": [
            {"failure_mode": m, "count": c} for m, c in mode_counts.most_common()
        ],
        "domain_distribution": [
            {"domain": d, "count": c} for d, c in domain_counts.most_common()
        ],
        "aggregate_fixes":   aggregate_fixes,
        "training_examples": training_examples,
        "priority_cases":    priority_cases,
        "total_diagnoses":   len(diagnoses),
        "total_ft_examples": len(training_examples),
    }


# ─────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────

def export_training_jsonl(report: dict, output_path: str) -> int:
    """Write fine-tuning examples to JSONL. Returns count written."""
    examples = (report.get("aggregate") or {}).get("training_examples", [])
    if not examples:
        return 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


def export_system_prompt_block(report: dict, output_path: str) -> str:
    """Write aggregate system prompt recommendations to a text file."""
    fixes = (report.get("aggregate") or {}).get("aggregate_fixes", [])
    if not fixes:
        return ""

    lines = [
        "# Consistency policy block — generated by contradish diagnose",
        "# Add this section to your system prompt to address detected drift patterns.",
        "",
    ]
    for fix in fixes:
        mode  = fix["failure_mode"].replace("_", " ").lower()
        count = fix["addresses_count"]
        lines.append(f"# Addresses: {mode} ({count} failure{'s' if count > 1 else ''} detected)")
        lines.append(fix["system_prompt_fix"])
        lines.append("")

    block = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(block)
    return block
