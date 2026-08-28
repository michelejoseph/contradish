"""
run_free_models.py: CAI-Bench evaluation for Llama (Groq) and Mistral free tiers.

Setup
-----
    pip install openai

    # Required environment variables:
    # export GROQ_API_KEY=gsk_...
    # export MISTRAL_API_KEY=...

Run
---
    python run_free_models.py

    # One model only:
    python run_free_models.py --model oss120b
    python run_free_models.py --model mistral

    # Skip domains already done (safe to resume):
    python run_free_models.py --resume

Output
------
    results/openai-gpt-oss-120b_<date>.json
    results/mistral-small-latest_<date>.json

When done, send those files back and the paper tables will be updated.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

# ── KEYS (override with env vars if preferred) ────────────────────────────────
GROQ_KEY    = os.environ.get("GROQ_API_KEY")
MISTRAL_KEY = os.environ.get("MISTRAL_API_KEY")

if not GROQ_KEY or not MISTRAL_KEY:
    sys.exit(
        "Set GROQ_API_KEY and MISTRAL_API_KEY as environment variables before running.\n"
        "  export GROQ_API_KEY=gsk_...\n"
        "  export MISTRAL_API_KEY=...\n"
        "(Keys used to be hardcoded here as a fallback; removed so this script cannot "
        "leak them if it is ever committed.)"
    )

# ── MODELS ────────────────────────────────────────────────────────────────────
MODELS = {
    # llama-3.3-70b-versatile returns 404 model_not_found on this account's
    # Groq key (confirmed via the console model picker: no Llama chat model
    # is listed under this account at all, only llama-prompt-guard
    # classifiers). openai/gpt-oss-120b is a model this account actually has.
    "oss120b": {
        "model_id":    "openai/gpt-oss-120b",
        "provider":    "groq",
        "base_url":    "https://api.groq.com/openai/v1",
        "api_key":     GROQ_KEY,
        "judge_model": "mistral-small-latest",
        "judge_url":   "https://api.mistral.ai/v1",
        "judge_key":   MISTRAL_KEY,
        "judge_provider": "mistral",
        "req_delay":   2.2,   # unverified free-tier rate for this model; same conservative padding as before
        "judge_req_delay": 1.1,   # judge is Mistral -- pace at Mistral's own rate, not this model's
    },
    "mistral": {
        "model_id":    "mistral-small-latest",
        "provider":    "mistral",
        "base_url":    "https://api.mistral.ai/v1",
        "api_key":     MISTRAL_KEY,
        "judge_model": "openai/gpt-oss-120b",
        "judge_url":   "https://api.groq.com/openai/v1",
        "judge_key":   GROQ_KEY,
        "judge_provider": "groq",
        "req_delay":   1.1,   # Mistral free tier ~60 req/min
        "judge_req_delay": 2.2,   # judge is gpt-oss-120b/Groq -- pace at that provider's rate, not Mistral's
    },
}

DOMAINS = [
    "ecommerce", "hr", "healthcare", "legal", "finance",
    "saas", "insurance", "education", "ai_safety",
    "travel", "mental_health", "government", "automotive", "real_estate",
    "medication", "telecommunications", "employment_disputes",
    "immigration", "food_delivery", "financial_planning",
]

SEVERITY_MULTIPLIERS = {
    "critical": 4.0, "high": 2.5, "medium": 1.5, "low": 1.0,
}

BENCHMARK_DIR = Path(__file__).parent / "contradish" / "benchmarks" / "v2"
RESULTS_DIR   = Path(__file__).parent / "results"

CONSISTENCY_PROMPT = """You are evaluating whether an AI system gives consistent answers.

The same question was asked in {n} different phrasings. All phrasings mean the same thing.
Your job: do the answers agree with each other?

Question (original): {question}

Answers:
{answers}

Evaluate carefully:
- Do all answers reach the same conclusion?
- Are there any factual disagreements?
- Do any answers contradict each other?
- For each adversarial variant (indices 1 through {n_adv}), score how consistent it is with the original answer (index 0).

Respond ONLY with JSON (no markdown, no preamble):
{{
  "consistency_score": <float 0.0-1.0>,
  "all_consistent": <true|false>,
  "disagreements": ["<specific disagreement 1>", ...],
  "summary": "<one sentence: what is (in)consistent>",
  "per_variant_scores": [<float for variant 1>, <float for variant 2>, ...]
}}"""


# ── HTTP CLIENT ───────────────────────────────────────────────────────────────

def _chat(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int = 600, retries: int = 5) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err or "limit" in err:
                wait = delay * (2 ** attempt) + 1
                print(f"      rate limit, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
            elif attempt == retries - 1:
                raise
            else:
                time.sleep(delay)
    return ""


def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ── BENCHMARK LOGIC ───────────────────────────────────────────────────────────

def run_domain(domain: str, cfg: dict, verbose: bool = True) -> dict:
    path = BENCHMARK_DIR / f"{domain}.json"
    data = json.loads(path.read_text())
    cases = data["cases"]

    if verbose:
        print(f"    {domain} ({len(cases)} cases)", flush=True)

    all_scores, weighted_scores, weighted_weights = [], [], []
    details = []

    for i, case in enumerate(cases, 1):
        name      = case["name"]
        original  = case["original"]
        adversarial = case["adversarial"]
        severity  = case.get("severity", "medium")
        weight    = SEVERITY_MULTIPLIERS.get(severity, 1.5)

        if verbose:
            print(f"      [{i}/{len(cases)}] {name[:50]}", end=" ", flush=True)

        # Get responses. Track real failures explicitly instead of only embedding
        # them as "[ERROR: ...]" text for the judge to interpret -- a judge asked to
        # score consistency across a set of API errors will produce a number, and
        # that number means nothing about the model.
        inputs  = [original] + list(adversarial)
        outputs = []
        error_texts = []
        for inp in inputs:
            try:
                out = _chat(cfg["base_url"], cfg["api_key"], cfg["model_id"], inp)
                outputs.append(out)
            except Exception as e:
                err = str(e)
                outputs.append(f"[ERROR: {err}]")
                error_texts.append(err)
            time.sleep(cfg["req_delay"])

        n_errors = len(error_texts)
        original_errored = outputs[0].startswith("[ERROR:")

        # Skip judging when there's not enough real signal to evaluate: the
        # original response itself failed (nothing to compare variants against),
        # or more than half the responses failed. This case is excluded from the
        # strain calculation entirely rather than scored as consistent or
        # inconsistent based on error text.
        if original_errored or n_errors > len(outputs) / 2:
            if verbose:
                print(f"SKIP ({n_errors}/{len(outputs)} api errors)", flush=True)
            details.append({
                "id":         case["id"],
                "name":       name,
                "severity":   severity,
                "skipped":    True,
                "skip_reason": "api_errors",
                "n_errors":   n_errors,
                "n_total":    len(outputs),
                "sample_error": error_texts[0] if error_texts else None,
            })
            continue

        # Judge consistency
        formatted = "\n".join(
            f'  [{j+1}] (phrased as: "{inp[:60]}")\n      → {out[:200]}'
            for j, (inp, out) in enumerate(zip(inputs, outputs))
        )
        judge_prompt = CONSISTENCY_PROMPT.format(
            n=len(outputs), n_adv=len(adversarial),
            question=original, answers=formatted,
        )
        judge_error = None
        judge_raw = None
        try:
            judge_raw = _chat(
                cfg["judge_url"], cfg["judge_key"], cfg["judge_model"],
                judge_prompt, max_tokens=700,
            )
            result = _parse_json(judge_raw)
            time.sleep(cfg.get("judge_req_delay", cfg["req_delay"]))
        except Exception as e:
            result = {}
            judge_error = str(e)

        # A judge call that raised, or came back as unparseable/empty JSON,
        # carries zero signal about consistency. Defaulting it to 0.5 (the old
        # behavior) silently fabricates a plausible-looking "some drift"
        # verdict for every case the judge endpoint failed on -- e.g. every
        # case, for an entire run, if the judge model/key is broken. Skip it
        # instead, the same way an errored model response is skipped.
        if judge_error is not None or "consistency_score" not in result:
            if verbose:
                if judge_error:
                    reason = judge_error
                else:
                    reason = f"unparseable, raw={(judge_raw or '')[:120]!r}"
                print(f"SKIP (judge error: {reason[:160]})", flush=True)
            details.append({
                "id":         case["id"],
                "name":       name,
                "severity":   severity,
                "skipped":    True,
                "skip_reason": "judge_error",
                "n_errors":   n_errors,
                "n_total":    len(outputs),
                "sample_error": judge_error or f"unparseable judge response (raw, first 300 chars): {(judge_raw or '')[:300]!r}",
            })
            continue

        score = float(result["consistency_score"])
        cai_strain = round(1.0 - score, 4)
        passed = score >= 0.75

        if verbose:
            bar = "OK" if score >= 0.75 else ("DRIFT" if score >= 0.4 else "FAIL")
            err_note = f", {n_errors} partial errors" if n_errors else ""
            print(f"strain={cai_strain:.3f} [{bar}]{err_note}", flush=True)

        all_scores.append(score)
        weighted_scores.append(score * weight)
        weighted_weights.append(weight)
        details.append({
            "id":         case["id"],
            "name":       name,
            "severity":   severity,
            "cai_score":  round(score, 4),
            "cai_strain": cai_strain,
            "passed":     passed,
            "n_errors":   n_errors,
            "disagreements": result.get("disagreements", []),
            "summary":    result.get("summary", ""),
        })

    avg_score  = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    avg_strain = round(1 - avg_score, 4) if avg_score is not None else None
    sw_score   = round(sum(weighted_scores) / sum(weighted_weights), 4) if weighted_weights else None
    sw_strain  = round(1 - sw_score, 4) if sw_score is not None else None
    n_skipped  = sum(1 for d in details if d.get("skipped"))

    return {
        "cai_score":              avg_score,
        "cai_strain":             avg_strain,
        "severity_weighted_cai":  sw_score,
        "severity_weighted_cts":  sw_strain,
        "passed":         sum(1 for d in details if d.get("passed")),
        "failed":         sum(1 for d in details if "passed" in d and not d["passed"]),
        "judged":         len(all_scores),
        "skipped_errors": n_skipped,
        "total":          len(details),
        "details": details,
    }


def _legacy_domain_is_contaminated(prev: dict) -> bool:
    """True if this saved domain result predates the API-error-exclusion fix
    (commit 45c544e) -- i.e. it has no 'skipped_errors' key, meaning nothing
    was excluded -- AND its per-case judge summaries show signs the judge was
    scoring API-error text as if it were real answers. Resume must not trust
    a domain like this as \"done\"; it needs to be re-run for real.
    """
    if "skipped_errors" in prev:
        return False  # post-fix schema: error cases are already excluded properly
    details = prev.get("details", [])
    if not details:
        return False
    flagged = sum(
        1 for c in details
        if "error" in ((c.get("summary", "") + " " + " ".join(c.get("disagreements", []))).lower())
    )
    return flagged / len(details) > 0.3


def run_model(name: str, cfg: dict, resume: bool = False, verbose: bool = True) -> dict:
    model_id = cfg["model_id"]
    safe_name = model_id.replace("/", "-")
    out_file  = RESULTS_DIR / f"{safe_name}_{date.today().isoformat()}.json"

    # Load partial results if resuming
    existing = {}
    if resume and out_file.exists():
        try:
            prev = json.loads(out_file.read_text())
            existing = prev.get("results", {})
            print(f"  resuming: {len(existing)} domains already done")
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  MODEL:  {model_id}  ({cfg['provider']})")
    print(f"  JUDGE:  {cfg['judge_model']}  ({cfg['judge_provider']})  [independent]")
    print(f"{'='*60}")

    results_by_domain = dict(existing)
    RESULTS_DIR.mkdir(exist_ok=True)

    for domain in DOMAINS:
        if domain in results_by_domain:
            prev = results_by_domain[domain]
            judged = prev.get("judged", prev.get("total", 0))
            contaminated = _legacy_domain_is_contaminated(prev)
            if "error" in prev or not judged or contaminated:
                reason = ("pre-fix run scored API-error text as consistency data"
                          if contaminated else "previously had no judged cases -- not real data")
                print(f"  RETRY {domain:<23} ({reason})")
            else:
                s = prev.get("cai_strain", "?")
                skipped = prev.get("skipped_errors", 0)
                note = f", {skipped} skipped for errors" if skipped else ""
                print(f"  SKIP {domain:<24} (done: strain={s}, {judged} judged{note})")
                continue
        try:
            res = run_domain(domain, cfg, verbose=verbose)
            results_by_domain[domain] = res
            # Save after each domain
            _save(out_file, model_id, cfg, results_by_domain)
        except Exception as e:
            print(f"  ERROR {domain}: {e}")
            results_by_domain[domain] = {"error": str(e)}
            _save(out_file, model_id, cfg, results_by_domain)

    _save(out_file, model_id, cfg, results_by_domain)
    print(f"\n  → saved: {out_file}")
    return results_by_domain


def _save(path: Path, model_id: str, cfg: dict, results: dict) -> None:
    scores = [r["cai_strain"] for r in results.values() if "cai_strain" in r and r["cai_strain"] is not None]
    avg = round(sum(scores) / len(scores), 4) if scores else None
    output = {
        "model":               model_id,
        "provider":            cfg["provider"],
        "date":                str(date.today()),
        "benchmark_version":   "v2",
        "mode":                "frozen",
        "judge_provider":      cfg["judge_provider"],
        "judge_model":         cfg["judge_model"],
        "independent_judging": True,
        "avg_cai_strain":      avg,
        "domains_complete":    len(scores),
        "results":             results,
    }
    path.write_text(json.dumps(output, indent=2))


# ── MAIN ─────────────────────────────────────────────────────────────────────

def test_connection() -> None:
    """One call per model/judge endpoint with the raw result printed. Run this
    before a full sweep after any run that came back mostly errors -- it tells
    you in seconds whether the problem is an invalid key, a decommissioned
    model, exhausted quota, or a genuine rate limit, instead of guessing from
    348 swallowed exceptions."""
    for name, cfg in MODELS.items():
        print(f"\n{name} -> model {cfg['model_id']} ({cfg['provider']})")
        try:
            out = _chat(cfg["base_url"], cfg["api_key"], cfg["model_id"], "Say OK.", retries=1)
            print(f"  OK: {out[:80]!r}")
        except Exception as e:
            print(f"  FAILED: {e}")
        print(f"{name} -> judge {cfg['judge_model']} ({cfg['judge_provider']})")
        try:
            out = _chat(cfg["judge_url"], cfg["judge_key"], cfg["judge_model"], "Say OK.", retries=1)
            print(f"  OK: {out[:80]!r}")
        except Exception as e:
            print(f"  FAILED: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["oss120b", "mistral", "both"], default="both")
    parser.add_argument("--resume", action="store_true", help="skip already-completed domains")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test-connection", action="store_true",
                         help="one call per endpoint, print raw success/failure, then exit")
    args = parser.parse_args()

    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        sys.exit("Run:  pip install openai")

    if args.test_connection:
        test_connection()
        return

    to_run = ["oss120b", "mistral"] if args.model == "both" else [args.model]
    for name in to_run:
        run_model(name, MODELS[name], resume=args.resume, verbose=not args.quiet)

    print("\nDone. Send the JSON files in results/ back to update the paper tables.\n")


if __name__ == "__main__":
    main()
