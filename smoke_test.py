"""
smoke_test.py -- validate one model + judge combo before committing to a
full run_benchmark_full.py pass (which can take hours across 20 domains).

Makes 2-3 real API calls total (1 direct call to the model under test, then
one real contradish case -- 1 paraphrase -- through the actual Suite path
run_benchmark_full.py uses, including the judge). Takes seconds, not hours,
and catches auth/quota/wiring problems before you burn a real run on them.

Usage
-----
    export GOOGLE_API_KEY=...
    export OPENAI_API_KEY=...     # judge, per JUDGE_MAP in run_benchmark_full.py
    python3 smoke_test.py gemini/gemini-1.5-pro --judge-provider openai

    export GROQ_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python3 smoke_test.py groq/llama-3.1-70b-versatile --key-env GROQ_API_KEY --judge-provider anthropic
"""
import argparse
import os
import sys


def _guess_key_env(model: str) -> str:
    if "gemini" in model:
        return "GOOGLE_API_KEY"
    if "mistral" in model:
        return "MISTRAL_API_KEY"
    if "groq" in model or "llama" in model:
        return "GROQ_API_KEY"
    if "together" in model:
        return "TOGETHER_API_KEY"
    return "OPENAI_API_KEY"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="LiteLLM model string, e.g. gemini/gemini-1.5-pro")
    p.add_argument("--key-env", default=None,
                   help="Env var holding the model's own API key (default: guessed from the model string)")
    p.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic"],
                   help="Which cross-provider judge to use, per JUDGE_MAP in run_benchmark_full.py "
                        "(needs OPENAI_API_KEY or ANTHROPIC_API_KEY set to match)")
    args = p.parse_args()

    key_env = args.key_env or _guess_key_env(args.model)
    key = os.environ.get(key_env)
    if not key:
        sys.exit(f"Set {key_env} first -- that's the model's own API key.")

    judge_key_env = "OPENAI_API_KEY" if args.judge_provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(judge_key_env):
        sys.exit(f"Set {judge_key_env} first -- that's the judge's API key (--judge-provider {args.judge_provider}).")

    from contradish.adapters import wrap_litellm
    from contradish import Suite, TestCase

    print(f"1/3  calling {args.model} directly ...")
    app = wrap_litellm(model=args.model, api_key=key)
    try:
        out = app("Say OK and nothing else.")
    except Exception as e:
        sys.exit(f"FAILED calling {args.model}: {e}")
    print(f"     -> {out!r}")

    print("2/3  running one real contradish case (1 paraphrase) end-to-end, judge included ...")
    suite = Suite(app=app, provider=args.judge_provider)
    suite.add(TestCase(input="Can I get a refund after 45 days?", name="smoke test"))
    report = suite.run(paraphrases=1, verbose=False)
    result = report.results[0]

    print(f"3/3  result: skipped={result.skipped}  cai_strain={result.cai_strain}")
    if result.skipped:
        sys.exit(
            f"FAILED: the case was skipped ({result.skip_reason}, {result.n_errors} app-call errors) -- "
            f"fix this before running the full benchmark, it will hit the same problem 360 more times."
        )

    print("\nOK -- model and judge are both reachable through the real contradish path used by "
          "run_benchmark_full.py. Safe to run the full benchmark.")


if __name__ == "__main__":
    main()
