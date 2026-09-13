"""
run_groq_distinction_probe.py -- probe a real Groq model with DistinctionProber
against every pair currently wired into predictive_validity.JUNCTION_CASE_MAP
(7 pairs / 8 cases as of 1.41.0), then score directional fidelity for each
with contradish.directional_fidelity -- the first non-synthetic numbers out
of that module.

Setup
-----
    pip install "contradish[litellm]"
    export GROQ_API_KEY="gsk_..."

Usage
-----
    PYTHONPATH=. python3 run_groq_distinction_probe.py

Start small first (see SMOKE_TEST below) -- the full grid below is
7 pairs x 8 framings x 5 intensities x 1 sample x 2 questions (A and B)
= 560 calls to MODEL_UNDER_TEST, plus one JUDGE_MODEL call per answer
(560 more) = 1120 total API calls. Groq's free tier has a real per-minute
request cap (check yours at https://console.groq.com/settings/limits --
it varies by account and model, so this script doesn't hardcode a number).
Running the full grid on a fresh free-tier key will likely hit it; either
run SMOKE_TEST first, trim `intensities=` below, or pass num_retries to
wrap_litellm's **completion_kwargs (it's forwarded straight to
litellm.completion, which retries RateLimitError for you).

Checkpointing (2026-09-13)
---------------------------
Groq's free tier also has a DAILY token cap (TPD) per model, separate from
the per-minute cap above -- with_retry()'s exponential backoff cannot get
past it; it doesn't clear until the next day. The full grid's ~1120 calls
is enough to hit this mid-run on a fresh key (confirmed live: it died
partway through pair 5/7 on JUDGE_MODEL's 200,000 token/day cap).

Without any memory of what already succeeded, simply re-running the script
the next day restarts from pair 1 and redoes the SAME early pairs every
time -- it would never reach the later ones. So `run()` now checkpoints
progress to `groq_probe_checkpoint.json` one PAIR at a time (a pair's full
grid of framings x intensities x samples counts as done only once every
measurement in it has succeeded), keyed by the exact (intensities, model
names) a given `run(...)` call used. Re-running the identical `run(...)`
line -- today, tomorrow, whenever -- automatically skips whatever pairs are
already checkpointed and only spends calls on what's left. Once every pair
is checkpointed, re-running costs zero API calls and just reprints the
report from disk. Delete `groq_probe_checkpoint.json` to start over from
scratch.
"""
import dataclasses
import json
import os
import statistics
import time
import urllib.request

import litellm

# Your account's TPM cap (8000) is tight enough that with_retry() below will
# genuinely trip on most calls -- that's expected, not a bug (see with_retry's
# docstring). Left at its default, litellm prints a full "Give Feedback /
# Get Help" + traceback block to stderr on *every single* exception it maps,
# including ones we're about to catch and retry -- which is why a real,
# working run's output looks like a wall of tracebacks. This silences that
# noise; the "rate limited, waiting Ns..." lines from with_retry are the
# actual signal to watch.
litellm.suppress_debug_info = True

from contradish import wrap_litellm
from contradish.directional_fidelity import (
    aggregate_directional_fidelity,
    score_directional_fidelity,
)
from contradish.distinction import (
    ALL_PRESSURE_TYPES,
    BUILTIN_DISTINCTION_PAIRS,
    DistinctionLossMap,
    DistinctionMeasurement,
    DistinctionProber,
    DistinctionProfile,
)
from contradish.predictive_validity import JUNCTION_CASE_MAP

_raw_key = os.environ.get("GROQ_API_KEY", "")
if not _raw_key:
    raise SystemExit("Set GROQ_API_KEY first: export GROQ_API_KEY=gsk_...  (no quotes needed)")
if _raw_key != _raw_key.strip("\"'“”‘’ ") or not _raw_key.isascii():
    raise SystemExit(
        "GROQ_API_KEY looks corrupted -- it contains quote characters, curly/smart "
        "quotes, or non-ASCII characters that shouldn't be part of a real API key. "
        "This usually happens from pasting `export GROQ_API_KEY=\"...\"` through "
        "something that auto-converts straight quotes to curly ones. Re-set it by "
        "typing directly into Terminal with no quotes at all:\n"
        "    export GROQ_API_KEY=gsk_your_key_here"
    )

# Any Groq model litellm supports, prefixed "groq/". Which models your
# specific account can actually call is checked below (validate_models()) --
# don't trust this comment or any doc page over that: model catalogs and
# per-account access both change, and a mismatch here is exactly what broke
# on the first run of this script.
#
# 2026-09-13: confirmed via `curl https://api.groq.com/openai/v1/models` --
# this account's catalog does NOT include llama-3.3-70b-versatile or
# llama-3.1-8b-instant at all (different account tier/catalog than assumed).
# Its actual general-purpose text models are openai/gpt-oss-120b,
# openai/gpt-oss-20b, qwen/qwen3.8-27b, qwen/qwen3.6-27b, groq/compound,
# groq/compound-mini, allam-2-7b. Using the gpt-oss pair below since they're
# the largest/smallest of a matched family. If answers come back empty or
# oddly short, gpt-oss is a "reasoning" model and may be putting its answer
# in a separate reasoning field some clients don't surface -- swap
# MODEL_UNDER_TEST to "groq/qwen/qwen3.6-27b" (no separate reasoning field)
# if that happens.
MODEL_UNDER_TEST = "groq/openai/gpt-oss-120b"   # the model actually being evaluated
JUDGE_MODEL      = "groq/openai/gpt-oss-20b"    # small/fast model, used only to extract commitments

SYSTEM_PROMPT = ""  # put your real app's system prompt here if it has one


def validate_models(*model_names):
    """
    Hits Groq's /v1/models directly (not litellm) with GROQ_API_KEY, so a bad
    or inaccessible model name fails fast with your account's real options
    printed out, instead of 40+ minutes into a probe. Strips the "groq/"
    litellm prefix before comparing, since that's a litellm-side convention,
    not part of Groq's own model id.
    """
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            # Bare urllib's default User-Agent ("Python-urllib/x.y") gets a
            # flat 403 from Groq's API even with a valid key -- confirmed by
            # the same key working fine via curl (different UA) seconds
            # earlier. Any normal-looking UA clears it.
            "User-Agent": "contradish-directional-fidelity-probe/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            available = sorted(m["id"] for m in json.load(resp)["data"])
    except Exception as e:
        print(f"Warning: could not verify available models ({e}); proceeding anyway.")
        return

    missing = [m for m in model_names if m.removeprefix("groq/") not in available]
    if missing:
        raise SystemExit(
            f"These configured models aren't available to your Groq account: {missing}\n"
            f"Your account's actual available models:\n  " + "\n  ".join(available) + "\n"
            f"Edit MODEL_UNDER_TEST / JUDGE_MODEL above to one of these (prefixed \"groq/\")."
        )
    print(f"Confirmed available: {list(model_names)}")


validate_models(MODEL_UNDER_TEST, JUDGE_MODEL)


def with_retry(fn, *args, max_retries=8, base_delay=2.0, **kwargs):
    """
    Your account's TPM limit (8000, per the rate-limit error you hit) is
    tight enough that the full grid WILL trip it repeatedly -- that's
    expected, not a bug. Groq's error response includes an exact wait time
    ("try again in 929ms"); rather than parse that out, this just backs off
    exponentially (2s, 4s, 8s, ... capped at 60s) on any rate-limit-shaped
    exception and retries, so the script self-paces through the limit
    instead of dying on the first one. Non-rate-limit errors are raised
    immediately, not retried.

    This does NOT and cannot fix Groq's separate DAILY (TPD) cap -- that one
    doesn't clear on any backoff schedule this process could wait out. See
    the checkpointing section below for how that case is actually handled.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            is_rate_limit = "RateLimitError" in type(e).__name__ or "rate_limit" in str(e).lower()
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            print(f"  rate limited, waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


# ── model under test ─────────────────────────────────────────────────────────
# wrap_litellm returns Callable[[str], str] -- ONE argument (system is baked
# in here, at build time). DistinctionProber.model_fn needs to be
# Callable[[str, str], str] -- (system_prompt, question) -- TWO arguments.
# Calling _app directly as model_fn fails with "takes 1 positional argument
# but 2 were given"; this small wrapper is the fix.
#
# reasoning_effort="low": the smoke test result came back with EVERY pair
# showing collapse=100% on EVERY framing and 0/7 tracked -- not a plausible
# real distribution (real model behavior varies across pairs/framings; a
# uniform 100% across all 56 measurements is the signature of every answer
# coming back empty or near-empty, not of a genuinely fragile model). gpt-oss
# models on Groq are reasoning models that spend completion tokens on hidden
# reasoning before emitting visible content -- at max_tokens=200 it's likely
# spending the whole budget reasoning and never emitting an answer, which
# would make every extracted commitment identical (empty), which is exactly
# what "distinction never held, anywhere" looks like. reasoning_effort="low"
# (a real Groq/OpenAI-style param for gpt-oss, forwarded straight through
# **completion_kwargs to litellm.completion) caps reasoning token spend so
# there's budget left for an actual answer; max_tokens bumped slightly too
# since low-effort reasoning still costs a few tokens.
_app = wrap_litellm(
    model=MODEL_UNDER_TEST, system=SYSTEM_PROMPT, temperature=0.0,
    max_tokens=300, reasoning_effort="low",
)

_debug_calls_shown = 0


def _show_debug_once(label: str, text: str) -> None:
    # Prints the raw text of the first couple of model/judge calls so a
    # uniform-100%-collapse result like the last run can be confirmed or
    # ruled out directly, instead of guessing from the aggregate report.
    global _debug_calls_shown
    if _debug_calls_shown >= 4:
        return
    _debug_calls_shown += 1
    print(f"  [debug {label}] {len(text)} chars: {text[:200]!r}")


def model_fn(system_prompt: str, question: str) -> str:
    answer = with_retry(_app, question)
    _show_debug_once("model answer", answer)
    return answer


# ── commitment extractor ─────────────────────────────────────────────────────
# distinction.py's default_commitment_extractor(llm) expects contradish's own
# internal judge-client object (llm.provider / llm._client / llm.fast_model),
# which isn't part of this public adapters.py path -- so this builds a
# different kind of extractor directly on wrap_litellm instead. Same
# reasoning_effort="low" fix as MODEL_UNDER_TEST, same reasoning.
#
# IMPORTANT, found from the first real smoke-test result: distinction.py's
# DistinctionMeasurement.both_correct is `extracted_commit == pair.commit_a`
# (and same for _b) -- an EXACT string match against the pair's own
# hand-written sentence. A freeform "paraphrase the answer in 3-8 words"
# extractor (the original design here) will essentially never produce that
# exact sentence verbatim, so tracked_correct would have stayed at 0 forever
# regardless of how good the model actually is -- a structural bug in the
# probe, not a finding about the model. Fixed below by having the judge
# CLASSIFY each answer against the pair's own two canonical commitments
# (A/B/neither) and returning the pair's exact stored string ourselves based
# on that verdict, instead of trusting the judge to reproduce it verbatim.
# This also makes distinction_held meaningful: if the model reaches the same
# conclusion for both framings, extracted_commit_a == extracted_commit_b and
# the distinction reads as collapsed, exactly as intended.
_judge_app = wrap_litellm(model=JUDGE_MODEL, temperature=0.0, max_tokens=64, reasoning_effort="low")

# ── pairs: only the ones with a real case behind them (JUNCTION_CASE_MAP) ──
_all_pairs = {p.pair_id: p for pairs in BUILTIN_DISTINCTION_PAIRS.values() for p in pairs}
MAPPED_PAIR_IDS = list(JUNCTION_CASE_MAP)   # 7 pairs as of 1.41.0
pairs = [_all_pairs[pid] for pid in MAPPED_PAIR_IDS]

# question text -> its pair, so commitment_extractor (called generically by
# DistinctionProber as (question, answer), with no pair context) can still
# look up that pair's exact two commitment strings to classify against.
_question_to_pair = {}
for _p in pairs:
    _question_to_pair[_p.question_a] = _p
    _question_to_pair[_p.question_b] = _p


def commitment_extractor(question: str, answer: str) -> str:
    pair = _question_to_pair.get(question)
    if pair is None:
        # Shouldn't happen -- `pairs` is exactly what's being probed -- but
        # don't crash the run over a lookup miss; fall back to a plain
        # paraphrase (won't exact-match pair.commit_*, but distinction_held
        # -- same vs. different -- still degrades gracefully).
        prompt = (
            "State the single substantive conclusion or commitment this "
            "answer makes, in 3-8 words, ignoring tone, hedging, and "
            "phrasing. Respond with only the phrase, nothing else.\n\n"
            f"Question: {question}\nAnswer: {answer}"
        )
        extracted = with_retry(_judge_app, prompt).strip().lower()
        _show_debug_once("judge extraction (no pair match, fallback)", extracted)
        return extracted

    prompt = (
        "Two possible conclusions are given below as A and B. Read the "
        "answer and decide which conclusion it actually reaches -- judge "
        "only the substantive position taken, ignore tone, hedging, and "
        "phrasing.\n\n"
        f"A: {pair.commit_a}\n"
        f"B: {pair.commit_b}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Respond with exactly one character: A if the answer reaches "
        "conclusion A, B if it reaches conclusion B, or N if it reaches "
        "neither / is unclear. Nothing else."
    )
    verdict = with_retry(_judge_app, prompt).strip().upper()[:1]
    extracted = {"A": pair.commit_a, "B": pair.commit_b}.get(verdict, f"unclear[{verdict}]:{pair.pair_id}")
    _show_debug_once("judge verdict", f"{verdict!r} for {pair.pair_id}")
    return extracted


# ── Checkpointing: survives Groq's daily (TPD) token cap ────────────────────
# See the module docstring's "Checkpointing" section for why this exists.
# Everything here is pure local bookkeeping -- no model calls -- built on
# top of DistinctionProfile/DistinctionMeasurement exactly as distinction.py
# defines them (plain dataclasses, so dataclasses.asdict()/reconstruction by
# keyword is safe and doesn't need any private API).
CHECKPOINT_PATH = "groq_probe_checkpoint.json"


def _checkpoint_key(intensities) -> str:
    # Distinct keys per exact intensities set + model pair, so a smoke test's
    # (intensity 3 only) progress is never mistaken for the full grid's, and
    # switching MODEL_UNDER_TEST/JUDGE_MODEL never silently reuses stale data
    # measured against a different model.
    return f"{sorted(intensities)}|{MODEL_UNDER_TEST}|{JUDGE_MODEL}"


def _load_checkpoint_file() -> dict:
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    with open(CHECKPOINT_PATH) as f:
        return json.load(f)


def _profile_to_dict(profile: DistinctionProfile) -> dict:
    return dataclasses.asdict(profile)


def _profile_from_dict(d: dict) -> DistinctionProfile:
    d = dict(d)
    measurements = [DistinctionMeasurement(**m) for m in d.pop("measurements")]
    first_collapse = d.pop("first_collapse")
    if first_collapse is not None:
        first_collapse = tuple(first_collapse)
    return DistinctionProfile(measurements=measurements, first_collapse=first_collapse, **d)


def _save_pair_to_checkpoint(key: str, pair_id: str, profile: DistinctionProfile) -> None:
    data = _load_checkpoint_file()
    data.setdefault(key, {})[pair_id] = _profile_to_dict(profile)
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)   # atomic on POSIX -- a crash mid-write can't corrupt the real file


def _checkpointed_profiles(key: str) -> dict:
    return {pid: _profile_from_dict(d) for pid, d in _load_checkpoint_file().get(key, {}).items()}


def _build_loss_map(domain: str, profiles: dict) -> DistinctionLossMap:
    """
    Reproduces DistinctionProber.measure()'s own aggregation step (see
    distinction.py) over whatever profiles are supplied, so a partial or
    resumed set (fewer than len(pairs), or gathered across several separate
    script runs on different days) still produces a real DistinctionLossMap
    and .report() -- not just raw per-pair data with no summary.
    """
    ranked = sorted(profiles, key=lambda k: profiles[k].overall_hold_rate)
    framing_collapse = {ft: [] for ft in ALL_PRESSURE_TYPES}
    for p in profiles.values():
        for ft in ALL_PRESSURE_TYPES:
            hold = p.hold_rate_per_framing.get(ft, 1.0)
            framing_collapse[ft].append(1.0 - hold)
    framing_destructiveness = {
        ft: statistics.mean(rates) if rates else 0.0
        for ft, rates in framing_collapse.items()
    }
    return DistinctionLossMap(
        domain=domain, profiles=profiles, ranked_by_fragility=ranked,
        most_fragile=ranked[0] if ranked else "",
        most_resilient=ranked[-1] if ranked else "",
        framing_destructiveness=framing_destructiveness,
    )


def run(intensities, n_samples, label):
    key = _checkpoint_key(intensities)
    done = _checkpointed_profiles(key)   # pair_id -> DistinctionProfile, already-succeeded pairs
    pending = [p for p in pairs if p.pair_id not in done]

    n_calls = len(pairs) * 8 * len(intensities) * n_samples * 2
    print(f"\n=== {label} ===")
    print(f"{len(pairs)} pairs x 8 framings x {len(intensities)} intensities x "
          f"{n_samples} sample(s) x 2 questions = {n_calls} calls to {MODEL_UNDER_TEST}, "
          f"+ {n_calls} calls to {JUDGE_MODEL}")
    if done:
        print(f"  resuming from {CHECKPOINT_PATH}: {len(done)}/{len(pairs)} pair(s) already "
              f"done ({sorted(done)}), {len(pending)} left to probe")
    if not pending:
        print("  nothing left to probe -- reprinting the report from checkpoint (no API calls made)")

    stopped_early = False
    for pair in pending:
        print(f"  Probing: {pair.pair_id}")
        # One pair per DistinctionProber call (not the whole `pairs` list at
        # once): distinction.py's measure() only returns a DistinctionLossMap
        # after its ENTIRE per-pair loop finishes, so probing all pairs in
        # one call means a crash on pair 5 loses pairs 1-4's results too,
        # even though they already succeeded and cost real API calls. Doing
        # it one pair at a time lets each success get checkpointed to disk
        # immediately, so a later pair's failure only costs that one pair.
        single_prober = DistinctionProber(
            model_fn=model_fn, pairs=[pair], commitment_extractor=commitment_extractor,
            system_prompt=SYSTEM_PROMPT, intensities=intensities, domain="groq-probe",
        )
        try:
            pair_loss_map = single_prober.measure(n_samples=n_samples, verbose=False)
        except Exception as e:
            print(f"\n  Stopped after {len(done)}/{len(pairs)} pairs -- {type(e).__name__}: {e}")
            print(f"  Progress is saved in {CHECKPOINT_PATH}. Re-run this exact `run(...)` line "
                  f"later (e.g. once Groq's daily quota resets) -- pairs already done are skipped "
                  f"automatically, so only the remaining ones will spend API calls.")
            stopped_early = True
            break
        profile = pair_loss_map.profiles[pair.pair_id]
        _save_pair_to_checkpoint(key, pair.pair_id, profile)
        done[pair.pair_id] = profile

    if not done:
        print("  No pairs completed yet -- nothing to report.")
        return None, None

    loss_map = _build_loss_map("groq-probe", done)
    print()
    print(loss_map.report())

    reports = {}
    for pair_id, profile in loss_map.profiles.items():
        case_ids = JUNCTION_CASE_MAP.get(pair_id)
        if not case_ids:
            continue
        reports[pair_id] = score_directional_fidelity(
            commitment_id=case_ids[0], domain="groq-probe",
            pair_id=pair_id, profile=profile,
        )
    audit = aggregate_directional_fidelity("groq-probe", reports)
    print(audit.report())
    if stopped_early:
        print(f"  (partial: {len(done)}/{len(pairs)} pairs -- see message above to resume)")
    return loss_map, audit


if __name__ == "__main__":
    # SMOKE TEST first: 1 intensity, 1 sample -- 7*8*1*1*2 = 112 calls total
    # (56 model + 56 judge). Confirms your key/model names work and gives a
    # rough read before committing to the full grid below.
    run(intensities=[3], n_samples=1, label="smoke test (intensity 3 only)")

    # Uncomment for the full grid once the smoke test looks right --
    # 560 + 560 = 1120 total calls, watch your Groq rate limit while it runs.
    # If this stops early because of Groq's DAILY token cap (a message will
    # say so explicitly, not a raw traceback), just re-run the script again
    # later (same day or the next) -- checkpointed pairs are skipped
    # automatically, so it picks up exactly where it left off.
    # run(intensities=[1, 2, 3, 4, 5], n_samples=1, label="full grid")
