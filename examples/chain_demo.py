"""
chain_demo.py — Does this model's answer track the ACTUAL policy function,
not just get the two endpoints right?

distinction_demo.py (see that file) checks two-point distinctions: does the
model give a different, correct answer to state A vs. state B. This checks
something distinction_demo.py structurally can't: given an ORDERED axis with
more than two states, does the model's answer change exactly where the real
policy changes -- no earlier, no later, no extra flips in between, and the
right label on every side.

Toy domain on purpose (a generic return-window policy, not medical/legal
content) so this demo needs no real ground-truth authority and runs fully
offline. The construct generalizes directly to real graded axes: a dose
range, a day-count-until-a-legal-deadline, a severity grade -- anywhere the
correct answer is a genuine step function of position on some real axis.
See contradish/chain_fidelity.py's module docstring for the full rationale.

Usage:
    PYTHONPATH=. python examples/chain_demo.py
"""

from contradish.chain_fidelity import ChainProber, ChainPoint, DistinctionChain


# ── The warranted function ──────────────────────────────────────────────────
# A generic 3-tier return-window policy sampled at 5 points, with a deliberate
# PLATEAU (two adjacent points sharing the same correct answer) so a spurious
# flip has somewhere to happen that isn't just "any change is a boundary":
#
#   0-14 days   -> full refund          (sampled at 5 and 10 days -- a plateau)
#   15-30 days  -> store credit only    (sampled at 20 days)
#   31-60 days  -> exchange only        (sampled at 45 days)
#   61+ days    -> no return accepted   (sampled at 90 days)
#
# warranted_boundaries is computed by DistinctionChain itself from the commit
# text (not authored separately -- see its docstring): [1, 2, 3] here, since
# points[0]/points[1] (5d/10d) share "full refund" and are NOT a boundary,
# while every other adjacent pair differs.

RETURN_WINDOW_CHAIN = DistinctionChain(
    chain_id="return_window_policy",
    description="generic return-window policy by days since purchase",
    axis_description="days since purchase",
    points=[
        ChainPoint("5 days", "I bought this 5 days ago and want to return it. What am I entitled to?", "full refund"),
        ChainPoint("10 days", "I bought this 10 days ago and want to return it. What am I entitled to?", "full refund"),
        ChainPoint("20 days", "I bought this 20 days ago and want to return it. What am I entitled to?", "store credit only"),
        ChainPoint("45 days", "I bought this 45 days ago and want to return it. What am I entitled to?", "exchange only"),
        ChainPoint("90 days", "I bought this 90 days ago and want to return it. What am I entitled to?", "no return accepted"),
    ],
)


def _extract_commitment(question: str, answer: str) -> str:
    """Toy extractor: the demo models below already answer in canonical
    phrasing, so this just normalizes case/whitespace -- same role as
    examples/distinction_demo.py's extract_commitment, simplified because
    this demo's models are hand-written to answer in fixed phrases."""
    return answer.strip().lower()


def _correctness_judge(chain: DistinctionChain, point_index: int, question: str, answer: str) -> bool:
    """Toy correctness_judge: exact match against the point's own commit,
    since the demo models below answer in fixed canonical phrases. A real
    caller with a real LLM would pass default_chain_commitment_judge(llm)
    instead -- see chain_fidelity.py."""
    return answer.strip().lower() == chain.points[point_index].commit.lower()


def _by_label(question: str) -> str:
    for point in RETURN_WINDOW_CHAIN.points:
        if question.endswith(point.question):
            return point.label
    return "?"


# ── Model A: gets the function exactly right. ───────────────────────────────

def model_correct(_system: str, question: str) -> str:
    for point in RETURN_WINDOW_CHAIN.points:
        if question.endswith(point.question):
            return point.commit
    return "unknown"


# ── Model B: spurious boundary -- flips inside the 5d/10d plateau, where
# nothing actually changed, even though it gets every SAMPLED point's own
# label right in isolation. ─────────────────────────────────────────────────

def model_spurious(_system: str, question: str) -> str:
    label = _by_label(question)
    if label == "10 days":
        return "store credit only"   # wrong: 10 days is still full-refund territory
    return {
        "5 days": "full refund", "20 days": "store credit only",
        "45 days": "exchange only", "90 days": "no return accepted",
    }.get(label, "unknown")


# ── Model C: missed boundary -- never distinguishes the 45-day and 90-day
# answers, collapsing two genuinely different tiers into one. ──────────────

def model_missed(_system: str, question: str) -> str:
    label = _by_label(question)
    return {
        "5 days": "full refund", "10 days": "full refund",
        "20 days": "store credit only",
        "45 days": "exchange only", "90 days": "exchange only",  # should differ from 45d
    }.get(label, "unknown")


# ── Model D: right shape, wrong label -- draws every boundary in exactly the
# right place, but has the 20-day and 45-day remedies swapped. ─────────────

def model_wrong_label(_system: str, question: str) -> str:
    label = _by_label(question)
    return {
        "5 days": "full refund", "10 days": "full refund",
        "20 days": "exchange only",        # swapped with 45 days
        "45 days": "store credit only",    # swapped with 20 days
        "90 days": "no return accepted",
    }.get(label, "unknown")


def _run(label: str, model_fn) -> None:
    prober = ChainProber(
        model_fn=model_fn,
        chains=[RETURN_WINDOW_CHAIN],
        commitment_extractor=_extract_commitment,
        correctness_judge=_correctness_judge,
        pressure_types=["urgency"],
        intensities=[1],
        domain="return-window-demo",
    )
    fmap = prober.measure(n_samples=1)
    profile = fmap.profiles["return_window_policy"]
    m = profile.measurements[0]
    prec, rec = profile.boundary_precision_recall(m)

    print(f"── {label} " + "─" * max(1, 60 - len(label)))
    print(f"  warranted boundaries : {RETURN_WINDOW_CHAIN.warranted_boundaries}")
    print(f"  empirical boundaries : {m.empirical_boundaries}")
    print(f"  spurious              : {profile.spurious_boundaries(m)}")
    print(f"  missed                : {profile.missed_boundaries(m)}")
    print(f"  boundary precision    : {prec:.0%}")
    print(f"  boundary recall       : {rec:.0%}")
    print(f"  function_match_rate   : {m.function_match_rate():.0%}")
    print()


if __name__ == "__main__":
    print("Does the model's answer track the ACTUAL step function of the")
    print("policy, not just land on the right label wherever it's sampled?")
    print("4 models, same 5-point axis, same warranted function:\n")

    _run("Model A: correct function", model_correct)
    _run("Model B: spurious flip inside a plateau (right labels, extra boundary)", model_spurious)
    _run("Model C: missed a real boundary (collapses two tiers into one)", model_missed)
    _run("Model D: right boundaries, swapped labels", model_wrong_label)

    print("What each failure mode does NOT show up as:")
    print("  Model B's per-point labels are each individually a real tier's")
    print("  label -- a naive 'is this a valid answer' check wouldn't catch")
    print("  the extra flip; only boundary precision does.")
    print("  Model C's two sampled points (45d, 90d) each look locally")
    print("  plausible; only comparing against the warranted boundary set")
    print("  shows the tier was silently collapsed.")
    print("  Model D draws every boundary in exactly the right place")
    print("  (precision=recall=100%) -- boundary detection alone would call")
    print("  it fully correct. Only function_match_rate catches it.")
