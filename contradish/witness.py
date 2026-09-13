"""
contradish/witness.py -- Multi-witness convergence: no serious finding rests on
one witness.

─────────────────────────────────────────────────────────────────────────────
PRINCIPLE (user-specified, 2026-09-13)
─────────────────────────────────────────────────────────────────────────────
"Serious claims should never rest on a single witness. Do not let one judge
model declare that a distinction was preserved or lost. For important
findings, require independent evidence from multiple methods. A Contradish
finding becomes much stronger when several independent measurements
converge."

Every judge call this repo makes -- default_restatement_judge,
default_commitment_extractor, default_hedge_judge (sacrifice.py), the Judge
class's consistency scoring -- already carries an explicit disclaimer that it
"inherits the judge's own noise." That disclaimer has, until now, been the
end of the conversation: callers were told to write their own judge if they
needed something more trustworthy, but nothing in the package made it easy
to actually COMBINE independent judges and require them to agree before a
finding counts. This module is that missing piece.

WitnessPanel is deliberately generic -- not tied to hedge_judge, KBV,
sacrifice, or any other specific measurement. It wraps N>=2 callables of
identical signature and produces a single combined callable of the same
signature, so it drops into any judge/classifier slot in this package
(hedge_judge, restatement_judge, usage_judge, commitment_extractor, or a
caller's own classifier) without that call site needing to know convergence
is happening.

─────────────────────────────────────────────────────────────────────────────
WHAT "INDEPENDENT" DOES AND DOESN'T MEAN HERE
─────────────────────────────────────────────────────────────────────────────
WitnessPanel enforces agreement, not independence -- it has no way to verify
that two witnesses are actually independent. Two calls to the SAME judge
model with the SAME prompt are not independent witnesses: they share
whatever blind spot or systematic bias that model has, and will tend to
agree with each other for the wrong reasons just as often as the right ones.
Genuine independence comes from the caller's choice of witnesses: different
model providers, a model judge paired with a programmatic/heuristic
classifier, a restatement-based check paired with a behavioral check. This
module records and reports the disagreement rate so that choice can be
audited after the fact (a panel whose members never disagree is a panel
that isn't actually independent, whatever its members claim to be) -- but it
cannot make a bad choice of witnesses good.

─────────────────────────────────────────────────────────────────────────────
POLARITY: DISAGREEMENT DEFAULTS TO THE CONSERVATIVE ANSWER
─────────────────────────────────────────────────────────────────────────────
By construction, requiring convergence can only make a finding HARDER to
claim, never easier: on disagreement, `combine()` returns the caller-supplied
`default_on_disagreement` (default: False), which every call site in this
package treats as "the failure condition was not confirmed" (e.g. hedge_judge
False = hedged = not a sacrifice instance; usage_judge False = didn't use the
claim = not a provenance-collapse instance). This is a one-way ratchet: a
multi-witness panel can only shrink an already-measured rate toward zero
relative to any single witness, never inflate it.

Usage::

    from contradish.witness import WitnessPanel
    from contradish.sacrifice import default_hedge_judge, measure_sacrifice

    panel = WitnessPanel(witnesses={
        "anthropic_judge": default_hedge_judge(anthropic_llm),
        "openai_judge":    default_hedge_judge(openai_llm),
    })
    combined_hedge_judge = panel.combine()
    sacrifice_report = measure_sacrifice(pairs, loss_map, kbv_report,
                                          hedge_judge=combined_hedge_judge)
    print(panel.convergence_report().summary())
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class WitnessCall:
    """One combined-judge invocation: what each witness said, and whether they agreed."""
    call_args_repr:    str            # str(args)/str(kwargs), truncated -- for inspection, not identity
    votes:             dict[str, Any] # witness_name -> that witness's verdict
    agreed:            bool
    combined_verdict:  Any

    def to_dict(self) -> dict:
        return {
            "call_args_repr": self.call_args_repr,
            "votes": self.votes,
            "agreed": self.agreed,
            "combined_verdict": self.combined_verdict,
        }


@dataclass
class ConvergenceReport:
    """
    Summary of every call a WitnessPanel's combined judge has made so far.

    disagreement_rate near 0 across MANY calls, for witnesses that are
    genuinely different methods, is itself a finding worth reporting --
    it's evidence the underlying signal is robust to how it's measured,
    not an artifact of one method's idiosyncrasies.
    """
    witness_names:      list[str]
    n_calls:            int
    n_agreed:           int
    n_disagreed:        int
    disagreement_rate:  float
    disagreements:      list[WitnessCall] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.n_calls} combined-judge call(s) across witnesses {self.witness_names}  *  "
            f"{self.n_disagreed} disagreement(s) ({self.disagreement_rate:.0%})"
        )

    def to_dict(self) -> dict:
        return {
            "witness_names": self.witness_names,
            "n_calls": self.n_calls,
            "n_agreed": self.n_agreed,
            "n_disagreed": self.n_disagreed,
            "disagreement_rate": self.disagreement_rate,
            "disagreements": [d.to_dict() for d in self.disagreements],
        }


class WitnessPanel:
    """
    Wraps >=2 independent classifier functions of identical signature into one
    combined function requiring their agreement.

    Parameters
    ----------
    witnesses
        name -> callable, all with the same call signature. Every witness is
        called on every invocation of the combined function (no short-
        circuiting), so the convergence report is always complete.
    """

    def __init__(self, witnesses: dict[str, Callable[..., Any]]):
        if len(witnesses) < 2:
            raise ValueError(
                "WitnessPanel requires at least 2 witnesses -- a single "
                "witness is exactly the practice this module exists to stop."
            )
        self.witnesses = dict(witnesses)
        self._calls: list[WitnessCall] = []

    def combine(
        self,
        default_on_disagreement: Any = False,
        require_unanimous: bool = True,
    ) -> Callable[..., Any]:
        """
        Return a single callable with the same signature as the wrapped
        witnesses.

        require_unanimous=True (default, recommended for any finding this
        package would report as confirmed): every witness must return the
        same verdict, or the result is `default_on_disagreement` and the
        call is logged as a disagreement.

        require_unanimous=False: majority vote among >2 witnesses decides
        the verdict; a call is still logged as a disagreement whenever the
        vote isn't unanimous, even though the majority verdict is returned
        (not `default_on_disagreement`) -- use this only when you have >=3
        witnesses and have decided majority-rules is the right bar for your
        use case, which is a weaker bar than the module's default.
        """
        def combined(*args, **kwargs) -> Any:
            votes = {name: fn(*args, **kwargs) for name, fn in self.witnesses.items()}
            values = list(votes.values())

            if require_unanimous:
                agreed = all(v == values[0] for v in values)
                verdict = values[0] if agreed else default_on_disagreement
            else:
                tally = Counter(values)
                winner, count = tally.most_common(1)[0]
                agreed = count == len(values)
                verdict = winner

            call_repr = f"args={args!r} kwargs={kwargs!r}"
            if len(call_repr) > 200:
                call_repr = call_repr[:200] + "...(truncated)"

            self._calls.append(WitnessCall(
                call_args_repr=call_repr, votes=votes,
                agreed=agreed, combined_verdict=verdict,
            ))
            return verdict
        return combined

    def convergence_report(self) -> ConvergenceReport:
        n = len(self._calls)
        n_agreed = sum(1 for c in self._calls if c.agreed)
        disagreements = [c for c in self._calls if not c.agreed]
        return ConvergenceReport(
            witness_names=list(self.witnesses),
            n_calls=n,
            n_agreed=n_agreed,
            n_disagreed=len(disagreements),
            disagreement_rate=round(len(disagreements) / n, 4) if n else 0.0,
            disagreements=disagreements,
        )

    def reset(self) -> None:
        """Clear recorded calls (e.g. between independent study runs reusing the same panel)."""
        self._calls = []

    @property
    def calls(self) -> list[WitnessCall]:
        """
        Read-only view of every call recorded so far, in order. Exposed so
        callers that need per-item detail beyond the aggregate
        ConvergenceReport (e.g. contradish/ground_truth_audit.py, which
        needs to know exactly which item each call corresponds to) can walk
        the same records convergence_report() summarizes, without reaching
        into a private attribute.
        """
        return list(self._calls)


def build_witnessed(
    judge_factory: Callable[[Any], Callable[..., Any]],
    llms: list,
    names: Optional[list[str]] = None,
    default_on_disagreement: Any = False,
    require_unanimous: bool = True,
) -> "tuple[Callable[..., Any], WitnessPanel]":
    """
    Convenience constructor: turn a single-LLM judge factory (the pattern
    every default_* judge in this package follows -- default_hedge_judge(llm),
    default_restatement_judge(llm), default_usage_judge(llm),
    default_commitment_extractor(llm)) into a witnessed judge in one call,
    instead of hand-building a WitnessPanel every time.

    This exists because the honest default should be the easy path. Every
    default_*_judge(llm) in this package is a single LLM call, and every one
    of them is documented as "inherits the judge's own noise, write your own
    for anything you plan to rely on" -- which nobody does in practice if
    doing it right is more code than doing it wrong. build_witnessed makes
    the witnessed version exactly as short as the unwitnessed one:

        # unwitnessed (single point of failure):
        hedge_judge = default_hedge_judge(llm)

        # witnessed (requires >=2 independently chosen LLMClients, ideally
        # different providers, so the two judges don't share a blind spot):
        hedge_judge, panel = build_witnessed(default_hedge_judge, [anthropic_llm, openai_llm])

    Args:
        judge_factory: a function llm -> judge_fn, e.g. default_hedge_judge.
        llms:          >=2 LLMClient-like objects to build one witness from
                       each. Passing two clients for the SAME provider/model
                       is allowed but defeats the purpose -- see the module
                       docstring's note on independence vs. agreement.
        names:         optional display names for the witnesses (for the
                       ConvergenceReport); defaults to "<provider>:<model>"
                       per llm, falling back to "<provider>" if no model
                       attribute is found.
        default_on_disagreement, require_unanimous: passed through to
                       WitnessPanel.combine().

    Returns:
        (combined_judge_fn, panel) -- pass combined_judge_fn wherever the
        single-witness judge would have gone; keep `panel` to call
        panel.convergence_report() afterward.
    """
    if len(llms) < 2:
        raise ValueError(
            "build_witnessed needs >=2 LLMClient-like objects -- a single "
            "one is exactly the unwitnessed default this function exists "
            "to make easy to avoid."
        )
    if names is None:
        names = []
        for llm in llms:
            provider = getattr(llm, "provider", "unknown")
            model = getattr(llm, "judge_model", None) or getattr(llm, "fast_model", None)
            names.append(f"{provider}:{model}" if model else str(provider))
        # de-duplicate names (e.g. two llms with the same provider/model repr)
        seen: dict[str, int] = {}
        deduped = []
        for n in names:
            seen[n] = seen.get(n, 0) + 1
            deduped.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
        names = deduped

    witnesses = {name: judge_factory(llm) for name, llm in zip(names, llms)}
    panel = WitnessPanel(witnesses=witnesses)
    combined = panel.combine(
        default_on_disagreement=default_on_disagreement,
        require_unanimous=require_unanimous,
    )
    return combined, panel
