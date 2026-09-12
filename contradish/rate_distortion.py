"""
rate_distortion.py: the information-graded resolution curve.

resolution.py answers a binary question: is there a hidden variable that,
stated as a plain fact, causally resolves this collapsing distinction? Real
deployments almost never hand a model a plain, fully-confirmed fact. They
hand it a chart note that "suggests" something, a support ticket where the
customer "believes" they're covered, a form field that's blank. The
question that actually matters in production is not "does full information
fix this," it's: as information about the disambiguating condition goes
from nothing to certain, does the model's accuracy rise smoothly with it,
or does it stay stuck at the collapsed baseline until the last possible
instant and then jump? The first is a model you can trust with partial
information. The second is a model that is only ever as good as its worst
sentence of context, and there is no way to tell that apart from a plain
accuracy score.

This module measures that curve directly, black-box, on any candidate a
resolution.py run has already validated (or on any pole statements supplied
directly). It is the behavioral, black-box sibling of a specific internal
research result: on a small hand-built transformer, collateral damage to an
unrelated constraint during narrow fine-tuning was found to scale
monotonically with the bits of missing information about a hidden
disambiguating variable, across a graded noisy channel (pooled Spearman
r=+0.96 vs. noise level, over 20 seeds per setting). That result measured
weight-level damage under real gradient descent and required building the
network by hand; it is not re-tested here, and nothing in this module
proves it generalizes to how frontier models are actually fine-tuned.

What transfers, and what this module actually tests, is the shape of the
claim: is degradation graded or is it a cliff. Instead of noise injected
into weights, the noise here is injected into language: the disambiguating
condition is asserted with a graded ladder of hedges from "no information
at all" to "stated as a certain fact," and the module measures the model's
accuracy at each rung. A Spearman correlation between rung and accuracy
gives the same kind of number the original result reported, on the
question contradish can actually ask of a real, black-box model today:
does this model's grip on a distinction it demonstrably has (per
resolution.py) degrade gracefully under partial information, or does it
fall off a cliff. That is a new, useful, and -- as of the research pass
behind this module -- unclaimed measurement: nothing in the eval/guardrail
market (Petri, PromptPex, Giskard, TruLens-class tools, LMUnit) reports a
graded information curve for constraint resolution; they report a pass/fail
or a single score.

Usage::

    from contradish.resolution import discover_resolution
    from contradish.rate_distortion import measure_rate_distortion_for_resolution

    result = discover_resolution(pair=pair, model_fn=my_model,
                                  commitment_extractor=my_extractor, llm=llm)
    curve = measure_rate_distortion_for_resolution(
        result, model_fn=my_model, commitment_extractor=my_extractor,
    )
    if curve is not None:
        print(curve.report())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .distinction import DistinctionPair, ModelFn
from .resolution import ResolutionResult

RATE_DISTORTION_SCHEMA_VERSION = "1.0"


# Graded certainty ladder, no-information to full-information. Each template
# takes the pole statement (a short factual clause, trailing period
# stripped) and wraps it at a specific hedge strength. None means "assert
# nothing about this variable at all" -- the unconditioned baseline.
_CERTAINTY_LEVELS: list[tuple[str, Optional[str]]] = [
    ("no_information",   None),
    ("weak_hint",         "There is a slight, unconfirmed indication that {pole}"),
    ("moderate_signal",   "The available information suggests, but does not confirm, that {pole}"),
    ("strong_signal",     "It is very likely, though not fully confirmed, that {pole}"),
    ("full_information",  "{pole}"),
]


@dataclass
class RateDistortionPoint:
    """One rung of the certainty ladder and the measured accuracy there."""
    level_name:    str
    level_index:   int      # 0 (no information) .. len(_CERTAINTY_LEVELS)-1 (full information)
    accuracy_a:    float    # fraction of question_a probes matching pair.commit_a
    accuracy_b:    float    # fraction of question_b probes matching pair.commit_b
    mean_accuracy: float


@dataclass
class RateDistortionCurveResult:
    """
    The measured information-accuracy curve for one distinction pair and one
    candidate disambiguating condition.

    shape
        "graded"      -- accuracy rises with information and correlates with
                         it (correlation >= graded_threshold): this model's
                         resolution of the distinction degrades gracefully
                         under partial information.
        "threshold"   -- accuracy ends up higher at full information than at
                         none, but not smoothly: most of the improvement
                         happens at or near the last rung. A brittle model:
                         reliable only with a fully-stated fact, worthless
                         with a hedge.
        "insensitive" -- accuracy does not meaningfully improve even at full
                         information (full_information accuracy - baseline
                         accuracy below insensitive_threshold). This condition
                         does not actually help this model; a companion,
                         honest negative result to resolution.py's own
                         "not resolved."
    correlation
        Spearman rank correlation between rung index and mean_accuracy
        across all rungs. Reported even when shape is not "graded" -- the
        number, not just the label, is what a claim like this should be
        checked against.
    """
    pair_id:      str
    description:  str
    condition:    str
    points:       list[RateDistortionPoint] = field(default_factory=list)
    correlation:  float = 0.0
    shape:        str = "insensitive"

    def summary(self) -> str:
        return (
            f"{self.pair_id} / '{self.condition}': {self.shape} "
            f"(correlation={self.correlation:+.2f}, "
            f"{self.points[0].mean_accuracy:.0%} -> {self.points[-1].mean_accuracy:.0%} "
            f"as information rises)"
        ) if self.points else f"{self.pair_id}: no points measured"

    def report(self) -> str:
        W   = 72
        sep = "-" * W
        bar = lambda r, w=16: "#" * round(r * w) + "." * (w - round(r * w))

        lines = [
            "",
            f"  RATE-DISTORTION CURVE  ·  {self.pair_id}",
            sep,
            f"  {self.description}",
            f"  candidate condition: {self.condition}",
            "",
            "  INFORMATION -> ACCURACY  (no information -> full information)",
            "",
        ]
        for p in self.points:
            lines.append(
                f"  [{p.level_index}] {p.level_name:<18}  "
                f"{bar(p.mean_accuracy)}  {p.mean_accuracy:.0%}  "
                f"(a={p.accuracy_a:.0%}, b={p.accuracy_b:.0%})"
            )
        lines += [
            "",
            f"  shape: {self.shape.upper()}   correlation: {self.correlation:+.2f}",
        ]
        if self.shape == "graded":
            lines.append(
                "    accuracy rises smoothly with information -- this model "
                "handles partial, hedged context gracefully on this distinction."
            )
        elif self.shape == "threshold":
            lines.append(
                "    accuracy only recovers near full information -- this model "
                "is brittle on this distinction: reliable with a stated fact, "
                "worthless with a hedge."
            )
        else:
            lines.append(
                "    accuracy does not meaningfully improve even at full "
                "information -- this candidate does not actually help this "
                "model on this distinction."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema_version": RATE_DISTORTION_SCHEMA_VERSION,
            "pair_id": self.pair_id,
            "description": self.description,
            "condition": self.condition,
            "points": [
                {
                    "level_name": p.level_name,
                    "level_index": p.level_index,
                    "accuracy_a": round(p.accuracy_a, 4),
                    "accuracy_b": round(p.accuracy_b, 4),
                    "mean_accuracy": round(p.mean_accuracy, 4),
                }
                for p in self.points
            ],
            "correlation": round(self.correlation, 4),
            "shape": self.shape,
        }


def _spearman_r(xs: list[float], ys: list[float]) -> float:
    """
    Spearman rank correlation, average ranks for ties, no external
    dependency. Returns 0.0 (no claimed relationship) when either series has
    zero variance -- a flat line correlates with nothing, and that is the
    honest number, not an error.
    """
    n = len(xs)
    if n < 2:
        return 0.0

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-indexed, averaged over the tie block
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov   = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((r - mean_rx) ** 2 for r in rx)
    var_y = sum((r - mean_ry) ** 2 for r in ry)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def measure_rate_distortion_curve(
    pair:                  DistinctionPair,
    pole_a_statement:      str,
    pole_b_statement:      str,
    model_fn:              ModelFn,
    commitment_extractor:  Callable[[str, str], str],
    system_prompt:         str = "",
    validation_samples:    int = 2,
    graded_threshold:      float = 0.7,
    insensitive_threshold: float = 0.15,
    verbose:               bool = False,
) -> RateDistortionCurveResult:
    """
    Measure how a model's accuracy on `pair` changes as stated certainty
    about a candidate disambiguating condition rises from nothing to a
    fully-stated fact.

    Args:
        pair:                   the DistinctionPair being characterized.
        pole_a_statement:       short factual clause supporting situation A
                                (typically a validated ResolutionCandidate's
                                pole_a_statement -- see
                                measure_rate_distortion_for_resolution).
        pole_b_statement:       short factual clause supporting situation B.
        model_fn:               (system_prompt, question) -> answer.
        commitment_extractor:   (question, answer) -> commitment string.
        system_prompt:          passed to model_fn on every call.
        validation_samples:     probes per question per certainty rung.
        graded_threshold:       minimum correlation to call the curve "graded".
        insensitive_threshold:  minimum (full - baseline) accuracy gain
                                required before a curve can be "graded" or
                                "threshold" rather than "insensitive".
        verbose:                print progress.

    Returns:
        A RateDistortionCurveResult with one point per certainty rung.
    """
    points: list[RateDistortionPoint] = []

    for idx, (level_name, template) in enumerate(_CERTAINTY_LEVELS):
        if template is None:
            aug_a, aug_b = pair.question_a, pair.question_b
        else:
            clause_a = template.format(pole=pole_a_statement.rstrip("."))
            clause_b = template.format(pole=pole_b_statement.rstrip("."))
            aug_a = f"{pair.question_a} {clause_a}."
            aug_b = f"{pair.question_b} {clause_b}."

        if verbose:
            print(f"  Rung {idx} ({level_name}): {pair.pair_id}")

        hits_a: list[bool] = []
        hits_b: list[bool] = []
        for _ in range(max(1, validation_samples)):
            com_a = commitment_extractor(aug_a, model_fn(system_prompt, aug_a))
            com_b = commitment_extractor(aug_b, model_fn(system_prompt, aug_b))
            hits_a.append(com_a == pair.commit_a)
            hits_b.append(com_b == pair.commit_b)

        acc_a = sum(hits_a) / len(hits_a)
        acc_b = sum(hits_b) / len(hits_b)
        points.append(RateDistortionPoint(
            level_name    = level_name,
            level_index   = idx,
            accuracy_a    = acc_a,
            accuracy_b    = acc_b,
            mean_accuracy = (acc_a + acc_b) / 2,
        ))

    xs = [p.level_index for p in points]
    ys = [p.mean_accuracy for p in points]
    correlation = _spearman_r(xs, ys)

    baseline_acc = points[0].mean_accuracy
    full_acc     = points[-1].mean_accuracy
    net_gain     = full_acc - baseline_acc

    if net_gain < insensitive_threshold:
        shape = "insensitive"
    elif correlation >= graded_threshold:
        shape = "graded"
    else:
        shape = "threshold"

    return RateDistortionCurveResult(
        pair_id     = pair.pair_id,
        description = pair.description,
        condition   = f"{pole_a_statement.rstrip('.')} / {pole_b_statement.rstrip('.')}",
        points      = points,
        correlation = correlation,
        shape       = shape,
    )


def measure_rate_distortion_for_resolution(
    result:                ResolutionResult,
    model_fn:              ModelFn,
    commitment_extractor:  Callable[[str, str], str],
    pair:                  Optional[DistinctionPair] = None,
    **kwargs,
) -> Optional[RateDistortionCurveResult]:
    """
    Convenience wrapper: run measure_rate_distortion_curve on the winning
    candidate of an already-computed resolution.discover_resolution() result.

    Returns None (not an error -- there is nothing to characterize) when
    `result.best` is None, i.e. no candidate was ever proposed.

    Args:
        result:  a ResolutionResult from discover_resolution().
        pair:    the original DistinctionPair. Required, since ResolutionResult
                only carries the pair's id/description/labels, not its
                questions or commitments -- pass the same pair object you
                gave discover_resolution.
    """
    if result.best is None:
        return None
    if pair is None:
        raise ValueError(
            "measure_rate_distortion_for_resolution needs the original "
            "DistinctionPair (ResolutionResult doesn't carry question_a/"
            "question_b/commit_a/commit_b) -- pass the same pair object "
            "you gave discover_resolution."
        )
    return measure_rate_distortion_curve(
        pair                 = pair,
        pole_a_statement     = result.best.pole_a_statement,
        pole_b_statement     = result.best.pole_b_statement,
        model_fn             = model_fn,
        commitment_extractor = commitment_extractor,
        **kwargs,
    )
