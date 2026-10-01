"""
contradish.counterfactual.simulate -- synthetic agents with KNOWN properties,
emitted in the same record format a real run produces.

This exists for two things only, and neither is an empirical finding:

  1. Checking the analysis itself: when reliability failures are generated
     from accuracy alone (gamma = 0), H2 must reject at no more than its
     nominal rate even though update fidelity is correlated with accuracy
     (rho > 0); when they also depend on update fidelity (gamma > 0), it must
     reject often.
  2. Planning: how many models a real study needs for a given effect.

Generative model, per model m with latent ability a_m and update trait u_m
(correlation rho), and task j with difficulty d_j:

    knows_mj      ~ Bernoulli(sigmoid(1.2 (a_m - d_j) + 0.5))
    slip_m        = sigmoid(-2.2 - eta a_m - gamma u_m)      per-run lapse
    base run      passes iff knows and no slip
    changed run   if knows and no slip: reaches the amended outcome with
                  prob sigmoid(1 + 1.5 u_m - e_j); otherwise it reproduces
                  the base outcome (rigidity) with prob 0.7.
                  With update_hurdle=False it behaves like a control run.
    invariant run base outcome kept unless it drifts, prob sigmoid(-3 - u_m)
    control run   base outcome kept unless perturbed, prob 0.03
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from contradish.counterfactual.core import Record

__all__ = ["simulate_records", "power"]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_records(
    n_models: int = 12,
    n_tasks: int = 168,
    n_changed: int = 40,
    n_invariant: int = 80,
    runs: int = 5,
    rho: float = 0.5,
    gamma: float = 0.0,
    eta: float = 0.6,
    update_hurdle: bool = True,
    benchmark: str = "synthetic",
    seed: int = 0,
) -> list:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n_models)
    u = rho * a + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.standard_normal(n_models)
    d = rng.standard_normal(n_tasks)
    e = 0.7 * rng.standard_normal(n_tasks)
    tasks = [f"t{j:03d}" for j in range(n_tasks)]
    order = rng.permutation(n_tasks)
    changed = order[:n_changed]
    invariant = order[n_changed:n_changed + n_invariant]

    out = []
    for m in range(n_models):
        name = f"model{m:02d}"
        knows = rng.random(n_tasks) < _sig(1.2 * (a[m] - d) + 0.5)
        slip = _sig(-2.2 - eta * a[m] - gamma * u[m])
        drift = _sig(-3.0 - u[m])

        def ok():
            return rng.random() >= slip

        for j in range(n_tasks):
            for k in range(runs):
                out.append(Record(benchmark, name, "d", tasks[j], "base", "base", k, bool(knows[j] and ok())))
        for j in changed:
            for k in range(runs):
                live = bool(knows[j] and ok())
                if not live:
                    pa, pb = False, False
                elif not update_hurdle:
                    pa = rng.random() >= 0.03
                    pb = False
                else:
                    pa = rng.random() < _sig(1.0 + 1.5 * u[m] - e[j])
                    pb = (not pa) and (rng.random() < 0.7)
                out.append(Record(benchmark, name, "d", tasks[j], "amend", "changed", k, bool(pb), bool(pa)))
                out.append(Record(benchmark, name, "d", tasks[j], "control", "control", k,
                                  bool(knows[j] and ok() and rng.random() >= 0.03)))
        for j in invariant:
            for k in range(runs):
                out.append(Record(benchmark, name, "d", tasks[j], "amend", "invariant", k,
                                  bool(knows[j] and ok() and rng.random() >= drift)))
    return out


def power(n_models: int, gamma: float, rho: float = 0.5, sims: int = 200, runs: int = 5,
          update_hurdle: bool = True, permutations: int = 999, seed: int = 0,
          n_tasks: int = 168, n_changed: int = 40, n_invariant: int = 80,
          alpha: Optional[float] = None, acc_degree: int = 1) -> dict:
    """
    Rejection rates of H1 and H2 over `sims` simulated studies, at the
    pre-registered thresholds (ALPHA_H1, ALPHA_H2) unless `alpha` overrides both.
    """
    from contradish.counterfactual.analysis import (
        ALPHA_H1, ALPHA_H2, h1_update_vs_control, h2_incremental_validity,
    )
    a1 = ALPHA_H1 if alpha is None else alpha
    a2 = ALPHA_H2 if alpha is None else alpha
    from contradish.counterfactual.core import all_model_metrics

    r1 = r2 = n1 = n2 = 0
    for s in range(sims):
        recs = simulate_records(n_models=n_models, n_tasks=n_tasks, n_changed=n_changed,
                                n_invariant=n_invariant, runs=runs, rho=rho, gamma=gamma,
                                update_hurdle=update_hurdle, seed=seed * 100003 + s)
        h1 = h1_update_vs_control(recs, seed=s)
        if h1.p_value is not None:
            n1 += 1
            r1 += h1.p_value < a1 and h1.excess > 0
        h2 = h2_incremental_validity(all_model_metrics(recs), seed=s, permutations=permutations,
                                     acc_degree=acc_degree)
        if h2.p_value is not None:
            n2 += 1
            r2 += h2.p_value < a2 and h2.beta_buf < 0
    return {
        "n_models": n_models, "gamma": gamma, "rho": rho, "runs": runs, "sims": sims,
        "update_hurdle": update_hurdle, "alpha_h1": a1, "alpha_h2": a2,
        "h1_reject_rate": r1 / n1 if n1 else None,
        "h2_reject_rate": r2 / n2 if n2 else None,
    }
