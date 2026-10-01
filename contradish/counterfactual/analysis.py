"""
contradish.counterfactual.analysis -- the pre-registered tests.

Two questions, each with one primary test (see counterfactual/PREREGISTRATION.md):

H1  Is warranted behavioral updating a separate hurdle from task accuracy?
    Take only (model, task) pairs the model has mastered under the base
    policy. Give the same task two perturbations of its policy: a
    substantive amendment that warrants a different outcome, and a
    meaning-preserving rewording (control). If updating were nothing beyond
    accuracy plus sensitivity to perturbation, the failure rate would be the
    same under both. The test statistic is the mean, over tasks, of
    (failure under amendment - failure under control), averaged over models;
    inference is a one-sided sign-flip permutation test over tasks.

H2  Does update fidelity explain reliability failures after controlling for
    task accuracy?  (incremental validity)
    Across models, regress the reliability-failure rate on held-out
    criterion tasks on ACC and BUF (both measured on different tasks, all
    three on the empirical-logit scale), with a separate intercept per
    benchmark. BUF is net of the reworded-policy control (see core.py), so
    it does not contain run-to-run flakiness by construction. The test is on BUF's coefficient,
    one-sided (higher BUF -> fewer reliability failures), by Freedman-Lane
    permutation within benchmark, at alpha = 0.025 (see ALPHA_H2). Reported with the partial correlation and
    the gain in R^2 over the accuracy-only model.

Descriptive, not a gate: the correlation between ACC and CUF across models,
corrected for the unreliability of each (split-half, Spearman-Brown).

Requires numpy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np

from contradish.counterfactual.core import all_model_metrics, criterion_split

__all__ = [
    "H1Result", "H2Result", "h1_update_vs_control", "h2_incremental_validity",
    "disattenuated_correlation", "analyze", "format_report",
]


# ─────────────────────────────────────────────────────────────────────────────
# H1
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class H1Result:
    n_tasks: int = 0
    n_pairs: int = 0                 # mastered (model, task) pairs contributing
    n_models: int = 0
    fail_amended: Optional[float] = None
    fail_control: Optional[float] = None
    excess: Optional[float] = None   # mean over tasks of (amended - control) failure
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_value: Optional[float] = None  # one-sided: excess > 0
    rigid_share: Optional[float] = None   # of update failures, share that reproduced the base outcome
    note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _sign_flip_p(d: np.ndarray, rng: np.random.Generator, draws: int = 20000) -> float:
    """One-sided p for mean(d) > 0 under symmetric-about-zero null."""
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    obs = d.sum()
    if d.size <= 16:
        signs = np.array(list(product((-1.0, 1.0), repeat=d.size)))
        null = signs @ d
        return float((null >= obs - 1e-12).mean())
    signs = rng.choice((-1.0, 1.0), size=(draws, d.size))
    null = signs @ d
    return float(((null >= obs - 1e-12).sum() + 1) / (draws + 1))


def h1_update_vs_control(records: list, mastery: str = "all", seed: int = 0) -> H1Result:
    base = defaultdict(list)
    for r in records:
        if r.role == "base":
            base[(r.model, r.task_key)].append(r.pass_base)

    def mastered(model, key):
        runs = base.get((model, key))
        if not runs:
            return False
        return all(runs) if mastery == "all" else sum(runs) * 2 > len(runs)

    upd = defaultdict(list)     # (model, task) -> [pass_amended]
    old = defaultdict(list)     # (model, task) -> [pass_base] on changed runs
    ctl = defaultdict(list)     # (model, task) -> [pass_base] on control runs
    for r in records:
        if not mastered(r.model, r.task_key):
            continue
        if r.role == "changed":
            upd[(r.model, r.task_key)].append(bool(r.pass_amended))
            old[(r.model, r.task_key)].append(bool(r.pass_base))
        elif r.role == "control":
            ctl[(r.model, r.task_key)].append(bool(r.pass_base))

    per_task = defaultdict(list)
    fa, fc, rigid_num, rigid_den = [], [], 0.0, 0.0
    models = set()
    for key in upd:
        if key not in ctl:
            continue
        u = 1 - float(np.mean(upd[key]))
        c = 1 - float(np.mean(ctl[key]))
        per_task[key[1]].append(u - c)
        fa.append(u)
        fc.append(c)
        models.add(key[0])
        fails = [o for p, o in zip(upd[key], old[key]) if not p]
        rigid_num += sum(fails)
        rigid_den += len(fails)

    res = H1Result(n_tasks=len(per_task), n_pairs=len(fa), n_models=len(models))
    if len(per_task) < 2:
        res.note = "needs >= 2 tasks with both an amended and a control run on a mastered task"
        return res
    d = np.array([np.mean(v) for v in per_task.values()])
    rng = np.random.default_rng(seed)
    res.fail_amended = float(np.mean(fa))
    res.fail_control = float(np.mean(fc))
    res.excess = float(d.mean())
    boots = d[rng.integers(0, d.size, size=(5000, d.size))].mean(axis=1)
    res.ci_low, res.ci_high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    res.p_value = _sign_flip_p(d, rng)
    res.rigid_share = float(rigid_num / rigid_den) if rigid_den else None
    return res


# ─────────────────────────────────────────────────────────────────────────────
# H2
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class H2Result:
    n_models: int = 0
    n_dropped: int = 0                    # models with no BUF or REL_FAIL (e.g. nothing mastered)
    n_benchmarks: int = 0
    beta_buf: Optional[float] = None      # logit REL_FAIL per logit BUF, accuracy held fixed
    beta_acc: Optional[float] = None
    t_buf: Optional[float] = None
    p_value: Optional[float] = None       # one-sided: beta_buf < 0
    partial_r: Optional[float] = None     # REL_FAIL ~ BUF | ACC, benchmark
    r2_acc_only: Optional[float] = None
    r2_full: Optional[float] = None
    delta_r2: Optional[float] = None
    r_acc_buf: Optional[float] = None     # how collinear the two predictors are
    note: str = ""
    models: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _elogit(p, n) -> np.ndarray:
    """log((p n + 0.5) / ((1 - p) n + 0.5)): a logit that stays finite at 0 and 1."""
    p = np.asarray(p, float)
    n = np.asarray(n, float)
    return np.log((p * n + 0.5) / ((1 - p) * n + 0.5))


def _ols(X: np.ndarray, y: np.ndarray) -> tuple:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def _t_last(X: np.ndarray, y: np.ndarray) -> float:
    beta, resid = _ols(X, y)
    dof = X.shape[0] - np.linalg.matrix_rank(X)
    if dof <= 0:
        return float("nan")
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[-1, -1], 0.0)))
    return float(beta[-1] / se) if se > 0 else float("nan")


def h2_incremental_validity(metrics: list, seed: int = 0, permutations: int = 20000,
                            acc_degree: int = 1) -> H2Result:
    rows = [m for m in metrics if m.rel_fail is not None and m.acc is not None and m.buf is not None]
    res = H2Result(n_models=len(rows), n_dropped=len(metrics) - len(rows),
                   n_benchmarks=len({m.benchmark for m in rows}),
                   models=[(m.benchmark, m.model) for m in rows])
    benchmarks = sorted({m.benchmark for m in rows})
    n_params = len(benchmarks) + 1 + acc_degree
    if len(rows) < n_params + 2:
        res.note = (f"needs at least {n_params + 2} models with ACC, BUF and REL_FAIL "
                    f"(have {len(rows)}): the test is across models")
        return res
    # Empirical logits: rates near 0 or 1 are not linear in each other.
    y = _elogit([m.rel_fail for m in rows], [max(m.n_capable, 1) for m in rows])
    acc = _elogit([m.acc for m in rows], [max(m.n_acc_tasks, 1) for m in rows])
    buf = _elogit([m.buf for m in rows], [max(m.n_changed + m.n_invariant, 1) for m in rows])
    strata = np.array([benchmarks.index(m.benchmark) for m in rows])
    D = np.zeros((len(rows), len(benchmarks)))
    D[np.arange(len(rows)), strata] = 1.0
    acc_c = acc - acc.mean()
    Z = np.column_stack([D] + [acc_c ** d for d in range(1, acc_degree + 1)])   # reduced model
    X = np.column_stack([Z, buf])                # full: + update fidelity

    beta, resid_full = _ols(X, y)
    gamma, resid_red = _ols(Z, y)
    sst = float(((y - D @ np.linalg.lstsq(D, y, rcond=None)[0]) ** 2).sum())   # within-benchmark
    res.beta_buf, res.beta_acc = float(beta[-1]), float(beta[len(benchmarks)])
    res.t_buf = _t_last(X, y)
    if sst > 0:
        res.r2_acc_only = 1 - float(resid_red @ resid_red) / sst
        res.r2_full = 1 - float(resid_full @ resid_full) / sst
        res.delta_r2 = res.r2_full - res.r2_acc_only
    # Partial correlation: residualize BUF on Z too.
    _, buf_resid = _ols(Z, buf)
    denom = float(np.sqrt((resid_red @ resid_red) * (buf_resid @ buf_resid)))
    res.partial_r = float(resid_red @ buf_resid) / denom if denom > 0 else None
    _, acc_w = _ols(D, acc)
    _, buf_w = _ols(D, buf)
    dd = float(np.sqrt((acc_w @ acc_w) * (buf_w @ buf_w)))
    res.r_acc_buf = float(acc_w @ buf_w) / dd if dd > 0 else None

    if not np.isfinite(res.t_buf):
        res.note = "BUF has no variance left after accounting for accuracy"
        return res
    # Freedman-Lane: permute reduced-model residuals within benchmark. All
    # permutations are evaluated at once: by Frisch-Waugh-Lovell the BUF
    # coefficient is (x . y*) / (x . x) with x = BUF residualized on Z.
    rng = np.random.default_rng(seed)
    fitted = Z @ gamma
    n = len(rows)
    idx_by = [np.where(strata == s)[0] for s in range(len(benchmarks))]
    perms = np.tile(np.arange(n), (permutations, 1))
    for idx in idx_by:
        if idx.size > 1:
            perms[:, idx] = rng.permuted(np.tile(idx, (permutations, 1)), axis=1)
    Ystar = fitted[None, :] + resid_red[perms]
    xx = float(buf_resid @ buf_resid)
    dof = n - np.linalg.matrix_rank(X)
    M = np.eye(n) - X @ np.linalg.pinv(X)
    betas = Ystar @ buf_resid / xx
    rss = ((Ystar @ M) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = betas / np.sqrt(rss / dof / xx)
    ok = np.isfinite(ts)
    valid = int(ok.sum())
    count = int((ts[ok] <= res.t_buf + 1e-12).sum())
    res.p_value = float((count + 1) / (valid + 1)) if valid else None
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Descriptive: are ACC and CUF the same thing measured twice?
# ─────────────────────────────────────────────────────────────────────────────

def _half(key) -> int:
    import hashlib
    return int(hashlib.sha256(("half|" + "|".join(key)).encode()).hexdigest()[:8], 16) % 2


def disattenuated_correlation(records: list, mastery: str = "all") -> dict:
    """
    Correlation of ACC with CUF across models, and that correlation divided
    by the square root of the product of their split-half reliabilities
    (Spearman-Brown). A value well under 1 says the two scores rank models
    differently for reasons other than measurement noise. With few models
    this is a description, not a test.
    """
    from contradish.counterfactual.core import model_metrics
    pairs = sorted({(r.benchmark, r.model) for r in records})
    if len(pairs) < 4:
        return {"n_models": len(pairs), "note": "needs >= 4 models"}
    crit = criterion_split(records)
    halves = [[r for r in records if _half(r.task_key) == h] for h in (0, 1)]
    full, h0, h1 = [], [], []
    for b, mdl in pairs:
        full.append(model_metrics(records, b, mdl, crit, mastery))
        h0.append(model_metrics(halves[0], b, mdl, crit, mastery))
        h1.append(model_metrics(halves[1], b, mdl, crit, mastery))

    def col(ms, attr):
        return np.array([np.nan if getattr(m, attr) is None else getattr(m, attr) for m in ms], float)

    def corr(a, b):
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            return None
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    def sb(r):
        return None if r is None or r <= -1 else 2 * r / (1 + r)

    r_obs = corr(col(full, "acc"), col(full, "cuf"))
    rel_acc = sb(corr(col(h0, "acc"), col(h1, "acc")))
    rel_cuf = sb(corr(col(h0, "cuf"), col(h1, "cuf")))
    out = {"n_models": len(pairs), "r_acc_cuf": r_obs, "reliability_acc": rel_acc,
           "reliability_cuf": rel_cuf, "r_disattenuated": None}
    if r_obs is not None and rel_acc and rel_cuf and rel_acc > 0 and rel_cuf > 0:
        out["r_disattenuated"] = float(r_obs / np.sqrt(rel_acc * rel_cuf))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Putting it together
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_H1 = 0.05
# H2 controls for accuracy with a measured, hence imperfect, accuracy score.
# When accuracy and update fidelity are strongly correlated that leaves some
# residual confounding, and a test at 0.05 rejects too often (up to ~8% in
# simulation). 0.025 keeps the simulated false-positive rate at or under
# ~3.5% in all but the most extreme cell (see PREREGISTRATION.md).
ALPHA_H2 = 0.025


def analyze(records: list, mastery: str = "all", seed: int = 0, permutations: int = 20000) -> dict:
    """Run every pre-registered analysis on a set of records."""
    crit = criterion_split(records)
    metrics = all_model_metrics(records, crit, mastery)
    h1 = h1_update_vs_control(records, mastery, seed)
    h2 = h2_incremental_validity(metrics, seed, permutations)
    by_bench = {}
    for b in sorted({r.benchmark for r in records}):
        sub = [r for r in records if r.benchmark == b]
        by_bench[b] = {
            "h1": h1_update_vs_control(sub, mastery, seed).to_dict(),
            "h2": h2_incremental_validity([m for m in metrics if m.benchmark == b], seed, permutations).to_dict(),
        }
    return {
        "n_records": len(records),
        "benchmarks": sorted({r.benchmark for r in records}),
        "n_criterion_tasks": len(crit),
        "mastery": mastery,
        "alpha_h1": ALPHA_H1,
        "alpha_h2": ALPHA_H2,
        "metrics": [m.to_dict() for m in metrics],
        "h1": h1.to_dict(),
        "h2": h2.to_dict(),
        "by_benchmark": by_bench,
        "acc_vs_cuf": disattenuated_correlation(records, mastery),
        "verdict": {
            "h1_supported": bool(h1.p_value is not None and h1.p_value < ALPHA_H1 and (h1.excess or 0) > 0),
            "h2_supported": bool(h2.p_value is not None and h2.p_value < ALPHA_H2 and (h2.beta_buf or 0) < 0),
            "h1_testable": h1.p_value is not None,
            "h2_testable": h2.p_value is not None,
        },
    }


def format_report(result: dict) -> str:
    def f(x, pct=False, nd=3):
        if x is None:
            return "n/a"
        return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"

    lines = ["", "  CONTRADISH COUNTERFACTUAL SUITE  *  " + ", ".join(result["benchmarks"]), "-" * 78, ""]
    lines.append("  model                          ACC     CUF    RIGID   HOLD    CTRL    BUF   REL_FAIL")
    lines.append("  (BUF is net of the CTRL floor: mean of min(1, CUF/CTRL) and min(1, HOLD/CTRL))")
    for m in result["metrics"]:
        lines.append(
            f"  {(m['benchmark'] + '/' + m['model'])[:29]:<29s} "
            f"{f(m['acc'], 1):>6s}  {f(m['cuf'], 1):>6s}  {f(m['rigid'], 1):>6s}  {f(m['hold'], 1):>6s}  "
            f"{f(m['ctrl'], 1):>6s}  {f(m['buf'], 1):>6s}  {f(m['rel_fail'], 1):>6s}"
        )
    h1, h2, v = result["h1"], result["h2"], result["verdict"]
    lines += ["", "  H1  updating is a hurdle beyond accuracy (mastered tasks: amendment vs reworded control)"]
    if v["h1_testable"]:
        lines.append(
            f"      failure under amendment {f(h1['fail_amended'], 1)}  vs  under control {f(h1['fail_control'], 1)}"
            f"   excess {f(h1['excess'], 1)}  (95% CI {f(h1['ci_low'], 1)} to {f(h1['ci_high'], 1)})"
        )
        lines.append(
            f"      one-sided p = {f(h1['p_value'], nd=4)}   tasks={h1['n_tasks']}  models={h1['n_models']}"
            f"  mastered pairs={h1['n_pairs']}"
            + (f"   rigid share of update failures {f(h1['rigid_share'], 1)}" if h1['rigid_share'] is not None else "")
        )
        lines.append("      -> " + ("SUPPORTED" if v["h1_supported"] else "not supported"))
    else:
        lines.append(f"      not testable: {h1['note']}")
    lines += ["", "  H2  update fidelity explains reliability failures after controlling for accuracy"]
    if v["h2_testable"]:
        lines.append(
            f"      beta_BUF = {f(h2['beta_buf'])}  (t = {f(h2['t_buf'], nd=2)})   partial r = {f(h2['partial_r'], nd=2)}"
            f"   R2 accuracy-only {f(h2['r2_acc_only'], nd=2)} -> with BUF {f(h2['r2_full'], nd=2)}"
            f"  (gain {f(h2['delta_r2'], nd=2)})"
        )
        lines.append(
            f"      one-sided permutation p = {f(h2['p_value'], nd=4)}   models={h2['n_models']}"
            + (f" ({h2['n_dropped']} dropped: no mastered tasks)" if h2.get('n_dropped') else "") +
            f"  benchmarks={h2['n_benchmarks']}   r(ACC, BUF) = {f(h2['r_acc_buf'], nd=2)}"
        )
        lines.append("      -> " + ("SUPPORTED" if v["h2_supported"] else "not supported")
                     + f"  (threshold p < {result['alpha_h2']})")
    else:
        lines.append(f"      not testable: {h2['note']}")
    d = result["acc_vs_cuf"]
    if d.get("r_acc_cuf") is not None:
        lines += ["", f"  ACC vs CUF across models: r = {f(d['r_acc_cuf'], nd=2)}"
                      f"   corrected for unreliability: {f(d.get('r_disattenuated'), nd=2)}"
                      f"   (reliability ACC {f(d.get('reliability_acc'), nd=2)}, CUF {f(d.get('reliability_cuf'), nd=2)})"]
    lines.append("")
    return "\n".join(lines)
