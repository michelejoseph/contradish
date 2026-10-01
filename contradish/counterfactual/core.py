"""
contradish.counterfactual.core -- the benchmark-independent record format and
the per-model quantities every Counterfactual Suite adapter reports.

One record is one scored run of one agent on one task under one policy
state. Any benchmark with (a) a deterministic pass/fail scorer and (b) a way
to amend the governing policy and re-derive the expected outcome can emit
these records; everything downstream (metrics, the statistical tests) only
reads records.

    benchmark   "state-bench", "appworld", "tau3-bench", ...
    model       agent / model identifier
    domain      benchmark domain (stratum)
    task_id
    condition   "base", or the amendment id the run was made under
    role        base       unamended policy
                changed    amendment that WARRANTS a different outcome here
                invariant  amendment that does not bear on this task
                control    meaning-preserving rewording of the policy
    run         repeat index
    pass_base   the run's outcome satisfies the BASE expected outcome
    pass_amended  (changed only) it satisfies the AMENDED expected outcome

Two properties are computed from disjoint evidence wherever they are
compared:

  ordinary task accuracy (ACC)
      mean base pass rate.

  warranted behavioral updating, conditional on competence
      Only (model, task) pairs the model has MASTERED under the base policy
      (passed every base run) are counted, so a failure under amendment
      cannot be a restatement of "the model cannot do this task":
        CUF   P(amended outcome reached | mastered, changed)   raw update rate
        RIGID P(base outcome produced   | mastered, changed)   rigidity
        HOLD  P(base outcome kept       | mastered, invariant) 1 - raw drift
        CTRL  P(base outcome kept       | mastered, control)   perturbation floor

      CUF and HOLD still contain the model's ordinary run-to-run flakiness
      and its sensitivity to ANY change in the policy text: a model that
      fails one run in five fails amended runs too, for reasons that have
      nothing to do with updating. CTRL, measured on the same mastered tasks
      under a meaning-preserving rewording, carries exactly those two things
      and no update requirement. Dividing it out leaves the update-specific
      part:

        UPD   min(1, CUF / CTRL)     update fidelity, net of the control
        KEEP  min(1, HOLD / CTRL)    non-drift, net of the control
        BUF   (UPD + KEEP) / 2       Behavioral Update Fidelity

      Without this step BUF would correlate with reliability by
      construction, and the H2 test would be circular.

  reliability failure (REL_FAIL), the criterion
      On held-out criterion tasks: of the tasks the model passed at least
      once in k base runs, the share it did not pass every time. This is
      the gap between pass@1 and pass^k, conditioned on capability.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

__all__ = [
    "Record", "load_records", "dump_records", "criterion_split", "is_criterion_key",
    "ModelMetrics", "model_metrics", "all_model_metrics",
]

ROLES = ("base", "changed", "invariant", "control")


@dataclass
class Record:
    benchmark: str
    model: str
    domain: str
    task_id: str
    condition: str
    role: str
    run: int
    pass_base: bool
    pass_amended: Optional[bool] = None

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        if self.role == "changed" and self.pass_amended is None:
            raise ValueError("a 'changed' record needs pass_amended")

    @property
    def task_key(self) -> tuple:
        return (self.benchmark, self.domain, self.task_id)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        pa = d.get("pass_amended")
        return cls(
            benchmark=str(d["benchmark"]), model=str(d["model"]), domain=str(d.get("domain", "")),
            task_id=str(d["task_id"]), condition=str(d["condition"]), role=str(d["role"]),
            run=int(d.get("run", 0)), pass_base=bool(d["pass_base"]),
            pass_amended=None if pa is None else bool(pa),
        )


def load_records(path) -> list:
    """Read records from a .jsonl file (one JSON object per line) or a .json list."""
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return [Record.from_dict(d) for d in json.loads(text)]
    return [Record.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def dump_records(records: Iterable, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict() if isinstance(r, Record) else r) + "\n")


def criterion_split(records: list, fraction: float = 0.5, salt: str = "contradish-cf-v1") -> set:
    """
    The held-out criterion tasks: a deterministic (hash-based) `fraction` of
    the tasks that appear ONLY under the base condition. Tasks that were ever
    run under an amendment are never criterion tasks, so the reliability
    criterion shares no task with the update-fidelity measurement.
    """
    amended = {r.task_key for r in records if r.role != "base"}
    base_only = sorted({r.task_key for r in records if r.role == "base"} - amended)
    return {key for key in base_only if is_criterion_key(key, fraction, salt)}


def is_criterion_key(key: tuple, fraction: float = 0.5, salt: str = "contradish-cf-v1") -> bool:
    """Whether a (benchmark, domain, task_id) key falls in the reserved criterion half."""
    h = int(hashlib.sha256((salt + "|" + "|".join(key)).encode()).hexdigest()[:8], 16)
    return h < int(fraction * 2 ** 32)


@dataclass
class ModelMetrics:
    benchmark: str
    model: str
    acc: Optional[float] = None          # ordinary task accuracy (predictor tasks)
    n_acc_tasks: int = 0
    cuf: Optional[float] = None
    rigid: Optional[float] = None
    n_changed: int = 0                   # mastered changed (task, amendment) pairs
    hold: Optional[float] = None
    n_invariant: int = 0
    ctrl: Optional[float] = None
    n_control: int = 0
    upd: Optional[float] = None          # CUF net of the control floor
    keep: Optional[float] = None         # HOLD net of the control floor
    buf: Optional[float] = None
    rel_fail: Optional[float] = None     # criterion
    n_capable: int = 0
    n_criterion_tasks: int = 0
    base_runs: int = 0                   # modal number of base runs per task

    def to_dict(self) -> dict:
        return asdict(self)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def model_metrics(records: list, benchmark: str, model: str, criterion: Optional[set] = None,
                  mastery: str = "all") -> ModelMetrics:
    """
    Metrics for one model on one benchmark.

    criterion  task keys reserved for the reliability criterion (see
               criterion_split); ACC is computed on the remaining tasks.
    mastery    "all": a task is mastered if every base run passed (default).
               "majority": if more than half did.
    """
    criterion = criterion or set()
    rs = [r for r in records if r.benchmark == benchmark and r.model == model]
    base = defaultdict(list)
    for r in rs:
        if r.role == "base":
            base[r.task_key].append(r.pass_base)

    def mastered(key) -> bool:
        runs = base.get(key)
        if not runs:
            return False
        return all(runs) if mastery == "all" else sum(runs) * 2 > len(runs)

    m = ModelMetrics(benchmark=benchmark, model=model)
    pred = {k: v for k, v in base.items() if k not in criterion}
    m.acc = _mean(sum(v) / len(v) for v in pred.values())
    m.n_acc_tasks = len(pred)
    if base:
        counts = defaultdict(int)
        for v in base.values():
            counts[len(v)] += 1
        m.base_runs = max(counts, key=counts.get)

    # Per (task, condition) means first, so a pair run more often does not count more.
    def pair_mean(role, attr):
        groups = defaultdict(list)
        for r in rs:
            if r.role == role and mastered(r.task_key):
                groups[(r.task_key, r.condition)].append(bool(getattr(r, attr)))
        return [sum(v) / len(v) for v in groups.values()]

    changed_upd = pair_mean("changed", "pass_amended")
    changed_old = pair_mean("changed", "pass_base")
    inv = pair_mean("invariant", "pass_base")
    ctl = pair_mean("control", "pass_base")
    m.cuf, m.rigid, m.n_changed = _mean(changed_upd), _mean(changed_old), len(changed_upd)
    m.hold, m.n_invariant = _mean(inv), len(inv)
    m.ctrl, m.n_control = _mean(ctl), len(ctl)
    if m.ctrl:
        if m.cuf is not None:
            m.upd = min(1.0, m.cuf / m.ctrl)
        if m.hold is not None:
            m.keep = min(1.0, m.hold / m.ctrl)
        parts = [x for x in (m.upd, m.keep) if x is not None]
        if parts:
            m.buf = sum(parts) / len(parts)

    crit = {k: v for k, v in base.items() if k in criterion and len(v) >= 2}
    capable = [v for v in crit.values() if any(v)]
    m.n_criterion_tasks = len(crit)
    m.n_capable = len(capable)
    if capable:
        m.rel_fail = sum(1 for v in capable if not all(v)) / len(capable)
    return m


def all_model_metrics(records: list, criterion: Optional[set] = None, mastery: str = "all") -> list:
    criterion = criterion_split(records) if criterion is None else criterion
    pairs = sorted({(r.benchmark, r.model) for r in records})
    return [model_metrics(records, b, mdl, criterion, mastery) for b, mdl in pairs]
