"""
Contradish x STATE-Bench: counterfactual policy amendments for Microsoft's
STATE-Bench (https://github.com/microsoft/STATE-Bench, MIT).

STATE-Bench scores an agent on whether it leaves a sandbox database in the
correct final state under a fixed policy. That is ordinary task accuracy.
This adapter adds the counterfactual axis: amend one policy rule, and ask
whether the agent's final state changes on exactly the tasks the amendment
governs (and to the right values), and stays put everywhere else.

HOW THE GROUND TRUTH IS DERIVED (no model, no human labelling)
---------------------------------------------------------------
STATE-Bench's policy is executable: each domain's `policies.py` holds named
constants that its environment's tools compute with, and its `get_policies`
tool returns the matching prose. STATE-Bench also ships successful gold
trajectories for its train split. So for an amendment that changes a
constant and the prose that states it:

  1. Replay the gold tool calls against a fresh environment under the BASE
     policy. Keep the task only if that reproduces STATE-Bench's own
     checked-in expected state (scored by STATE-Bench's own scorer).
  2. Replay them again under the AMENDED policy. Where the gold agent
     submitted a number the environment had computed for it (the net refund
     on `process_return`), submit the number the environment computes now.
  3. Classify the task for this amendment:
       changed    the gold path's final state differs: the amendment
                  WARRANTS a different outcome, and the amended final state
                  is the new expected state.
       invariant  the final state is identical: nothing the amendment
                  touches bears on this task; the base expected state stands.
       excluded   the gold path is no longer valid evidence (a tool call
                  flipped between success/rejection/error, or the agent
                  supplied a number that the amendment made stale, or the
                  amended expected state cannot be expressed/verified).
                  Reported with the reason; never scored.

The derivation is deliberately conservative: amendments are restricted to
rules whose effect is computed by the environment and visible in tool
results, and anything the replay cannot vouch for is excluded rather than
guessed. A meaning-preserving control amendment (prose reworded, no constant
changed) must classify every task `invariant`; that is checked.

This module needs a STATE-Bench checkout on disk (pass its root). Deriving
the suite makes no model calls. Running a model against it uses STATE-Bench's
own orchestrator with an amended environment (see amended_domain()).
"""

from __future__ import annotations

import copy
import dataclasses
import glob
import importlib
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "PolicyAmendment",
    "AMENDMENTS",
    "DerivedCase",
    "load_state_bench",
    "apply_amendment",
    "amended_domain",
    "replay",
    "derive_case",
    "derive_suite",
    "amended_task",
    "case_from_dict",
    "score_run",
    "scripted_records",
    "live_records",
    "SUITE_VERSION",
]

SUITE_VERSION = "0.1"
BASE = "base"
DOMAINS = ("customer_support",)


# ─────────────────────────────────────────────────────────────────────────────
# Amendments
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyAmendment:
    """
    One counterfactual change to a STATE-Bench domain policy.

    constants   {NAME: new_value} set on the domain's `policies` module. A
                dotted "NAME.key" sets one key of a dict constant.
    text_edits  [(old, new)] applied to everything `get_policies` returns, so
                the prose the agent reads states the amended rule. Every edit
                must match somewhere in the domain's policy text
                (validate_text_edits checks), otherwise the agent would be
                told the old rule while being scored on the new one.
    control     True for a meaning-preserving rewording: no constants, and
                every task must come out `invariant`.
    """
    id: str
    domain: str
    description: str
    constants: dict = field(default_factory=dict)
    text_edits: list = field(default_factory=list)
    control: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "domain": self.domain, "description": self.description,
            "constants": dict(self.constants), "text_edits": [list(e) for e in self.text_edits],
            "control": self.control,
        }


# v0.1 covers STATE-Bench's customer_support domain. Candidate amendments in
# the other domains, and a cancellation intercept-fee amendment here, were
# dropped by the derivation's own guards: in every task they touched, the
# environment computes and writes the amended value itself, so an agent that
# ignored the amendment would still reach the amended state. Such cases test
# nothing about updating. (Travel keeps its fee schedule outside its
# policies module; shopping_assistant computes cart totals server-side.)
AMENDMENTS: dict = {a.id: a for a in [
    # ── customer_support ────────────────────────────────────────────────────
    PolicyAmendment(
        "cs_restocking_fee_20", "customer_support",
        "Restocking fee on opened electronics returned for changed_mind: 15% -> 20%.",
        constants={"RESTOCKING_FEE_PCT": 20},
        text_edits=[("15% restocking fee for opened electronics", "20% restocking fee for opened electronics")],
    ),
    PolicyAmendment(
        "cs_return_shipping_fee_12", "customer_support",
        "Return shipping fee on low-value orders: $8 -> $12.",
        constants={"PAID_RETURN_SHIPPING_FEE": 12},
        text_edits=[("customer pays $8 return shipping fee", "customer pays $12 return shipping fee")],
    ),
    PolicyAmendment(
        "cs_repeat_surcharge_10", "customer_support",
        "Repeat-category return surcharge: $5 -> $10 per additional return.",
        constants={"REPEAT_CATEGORY_SURCHARGE": 10},
        text_edits=[("$5 surcharge per additional return", "$10 surcharge per additional return")],
    ),
    PolicyAmendment(
        "cs_bulk_clawback_8", "customer_support",
        "Bulk-discount clawback: $5 -> $8 per remaining item.",
        constants={"BULK_CLAWBACK_PER_ITEM": 8},
        text_edits=[("$5 per remaining item, deducted from the return refund",
                     "$8 per remaining item, deducted from the return refund")],
    ),
    PolicyAmendment(
        "cs_control_reworded", "customer_support",
        "Control: restocking-fee and return-shipping sentences reworded, nothing changed.",
        text_edits=[
            ("15% restocking fee for opened electronics returned for changed_mind.",
             "Opened electronics sent back because the customer changed their mind carry a restocking fee of 15%."),
            ("Orders with subtotal < $50: customer pays $8 return shipping fee.",
             "When an order's subtotal is under $50, the customer is charged an $8 fee for return shipping."),
        ],
        control=True,
    ),
    PolicyAmendment(
        "cs_shipping_clawback_12", "customer_support",
        "Free-shipping clawback on returns that drop an order below the threshold: $8 -> $12.",
        constants={"STANDARD_SHIPPING_COST": 12},
        text_edits=[("$8 standard shipping fee is deducted from the return refund",
                     "$12 standard shipping fee is deducted from the return refund")],
    ),
    PolicyAmendment(
        "cs_gold_restocking_discount_75", "customer_support",
        "Gold members' restocking-fee discount: 50% -> 75%.",
        constants={"RESTOCKING_DISCOUNT_BY_TIER.gold": 0.75},
        text_edits=[("50% off the restocking fee", "75% off the restocking fee")],
    ),
]}

# Optional {amendment_id: (topic substring, ...)} restricting where text edits
# apply, for an edit too short to be safe across every policy topic.
TEXT_SCOPE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Loading STATE-Bench
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StateBench:
    root: str
    schemas: Any
    scoring: Any
    env_loader: Any

    def domain(self, name: str):
        return importlib.import_module(f"state_bench.domains.{name}.config").get_config()

    def policies(self, name: str):
        return importlib.import_module(f"state_bench.domains.{name}.policies")

    def task(self, domain: str, task_id: str):
        path = os.path.join(self.root, "state_bench", "domains", domain, "tasks", f"{task_id}.json")
        with open(path, encoding="utf-8") as f:
            return self.schemas.TaskDefinition.from_dict(json.load(f))

    def gold_trajectory(self, domain: str, task_id: str) -> Optional[list]:
        path = os.path.join(self.root, "datasets", "train_task_trajectories", domain, f"{task_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("conversation", [])

    def gold_task_ids(self, domain: str) -> list:
        d = os.path.join(self.root, "datasets", "train_task_trajectories", domain)
        return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(d, "*.json")))

    def commit(self) -> str:
        head = os.path.join(self.root, ".git", "HEAD")
        try:
            ref = open(head).read().strip()
            if ref.startswith("ref: "):
                return open(os.path.join(self.root, ".git", ref[5:])).read().strip()
            return ref
        except OSError:
            return "unknown"


def load_state_bench(root: str) -> StateBench:
    """Import a STATE-Bench checkout (no network, no model client needed)."""
    root = os.path.abspath(root)
    if not os.path.isdir(os.path.join(root, "state_bench")):
        raise FileNotFoundError(f"{root} is not a STATE-Bench checkout (no state_bench/ package)")
    if root not in sys.path:
        sys.path.insert(0, root)
    return StateBench(
        root=root,
        schemas=importlib.import_module("state_bench.schemas"),
        scoring=importlib.import_module("state_bench.scoring"),
        env_loader=importlib.import_module("state_bench.env_loader"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Applying an amendment
# ─────────────────────────────────────────────────────────────────────────────

def _edit_text(obj: Any, edits: list, hits: Optional[list] = None) -> Any:
    """Apply (old, new) replacements to every string inside a JSON-like value."""
    if isinstance(obj, str):
        for i, (old, new) in enumerate(edits):
            if old in obj:
                obj = obj.replace(old, new)
                if hits is not None:
                    hits[i] += 1
        return obj
    if isinstance(obj, list):
        return [_edit_text(x, edits, hits) for x in obj]
    if isinstance(obj, dict):
        return {k: _edit_text(v, edits, hits) for k, v in obj.items()}
    return obj


def _in_scope(amendment: PolicyAmendment, topic: str) -> bool:
    scope = TEXT_SCOPE.get(amendment.id)
    return scope is None or any(s in (topic or "") for s in scope)


@contextmanager
def apply_amendment(sb: StateBench, amendment: Optional[PolicyAmendment]):
    """Patch the domain's policy constants for the duration of the block."""
    if amendment is None or not amendment.constants:
        yield
        return
    pol = sb.policies(amendment.domain)
    saved = {}
    try:
        for name, value in amendment.constants.items():
            if "." in name:
                const, key = name.split(".", 1)
                d = getattr(pol, const)
                saved[name] = d[key]
                d[key] = value
            else:
                saved[name] = getattr(pol, name)
                setattr(pol, name, value)
        yield
    finally:
        for name, value in saved.items():
            if "." in name:
                const, key = name.split(".", 1)
                getattr(pol, const)[key] = value
            else:
                setattr(pol, name, value)


def _amend_env(env: Any, amendment: Optional[PolicyAmendment]) -> Any:
    """Make env.get_policies return the amended prose."""
    if amendment is None or not amendment.text_edits:
        return env
    original = env.get_policies

    def get_policies(params):
        out = original(params)
        if _in_scope(amendment, (params or {}).get("topic", "")):
            return _edit_text(copy.deepcopy(out), amendment.text_edits)
        return out

    env.get_policies = get_policies
    return env


def amended_domain(sb: StateBench, amendment: PolicyAmendment):
    """
    A DomainConfig whose environments serve the amended policy prose, for use
    with STATE-Bench's own orchestrator (state_bench.orchestrator.run_task).
    Wrap the run in `with apply_amendment(sb, amendment):` so the constants
    are amended too.
    """
    cfg = sb.domain(amendment.domain)
    base_cls = cfg.environment_class

    class AmendedEnvironment(base_cls):  # type: ignore[misc, valid-type]
        def get_policies(self, params):
            out = super().get_policies(params)
            if _in_scope(amendment, (params or {}).get("topic", "")):
                return _edit_text(copy.deepcopy(out), amendment.text_edits)
            return out

    AmendedEnvironment.__name__ = f"{base_cls.__name__}__{amendment.id}"
    return dataclasses.replace(cfg, environment_class=AmendedEnvironment)


def policy_topics(sb: StateBench, domain: str) -> list:
    """Topic names the domain's get_policies tool accepts."""
    cfg = sb.domain(domain)
    for schema in cfg.tool_schemas:
        fn = schema.get("function", schema)
        if fn.get("name") == "get_policies":
            props = fn.get("parameters", {}).get("properties", {})
            enum = props.get("topic", {}).get("enum")
            if enum:
                return list(enum)
    return []


def validate_text_edits(sb: StateBench, amendment: PolicyAmendment, env: Any) -> list:
    """Edits that match no policy text the agent could read. Empty means all good."""
    hits = [0] * len(amendment.text_edits)
    for topic in policy_topics(sb, amendment.domain):
        if _in_scope(amendment, topic):
            _edit_text(env.get_policies({"topic": topic}), amendment.text_edits, hits)
    return [amendment.text_edits[i][0] for i, h in enumerate(hits) if h == 0]


# ─────────────────────────────────────────────────────────────────────────────
# Gold-trajectory replay
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Replay:
    calls: list             # [(name, args_used, result)]
    diff: Any               # state_bench.schemas.StateDiff
    snapshot: dict
    substituted: int = 0    # agent-supplied amounts replaced by the env's amended computation
    notes: list = field(default_factory=list)


def _tool_calls(trajectory: list) -> list:
    out = []
    for msg in trajectory:
        for tc in msg.get("tool_calls") or []:
            out.append((tc["name"], copy.deepcopy(tc.get("arguments") or {})))
    return out


def _new_env(sb: StateBench, cfg: Any, task: Any) -> Any:
    env_data, _ = sb.env_loader.load_task_environment(cfg, task)
    return cfg.environment_class(env_data.deep_copy(), now=task.now)


def _signature(result: Any) -> tuple:
    """Coarse outcome class of a tool call: did it error, and what status did it report."""
    if not isinstance(result, dict):
        return ("value",)
    return ("error" in result, result.get("status"))


def replay(sb: StateBench, domain: str, task: Any, trajectory: list,
           amendment: Optional[PolicyAmendment] = None,
           base: Optional["Replay"] = None) -> Replay:
    """
    Execute a gold trajectory's tool calls against a fresh environment.

    With `amendment` and the `base` replay, an agent-supplied `amount` that
    equalled what the environment computed under the base policy is replaced
    by what the environment computes under the amended one (found by a dry
    run on a copy of the environment).
    """
    cfg = sb.domain(domain)
    with apply_amendment(sb, amendment):
        env = _amend_env(_new_env(sb, cfg, task), amendment)
        before = env.get_full_snapshot()
        calls, substituted, notes = [], 0, []
        for i, (name, args) in enumerate(_tool_calls(trajectory)):
            handler = env.tool_handlers.get(name) if name != "get_policies" else env.get_policies
            if handler is None:
                continue        # memory / non-domain tools
            if amendment is not None and base is not None and "amount" in args and i < len(base.calls):
                base_result = base.calls[i][2] if isinstance(base.calls[i][2], dict) else {}
                base_computed = base_result.get("policy_computed_amount")
                if base_computed is not None and _num_eq(args["amount"], base_computed):
                    dry = copy.deepcopy(env)
                    dry_handler = dry.tool_handlers.get(name)
                    dry_result = dry_handler(copy.deepcopy(args)) if dry_handler else {}
                    new_computed = dry_result.get("policy_computed_amount") if isinstance(dry_result, dict) else None
                    if new_computed is not None and not _num_eq(new_computed, args["amount"]):
                        args["amount"] = new_computed
                        substituted += 1
            result = handler(copy.deepcopy(args))
            calls.append((name, args, result))
        after = env.get_full_snapshot()
    diff = sb.schemas.StateDiff.compute(before, after)
    return Replay(calls=calls, diff=diff, snapshot=after, substituted=substituted, notes=notes)


def _num_eq(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def _numbers(obj: Any, out: Optional[set] = None) -> set:
    out = set() if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Deriving the counterfactual ground truth
# ─────────────────────────────────────────────────────────────────────────────

CHANGED = "changed"
INVARIANT = "invariant"
EXCLUDED = "excluded"


@dataclass
class DerivedCase:
    domain: str
    task_id: str
    amendment_id: str
    cls: str                                 # changed | invariant | excluded
    reason: str = ""                         # why excluded, or what changed
    state_requirements: Optional[list] = None  # amended expected state (changed cases)
    changed_fields: list = field(default_factory=list)   # [(entity, key, field, base, amended)]
    tier: str = ""                           # changed cases: clean | rewritten
    # Fields a non-updating agent leaves at their base-policy value:
    # [(entity, key, field, value)]. A run that reproduces all of them (and
    # misses the amended state) is scored as rigidity.
    rigid_fields: list = field(default_factory=list)
    rewrites: dict = field(default_factory=dict)         # {"$39": "$36"} applied to the task script

    def to_dict(self) -> dict:
        d = {"domain": self.domain, "task_id": self.task_id, "amendment_id": self.amendment_id,
             "class": self.cls}
        if self.reason:
            d["reason"] = self.reason
        if self.cls == CHANGED:
            d["state_requirements"] = self.state_requirements
            d["changed_fields"] = [list(c) for c in self.changed_fields]
            d["tier"] = self.tier
            d["rigid_fields"] = [list(c) for c in self.rigid_fields]
            if self.rewrites:
                d["rewrites"] = dict(self.rewrites)
        return d


def _base_ok(sb: StateBench, task: Any, rep: Replay) -> bool:
    score = sb.scoring.evaluate_state_requirements(task, rep.diff)
    return score is None or score.score == 1


def _amended_requirements(requirements: list, base_snap: dict, amended_snap: dict) -> tuple:
    """
    Rewrite direct field assertions whose value the amendment changed. Returns
    (requirements, changed_fields). Other requirement shapes are kept as-is;
    the caller verifies the result against the amended replay.
    """
    out, changed = [], []
    for req in requirements:
        req = copy.deepcopy(req)
        if {"entity_type", "record_key", "field", "expected_value"} <= set(req):
            et, key, fld = req["entity_type"], str(req["record_key"]), req["field"]
            b = (base_snap.get(et, {}).get(key) or {}).get(fld)
            a = (amended_snap.get(et, {}).get(key) or {}).get(fld)
            if b != a and req["expected_value"] == b:
                changed.append((et, key, fld, b, a))
                req["expected_value"] = a
        out.append(req)
    return out, changed


def _script_strings(task: Any, judge: bool = False) -> list:
    """
    judge=False: every string the simulated user is built from (opening
    message + user_simulator). judge=True: what only the conversation judge
    sees (task_summary + task_requirements), which this suite does not score.
    """
    sim = task.user_simulator.to_dict() if hasattr(task.user_simulator, "to_dict") else {}
    out = [task.task_summary or ""] if judge else [task.opening_message or ""]

    def walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(task.task_requirements if judge else sim)
    return out


def _rule_tokens(amendment: PolicyAmendment) -> list:
    """Old-rule figures an amendment replaces, e.g. "15%" or "$8"."""
    toks = []
    for old, new in amendment.text_edits:
        o = re.findall(r"\$\d+(?:\.\d+)?|\d+(?:\.\d+)?%", old)
        n = re.findall(r"\$\d+(?:\.\d+)?|\d+(?:\.\d+)?%", new)
        toks += [t for t in o if t not in n]
    return sorted(set(toks))


def _script_check(task: Any, amendment: PolicyAmendment, changed: list) -> tuple:
    """
    A task's simulated-user script was written for the base policy. If it
    cites an amount the amendment changes, running it unedited would have the
    simulated user push the agent toward the old answer. Returns
    (tier, rewrites, exclusion_reason):

      clean      nothing cited.
      rewritten  only dollar amounts ("$39") that map to exactly one amended
                 value are cited; amended_task() rewrites them.
      excluded   the script cites the old rule itself ("15%"), or a changed
                 amount as a bare number that cannot be rewritten safely.
    """
    text = "\n".join(_script_strings(task))
    judge_text = "\n".join(_script_strings(task, judge=True))
    for tok in _rule_tokens(amendment):
        if re.search(r"(?<![\d.])" + re.escape(tok) + r"(?![\d])", text):
            return ("", {}, f"task script cites the amended rule ({tok})")
    mapping: dict = {}
    for _, _, _, b, a in changed:
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            continue
        key = _fmt_num(b)
        if key in mapping and mapping[key] != _fmt_num(a):
            if re.search(r"(?<![\d.])\$?" + re.escape(key) + r"(?![\d.])", text):
                return ("", {}, f"task script cites {key}, which maps to more than one amended value")
            mapping[key] = None
        else:
            mapping[key] = _fmt_num(a)
    rewrites = {}
    for b, a in mapping.items():
        if a is None:
            continue
        dollar = re.search(r"\$" + re.escape(b) + r"(?![\d.])", text)
        bare = re.search(r"(?<![\d.$])" + re.escape(b) + r"(?![\d.])", text)
        if bare:
            return ("", {}, f"task script cites the changed amount {b} as a bare number")
        if dollar or re.search(r"\$" + re.escape(b) + r"(?![\d.])", judge_text):
            rewrites["$" + b] = "$" + a
    sim_rewritten = any(re.search(re.escape(k) + r"(?![\d.])", text) for k in rewrites)
    return ("rewritten" if sim_rewritten else "clean", rewrites, "")


def _fmt_num(x: Any) -> str:
    return str(int(x)) if float(x) == int(x) else str(x)


def _rewrite(obj: Any, rewrites: dict) -> Any:
    if isinstance(obj, str):
        for old, new in rewrites.items():
            obj = re.sub(re.escape(old) + r"(?![\d.])", new, obj)
        return obj
    if isinstance(obj, list):
        return [_rewrite(x, rewrites) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite(v, rewrites) for k, v in obj.items()}
    return obj


def derive_case(sb: StateBench, domain: str, task_id: str, amendment: PolicyAmendment,
                base: Optional[Replay] = None) -> DerivedCase:
    """Classify one task under one amendment, and derive its amended expected state."""
    task = sb.task(domain, task_id)
    traj = sb.gold_trajectory(domain, task_id)

    def excluded(reason):
        return DerivedCase(domain, task_id, amendment.id, EXCLUDED, reason)

    if traj is None:
        return excluded("no gold trajectory")
    if base is None:
        base = replay(sb, domain, task, traj)
    if not _base_ok(sb, task, base):
        return excluded("gold trajectory does not reproduce the checked-in expected state")

    am = replay(sb, domain, task, traj, amendment=amendment, base=base)
    if len(am.calls) != len(base.calls):
        return excluded("replay length changed")

    changed_numbers: set = set()
    for (bn, bargs, bres), (an, aargs, ares) in zip(base.calls, am.calls):
        if bn == "get_policies":
            continue
        if _signature(bres) != _signature(ares):
            return excluded(f"tool outcome changed for {bn}: {_signature(bres)} -> {_signature(ares)}")
        changed_numbers |= (_numbers(bres) - _numbers(ares))
    # An agent-supplied number that the amendment made stale (and that the
    # replay had no rule to recompute) means the gold path is no longer right.
    for (bn, bargs, bres), (an, aargs, ares) in zip(base.calls, am.calls):
        if bn == "get_policies":
            continue
        for k, v in aargs.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if float(v) in changed_numbers and bargs.get(k) == v and float(v) != 0.0:
                return excluded(f"agent-supplied {bn}.{k}={v} depends on an amended quantity")

    if am.diff.to_dict() == base.diff.to_dict():
        return DerivedCase(domain, task_id, amendment.id, INVARIANT)
    if amendment.control:
        return excluded("control amendment changed the final state")

    reqs, changed = _amended_requirements(task.state_requirements, base.snapshot, am.snapshot)
    if not changed:
        return excluded("final state changed in a way the task's requirements do not express")
    amended = dataclasses.replace(task, state_requirements=reqs)
    ok = sb.scoring.evaluate_state_requirements(amended, am.diff)
    if ok is not None and ok.score != 1:
        return excluded("amended expected state fails verification against the amended replay")
    # The base expectation must genuinely fail under the amendment, and vice versa.
    cross = sb.scoring.evaluate_state_requirements(task, am.diff)
    if cross is None or cross.score == 1:
        return excluded("base expected state still satisfied after amendment")
    # A non-updating agent: the same gold calls with the amounts it would
    # have submitted under the old policy. If that still reaches the amended
    # state, the environment produces the outcome whatever the agent does and
    # the case tests nothing about updating.
    rigid = replay(sb, domain, task, traj, amendment=amendment, base=None)
    rigid_score = sb.scoring.evaluate_state_requirements(amended, rigid.diff)
    if rigid_score is None or rigid_score.score == 1:
        return excluded("amended outcome is produced by the environment regardless of the agent")
    rigid_fields = []
    for et, key, fld, b, a in changed:
        v = (rigid.snapshot.get(et, {}).get(key) or {}).get(fld)
        if v != a:
            rigid_fields.append((et, key, fld, v))
    tier, rewrites, why = _script_check(task, amendment, changed)
    if why:
        return excluded(why)
    return DerivedCase(domain, task_id, amendment.id, CHANGED,
                       reason=f"{len(changed)} field(s) change", state_requirements=reqs,
                       changed_fields=changed, tier=tier, rewrites=rewrites,
                       rigid_fields=rigid_fields)


def case_from_dict(d: dict) -> DerivedCase:
    return DerivedCase(
        domain=d["domain"], task_id=d["task_id"], amendment_id=d["amendment_id"], cls=d["class"],
        reason=d.get("reason", ""), state_requirements=d.get("state_requirements"),
        changed_fields=[tuple(x) for x in d.get("changed_fields", [])], tier=d.get("tier", ""),
        rewrites=d.get("rewrites", {}), rigid_fields=[tuple(x) for x in d.get("rigid_fields", [])],
    )


def score_run(sb: StateBench, case: DerivedCase, state_diff: Any, control: bool = False) -> tuple:
    """
    Score one run's final state for one derived case.
    Returns (pass_base, pass_amended):
      invariant / control   pass_base = base expected state reached; pass_amended None.
      changed               pass_amended = amended expected state reached;
                            pass_base = the run reproduced the OLD outcome on
                            every field a non-updating agent leaves stale
                            (rigidity), and did not reach the amended state.
    """
    base_task = sb.task(case.domain, case.task_id)

    def ok(task):
        sc = sb.scoring.evaluate_state_requirements(task, state_diff)
        return bool(sc is None or sc.score == 1)

    if case.cls != CHANGED or control:
        return ok(base_task), None
    updated = ok(amended_task(sb, case))
    rigid = False
    if not updated and case.rigid_fields:
        snap = sb.scoring._reconstruct_final_snapshot(base_task, state_diff)
        rigid = all(
            (snap.get(et, {}).get(str(key)) or {}).get(fld) == v
            for et, key, fld, v in case.rigid_fields
        )
    return rigid, updated


def amended_task(sb: StateBench, case: DerivedCase):
    """The STATE-Bench TaskDefinition to score a run against under this amendment."""
    task = sb.task(case.domain, case.task_id)
    if case.cls != CHANGED:
        return task
    d = task.to_dict()
    if case.rewrites:
        for k in ("opening_message", "task_summary", "user_simulator", "task_requirements"):
            if k in d:
                d[k] = _rewrite(d[k], case.rewrites)
    d["state_requirements"] = copy.deepcopy(case.state_requirements)
    if getattr(task, "task_env_path", None):
        d["task_env_path"] = task.task_env_path
    return sb.schemas.TaskDefinition.from_dict(d)


def derive_suite(sb: StateBench, amendments: Optional[list] = None) -> dict:
    """
    Derive the full Contradish x STATE-Bench manifest: for every amendment,
    every replay-verified gold task classified changed / invariant / excluded.
    """
    amendments = list(amendments) if amendments is not None else list(AMENDMENTS.values())
    manifest = {
        "suite": "contradish-x-state-bench",
        "suite_version": SUITE_VERSION,
        "state_bench_commit": sb.commit(),
        "amendments": [],
        "replay": {},
        "cases": [],
    }
    bases: dict = {}
    for domain in sorted({a.domain for a in amendments}):
        ids = sb.gold_task_ids(domain)
        verified = 0
        for tid in ids:
            task = sb.task(domain, tid)
            rep = replay(sb, domain, task, sb.gold_trajectory(domain, tid))
            if _base_ok(sb, task, rep):
                bases[(domain, tid)] = rep
                verified += 1
        manifest["replay"][domain] = {"gold_tasks": len(ids), "replay_verified": verified}

    for a in amendments:
        cfg = sb.domain(a.domain)
        ids = [tid for (d, tid) in bases if d == a.domain]
        unmatched = []
        if ids:
            probe_task = sb.task(a.domain, ids[0])
            unmatched = validate_text_edits(sb, a, _new_env(sb, cfg, probe_task))
        counts = {CHANGED: 0, INVARIANT: 0, EXCLUDED: 0, "changed_clean": 0, "changed_rewritten": 0}
        for tid in sorted(ids):
            case = derive_case(sb, a.domain, tid, a, base=bases[(a.domain, tid)])
            counts[case.cls] += 1
            if case.cls == CHANGED:
                counts["changed_" + case.tier] += 1
            manifest["cases"].append(case.to_dict())
        entry = a.to_dict()
        entry["counts"] = counts
        entry["unmatched_text_edits"] = unmatched
        entry["valid"] = (not unmatched) and (counts[CHANGED] > 0 or a.control) \
            and not (a.control and counts[EXCLUDED] > 0)
        manifest["amendments"].append(entry)
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Producing records
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK = "state-bench"


def _plan(manifest: dict, invariant_per_amendment: Optional[int], seed: int) -> list:
    """
    [(amendment_id or None, case dict, role)] to run: every changed case,
    a seeded sample of invariant cases per substantive amendment, and, under
    each domain's control amendment, the tasks that are `changed` under some
    substantive amendment (the paired control H1 needs).
    """
    import random
    from contradish.counterfactual.core import is_criterion_key
    rng = random.Random(seed)
    amend = {a["id"]: a for a in manifest["amendments"] if a.get("valid", True)}
    plan = []
    changed_tasks: dict = {}
    for a in amend.values():
        if a["control"]:
            continue
        cases = [c for c in manifest["cases"] if c["amendment_id"] == a["id"]]
        ch = [c for c in cases if c["class"] == CHANGED]
        # Tasks reserved for the reliability criterion are never run under an
        # amendment, so the criterion shares no task with the BUF measurement.
        inv = sorted((c for c in cases if c["class"] == INVARIANT
                      and not is_criterion_key((BENCHMARK, c["domain"], c["task_id"]))),
                     key=lambda c: c["task_id"])
        if invariant_per_amendment is not None and len(inv) > invariant_per_amendment:
            inv = rng.sample(inv, invariant_per_amendment)
        plan += [(a["id"], c, "changed") for c in ch] + [(a["id"], c, "invariant") for c in inv]
        changed_tasks.setdefault(a["domain"], set()).update(c["task_id"] for c in ch)
    for a in amend.values():
        if not a["control"]:
            continue
        for c in manifest["cases"]:
            if c["amendment_id"] == a["id"] and c["class"] == INVARIANT \
                    and c["task_id"] in changed_tasks.get(a["domain"], set()):
                plan.append((a["id"], c, "control"))
    return plan


def _base_tasks(manifest: dict) -> list:
    return sorted({(c["domain"], c["task_id"]) for c in manifest["cases"] if c["class"] != EXCLUDED
                   or c.get("reason", "") != "gold trajectory does not reproduce the checked-in expected state"})


def scripted_records(sb: StateBench, manifest: dict, agent: str = "oracle", model: Optional[str] = None,
                     invariant_per_amendment: Optional[int] = None, seed: int = 0) -> list:
    """
    Records for two scripted agents that need no model, used to check the
    suite itself end to end:

      oracle  replays the gold trajectory and submits what the amended
              policy computes. Must pass everything.
      rigid   replays the gold trajectory with the numbers the OLD policy
              computed. Must hold every invariant and control case, and be
              scored as rigidity on every changed case.
    """
    from contradish.counterfactual.core import Record
    if agent not in ("oracle", "rigid"):
        raise ValueError("agent must be 'oracle' or 'rigid'")
    model = model or f"scripted-{agent}"
    out = []
    bases = {}
    for domain, tid in _base_tasks(manifest):
        task = sb.task(domain, tid)
        rep_ = replay(sb, domain, task, sb.gold_trajectory(domain, tid))
        bases[(domain, tid)] = rep_
        out.append(Record(BENCHMARK, model, domain, tid, BASE, "base", 0, _base_ok(sb, task, rep_)))
    for aid, c, role in _plan(manifest, invariant_per_amendment, seed):
        a = AMENDMENTS[aid]
        case = case_from_dict(c)
        task = sb.task(case.domain, case.task_id)
        traj = sb.gold_trajectory(case.domain, case.task_id)
        base = bases[(case.domain, case.task_id)] if agent == "oracle" else None
        rep_ = replay(sb, case.domain, task, traj, amendment=a, base=base)
        pb, pa = score_run(sb, case, rep_.diff, control=(role == "control"))
        out.append(Record(BENCHMARK, model, case.domain, case.task_id, aid, role, 0, pb, pa))
    return out


def live_records(sb: StateBench, manifest: dict, model: str, client: Any, simulator_client: Any = None,
                 runs: int = 5, invariant_per_amendment: Optional[int] = 12, seed: int = 0,
                 agent_class: Any = None, on_record: Any = None) -> list:
    """
    Run a real agent through STATE-Bench's own orchestrator under the base
    policy and every amendment, and score the final states.

    client / simulator_client are STATE-Bench LLM clients
    (state_bench.client.build_llm_client / build_user_sim_client); this
    function adds nothing to how STATE-Bench talks to a model. Scoring uses
    only STATE-Bench's deterministic state scorer, for the base and amended
    conditions alike, so accuracy and update fidelity are measured with the
    same instrument and no LLM judge.

    NOTE: this path needs model credentials and has not been exercised in
    the environment this suite was built in; scripted_records() covers the
    derivation and scoring logic it shares.
    """
    from contradish.counterfactual.core import Record
    orchestrator = importlib.import_module("state_bench.orchestrator")
    out = []

    def emit(rec):
        out.append(rec)
        if on_record is not None:
            on_record(rec)

    def one(domain_cfg, task):
        env_data, _ = sb.env_loader.load_task_environment(domain_cfg, task)
        traj = orchestrator.run_task(task, env_data, task.user_id, client, domain=domain_cfg,
                                     simulator_client=simulator_client, agent_class=agent_class)
        return traj.state_diff

    for domain, tid in _base_tasks(manifest):
        cfg = sb.domain(domain)
        task = sb.task(domain, tid)
        for k in range(runs):
            diff = one(cfg, task)
            sc = sb.scoring.evaluate_state_requirements(task, diff)
            emit(Record(BENCHMARK, model, domain, tid, BASE, "base", k, bool(sc is None or sc.score == 1)))
    for aid, c, role in _plan(manifest, invariant_per_amendment, seed):
        a = AMENDMENTS[aid]
        case = case_from_dict(c)
        cfg = amended_domain(sb, a)
        task = amended_task(sb, case)       # simulated-user script with amended amounts
        for k in range(runs):
            with apply_amendment(sb, a):
                diff = one(cfg, task)
            pb, pa = score_run(sb, case, diff, control=(role == "control"))
            emit(Record(BENCHMARK, model, case.domain, case.task_id, aid, role, k, pb, pa))
    return out
