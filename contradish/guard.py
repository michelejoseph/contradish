"""
contradish/guard.py -- one call, before an agent's tool calls run.

    from contradish import Guard

    guard = Guard("policy.json", mode="shadow", log="decisions.jsonl")
    decision = guard.check(evidence, proposed_calls, as_of="2026-10-20", request_id="req-123")
    run_these = decision.calls_to_run()          # shadow: everything proposed; enforce: minus what is blocked

`evidence` is what your system has on record for this request, each fact with the source it came from. `proposed_calls`
are the tool calls your agent wants to make, in any of: contradish Calls, {"tool", "args"} dicts, an OpenAI chat
message (or its `tool_calls`), or Anthropic `tool_use` content blocks.

The decision answers two questions about the same request, exactly, from the policy file alone (no model, no judge):

    may it act?        an irreversible call the established evidence does not justify is BLOCKED (`justify`)
    is it withholding? a benefit the established evidence already entitles the person to, not delivered or
                       delivered short, is OWED (`entitled`)

and, when the case is not settled, which facts to ASK for and from whom (the fewest that settle it). Outcomes, in
order of precedence: deliver (something owed is being withheld), block, verify, review (a reversible call the
evidence does not justify), allow.

Every decision carries a certificate: the policy, the updates, the evidence, the proposed calls and the derivation,
with a sha256 digest. `python contradish/evidence_check.py decisions.jsonl` re-derives each one with a separate,
standard-library implementation that enumerates every value instead of using regions, so a decision it accepts has
been established twice, by two methods. In shadow mode nothing is blocked: the log is the record of what the guard
would have done, and `contradish report decisions.jsonl` turns it into a value report.

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from contradish.entitlement import (ENTITLED, EXCLUDED, POSSIBLE, WITHHOLDING, check_entitlement, governing_from_spec,
                                    load_policy)
from contradish.justification import (JUSTIFIED, VERIFY_TOOL, Evidence, InformationState, check_actions,
                                      verification_plan)
from contradish.policy_program import Call, PolicyProgram

__all__ = ["Guard", "Decision", "calls_from", "evidence_from", "run_examples", "check_policy", "DECISION_SCHEMA"]

DECISION_SCHEMA = "contradish.decision/1"
OUTCOMES = ("deliver", "block", "verify", "review", "allow")


# ─────────────────────────────────────────────────────────────────────────────
# Adapters
# ─────────────────────────────────────────────────────────────────────────────

def _get(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def calls_from(x: Any) -> list:
    """Tool calls from: contradish Calls, {"tool","args"} dicts, an OpenAI message / its tool_calls, Anthropic
    tool_use blocks (or a message whose content holds them). Anything else is ignored."""
    if x is None:
        return []
    if isinstance(x, Call):
        return [x]
    tc = _get(x, "tool_calls")
    if tc is not None and not isinstance(x, list):
        return calls_from(list(tc))
    content = _get(x, "content")
    if isinstance(content, list) and not isinstance(x, list):
        return calls_from(content)
    if not isinstance(x, (list, tuple)):
        x = [x]
    out = []
    for c in x:
        if isinstance(c, Call):
            out.append(c)
            continue
        if _get(c, "type") == "tool_use":                                 # Anthropic
            out.append(Call.of(_get(c, "name"), dict(_get(c, "input") or {})))
            continue
        fn = _get(c, "function")
        if fn is not None:                                                 # OpenAI
            args = _get(fn, "arguments") or {}
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            out.append(Call.of(_get(fn, "name"), dict(args)))
            continue
        tool = _get(c, "tool") or _get(c, "name")
        if isinstance(tool, str):
            out.append(Call.of(tool, dict(_get(c, "args") or _get(c, "input") or _get(c, "arguments") or {})))
    return out


def evidence_from(x: Any, default_source: Optional[str] = None) -> list:
    """Evidence from: Evidence objects; {"fact","value"|"lo"/"hi"|"values","source"} dicts; or a mapping
    {fact: (value, source)} / {fact: {"value":..., "source":...}} / {fact: value} with `default_source`."""
    if isinstance(x, dict) and not ({"fact", "source"} <= set(x)):
        out = []
        for f, v in x.items():
            if isinstance(v, tuple):
                out.append(Evidence(f, v[1], value=v[0]))
            elif isinstance(v, dict):
                out.append(Evidence.from_dict({"fact": f, **v}))
            else:
                if default_source is None:
                    raise ValueError(f"fact {f!r} has no source: pass (value, source) or default_source")
                out.append(Evidence(f, default_source, value=v))
        return out
    if isinstance(x, (dict, Evidence)):
        x = [x]
    return [e if isinstance(e, Evidence) else Evidence.from_dict(e) for e in (x or [])]


# ─────────────────────────────────────────────────────────────────────────────
# Decision
# ─────────────────────────────────────────────────────────────────────────────

def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(obj) -> str:
    return "sha256:" + hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


def _ct(c: Optional[Call]) -> str:
    return "nothing" if c is None else f"{c.tool}(" + ", ".join(f"{k}={v}" for k, v in c.args) + ")"


@dataclass
class Decision:
    outcome: str                      # deliver | block | verify | review | allow
    mode: str                         # shadow | enforce
    proposed: list
    blocked: list = field(default_factory=list)
    owed: list = field(default_factory=list)
    ask: list = field(default_factory=list)          # [{"fact", "from"}]
    review: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    certificate: dict = field(default_factory=dict)
    required: Optional[list] = None   # when the evidence settles every step: exactly the calls the policy requires

    @property
    def ok(self) -> bool:
        return self.outcome == "allow"

    def calls_to_run(self, deliver_owed: bool = False) -> list:
        """What to execute. Shadow mode changes nothing. Enforce mode drops blocked calls. With `deliver_owed`,
        when the evidence settles the whole case, exactly the calls the policy requires; otherwise the proposed
        calls minus everything not justified, plus what is owed."""
        if self.mode == "shadow":
            return list(self.proposed)
        keep = [c for c in self.proposed if c not in self.blocked]
        if deliver_owed:
            if self.required is not None:
                return list(self.required)
            owed_tools = {c.tool for c in self.owed}
            keep = [c for c in keep if c.tool not in owed_tools and c not in self.review] + list(self.owed)
        return keep

    def to_dict(self) -> dict:
        return {"outcome": self.outcome, "mode": self.mode,
                "proposed": [c.to_dict() for c in self.proposed], "blocked": [c.to_dict() for c in self.blocked],
                "owed": [c.to_dict() for c in self.owed], "ask": self.ask,
                "review": [c.to_dict() for c in self.review], "reasons": self.reasons}

    def explain(self) -> str:
        lines = [f"{self.outcome.upper()}" + (" (shadow: nothing was stopped)" if self.mode == "shadow" and self.outcome != "allow" else "")]
        lines += [f"  {r}" for r in self.reasons]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# The guard
# ─────────────────────────────────────────────────────────────────────────────

class Guard:
    """
    Guard(policy, mode="shadow" | "enforce", log=None)

    policy   a built-in program name ("pharmacy_refills", "returns"), a path to a policy JSON file, or a spec dict.
             The file may hold `updates` (each with `source` and `effective`), `fact_authority` (which sources
             establish which facts), `irreversible_tools` and `benefit_steps`.
    log      a JSONL path: every decision's certificate is appended (the input to `contradish report` and to the
             independent checker).
    """

    def __init__(self, policy: Any, mode: str = "shadow", log: Optional[str] = None,
                 updates: Optional[list] = None, require_approval: bool = False):
        if mode not in ("shadow", "enforce"):
            raise ValueError("mode must be 'shadow' or 'enforce'")
        if isinstance(policy, dict):
            spec = dict(policy)
            raw = spec.pop("updates", [])
        else:
            spec, raw = load_policy(policy)
        from contradish.drafting import APPROVAL_FIELDS, approval_status
        st = approval_status(dict(spec, updates=raw))
        self.approval = dict(st, **{k: (spec.get("approval") or {}).get(k) for k in ("by", "role", "at", "digest")
                                    if (spec.get("approval") or {}).get(k)})
        refuse = ("draft", "edited after approval") + (("unreviewed",) if require_approval else ())
        if st["state"] in refuse:
            msg = (f"policy {spec.get('id')!r} is {st['state']} ({st['detail']}). "
                   "Review it (contradish guard review) and approve it (contradish guard approve) first.")
            if mode == "enforce":
                raise PermissionError("will not enforce: " + msg)
            import warnings
            warnings.warn("shadow mode on a policy nobody has approved: " + msg, stacklevel=2)
        spec = {k: v for k, v in spec.items() if k not in APPROVAL_FIELDS}
        if updates is not None:
            raw = list(raw) + list(updates)
        self.base_spec = PolicyProgram(spec).spec          # compiled, canonical
        self.updates = [dict(u) for u in raw]
        self.mode = mode
        self.log = log
        missing = [f for f in self.base_spec.get("facts", {}) if f not in (self.base_spec.get("fact_authority") or {})]
        if self.base_spec["id"] != "returns" and missing:
            raise ValueError(f"policy has no fact_authority for {missing}: say which sources establish each fact")

    def check(self, evidence: Any, calls: Any, as_of: Optional[str] = None, request_id: Optional[str] = None,
              context: Optional[dict] = None) -> Decision:
        as_of = as_of or _dt.date.today().isoformat()
        g = governing_from_spec(self.base_spec, self.updates, as_of)
        state = InformationState(evidence_from(evidence), g.fact_authority)
        proposed = calls_from(calls)
        j = check_actions(g.current, state, proposed, g.irreversible, plan=False)
        e = check_entitlement(g, state, proposed)
        by_step = {c.step: c for c in j.checks}

        blocked = [c.proposed for c in j.checks if c.proposed is not None and c.irreversible and c.verdict != JUSTIFIED]
        review = [c.proposed for c in j.checks if c.proposed is not None and not c.irreversible and c.verdict != JUSTIFIED]
        owed = [f.owed for f in e.findings if f.status == ENTITLED and f.proposed != f.owed]
        unsettled = [c.step for c in j.checks if c.proposed is not None and c.irreversible and c.verdict == "UNJUSTIFIED"]
        unsettled += [f.step for f in e.findings if f.status == POSSIBLE and f.proposed is None and f.step not in unsettled]
        ask = []
        if unsettled:
            plan = verification_plan(g.current, state, unsettled)
            for fct in (plan.minimal_sets[0] if plan.minimal_sets else []):
                ask.append({"fact": fct, "from": list(g.fact_authority.get(fct, []))})
        outcome = ("deliver" if owed else "block" if blocked else "verify" if ask else "review" if review else "allow")
        from contradish.entitlement import _settled_situation
        sit = _settled_situation(g.current, state, (g.base,))
        required = [c for c in g.current.run(sit).values() if c is not None] if sit is not None else None

        reasons = []
        for f in e.findings:
            if f.status == ENTITLED and f.proposed != f.owed:
                why = {"missed_change": "an amendment in force entitles it ("
                                        + ", ".join(f.amendments) + f"; before it: {f.old_rule_would_say})",
                       "evidence_sufficient": "what is already established entitles it"}.get(f.cause, "")
                ev = "; ".join(f"{x.source}: {x.describe()}" for x in f.sufficient_evidence)
                reasons.append(f"owed {_ct(f.owed)}, proposed {_ct(f.proposed)}: {why}. Enough already: {ev}.")
        for c in blocked:
            st = by_step[next(s for s, x in by_step.items() if x.proposed == c)]
            reasons.append(f"blocked {_ct(c)}: irreversible, and the established evidence "
                           + ("rules it out" if st.verdict == "WRONG" else
                              f"permits it in only {st.permitted_in} of {st.of} situations it allows") + ".")
        if ask:
            reasons.append("ask for: " + "; ".join(f"{a['fact']} from {' or '.join(a['from']) or '(no source)'}" for a in ask))
        for c in review:
            reasons.append(f"review {_ct(c)}: reversible, not justified by what is established.")

        derivation = {
            "applied_updates": [a["id"] for a in g.applied],
            "steps": {c.step: {"proposed": c.proposed.to_dict() if c.proposed else None, "verdict": c.verdict,
                               "permitted_in": c.permitted_in, "of": c.of} for c in j.checks},
            "benefits": {f.step: {"status": f.status, "owed": f.owed.to_dict() if f.owed else None,
                                  "verdict": f.verdict, "cause": f.cause, "amendments": f.amendments}
                         for f in e.findings},
            "outcome": outcome, "blocked": [c.to_dict() for c in blocked], "owed": [c.to_dict() for c in owed],
            "ask": ask, "review": [c.to_dict() for c in review],
            "required": [c.to_dict() for c in required] if required is not None else None,
        }
        body = {"schema": DECISION_SCHEMA, "kind": "decision", "policy": self.base_spec, "updates": self.updates,
                "fact_authority": g.fact_authority, "irreversible_tools": sorted(g.irreversible),
                "benefit_steps": list(g.benefit_steps), "as_of": as_of, "evidence": [x.to_dict() for x in state.evidence],
                "proposed": [c.to_dict() for c in proposed], "derivation": derivation,
                "request_id": request_id, "context": context or {}, "mode": self.mode, "approval": self.approval,
                "logged_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        cert = dict(body, digest=_digest(body))
        d = Decision(outcome, self.mode, proposed, blocked, owed, ask, review, reasons, cert, required)
        if self.log:
            with open(self.log, "a", encoding="utf-8") as fh:
                fh.write(_canon(cert) + "\n")
        return d

    def wrap(self, agent: Callable, evidence: Callable, as_of: Optional[Callable] = None) -> Callable:
        """agent(*a, **kw) -> tool calls; evidence(*a, **kw) -> evidence for the same request. Returns a function
        with the agent's signature that returns (calls_to_run, decision)."""
        def guarded(*a, **kw):
            d = self.check(evidence(*a, **kw), agent(*a, **kw), as_of=as_of(*a, **kw) if as_of else None,
                           request_id=kw.get("request_id"))
            return d.calls_to_run(), d
        return guarded


# ─────────────────────────────────────────────────────────────────────────────
# Authoring checks: is the encoded policy what the team means?
# ─────────────────────────────────────────────────────────────────────────────

def run_examples(policy: Any, examples: Optional[list] = None) -> list:
    """
    Check the encoding against examples the team writes in their own terms:
        {"name": "...", "facts": {...}, "calls": [{"tool": ..., "args": {...}}, ...]}   the exact calls expected
        {"name": "...", "facts": {...}, "includes": ["dispense_refill"], "excludes": ["contact_prescriber"]}
    `examples` default to the policy file's own "examples". Returns [(name, problem)]; empty means all pass.
    """
    if isinstance(policy, dict):
        spec = dict(policy)
        spec.pop("updates", None)
    else:
        spec, _ = load_policy(policy)
    examples = examples if examples is not None else spec.pop("examples", [])
    spec.pop("examples", None)
    p = PolicyProgram(spec)
    problems = []
    for i, ex in enumerate(examples):
        name = ex.get("name", f"example {i + 1}")
        try:
            got = [c for c in p.run(ex["facts"]).values() if c is not None]
        except Exception as err:  # noqa: BLE001
            problems.append((name, f"could not evaluate: {err}"))
            continue
        tools = {c.tool for c in got}
        if "calls" in ex:
            want = calls_from(ex["calls"])
            if sorted(map(_ct, want)) != sorted(map(_ct, got)):
                problems.append((name, f"expected {sorted(map(_ct, want))}, the policy gives {sorted(map(_ct, got))}"))
        for t in ex.get("includes", []):
            if t not in tools:
                problems.append((name, f"expected a call to {t}; the policy gives {sorted(map(_ct, got))}"))
        for t in ex.get("excludes", []):
            if t in tools:
                problems.append((name, f"expected no call to {t}; the policy gives {sorted(map(_ct, got))}"))
    return problems


def check_policy(policy: Any) -> list:
    """[(level, message)] for a policy file: what would make the guard wrong or unusable."""
    if isinstance(policy, dict):
        spec = dict(policy)
        raw = spec.pop("updates", [])
    else:
        spec, raw = load_policy(policy)
    spec.pop("examples", None)
    out = []
    try:
        p = PolicyProgram(spec)
    except Exception as err:  # noqa: BLE001
        return [("error", f"the policy does not load: {err}")]
    auth = spec.get("fact_authority") or {}
    for f in p.facts:
        if f not in auth:
            out.append(("error", f"fact {f!r}: no source is declared to establish it (fact_authority)"))
        if p.facts[f].get("type") not in ("int", "enum", "bool", "number"):
            out.append(("error", f"fact {f!r}: type must be int, enum, bool or number"))
        if p.facts[f].get("type") == "int" and ("min" not in p.facts[f] or "max" not in p.facts[f]):
            out.append(("error", f"fact {f!r}: an int fact needs min and max"))
    tools = {t["tool"] for t in p.tools()}
    for t in spec.get("irreversible_tools") or []:
        if t not in tools:
            out.append(("error", f"irreversible tool {t!r} is not a tool of any step"))
    if not spec.get("irreversible_tools"):
        out.append(("warning", "no irreversible_tools declared: nothing will ever be blocked"))
    for s in spec.get("benefit_steps") or []:
        if s not in p.step_ids():
            out.append(("error", f"benefit step {s!r} is not a step"))
    if not spec.get("benefit_steps"):
        out.append(("warning", "no benefit_steps declared: withholding will not be checked"))
    for u in raw:
        if u.get("source") not in p.sources:
            out.append(("error", f"update {u.get('id')!r}: unknown source {u.get('source')!r}"))
        if u.get("effective") is not None and len(str(u["effective"])) != 10:
            out.append(("error", f"update {u.get('id')!r}: effective must be YYYY-MM-DD"))
    from contradish.action_frontier import situation_universe
    try:
        n = len(situation_universe([p]))
        out.append(("info", f"{n} situations in the policy's exact partition"))
    except ValueError as err:
        out.append(("error", str(err)))
    examples = (policy.get("examples") if isinstance(policy, dict) else load_policy_examples(policy))
    if not examples:
        out.append(("warning", "no examples: add a few cases in your own terms so `contradish policy test` can "
                               "check the encoding says what you mean"))
    return out


def load_policy_examples(name_or_path: str) -> list:
    path = name_or_path if name_or_path.endswith(".json") else os.path.join(
        os.path.dirname(__file__), "contracts", "programs", name_or_path + ".json")
    with open(path) as f:
        return json.load(f).get("examples", [])


# ─────────────────────────────────────────────────────────────────────────────
# The value report: what a period of decisions shows
# ─────────────────────────────────────────────────────────────────────────────

def _independent_check():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "evidence_check.py")
    spec = importlib.util.spec_from_file_location("contradish_evidence_check_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def value_report(log_path: str, verify: bool = True, examples: int = 5) -> tuple:
    """(markdown, summary) for a decision log. With `verify`, every decision is re-derived by the standalone
    checker first, and only verified decisions are counted."""
    certs = [json.loads(line) for line in open(log_path, encoding="utf-8") if line.strip()]
    checked = None
    if verify:
        chk = _independent_check()
        res = [chk.check_decision(c) for c in certs]
        checked = {"verified": sum(r["verdict"] == "VERIFIED" for r in res),
                   "rejected": sum(r["verdict"] == "REJECTED" for r in res),
                   "unchecked": sum(r["verdict"] == "UNCHECKED" for r in res)}
        certs = [c for c, r in zip(certs, res) if r["verdict"] == "VERIFIED"]
    out = {o: 0 for o in OUTCOMES}
    withheld_by, cause_by, amendments = {}, {}, {}
    blocked_by, asked_by = {}, {}
    ex = {o: [] for o in OUTCOMES}
    requests = set()
    dates = sorted(c.get("as_of") for c in certs if c.get("as_of"))
    modes = sorted({c.get("mode") for c in certs})
    for c in certs:
        d = c["derivation"]
        o = d["outcome"]
        out[o] += 1
        requests.add(c.get("request_id") or c["digest"])
        for sid, b in d["benefits"].items():
            if b["status"] == "ENTITLED" and b["verdict"] != "HONORED":
                tool = b["owed"]["tool"]
                withheld_by[tool] = withheld_by.get(tool, 0) + 1
                cause_by[b["cause"]] = cause_by.get(b["cause"], 0) + 1
                if b["cause"] == "missed_change":
                    for a in b.get("amendments") or []:
                        amendments[a] = amendments.get(a, 0) + 1
        for call in d["blocked"]:
            blocked_by[call["tool"]] = blocked_by.get(call["tool"], 0) + 1
        for a in d["ask"]:
            key = f"{a['fact']} (from {' or '.join(a['from'])})"
            asked_by[key] = asked_by.get(key, 0) + 1
        if len(ex[o]) < examples and o != "allow":
            what = {"deliver": ", ".join(_ct(Call.from_dict(x)) for x in d["owed"]),
                    "block": ", ".join(_ct(Call.from_dict(x)) for x in d["blocked"]),
                    "verify": ", ".join(a["fact"] for a in d["ask"]),
                    "review": ", ".join(_ct(Call.from_dict(x)) for x in d["review"])}[o]
            ex[o].append((c.get("request_id") or c["digest"][:19], c.get("as_of"), what))
    n = len(certs)
    n_owed = sum(bool(c["derivation"]["owed"]) for c in certs)
    n_blocked = sum(bool(c["derivation"]["blocked"]) for c in certs)
    n_ask = sum(bool(c["derivation"]["ask"]) for c in certs)
    n_review = sum(bool(c["derivation"]["review"]) for c in certs)
    pl = lambda k, w: f"{k} {w}" + ("" if k == 1 else "s")
    shadow = modes == ["shadow"]
    would = "would have been" if shadow else "were"
    L = ["# What the guard found", ""]
    L.append(f"{n} decisions on {len(requests)} requests" + (f", {dates[0]} to {dates[-1]}" if dates else "")
             + f" (mode: {', '.join(m for m in modes if m)}).")
    if checked is not None:
        L.append(f"Every decision was re-derived independently by `evidence_check.py` before counting: "
                 f"{checked['verified']} verified, {checked['rejected']} rejected, {checked['unchecked']} too large "
                 "to enumerate. Only verified decisions are counted below.")
    L += ["", "| outcome | decisions | share |", "|---|---|---|"]
    for o in OUTCOMES:
        L.append(f"| {o} | {out[o]} | {out[o] / n:.0%} |" if n else f"| {o} | 0 | |")
    L += ["", f"## Withheld from people: {pl(n_owed, 'decision')}",
          f"Something the established evidence already entitled the person to {would} withheld, delayed or cut short."]
    for t, k in sorted(withheld_by.items(), key=lambda x: -x[1]):
        L.append(f"- {t}: {pl(k, 'time')}")
    if cause_by:
        L.append("Why: " + "; ".join(f"{ {'missed_change': 'an amendment in force was not applied', 'evidence_sufficient': 'the evidence already settled it'}.get(k, k)} ({v})" for k, v in cause_by.items()))
    if amendments:
        L.append("Amendments missed: " + ", ".join(f"{a} ({k})" for a, k in amendments.items()))
    L += ["", f"## Irreversible actions without justification: {pl(n_blocked, 'decision')}",
          f"An irreversible call the established evidence did not justify {would} stopped before it ran."]
    for t, k in sorted(blocked_by.items(), key=lambda x: -x[1]):
        L.append(f"- {t}: {pl(k, 'call')}")
    L += ["", f"## Not settled yet: {pl(n_ask, 'decision')}", "The fewest facts that settle each case, and who can establish them:"]
    for t, k in sorted(asked_by.items(), key=lambda x: -x[1])[:10]:
        L.append(f"- {t}: {k}")
    L += ["", f"## For review: {pl(n_review, 'decision')}", "A reversible call the evidence did not justify."]
    for o in ("deliver", "block", "verify", "review"):
        if ex[o]:
            L += ["", f"### Examples: {o}", "", "| request | date | what |", "|---|---|---|"]
            L += [f"| {r} | {d} | {w} |" for r, d, w in ex[o]]
    L += ["", "Each line above is backed by a certificate in the log. To check any of them without trusting contradish: "
              f"`python contradish/evidence_check.py {os.path.basename(log_path)}`."]
    summary = {"decisions": n, "requests": len(requests), "outcomes": out, "withheld_by_tool": withheld_by,
               "withheld_by_cause": cause_by, "amendments_missed": amendments, "blocked_by_tool": blocked_by,
               "asked": asked_by, "independent_check": checked, "period": [dates[0], dates[-1]] if dates else None}
    return "\n".join(L), summary
