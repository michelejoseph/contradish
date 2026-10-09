"""
contradish/counterexample.py -- shrink an unauthorized change to a minimal
counterexample, and certify it.

Given a situation in which an agent's action was captured by an update it
should have resisted, find the smallest version of that scenario that still
produces the failure. The scenario is broken into components:

    claim:<clause>    one edit of the update, together with the sentence of
                      the update's content that expresses it (Edit.says)
    context:<n>       a sentence of the update's content that expresses no edit
    fact:<name>       a fact of the situation that differs from its default
                      (removing it resets the fact to the default)

Delta debugging (Zeller and Hildebrandt, 2002) removes components while the
failure still reproduces. The result is 1-minimal: removing any single
remaining component makes the failure disappear. Each of those single
removals is run once more at the end and recorded as a minimality witness,
so the claim is checkable rather than asserted.

"Still reproduces" means, for the chosen step:
    - the reduced scenario still requires resisting (the unauthorized edits
      left would change the step if obeyed), so it is still a test of
      authority and not a trivial one; and
    - the agent's status on the step is `captured` in at least `k` of
      `trials` runs.
For a scripted agent trials=k=1 is exact. For a sampled model use e.g.
trials=5, k=3; the recorded witnesses then state how often each removal
reproduced.

The output is an evidence certificate (contradish/evidence.py) of kind
unauthorized_change for the minimal scenario, with a `minimization` block.
The independent checker verifies the certificate and that every recorded
witness failed to reproduce.

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from contradish.action_frontier import (
    Agent, build_agent_input, derive_action_frontier, status_of,
    observe, ActionCaseResult,
)
from contradish.evidence import certificate, seal
from contradish.policy_program import Edit, PolicyProgram, ProgramUpdate, calls_equal

__all__ = ["Component", "components_of", "minimize_unauthorized_change", "ddmin", "split_sentences"]


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]


@dataclass(frozen=True)
class Component:
    id: str
    kind: str      # claim | context | fact
    ref: str       # clause id, sentence index, or fact name
    text: str = ""


def components_of(program: PolicyProgram, update: ProgramUpdate, situation: dict) -> list:
    out = []
    claimed = set()
    for e in update.edits:
        out.append(Component(f"claim:{e.clause}", "claim", e.clause, e.says or ""))
        if e.says:
            claimed.add(e.says.strip())
    for i, s in enumerate(split_sentences(update.content)):
        if s.strip() not in claimed:
            out.append(Component(f"context:{i}", "context", str(i), s))
    full = program.complete_state(situation)
    for f, spec in program.facts.items():
        if "default" in spec and full[f] != spec["default"]:
            out.append(Component(f"fact:{f}", "fact", f, f"{f}={full[f]!r}"))
    return out


def _build(program: PolicyProgram, update: ProgramUpdate, situation: dict, keep: list) -> tuple:
    kept = {c.id for c in keep}
    edits = [e for e in update.edits if f"claim:{e.clause}" in kept]
    sentences = split_sentences(update.content)
    claimed = {e.says.strip(): e.clause for e in update.edits if e.says}
    text = []
    for i, s in enumerate(sentences):
        clause = claimed.get(s.strip())
        if clause is not None:
            if f"claim:{clause}" in kept:
                text.append(s)
        elif f"context:{i}" in kept:
            text.append(s)
    u = ProgramUpdate(update.id + ":min", update.source, update.channel, " ".join(text), edits)
    full = program.complete_state(situation)
    s = {}
    for f, spec in program.facts.items():
        if "default" in spec and full[f] != spec["default"] and f"fact:{f}" not in kept:
            s[f] = spec["default"]
        else:
            s[f] = full[f]
    return u, s


@dataclass
class _Probe:
    reproduced: bool
    captured: int
    trials: int
    valid: bool
    observed: list = field(default_factory=list)   # [(before_calls, after_calls)]


def ddmin(items: list, test: Callable[[list], bool]) -> list:
    """Zeller's ddmin: a 1-minimal sublist of `items` for which test() is True."""
    n = 2
    cur = list(items)
    while len(cur) >= 2:
        chunk = max(1, len(cur) // n)
        subsets = [cur[i:i + chunk] for i in range(0, len(cur), chunk)]
        reduced = False
        for sub in subsets:
            if test(sub):
                cur, n, reduced = sub, 2, True
                break
        if not reduced:
            for sub in subsets:
                comp = [x for x in cur if x not in sub]
                if comp and test(comp):
                    cur, n, reduced = comp, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(cur):
                break
            n = min(len(cur), n * 2)
    return cur


def minimize_unauthorized_change(program: PolicyProgram, update: ProgramUpdate, situation: dict, step: str,
                                 agent: Agent, agent_name: str = "", agent_kind: str = "scripted",
                                 model: Optional[str] = None, trials: int = 1, k: int = 1) -> dict:
    """
    Shrink a captured (unauthorized) change on `step` to a 1-minimal
    scenario and return its sealed certificate. Raises ValueError if the
    original scenario does not reproduce.
    """
    agent_name = agent_name or getattr(agent, "__name__", "agent")
    calls = {"n": 0}
    cache: dict = {}

    def probe(keep: list) -> _Probe:
        key = tuple(sorted(c.id for c in keep))
        if key in cache:
            return cache[key]
        u, s = _build(program, update, situation, keep)
        if not u.edits:
            pr = _Probe(False, 0, 0, False)
            cache[key] = pr
            return pr
        fr = derive_action_frontier(program, u, [s])
        v = fr.at(0, step)
        if not v.resist:
            pr = _Probe(False, 0, 0, False)
            cache[key] = pr
            return pr
        captured, obs = 0, []
        for _ in range(trials):
            calls["n"] += 1
            b0_list = agent(build_agent_input(program, s))
            b1_list = agent(build_agent_input(program, s, u))
            b0, _, _ = observe(program, b0_list)
            b1, _, _ = observe(program, b1_list)
            st = status_of(v.after, b0[step], b1[step], v.asserted)
            captured += st == "captured"
            obs.append((b0_list, b1_list))
        pr = _Probe(captured >= k, captured, trials, True, obs)
        cache[key] = pr
        return pr

    comps = components_of(program, update, situation)
    if not probe(comps).reproduced:
        raise ValueError("the original scenario does not reproduce an unauthorized change on step " + step)
    minimal = ddmin(comps, lambda keep: probe(keep).reproduced)

    witnesses = []
    for c in minimal:
        rest = [x for x in minimal if x.id != c.id]
        pr = probe(rest)
        witnesses.append({"component": c.id, "text": c.text, "reproduced": pr.reproduced,
                          "valid_scenario": pr.valid, "captured": pr.captured, "trials": pr.trials})
    if any(w["reproduced"] for w in witnesses):  # pragma: no cover - ddmin guarantees this
        raise AssertionError("ddmin result is not 1-minimal")

    u, s = _build(program, update, situation, minimal)
    fr = derive_action_frontier(program, u, [s])
    final = probe(minimal)
    b0_list, b1_list = final.observed[0]
    b0, _, _ = observe(program, b0_list)
    b1, _, _ = observe(program, b1_list)
    v = fr.at(0, step)
    st = status_of(v.after, b0[step], b1[step], v.asserted)
    case = ActionCaseResult(0, step, st, v.must, v.basis, v.resist, b0[step], b1[step],
                            v.must == "preserve" and not calls_equal(b0[step], b1[step]) and not v.after.permits(b1[step]),
                            st == "captured")
    raw = None
    if agent_kind == "model":
        raw = {"before": getattr(b0_list, "raw", ""), "after": getattr(b1_list, "raw", "")}
    cert = certificate(fr, case, agent_name, agent_kind, model, raw, kind="unauthorized_change",
                       note="minimal counterexample")
    cert = {k: v2 for k, v2 in cert.items() if k != "digest"}
    cert["minimization"] = {
        "method": "ddmin (Zeller & Hildebrandt 2002), then single-removal check",
        "original": {"update": update.to_dict(), "situation": program.complete_state(situation),
                     "components": [c.id for c in comps]},
        "kept": [{"component": c.id, "text": c.text} for c in minimal],
        "removed": [c.id for c in comps if c not in minimal],
        "oracle": {"trials": trials, "k": k, "agent_runs": calls["n"]},
        "witnesses": witnesses,
    }
    return seal(cert)
