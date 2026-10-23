"""
contradish/drafting.py -- from a written policy to a reviewed, signed-off policy.json.

Encoding a policy by hand is the slowest part of adopting the guard. A model can draft it; nothing a model drafts
should govern anything until a person has checked it. This module drafts, then makes the checking fast and exact:

    1. ENCODER   a model reads the policy text and writes policy.json: facts, clauses with readable conditions, the
                 tool calls each clause prescribes, which tools are irreversible, which deliver a benefit, and, for
                 every clause, number and fact, the exact sentence of the policy it came from.
    2. GROUNDING deterministic, no model: every quoted source must appear in the policy text (whitespace and case
                 aside); every number in a clause must appear in its quoted source; every sentence of the policy
                 that no element cites is listed as uncovered; the draft must compile and pass `check_policy`.
                 Which system establishes each fact is rarely in a policy's text, so the draft's `fact_authority` is
                 always marked as assumed until a person confirms it.
    3. EXAMINER  a second, independent model call sees only the policy text and the names of the facts and tools
                 (not the draft's logic) and writes test cases: situations and the calls it expects. The draft is
                 run on every case; each disagreement is a place where the encoding, the examiner or the policy's
                 wording is wrong, and a person decides which.
    4. BOUNDARY  deterministic: from the draft's exact situation partition, one question per threshold and per
                 value of every fact, around a typical case, in plain words, with what the draft decides. A reviewer
                 answers questions about outcomes, not code; confirmed answers become examples.
    5. SIGN-OFF  the draft has status "draft". `approve` records who approved it, when, in what role, bound to the
                 digest of everything the guard reads. The guard refuses to enforce a draft, or a policy edited after
                 its approval; every decision certificate carries the approval.

    contradish guard draft policy.md --model anthropic:claude-sonnet-4-5 --out policy.json
    contradish guard review policy.json            # review.md: sources, flags, uncovered text, disagreements, boundary
    contradish guard approve policy.json --by "Dana Reyes" --role "pharmacy operations lead"

Models are reached over plain HTTPS (anthropic, openai, groq, openrouter, together, mistral, deepseek, xai, gemini,
ollama) or through any function chat(messages) -> str.

Behavioral Update Fidelity was introduced by Michele Joseph in 2026.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from contradish.policy_program import PolicyProgram, ProgramError

__all__ = ["Draft", "draft_policy", "examine", "boundary_questions", "grounding", "review_markdown",
           "approve", "approval_status", "approval_digest", "disagreements", "blockers", "chat_for_model", "ENCODER_PROMPT", "EXAMINER_PROMPT"]

APPROVAL_FIELDS = ("status", "approval", "provenance", "assumptions", "review")


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def chat_for_model(model: str, max_tokens: int = 8000) -> Callable[[list], str]:
    """'provider:model' -> chat(messages) -> str, over HTTPS, no SDK (the study runner's transport)."""
    from contradish.study import call_model, parse_model
    spec = parse_model(model)
    return lambda messages: call_model(spec, messages, max_tokens=max_tokens, temperature=0.0)


def _json_from(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("the model's reply contains no JSON object")
    raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # common model slips: trailing commas
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))


# ─────────────────────────────────────────────────────────────────────────────
# 1. The encoder
# ─────────────────────────────────────────────────────────────────────────────

ENCODER_PROMPT = """You encode a written policy as a machine-checkable policy file. Be literal: encode what the text
says, not what would be sensible. When the text is silent or ambiguous, say so in "assumptions" instead of guessing
silently.

Reply with ONE JSON object:
{
  "id": "short_snake_case_name",
  "title": "...",
  "facts": {"<fact>": {"type": "int"|"enum"|"bool"|"number", "min": .., "max": .., "values": [..], "default": ..,
                       "source": "<exact quote from the policy that makes this fact matter>"}},
  "fact_authority": {"<fact>": ["<system or role that would establish it>"]},
  "sources": {"policy_owner": {"governs": ["*"]}},
  "clauses": {
    "<C1>": {"text": "<the rule in one sentence>", "source": "<exact quote>",
             "params": {"<name>": <number>}, "define": {"<name>": "<readable condition>"},
             "steps": {"<step>": {"tool": "<snake_case_tool>", "when": "<readable condition>",
                                  "args": {"<arg>": "<readable expression>"}, "description": "..."}}}
  },
  "irreversible_tools": ["<tools whose effect cannot be undone: money moved, goods dispensed, a person told>"],
  "benefit_steps": ["<steps that deliver something to the person>"],
  "assumptions": ["<everything you had to assume>"]
}

Rules:
- Every number the policy states becomes a param of the clause that states it. Conditions refer to params by name.
- Conditions use: and, or, not, == != < <= > >=, + - * /, x if c else y, min(), max(), round(), 'quoted text',
  x in ('a', 'b'), true, false. Names are facts, params or other definitions; each name is owned by one clause.
- Ints need min and max. Enums list every value.
- "source" must be copied exactly from the policy text: whole sentences or clauses, no paraphrase.
- Every step needs its own tool. Use the tool names given below when they fit.
{tools}
POLICY TEXT:
{text}"""

EXAMINER_PROMPT = """You check a policy encoding written by someone else, without seeing it. From the policy text alone,
write test cases that a careful reader would agree on, especially near every limit and exception the text states.

Facts you can set (use exactly these names and values): {facts}
Tools the system can call: {tools}

Reply with ONE JSON object:
{{"cases": [{{"name": "<short description>", "facts": {{"<fact>": <value>, ...}},
             "includes": ["<tools the policy requires here>"], "excludes": ["<tools it forbids here>"],
             "source": "<exact quote from the policy this case tests>"}}]}}

Give values for every fact in every case. Write {n} cases.

POLICY TEXT:
{text}"""


@dataclass
class Draft:
    spec: dict                                   # the draft policy.json (status "draft")
    flags: list = field(default_factory=list)    # [(level, where, message)]
    uncovered: list = field(default_factory=list)
    disagreements: list = field(default_factory=list)
    boundary: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def errors(self) -> list:
        return [f for f in self.flags if f[0] == "error"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _sentences(text: str) -> list:
    out = []
    for block in re.split(r"\n\s*\n|\n(?=\s*[-*\d#])", text):
        if block.strip().startswith("#"):
            continue                                   # a heading is not a rule
        for s in re.split(r"(?<=[.;:!?])\s+(?=[A-Z(\[])", block.strip()):
            s = s.strip(" -*#\t\n")
            if len(re.findall(r"[A-Za-z]{2,}", s)) >= 4:
                out.append(s)
    return out


def _numbers(s: str) -> set:
    return {float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", (s or "").replace(",", ""))}


def grounding(spec: dict, text: str) -> tuple:
    """([(level, where, message)], [uncovered sentences]). Deterministic: no model."""
    flags = []
    body = _norm(text)
    cited = []
    for where, src in _sources(spec):
        if not src:
            flags.append(("warning", where, "no source quote: nothing in the policy text is cited for this"))
            continue
        cited.append(_norm(src))
        if _norm(src) not in body:
            flags.append(("error", where, f"the quoted source is not in the policy text: {src[:90]!r}"))
    for cid, c in (spec.get("clauses") or {}).items():
        src = c.get("source") or ""
        nums = _numbers(src) | _numbers(c.get("text", ""))
        for name, v in (c.get("params") or {}).items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or float(v) in nums:
                continue
            if any(abs(float(v) * k - n) < 1e-9 for n in nums for k in (100, 0.01)):
                flags.append(("review", f"clause {cid} param {name}",
                              f"{v} is the source's number rescaled (percent vs fraction): check the conditions use it so"))
            else:
                flags.append(("error", f"clause {cid} param {name}",
                              f"the value {v} does not appear in the clause's source quote"))
    for f, fs in (spec.get("facts") or {}).items():
        if f in (spec.get("fact_authority") or {}):
            flags.append(("review", f"fact {f}", f"who establishes it is assumed: {spec['fact_authority'][f]}. "
                                                 "Confirm the system of record."))
    uncovered = [s for s in _sentences(text) if not any(_norm(s) in c or c in _norm(s) for c in cited if c)]
    return flags, uncovered


def _sources(spec: dict):
    for f, fs in (spec.get("facts") or {}).items():
        yield f"fact {f}", fs.get("source")
    for cid, c in (spec.get("clauses") or {}).items():
        yield f"clause {cid}", c.get("source")


def _clean(spec: dict) -> dict:
    """Move provenance out of the parts the policy program reads, and fill structural defaults."""
    s = copy.deepcopy(spec)
    prov = {"facts": {}, "clauses": {}}
    for f, fs in (s.get("facts") or {}).items():
        prov["facts"][f] = fs.pop("source", None)
        if fs.get("type") == "enum" and "default" not in fs and fs.get("values"):
            fs["default"] = fs["values"][0]
        if fs.get("type") == "bool" and "default" not in fs:
            fs["default"] = False
        if fs.get("type") == "int" and "default" not in fs and "min" in fs:
            fs["default"] = fs["min"]
    for cid, c in (s.get("clauses") or {}).items():
        prov["clauses"][cid] = c.pop("source", None)
    s.setdefault("schema", "contradish.policy_program/1.0")
    s.setdefault("version", "1")
    s.setdefault("sources", {"policy_owner": {"role": "policy_owner", "governs": ["*"]}})
    s["provenance"] = prov
    s["status"] = "draft"
    return s


def draft_policy(text: str, chat: Callable[[list], str], tools: Optional[list] = None, examiner: Optional[Callable] = None,
                 n_cases: int = 12, models: Optional[dict] = None) -> Draft:
    """Encode `text` with `chat`, ground it, cross-examine it (with `examiner`, default the same model in a separate
    call that never sees the draft), and derive the boundary questions."""
    tl = ("\nTool names to use when they fit: " + ", ".join(tools) + "\n") if tools else "\n"
    reply = chat([{"role": "user", "content": ENCODER_PROMPT.replace("{tools}", tl).replace("{text}", text)}])
    raw = _json_from(reply)
    assumptions = list(raw.pop("assumptions", []) or [])
    flags, uncovered = grounding(raw, text)
    spec = _clean(raw)
    spec["assumptions"] = assumptions
    spec["provenance"]["text"] = text
    spec["provenance"]["models"] = models or {}
    spec["provenance"]["drafted_at"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for a in assumptions:
        flags.append(("review", "assumption", a))
    from contradish.guard import check_policy
    try:
        program = PolicyProgram({k: v for k, v in spec.items() if k not in APPROVAL_FIELDS})
    except ProgramError as e:
        flags.append(("error", "compile", str(e)))
        return Draft(spec, flags, uncovered, raw={"encoder": reply})
    for lvl, msg in check_policy(spec):
        if lvl == "error":
            flags.append(("error", "lint", msg))
    d = Draft(spec, flags, uncovered, raw={"encoder": reply})
    ex = examine(spec, text, examiner or chat, n_cases)
    d.disagreements = ex["disagreements"]
    d.raw["examiner"] = ex["reply"]
    spec["review"] = {"examiner_cases": ex["cases"]}
    d.boundary = boundary_questions(program)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 2. The examiner
# ─────────────────────────────────────────────────────────────────────────────

def _facts_vocab(spec: dict) -> str:
    out = []
    for f, fs in spec["facts"].items():
        t = fs.get("type")
        if t == "enum":
            out.append(f"{f} (one of {fs['values']})")
        elif t == "int":
            out.append(f"{f} (integer {fs.get('min')} to {fs.get('max')})")
        else:
            out.append(f"{f} ({t})")
    return "; ".join(out)


def examine(spec: dict, text: str, chat: Callable[[list], str], n: int = 12) -> dict:
    p = PolicyProgram({k: v for k, v in spec.items() if k not in APPROVAL_FIELDS + ("updates", "examples")})
    tools = sorted({t["tool"] for t in p.tools()})
    prompt = EXAMINER_PROMPT.format(facts=_facts_vocab(spec), tools=", ".join(tools), n=n, text=text)
    reply = chat([{"role": "user", "content": prompt}])
    cases = list(_json_from(reply).get("cases", []))
    dis = disagreements(spec, cases)
    return {"cases": cases, "disagreements": dis, "reply": reply}


def disagreements(spec: dict, cases: list) -> list:
    """Examiner cases the policy, as now encoded, does not satisfy."""
    p = PolicyProgram({k: v for k, v in spec.items() if k not in APPROVAL_FIELDS + ("updates", "examples")})
    dis = []
    for c in cases:
        facts = {k: v for k, v in (c.get("facts") or {}).items() if k in p.facts}
        try:
            got = {x.tool for x in p.run(facts).values() if x is not None}
        except Exception as err:  # noqa: BLE001
            dis.append({"case": c.get("name"), "facts": facts, "source": c.get("source"),
                        "problem": f"the encoding cannot evaluate it: {err}"})
            continue
        missing = [t for t in c.get("includes", []) if t not in got]
        extra = [t for t in c.get("excludes", []) if t in got]
        if missing or extra:
            dis.append({"case": c.get("name"), "facts": facts, "source": c.get("source"),
                        "draft_calls": sorted(got), "missing": missing, "unexpected": extra})
    return dis


# ─────────────────────────────────────────────────────────────────────────────
# 3. Boundary questions
# ─────────────────────────────────────────────────────────────────────────────

def _say(f: str, v) -> str:
    if isinstance(v, bool):
        return f"{f.replace('_', ' ')}: {'yes' if v else 'no'}"
    return f"{f.replace('_', ' ')} {v}"


def _decision(p: PolicyProgram, sit: dict) -> list:
    return sorted(f"{c.tool}(" + ", ".join(f"{k}={v}" for k, v in c.args) + ")"
                  for c in p.run(sit).values() if c is not None)


def boundary_questions(p: PolicyProgram, base: Optional[dict] = None, limit: int = 40) -> list:
    """
    Plain-language questions at every place the draft's decision can change: starting from a typical case (each
    fact's default, or `base`), each int fact just below, at and above every threshold the policy compares it to,
    and every value of every enum and bool fact. Each: {"facts", "question", "draft_says", "changes"}. Only cases
    where the decision differs from a neighbor are kept, so every question is about a boundary.
    """
    from contradish.action_frontier import _thresholds
    import math
    sit0 = {f: (base or {}).get(f, s.get("default")) for f, s in p.facts.items()}
    out, seen = [], set()

    def add(sit, note):
        key = json.dumps(sit, sort_keys=True, default=str)
        if key in seen:
            return
        seen.add(key)
        facts = ", ".join(_say(f, v) for f, v in sit.items())
        out.append({"facts": dict(sit), "question": f"{facts}. {note}", "draft_says": _decision(p, sit)})

    d0 = _decision(p, sit0)
    add(sit0, "A typical case.")
    for f, s in p.facts.items():
        t = s.get("type")
        if t == "int":
            for th in sorted(_thresholds(p, f)):
                pts = sorted({math.floor(th) - 1, math.floor(th), math.ceil(th), math.ceil(th) + 1})
                prev = None
                for v in pts:
                    if s.get("min", -10 ** 9) <= v <= s.get("max", 10 ** 9):
                        sit = dict(sit0, **{f: v})
                        dv = _decision(p, sit)
                        if prev is not None and dv != prev[1]:
                            add(dict(sit0, **{f: prev[0]}), f"Just below where the decision changes ({f} {prev[0]}).")
                            add(sit, f"Where it changes ({f} {v}).")
                        prev = (v, dv)
        elif t in ("enum", "bool"):
            for v in ([False, True] if t == "bool" else s["values"]):
                if v != sit0[f]:
                    sit = dict(sit0, **{f: v})
                    if _decision(p, sit) != d0:
                        add(sit, f"Only {f.replace('_', ' ')} differs from the typical case.")
    return out[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# 4. The review sheet
# ─────────────────────────────────────────────────────────────────────────────

def _code(e) -> str:
    return e if isinstance(e, str) else json.dumps(e)


def review_markdown(spec: dict, text: Optional[str] = None, draft: Optional[Draft] = None) -> str:
    """The sheet a reviewer works through before approving. Regenerable from policy.json alone."""
    clean = {k: v for k, v in spec.items() if k not in APPROVAL_FIELDS}
    p = PolicyProgram(clean)
    prov = spec.get("provenance") or {}
    text = text if text is not None else prov.get("text")
    flags = list(draft.flags) if draft else []
    uncovered = list(draft.uncovered) if draft else []
    if text is not None and not draft:
        with_src = dict(clean,
                        facts={f: dict(fs, source=(prov.get("facts") or {}).get(f)) for f, fs in clean["facts"].items()},
                        clauses={c: dict(v, source=(prov.get("clauses") or {}).get(c)) for c, v in clean["clauses"].items()})
        flags, uncovered = grounding(with_src, text)
        flags += [("review", "assumption", a) for a in spec.get("assumptions") or []]
    from contradish.guard import check_policy, run_examples
    flags += [("error", "lint", m) for lvl, m in check_policy(clean) if lvl == "error" and not draft]
    flags += [("error", f"example {n}", m) for n, m in run_examples(dict(clean, examples=spec.get("examples", [])))]
    dis = list(draft.disagreements) if draft else disagreements(spec, (spec.get("review") or {}).get("examiner_cases", []))
    boundary = draft.boundary if draft else boundary_questions(p)
    st = approval_status(spec)
    L = [f"# Review: {spec.get('title') or spec.get('id')}", "",
         f"Status: **{st['state']}**" + (f" ({st['detail']})" if st.get("detail") else ""), ""]
    errs = [f for f in flags if f[0] == "error"]
    L += [f"## 1. Must fix before approval ({len(errs)})", ""]
    L += [f"- **{w}**: {m}" for _, w, m in errs] or ["- nothing"]
    L += ["", f"## 2. Disagreements with an independent reading ({len(dis)})", "",
          "A second reading of the policy text, which never saw this encoding, expects something different here. "
          "Decide which is right: the encoding, the test, or the policy's wording.", ""]
    for d in dis:
        what = d.get("problem") or (f"the encoding calls {', '.join(d.get('draft_calls') or []) or 'nothing'}"
                                    + (f"; the reading expects {', '.join(d['missing'])}" if d.get("missing") else "")
                                    + (f"; the reading forbids {', '.join(d['unexpected'])}" if d.get("unexpected") else ""))
        facts = ", ".join(_say(f, v) for f, v in (d.get("facts") or {}).items())
        L.append(f"- **{d['case']}** ({facts}): {what}. Policy: \"{(d.get('source') or '')[:200]}\"")
    if not dis:
        L.append("- none")
    L += ["", "## 3. Each rule against the sentence it came from", ""]
    for cid, c in clean["clauses"].items():
        L.append(f"**{cid}**: {c.get('text', '')}")
        src = (prov.get("clauses") or {}).get(cid)
        L.append(f"> {src}" if src else "> (no source cited)")
        for k, v in (c.get("params") or {}).items():
            L.append(f"- number `{k}` = {v}")
        for k, v in (c.get("define") or {}).items():
            L.append(f"- `{k}` means `{_code(v)}`")
        for sid, s in (c.get("steps") or {}).items():
            irr = " (irreversible)" if s["tool"] in (spec.get("irreversible_tools") or []) else ""
            ben = " (a benefit to the person)" if sid in (spec.get("benefit_steps") or []) else ""
            L.append(f"- calls `{s['tool']}`{irr}{ben} when `{_code(s.get('when', True))}`")
        L.append("")
    L += ["## 4. Who establishes each fact (assumed until you confirm)", "",
          "| fact | established by | why it matters |", "|---|---|---|"]
    for f, fs in clean["facts"].items():
        L.append(f"| {f} | {', '.join(clean.get('fact_authority', {}).get(f, ['(nobody: fix this)']))} | "
                 f"{((prov.get('facts') or {}).get(f) or '')[:100]} |")
    L += ["", f"## 5. Boundary questions ({len(boundary)})", "",
          "Every place the draft's decision changes. Tick each one you agree with; add the ones you correct to "
          "`examples` so `contradish guard test` keeps them true.", ""]
    for q in boundary:
        L.append(f"- [ ] {q['question']} The draft decides: {', '.join(q['draft_says']) or 'no calls'}.")
    L += ["", f"## 6. Policy text no rule cites ({len(uncovered)})", ""]
    L += [f"- \"{u}\"" for u in uncovered] or ["- none"]
    rev = [f for f in flags if f[0] in ("review", "warning")]
    L += ["", f"## 7. Assumptions and other notes ({len(rev)})", ""]
    L += [f"- {w}: {m}" for _, w, m in rev] or ["- none"]
    L += ["", "When everything above is resolved: `contradish guard approve policy.json --by \"Name\" --role \"...\"`."]
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sign-off
# ─────────────────────────────────────────────────────────────────────────────

def approval_digest(spec: dict) -> str:
    """
    sha256 over everything that decides what the guard does: the compiled policy and its updates. Status, approval,
    provenance, review notes and examples are left out: examples and notes do not change a decision, so a reviewer
    can add confirmed examples without re-approving. Adding an update does change decisions, so it needs approval.
    """
    body = {k: v for k, v in spec.items() if k not in APPROVAL_FIELDS + ("updates", "examples")}
    canon = json.dumps({"policy": PolicyProgram(body).spec, "updates": spec.get("updates", [])},
                       sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


def blockers(spec: dict) -> dict:
    """What stands between this policy and approval: {"errors": [...], "disagreements": [...]}."""
    clean = {k: v for k, v in spec.items() if k not in APPROVAL_FIELDS}
    from contradish.guard import check_policy, run_examples
    errors = [m for lvl, m in check_policy(clean) if lvl == "error"]
    if not errors:
        errors += [f"example {n!r}: {m}" for n, m in run_examples(clean)]
    prov = spec.get("provenance") or {}
    if prov.get("text") and not errors:
        with_src = dict(clean,
                        facts={f: dict(fs, source=(prov.get("facts") or {}).get(f)) for f, fs in clean["facts"].items()},
                        clauses={c: dict(v, source=(prov.get("clauses") or {}).get(c)) for c, v in clean["clauses"].items()})
        errors += [f"{w}: {m}" for lvl, w, m in grounding(with_src, prov["text"])[0] if lvl == "error"]
    dis = [] if errors else disagreements(spec, (spec.get("review") or {}).get("examiner_cases", []))
    return {"errors": errors, "disagreements": dis}


def approve(spec: dict, by: str, role: str = "", note: str = "", at: Optional[str] = None,
            despite_disagreements: bool = False) -> dict:
    """
    A copy of `spec`, approved by `by`, bound to its digest. Refused while the policy has errors, failing examples or
    source quotes not in its text, and while an independent reading still disagrees with it, unless the reviewer
    says they checked each disagreement and the encoding is right (`despite_disagreements`, recorded).
    """
    if not by.strip():
        raise ValueError("approval needs a name")
    b = blockers(spec)
    if b["errors"]:
        raise ValueError("cannot approve: " + "; ".join(b["errors"]))
    if b["disagreements"] and not despite_disagreements:
        raise ValueError(f"cannot approve: {len(b['disagreements'])} disagreement(s) with the independent reading "
                         "are unresolved: " + "; ".join(str(d["case"]) for d in b["disagreements"])
                         + ". Fix the encoding, or confirm the encoding is right and approve despite them.")
    out = copy.deepcopy(spec)
    out["status"] = "approved"
    out["approval"] = {"by": by, "role": role, "note": note,
                       "at": at or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "digest": approval_digest(spec)}
    if b["disagreements"]:
        out["approval"]["overrode"] = [d["case"] for d in b["disagreements"]]
    return out


def approval_status(spec: dict) -> dict:
    """{"state": "unreviewed" | "draft" | "approved" | "edited after approval", "detail": ...}"""
    st = spec.get("status")
    ap = spec.get("approval")
    if ap:
        if ap.get("digest") != approval_digest(spec):
            return {"state": "edited after approval",
                    "detail": f"approved by {ap.get('by')} on {ap.get('at', '')[:10]}, changed since"}
        return {"state": "approved", "detail": f"by {ap.get('by')}" + (f", {ap['role']}" if ap.get("role") else "")
                + f", {ap.get('at', '')[:10]}"}
    if st == "draft":
        return {"state": "draft", "detail": "drafted by a model; not reviewed"}
    return {"state": "unreviewed", "detail": "written by hand; no approval recorded"}
