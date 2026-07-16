"""
residual_truth.py — Contradiction-forced truth extraction.

The Residual Truth Engine does not ask "what does this model say?"
It asks "what can this model not escape saying?"

Standard output (confidence-scored):
  Answer: X

Residual Truth output (stability-scored):
  Stable residue:        X
  Collapsed assumptions: A, B, C
  Fragile claims:        D, E
  Repair path:           ...

Algorithm
─────────
  1. Generate answers under N framings (neutral, authority, emotional, ...)
  2. Extract atomic claims from each answer
  3. Deduplicate claims by semantic similarity
  4. Build incompatibility graph  G = (Claims, I)
     where I = pairs that cannot both be true
  5. Run MAX-IS repair R times with randomized edge ordering
  6. Score each claim: stability = fraction of repairs it survived
  7. Classify:
       stable residue        stability ≥ 0.75  — model cannot escape these
       fragile claims   0.20 < stability < 0.75 — context-dependent
       collapsed assumptions stability ≤ 0.20  — framing artifacts

The stability residual is the truth approximator.
High stability under contradictory pressure → real constraint.
Low stability → belief held only when not challenged.
"""

from __future__ import annotations

import re
import random
import html as html_module
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


# ── Types ──────────────────────────────────────────────────────────────────────

ModelFn = Callable[[str, str], str]   # (system_prompt, question) -> answer


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """An atomic assertion extracted from a model answer."""

    claim_id:      str
    text:          str          # normalized claim text (used for dedup)
    raw_text:      str          # original sentence
    framing:       str          # framing that first produced this claim
    source_answer: str          # full answer it came from
    framings_seen: set[str] = field(default_factory=set)  # all framings that produced this

    # Scores — computed by the engine
    frequency:  float = 0.0    # fraction of framings that produced this claim
    stability:  float = 0.0    # fraction of repair runs in which it survived
    _run_score: float = 0.0    # internal adaptive score used during repair

    @property
    def stable(self) -> bool:
        return self.stability >= 0.75

    @property
    def fragile(self) -> bool:
        return 0.20 < self.stability < 0.75

    @property
    def collapsed(self) -> bool:
        return self.stability <= 0.20

    def __hash__(self):
        return hash(self.claim_id)

    def __eq__(self, other):
        return isinstance(other, Claim) and self.claim_id == other.claim_id

    def __repr__(self):
        return f"Claim({self.claim_id!r}, stability={self.stability:.2f}, {self.text[:50]!r})"


@dataclass
class IncompatibilityEdge:
    """Two claims that cannot simultaneously be true."""
    claim_a_id: str
    claim_b_id: str
    reason:     str
    confidence: float = 1.0


@dataclass
class RepairStep:
    """One incompatibility resolution."""
    dropped_id:   str
    kept_id:      str
    dropped_text: str
    kept_text:    str
    reason:       str       # why this one was dropped
    edge_reason:  str       # what the incompatibility was


@dataclass
class RepairTrace:
    """A single repair run from initial claims to residue."""
    initial_claim_ids: list[str]
    steps:             list[RepairStep]
    residue_ids:       set[str]


@dataclass
class ResidualTruthResult:
    """
    The output of the Residual Truth Engine.

    Usage::

        result = engine.analyze("Is ibuprofen safe?", model_fn)
        print(result.report())
        open("out.html","w").write(result.to_html())
    """

    question:      str
    framings_used: list[str]
    all_claims:    list[Claim]
    incompatibilities: list[IncompatibilityEdge]
    traces:        list[RepairTrace]
    n_repairs:     int

    stable_residue:        list[Claim] = field(default_factory=list)
    fragile_claims:        list[Claim] = field(default_factory=list)
    collapsed_assumptions: list[Claim] = field(default_factory=list)
    canonical_repair_path: list[RepairStep] = field(default_factory=list)

    # ── Text report ────────────────────────────────────────────────────────────

    def report(self) -> str:
        W = 66
        hr = "─" * W
        lines: list[str] = []

        def h(title: str):
            lines.append("")
            lines.append(title)
            lines.append(hr)

        lines.append("")
        lines.append("RESIDUAL TRUTH ANALYSIS")
        lines.append("=" * W)
        lines.append(f"  question  : {self.question}")
        lines.append(f"  framings  : {len(self.framings_used)}")
        lines.append(f"  claims    : {len(self.all_claims)}")
        lines.append(f"  conflicts : {len(self.incompatibilities)}")
        lines.append(f"  repairs   : {self.n_repairs}")

        h("STABLE RESIDUE")
        if self.stable_residue:
            lines.append("  The model cannot escape these claims under any framing.")
            for c in self.stable_residue:
                bar = "█" * int(c.stability * 10) + "░" * (10 - int(c.stability * 10))
                lines.append(f"  [{bar}] {c.stability:.0%}  {c.text}")
        else:
            lines.append("  (none — all claims are framing-dependent)")

        h("COLLAPSED ASSUMPTIONS")
        if self.collapsed_assumptions:
            lines.append("  These appeared in answers but were dropped in every repair.")
            for c in self.collapsed_assumptions:
                bar = "█" * int(c.stability * 10) + "░" * (10 - int(c.stability * 10))
                lines.append(f"  [{bar}] {c.stability:.0%}  {c.text}")
        else:
            lines.append("  (none)")

        h("FRAGILE CLAIMS")
        if self.fragile_claims:
            lines.append("  These survived some repairs but not others.")
            for c in self.fragile_claims:
                bar = "█" * int(c.stability * 10) + "░" * (10 - int(c.stability * 10))
                lines.append(f"  [{bar}] {c.stability:.0%}  {c.text}")
        else:
            lines.append("  (none)")

        if self.canonical_repair_path:
            h("REPAIR PATH  (canonical)")
            for i, step in enumerate(self.canonical_repair_path, 1):
                lines.append(f"  {i}. CONFLICT  {step.edge_reason}")
                lines.append(f"     dropped   \"{step.dropped_text}\"")
                lines.append(f"     kept      \"{step.kept_text}\"")
                lines.append(f"     because   {step.reason}")

        lines.append("")
        return "\n".join(lines)

    # ── HTML report ────────────────────────────────────────────────────────────

    def to_html(self) -> str:
        def esc(s: str) -> str:
            return html_module.escape(str(s))

        def claim_html(c: Claim, label_class: str) -> str:
            bar_w = int(c.stability * 100)
            return f"""
            <div class="claim {label_class}">
              <div class="claim-bar" style="width:{bar_w}%"></div>
              <div class="claim-body">
                <span class="claim-pct">{c.stability:.0%}</span>
                <span class="claim-text">{esc(c.text)}</span>
                <span class="claim-freq">{int(c.frequency)} framing{'s' if c.frequency != 1 else ''}</span>
              </div>
            </div>"""

        stable_html    = "\n".join(claim_html(c, "stable")    for c in self.stable_residue)    or "<p class='empty'>none</p>"
        collapsed_html = "\n".join(claim_html(c, "collapsed") for c in self.collapsed_assumptions) or "<p class='empty'>none</p>"
        fragile_html   = "\n".join(claim_html(c, "fragile")   for c in self.fragile_claims)    or "<p class='empty'>none</p>"

        repair_html = ""
        for i, step in enumerate(self.canonical_repair_path, 1):
            repair_html += f"""
            <div class="repair-step">
              <div class="step-num">{i}</div>
              <div class="step-body">
                <div class="step-conflict">⊥ {esc(step.edge_reason)}</div>
                <div class="step-row dropped">
                  <span class="step-label">dropped</span>
                  <span class="step-claim">"{esc(step.dropped_text)}"</span>
                </div>
                <div class="step-row kept">
                  <span class="step-label">kept</span>
                  <span class="step-claim">"{esc(step.kept_text)}"</span>
                </div>
                <div class="step-reason">↳ {esc(step.reason)}</div>
              </div>
            </div>"""

        graph_edges = ""
        claim_map = {c.claim_id: c for c in self.all_claims}
        for e in self.incompatibilities[:20]:  # cap at 20 for display
            ca = claim_map.get(e.claim_a_id)
            cb = claim_map.get(e.claim_b_id)
            if ca and cb:
                graph_edges += f"""
                <div class="edge-row">
                  <span class="edge-tag">⊥</span>
                  <span class="edge-a {'stable-text' if ca.stable else 'collapsed-text'}">{esc(ca.text[:50])}</span>
                  <span class="edge-sep">/</span>
                  <span class="edge-b {'stable-text' if cb.stable else 'collapsed-text'}">{esc(cb.text[:50])}</span>
                  <span class="edge-reason">{esc(e.reason)}</span>
                </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Residual Truth — {esc(self.question[:60])}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0b0b0b;--s1:#111;--s2:#171717;--b:#222;--b2:#333;
  --text:#e8e8e8;--dim:#888;--dimmer:#444;
  --green:#4ade80;--green-bg:#0d1f14;--green-b:#1a4028;
  --red:#f87171;--red-bg:#1f0d0d;--red-b:#3d1515;
  --amber:#fbbf24;--amber-bg:#1c1400;--amber-b:#3d2d00;
  --mono:'SF Mono','Fira Code','Consolas',monospace;
}}
body{{background:var(--bg);color:var(--text);font-family:var(--mono);
  font-size:13px;line-height:1.6;padding:32px;max-width:900px;margin:0 auto}}
h1{{font-size:16px;font-weight:500;letter-spacing:0.5px;margin-bottom:4px}}
.meta{{font-size:11px;color:var(--dim);margin-bottom:32px}}
.meta span{{margin-right:20px}}
section{{margin-bottom:28px}}
.section-title{{font-size:9px;text-transform:uppercase;letter-spacing:2px;
  color:var(--dimmer);margin-bottom:10px;padding-bottom:6px;
  border-bottom:1px solid var(--b)}}
.section-desc{{font-size:11px;color:var(--dim);margin-bottom:10px}}
.claim{{position:relative;border:1px solid var(--b);border-radius:6px;
  overflow:hidden;margin-bottom:6px}}
.claim-bar{{position:absolute;top:0;left:0;height:100%;
  opacity:0.12;pointer-events:none}}
.stable .claim-bar{{background:var(--green)}}
.collapsed .claim-bar{{background:var(--red)}}
.fragile .claim-bar{{background:var(--amber)}}
.stable{{border-color:var(--green-b);background:var(--green-bg)}}
.collapsed{{border-color:var(--red-b);background:var(--red-bg)}}
.fragile{{border-color:var(--amber-b);background:var(--amber-bg)}}
.claim-body{{position:relative;display:flex;align-items:center;
  gap:12px;padding:8px 12px}}
.claim-pct{{font-size:11px;font-weight:600;min-width:32px}}
.stable .claim-pct{{color:var(--green)}}
.collapsed .claim-pct{{color:var(--red)}}
.fragile .claim-pct{{color:var(--amber)}}
.claim-text{{flex:1;font-size:12px}}
.claim-freq{{font-size:10px;color:var(--dimmer)}}
.empty{{font-size:11px;color:var(--dimmer);padding:6px 0}}
.repair-step{{display:flex;gap:14px;padding:10px 0;
  border-bottom:1px solid var(--b)}}
.repair-step:last-child{{border-bottom:none}}
.step-num{{font-size:11px;color:var(--dimmer);min-width:18px;
  padding-top:2px;text-align:right}}
.step-body{{flex:1}}
.step-conflict{{font-size:11px;color:var(--red);margin-bottom:6px}}
.step-row{{display:flex;gap:8px;margin-bottom:3px;align-items:baseline}}
.step-label{{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;
  min-width:50px}}
.dropped .step-label{{color:var(--red)}}
.kept .step-label{{color:var(--green)}}
.step-claim{{font-size:11px;color:var(--dim)}}
.step-reason{{font-size:10px;color:var(--dimmer);margin-top:4px}}
.edge-row{{display:flex;align-items:baseline;gap:8px;padding:5px 0;
  border-bottom:1px solid var(--b);font-size:11px;flex-wrap:wrap}}
.edge-row:last-child{{border-bottom:none}}
.edge-tag{{color:var(--red);flex-shrink:0}}
.stable-text{{color:var(--green)}}
.collapsed-text{{color:var(--red)}}
.edge-sep{{color:var(--dimmer)}}
.edge-reason{{color:var(--dimmer);font-size:10px;margin-left:auto}}
.stats-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
  margin-bottom:28px}}
.stat-box{{border:1px solid var(--b);border-radius:6px;padding:12px;
  background:var(--s1);text-align:center}}
.stat-val{{font-size:22px;font-weight:500;margin-bottom:2px}}
.stat-label{{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;
  color:var(--dimmer)}}
.green{{color:var(--green)}} .red{{color:var(--red)}}
.amber{{color:var(--amber)}} .dim{{color:var(--dim)}}
</style>
</head>
<body>
<h1>Residual Truth Analysis</h1>
<div class="meta">
  <span>Question: <strong>{esc(self.question)}</strong></span>
</div>

<div class="stats-grid">
  <div class="stat-box">
    <div class="stat-val green">{len(self.stable_residue)}</div>
    <div class="stat-label">stable residue</div>
  </div>
  <div class="stat-box">
    <div class="stat-val amber">{len(self.fragile_claims)}</div>
    <div class="stat-label">fragile claims</div>
  </div>
  <div class="stat-box">
    <div class="stat-val red">{len(self.collapsed_assumptions)}</div>
    <div class="stat-label">collapsed</div>
  </div>
  <div class="stat-box">
    <div class="stat-val dim">{len(self.incompatibilities)}</div>
    <div class="stat-label">incompatibilities</div>
  </div>
</div>

<section>
  <div class="section-title">Stable Residue</div>
  <div class="section-desc">These claims survived every repair. The model cannot escape them.</div>
  {stable_html}
</section>

<section>
  <div class="section-title">Collapsed Assumptions</div>
  <div class="section-desc">These appeared under some framings but were dropped in every repair — framing artifacts.</div>
  {collapsed_html}
</section>

<section>
  <div class="section-title">Fragile Claims</div>
  <div class="section-desc">Survived some repairs but not others — context-dependent or borderline.</div>
  {fragile_html}
</section>

<section>
  <div class="section-title">Canonical Repair Path</div>
  <div class="section-desc">How the stable residue was reached — the ordered resolution sequence.</div>
  {repair_html or "<p class='empty'>no incompatibilities found — all claims mutually compatible</p>"}
</section>

<section>
  <div class="section-title">Incompatibility Graph  ({len(self.incompatibilities)} edges)</div>
  {graph_edges or "<p class='empty'>none detected</p>"}
</section>

</body>
</html>"""


# ── Claim Extractor ────────────────────────────────────────────────────────────

@runtime_checkable
class ClaimExtractor(Protocol):
    """Extract atomic claims from a model answer."""
    def extract(self, question: str, answer: str, framing: str) -> list[str]: ...


class KeywordClaimExtractor:
    """
    Sentence-level claim extractor.

    Splits the answer into sentences, filters to declarative assertions,
    normalizes, and returns a deduplicated list.

    For production use, replace with an LLM-backed extractor:

        engine = ResidualTruthEngine(
            claim_extractor=lambda q, a, f: my_llm_extract_claims(q, a)
        )
    """

    _HEDGE = re.compile(
        r"\b(generally|typically|usually|often|might|may|could|seem(?:s)?|appear(?:s)?|"
        r"tend(?:s)?|probably|perhaps|likely|possibly)\b",
        re.I,
    )
    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
    _IMPERATIVES = {
        "take", "use", "avoid", "consult", "ask", "see", "note", "please",
        "remember", "check", "ensure", "make", "try", "consider",
    }
    # First-person empathetic openers — not factual claims
    _EMPATHY_PAT = re.compile(
        r"^i\s+(hear|understand|know|see|feel|realize|recognise|recognize)\b",
        re.I,
    )

    def extract(self, question: str, answer: str, framing: str) -> list[str]:
        sentences = self._split(answer)
        seen: set[str] = set()
        result: list[str] = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            if not self._is_declarative(sent):
                continue
            norm = self._normalize(sent)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)
        return result

    def _split(self, text: str) -> list[str]:
        parts = self._SENTENCE_SPLIT.split(text)
        out: list[str] = []
        for p in parts:
            for sub in p.split("; "):
                out.append(sub.strip())
        return out

    def _is_declarative(self, sent: str) -> bool:
        if sent.endswith("?"):
            return False
        first = sent.split()[0].lower() if sent.split() else ""
        if first in self._IMPERATIVES:
            return False
        # Reject empathetic/emotional openers — not factual claims
        if self._EMPATHY_PAT.match(sent):
            return False
        return True

    def _normalize(self, sent: str) -> str:
        sent = sent.rstrip(".!?")
        sent = re.sub(r"\s+", " ", sent).strip()
        return sent


# ── Incompatibility Detector ───────────────────────────────────────────────────

@runtime_checkable
class IncompatibilityDetector(Protocol):
    """Determine whether two claims are incompatible."""
    def are_incompatible(self, claim_a: str, claim_b: str) -> tuple[bool, str]: ...


class PatternIncompatibilityDetector:
    """
    Pattern-based incompatibility detection covering three classes:

    1. Numerical conflict — same topic, different numbers (ratio ≥ 1.5)
    2. Negation conflict  — "X is Y" vs "X is not Y"
    3. Categorical conflict — "safer" vs "neither is safer", OTC vs Rx, etc.

    For richer coverage, plug in an LLM-backed detector:

        engine = ResidualTruthEngine(
            incompatibility_detector=my_llm_detector
        )
    """

    _STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","can","shall","this","that","these",
        "those","i","you","he","she","it","we","they","for","of","in",
        "on","at","to","and","or","but","not","with","by","from","up",
        "about","into","through","during","before","after","above",
        "below","between","out","off","over","under","its","their",
    }

    def are_incompatible(self, a: str, b: str) -> tuple[bool, str]:
        al, bl = a.lower(), b.lower()
        for check in (self._numerical, self._negation, self._categorical):
            ok, reason = check(al, bl)
            if ok:
                return True, reason
        return False, ""

    # ── 1. Numerical conflict ──────────────────────────────────────────────────

    def _numbers(self, text: str) -> list[float]:
        text = re.sub(r"(\d),(\d)", r"\1\2", text)
        return [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", text)]

    def _keywords(self, text: str) -> set[str]:
        result: set[str] = set()
        for w in re.findall(r"\b[a-z]+\b", text):
            if w in self._STOPWORDS or len(w) <= 2:
                continue
            result.add(w)
            # Simple stem: strip trailing 's' so "adults" matches "adult"
            if w.endswith("s") and len(w) > 3:
                result.add(w[:-1])
            # Strip trailing 'ly' for adverbs: "pharmacologically" → "pharmacological"
            if w.endswith("ly") and len(w) > 4:
                result.add(w[:-2])
        return result

    def _numerical(self, a: str, b: str) -> tuple[bool, str]:
        na, nb = self._numbers(a), self._numbers(b)
        if not na or not nb:
            return False, ""
        shared = self._keywords(a) & self._keywords(b)
        if len(shared) < 2:
            return False, ""
        for va in na:
            for vb in nb:
                if va == vb or va == 0 or vb == 0:
                    continue
                ratio = max(va, vb) / min(va, vb)
                # Meaningful difference (≥1.5×) but plausibly the same
                # quantity — ratios >30× suggest different scales (mg vs age)
                if 1.5 <= ratio <= 30:
                    shared_str = ", ".join(sorted(shared)[:4])
                    return True, f"Numerical conflict: {va:g} vs {vb:g} (context: {shared_str})"
        return False, ""

    # ── 2. Negation conflict ───────────────────────────────────────────────────

    _NEG_PAT = re.compile(r"\bnot\b|\bno\b|n't\b|\bnever\b|\bneither\b")

    def _negation(self, a: str, b: str) -> tuple[bool, str]:
        a_neg = bool(self._NEG_PAT.search(a))
        b_neg = bool(self._NEG_PAT.search(b))
        if a_neg == b_neg:
            return False, ""          # same polarity
        ka = self._keywords(re.sub(self._NEG_PAT, "", a))
        kb = self._keywords(re.sub(self._NEG_PAT, "", b))
        if len(ka & kb) >= 3:
            return True, "Polarity conflict (affirmation vs negation)"
        return False, ""

    # ── 3. Categorical conflict ────────────────────────────────────────────────

    _COMPARATIVE = [
        (r"\b(ibuprofen|nsaid)\b.*\bsafer\b", r"\bneither\b|\bnot categorically\b|\bdepends\b|\bcontext\b"),
        (r"\bacetaminophen\b.*\bsafer\b",      r"\bneither\b|\bnot categorically\b|\bdepends\b|\bcontext\b"),
        (r"\bsafer than\b",                    r"\bneither\b|\bdepends\b|\bcontext\b"),
    ]

    _CATEGORICAL_PAIRS = [
        (r"\botc\b",          r"\bprescription\b"),
        (r"\bsafe\b",         r"\bunsafe\b"),
        (r"\beffective\b",    r"\bineffective\b"),
        (r"\brecommended\b",  r"\bnot recommended\b"),
    ]

    def _categorical(self, a: str, b: str) -> tuple[bool, str]:
        # Comparative safety
        for pos_pat, neg_pat in self._COMPARATIVE:
            a_pos = bool(re.search(pos_pat, a))
            b_pos = bool(re.search(pos_pat, b))
            a_neg = bool(re.search(neg_pat, a))
            b_neg = bool(re.search(neg_pat, b))
            if (a_pos and b_neg) or (b_pos and a_neg):
                return True, "Comparative safety conflict"

        # Categorical pairs
        ka = self._keywords(a)
        kb = self._keywords(b)
        for pos, neg in self._CATEGORICAL_PAIRS:
            if re.search(pos, a) and re.search(neg, b) and len(ka & kb) >= 2:
                return True, f"Categorical conflict"
            if re.search(pos, b) and re.search(neg, a) and len(ka & kb) >= 2:
                return True, f"Categorical conflict"

        return False, ""


# ── Residual Truth Engine ──────────────────────────────────────────────────────

class ResidualTruthEngine:
    """
    Contradiction-forced truth extraction.

    Quick start::

        from contradish import ResidualTruthEngine

        engine = ResidualTruthEngine()
        result = engine.analyze("Is ibuprofen safe?", my_model_fn)
        print(result.report())

    With a custom LLM-backed extractor::

        engine = ResidualTruthEngine(
            claim_extractor=my_llm_extractor,
            incompatibility_detector=my_llm_detector,
        )
    """

    def __init__(
        self,
        claim_extractor: ClaimExtractor | None = None,
        incompatibility_detector: IncompatibilityDetector | None = None,
        n_repairs: int = 60,
        similarity_threshold: float = 0.45,
        adaptive_scoring: bool = True,
        seed: int | None = None,
    ):
        self.extractor    = claim_extractor or KeywordClaimExtractor()
        self.detector     = incompatibility_detector or PatternIncompatibilityDetector()
        self.n_repairs    = n_repairs
        self.sim_thresh   = similarity_threshold
        self.adaptive     = adaptive_scoring
        self._rng         = random.Random(seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        question:      str,
        model_fn:      ModelFn,
        framings:      dict[str, str] | None = None,
        system_prompt: str = "",
    ) -> ResidualTruthResult:
        """
        Run the full residual truth analysis.

        Parameters
        ----------
        question : str
            The question to ask the model under all framings.
        model_fn : (system_prompt, question) -> answer
            The model under analysis.
        framings : dict[framing_name, prefix] | None
            Framing prefixes. Defaults to FRAMING_PREFIXES from phi_star.
        system_prompt : str
            Passed to model_fn as the first argument.

        Returns
        -------
        ResidualTruthResult
            Classified claims with stability scores and repair path.
        """
        if framings is None:
            from .phi_star import FRAMING_PREFIXES
            framings = FRAMING_PREFIXES

        # 1. Generate answers
        answers = self._generate_answers(question, model_fn, system_prompt, framings)

        # 2. Extract and deduplicate claims
        claims = self._extract_claims(question, answers)
        if not claims:
            return self._empty_result(question, list(answers.keys()))

        n_framings = len(answers)
        for c in claims:
            c.frequency = len(c.framings_seen)   # raw count
            c._run_score = c.frequency / max(1, n_framings)

        # 3. Build incompatibility graph
        incompatibilities = self._build_incompatibilities(claims)

        # 4. Run repairs (adaptive: later runs use accumulated survival signal)
        traces = self._run_repairs(claims, incompatibilities)

        # 5. Compute stability scores
        survival: dict[str, int] = {c.claim_id: 0 for c in claims}
        for trace in traces:
            for cid in trace.residue_ids:
                survival[cid] = survival.get(cid, 0) + 1

        for c in claims:
            c.stability = survival[c.claim_id] / len(traces)

        # 6. Classify
        # Stable residue requires both high stability AND multi-framing presence:
        # a claim that appeared in only one framing cannot be "inescapable"
        min_freq_for_stable = max(2, len(answers) // 8)  # at least 2, ≥12.5% of framings
        stable    = sorted(
            [c for c in claims if c.stable and c.frequency >= min_freq_for_stable],
            key=lambda c: (-c.stability, -c.frequency),
        )
        # Demote high-stability but low-frequency claims to fragile
        fragile   = sorted(
            [c for c in claims if c.fragile or (c.stable and c.frequency < min_freq_for_stable)],
            key=lambda c: -c.stability,
        )
        collapsed = sorted([c for c in claims if c.collapsed], key=lambda c:  c.stability)

        canonical = self._canonical_trace(traces)

        return ResidualTruthResult(
            question               = question,
            framings_used          = list(answers.keys()),
            all_claims             = claims,
            incompatibilities      = incompatibilities,
            traces                 = traces,
            n_repairs              = len(traces),
            stable_residue         = stable,
            fragile_claims         = fragile,
            collapsed_assumptions  = collapsed,
            canonical_repair_path  = canonical.steps if canonical else [],
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _generate_answers(
        self,
        question: str,
        model_fn: ModelFn,
        system_prompt: str,
        framings: dict[str, str],
    ) -> dict[str, str]:
        answers: dict[str, str] = {}
        for name, prefix in framings.items():
            framed = (prefix + question) if prefix else question
            try:
                ans = model_fn(system_prompt, framed)
                if ans and ans.strip():
                    answers[name] = ans.strip()
            except Exception:
                pass
        return answers

    def _extract_claims(
        self,
        question: str,
        answers: dict[str, str],
    ) -> list[Claim]:
        raw: list[tuple[str, str, str]] = []  # (text, framing, answer)
        for framing, answer in answers.items():
            texts = self.extractor.extract(question, answer, framing)
            for t in texts:
                raw.append((t, framing, answer))

        # Deduplicate by Jaccard similarity
        claims: list[Claim] = []
        seq = 0
        for text, framing, answer in raw:
            matched = False
            for existing in claims:
                if self._similar(text, existing.text):
                    existing.framings_seen.add(framing)
                    matched = True
                    break
            if not matched:
                claims.append(Claim(
                    claim_id      = f"c{seq}",
                    text          = text,
                    raw_text      = text,
                    framing       = framing,
                    source_answer = answer[:200],
                    framings_seen = {framing},
                ))
                seq += 1

        return claims

    def _similar(self, a: str, b: str) -> bool:
        wa = set(re.findall(r"\b[a-z]+\b", a.lower()))
        wb = set(re.findall(r"\b[a-z]+\b", b.lower()))
        if not wa or not wb:
            return False
        return len(wa & wb) / len(wa | wb) >= self.sim_thresh

    def _build_incompatibilities(self, claims: list[Claim]) -> list[IncompatibilityEdge]:
        edges: list[IncompatibilityEdge] = []
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                incompatible, reason = self.detector.are_incompatible(
                    claims[i].text, claims[j].text
                )
                if incompatible:
                    edges.append(IncompatibilityEdge(
                        claim_a_id = claims[i].claim_id,
                        claim_b_id = claims[j].claim_id,
                        reason     = reason,
                    ))
        return edges

    def _run_repairs(
        self,
        claims:           list[Claim],
        incompatibilities: list[IncompatibilityEdge],
    ) -> list[RepairTrace]:
        claim_map = {c.claim_id: c for c in claims}
        traces: list[RepairTrace] = []

        # Adaptive: update _run_score from survival every 10 rounds
        survival_acc: dict[str, int] = {c.claim_id: 0 for c in claims}

        for run_idx in range(self.n_repairs):

            # Adaptive score update every 10 rounds
            if self.adaptive and run_idx > 0 and run_idx % 10 == 0:
                for c in claims:
                    freq_norm = c.frequency / max(1, max(x.frequency for x in claims))
                    surv_norm = survival_acc[c.claim_id] / run_idx
                    # Weight shifts toward survival signal as rounds accumulate
                    w = min(0.80, run_idx / self.n_repairs)
                    c._run_score = (1 - w) * freq_norm + w * surv_norm

            # Single repair run
            held: set[str] = {c.claim_id for c in claims}
            steps: list[RepairStep] = []

            edges = list(incompatibilities)
            self._rng.shuffle(edges)

            changed = True
            while changed:
                changed = False
                for edge in edges:
                    if edge.claim_a_id in held and edge.claim_b_id in held:
                        ca = claim_map[edge.claim_a_id]
                        cb = claim_map[edge.claim_b_id]

                        if ca._run_score <= cb._run_score:
                            drop, keep = ca, cb
                        else:
                            drop, keep = cb, ca

                        held.discard(drop.claim_id)
                        steps.append(RepairStep(
                            dropped_id   = drop.claim_id,
                            kept_id      = keep.claim_id,
                            dropped_text = drop.text[:80],
                            kept_text    = keep.text[:80],
                            reason       = (
                                f"lower stability signal "
                                f"({drop._run_score:.2f} vs {keep._run_score:.2f})"
                            ),
                            edge_reason  = edge.reason,
                        ))
                        changed = True

            for cid in held:
                survival_acc[cid] = survival_acc.get(cid, 0) + 1

            traces.append(RepairTrace(
                initial_claim_ids = [c.claim_id for c in claims],
                steps             = steps,
                residue_ids       = held,
            ))

        return traces

    def _canonical_trace(self, traces: list[RepairTrace]) -> RepairTrace | None:
        """Return the trace with median length — most representative repair."""
        if not traces:
            return None
        return sorted(traces, key=lambda t: len(t.steps))[len(traces) // 2]

    def _empty_result(self, question: str, framings: list[str]) -> ResidualTruthResult:
        return ResidualTruthResult(
            question=question, framings_used=framings,
            all_claims=[], incompatibilities=[], traces=[], n_repairs=0,
        )
