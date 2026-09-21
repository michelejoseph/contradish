"""
policy_contradiction.py -- finds textual contradictions inside a company's
own published policy documents (Terms of Service, Privacy Policy, compliance
pages, and -- optionally -- marketing copy that makes policy-shaped claims).

This is the same underlying question contradish already asks of a MODEL's
answers ("does this system say the same thing twice, when it should?")
pointed at a different target: not "does the AI contradict itself across
rephrasings of one question," but "does the COMPANY contradict itself
across its own published sentences." Same shape of question, different
substrate -- prose the company itself wrote and shipped, not model output.

─────────────────────────────────────────────────────────────────────────────
WHY THIS MODULE EXISTS -- AND WHAT IT DOES NOT CLOSE
─────────────────────────────────────────────────────────────────────────────
This is a text-only, sentence-level scanner over documents a company
publishes. What it DOES do: extract normative/claim-bearing sentences from
one or more pages, cluster them by shared topic, and ask a judge (an LLM,
or a human reviewer using the same interface) whether any two claims commit
the company to incompatible things for some realistic real-world case --
e.g. "no refunds after 30 days" published on one page and "full refund any
time within 90 days" published on another.

What it does NOT do:
  - It does not verify the company's actual BEHAVIOR against any of its
    policy text -- that is a real-world-observation problem, structurally
    the same scope limit contradish.compliance_gap already states for model
    transcripts (verbal commitment vs. actual behavior is only checkable
    when both are visible in the same text). A company can publish two
    perfectly consistent policies and still violate both in practice; this
    module cannot see that.
  - It is not a formal theorem-prover over the policy text. Judging whether
    two sentences are "genuinely incompatible" requires real-world/legal
    reasoning (does "share with service providers under contract" conflict
    with "we don't sell your data"? -- no, those are different concepts,
    a careless keyword-matcher would flag it anyway). The judge step uses
    an LLM (or a human using the identical interface) and inherits that
    model's reasoning quality and its known failure modes: hallucinated
    "conflicts" between statements that are actually compatible once you
    read the qualifying clause, and missed conflicts that are only
    connected by real-world knowledge the model doesn't surface. Every
    finding is delivered with confidence + explanation so a human can
    re-check it fast; nothing here should be treated as a legal
    determination without that check.
  - Sentence extraction and topic clustering are both heuristic
    (regex/keyword-based, not a semantic parser). A claim phrased in a way
    the extractor doesn't recognize as claim-bearing is invisible to
    everything downstream. This is a recall/precision tradeoff stated
    plainly, not a hidden gap.

    from contradish.policy_contradiction import (
        extract_policy_claims, generate_candidate_pairs,
        PolicyContradictionJudge, audit_policies,
    )
    from contradish.llm import LLMClient

    claims = []
    claims += extract_policy_claims(open("legal.html").read(), source="legal.html")
    claims += extract_policy_claims(open("compliance.html").read(), source="compliance.html")

    judge = PolicyContradictionJudge(LLMClient())  # needs ANTHROPIC_API_KEY or OPENAI_API_KEY
    report = audit_policies(claims, judge, max_pairs=40)
    report.print_summary()

For a run with no LLM key available, ManualJudgeClient + audit_policies_manual()
give a human (or an LLM session standing in as reviewer) the same interface --
see policy-audit/run_policy_audit.py for a full worked example.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────

# Sentences worth comparing are the ones that actually commit the company
# to something: a modal ("will", "may not", "must"), a negation/absolute
# ("never", "always", "no", "only"), a right/obligation verb (share, sell,
# retain, delete, disclose, guarantee, refund, cancel, terminate, permit,
# require, collect, charge, waive, indemnify, warrant), or a concrete
# quantity (a number -- money, days, percentages) that policy terms are
# usually made of. Anything else (nav links, headings, boilerplate,
# copyright lines) is dropped.
_CLAIM_SIGNAL_RE = re.compile(
    r"\b(will|shall|must|may not|won'?t|don'?t|does not|do not|never|"
    r"always|only|no(?:ne|thing)?\b|all\b|share|shares|shared|sharing|"
    r"sell|sold|sells|retain|retains|delete|deletes|disclose|discloses|"
    r"guarantee|guarantees|refund|refunds|cancel|cancels|terminate|"
    r"terminates|permit|permits|require|requires|collect|collects|"
    r"charge|charges|waive|waives|indemnif\w*|warrant\w*|license|"
    r"licenses|liable|liability|responsible|entitled|eligible|"
    r"obligat\w*|consent\w*|opt[\s-]?in|opt[\s-]?out|\d)",
    re.IGNORECASE,
)

# Boilerplate/nav lines to drop outright even if they trip the signal regex.
_JUNK_RE = re.compile(
    r"^(contradish|©|copyright|all rights reserved|sign in|sign up|"
    r"docs|blog|demo|compliance|cai-bench|home)\W*$",
    re.IGNORECASE,
)

_SENT_SPLIT_RE = re.compile(
    r"(?<!\bU\.S)(?<!\bInc)(?<!\be\.g)(?<!\bi\.e)(?<!\bDr)(?<!\bMr)(?<!\bMs)"
    r"(?<=[.!?])\s+(?=[A-Z(\"])"
)

_WS_RE = re.compile(r"\s+")


@dataclass
class PolicyClaim:
    claim_id: str
    source: str              # e.g. "legal.html"
    section: str             # e.g. "Terms of Service > 5. Fees"
    text: str
    topics: set = field(default_factory=set, repr=False)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "source": self.source,
            "section": self.section,
            "text": self.text,
        }


def _split_sentences(block: str) -> list:
    block = _WS_RE.sub(" ", block).strip()
    if not block:
        return []
    parts = _SENT_SPLIT_RE.split(block)
    return [p.strip() for p in parts if p.strip()]


def _is_claim_bearing(sentence: str) -> bool:
    if len(sentence.split()) < 6:
        return False
    if _JUNK_RE.match(sentence.strip()):
        return False
    return bool(_CLAIM_SIGNAL_RE.search(sentence))


def extract_policy_claims(html: str, source: str) -> list:
    """
    Parse `html`, walk block-level content (p, li, td/th) in document order,
    track the nearest preceding heading (h1-h4) as a section breadcrumb,
    split each block into sentences, and keep the claim-bearing ones.

    Falls back to a regex HTML-stripper (no section tracking) if bs4 isn't
    installed -- degraded but functional, and says so in each claim's
    `section` field so a report never silently implies structure it
    doesn't have.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        return _extract_policy_claims_no_bs4(html, source)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    breadcrumb = []          # stack of (level, text)
    claims = []
    idx = 0

    body = soup.body or soup
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        name = el.name
        if name in ("h1", "h2", "h3", "h4"):
            level = int(name[1])
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            breadcrumb = [b for b in breadcrumb if b[0] < level]
            breadcrumb.append((level, text))
            continue

        block_text = el.get_text(" ", strip=True)
        if not block_text:
            continue
        section = " > ".join(t for _, t in breadcrumb) or "(no heading)"
        for sentence in _split_sentences(block_text):
            if not _is_claim_bearing(sentence):
                continue
            idx += 1
            claims.append(
                PolicyClaim(
                    claim_id=f"{source}#{idx}",
                    source=source,
                    section=section,
                    text=sentence,
                    topics=_topic_signature(sentence),
                )
            )
    return claims


def _extract_policy_claims_no_bs4(html: str, source: str) -> list:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    import html as html_mod
    text = html_mod.unescape(text)
    claims = []
    idx = 0
    for sentence in _split_sentences(text):
        if not _is_claim_bearing(sentence):
            continue
        idx += 1
        claims.append(
            PolicyClaim(
                claim_id=f"{source}#{idx}",
                source=source,
                section="(bs4 not installed -- no section tracking)",
                text=sentence,
                topics=_topic_signature(sentence),
            )
        )
    return claims


# ─────────────────────────────────────────────────────────────────────────
# Candidate pairing -- cheap, deterministic, no model calls. Keeps the
# judge's (expensive, LLM-backed) attention on pairs that plausibly share a
# real-world subject, instead of paying for O(n^2) unrelated comparisons.
# ─────────────────────────────────────────────────────────────────────────

_STOPWORDS = set(
    "a an the this that these those of to for in on at by with without "
    "and or but if then unless when where who whom which what how why "
    "you your we our us they their it its is are was were be been being "
    "do does did will would shall should may might must can could not no "
    "as from into out up down over under again further once here there "
    "all any both each few more most other some such only own same so "
    "than too very s t just don now".split()
)

# Curated synonym buckets for the topics policy documents actually argue
# about. Deliberately small and inspectable -- this is what decides which
# claims get compared, so it needs to be legible, not a black-box embedding.
_TOPIC_BUCKETS = {
    "data_sharing": {"share", "shares", "shared", "sharing", "sell", "sold",
                      "sells", "disclose", "discloses", "third", "party",
                      "parties", "provider", "providers", "transfer"},
    "data_deletion": {"delete", "deletes", "deletion", "retain", "retains",
                       "retention", "remove", "erase"},
    "confidentiality": {"confidential", "confidentiality", "nda", "private",
                         "privacy"},
    "pricing": {"free", "fee", "fees", "charge", "charges", "price",
                "pricing", "paid", "payment", "cost"},
    "liability": {"liable", "liability", "damages", "indemnif", "warrant",
                  "disclaim", "disclaimer"},
    "security": {"security", "secure", "encrypt", "encryption", "breach"},
    "ip": {"intellectual", "property", "license", "licenses", "ownership",
           "own", "owns"},
    "termination": {"terminate", "terminates", "termination", "suspend",
                     "cancel", "cancels", "cancellation"},
    "training_data": {"train", "training", "trains", "dataset", "benchmark",
                        "opt-in", "optin", "opt", "public"},
}
# "account"/"accounts"/"credentials" were deliberately left out of the
# buckets above: in a ToS, nearly every sentence mentions an account in
# passing, so scoring on it just floods the candidate list with pairs that
# share incidental vocabulary, not a real policy axis.


def _topic_signature(sentence: str) -> set:
    words = {w.lower().strip(".,;:()\"'") for w in sentence.split()}
    words = {w for w in words if w and w not in _STOPWORDS and len(w) > 2}
    sig = set()
    for bucket, keys in _TOPIC_BUCKETS.items():
        if any(any(k in w for k in keys) for w in words):
            sig.add(bucket)
    # also keep raw content words with 3+ chars as a fallback signal, so
    # topics outside the curated buckets can still cluster on shared nouns
    sig |= {w for w in words if len(w) >= 5}
    return sig


def _corpus_common_words(claims: list, max_doc_freq: float = 0.15) -> set:
    """Words that show up in more than `max_doc_freq` of all claims are too
    generic in THIS corpus to signal a shared topic (e.g. "account" in a
    ToS, "information" in a privacy policy) -- drop them from the raw-word
    fallback signal so they stop flooding every pair with incidental
    overlap. Computed per-corpus, not a fixed stopword list, so it adapts
    to whatever the document is actually about."""
    from collections import Counter
    doc_count = Counter()
    n = max(1, len(claims))
    for c in claims:
        words = {w.lower().strip(".,;:()\"'") for w in c.text.split()}
        for w in words:
            if w and w not in _STOPWORDS and len(w) >= 5:
                doc_count[w] += 1
    return {w for w, cnt in doc_count.items() if cnt / n > max_doc_freq}


def generate_candidate_pairs(
    claims: list,
    min_shared_buckets: int = 1,
    max_pairs: int = 60,
) -> list:
    """
    Pair claims whose topic signatures overlap, preferring cross-document
    (or at least cross-section) pairs -- a contradiction inside one
    sentence's own paragraph is much less interesting than one between two
    places in the document that were probably written at different times.
    Ranked by overlap size, capped at `max_pairs`.
    """
    common = _corpus_common_words(claims)

    def signature(c: PolicyClaim) -> set:
        return {t for t in c.topics if t in _TOPIC_BUCKETS or t not in common}

    scored = []
    n = len(claims)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = claims[i], claims[j]
            if a.text == b.text:
                continue
            overlap = signature(a) & signature(b)
            # named-bucket overlap only (ignore the generic word fallback
            # when scoring cross-doc relevance -- a shared rare word like
            # "hashed" isn't a topic, it's a coincidence)
            named_overlap = overlap & set(_TOPIC_BUCKETS.keys())
            score = len(named_overlap) if named_overlap else (1 if len(overlap) >= 2 else 0)
            if score < min_shared_buckets:
                continue
            cross_doc = a.source != b.source
            scored.append((score + (1 if cross_doc else 0), a, b))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [(a, b) for _, a, b in scored[:max_pairs]]


# ─────────────────────────────────────────────────────────────────────────
# Judging -- same shape as contradish.judge.Judge: a thin wrapper around an
# LLMClient-like object exposing .complete_json(prompt) -> dict. Any object
# with that method works, including a human reviewer's own judgments piped
# in through the same interface (see ManualJudgeClient below).
# ─────────────────────────────────────────────────────────────────────────

_CONTRADICTION_PROMPT = """You are auditing a company's own published policy documents for internal contradictions -- not fact-checking against outside law, just checking whether the company's OWN sentences are mutually satisfiable.

Claim A (from {source_a}, section "{section_a}"):
"{text_a}"

Claim B (from {source_b}, section "{section_b}"):
"{text_b}"

Question: is there a realistic real-world situation in which the company CANNOT honor both claims at once -- i.e. do they impose genuinely incompatible obligations, permissions, or facts about the same underlying subject?

Do NOT flag as a contradiction:
- Two claims about different subjects that merely share vocabulary.
- A general statement and a specific, non-conflicting exception to it (e.g. "we don't sell your data" and "we share data with service providers under contract" describe different things -- selling vs. a scoped processing relationship -- and are NOT a contradiction unless the general statement explicitly rules out the specific case too).
- A claim and its own later amendment/clarification if the text makes clear the later one supersedes or narrows the first.

DO flag as a contradiction:
- Direct negation of the same fact or rule (A says X, B says not-X, same subject, same conditions).
- Numeric/quantity conflicts (two different numbers given as THE answer to the same question -- a time window, a dollar figure, a percentage).
- Scope conflicts (A grants a right/permission unconditionally, B denies it under conditions that should be covered by A's "unconditional").
- Temporal conflicts (A says something is permanent/always true, B describes a case where it changes, with no reconciling condition).

Respond with ONLY a JSON object, no other text:
{{
  "is_contradiction": true or false,
  "contradiction_type": "direct_negation" | "numeric_conflict" | "scope_conflict" | "temporal_conflict" | "none",
  "shared_subject": "one short phrase naming what both claims are actually about",
  "confidence": 0.0 to 1.0,
  "severity": "high" | "medium" | "low" | "informational",
  "explanation": "1-3 sentences. If it's a contradiction, state the concrete scenario where the company can't satisfy both. If not, say why the apparent overlap doesn't actually conflict."
}}
"""


@dataclass
class ContradictionFinding:
    claim_a: PolicyClaim
    claim_b: PolicyClaim
    is_contradiction: bool
    contradiction_type: str
    shared_subject: str
    confidence: float
    severity: str
    explanation: str

    def to_dict(self) -> dict:
        return {
            "claim_a": self.claim_a.to_dict(),
            "claim_b": self.claim_b.to_dict(),
            "is_contradiction": self.is_contradiction,
            "contradiction_type": self.contradiction_type,
            "shared_subject": self.shared_subject,
            "confidence": self.confidence,
            "severity": self.severity,
            "explanation": self.explanation,
        }


class PolicyContradictionJudge:
    """Wraps any object with .complete_json(prompt) -> dict (contradish's
    LLMClient, or ManualJudgeClient below) as a policy-contradiction judge."""

    def __init__(self, llm):
        self.llm = llm

    def judge_pair(self, claim_a: PolicyClaim, claim_b: PolicyClaim) -> ContradictionFinding:
        prompt = _CONTRADICTION_PROMPT.format(
            source_a=claim_a.source, section_a=claim_a.section, text_a=claim_a.text,
            source_b=claim_b.source, section_b=claim_b.section, text_b=claim_b.text,
        )
        try:
            result = self.llm.complete_json(prompt)
        except Exception as e:
            result = {"is_contradiction": False, "contradiction_type": "none",
                      "shared_subject": "", "confidence": 0.0,
                      "severity": "informational", "explanation": f"judge call failed: {e}"}
        if not isinstance(result, dict):
            result = {}

        def safe_float(v, default=0.5):
            try:
                return max(0.0, min(1.0, float(v))) if v is not None else default
            except (TypeError, ValueError):
                return default

        return ContradictionFinding(
            claim_a=claim_a,
            claim_b=claim_b,
            is_contradiction=bool(result.get("is_contradiction", False)),
            contradiction_type=result.get("contradiction_type", "none"),
            shared_subject=result.get("shared_subject", ""),
            confidence=safe_float(result.get("confidence")),
            severity=result.get("severity", "informational"),
            explanation=result.get("explanation", ""),
        )


def pair_key(claim_a: PolicyClaim, claim_b: PolicyClaim) -> str:
    """Stable identifier for a claim pair, independent of candidate-list
    ordering or of how many pairs a given run generated -- safe to use as a
    dict key even if generate_candidate_pairs is rerun with a different
    max_pairs, or re-ranks pairs after an extraction-heuristic tweak."""
    return "|".join(sorted([claim_a.claim_id, claim_b.claim_id]))


class ManualJudgeClient:
    """A lookup table of pre-computed judge verdicts, keyed by
    pair_key(claim_a, claim_b), for running a real audit through a human
    reviewer -- or an LLM session with no API key wired in, standing in as
    reviewer #1 the same way contradish.benchmark_ground_truth_audit uses
    '>=2 LLM reviewer models' -- instead of a live model call.

    Only the pairs worth a real explanation need an entry; every other
    candidate pair falls back to `default` (e.g. "reviewed, not a genuine
    conflict, shares only incidental vocabulary"). Used by
    audit_policies_manual() below. Swap that call for
    audit_policies(claims, PolicyContradictionJudge(LLMClient()), ...) once
    a live API key is available and nothing else about how findings are
    consumed changes -- both paths return the same AuditReport /
    ContradictionFinding shapes.
    """

    def __init__(self, judgments: dict, default: Optional[dict] = None):
        self._judgments = judgments  # {pair_key(a, b): verdict dict}
        self._default = default or {
            "is_contradiction": False,
            "contradiction_type": "none",
            "shared_subject": "",
            "confidence": 0.5,
            "severity": "informational",
            "explanation": "Reviewed; shares topic vocabulary with its pair "
                            "but not a genuine conflict.",
        }

    def verdict_for(self, claim_a: PolicyClaim, claim_b: PolicyClaim) -> dict:
        return self._judgments.get(pair_key(claim_a, claim_b), self._default)


# ─────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class AuditReport:
    claims: list
    pairs_considered: int
    findings: list  # ContradictionFinding, only the ones judged True

    def print_summary(self):
        print(f"Extracted {len(self.claims)} policy-bearing claims.")
        print(f"Evaluated {self.pairs_considered} candidate pairs.")
        print(f"Found {len(self.findings)} likely contradictions.\n")
        for f in sorted(self.findings, key=lambda x: -x.confidence):
            print(f"[{f.severity.upper()} · {f.contradiction_type} · conf {f.confidence:.2f}] "
                  f"{f.shared_subject}")
            print(f"  A ({f.claim_a.source} / {f.claim_a.section}): {f.claim_a.text}")
            print(f"  B ({f.claim_b.source} / {f.claim_b.section}): {f.claim_b.text}")
            print(f"  -> {f.explanation}\n")

    def to_json(self) -> str:
        return json.dumps({
            "claims_extracted": len(self.claims),
            "pairs_considered": self.pairs_considered,
            "findings": [f.to_dict() for f in self.findings],
        }, indent=2)


def audit_policies(claims: list, judge: PolicyContradictionJudge, max_pairs: int = 40) -> AuditReport:
    pairs = generate_candidate_pairs(claims, max_pairs=max_pairs)
    findings = []
    for a, b in pairs:
        result = judge.judge_pair(a, b)
        if result.is_contradiction:
            findings.append(result)
    return AuditReport(claims=claims, pairs_considered=len(pairs), findings=findings)


def audit_policies_manual(
    claims: list, manual_judge: ManualJudgeClient, max_pairs: int = 40
) -> AuditReport:
    """Same contract and return shape as audit_policies(), but resolves each
    candidate pair's verdict from a ManualJudgeClient lookup table instead of
    a live LLM call -- for running a real, honestly-labeled audit when no
    API key is available, or for a human-reviewed pass."""
    pairs = generate_candidate_pairs(claims, max_pairs=max_pairs)
    findings = []
    for a, b in pairs:
        v = manual_judge.verdict_for(a, b)

        def safe_float(x, default=0.5):
            try:
                return max(0.0, min(1.0, float(x)))
            except (TypeError, ValueError):
                return default

        finding = ContradictionFinding(
            claim_a=a,
            claim_b=b,
            is_contradiction=bool(v.get("is_contradiction", False)),
            contradiction_type=v.get("contradiction_type", "none"),
            shared_subject=v.get("shared_subject", ""),
            confidence=safe_float(v.get("confidence")),
            severity=v.get("severity", "informational"),
            explanation=v.get("explanation", ""),
        )
        if finding.is_contradiction:
            findings.append(finding)
    return AuditReport(claims=claims, pairs_considered=len(pairs), findings=findings)
