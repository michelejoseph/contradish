"""
Admissibility Engine — structurally-calibrated production safety layer.

The existing Firewall measures only CAI Strain: does this response contradict
something the model said earlier in the same conversation? That's internal
consistency. A model can pass the Firewall completely and be consistently wrong.

This module adds the second axis: Reality Strain — distance from the external
truth fixed point of the domain. It unifies both into a single production layer
with thresholds that are not uniform but proportional to load_bearing_weight.

The key architectural difference from statistical classifiers:

  Statistical classifier:  threshold is a single tuned constant.
  Admissibility Engine:    threshold is derived from structural importance.
                           High load_bearing_weight → tight threshold → protective.
                           Low load_bearing_weight  → loose threshold → permissive.

This directly solves the calibration problem: reduce false alarms on distinctions
that are structurally peripheral while preserving full protection on distinctions
that are load-bearing. The threshold is not tuned from data — it is derived from
the domain's fixed point structure.

The engine also self-calibrates from production feedback. When a case generates
a false alarm (human says it was fine), the estimated load_bearing_weight for
that case decreases, loosening the threshold. When a case generates a missed
alarm (human says it was wrong), the weight increases. Over time the calibration
converges toward the actual structural importance of each domain distinction.

Usage (production):
    engine = AdmissibilityEngine.from_ground_truth()
    result = engine.check(user_query, model_response)

    if result.verdict == "block":
        return result.safe_fallback
    if result.verdict == "warn":
        alert_team(result)
    return model_response

Usage (feedback loop):
    engine.feedback(result.matched_case_id, outcome="false_alarm")
    engine.feedback(result.matched_case_id, outcome="missed_alarm")
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .llm import LLMClient
from .judge import Judge


# ── Verdict type ──────────────────────────────────────────────────────────────

Verdict = Literal["pass", "warn", "block"]
Outcome = Literal["false_alarm", "correct_alarm", "correct_pass", "missed_alarm"]

_GROUND_TRUTH_DIR = Path(__file__).parent.parent / "ground-truth"
_DOMAIN_FILES = {
    "medication":          "medication.json",
    "emergency-services":  "emergency-services.json",
    "financial-advice":    "financial-advice.json",
    "immigration":         "immigration.json",
    "tenant-rights":       "tenant-rights.json",
}


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class AdmissibilityResult:
    """
    The result of a single engine.check() call.

    verdict:             "pass" | "warn" | "block"
    reality_strain:      0.0 (correct) to 1.0 (completely wrong)
    threshold:           the per-case threshold used (derived from load_bearing_weight)
    load_bearing_weight: how structurally important this distinction is (0–1)
    matched_case_id:     which ground truth entry was matched
    matched_domain:      which domain it came from
    similarity_score:    how confident the domain index was in the match (0–1)
    auto_fail:           True if a critical claim was absent or disqualifying claim present
    judge_summary:       one-line explanation from the judge
    safe_fallback:       message to return in block mode
    elapsed_ms:          time taken for this check
    """
    query:               str
    response:            str
    verdict:             Verdict
    reality_strain:      float
    threshold:           float
    load_bearing_weight: float
    matched_case_id:     str
    matched_domain:      str
    similarity_score:    float
    auto_fail:           bool
    judge_summary:       str
    safe_fallback:       str = (
        "I want to make sure I give you accurate information on this. "
        "Let me connect you with a qualified source who can help directly."
    )
    elapsed_ms:          float = 0.0

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    @property
    def flagged(self) -> bool:
        return self.verdict in ("warn", "block")


# ── Domain Index ──────────────────────────────────────────────────────────────

class DomainIndex:
    """
    Maps any input query to the nearest ground truth entry.

    Uses TF-IDF keyword overlap for fast, zero-API-cost lookup. The match is
    intentionally conservative — if no entry scores above the similarity floor,
    the engine skips scoring rather than matching incorrectly. A false abstention
    (no match found) is safer than a false match (wrong domain entry used).

    For higher-precision matching at higher cost, set semantic=True in
    AdmissibilityEngine, which uses the judge LLM to confirm ambiguous matches.
    """

    def __init__(self, entries: list[dict]):
        self._entries   = entries
        self._index     = [self._fingerprint(e) for e in entries]

    @staticmethod
    def _fingerprint(entry: dict) -> set[str]:
        """Extract meaningful tokens from question + critical_claim."""
        text = (
            entry.get("question", "") + " " +
            entry.get("critical_claim", "") + " " +
            entry.get("gold_standard", "")
        ).lower()
        # Remove stopwords; keep tokens with 3+ chars
        stopwords = {
            "the", "and", "for", "are", "that", "this", "with", "from",
            "what", "does", "can", "you", "have", "not", "all", "any",
            "how", "who", "when", "where", "they", "their", "your", "its",
            "been", "will", "would", "should", "could", "must", "may",
        }
        tokens = re.findall(r"\b[a-z]{3,}\b", text)
        return {t for t in tokens if t not in stopwords}

    def nearest(
        self,
        query: str,
        floor: float = 0.08,
    ) -> tuple[Optional[dict], float]:
        """
        Return the nearest entry and its similarity score.
        Returns (None, 0.0) if no entry scores above floor.

        Similarity = |query_tokens ∩ entry_tokens| / |query_tokens ∪ entry_tokens|
        (Jaccard coefficient on the token sets)
        """
        q_tokens = self._fingerprint({"question": query, "critical_claim": "", "gold_standard": ""})
        if not q_tokens:
            return None, 0.0

        best_entry, best_score = None, 0.0
        for entry, e_tokens in zip(self._entries, self._index):
            if not e_tokens:
                continue
            intersection = len(q_tokens & e_tokens)
            union        = len(q_tokens | e_tokens)
            score        = intersection / union if union else 0.0
            if score > best_score:
                best_score, best_entry = score, entry

        if best_score < floor:
            return None, best_score
        return best_entry, best_score

    @classmethod
    def from_domain_data(cls, domain_data_list: list[dict]) -> "DomainIndex":
        entries = []
        for data in domain_data_list:
            for q in data.get("questions", []):
                entries.append(q)
        return cls(entries)


# ── Threshold Policy ──────────────────────────────────────────────────────────

class ThresholdPolicy:
    """
    Maps load_bearing_weight to a per-case threshold.

    base_threshold / load_bearing_weight — so high-weight cases get tight
    thresholds and low-weight cases get loose ones.

    Clip to [min_threshold, max_threshold] to prevent extremes.

        lbw=0.9 (heart attack — call 911): threshold = 0.20 / 0.9 = 0.22
        lbw=0.5 (ibuprofen + acetaminophen): threshold = 0.20 / 0.5 = 0.40
        lbw=0.3 (peripheral detail): threshold = 0.20 / 0.3 = 0.67 (capped at 0.70)

    The base_threshold is the threshold you'd apply to a case with lbw=1.0.
    Set it to the strictest acceptable Reality Strain for your most critical cases.

    warn_multiplier: warn at threshold × warn_multiplier before hard blocking.
    block at threshold, warn at threshold × warn_multiplier.
    """

    def __init__(
        self,
        base_threshold:  float = 0.20,
        warn_multiplier: float = 1.5,
        min_threshold:   float = 0.10,
        max_threshold:   float = 0.75,
    ):
        self.base   = base_threshold
        self.mult   = warn_multiplier
        self.min_t  = min_threshold
        self.max_t  = max_threshold

    def block_threshold(self, lbw: float) -> float:
        raw = self.base / max(lbw, 0.01)
        return round(min(self.max_t, max(self.min_t, raw)), 4)

    def warn_threshold(self, lbw: float) -> float:
        return round(min(self.max_t, self.block_threshold(lbw) * self.mult), 4)

    def verdict(self, reality_strain: float, lbw: float, auto_fail: bool) -> Verdict:
        if auto_fail:
            return "block"
        block_t = self.block_threshold(lbw)
        warn_t  = self.warn_threshold(lbw)
        if reality_strain >= block_t:
            return "block"
        if reality_strain >= warn_t:
            return "warn"
        return "pass"


# ── Calibration Store ─────────────────────────────────────────────────────────

class CalibrationStore:
    """
    Persists and updates load_bearing_weight estimates from production feedback.

    The static load_bearing_weight in the ground truth files is a prior.
    Production feedback is the likelihood update. The posterior weight moves:
      - Down when a case generates false alarms (over-protective)
      - Up when a case generates missed alarms (under-protective)

    Update rule (Bayesian-flavored bounded gradient):
        new_weight = clip(weight + delta, min_weight, max_weight)
        delta = +step on missed_alarm, -step on false_alarm

    The step size decreases with observation count (trust the prior less as
    evidence accumulates, but don't move too fast on sparse signal).
    """

    STEP        = 0.05
    MIN_WEIGHT  = 0.10
    MAX_WEIGHT  = 0.99

    def __init__(self, path: Optional[Path] = None):
        self._path  = path
        self._store: dict[str, dict] = {}
        if path and path.exists():
            with open(path) as f:
                self._store = json.load(f)

    def get_weight(self, case_id: str, base_weight: float) -> float:
        """Return the calibrated weight for a case (falls back to base_weight)."""
        if case_id in self._store:
            return self._store[case_id]["weight"]
        return base_weight

    def record_outcome(self, case_id: str, base_weight: float, outcome: Outcome) -> float:
        """Update the weight based on a production outcome. Returns new weight."""
        if case_id not in self._store:
            self._store[case_id] = {
                "weight":           base_weight,
                "false_alarms":     0,
                "missed_alarms":    0,
                "correct_alarms":   0,
                "correct_passes":   0,
                "observations":     0,
            }

        record = self._store[case_id]
        record["observations"] += 1

        # Decreasing step: move quickly at first, stabilize with evidence
        n    = record["observations"]
        step = self.STEP / math.sqrt(n)

        if outcome == "false_alarm":
            record["false_alarms"] += 1
            delta = -step
        elif outcome == "missed_alarm":
            record["missed_alarms"] += 1
            delta = +step
        elif outcome == "correct_alarm":
            record["correct_alarms"] += 1
            delta = +step * 0.5   # mild reinforcement
        else:  # correct_pass
            record["correct_passes"] += 1
            delta = 0.0

        new_weight = round(
            min(self.MAX_WEIGHT, max(self.MIN_WEIGHT, record["weight"] + delta)), 4
        )
        record["weight"] = new_weight

        if self._path:
            with open(self._path, "w") as f:
                json.dump(self._store, f, indent=2)

        return new_weight

    def calibration_summary(self) -> dict:
        return {
            cid: {
                "weight":         r["weight"],
                "false_alarms":   r["false_alarms"],
                "missed_alarms":  r["missed_alarms"],
                "observations":   r["observations"],
                "precision":      round(
                    r["correct_alarms"] / max(r["correct_alarms"] + r["false_alarms"], 1), 3
                ),
                "recall":         round(
                    r["correct_alarms"] / max(r["correct_alarms"] + r["missed_alarms"], 1), 3
                ),
            }
            for cid, r in self._store.items()
        }


# ── Semantic Confirmer (optional, higher cost) ────────────────────────────────

_SEMANTIC_MATCH_PROMPT = """You are verifying whether a user query is semantically equivalent
to a specific ground-truth question that a safety system wants to evaluate it against.

User query:
{query}

Candidate ground-truth question:
{candidate}

Is the user query asking about the same substantive topic as the candidate question?
"Same topic" means: a correct answer to the candidate question would substantially
address the user's query, or a wrong answer to the candidate question would
constitute a wrong answer to the user's query.

Respond ONLY with JSON:
{{"match": true|false, "confidence": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


# ── Admissibility Engine ───────────────────────────────────────────────────────

class AdmissibilityEngine:
    """
    Structurally-calibrated production safety layer.

    Checks model responses against the load-bearing fixed point of their domain.
    Thresholds are derived from structural importance (load_bearing_weight),
    not tuned statistically — high-weight distinctions get tight protection,
    peripheral distinctions get loose protection.

    The engine self-calibrates from production feedback, updating weight
    estimates as false alarms and missed alarms accumulate.

    Args:
        judge:              Judge instance for scoring (required).
        domain_data:        List of domain JSON dicts (loaded from ground-truth/).
                            Pass None to auto-load from default ground-truth directory.
        threshold_policy:   ThresholdPolicy to use. Defaults to base=0.20.
        calibration_path:   Path to persist calibration state. None = in-memory only.
        semantic_confirm:   If True, use the judge to confirm ambiguous domain matches.
                            Increases accuracy; costs one extra LLM call on ambiguous cases.
        similarity_floor:   Minimum keyword similarity to attempt scoring.
                            Queries below this floor are passed through (no ground truth match).
        semantic_floor:     Similarity score below which semantic confirmation is triggered.

    Example:
        engine = AdmissibilityEngine.from_ground_truth(api_key="sk-...")
        result = engine.check(user_query, model_response)
        if result.verdict == "block":
            return safe_message
    """

    def __init__(
        self,
        judge:             Judge,
        domain_data:       Optional[list[dict]] = None,
        threshold_policy:  Optional[ThresholdPolicy] = None,
        calibration_path:  Optional[Path] = None,
        semantic_confirm:  bool = True,
        similarity_floor:  float = 0.08,
        semantic_floor:    float = 0.25,
    ):
        self._judge         = judge
        self._policy        = threshold_policy or ThresholdPolicy()
        self._calibration   = CalibrationStore(calibration_path)
        self._semantic      = semantic_confirm
        self._sim_floor     = similarity_floor
        self._sem_floor     = semantic_floor
        self.events:        list[AdmissibilityResult] = []

        if domain_data is None:
            domain_data = self._load_default_domains()
        self._index = DomainIndex.from_domain_data(domain_data)

    @classmethod
    def from_ground_truth(
        cls,
        api_key:           Optional[str] = None,
        provider:          Optional[str] = None,
        domains:           Optional[list[str]] = None,
        base_threshold:    float = 0.20,
        calibration_path:  Optional[Path] = None,
        semantic_confirm:  bool = True,
    ) -> "AdmissibilityEngine":
        """
        Convenience constructor: load ground truth from default directory,
        build LLM client and judge, return a ready engine.
        """
        llm   = LLMClient(api_key=api_key, provider=provider)
        judge = Judge(llm)
        data  = cls._load_default_domains(domains)
        return cls(
            judge=judge,
            domain_data=data,
            threshold_policy=ThresholdPolicy(base_threshold=base_threshold),
            calibration_path=calibration_path,
            semantic_confirm=semantic_confirm,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        query:          str,
        response:       str,
        safe_fallback:  Optional[str] = None,
    ) -> AdmissibilityResult:
        """
        Check a model response against the domain's truth fixed point.

        1. Find the nearest ground truth entry for this query (domain index).
        2. If no entry matches, return verdict="pass" (outside known domain).
        3. Optionally confirm the match semantically (one extra LLM call if ambiguous).
        4. Score the response (one LLM call: disq check + critical claim + truth).
        5. Derive threshold from load_bearing_weight (zero extra calls).
        6. Return AdmissibilityResult with verdict, strain, threshold, case_id.
        """
        t0 = time.time()

        # Step 1: Domain index lookup
        entry, sim_score = self._index.nearest(query, floor=self._sim_floor)

        # No match → outside known domain → pass through (no false alarm)
        if entry is None:
            return AdmissibilityResult(
                query=query, response=response,
                verdict="pass",
                reality_strain=0.0,
                threshold=1.0,
                load_bearing_weight=0.0,
                matched_case_id="none",
                matched_domain="unknown",
                similarity_score=sim_score,
                auto_fail=False,
                judge_summary="No matching ground truth entry found — outside known domain.",
                safe_fallback=safe_fallback or AdmissibilityResult.safe_fallback,
                elapsed_ms=round((time.time() - t0) * 1000, 1),
            )

        # Step 2: Optional semantic confirmation on weak keyword matches
        if self._semantic and sim_score < self._sem_floor:
            confirmed, confidence = self._confirm_semantic_match(query, entry)
            if not confirmed:
                return AdmissibilityResult(
                    query=query, response=response,
                    verdict="pass",
                    reality_strain=0.0,
                    threshold=1.0,
                    load_bearing_weight=0.0,
                    matched_case_id=entry.get("id", "unknown"),
                    matched_domain=entry.get("domain", "unknown"),
                    similarity_score=sim_score,
                    auto_fail=False,
                    judge_summary=f"Keyword match too weak and semantic confirmation rejected "
                                  f"(sim={sim_score:.2f}, confidence={confidence:.2f}).",
                    safe_fallback=safe_fallback or AdmissibilityResult.safe_fallback,
                    elapsed_ms=round((time.time() - t0) * 1000, 1),
                )

        # Step 3: Get calibrated load_bearing_weight
        base_lbw = entry.get("load_bearing_weight", 0.5)
        case_id  = entry.get("id", "unknown")
        domain   = entry.get("domain", "unknown")
        lbw      = self._calibration.get_weight(case_id, base_lbw)

        # Step 4: Score (single LLM call)
        scored = self._judge.evaluate_reality_strain(
            question=entry["question"],
            gold_standard=entry.get("gold_elaborated") or entry["gold_standard"],
            model_output=response,
            critical_claim=entry.get("critical_claim", ""),
            disqualifying_claims=entry.get("disqualifying_claims", []),
        )

        # Step 5: Apply threshold policy
        rs      = scored["reality_strain"]
        verdict = self._policy.verdict(rs, lbw, scored["auto_fail"])
        block_t = self._policy.block_threshold(lbw)

        result = AdmissibilityResult(
            query=query,
            response=response,
            verdict=verdict,
            reality_strain=rs,
            threshold=block_t,
            load_bearing_weight=lbw,
            matched_case_id=case_id,
            matched_domain=domain,
            similarity_score=sim_score,
            auto_fail=scored["auto_fail"],
            judge_summary=scored["summary"],
            safe_fallback=safe_fallback or AdmissibilityResult.safe_fallback,
            elapsed_ms=round((time.time() - t0) * 1000, 1),
        )
        self.events.append(result)
        return result

    def feedback(self, case_id: str, outcome: Outcome, base_weight: float = 0.5) -> float:
        """
        Record a human-verified outcome for a case. Updates calibrated weight.

        outcome:
            "false_alarm"   — engine flagged, but response was actually fine
            "missed_alarm"  — engine passed, but response was actually wrong
            "correct_alarm" — engine flagged correctly
            "correct_pass"  — engine passed correctly

        Returns the updated load_bearing_weight for this case.
        """
        return self._calibration.record_outcome(case_id, base_weight, outcome)

    def summary(self) -> dict:
        """Aggregate statistics for all checks since initialization."""
        total   = len(self.events)
        blocked = sum(1 for e in self.events if e.verdict == "block")
        warned  = sum(1 for e in self.events if e.verdict == "warn")
        passed  = sum(1 for e in self.events if e.verdict == "pass")
        matched = sum(1 for e in self.events if e.matched_case_id != "none")
        rs_vals = [e.reality_strain for e in self.events if e.matched_case_id != "none"]
        mean_rs = round(sum(rs_vals) / len(rs_vals), 4) if rs_vals else None

        by_domain: dict[str, dict] = {}
        for e in self.events:
            d = e.matched_domain
            if d not in by_domain:
                by_domain[d] = {"checks": 0, "blocked": 0, "warned": 0, "strain_sum": 0.0}
            by_domain[d]["checks"]     += 1
            by_domain[d]["blocked"]    += e.verdict == "block"
            by_domain[d]["warned"]     += e.verdict == "warn"
            by_domain[d]["strain_sum"] += e.reality_strain

        domain_summary = {
            d: {
                "checks":            v["checks"],
                "blocked":           v["blocked"],
                "warned":            v["warned"],
                "mean_reality_strain": round(v["strain_sum"] / v["checks"], 4) if v["checks"] else None,
            }
            for d, v in by_domain.items()
        }

        return {
            "total_checks":        total,
            "matched_to_domain":   matched,
            "verdict_pass":        passed,
            "verdict_warn":        warned,
            "verdict_block":       blocked,
            "alarm_rate":          round((warned + blocked) / total, 3) if total else 0.0,
            "mean_reality_strain": mean_rs,
            "by_domain":           domain_summary,
            "calibration":         self._calibration.calibration_summary(),
        }

    def threshold_report(self) -> list[dict]:
        """
        Show the per-case threshold derived from load_bearing_weight.
        Useful for understanding where protection is tight vs. loose.
        """
        rows = []
        for data in self._load_default_domains():
            for q in data.get("questions", []):
                case_id  = q.get("id", "?")
                base_lbw = q.get("load_bearing_weight", 0.5)
                lbw      = self._calibration.get_weight(case_id, base_lbw)
                rows.append({
                    "case_id":           case_id,
                    "domain":            q.get("domain", "?"),
                    "load_bearing_weight": lbw,
                    "block_threshold":   self._policy.block_threshold(lbw),
                    "warn_threshold":    self._policy.warn_threshold(lbw),
                    "question":          q.get("question", "")[:80],
                })
        return sorted(rows, key=lambda r: r["load_bearing_weight"], reverse=True)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _confirm_semantic_match(self, query: str, entry: dict) -> tuple[bool, float]:
        """Ask the judge whether this query is genuinely about the candidate entry."""
        prompt = _SEMANTIC_MATCH_PROMPT.format(
            query=query[:300],
            candidate=entry.get("question", "")[:300],
        )
        try:
            result = self._judge.llm.complete_json(prompt)
            match      = bool(result.get("match", False))
            confidence = float(result.get("confidence", 0.0))
            return match, confidence
        except Exception:
            return True, 0.5  # fail open: if confirmation errors, proceed with match

    @staticmethod
    def _load_default_domains(domains: Optional[list[str]] = None) -> list[dict]:
        target = domains or list(_DOMAIN_FILES.keys())
        loaded = []
        for domain in target:
            fname = _DOMAIN_FILES.get(domain)
            if not fname:
                continue
            path = _GROUND_TRUTH_DIR / fname
            if path.exists():
                with open(path) as f:
                    loaded.append(json.load(f))
        return loaded
