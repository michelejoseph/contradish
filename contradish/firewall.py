"""
Firewall: real-time contradiction detection for production LLM apps.

Wraps your existing LLM app and intercepts responses that contradict
something the app committed to earlier in the same conversation. Two modes:

  "monitor" -- pass everything through, flag contradictions in the result.
  "block"   -- return a corrected (or safe fallback) reply on a contradiction.

This is the production counterpart to contradish's offline CAI testing.
Use testing (Suite) to catch failures before deploy. Use Firewall to catch
the ones that slip through in production.

Memory-aware by default
------------------------
Instead of comparing each reply against the most recent few raw turns, the
Firewall distills every reply into atomic *commitments* (normalized, checkable
assertions with a topic key and provenance), stores them per session, and for
each new reply retrieves only the *relevant* prior commitments and judges
contradiction claim-vs-claim. The contradiction that matters is usually with a
commitment made arbitrarily far back ("refund window is 30 days" at turn 3,
contradicted at turn 60); recency-based comparison misses it, relevance-based
retrieval catches it. See contradish.memory.ConversationMemory.

On a contradiction it can also *repair* the reply: rewrite it to honor the
established commitment and cite it, instead of returning a generic fallback.
This is the runtime analogue of the offline `improve` loop.

Cost note: the memory-aware path costs up to 1 extraction call per reply, plus
1 detection call when there is a relevant prior, plus 1 repair call on a
detected contradiction (rare). Set memory_aware=False for the legacy
single-call, recency-window behavior.

Example:
    from contradish import Firewall

    firewall = Firewall(app=my_llm_app, mode="monitor")

    result = firewall.check("Can I get a refund after 45 days?", session="user-42")
    print(result.response)
    if result.contradiction_detected:
        print(f"Contradiction: {result.explanation}")
        print(f"Grounded on:   {result.grounded_on}")
        print(f"Suggested fix: {result.repaired_response}")

    print(firewall.summary())
"""

from typing import Callable, Optional
from .models import FirewallResult
from .llm import LLMClient
from .caches import FirewallCache, InMemoryCache
from .memory import ConversationMemory


# Legacy (memory_aware=False) recency-window contradiction check prompt.
_CONTRADICTION_CHECK_PROMPT = """You are checking whether an LLM app's new response contradicts any of its recent responses on the same topic.

New query: {query}
New response: {new_response}

Recent responses from the same app:
{cache_pairs}

Does the new response contradict any recent response on a shared topic or policy?
A contradiction means the app gives meaningfully different answers to questions that should have the same answer.
Ignore differences in phrasing or tone. Only flag logical/policy contradictions.

Return JSON only, no markdown:
{{"contradiction": true/false, "matched_query": "the exact query it contradicts, or null", "explanation": "one sentence description of the contradiction, or null"}}"""

_DEFAULT_FALLBACK = (
    "I want to make sure I give you accurate information. "
    "Let me connect you with someone who can help directly."
)


class Firewall:
    """
    Real-time contradiction detection layer for production LLM apps.

    Args:
        app:               Your LLM callable (str -> str).
        mode:              "monitor" logs contradictions without blocking.
                           "block" returns a corrected (or fallback) reply.
        api_key:           API key for the judge. Reads from env
                           (ANTHROPIC_API_KEY / OPENAI_API_KEY) if omitted.
        provider:          "anthropic" or "openai". Auto-detected if omitted.
        window:            Max raw turns retained in the legacy audit cache.
        fallback_response: Message returned when blocking and repair is off or
                           fails. Uses a safe default if omitted.
        cache:             Raw-turn audit cache backend (FirewallCache). Default
                           InMemoryCache(window=window). Used for the audit log
                           and for the legacy detection path.
        memory_aware:      When True (default), use commitment-level memory with
                           relevance retrieval. When False, use the legacy
                           recency-window single-call check.
        memory:            A ConversationMemory. Built internally when omitted
                           and memory_aware is True. Pass one with a
                           RedisCommitmentStore for shared multi-worker state.
        repair:            When True (default) and a contradiction is detected,
                           generate a corrected reply. In block mode that
                           corrected reply is returned; in monitor mode it is
                           offered via result.repaired_response while the
                           original passes through.

    Example (monitor, log but don't block):
        firewall = Firewall(app=my_app, mode="monitor")
        result = firewall.check(user_query, session=user_id)
        if result.contradiction_detected:
            alert_team(result)

    Example (block + repair, return a consistent reply):
        firewall = Firewall(app=my_app, mode="block")
        result = firewall.check(user_query, session=user_id)
        return result.response   # corrected to honor the prior commitment
    """

    def __init__(
        self,
        app:               Callable[[str], str],
        mode:              str = "monitor",
        api_key:           Optional[str] = None,
        provider:          Optional[str] = None,
        window:            int = 50,
        fallback_response: Optional[str] = None,
        cache:             Optional[FirewallCache] = None,
        memory_aware:      bool = True,
        memory:            Optional[ConversationMemory] = None,
        repair:            bool = True,
    ):
        if mode not in ("monitor", "block"):
            raise ValueError(f"mode must be 'monitor' or 'block', got: {mode!r}")

        self.app          = app
        self.mode         = mode
        self._llm         = LLMClient(api_key=api_key, provider=provider)
        self.window       = window
        self._fallback    = fallback_response or _DEFAULT_FALLBACK
        self.cache        = cache if cache is not None else InMemoryCache(window=window)
        self.memory_aware = memory_aware
        self.repair       = repair
        if memory_aware:
            self.memory = memory if memory is not None else ConversationMemory(llm=self._llm)
        else:
            self.memory = memory
        self.events: list[FirewallResult] = []

    # ── Public ─────────────────────────────────────────────────────────────────

    def check(self, query: str, session: str = "default") -> FirewallResult:
        """
        Send query to your app, check the reply for contradictions.

        Args:
            query:   the user input to pass to your app.
            session: conversation / user scope. Contradiction checks only
                     compare within a session, so one user's history never
                     pollutes another's. Defaults to a single shared scope.

        Returns a FirewallResult. In block mode with a detected contradiction,
        result.response holds the repaired reply (or the fallback if repair is
        off or failed) and result.blocked is True.
        """
        response = self.app(query)

        if self.memory_aware and self.memory is not None:
            return self._check_memory_aware(query, response, session)
        return self._check_legacy(query, response, session)

    def summary(self) -> dict:
        """
        Aggregate stats for all queries processed since initialization.

        Returns:
            Dict with total_queries, contradictions_detected, responses_blocked,
            responses_repaired, and contradiction_rate (0-1).
        """
        total          = len(self.events)
        contradictions = sum(1 for e in self.events if e.contradiction_detected)
        blocked        = sum(1 for e in self.events if e.blocked)
        repaired       = sum(1 for e in self.events if e.repaired_response)
        return {
            "total_queries":           total,
            "contradictions_detected": contradictions,
            "responses_blocked":       blocked,
            "responses_repaired":      repaired,
            "contradiction_rate":      round(contradictions / total, 3) if total else 0.0,
        }

    def reset(self, session: Optional[str] = None) -> None:
        """
        Clear stored state. With a session, clears only that conversation's
        memory. With no argument, clears the raw cache, the event log, and all
        conversation memory.
        """
        if session is not None:
            if self.memory is not None:
                self.memory.clear(session)
            return
        self.cache.clear()
        self.events.clear()
        if self.memory is not None:
            self.memory.clear()

    # ── Internal: memory-aware path ──────────────────────────────────────────────

    def _check_memory_aware(self, query: str, response: str, session: str) -> FirewallResult:
        finding = self.memory.check(session, query, response)
        # Persist the new reply's commitments after checking (don't match self).
        new_commitments = getattr(finding, "_new_commitments", None)
        if new_commitments is None:
            new_commitments = self.memory.extract(query, response, session=session)
        self.memory.ingest_commitments(new_commitments)
        # Keep a raw audit trail (also preserves legacy cache semantics).
        self.cache.append(query, response)

        contradiction = finding.contradiction
        repaired = None
        if contradiction and self.repair:
            repaired = self.memory.repair(query, response, finding)

        blocked = contradiction and self.mode == "block"
        if blocked:
            out_response = repaired if repaired else self._fallback
        else:
            out_response = response

        result = FirewallResult(
            query=query,
            response=out_response,
            blocked=blocked,
            contradiction_detected=contradiction,
            cached_query=finding.prior_query,
            cached_response=finding.prior_response,
            explanation=finding.explanation,
            repaired_response=repaired,
            grounded_on=finding.prior_claim,
            confidence=finding.confidence,
            session=session,
        )
        self.events.append(result)
        return result

    # ── Internal: legacy recency-window path ─────────────────────────────────────

    def _check_legacy(self, query: str, response: str, session: str) -> FirewallResult:
        contradiction, matched_query, matched_response, explanation = (
            self._check_contradiction(query, response)
        )
        self.cache.append(query, response)

        blocked = contradiction and self.mode == "block"
        result = FirewallResult(
            query=query,
            response=self._fallback if blocked else response,
            blocked=blocked,
            contradiction_detected=contradiction,
            cached_query=matched_query,
            cached_response=matched_response,
            explanation=explanation,
            session=session,
        )
        self.events.append(result)
        return result

    def _check_contradiction(
        self,
        query:    str,
        response: str,
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Legacy path: compare query+response against the recent raw cache.
        Returns (contradiction, matched_query, matched_response, explanation).
        """
        if self.cache.size() == 0:
            return False, None, None, None

        sample = self.cache.recent(15)
        if not sample:
            return False, None, None, None

        cache_pairs = "\n".join(
            f"Q: {c['query']}\nA: {c['response'][:300]}"
            for c in sample
        )

        prompt = _CONTRADICTION_CHECK_PROMPT.format(
            query=query,
            new_response=response[:400],
            cache_pairs=cache_pairs,
        )

        try:
            result = self._llm.complete_json(prompt, model=self._llm.fast_model)
            if isinstance(result, dict) and result.get("contradiction"):
                matched_query = result.get("matched_query")
                matched_entry = next(
                    (c for c in reversed(sample) if c["query"] == matched_query),
                    None,
                )
                return (
                    True,
                    matched_query,
                    matched_entry["response"] if matched_entry else None,
                    result.get("explanation"),
                )
        except Exception:
            pass

        return False, None, None, None
