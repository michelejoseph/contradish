"""
Firewall: real-time contradiction detection for production LLM apps.

Wraps your existing LLM app and intercepts responses that contradict
something the app said previously. Two modes:

  "monitor" -- pass everything through, flag contradictions in the result.
  "block"   -- return a safe fallback whenever a contradiction is detected.

This is the production counterpart to contradish's offline CAI testing.
Use testing (Suite) to catch failures before deploy.
Use Firewall to catch the ones that slip through in production.

Example:
    from contradish import Firewall

    firewall = Firewall(app=my_llm_app, mode="monitor")

    result = firewall.check("Can I get a refund after 45 days?")
    print(result.response)
    if result.contradiction_detected:
        print(f"Contradiction: {result.explanation}")
        # log it, alert, or route to human review

    # Get a summary of all traffic since startup
    print(firewall.summary())
"""

from typing import Callable, Optional
from .models import FirewallResult
from .llm import LLMClient


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
                           "block" returns a safe fallback instead.
        api_key:           API key for the contradiction judge. Reads from
                           env (ANTHROPIC_API_KEY / OPENAI_API_KEY) if omitted.
        provider:          "anthropic" or "openai". Auto-detected if omitted.
        window:            Max number of recent responses to compare against.
        fallback_response: Custom message to return when blocking. Uses a
                           safe default if omitted.

    Example (monitor mode, log but don't block):
        firewall = Firewall(app=my_app, mode="monitor")
        result = firewall.check(user_query)
        if result.contradiction_detected:
            alert_team(result)

    Example (block mode, prevent contradictory responses):
        firewall = Firewall(app=my_app, mode="block",
                            fallback_response="Let me get a team member to help.")
        result = firewall.check(user_query)
        return result.response  # safe even if contradiction detected
    """

    def __init__(
        self,
        app:               Callable[[str], str],
        mode:              str = "monitor",
        api_key:           Optional[str] = None,
        provider:          Optional[str] = None,
        window:            int = 50,
        fallback_response: Optional[str] = None,
    ):
        if mode not in ("monitor", "block"):
            raise ValueError(f"mode must be 'monitor' or 'block', got: {mode!r}")

        self.app       = app
        self.mode      = mode
        self._llm      = LLMClient(api_key=api_key, provider=provider)
        self.window    = window
        self._fallback = fallback_response or _DEFAULT_FALLBACK
        self._cache: list[dict] = []   # [{query, response}]
        self.events: list[FirewallResult] = []

    # ── Public ─────────────────────────────────────────────────────────────────

    def check(self, query: str) -> FirewallResult:
        """
        Send query to your app, check the response for contradictions.

        Returns a FirewallResult. If mode="block" and a contradiction is
        detected, result.response will contain the safe fallback.
        result.blocked indicates whether the original was suppressed.
        """
        response = self.app(query)

        contradiction, matched_query, matched_response, explanation = (
            self._check_contradiction(query, response)
        )

        # Cache after checking (don't compare against self)
        self._cache.append({"query": query, "response": response})
        if len(self._cache) > self.window:
            self._cache.pop(0)

        blocked = contradiction and self.mode == "block"

        result = FirewallResult(
            query=query,
            response=self._fallback if blocked else response,
            blocked=blocked,
            contradiction_detected=contradiction,
            cached_query=matched_query,
            cached_response=matched_response,
            explanation=explanation,
        )
        self.events.append(result)
        return result

    def summary(self) -> dict:
        """
        Aggregate stats for all queries processed since initialization.

        Returns:
            Dict with total_queries, contradictions_detected, responses_blocked,
            and contradiction_rate (0-1).
        """
        total         = len(self.events)
        contradictions = sum(1 for e in self.events if e.contradiction_detected)
        blocked        = sum(1 for e in self.events if e.blocked)
        return {
            "total_queries":            total,
            "contradictions_detected":  contradictions,
            "responses_blocked":        blocked,
            "contradiction_rate":       round(contradictions / total, 3) if total else 0.0,
        }

    def reset(self) -> None:
        """Clear the response cache and event log."""
        self._cache.clear()
        self.events.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _check_contradiction(
        self,
        query:    str,
        response: str,
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Compare query+response against recent cache.
        Returns (contradiction, matched_query, matched_response, explanation).
        """
        if not self._cache:
            return False, None, None, None

        # Use last N entries for comparison
        sample = self._cache[-min(15, len(self._cache)):]
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
                    (c for c in reversed(self._cache) if c["query"] == matched_query),
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
