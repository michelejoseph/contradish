"""
Unified LLM client. Works with OpenAI or Anthropic.
Auto-detects from environment variables if no key/provider is specified.
"""

import os
import json
import re
import time
import random
from typing import Callable, Optional


# Retry policy. A benchmark run fires thousands of calls; a single transient
# rate-limit or 5xx should not fail the whole run. Non-transient errors
# (auth, bad request) fail fast and are never retried.
_MAX_RETRIES   = 4       # total attempts = 1 + _MAX_RETRIES
_BASE_DELAY    = 1.0     # seconds; doubles each retry
_MAX_DELAY     = 30.0    # cap on a single backoff sleep

# Substrings / class names that signal a transient, retryable failure. Matched
# against the exception class name and message so we don't need to import the
# provider SDKs' exception types (they are optional dependencies).
_TRANSIENT_MARKERS = (
    "ratelimit", "rate limit", "429",
    "timeout", "timedout", "timed out",
    "apiconnection", "connection", "connectionerror",
    "internalserver", "serviceunavailable", "service unavailable",
    "overloaded", "502", "503", "504", "500",
    "temporarily", "try again",
)


def _is_transient(exc: BaseException) -> bool:
    """Return True if this exception looks worth retrying."""
    # An explicit status_code attribute is the most reliable signal.
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    blob = (type(exc).__name__ + " " + str(exc)).lower()
    return any(m in blob for m in _TRANSIENT_MARKERS)


class LLMClient:
    """
    Thin wrapper around OpenAI and Anthropic SDKs.
    Automatically detects which provider to use from environment variables.

    Priority:
        1. Explicit provider + api_key arguments
        2. ANTHROPIC_API_KEY in environment  →  uses Anthropic
        3. OPENAI_API_KEY    in environment  →  uses OpenAI

    For benchmark judging, use make_judge_client() to get a cross-provider
    judge (e.g. OpenAI judges Anthropic models, Anthropic judges OpenAI models).
    This eliminates same-provider self-preference bias in consistency scoring.
    """

    ANTHROPIC_JUDGE_MODEL  = "claude-sonnet-4-6"
    ANTHROPIC_FAST_MODEL   = "claude-haiku-4-5-20251001"
    OPENAI_JUDGE_MODEL     = "gpt-4o"
    OPENAI_FAST_MODEL      = "gpt-4o-mini"

    def __init__(
        self,
        api_key:  Optional[str] = None,
        provider: Optional[str] = None,   # "anthropic" | "openai"
    ):
        self.provider, self.api_key = self._resolve(api_key, provider)
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def judge_model(self) -> str:
        return self.ANTHROPIC_JUDGE_MODEL if self.provider == "anthropic" else self.OPENAI_JUDGE_MODEL

    @property
    def fast_model(self) -> str:
        return self.ANTHROPIC_FAST_MODEL if self.provider == "anthropic" else self.OPENAI_FAST_MODEL

    def complete(self, prompt: str, model: Optional[str] = None, max_tokens: int = 1024) -> str:
        """Send a prompt and return the raw text response."""
        m = model or self.judge_model
        if self.provider == "anthropic":
            return self._anthropic_complete(prompt, m, max_tokens)
        else:
            return self._openai_complete(prompt, m, max_tokens)

    def complete_json(self, prompt: str, model: Optional[str] = None) -> dict:
        """Send a prompt, parse and return a JSON dict."""
        raw = self.complete(prompt, model=model)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, api_key: Optional[str], provider: Optional[str]):
        """Resolve provider + key, falling back to environment."""
        # Explicit
        if api_key and provider:
            return provider.lower(), api_key

        # Explicit key, infer provider
        if api_key and not provider:
            if api_key.startswith("sk-ant-"):
                return "anthropic", api_key
            return "openai", api_key

        # No explicit key: check environment
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        openai_key    = os.environ.get("OPENAI_API_KEY",    "").strip()

        if provider == "anthropic" and anthropic_key:
            return "anthropic", anthropic_key
        if provider == "openai" and openai_key:
            return "openai", openai_key

        # Auto-detect: prefer Anthropic
        if anthropic_key:
            return "anthropic", anthropic_key
        if openai_key:
            return "openai", openai_key

        raise EnvironmentError(
            "\n\n  contradish needs an API key to run the judge layer.\n\n"
            "  Set one of:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "    export OPENAI_API_KEY=sk-...\n\n"
            "  Or pass it directly:\n"
            "    Suite(api_key='sk-ant-...', app=my_app)\n"
        )

    def _build_client(self):
        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Install the Anthropic SDK:  pip install anthropic"
                )
        else:
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Install the OpenAI SDK:  pip install openai"
                )

    # Injectable for testing so the suite never actually sleeps.
    _sleep: Callable[[float], None] = staticmethod(time.sleep)

    def _call_with_retry(self, fn: Callable[[], str]) -> str:
        """
        Run fn(), retrying transient failures with exponential backoff + jitter.

        Retries on rate limits, timeouts, connection errors, and 5xx. Fails
        fast (re-raises immediately) on anything that doesn't look transient,
        such as auth or bad-request errors, where retrying is pointless.
        """
        attempt = 0
        while True:
            try:
                return fn()
            except BaseException as exc:  # noqa: BLE001 - we re-raise non-transient
                if attempt >= _MAX_RETRIES or not _is_transient(exc):
                    raise
                delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** attempt))
                # Honor a Retry-After hint if the SDK exception carries one.
                retry_after = getattr(exc, "retry_after", None)
                if isinstance(retry_after, (int, float)) and retry_after > 0:
                    delay = min(_MAX_DELAY, float(retry_after))
                delay += random.uniform(0, delay * 0.25)  # jitter
                self._sleep(delay)
                attempt += 1

    def _anthropic_complete(self, prompt: str, model: str, max_tokens: int) -> str:
        def _do() -> str:
            msg = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        return self._call_with_retry(_do)

    def _openai_complete(self, prompt: str, model: str, max_tokens: int) -> str:
        def _do() -> str:
            resp = self._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        return self._call_with_retry(_do)

    @classmethod
    def make_judge_client(
        cls,
        model_provider: str,
        judge_provider: Optional[str] = None,
        judge_api_key:  Optional[str] = None,
    ) -> "LLMClient":
        """
        Return an LLMClient for the judge, preferring cross-provider judging.

        By default, if the model under test is from Anthropic, returns an
        OpenAI judge (and vice versa). This eliminates same-provider
        self-preference bias -- a common and valid criticism of LLM-as-judge
        benchmarks.

        Falls back to same provider if no cross-provider key is available,
        and prints a warning.

        Args:
            model_provider: Provider of the model being evaluated ("anthropic" | "openai").
            judge_provider: Explicit judge provider override.
            judge_api_key:  Explicit judge API key override.

        Returns:
            LLMClient configured for the judge.
        """
        # Explicit override wins
        if judge_provider and judge_api_key:
            return cls(api_key=judge_api_key, provider=judge_provider)

        # Default cross-provider logic
        opposite = "openai" if model_provider == "anthropic" else "anthropic"
        env_key_name = "OPENAI_API_KEY" if opposite == "openai" else "ANTHROPIC_API_KEY"
        cross_key = os.environ.get(env_key_name, "").strip()

        if cross_key:
            return cls(api_key=cross_key, provider=opposite)

        # Fall back to same provider with warning
        same_key_name = "ANTHROPIC_API_KEY" if model_provider == "anthropic" else "OPENAI_API_KEY"
        same_key = os.environ.get(same_key_name, "").strip()
        if same_key:
            print(
                f"\n  [judge] WARNING: using same-provider judging ({model_provider}).\n"
                f"  For independent results, set {env_key_name} to enable cross-provider judging.\n"
            )
            return cls(api_key=same_key, provider=model_provider)

        raise EnvironmentError(
            f"No API key found for judge. Set {env_key_name} (preferred) or {same_key_name}."
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Best-effort: extract first {...} block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {}
