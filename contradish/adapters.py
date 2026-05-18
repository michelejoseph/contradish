"""
adapters: turn any model into the `Callable[[str], str]` shape contradish expects.

Contradish treats your app as a black box: a function that takes a user question
(str) and returns a response (str). That keeps the testing layer unopinionated —
but it means a developer on Bedrock, Vertex, Gemini, OpenRouter, Together, Groq,
Mistral, Ollama, vLLM, or any other provider has to write the wrapper themselves
before they can run a single test.

This module is the wrapper. Pick one:

    from contradish import wrap_litellm, Suite

    # Any of ~100 LiteLLM-supported models
    app = wrap_litellm(
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        system="You are a support agent. Refunds within 30 days only.",
    )
    Suite(app=app).from_policy("ecommerce").run()

    # Any OpenAI-compatible endpoint (vLLM, Ollama, OpenRouter, Together, Groq, etc.)
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    app = wrap_openai_compatible(client, model="meta-llama/Llama-3.1-70B-Instruct",
                                 system=SYSTEM_PROMPT)

Both helpers return a plain `Callable[[str], str]`. Nothing about the rest of
contradish changes — Suite, RegressionSuite, Firewall, improve(), all accept
the callable the same way.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


# ──────────────────────────────────────────────────────────────────────────────
# LiteLLM — one shim, ~100 providers
# ──────────────────────────────────────────────────────────────────────────────

def wrap_litellm(
    model:      str,
    system:     Optional[str] = None,
    max_tokens: int           = 512,
    temperature: float        = 0.0,
    **completion_kwargs:      Any,
) -> Callable[[str], str]:
    """
    Wrap any LiteLLM-supported model as a contradish-compatible app callable.

    LiteLLM normalizes ~100 providers (Anthropic, OpenAI, Bedrock, Vertex,
    Gemini, OpenRouter, Together, Groq, Mistral, Ollama, vLLM, …) behind a
    single `completion()` call. Provider-specific auth still uses the
    provider's own environment variables — see https://docs.litellm.ai for the
    matrix.

    Args:
        model:             A LiteLLM model string. Examples:
                             "gpt-4o"
                             "claude-sonnet-4-6"
                             "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
                             "vertex_ai/gemini-1.5-pro"
                             "openrouter/meta-llama/llama-3.1-70b-instruct"
                             "ollama/llama3"
        system:            Optional system prompt. Prepended to every call.
        max_tokens:        Token cap on the response. Default 512.
        temperature:       Sampling temperature. Default 0.0 for stability.
        completion_kwargs: Forwarded directly to `litellm.completion`. Use
                           this for provider-specific knobs (api_base,
                           aws_region_name, vertex_project, …).

    Returns:
        A `Callable[[str], str]` you can pass to Suite(app=...), Firewall(app=...),
        RegressionSuite, or improve().

    Raises:
        ImportError if litellm isn't installed. Install with:
            pip install "contradish[litellm]"
        or
            pip install litellm

    Example:
        from contradish import Suite, wrap_litellm

        app = wrap_litellm(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            system="You are a support agent. Refunds within 30 days only.",
        )
        Suite.from_policy("ecommerce", app=app).run()
    """
    try:
        import litellm
    except ImportError as e:
        raise ImportError(
            "litellm is not installed. Install with:\n"
            "    pip install \"contradish[litellm]\"\n"
            "  or\n"
            "    pip install litellm"
        ) from e

    def app(question: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        resp = litellm.completion(
            model       = model,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            **completion_kwargs,
        )
        # LiteLLM normalizes to OpenAI response shape regardless of provider.
        return (resp.choices[0].message.content or "").strip()

    return app


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible — vLLM, Ollama, OpenRouter, Together, Groq, Fireworks, …
# ──────────────────────────────────────────────────────────────────────────────

def wrap_openai_compatible(
    client:     Any,
    model:      str,
    system:     Optional[str] = None,
    max_tokens: int           = 512,
    temperature: float        = 0.0,
    **completion_kwargs:      Any,
) -> Callable[[str], str]:
    """
    Wrap a pre-built OpenAI-compatible client as a contradish app callable.

    Any provider that speaks the OpenAI Chat Completions API works: vLLM,
    Ollama, OpenRouter, Together, Groq, Fireworks, Anyscale, DeepInfra, the
    official OpenAI SDK pointed at any compatible base_url, etc. You build
    the client yourself (so contradish doesn't have to know about your auth);
    we just call `.chat.completions.create` on it.

    Args:
        client:            An OpenAI-compatible client instance. Must expose
                           `client.chat.completions.create(model=..., messages=...)`.
                           Use `openai.OpenAI(base_url=..., api_key=...)` for
                           most third-party providers.
        model:             Model identifier the endpoint expects.
        system:            Optional system prompt prepended to every call.
        max_tokens:        Token cap on the response.
        temperature:       Sampling temperature.
        completion_kwargs: Forwarded to `chat.completions.create`.

    Returns:
        A `Callable[[str], str]` ready to pass to Suite / Firewall / improve.

    Example (local vLLM):
        from openai import OpenAI
        from contradish import Suite, wrap_openai_compatible

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        app = wrap_openai_compatible(client, model="meta-llama/Llama-3.1-70B-Instruct",
                                     system="You are a support agent.")
        Suite.from_policy("ecommerce", app=app).run()

    Example (OpenRouter):
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
        app = wrap_openai_compatible(client, model="meta-llama/llama-3.1-70b-instruct")
    """
    def app(question: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        resp = client.chat.completions.create(
            model       = model,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            **completion_kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

    return app


__all__ = ["wrap_litellm", "wrap_openai_compatible"]
