"""
contradish/gateway.py -- the Decision Integrity Gateway: the Guard in front of ANY model endpoint.

Point your existing client at the gateway instead of the provider. Nothing else in your code changes:

    contradish gateway policy.json --upstream https://api.openai.com --mode shadow --ledger audit.jsonl
    client = OpenAI(base_url="http://localhost:8787/v1")           # OpenAI format: /v1/chat/completions
    client = Anthropic(base_url="http://localhost:8787")           # Anthropic format: /v1/messages

Any upstream that speaks either format works: OpenAI, Anthropic, Azure OpenAI, Google's and xAI's
OpenAI-compatible endpoints, Bedrock or Vertex behind LiteLLM, OpenRouter, Together, Groq, vLLM, Ollama.
Credentials pass through untouched (Authorization, x-api-key, anthropic-version); the gateway stores none.

For each response that proposes tool calls, the Guard decides allow / block / verify / review / deliver from the
policy file and the EVIDENCE YOUR BACKEND SUPPLIES for the request, never from the model's own text:

    header  X-Contradish-Context: {"evidence": {...}, "as_of": "2026-10-20", "request_id": "r-1"}
    or body "contradish": {...same...}          (removed before the request is forwarded)

  shadow   the response is returned unchanged; the decision is attached and logged.
  enforce  blocked calls are removed from the response before your code can run them; if none remain, the
           model's turn ends with a short explanation. A request with no evidence establishes nothing, so an
           irreversible call is blocked (fail closed). Streaming responses with tools are refused in enforce mode.

Every response carries X-Contradish-Outcome and X-Contradish-Digest headers and a "contradish" field; every
decision is appended to the log and, with --ledger, to the signed audit ledger (contradish.integrity).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from contradish.guard import Guard, calls_from
from contradish.policy_program import Call

__all__ = ["Gateway", "apply_decision", "serve"]

PASS_HEADERS = ("authorization", "x-api-key", "anthropic-version", "anthropic-beta", "openai-organization",
                "openai-project", "api-key", "x-goog-api-key", "content-type")
CTX_HEADER = "x-contradish-context"


def _same(a: Call, b: Call) -> bool:
    return a.tool == b.tool and a.args == b.args


def _blocked(call: Call, blocked: list) -> bool:
    return any(_same(call, b) for b in blocked)


def apply_decision(fmt: str, response: dict, decision, enforce: bool) -> dict:
    """Return the response the client should see. `fmt` is 'openai' or 'anthropic'."""
    out = json.loads(json.dumps(response))
    note = None
    if enforce and decision.blocked:
        note = "[contradish] Not executed: " + " ".join(r for r in decision.reasons if r.startswith("blocked"))
        if fmt == "openai":
            for ch in out.get("choices", []):
                msg = ch.get("message") or {}
                tcs = msg.get("tool_calls") or []
                keep = [t for t in tcs if not _blocked(calls_from([t])[0], decision.blocked)] if tcs else tcs
                if tcs:
                    msg["tool_calls"] = keep or None
                    if not keep:
                        msg.pop("tool_calls", None)
                        msg["content"] = ((msg.get("content") or "") + "\n" + note).strip()
                        ch["finish_reason"] = "stop"
        else:
            blocks = out.get("content") or []
            keep = [b for b in blocks if b.get("type") != "tool_use"
                    or not _blocked(calls_from([b])[0], decision.blocked)]
            if not any(b.get("type") == "tool_use" for b in keep):
                keep.append({"type": "text", "text": note})
                out["stop_reason"] = "end_turn"
            out["content"] = keep
    out["contradish"] = {"outcome": decision.outcome, "mode": decision.mode, "enforced": bool(enforce and note),
                         "reasons": decision.reasons, "ask": decision.ask,
                         "owed": [c.to_dict() for c in decision.owed],
                         "blocked": [c.to_dict() for c in decision.blocked],
                         "digest": decision.certificate.get("digest")}
    return out


def _proposed(fmt: str, response: dict) -> list:
    if fmt == "openai":
        calls = []
        for ch in response.get("choices", [])[:1]:
            calls += calls_from((ch.get("message") or {}).get("tool_calls") or [])
        return calls
    return calls_from([b for b in (response.get("content") or []) if b.get("type") == "tool_use"])


class Gateway:
    """The gateway as a library: `handle(path, headers, body) -> (status, headers, body)`; `serve()` wraps it in
    an HTTP server. `transport(url, headers, body) -> (status, body)` is replaceable (tests, custom clients)."""

    def __init__(self, guard: Guard, upstream: str, transport: Optional[Callable] = None, timeout: float = 600):
        self.guard = guard
        self.upstream = upstream.rstrip("/")
        self.timeout = timeout
        self.transport = transport or self._http
        self.lock = threading.Lock()

    def _http(self, url: str, headers: dict, body: bytes) -> tuple:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def handle(self, path: str, headers: dict, body: bytes) -> tuple:
        h = {k.lower(): v for k, v in headers.items()}
        if path.endswith("/chat/completions"):
            fmt = "openai"
        elif path.endswith("/messages"):
            fmt = "anthropic"
        else:
            return 404, {}, json.dumps({"error": "contradish gateway serves /v1/chat/completions and /v1/messages"}).encode()
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            return 400, {}, b'{"error": "request body is not JSON"}'
        ctx = req.pop("contradish", None)
        if ctx is None and CTX_HEADER in h:
            try:
                ctx = json.loads(h[CTX_HEADER])
            except ValueError:
                return 400, {}, b'{"error": "X-Contradish-Context is not JSON"}'
        ctx = ctx or {}
        enforce = self.guard.mode == "enforce"
        if req.get("stream") and req.get("tools") and enforce:
            return 400, {}, json.dumps({"error": "contradish enforce mode cannot inspect a streamed tool call; "
                                                 "send stream=false"}).encode()
        fwd = {k: v for k, v in h.items() if k in PASS_HEADERS}
        fwd.setdefault("content-type", "application/json")
        status, raw = self.transport(self.upstream + path, fwd, json.dumps(req).encode())
        if status != 200 or req.get("stream"):
            return status, {}, raw
        resp = json.loads(raw)
        proposed = _proposed(fmt, resp)
        if not proposed:
            return 200, {"X-Contradish-Outcome": "no-tool-calls"}, raw
        with self.lock:                                  # one writer to the log / ledger
            d = self.guard.check(ctx.get("evidence") or [], proposed, as_of=ctx.get("as_of"),
                                 request_id=ctx.get("request_id"),
                                 context={"gateway": fmt, "model": req.get("model"), "upstream": self.upstream})
        out = apply_decision(fmt, resp, d, enforce)
        return 200, {"X-Contradish-Outcome": d.outcome, "X-Contradish-Digest": d.certificate.get("digest", "")}, \
            json.dumps(out).encode()


def serve(gateway: Gateway, host: str = "127.0.0.1", port: int = 8787):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/") in ("/healthz", "/contradish"):
                g = gateway.guard
                from contradish.policy_program import digest as _pd
                body = json.dumps({"ok": True, "mode": g.mode, "policy": g.base_spec.get("id"),
                                   "policy_digest": _pd(g.base_spec), "approval": g.approval.get("state"),
                                   "upstream": gateway.upstream}).encode()
                self._send(200, {}, body)
            else:
                self._send(404, {}, b"{}")

        def do_POST(self):
            n = int(self.headers.get("content-length") or 0)
            status, hdrs, body = gateway.handle(self.path, dict(self.headers), self.rfile.read(n))
            self._send(status, hdrs, body)

        def _send(self, status, hdrs, body):
            self.send_response(status)
            self.send_header("content-type", "application/json")
            for k, v in hdrs.items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), H)
    return srv
