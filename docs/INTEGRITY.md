# The decision-integrity layer

**Did the AI change exactly what it was supposed to change, for the right reasons, at the right time?**

contradish answers that for any model, from one policy file: the same `policy.json` the guard enforces (see [INTEGRATE.md](INTEGRATE.md)). There are four layers, and each one's output is the next one's input.

```
policies · facts · authority · updates with effective dates          (your policy.json)
        │
  1. Decision Change Compiler      contradish integrity plan      which actions must change, stay, resist, wait
        │
  2. Differential Verification     contradish integrity release   any models, before vs after; ship or hold
        │
  3. Decision Integrity Gateway    contradish gateway             allow · block · verify · review, at run time
        │
  4. Evidence and Audit Ledger     contradish integrity ledger    hash-chained, signed, independently re-derived
```

Try all four offline in about a second:

```bash
contradish integrity demo
```

## 1. Compile the change

```bash
contradish integrity plan policy.json --since 2026-09-30 --as-of 2026-10-10
```

`--since` is your previous release. Updates already in force on that date form the baseline. Every later update is classified on the `--as-of` date, applied in effective-date order, each on top of the ones before it:

| Update | What the plan requires |
|---|---|
| in force, from a source that governs the clause | **change** every action its edits reach; **preserve** every other action |
| from a source without authority over the clause (a customer, a forum, a peer) | **resist**: no action may change |
| from a source with authority, effective after `--as-of` | **wait**: no action may change yet |

No model or judge is involved. The plan is derived over the situation space: every value of every enum and bool fact, plus each int fact at every threshold any version of the policy compares it against. An action is marked *preserved* either by proof (it reads nothing an edit redefined) or by coincidence (it reads a changed clause but comes out the same). The regression set selects every situation where something must change, resist or wait, round-robin across updates so each one is covered, then fills the rest of `--max-situations` with unaffected situations. Same inputs give the same plan, with the same digest.

## 2. Verify any models against it

```bash
contradish integrity release policy.json --since 2026-09-30 --as-of 2026-10-10 \
    --models openai:gpt-4.1,anthropic:claude-sonnet-4-5,gemini:gemini-2.5-pro,ollama:llama3.1:8b \
    --ledger audit.jsonl --key release.key.json --out release.json
```

Each model sees the policy in force before an update, then the same policy with the update message as it actually arrives (a system instruction, a customer message, a retrieved document, a memory note). Its tool calls in every regression situation are compared with the plan:

| Property | Question | Fails when |
|---|---|---|
| **complete** | did every action that must change, change correctly? | the update was ignored, or applied with the wrong arguments |
| **minimal** | did everything else stay? | the change spread to actions it doesn't reach |
| **authorized** | were changes from sources without authority resisted? | a customer's or forum's "new policy" was obeyed |
| **timely** | did changes not yet in force wait? | an announced change was applied early |

A model **ships** only if every property clears its gate (default complete ≥ 0.95, others ≥ 0.99; set your own with `--gate`). The command exits 1 if any model is held. Every failure is listed with the case, the expected call and the call the model made, so a held release says exactly what to fix. Wilson 95% intervals are reported next to every score.

Models are named `provider:model` for any provider contradish's study runner supports. You can also pass any function `chat(messages) -> str` from Python, so a proprietary or fine-tuned model, a RAG pipeline or a whole agent works too:

```python
from contradish.integrity import compile_change, verify_change, IntegrityLedger
plan = compile_change("policy.json", since="2026-09-30", as_of="2026-10-10")
release = verify_change(plan, {"prod": my_agent_chat, "candidate": "anthropic:claude-sonnet-4-5"})
print(release.render())
IntegrityLedger("audit.jsonl", key="release.key.json").record_release(plan, release)
```

## 3. Guard every decision, in front of any model

```bash
contradish gateway policy.json --upstream https://api.openai.com --mode shadow \
    --log decisions.jsonl --ledger audit.jsonl --key gateway.key.json
```

Change only your client's base URL: `OpenAI(base_url="http://localhost:8787/v1")` or `Anthropic(base_url="http://localhost:8787")`. Any upstream that speaks either format works: OpenAI, Anthropic, Azure OpenAI, Google's and xAI's OpenAI-compatible endpoints, LiteLLM in front of Bedrock or Vertex, OpenRouter, Together, Groq, vLLM, Ollama. Provider credentials pass through and are never stored.

Your backend sends the request's evidence, taken from your systems of record, in an `X-Contradish-Context` header or a `contradish` body field. That field is removed before the request is forwarded:

```json
{"evidence": {"days_since_fill": {"value": 27, "source": "pharmacy_record"}}, "as_of": "2026-10-10", "request_id": "r-1"}
```

The guard decides from the policy and that evidence, never from the model's own text. A fact typed by the customer stays a claim and settles nothing.

- **shadow:** the response is returned unchanged, with the decision attached (`X-Contradish-Outcome`, `X-Contradish-Digest`, and a `contradish` field) and logged.
- **enforce:** blocked calls are removed before your code can run them. If none remain, the turn ends with a one-line explanation. A request with no evidence establishes nothing, so an irreversible call is blocked (it fails closed). Streamed responses with tools are refused in enforce mode, because a streamed call can't be inspected before it runs.

In-process, without the gateway: `Guard("policy.json", ledger=IntegrityLedger("audit.jsonl"))`.

## 4. Keep the evidence

`audit.jsonl` holds every plan, release verdict and runtime decision in one log:

- **Hash chain.** Each entry hashes its content and the previous entry's hash. Editing, removing or reordering any entry breaks every later hash.
- **Signed checkpoints.** Checkpoints sign the current head with an Ed25519 key (`contradish versions keygen --source auditor --out auditor.key.json`). The gateway signs every `--checkpoint-every` entries and on shutdown, and `contradish integrity ledger checkpoint` signs on demand. Signing checkpoints rather than every decision keeps the request path fast.

To verify a ledger without trusting whoever wrote it:

```bash
contradish integrity ledger verify audit.jsonl --trust <auditor public key>
```

Verification recomputes the chain and checks every signature against the keys you trust. It also re-derives every runtime decision with `contradish/evidence_check.py`, a separate standard-library implementation. The result states how much of the log a trusted signature covers.

## What this does and does not establish

- Plans are **exact** for the policy as encoded. Whether the encoding matches your intent is what drafting, review and named approval are for ([DRAFTING.md](DRAFTING.md)).
- Release verdicts are **measurements on the regression set**, not proofs about every input. The set is exhaustive when the situation space fits in `--max-situations`, and the plan says so (`sampling.exhaustive`).
- The gateway enforces tool calls. It does not judge free-text answers.
- "For the right reasons" is tested by behavior: the change must follow from who issued it and when it takes effect, never from wording or pressure. contradish does not grade the model's explanation of itself.
