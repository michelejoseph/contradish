"""
Wire the contradish guard around your agent. Runs as-is (offline, with a stand-in model); then replace the two
marked functions with your own.

    python guard_example.py

1. `records_for(request)`: what your systems have on record for this request, each fact with the source it came
   from. Only sources named in policy.json's `fact_authority` establish a fact; anything else (what the customer
   typed, a note, a web page) is kept as a claim and settles nothing.
2. `agent(request)`: your agent's proposed tool calls, in whatever shape your SDK returns them: an OpenAI message,
   Anthropic content blocks, or [{"tool": ..., "args": {...}}].

Start in shadow mode: nothing is stopped, every decision is logged with a certificate. After a few weeks:
    contradish guard report decisions.jsonl          what it would have caught
    contradish guard verify decisions.jsonl          re-derive every decision without trusting contradish
Then switch to mode="enforce".
"""
import json
import os

from contradish import Guard

HERE = os.path.dirname(os.path.abspath(__file__))
guard = Guard(os.path.join(HERE, "policy.json"), mode="shadow", log=os.path.join(HERE, "decisions.jsonl"))


# ── 1. replace with your records lookup ──────────────────────────────────────
def records_for(request: dict) -> dict:
    """{fact: (value, source)}. Here: a pharmacy record, an identity check, and what the patient said."""
    db = {"rx-1001": {"days_since_fill": 22, "refills_remaining": 1, "schedule": "none"}}
    rec = db.get(request["rx"], {})
    out = {f: (v, "pharmacy_record") for f, v in rec.items()}
    out["identity_verified"] = (request.get("id_checked", False), "identity_check")
    if "patient_says_days" in request:                     # a claim: kept, settles nothing
        out.setdefault("days_since_fill", (request["patient_says_days"], "patient"))
    return out


# ── 2. replace with your agent ───────────────────────────────────────────────
def agent(request: dict):
    """A stand-in that still follows the old 80% early-refill rule, in OpenAI's tool-call shape."""
    return {"role": "assistant", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "tell_refill_date", "arguments": json.dumps({"day": 24})}},
        {"id": "2", "type": "function", "function": {"name": "notify_patient",
                                                     "arguments": json.dumps({"outcome": "not_yet_due"})}}]}


def handle(request: dict):
    proposed = agent(request)
    decision = guard.check(records_for(request), proposed, as_of=request["date"], request_id=request["id"])
    print(decision.explain())
    to_run = decision.calls_to_run(deliver_owed=True)   # shadow mode: exactly what was proposed
    for call in to_run:
        print("  executes now (shadow):", call.tool, dict(call.args))   # replace with your tool executor
    import dataclasses
    enforced = dataclasses.replace(decision, mode="enforce").calls_to_run(deliver_owed=True)
    print("  in enforce mode it would execute:", [(c.tool, dict(c.args)) for c in enforced] or "nothing, until verified")
    return decision


if __name__ == "__main__":
    handle({"id": "req-1", "rx": "rx-1001", "id_checked": True, "date": "2026-10-20"})
    print()
    handle({"id": "req-2", "rx": "rx-unknown", "id_checked": True, "patient_says_days": 22, "date": "2026-10-20"})
    print(f"\nLogged to {guard.log}. Next: contradish guard report {guard.log}")
