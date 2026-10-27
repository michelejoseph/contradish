# Authorization-preserving memory transformations

**A verified interface for changing what an agent remembers without changing what it is authorized to do.**

Every agent framework rewrites memory. It summarizes conversations, consolidates notes, forgets old entries, merges duplicates, and rolls "a delegation from Dana under the owner's grant" into "may act until October 20". These are internal state transformations, and today nothing checks them. A rewrite that looks harmless can widen what the agent will do: the October 20 summary keeps dispensing after Dana legitimately revokes her delegation. It can also narrow what the agent does, or leave the agent unable to decide.

contradish decides, for any proposed transformation, whether it is safe:

> A transformation of memory **M** into **M′** is *authorization-preserving* relative to a policy if, on every future the policy permits, the agent's required, permitted and forbidden actions under **M′** are exactly those under **M**.

The futures covered are:
- every day of the horizon, under the policy version in force that day;
- clock facts advancing day by day;
- each kept grant revoked by its issuer on the first day and on each version change;
- every pair of kept grants revoked on the first day.

A revocation cascades to sub-delegations.

## The interface

```python
from contradish.transform import check_transform, repair, compress, MemoryGuard

v = check_transform("policy.json", before, after, as_of="2026-09-30", reason="nightly summarization")
v["verdict"]            # "SAFE" | "UNSAFE"
v["counterexamples"]    # first of each kind: the day, the future, the step, what the agent does before and after

r = repair("policy.json", before, after, as_of)       # the largest safe part of the requested change
c = compress("policy.json", memory, as_of)            # drop every fact the rest of the memory makes irrelevant

guard = MemoryGuard("policy.json", memory, as_of=today, mode="enforce", on_unsafe="repair", ledger=ledger)
receipt = guard.propose(new_memory, reason="consolidate")   # commits only what is safe
```

From any language:

```bash
contradish transform serve policy.json --port 8788
curl -s localhost:8788/v1/memory/transform -d '{"before": ..., "after": ..., "as_of": "2026-09-30", "repair": true}'
```

Command line: `contradish transform demo | check | repair | compress | serve`.

## What a framework sends: `contradish.memory/1`

```json
{
  "facts": {
    "days_since_fill":   {"lo": 20, "hi": 23, "on": "2026-09-30"},
    "refills_remaining": {"lo": 1, "hi": 11},
    "schedule":          {"values": ["c3_c5", "none"]},
    "prescriber_reachable": {"drop": true}
  },
  "authority": {
    "agent_may_dispense": {"root": "pharmacy_owner", "scope": "dispense", "chains": [
      {"links": [{"id": "G1", "issuer": "pharmacy_owner"}, {"id": "G2", "issuer": "dana_lead"}],
       "from": "2026-09-30", "until": "2026-10-20"}]}
  },
  "unknown": []
}
```

- A fact is given exactly (`{"value": v}`), as a range (`lo`/`hi`), as a set of values, or dropped.
- A clock fact carries the date its value refers to (`on`).
- A delegated fact is a list of chains of (grant id, issuer), each with the interval it is in force. A chain with no links is authority nobody can revoke. That is exactly what a summary like "may act until October 20" amounts to, and why it is unsafe.

`contradish memory compile` produces this form from a raw history: records, claims, grants and revocations. Frameworks map their own memory items onto the policy's facts. That mapping is the one piece of adoption work.

## What comes back: `contradish.transform_verdict/1`

The verdict holds:
- `verdict`;
- `violations_by_kind` and `counterexamples`;
- `futures_checked`;
- the version timeline with compiled policies;
- `before` and `after` with their digests;
- a `digest` of the whole verdict.

The violation kinds, in order of precedence:

| kind | meaning |
|---|---|
| `authority_expansion` | after the change, the agent makes an irreversible call the memory before forbids |
| `obligation_forgotten` | an obligation open in the memory before is not in the memory after |
| `fulfillment_loss` | an obligation the memory before could pay on some future cannot be paid on it after: a parameter dropped (the card, the order), or the route lost |
| `withholding` | a call the memory before requires is no longer made |
| `changed_action` | the same calls with different arguments, or a new reversible call |
| `indeterminacy` | the memory before decides; after the change it cannot |
| `fabrication` | the memory before cannot decide; after the change it decides, which is certainty the history never supported |

## Why you can believe it

- **Independent re-derivation.** `python contradish/memory_check.py verdict.json` re-derives a verdict from its own contents. It uses the standard library only and imports nothing from contradish. Policies are evaluated with `evidence_check.py`'s separately written evaluator. It recomputes the timeline, every situation each memory admits, every future, the behaviour on every critical day, the verdict, the count of every violation kind, and every counterexample. It rejects a verdict that claims SAFE for an unsafe change, one that hides revocation futures, and one whose memory was swapped, even when its digest is recomputed.
- **Ledger re-checking.** `MemoryGuard` receipts go into the signed audit ledger (`contradish.integrity`). `verify_ledger` re-derives each verdict. It rejects the log if an enforcing guard ever committed an unsafe change.
- **The checker earned its place.** During development it caught a soundness bug in the producer: two memories were compared only on the days both considered critical.

## The demo

```
summarize authority as 'may dispense until 2026-10-20'      UNSAFE  authority_expansion
  if dana_lead revokes G2 on 2026-09-30: before, the refill goes to the pharmacist; after, the agent dispenses it
round the fill day to 'about three weeks ago' (18-23)       UNSAFE  indeterminacy   (cannot decide on 2026-10-01)
forget the direct grant G3 (looks redundant)                UNSAFE  withholding     (stops dispensing refills it may)
forget whether the prescriber is reachable                  SAFE    (it matters only when no refills are left)
MemoryGuard, all four at once: applies the last; keeps authority and the fill day as before
```

## Remembering what is owed is not enough

A memory can keep what the agent may do and what it owes, and still lose the ability to pay. "Owes the patient $40" keeps the obligation and drops the card and the order. No decision depends on which card, so a check that preserves decisions alone would allow it. Obligations are therefore kept as their own part of the memory, with every fulfilment parameter exact, and checked on their own terms; see [FULFILLMENT.md](FULFILLMENT.md).

## Limits

- **Futures.** The futures are the ones listed above. Three or more simultaneous revocations, and changes nobody has issued yet, are not covered.
- **Exactness.** Results are exact when each integer fact is compared with fact-free constants and there is at most one clock fact. That is the same condition the memory compiler needs.
- **Scope.** Only decisions the policy encodes are protected. Memory the policy does not read is outside the guarantee, so a free-text summary used for tone is neither checked nor constrained.
