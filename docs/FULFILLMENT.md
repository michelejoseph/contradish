# Can the agent still pay what it owes? (`contradish memory obligations`)

An agent can remember what it is allowed to do, remember what it owes, and still lose the ability to fulfil what it owes. There are three ways this happens:

1. **The memory drops what fulfilment needs.** "Owes a $40 refund" is kept, while the card and the order are dropped. No *decision* depends on which card, so a compression that preserves decisions throws it away.
2. **The capacity ends before the obligation does.** A grant expires, or a scheduled revocation takes effect, before the deadline.
3. **The capacity can be revoked.** The only route to fulfilment runs through grants someone may legitimately revoke.

## Declaring obligations

The policy declares obligation types under `memory.obligations`:

```json
"refund": {"tool": "issue_refund", "params": ["order_id", "card_token", "amount"],
           "authority": "agent_may_dispense", "sources": ["pharmacy_record"],
           "fallback": {"tool": "hand_to_pharmacist", "authority": null}}
```

The fallback is optional. The history records obligations, and their discharge, as events:

```json
{"kind": "obligation", "id": "O1", "type": "refund", "owed_to": "patient", "incurred": "2026-09-28",
 "due": "2026-10-25", "params": {"order_id": "A-1043", "card_token": "tok_4242", "amount": 40}, "source": "pharmacy_record"}
{"kind": "discharged", "id": "O1", "at": "2026-10-02"}
```

## What is checked

On a given day and future, an obligation can be paid if every parameter is known exactly and the agent holds the authority its type needs that day, or its fallback route is open that day. The futures are the same ones `contradish transform` uses: every day, scheduled versions, and single and paired revocations by issuers, with cascade.

`contradish memory obligations CERT.json policy.json` reports each open obligation as one of:

| status | meaning |
|---|---|
| `fulfillable` | a route is open on some day before the deadline, and stays open to the deadline |
| `act_by D` | with no revocation, every route closes on D, before the deadline: the last safe day to act |
| `stranded_by_schedule` | even with no revocation, no route is open on any day before the deadline |
| `params_missing` | the memory does not hold what paying it requires (named) |
| `overdue` | past its deadline |

Each status also comes with:

- **`stranding_futures`:** the legitimate revocations under which no route would remain. An obligation with any of these is *fragile*.
- **A remedy:** act earlier, re-establish the missing parameters, or add a fallback that does not run through a revocable grant.

In the demo, the $40 refund is due October 25. Every route to issuing it runs through grants that end October 10 and October 20, so the status is **act by October 19**. The refund is also fragile: if the owner revokes both chains today, it cannot be paid at all. The recall notice needs no grant, so it can be fulfilled on every future.

## In memory transformations

`check_transform` reports `obligation_forgotten` and `fulfillment_loss`. `repair` keeps an obligation intact whenever changing it is unsafe. The memory compiler keeps open obligations with exact parameters. `memory_check.py` re-derives the open obligations from the history, and the fulfilment violations in a verdict, independently.

## Limits

- Capacity here means parameters plus authority plus an open route. It does not cover external capacity the policy does not model, such as funds, a working payment rail, or the patient still being reachable at the stored contact.
- Fallback routes are checked for authority only, not for their own parameters.
