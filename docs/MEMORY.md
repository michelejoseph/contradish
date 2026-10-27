# The memory compiler (`contradish memory`)

**From an agent's full history and its policy, derive the smallest memory that decides every future action the policy permits exactly as the full history would. Certify the compression, and revalidate the certificate as authorization evolves.**

Two agents can agree on every fact about the present and still need different memories to act safely later, because what a memory must carry is fixed by the futures it will face. With a policy file those futures are concrete, and the compiler covers each of them over a horizon:

- **Time passing.** Clock facts advance (days since a fill), and decisions are taken on every future day.
- **Scheduled policy changes.** Updates already issued by a source with authority, effective later.
- **Delegation and revocation.** On any future day, a grant may be revoked by its issuer or by the root. Scheduled revocations and expiries take effect on their dates. A revoked grant ends every sub-delegation made under it, and a later grant does not restore them.

Two histories need the same memory exactly when every such future treats them the same. That is the Myhill–Nerode quotient, taken relative to the policy.

```bash
contradish memory demo
contradish memory compile policy.json --history history.json --agent agent --as-of 2026-09-30 --out mem.json
contradish memory check mem.json --history history.json          # independent checker
contradish memory revalidate mem.json policy.json --events new.json --out mem2.json   # exit 1 if invalidated
contradish memory show mem2.json                                  # the memory as the agent carries it
```

## What the compiler keeps

- **Facts.** For each fact, the compiler partitions its values into classes that no decision distinguishes. That holds on every day of the horizon, under the version in force that day, whatever the other facts are and whether or not the agent holds its delegation. The memory keeps the class of the *established* value, meaning the latest record from a source with authority over the fact; claims from anyone else are discarded. A clock fact is kept as a class on an anchor date, so the memory stays exact as days pass. In the demo, "day 22 since the fill" becomes "between 20 and 23 on 2026-09-30": every value in that range is not due today and due from October 1 under the insurer's change.
- **Authority.** For a delegated fact, the memory keeps the minimal chains of grants from the root to the agent that are in force at some point in the horizon. Each chain is kept as (grant id, issuer) links plus the interval it is in force. Everything else is discarded, because no permitted future can depend on it:
  - holders of intermediate grants;
  - sibling delegations and grants for other scopes;
  - expired and revoked grants;
  - revocations by people with no standing.

  Issuers are kept, because whether a future revocation counts depends on who issued the grant.

## The certificate

`contradish.memory_certificate/1` holds:

- the memory;
- every partition, with a **witness for every boundary**: two neighbouring classes, the other facts, the day and the step on which they are decided differently. This is the proof that no coarser memory is sufficient;
- the version timeline;
- what was discarded, and why;
- the history digest;
- a digest of the whole certificate.

`contradish/memory_check.py` re-derives all of it. It uses the standard library only and imports nothing from contradish; policies are evaluated with `evidence_check.py`'s separately written evaluator. It checks:

1. **Integrity.** The digest matches the content.
2. **Versions.** The timeline is exactly the authorized updates applied in effective-date order.
3. **Sufficiency.** Within every class, every value is decided identically on every critical day.
4. **Minimality.** Every witness verifies.
5. **The memory.** Given the history, each kept fact is the class of the established value, and each chain is exactly the chain the checker derives.
6. **Lineage.** The certificate supersedes the previous one.

Forged certificates are rejected even with a recomputed digest. The forgeries tested include:

- a coarser memory;
- a missing or moved witness;
- a dropped grant, or an ignored scheduled revocation;
- a patient's claim kept instead of the record;
- a tampered future threshold;
- a version from a source with no standing.

## Revalidation as authorization evolves

`revalidate(cert, policy, new_updates, new_events, as_of, history=None)` produces a new certificate that names the one it supersedes:

- **Reused partitions.** A partition is reused when no changed definition reads its fact or shares a reader with it. New delegation events or fact records never change a partition, so they reuse all of them.
- **A fact that still fits.** The kept memory is re-checked against the new partitions. A kept class that still lies inside one new class stays valid, and may become coarser.
- **A fact that no longer fits.** If a newly announced change tells apart values the memory merged, the certificate is **invalidated**. It names the update and the day the distinction becomes load-bearing, and says where to re-establish the fact. In the demo, an announced return to 90% from October 3 makes days 20–23 diverge. Given the history, the compiler recovers the exact value automatically.
- **A grant memory can't place.** A grant made under a grant the memory did not keep can't be evaluated from memory, so the certificate is invalidated unless the history is supplied.

Certificates go in the signed audit ledger with `IntegrityLedger.record_memory`, or `--ledger L --key K` on the command line. The ledger then holds the memory's whole lineage, hash-chained and signed.

## Limits

- **Minimality is per fact.** The memory is minimal among memories that keep each fact separately, one class per fact. A memory that ties facts together can be smaller when they interact; for example, while identity is unverified nothing else decides today.
- **Exactness conditions.** The partitions are exact when each integer fact is compared with fact-free constants, which is the condition under which contradish's situation universe is exact. Exactness also requires at most one clock fact.
- **Covered futures.** The futures covered are those the policy and history make possible: declared updates, revocation by anyone with standing, time. New grants never make two histories differ, so they are covered. An update nobody has issued yet is not covered; for that, keep the record recoverable.
- **Speed.** Compilation and checking take seconds on the example. Revalidating after a policy change recomputes the affected partitions, about 15 seconds in the demo.
