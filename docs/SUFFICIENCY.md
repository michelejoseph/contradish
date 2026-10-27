# What must a representation keep? (`contradish license`)

Two systems can agree on every fact about the present, and on every decision due today, and still need different representations to decide safely tomorrow. What a memory, a summary or a case file must carry is fixed by the futures it will face, not by the present. With a policy file those futures are concrete. They are the policy in force today, plus every update a source with authority has already issued that takes effect later. So what must be kept can be **computed**.

```bash
contradish license demo                                     # two notes that agree today; one fails tomorrow
contradish license policy.json --as-of 2026-09-30           # per fact: what today needs, what the future adds
contradish license check policy.json --case case.json --note note.json --as-of 2026-09-30
contradish license notes policy.json --model anthropic:claude-sonnet-4-5 --as-of 2026-09-30 --cases 24
```

## The compression license

For each fact, the license lists:

- **the distinctions today's decisions depend on**, such as "days since fill: 0–23 vs 24–90";
- **the distinctions the future versions add**, each with the update and date that make it load-bearing. For example, "21 vs 20, under `insurer_early_refill_70` from 2026-10-01."

Two values of a fact may be merged exactly when no decision step, under any version, treats them differently, whatever the other facts are. A fact no version reads may be dropped. Updates from a source without standing over their clause change nothing and are listed as ignored.

## Is a representation sufficient?

A note keeps each fact exactly (`22`), as a range (`{"lo": 0, "hi": 23}`), as a set (`{"values": [...]}`), or not at all. Under each version, the note is **sufficient** when every situation it is consistent with requires the same calls, and those are the calls the true case requires. The verdict reports:

- `sufficient_now` and `sufficient_future`;
- `blind_spot`: sufficient today but not for a future version, the failure the question names;
- a **counterexample**: the case and the nearest situation the note cannot tell it apart from, which the future version decides differently. For example: "the note cannot tell day 22 from day 20; under the insurer's change one is dispensed and the other is not";
- `contradicts_case`: facts the note states falsely, so a correct answer today was luck.

`future_equivalent(policy, case, note_a, note_b)` compares two representations of the same case. They are *equivalent now* if they settle today's decision the same way, and *future-equivalent* if they do under every version. It also names the first version where they come apart.

## Do a model's own notes carry the future?

`contradish license notes` gives a model cases under today's policy, asks for the decision, and asks for the note it would leave for whoever reopens the case. Half the cases are chosen so that a future version decides them differently. Every note is checked. The **future blind spot** is the share of notes that are sufficient today and insufficient for a version already announced.

## Limits

- The license is exact for policies whose decisions compare each integer fact with fact-free expressions, the condition under which contradish's situation universe is exact. Otherwise it is exact over the universe it lists.
- The futures are the changes the policy file declares. A representation can still be blindsided by a change nobody has issued yet. For that, keep the record recoverable.
