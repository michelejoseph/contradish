# From your written policy to an approved policy.json (`contradish guard draft`)

Encoding a policy by hand is the slowest part of adopting the guard. A model can draft the encoding. But nothing a model drafts should govern a refund, a refill or a claim until a person has checked it, so contradish drafts, makes the checking fast and exact, and refuses to enforce anything nobody approved.

```
contradish guard draft policy.md --model anthropic:claude-sonnet-4-5 --tools dispense_refill,notify_patient
contradish guard review policy.policy.json        # writes policy.policy.review.md
contradish guard approve policy.policy.json --by "Dana Reyes" --role "pharmacy operations lead"
```

Try it offline first: `contradish guard draft --demo`.

## What happens

1. **Encode.** A model reads your policy text and writes `policy.json`: facts, rules with readable conditions, the tool calls each rule prescribes, which tools are irreversible, which deliver a benefit, and, for every rule and fact, the exact sentence of your policy it came from. Anything it had to assume goes in `assumptions` instead of being guessed silently.
2. **Ground** (deterministic, no model):
   - every quoted sentence must appear in your text;
   - every number in a rule must appear in the sentence it cites;
   - a number rescaled from it (80% written as 0.8) is flagged for review;
   - every sentence of your text that no rule cites is listed;
   - the draft must load and pass `guard lint`.

   Which system establishes each fact is rarely written in a policy, so the draft's `fact_authority` is always flagged as assumed.
3. **Cross-examine.** A second, separate model call sees only your text and the names of the facts and tools. It never sees the draft's conditions. It writes test cases near every limit and exception the text states. The draft is run on each case. Every disagreement means the encoding, the test or the policy's wording is wrong, and a person decides which. You can use a different model as the examiner with `--examiner`.
4. **Ask about the boundaries** (deterministic). From the draft's exact situation partition, there is one plain-language question at every place its decision changes: just below and at every threshold, and every value of every yes/no or category fact, around a typical case. For example:

   > days since fill 24, refills remaining 1, ... The draft decides: notify_patient(outcome=not_yet_due), tell_refill_date(day=24).

   The reviewer answers questions about outcomes, not code.
5. **Sign off.** The draft has `status: "draft"`. `approve` records who approved it, in what role and when, bound to a digest of everything that decides what the guard does: the rules and the updates. Examples and review notes are outside the digest, so confirmed examples can be added without re-approving.

## What the guard does with it

| Policy | `mode="shadow"` | `mode="enforce"` |
|---|---|---|
| draft | runs, with a warning | **refused** |
| approved | runs | runs |
| changed after approval (a rule, a number, a condition or an update) | runs, with a warning | **refused** until approved again |
| written by hand, no approval recorded | runs | runs (or refused with `require_approval=True`) |

Every decision certificate carries the approval: who approved it, in what role, when, and the digest. "Which person vouched for the rule this decision applied?" has an answer for every decision.

## What `approve` refuses

- a policy that does not load, fails lint, or fails one of its own `examples`;
- a source quote that is not in the policy text;
- a disagreement with the independent reading that is still open.

To resolve a disagreement, fix the encoding (the disagreement disappears), or decide the encoding is right and pass `--despite-disagreements`. The cases you overrode are recorded in the approval.

## The demo

`contradish guard draft --demo` drafts the fictional refill procedure in `contradish/templates/refill_policy.md`. The encoder's and examiner's replies are scripted stand-ins, so the demo runs offline. Everything after those replies is the real pipeline.

The scripted encoding has one deliberate error. The policy says a refill is allowed "once 80% of it has elapsed", and the draft writes `days_since_fill > refill_day`, so a patient on day 24 is told to come back on day 24. The review catches it twice: the independent reading expects a refill on day 24, and the boundary questions show the draft's answer for day 24. Approval is refused until it is fixed. After the fix, the review is clean, approval succeeds, and in enforce mode the guard flags `dispense_refill(days=30)` as owed on day 24.

## Limits

- **The model can be wrong in ways both readings share.** If the encoder and the examiner misread a sentence the same way, nothing flags it. A different examiner model (`--examiner`) and the boundary questions reduce this; the person approving is the last check, and that is the point of the approval.
- **Grounding checks citations, not meaning.** A rule can quote the right sentence and encode it wrongly. The examiner and the boundary questions are there for that.
- **Only what decides a tool call is encoded.** "Be courteous" is listed as uncovered text, not checked.
- **No real-model results are published yet.** The pipeline is tested with scripted replies. How often a given model drafts a correct encoding, and how often the examiner catches its errors, still has to be measured.
