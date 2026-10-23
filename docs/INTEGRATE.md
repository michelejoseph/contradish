# Integrate contradish in a day, verify it yourself, show its value in a month

contradish's guard sits between your agent and your tools. Before a proposed tool call runs, it answers two questions about the request, exactly, from your policy file (no model, no judge, no network):

- **May it act?** An irreversible call that what is established does not justify is blocked.
- **Is it withholding?** Something the person is already entitled to, under every amendment in force, that the agent refuses, delays or cuts short is flagged as owed.

When the case is not settled, it says which facts to ask for, from whom: the fewest that settle it. Every decision is logged as a certificate that a separate program re-derives without trusting contradish.

```python
from contradish import Guard

guard = Guard("policy.json", mode="shadow", log="decisions.jsonl")
decision = guard.check(evidence, proposed_calls, as_of=today, request_id=request_id)
run_these = decision.calls_to_run()      # shadow: unchanged; enforce: minus what is blocked
```

## Day one

| Hour | Do | Check |
|---|---|---|
| 0:00 | `pip install git+https://github.com/michelejoseph/contradish` then `contradish guard demo` | the whole loop runs offline and prints a value report |
| 0:15 | `contradish guard init guard/` | `guard/policy.json` (a starter policy) and `guard/guard_example.py` |
| 0:30 to 4:00 | Encode your policy in `policy.json` (below), or draft it from your written policy: `contradish guard draft policy.md --model provider:model`, then work through the review sheet it writes ([DRAFTING.md](DRAFTING.md)). Write 10 to 20 `examples` in your own terms. | `contradish guard lint` and `contradish guard test` both pass; a drafted policy shows no errors or open disagreements in `contradish guard review` |
| 4:00 | In your agent's request handler, call `guard.check(records, proposed_calls, as_of=..., request_id=...)` in **shadow** mode. `guard_example.py` shows where your records lookup and your agent plug in. | decisions appear in `decisions.jsonl` |
| 5:00 | `contradish guard verify decisions.jsonl` | every decision `VERIFIED` by the standalone checker |
| 5:30 | Add `contradish guard lint` and `contradish guard test` to CI | a policy change that breaks an example fails the build |

Shadow mode never stops anything: it is safe to ship on day one. Before switching to enforce mode, a named person approves the policy: `contradish guard approve policy.json --by "Name" --role "..."`. The guard refuses to enforce a model-drafted policy nobody approved, or one changed after its approval, and every decision certificate records who approved the policy it applied.

### What goes in `policy.json`

```json
{
  "facts":            {"days_since_fill": {"type": "int", "min": 0, "max": 90}, "...": "..."},
  "fact_authority":   {"days_since_fill": ["pharmacy_record"], "identity_verified": ["identity_check"]},
  "irreversible_tools": ["dispense_refill", "notify_patient"],
  "benefit_steps":    ["dispense"],
  "sources":          {"pharmacy_owner": {"governs": ["*"]}, "insurer": {"governs": ["R1"]}},
  "clauses": {
    "R1": {"text": "A 30-day supply may be refilled once {early_refill_pct}% of it has elapsed.",
           "params": {"supply_days": 30, "early_refill_pct": 80},
           "define": {"refill_day": "supply_days * (early_refill_pct / 100)",
                      "refill_due": "days_since_fill >= refill_day"}},
    "R6": {"define": {"eligible": "identity_verified and refill_due and has_refill and not controlled_ii"},
           "steps": {"dispense": {"tool": "dispense_refill", "when": "eligible", "args": {"days": "supply_days"}}}}
  },
  "updates":  [{"id": "insurer_70", "source": "insurer", "effective": "2026-10-01",
                "edits": [{"clause": "R1", "params": {"early_refill_pct": 70}}]}],
  "examples": [{"name": "due, one refill left", "facts": {"days_since_fill": 25, "...": "..."},
                "includes": ["dispense_refill"]}]
}
```

- **facts**: what describes a request. Ints need `min` and `max`.
- **fact_authority**: which of your systems establishes each fact. Evidence from any other source (what the customer typed, a note, a web page) is kept as a claim and settles nothing. This is how the guard tells "the record says day 22" from "the customer says day 22".
- **irreversible_tools**: calls that cannot be undone (money moved, medicine dispensed, a person notified). Only these are blocked; other unjustified calls are flagged for review.
- **benefit_steps**: steps that deliver something to the person. These are checked for withholding.
- **clauses**: your policy, one clause per rule, each owning its numbers (`params`), its definitions (`define`) and the tool calls it prescribes (`steps`). Conditions are written as readable expressions: `and or not`, comparisons, `+ - * /`, `x if c else y`, `min max round`, `'quoted text'`, `x in ('a', 'b')`.
- **sources** and **updates**: who may change which clause, and the amendments with their effective dates. An update from a source without authority over its clause changes nothing; one not yet in force changes nothing yet.
- **examples**: cases in your own terms with the calls you expect (`calls`, `includes`, `excludes`). They are how you, and later an auditor, check that the encoding says what you mean.

`contradish/templates/starter_policy.json` is a complete, working example (a fictional pharmacy refill policy) written this way.

### What `guard.check` takes

- **evidence**: `{fact: (value, source)}`, or a list of `{"fact", "value", "source"}` (or `"lo"`/`"hi"` for a range, `"values"` for one of several).
- **proposed calls**: an OpenAI chat message or its `tool_calls`, Anthropic `tool_use` content blocks, or `[{"tool", "args"}]`.
- **as_of**: the request date (it decides which amendments are in force).

### What it returns

`decision.outcome` is one of `deliver` (something owed is being withheld), `block`, `verify`, `review` or `allow`, with:

- `decision.owed`: the exact calls owed;
- `decision.blocked`: calls not to execute;
- `decision.ask`: facts to establish, and from whom;
- `decision.required`: when the evidence settles everything, exactly the calls the policy requires;
- `decision.reasons`: plain sentences, including which amendment a withheld benefit depends on and the smallest set of established evidence that already entitles the person;
- `decision.certificate`: the record that gets logged.

`decision.calls_to_run(deliver_owed=True)` in enforce mode gives what to execute.

## Verify it yourself

`contradish/evidence_check.py` is a single file that uses only the Python standard library and imports nothing from contradish. For each decision it re-derives from the certificate alone:

- which updates were in force and had authority;
- which evidence is established;
- every situation consistent with it, by enumerating every value of every fact, where the guard uses regions;
- each step's verdict, each benefit's status and owed call;
- that the facts to ask for settle the case and that no smaller set does;
- the outcome.

A decision it accepts has been established twice, by two implementations using two methods. It rejects any certificate whose digest or derivation does not match.

```
python contradish/evidence_check.py decisions.jsonl        # or: contradish guard verify decisions.jsonl
```

On the built-in policies, 2,300 decisions across every outcome were re-derived this way with no disagreement. Your reviewers can copy that one file and read it.

## Month one

| Week | Do | Output |
|---|---|---|
| 1 | Shadow mode on real traffic. Add each policy change to `updates` with its effective date as it happens. | `decisions.jsonl` |
| 2 | `contradish guard report decisions.jsonl --out week2.md` | what the agent withheld that people were owed, which amendments it missed, which irreversible actions ran on unestablished facts, which facts were most often missing |
| 3 | Fix what the report shows: missed amendments go into the prompt or the retrieval; missing facts go into the records lookup. Fix the encoding where an example or a reviewer disagrees. Turn on `mode="enforce"` for the irreversible tools where you are confident. | a shorter report |
| 4 | `contradish guard report decisions.jsonl --out month1.md` | the value report: counts and examples, every line backed by a certificate anyone can re-check |

The report re-verifies every decision before counting it, so its numbers are only those an independent program confirmed.

## What it does not do

- **It is exact relative to your encoding.** If `policy.json` says something your policy does not, the guard is faithfully wrong, and visibly so: every decision names the clauses, amendments and evidence it used, and your examples are the check on the encoding.
- **It sees only the facts your policy names.** A deciding fact the policy never mentions is invisible to it.
- **It checks tool calls, not prose.** What the agent says in text around the calls is not checked.
- **Numeric amounts that depend on an unestablished number** (a refund on an unverified price) are checked at the number's declared values and bounds: dependence is detected, not proved.
- **It is not yet published on PyPI** in this version: install from the repository.
