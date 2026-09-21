# Policy contradiction audit -- contradish.com

Run against contradish.com's own `legal.html` (Terms of Service + Privacy
Policy) and `compliance.html`, using the new `contradish/policy_contradiction.py`
scanner as its own first dogfood target.

**39** claim-bearing sentences extracted, **72** candidate pairs evaluated,
**1** genuine contradiction found.

No `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` was available in the sandbox this
audit ran in, so the judge pass was done by a live reviewer session (me)
reading all 72 candidate pairs directly and returning the same verdict shape
a real LLM judge would -- the same role `contradish/benchmark_ground_truth_audit.py`
gives to "&ge;2 LLM reviewer models" for ground-truth auditing. Swap in
`PolicyContradictionJudge(LLMClient())` once a key is available and nothing
else about the pipeline changes (see `run_policy_audit.py --help`).

## The finding

**[MEDIUM &middot; scope_conflict &middot; confidence 0.55]** Whether the right to
delete personal data is unconditional or jurisdiction-dependent.

> **Privacy Policy &sect; 5, Data Retention:** "You can request deletion at
> any time (see Section 7)."
>
> **Privacy Policy &sect; 7, Your Rights:** "Depending on where you live, you
> may have the right to access, correct, export, or delete your personal
> information, or to object to certain processing."

Section 5 makes an absolute-sounding promise and cross-references Section 7
for "details." Section 7 -- the section it points to -- actually frames
deletion as one of several rights a person *may* have, *depending on where
they live*, not a universal guarantee. A reader who stops at Section 5's
cross-reference reasonably comes away thinking deletion is always available;
Section 7 says otherwise.

This isn't a hard logical impossibility -- both sentences can be literally
true at once (you can always *ask* for deletion; the company isn't always
*obligated* to honor it). It's a real scope mismatch between how confidently
Section 5 states the right and how conditionally Section 7 actually defines
it, in the exact section Section 5 sends the reader to for clarification.

**Suggested fix:** reword Section 5 to something like "you can request
deletion at any time, though the right to have it honored may depend on
where you live (see Section 7)" -- one clause, no product change, and it
closes the gap between what the promise implies and what the cross-reference
actually says.

## Near-misses (why these were *not* flagged)

The topic-clustering step that generates candidate pairs is deliberately a
recall tool, not a precision guarantee -- it pairs claims that share
vocabulary, and the judge step is where precision gets enforced. These five
are worth naming explicitly because each one is a plausible false positive a
cruder keyword-matching approach would flag, and each fails for a different
reason:

1. **"We don't sell your personal information" vs. sharing with contracted
   service providers.** Selling data and a scoped, contract-bound processing
   relationship with sub-processors are different concepts -- this is the
   standard sub-processor carve-out present in nearly every real privacy
   policy, not a conflict with the no-sale claim.

2. **"You retain all rights in Customer Content" vs. "We retain account
   information... for as long as your account is active."** Same word,
   "retain," used for two unrelated concepts: legal ownership vs.
   data-retention duration. A keyword-only matcher would flag this; it isn't
   a real conflict.

3. **"contradish is currently offered free of charge" vs. a liability cap
   phrased around "the amount you paid."** The cap is "the GREATER of (A)
   the amount you paid in the prior 12 months, OR (B) $100" -- the $100
   floor is exactly what makes the clause work correctly even when the
   amount paid is $0 under the current free offering. Boilerplate written
   with a future paid tier in mind, but not actually inconsistent with
   "currently free."

4. **The data-retention sentence immediately followed by the deletion-request
   sentence** (same section, adjacent sentences). Deliberately structured so
   the second is the override to the first -- unlike the real finding above,
   where the cross-referenced qualifier actually narrows the claim instead
   of sitting next to it as an obvious continuation.

5. **"We don't include Customer Content in any public dataset unless you opt
   in" vs. "we publish the CAI-Bench dataset on Hugging Face."** Only in
   tension if CAI-Bench were built from customer submissions by default --
   nothing in either sentence claims that, and CAI-Bench is contradish's own
   research benchmark, not an aggregation of customer data.

## What this run does and doesn't establish

This module is text-only, the same scope limit `contradish/compliance_gap.py`
already states for model transcripts: it checks a company's own published
sentences against each other, not against its actual behavior. A company can
publish two perfectly consistent policies and still violate both in
practice; this can't see that. Extraction and topic-clustering are both
heuristic (regex/keyword-based), so a claim phrased in a way the extractor
doesn't recognize as claim-bearing is invisible to everything downstream.
And this particular run's judge step was a single human reviewer standing in
for an LLM judge, not the real thing -- every finding ships with a
confidence score and a plain-language explanation specifically so it's fast
to re-check by hand, and the one finding above should be read as "worth a
copy edit," not as a legal determination.

## Files in this directory

- `../contradish/policy_contradiction.py` -- the reusable module (extraction,
  candidate pairing, judge interface, `ManualJudgeClient`/`audit_policies_manual`
  for no-API-key runs).
- `run_policy_audit.py` -- CLI: `python3 run_policy_audit.py legal.html compliance.html --manual-judgments policy_audit_legal_compliance_judgments.py`
- `policy_audit_legal_compliance_judgments.py` -- this run's manual judge pass, reusable as a regression check (rerun after any policy-page edit; if a previously-cleared pair starts scoring differently, something upstream in the extraction changed, not the policy text).
- `report.json` -- machine-readable output of this run.
- `REPORT.md` -- this file.
