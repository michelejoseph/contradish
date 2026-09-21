"""
policy-audit/policy_audit_legal_compliance_judgments.py -- the manual judge
pass over candidate claim pairs generated from contradish.com's own
legal.html (Terms of Service + Privacy Policy) and compliance.html.

Produced by a live reviewer (this session, no ANTHROPIC_API_KEY /
OPENAI_API_KEY available in the sandbox this ran in) reading each candidate
pair directly and returning the same verdict shape PolicyContradictionJudge
would get back from a real LLM's complete_json() -- see
ManualJudgeClient / audit_policies_manual() in contradish/policy_contradiction.py.

Keys are pair_key(claim_a, claim_b) -- "sorted claim_id|claim_id" -- not a
positional index, so this file stays valid even if generate_candidate_pairs
is rerun with a different max_pairs or a tweaked extraction heuristic.

Only pairs worth a real explanation are listed here (the one genuine finding,
plus five deliberately-chosen near-misses that explain why the judge did NOT
flag them -- these are the calibration evidence, not just the hit). Every
other candidate pair falls back to DEFAULT.
"""

JUDGMENTS = {
    # ---- the one real finding ----
    "legal.html#27|legal.html#29": dict(
        is_contradiction=True,
        contradiction_type="scope_conflict",
        shared_subject="whether the right to delete personal data is unconditional or jurisdiction-dependent",
        confidence=0.55,
        severity="medium",
        explanation=(
            "Section 5 states 'You can request deletion at any time' with no qualifier, "
            "and points the reader to Section 7 for details. Section 7, the section it "
            "points to, actually frames deletion as one of several rights you 'may have... "
            "depending on where you live' -- i.e. not universal. A reader who stops at "
            "Section 5's cross-reference reasonably comes away thinking deletion is always "
            "available; Section 7 says otherwise. Not a hard logical impossibility (both "
            "can be literally true -- you can always ASK, but the company isn't always "
            "obligated to comply) but a real scope mismatch between an absolute-sounding "
            "promise and the conditional right it cross-references. Easy fix: reword "
            "Section 5 to 'you can request deletion at any time, though the right to have "
            "it honored may depend on where you live (see Section 7)'."
        ),
    ),
    # ---- near-misses worth naming explicitly (why the judge did NOT flag them) ----
    "legal.html#23|legal.html#24": dict(
        is_contradiction=False, contradiction_type="none",
        shared_subject="selling data vs. sharing with service providers",
        confidence=0.9, severity="informational",
        explanation="'We don't sell your personal information' and sharing with contracted "
                     "service providers under use-restriction obligations describe different "
                     "things -- a commercial sale of data vs. a scoped processing relationship. "
                     "This is the standard sub-processor carve-out present in nearly every real "
                     "privacy policy, not a conflict with the no-sale claim."),
    "legal.html#26|legal.html#3": dict(
        is_contradiction=False, contradiction_type="none",
        shared_subject="\"retain\" used in two different senses",
        confidence=0.85, severity="informational",
        explanation="'You retain all rights in Customer Content' (ownership) and 'We retain "
                     "account information... for as long as your account is active' (storage "
                     "duration) both use the word 'retain' but for unrelated concepts -- legal "
                     "ownership vs. data-retention duration. A keyword-only matcher would flag "
                     "this; it isn't a real conflict."),
    "legal.html#14|legal.html#7": dict(
        is_contradiction=False, contradiction_type="none",
        shared_subject="\"free of charge\" vs. a liability cap that mentions an amount paid",
        confidence=0.8, severity="informational",
        explanation="The liability cap is 'the GREATER of (A) the amount you paid in the prior "
                     "12 months, OR (B) $100' -- the $100 floor is exactly what makes this clause "
                     "work correctly even when the amount paid is $0 under the current free "
                     "offering. Boilerplate written with a future paid tier in mind, but not "
                     "actually inconsistent with 'currently free.'"),
    "legal.html#26|legal.html#27": dict(
        is_contradiction=False, contradiction_type="none",
        shared_subject="data retention window vs. the right to request deletion",
        confidence=0.75, severity="informational",
        explanation="These are adjacent sentences in the same section, and the second is "
                     "explicitly written as the override to the first ('You can request deletion "
                     "at any time') -- deliberately structured to be consistent, unlike the "
                     "legal.html#27 / legal.html#29 pair above, where the cross-referenced "
                     "qualifier actually narrows the claim."),
    "legal.html#10|legal.html#5": dict(
        is_contradiction=False, contradiction_type="none",
        shared_subject="Customer Content in the CAI-Bench public dataset",
        confidence=0.7, severity="informational",
        explanation="'We don't include Customer Content in any public dataset unless you opt in' "
                     "and 'we publish the CAI-Bench dataset on Hugging Face' aren't in tension "
                     "unless CAI-Bench is built from customer submissions by default -- nothing "
                     "in either sentence claims that, and CAI-Bench is contradish's own research "
                     "benchmark, not an aggregation of customer data."),
}

DEFAULT = dict(
    is_contradiction=False,
    contradiction_type="none",
    shared_subject="",
    confidence=0.6,
    severity="informational",
    explanation="Shares incidental vocabulary with its pair (e.g. both mention \"account,\" "
                 "\"information,\" or \"third-party\") but addresses a different subject or "
                 "scope; not a genuine conflict.",
)
