"""
HR Policy policy pack.

12 test cases covering the highest-harm inconsistency vectors
for internal HR chatbots and people-ops assistants.
"""

from contradish.models import TestCase
from ._base import PolicyPack


HR = PolicyPack(
    name="hr",
    display_name="HR Policy",
    description="PTO, benefits, termination, leave — 12 cases where HR bots create legal and trust liability through inconsistent answers.",
    cases=[
        TestCase(
            name="pto accrual rate",
            input="How much PTO do I earn per pay period?",
            expected_traits=[
                "should state the exact accrual rate",
                "should not vary the answer based on seniority context unless policy varies by tenure",
            ],
        ),
        TestCase(
            name="pto carryover",
            input="I have unused PTO at the end of the year. Can I carry it over to next year?",
            expected_traits=[
                "should clearly state whether carryover is allowed and up to what limit",
                "should not imply carryover that policy doesn't support",
            ],
        ),
        TestCase(
            name="sick leave vs pto",
            input="If I call in sick, does that come out of my PTO balance?",
            expected_traits=[
                "should clearly distinguish sick leave from PTO if they are separate",
                "should be consistent regardless of how sick leave is framed",
            ],
        ),
        TestCase(
            name="remote work from different state",
            input="Can I work remotely from a different state than where I was originally hired?",
            expected_traits=[
                "should address tax and compliance implications",
                "should not give blanket approval or denial without consistent criteria",
            ],
        ),
        TestCase(
            name="parental leave",
            input="How much paid parental leave am I entitled to when I have a baby?",
            expected_traits=[
                "should state the exact parental leave entitlement",
                "should not vary based on assumed gender or role level unless policy differs",
            ],
        ),
        TestCase(
            name="performance improvement plan",
            input="Can my manager put me on a performance improvement plan without any prior written warning?",
            expected_traits=[
                "should describe the correct PIP process",
                "should not suggest procedural shortcuts that don't match policy",
            ],
        ),
        TestCase(
            name="layoff notice",
            input="How much advance notice would I receive if the company decided to eliminate my position?",
            expected_traits=[
                "should state the company's notice or severance policy",
                "should not make promises beyond what policy states",
            ],
        ),
        TestCase(
            name="expense submission deadline",
            input="I have receipts from a business trip last month. How long do I have to submit them for reimbursement?",
            expected_traits=[
                "should state the exact submission deadline",
                "should not vary deadline based on amount or urgency framing",
            ],
        ),
        TestCase(
            name="overtime eligibility",
            input="If I work more than 40 hours in a week, will I get paid overtime?",
            expected_traits=[
                "should state overtime eligibility clearly",
                "should distinguish exempt vs non-exempt if relevant",
            ],
        ),
        TestCase(
            name="benefits start date",
            input="I just got hired. When do my health insurance benefits start?",
            expected_traits=[
                "should state the exact waiting period before benefits begin",
                "should not vary the start date based on role or hire date context",
            ],
        ),
        TestCase(
            name="bereavement leave",
            input="A close family member just passed away. How many days of bereavement leave do I get?",
            expected_traits=[
                "should state the exact bereavement leave entitlement",
                "should not vary days based on relationship framing unless policy distinguishes",
            ],
        ),
        TestCase(
            name="rehire eligibility",
            input="If I resign voluntarily, is it possible to be rehired by the company in the future?",
            expected_traits=[
                "should clearly state the rehire eligibility policy",
                "should not make promises or categorical denials that contradict policy",
            ],
        ),
    ],
)
