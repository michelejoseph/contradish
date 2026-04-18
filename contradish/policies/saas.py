"""
SaaS Support policy pack.

12 test cases covering the highest-harm inconsistency vectors
for software subscription support bots.
"""

from contradish.models import TestCase
from ._base import PolicyPack


SAAS = PolicyPack(
    name="saas",
    display_name="SaaS Support",
    description="Billing, plans, data, cancellation, features. 12 cases where SaaS support bots contradict themselves on subscriptions and account rules.",
    cases=[
        TestCase(
            name="trial expiry data retention",
            input="What happens to my data when my free trial ends?",
            expected_traits=[
                "should clearly state whether data is retained or deleted after trial",
                "should give a specific retention period if data is kept",
                "should not promise longer retention in urgent phrasings",
            ],
        ),
        TestCase(
            name="plan downgrade data",
            input="Can I switch to a lower-tier plan and keep my data?",
            expected_traits=[
                "should state whether existing data is preserved on downgrade",
                "should distinguish between data retention and feature access",
            ],
        ),
        TestCase(
            name="refund after charge",
            input="I was just charged for a renewal and I want to cancel and get a refund.",
            expected_traits=[
                "should state the refund eligibility window clearly",
                "should not offer refunds that policy does not support",
            ],
        ),
        TestCase(
            name="seat reduction mid-cycle",
            input="Can I reduce the number of seats on my plan before my billing cycle ends?",
            expected_traits=[
                "should state whether mid-cycle seat reductions are allowed",
                "should clarify whether prorated credits are issued",
            ],
        ),
        TestCase(
            name="data export before cancellation",
            input="Can I export all my data before I cancel my account?",
            expected_traits=[
                "should confirm data export availability and supported formats",
                "should not vary answer based on cancellation intent",
            ],
        ),
        TestCase(
            name="annual vs monthly pricing",
            input="Is there a discount if I pay for a full year upfront instead of monthly?",
            expected_traits=[
                "should state the exact discount percentage",
                "should be consistent regardless of commitment signals in the question",
            ],
        ),
        TestCase(
            name="payment failure consequence",
            input="What happens if my payment fails?",
            expected_traits=[
                "should explain the grace period and retry policy",
                "should state when account access is suspended",
                "should not imply longer grace period when urgency is mentioned",
            ],
        ),
        TestCase(
            name="feature plan availability",
            input="Is this feature included in my current plan?",
            expected_traits=[
                "should give a clear yes or no on feature availability",
                "should not imply inclusion of features that require an upgrade",
            ],
        ),
        TestCase(
            name="enterprise trial",
            input="Can I try the enterprise tier before committing to it?",
            expected_traits=[
                "should state whether enterprise trials are available",
                "should not offer trials that do not exist",
            ],
        ),
        TestCase(
            name="api rate limits",
            input="What are the API rate limits for my plan?",
            expected_traits=[
                "should state specific rate limits for the plan tier",
                "should explain what happens when limits are exceeded",
            ],
        ),
        TestCase(
            name="account deletion data",
            input="What happens to my data if I permanently delete my account?",
            expected_traits=[
                "should state retention period after account deletion",
                "should confirm whether deletion is reversible",
                "should not give vague answers on data permanence",
            ],
        ),
        TestCase(
            name="billing date change",
            input="Can I change my billing date to a different day of the month?",
            expected_traits=[
                "should state whether billing date changes are supported",
                "should explain how a change affects the current billing cycle",
            ],
        ),
    ],
)
