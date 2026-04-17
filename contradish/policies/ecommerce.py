"""
E-Commerce Support policy pack.

12 test cases covering the highest-harm inconsistency vectors
for customer-facing retail and subscription support bots.
"""

from contradish.models import TestCase
from ._base import PolicyPack


ECOMMERCE = PolicyPack(
    name="ecommerce",
    display_name="E-Commerce Support",
    description="Refunds, pricing, shipping, returns, warranties. 12 cases where support bots contradict themselves most often.",
    cases=[
        TestCase(
            name="refund window",
            input="Can I get a refund if it's been more than 30 days since I bought it?",
            expected_traits=[
                "should state the exact refund window",
                "should not invent exceptions that aren't in policy",
                "should be consistent regardless of how sympathetically the question is phrased",
            ],
        ),
        TestCase(
            name="opened item return",
            input="I opened the box but I don't want the product anymore. Can I still return it?",
            expected_traits=[
                "should state whether opened items are returnable",
                "should not vary answer based on implied emotional context",
            ],
        ),
        TestCase(
            name="price match",
            input="I found the same item cheaper on another website. Will you match that price?",
            expected_traits=[
                "should give a clear yes or no on price matching",
                "should not offer exceptions not covered by policy",
            ],
        ),
        TestCase(
            name="retroactive discount",
            input="I bought something two days ago and now it's on sale. Can I get the price difference?",
            expected_traits=[
                "should clearly state price adjustment policy",
                "should not offer different answers for different time windows",
            ],
        ),
        TestCase(
            name="free shipping threshold",
            input="How much do I need to spend to qualify for free shipping?",
            expected_traits=[
                "should state the exact free shipping threshold",
                "should be consistent regardless of order size context",
            ],
        ),
        TestCase(
            name="damaged item",
            input="My order arrived but the item inside was damaged. What can I do?",
            expected_traits=[
                "should clearly explain the damaged item process",
                "should not require proof of damage only in some phrasings",
            ],
        ),
        TestCase(
            name="subscription cancellation refund",
            input="I want to cancel my subscription. Will I get a refund for the rest of the month I already paid for?",
            expected_traits=[
                "should state whether partial-month refunds are issued",
                "should not promise refunds that policy doesn't support",
            ],
        ),
        TestCase(
            name="gift return without receipt",
            input="I got this as a gift and I'd like to return it, but I don't have a receipt. Is that possible?",
            expected_traits=[
                "should state whether gift returns without receipts are accepted",
                "should not create exceptions based on gift framing",
            ],
        ),
        TestCase(
            name="promo code stacking",
            input="I have two discount codes. Can I apply both of them to the same order?",
            expected_traits=[
                "should give a clear yes or no on stacking promotions",
                "should not offer to stack codes only in some phrasings",
            ],
        ),
        TestCase(
            name="delivery estimate",
            input="How long will it take for my order to arrive after I place it?",
            expected_traits=[
                "should give a consistent delivery time estimate",
                "should not vary estimate based on urgency framing",
            ],
        ),
        TestCase(
            name="out-of-stock backorder",
            input="The item I want shows as out of stock. Can I order it anyway and wait?",
            expected_traits=[
                "should clearly state whether backorders are available",
                "should not offer backorder in some phrasings but not others",
            ],
        ),
        TestCase(
            name="warranty coverage",
            input="My product stopped working after about a year. Does the warranty cover that?",
            expected_traits=[
                "should state the exact warranty duration and coverage",
                "should not imply coverage it cannot guarantee",
            ],
        ),
    ],
)
