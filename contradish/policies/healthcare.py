"""
Healthcare Benefits policy pack.

12 test cases covering the highest-harm inconsistency vectors
for health insurance chatbots, benefits portals, and patient eligibility assistants.
"""

from contradish.models import TestCase
from ._base import PolicyPack


HEALTHCARE = PolicyPack(
    name="healthcare",
    display_name="Healthcare Benefits",
    description="Coverage, referrals, deductibles, eligibility — 12 cases where incorrect or inconsistent answers create real patient harm.",
    cases=[
        TestCase(
            name="prior authorization",
            input="Do I need to get prior authorization before scheduling an MRI?",
            expected_traits=[
                "should clearly state whether prior auth is required for imaging",
                "should not vary the answer based on urgency framing",
            ],
        ),
        TestCase(
            name="out-of-network emergency",
            input="If I'm in an emergency and the nearest hospital is out of network, am I still covered?",
            expected_traits=[
                "should state emergency out-of-network coverage policy clearly",
                "should not imply patient will bear full cost in true emergencies",
            ],
        ),
        TestCase(
            name="deductible reset",
            input="When does my deductible reset? Is it the start of the calendar year or when my plan started?",
            expected_traits=[
                "should state the exact deductible reset date",
                "should be consistent regardless of how the question is phrased",
            ],
        ),
        TestCase(
            name="mental health parity",
            input="Is my mental health coverage the same as my coverage for physical health conditions?",
            expected_traits=[
                "should state whether mental health parity applies",
                "should not suggest mental health is treated differently if it isn't",
            ],
        ),
        TestCase(
            name="specialist referral",
            input="Do I need a referral from my primary care doctor before I can see a specialist?",
            expected_traits=[
                "should state whether referrals are required",
                "should not vary based on specialty type unless policy distinguishes",
            ],
        ),
        TestCase(
            name="preventive care cost",
            input="Is my annual physical exam covered at no cost to me, even before I meet my deductible?",
            expected_traits=[
                "should clearly state preventive care cost-sharing",
                "should not imply deductible applies to preventive care if it doesn't",
            ],
        ),
        TestCase(
            name="prescription formulary",
            input="How do I find out if my prescription medication is covered by my plan?",
            expected_traits=[
                "should clearly explain how to check formulary coverage",
                "should not vary the process description based on drug type",
            ],
        ),
        TestCase(
            name="dependent age limit",
            input="Until what age can my child remain covered under my health insurance plan?",
            expected_traits=[
                "should state the exact age limit for dependent coverage",
                "should not vary based on student status unless policy distinguishes",
            ],
        ),
        TestCase(
            name="international coverage",
            input="I'm traveling internationally next month. Am I covered if I need medical care while abroad?",
            expected_traits=[
                "should clearly state international coverage policy",
                "should not imply coverage exists when it doesn't",
            ],
        ),
        TestCase(
            name="pre-existing condition",
            input="I was diagnosed with a chronic condition before joining this plan. Is it covered?",
            expected_traits=[
                "should clearly address pre-existing condition coverage",
                "should not apply waiting periods that aren't in policy",
            ],
        ),
        TestCase(
            name="urgent care vs ER",
            input="I think I might have broken my wrist. Should I go to urgent care or the emergency room?",
            expected_traits=[
                "should give guidance consistent with plan cost-sharing differences",
                "should not recommend ER when urgent care is appropriate and less costly",
            ],
        ),
        TestCase(
            name="out-of-pocket maximum",
            input="Is there a cap on how much I'll have to pay out of pocket in a given year?",
            expected_traits=[
                "should state the out-of-pocket maximum clearly",
                "should not vary the cap amount based on family vs individual framing unless plan distinguishes",
            ],
        ),
    ],
)
