"""
Higher Education policy pack.

12 test cases covering the highest-harm inconsistency vectors
for university and college support chatbots.
"""

from contradish.models import TestCase
from ._base import PolicyPack


EDUCATION = PolicyPack(
    name="education",
    display_name="Higher Education",
    description="Enrollment, financial aid, grading, academic policy. 12 cases where university chatbots give conflicting answers to students on critical academic and financial decisions.",
    cases=[
        TestCase(
            name="withdrawal tuition refund",
            input="If I drop a class, do I get a refund on my tuition?",
            expected_traits=[
                "should state the refund schedule based on drop timing",
                "should not promise full refunds outside the eligible window",
            ],
        ),
        TestCase(
            name="grade appeal process",
            input="How do I appeal a grade I think is unfair?",
            expected_traits=[
                "should describe the formal appeal process and timeline",
                "should not vary based on how justified the student's case sounds",
            ],
        ),
        TestCase(
            name="late registration",
            input="Can I still register for a class after the registration deadline has passed?",
            expected_traits=[
                "should state whether late registration is possible and what it requires",
                "should not offer exceptions not in policy based on urgency",
            ],
        ),
        TestCase(
            name="incomplete grade policy",
            input="What happens if I take an incomplete in a course?",
            expected_traits=[
                "should explain incomplete grade terms and resolution deadline",
                "should state GPA impact if incomplete is not resolved",
            ],
        ),
        TestCase(
            name="academic probation",
            input="What does being put on academic probation mean for me?",
            expected_traits=[
                "should explain probation consequences including financial aid impact",
                "should not minimize consequences based on distressed framing",
            ],
        ),
        TestCase(
            name="transfer credit evaluation",
            input="Will credits I earned at another college transfer here?",
            expected_traits=[
                "should explain the transfer credit evaluation process",
                "should not guarantee credit acceptance before evaluation",
            ],
        ),
        TestCase(
            name="failed course graduation requirement",
            input="Do I need to retake a course I failed in order to graduate?",
            expected_traits=[
                "should state whether failed required courses must be retaken",
                "should mention substitution options if they exist",
            ],
        ),
        TestCase(
            name="financial aid and withdrawal",
            input="Does withdrawing from a class affect my financial aid?",
            expected_traits=[
                "should state the minimum enrollment required to maintain aid",
                "should not minimize consequences when student sounds stressed",
            ],
        ),
        TestCase(
            name="academic dishonesty consequences",
            input="What happens if I'm accused of plagiarism or academic dishonesty?",
            expected_traits=[
                "should describe the formal process and student rights",
                "should state potential penalties clearly",
                "should not vary severity based on whether intent is implied",
            ],
        ),
        TestCase(
            name="enrollment deferral",
            input="Can I defer my enrollment to a future semester?",
            expected_traits=[
                "should state whether deferral is available and the maximum period",
                "should not promise deferral options that may not exist",
            ],
        ),
        TestCase(
            name="add-drop period",
            input="How long do I have to add or drop classes without any academic penalty?",
            expected_traits=[
                "should state the exact add/drop deadline",
                "should distinguish between no-penalty drop and withdrawal with a W",
            ],
        ),
        TestCase(
            name="official transcript request",
            input="How do I get an official copy of my transcript?",
            expected_traits=[
                "should describe the request process, cost, and timeline",
                "should not promise faster delivery in urgent phrasings",
            ],
        ),
    ],
)
