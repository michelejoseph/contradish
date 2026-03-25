"""
contradish — CAI testing for LLM applications.

Detects CAI failures: when your app gives contradictory answers to semantically
equivalent inputs. Returns a CAI score per rule.

Quickstart:
    pip install contradish

    from contradish import Suite, TestCase

    suite = Suite(app=my_llm_function)
    suite.add(TestCase(input="Can I get a refund after 45 days?"))
    report = suite.run()

    # Access CAI scores programmatically
    print(report.cai_score)           # aggregate score
    for r in report.results:
        print(r.test_case.name, r.cai_score)

Full docs: https://contradish.com
"""

from .suite  import Suite
from .models import TestCase, Report, TestResult, RiskLevel

__version__ = "0.3.0"
__all__ = ["Suite", "TestCase", "Report", "TestResult", "RiskLevel"]
