"""
contradish — reasoning stability testing for LLM applications.

Detect contradictions. Measure consistency. Catch regressions.

Quickstart:
    pip install contradish

    from contradish import Suite, TestCase

    suite = Suite(app=my_llm_function)
    suite.add(TestCase(input="Can I get a refund after 45 days?"))
    suite.run()

Full docs: https://contradish.com
"""

from .suite  import Suite
from .models import TestCase, Report, TestResult, RiskLevel

__version__ = "0.1.0"
__all__ = ["Suite", "TestCase", "Report", "TestResult", "RiskLevel"]
