"""
contradish — Reasoning stability testing for LLM applications.

pip install contradish

Usage:
    from contradish import Suite, TestCase

    suite = Suite(api_key="sk-ant-...", app=my_llm_function)
    suite.add_test(TestCase(input="Can I get a refund after 45 days?"))
    report = suite.run()
    print(report)
"""

from .suite import Suite
from .models import TestCase, Report, TestResult, RegressionResult
from .runner import Runner
from .judge import Judge
from .regression import RegressionSuite

__version__ = "0.1.0"
__all__ = [
    "Suite",
    "TestCase",
    "Report",
    "TestResult",
    "RegressionResult",
    "Runner",
    "Judge",
    "RegressionSuite",
]
