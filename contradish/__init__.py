"""
contradish — CAI testing for LLM applications.

Detects CAI failures: when your app gives contradictory answers to semantically
equivalent inputs. Returns a CAI score per rule.

Tools:
    Suite         — offline CAI testing (run before deploy)
    RegressionSuite — compare baseline vs candidate for CI/CD gates
    Firewall      — real-time contradiction detection in production
    PromptRepair  — auto-generate and test improved prompt variants

Quickstart:
    pip install contradish

    from contradish import Suite, TestCase

    suite = Suite(app=my_llm_function)
    suite.add(TestCase(input="Can I get a refund after 45 days?"))
    report = suite.run()

    print(report.cai_score)           # aggregate score: 0.0-1.0
    for r in report.results:
        print(r.test_case.name, r.cai_score)

Full docs: https://contradish.com
"""

from .suite        import Suite
from .regression   import RegressionSuite
from .firewall     import Firewall
from .repair       import PromptRepair
from .models       import (
    TestCase,
    Report,
    TestResult,
    RiskLevel,
    RegressionResult,
    FirewallResult,
    RepairResult,
)
from .policies     import load_policy, list_policies, PolicyPack
from .fingerprint  import fingerprint, FailureCluster
from .exporters    import to_langfuse, to_phoenix
from .audit        import to_audit_html

__version__ = "0.5.1"
__all__ = [
    "Suite",
    "RegressionSuite",
    "Firewall",
    "PromptRepair",
    "TestCase",
    "Report",
    "TestResult",
    "RiskLevel",
    "RegressionResult",
    "FirewallResult",
    "RepairResult",
    "load_policy",
    "list_policies",
    "PolicyPack",
    "fingerprint",
    "FailureCluster",
    "to_langfuse",
    "to_phoenix",
    "to_audit_html",
]
