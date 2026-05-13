"""
contradish: CAI Strain testing for LLM applications.

Detects CAI failures: when your app gives contradictory answers to semantically
equivalent inputs. ML literature calls this drift; contradish names and scores it.
Returns CAI Strain per rule (0–1, lower = more consistent).

Tools:
    Suite           -- offline CAI Strain testing (run before deploy)
    RegressionSuite -- compare baseline vs candidate for CI/CD gates
    Firewall        -- real-time contradiction detection in production
    PromptRepair    -- auto-generate and test improved prompt variants

Quickstart:
    pip install contradish

    from contradish import Suite, TestCase

    suite = Suite(app=my_llm_function)
    suite.add(TestCase(input="Can I get a refund after 45 days?"))
    report = suite.run()

    print(report.cai_strain)          # aggregate CAI Strain: 0.0-1.0, lower is better
    for r in report.results:
        print(r.test_case.name, r.cai_strain)

The legacy `report.cai_score` attribute (0–1, higher is better) is preserved
for backward compatibility; cai_score == 1 - cai_strain.

Full docs: https://contradish.com
"""

from .suite        import Suite
from .regression   import RegressionSuite
from .firewall     import Firewall
from .repair       import PromptRepair
from .improve      import improve, ImprovementResult
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

__version__ = "1.3.0"
__all__ = [
    "Suite",
    "RegressionSuite",
    "Firewall",
    "PromptRepair",
    "improve",
    "ImprovementResult",
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
