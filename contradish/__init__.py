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
from .findings     import findings_from, Finding
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
from .adapters     import wrap_litellm, wrap_openai_compatible
from .caches       import FirewallCache, InMemoryCache, RedisCache
from .prompt_analyzer import analyze_prompt, PromptAnalysis, PromptTension
from .judge_calibration import measure_judge_floor, JudgeCalibration
from .fairness     import audit_fairness, FairnessAudit, IdentityProfile
from .memory       import (
    ConversationMemory,
    Commitment,
    ContradictionFinding,
    CommitmentStore,
    InMemoryCommitmentStore,
    RedisCommitmentStore,
    EmbeddingRelevance,
    openai_embedder,
)

__version__ = "1.12.0"
__all__ = [
    "Suite",
    "RegressionSuite",
    "Firewall",
    "PromptRepair",
    "improve",
    "ImprovementResult",
    "findings_from",
    "Finding",
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
    "wrap_litellm",
    "wrap_openai_compatible",
    "FirewallCache",
    "InMemoryCache",
    "RedisCache",
    "analyze_prompt",
    "PromptAnalysis",
    "PromptTension",
    "measure_judge_floor",
    "JudgeCalibration",
    "audit_fairness",
    "FairnessAudit",
    "IdentityProfile",
    "ConversationMemory",
    "Commitment",
    "ContradictionFinding",
    "CommitmentStore",
    "InMemoryCommitmentStore",
    "RedisCommitmentStore",
    "EmbeddingRelevance",
    "openai_embedder",
]
