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
from .improve      import improve, improve_from_production, ImprovementResult
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
from .replay       import (
    replay,
    replay_transcript,
    load_transcript,
    ReplayReport,
    ReplayContradiction,
    ReplayTurn,
)
from .prompt_analyzer import commitments_from_analysis
from .reconcile    import (
    reconcile,
    ReconciliationReport,
    CommitmentMatch,
    cases_from_reconciliation,
)
from .ledger       import CommitmentLedger, LedgerEntry
from .report       import audit_report_html
from .admissibility import (
    AdmissibilityEngine,
    AdmissibilityResult,
    DomainIndex,
    ThresholdPolicy,
    CalibrationStore,
)
from .measurement import (
    ReasoningDimension,
    DIMENSIONS,
    MeasurementLaw,
    LAWS,
    ReasoningProfile,
    MeasurementUncertainty,
    profile_from_results,
    compare as compare_profiles,
)
from .epistemic import (
    EpistemicAudit,
    EpistemicProfile,
    DisagreementMap,
    InquiryScaffold,
)
from .observatory import (
    Constraint,
    ConstraintStatus,
    ConstraintProfile,
    ConstraintDelta,
    ConstraintObservatory,
    ConstraintProfiler,
)
from .structural_eval import (
    JunctionSensitivity,
    SensitivityProfile,
    StructuralDelta,
    StructuralEvaluationReport,
    StructuralEvaluator,
)
from .active_oracle import (
    GroundTruthSignal,
    ModelProbeResult,
    DiscoveryClassification,
    DiscoveryResult,
    ActiveOracle,
)
from .oracle import (
    NodeProbe,
    ConsensusNode,
    ConsensusTopology,
    TargetedPerturbation,
    OracleResult,
    TopologyOracle,
    TopologyRegistry,
)
from .topology import (
    ReasoningNode,
    ReasoningEdge,
    TopologyPath,
    FailureTopologyMap,
    topology_distance,
    topology_from_phi_star,
)
from .convergence import (
    ReasoningTrajectory,
    trajectory_similarity,
    population_trajectory_similarity,
    CrossSystemAnalyzer,
    CrossSystemResult,
    SystemPairResult,
    convergence_efficiency,
    EfficiencyResult,
)
from .phi_star import (
    PhiStarExplorer,
    PhiStarResult,
    DistinctionCluster,
    ConvergenceResult,
    Trajectory as PhiStarTrajectory,
    run_convergence,
    jaccard_similarity,
    first_sentence_extractor,
    FRAMING_PREFIXES,
    ALL_FRAMINGS,
)
from .theorems import (
    verify_all as verify_theorems,
    theorem_1_convexity,
    theorem_2_fixed_point,
    theorem_3_gradient_optimality,
    theorem_4_corner_optimality,
    theorem_5_convergence_monotonicity,
    theorem_6_threshold_optimality,
    theorem_7_contraction_and_local_traps,
    theorem_8_sag_bound,
    TheoremResult,
)

__version__ = "1.25.0"
__all__ = [
    "Suite",
    "RegressionSuite",
    "Firewall",
    "PromptRepair",
    "improve",
    "improve_from_production",
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
    "replay",
    "replay_transcript",
    "load_transcript",
    "ReplayReport",
    "ReplayContradiction",
    "ReplayTurn",
    "commitments_from_analysis",
    "reconcile",
    "ReconciliationReport",
    "CommitmentMatch",
    "cases_from_reconciliation",
    "CommitmentLedger",
    "LedgerEntry",
    "audit_report_html",
    "AdmissibilityEngine",
    "AdmissibilityResult",
    "DomainIndex",
    "ThresholdPolicy",
    "CalibrationStore",
    # observatory
    "Constraint",
    "ConstraintStatus",
    "ConstraintProfile",
    "ConstraintDelta",
    "ConstraintObservatory",
    "ConstraintProfiler",
    # active_oracle
    "GroundTruthSignal",
    "ModelProbeResult",
    "DiscoveryClassification",
    "DiscoveryResult",
    "ActiveOracle",
    # oracle
    "NodeProbe",
    "ConsensusNode",
    "ConsensusTopology",
    "TargetedPerturbation",
    "OracleResult",
    "TopologyOracle",
    "TopologyRegistry",
    # measurement
    "ReasoningDimension",
    "DIMENSIONS",
    "MeasurementLaw",
    "LAWS",
    "ReasoningProfile",
    "MeasurementUncertainty",
    "profile_from_results",
    "compare_profiles",
    # epistemic
    "EpistemicAudit",
    "EpistemicProfile",
    "DisagreementMap",
    "InquiryScaffold",
    # topology
    "ReasoningNode",
    "ReasoningEdge",
    "TopologyPath",
    "FailureTopologyMap",
    "topology_distance",
    "topology_from_phi_star",
    # convergence
    "ReasoningTrajectory",
    "trajectory_similarity",
    "population_trajectory_similarity",
    "CrossSystemAnalyzer",
    "CrossSystemResult",
    "SystemPairResult",
    "convergence_efficiency",
    "EfficiencyResult",
    # phi_star
    "PhiStarExplorer",
    "PhiStarResult",
    "DistinctionCluster",
    "ConvergenceResult",
    "PhiStarTrajectory",
    "run_convergence",
    "jaccard_similarity",
    "first_sentence_extractor",
    "FRAMING_PREFIXES",
    "ALL_FRAMINGS",
    # theorems
    "verify_theorems",
    "theorem_1_convexity",
    "theorem_2_fixed_point",
    "theorem_3_gradient_optimality",
    "theorem_4_corner_optimality",
    "theorem_5_convergence_monotonicity",
    "theorem_6_threshold_optimality",
    "theorem_7_contraction_and_local_traps",
    "theorem_8_sag_bound",
    "TheoremResult",
]
