"""
contradish: CAI Strain testing for LLM applications.

Detects CAI failures: when your app gives contradictory answers to semantically
equivalent inputs. ML literature calls this drift; contradish names and scores it.
Returns CAI Strain per rule (0-1, lower = more consistent).

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

The legacy `report.cai_score` attribute (0-1, higher is better) is preserved
for backward compatibility; cai_score == 1 - cai_strain.

A note on scope: everything above is covered by tests, documented in the
README, and reachable from a CLI command or the core Suite/Judge/benchmark
path. contradish also ships a second, larger tier of research modules
(topology/oracle/theorem-proving/reasoning-measurement machinery) that
predates that discipline being applied consistently. Those symbols are
still fully importable -- nothing is deleted or broken -- but they're
lazily loaded and raise a FutureWarning on first use so you know which
tier of guarantee you're getting. See the "_EXPERIMENTAL_SYMBOLS" comment
near the bottom of this file for the full list and the reasoning.

Full docs: https://contradish.com
"""

import importlib as _importlib
import warnings as _warnings

# _improve.py / _reconcile.py / _replay.py are underscore-prefixed on disk
# (unlike every other submodule here) because each one's main function is
# re-exported under the exact same name as its own file -- `improve()` from
# improve.py, `reconcile()` from reconcile.py, `replay()` from replay.py.
# `from .improve import improve` would bind `contradish.improve` to the
# *function*, silently clobbering the submodule reference the import system
# had just set on the `contradish` package object as a side effect of
# importing it -- so `contradish.improve.improve(...)` (attribute-chain
# access into the submodule) would raise AttributeError, unpredictably,
# depending on import order. There's no way to have one attribute slot be
# both things at once. Renaming the files sidesteps the collision entirely:
# `contradish.improve` is unambiguously the function (the documented,
# tested usage -- see README), and the submodule, if you need it directly,
# is `contradish._improve` (internal; `from contradish.improve import X`
# does NOT work now that there is no contradish/improve.py -- use
# `from contradish import improve` instead, or `contradish._improve` if you
# specifically need submodule internals not exposed at the top level).
from .suite        import Suite
from .regression   import RegressionSuite
from .firewall     import Firewall
from .repair       import PromptRepair
from ._improve     import improve, improve_from_production, ImprovementResult
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
from .prompt_analyzer import (
    analyze_prompt,
    PromptAnalysis,
    PromptTension,
    commitments_from_analysis,
)
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
from ._replay      import (
    replay,
    replay_transcript,
    load_transcript,
    ReplayReport,
    ReplayContradiction,
    ReplayTurn,
)
from ._reconcile   import (
    reconcile,
    ReconciliationReport,
    CommitmentMatch,
    cases_from_reconciliation,
)
from .ledger       import CommitmentLedger, LedgerEntry
from .report       import audit_report_html

# quickstart.analyze() backs the real `contradish analyze` CLI command, so
# its direct dependencies stay eager: domains.py (question sets) and
# conviction.py + cdr.py (the Consistency Diagnostic Report pipeline --
# used for real client-facing audits, not just a demo). Note that
# quickstart.analyze() itself also reaches into phi_star.py and
# residual_truth.py internally; those two are listed under
# _EXPERIMENTAL_SYMBOLS below like the rest of that cluster, since they
# carry the same lack of tests and docs even though one CLI path depends
# on them -- see the comment down there for the full disclosure.
from .quickstart   import analyze, QuickResult
from .domains      import (
    DomainPack,
    get_domain,
    list_domains,
    domain_questions,
    CUSTOMER_SERVICE,
    MEDICAL,
    LEGAL,
    FINANCIAL,
    SAFETY,
    HR,
)
from .conviction   import ConvictionProfiler, ConvictionReport, ConvictionResult
from .cdr          import generate_cdr

__version__ = "1.29.0"
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
    # quickstart / analyze
    "analyze",
    "QuickResult",
    # domains
    "DomainPack",
    "get_domain",
    "list_domains",
    "domain_questions",
    "CUSTOMER_SERVICE",
    "MEDICAL",
    "LEGAL",
    "FINANCIAL",
    "SAFETY",
    "HR",
    # conviction / cdr (Consistency Diagnostic Report pipeline)
    "ConvictionProfiler",
    "ConvictionReport",
    "ConvictionResult",
    "generate_cdr",
]

# ─────────────────────────────────────────────────────────────────────────
# Experimental research cluster.
#
# Everything named below is real, shipped code -- but as of this writing,
# none of it has any test coverage, none of it is documented in the
# README, and (with the sole exception of phi_star/residual_truth, which
# `contradish analyze` reaches into internally, undocumented, for its own
# implementation) none of it is wired into any CLI command or into the
# core Suite/Judge/benchmark path that the rest of this package holds
# itself to a testing bar for. theorems.py in particular claims to
# "prove" eight formal theorems about an admissibility space that no
# other module in the package actually consults.
#
# Nothing here is deleted -- every symbol below remains fully importable,
# `from contradish import X` still works exactly as before. Two things
# changed: (1) these modules are no longer imported eagerly at
# `import contradish` time, so the CI-gate path doesn't pay the cost of
# loading ~9,500 lines of unverified code it never touches; (2) accessing
# any of these names raises a FutureWarning once, telling you which tier
# of guarantee you're getting, so it's an informed choice rather than a
# silent one. Treat this tier as a research preview: the API may change
# or be removed without a major-version bump until it earns the same
# tests + docs bar as everything in __all__ above.
_EXPERIMENTAL_SYMBOLS = {
    # admissibility.py
    "AdmissibilityEngine":  ("admissibility", "AdmissibilityEngine"),
    "AdmissibilityResult":  ("admissibility", "AdmissibilityResult"),
    "DomainIndex":          ("admissibility", "DomainIndex"),
    "ThresholdPolicy":      ("admissibility", "ThresholdPolicy"),
    "CalibrationStore":     ("admissibility", "CalibrationStore"),
    # measurement.py
    "ReasoningDimension":     ("measurement", "ReasoningDimension"),
    "DIMENSIONS":             ("measurement", "DIMENSIONS"),
    "MeasurementLaw":         ("measurement", "MeasurementLaw"),
    "LAWS":                   ("measurement", "LAWS"),
    "ReasoningProfile":       ("measurement", "ReasoningProfile"),
    "MeasurementUncertainty": ("measurement", "MeasurementUncertainty"),
    "profile_from_results":   ("measurement", "profile_from_results"),
    "compare_profiles":       ("measurement", "compare"),
    # epistemic.py
    "EpistemicAudit":   ("epistemic", "EpistemicAudit"),
    "EpistemicProfile": ("epistemic", "EpistemicProfile"),
    "DisagreementMap":  ("epistemic", "DisagreementMap"),
    "InquiryScaffold":  ("epistemic", "InquiryScaffold"),
    # observatory.py
    "Constraint":            ("observatory", "Constraint"),
    "ConstraintStatus":      ("observatory", "ConstraintStatus"),
    "ConstraintProfile":     ("observatory", "ConstraintProfile"),
    "ConstraintDelta":       ("observatory", "ConstraintDelta"),
    "ConstraintObservatory": ("observatory", "ConstraintObservatory"),
    "ConstraintProfiler":    ("observatory", "ConstraintProfiler"),
    # topology_training.py
    "TrainingExample":     ("topology_training", "TrainingExample"),
    "TrainingCurriculum":  ("topology_training", "TrainingCurriculum"),
    "TopologyTrainer":     ("topology_training", "TopologyTrainer"),
    "generate_curriculum": ("topology_training", "generate_curriculum"),
    # residual_truth.py -- reached internally by quickstart.analyze(); see
    # the module-level comment above for why it's still listed here.
    "Claim":                          ("residual_truth", "Claim"),
    "IncompatibilityEdge":            ("residual_truth", "IncompatibilityEdge"),
    "RepairStep":                     ("residual_truth", "RepairStep"),
    "RepairTrace":                    ("residual_truth", "RepairTrace"),
    "ResidualTruthResult":            ("residual_truth", "ResidualTruthResult"),
    "KeywordClaimExtractor":          ("residual_truth", "KeywordClaimExtractor"),
    "PatternIncompatibilityDetector": ("residual_truth", "PatternIncompatibilityDetector"),
    "ResidualTruthEngine":            ("residual_truth", "ResidualTruthEngine"),
    # distinction.py
    "DistinctionPair":        ("distinction", "DistinctionPair"),
    "DistinctionMeasurement":  ("distinction", "DistinctionMeasurement"),
    "DistinctionProfile":     ("distinction", "DistinctionProfile"),
    "DistinctionLossMap":     ("distinction", "DistinctionLossMap"),
    "DistinctionProber":      ("distinction", "DistinctionProber"),
    # surrender.py
    "SurrenderSample":    ("surrender", "SurrenderSample"),
    "SurrenderPoint":     ("surrender", "SurrenderPoint"),
    "SurrenderCurve":     ("surrender", "SurrenderCurve"),
    "SurrenderProfiler":  ("surrender", "SurrenderProfiler"),
    "SurrenderAtlas":     ("surrender", "SurrenderAtlas"),
    "profile_surrender":  ("surrender", "profile_constraints"),
    "PRESSURE_LEVELS":    ("surrender", "PRESSURE_LEVELS"),
    "ALL_PRESSURE_TYPES": ("surrender", "ALL_PRESSURE_TYPES"),
    # cognitive_topology.py
    "ReliabilityPoint":          ("cognitive_topology", "ReliabilityPoint"),
    "ReliabilityGradient":       ("cognitive_topology", "ReliabilityGradient"),
    "IntegrationFinding":        ("cognitive_topology", "IntegrationFinding"),
    "CognitiveTopologyReport":   ("cognitive_topology", "CognitiveTopologyReport"),
    "CognitiveTopologyProfiler": ("cognitive_topology", "CognitiveTopologyProfiler"),
    # behavioral_drift.py
    "Interaction":                 ("behavioral_drift", "Interaction"),
    "SessionLog":                  ("behavioral_drift", "SessionLog"),
    "DriftSignal":                 ("behavioral_drift", "DriftSignal"),
    "ImplicitCorrectionCandidate": ("behavioral_drift", "ImplicitCorrectionCandidate"),
    "BehavioralDriftReport":       ("behavioral_drift", "BehavioralDriftReport"),
    "BehavioralDriftDetector":     ("behavioral_drift", "BehavioralDriftDetector"),
    # structural_eval.py
    "JunctionSensitivity":        ("structural_eval", "JunctionSensitivity"),
    "SensitivityProfile":         ("structural_eval", "SensitivityProfile"),
    "StructuralDelta":            ("structural_eval", "StructuralDelta"),
    "StructuralEvaluationReport": ("structural_eval", "StructuralEvaluationReport"),
    "StructuralEvaluator":        ("structural_eval", "StructuralEvaluator"),
    # active_oracle.py
    "GroundTruthSignal":       ("active_oracle", "GroundTruthSignal"),
    "ModelProbeResult":        ("active_oracle", "ModelProbeResult"),
    "DiscoveryClassification": ("active_oracle", "DiscoveryClassification"),
    "DiscoveryResult":         ("active_oracle", "DiscoveryResult"),
    "ActiveOracle":            ("active_oracle", "ActiveOracle"),
    # oracle.py
    "NodeProbe":            ("oracle", "NodeProbe"),
    "ConsensusNode":        ("oracle", "ConsensusNode"),
    "ConsensusTopology":    ("oracle", "ConsensusTopology"),
    "TargetedPerturbation": ("oracle", "TargetedPerturbation"),
    "OracleResult":         ("oracle", "OracleResult"),
    "TopologyOracle":       ("oracle", "TopologyOracle"),
    "TopologyRegistry":     ("oracle", "TopologyRegistry"),
    # topology.py
    "ReasoningNode":          ("topology", "ReasoningNode"),
    "ReasoningEdge":          ("topology", "ReasoningEdge"),
    "TopologyPath":           ("topology", "TopologyPath"),
    "FailureTopologyMap":     ("topology", "FailureTopologyMap"),
    "topology_distance":      ("topology", "topology_distance"),
    "topology_from_phi_star": ("topology", "topology_from_phi_star"),
    # convergence.py
    "ReasoningTrajectory":              ("convergence", "ReasoningTrajectory"),
    "trajectory_similarity":            ("convergence", "trajectory_similarity"),
    "population_trajectory_similarity": ("convergence", "population_trajectory_similarity"),
    "CrossSystemAnalyzer":              ("convergence", "CrossSystemAnalyzer"),
    "CrossSystemResult":                ("convergence", "CrossSystemResult"),
    "SystemPairResult":                 ("convergence", "SystemPairResult"),
    "convergence_efficiency":           ("convergence", "convergence_efficiency"),
    "EfficiencyResult":                 ("convergence", "EfficiencyResult"),
    # phi_star.py -- reached internally by quickstart.analyze(); see the
    # module-level comment above for why it's still listed here.
    "PhiStarExplorer":          ("phi_star", "PhiStarExplorer"),
    "PhiStarResult":            ("phi_star", "PhiStarResult"),
    "DistinctionCluster":       ("phi_star", "DistinctionCluster"),
    "ConvergenceResult":        ("phi_star", "ConvergenceResult"),
    "PhiStarTrajectory":        ("phi_star", "Trajectory"),
    "run_convergence":          ("phi_star", "run_convergence"),
    "jaccard_similarity":       ("phi_star", "jaccard_similarity"),
    "first_sentence_extractor": ("phi_star", "first_sentence_extractor"),
    "FRAMING_PREFIXES":         ("phi_star", "FRAMING_PREFIXES"),
    "ALL_FRAMINGS":             ("phi_star", "ALL_FRAMINGS"),
    # theorems.py -- not imported by any other module in the package.
    "verify_theorems":                       ("theorems", "verify_all"),
    "theorem_1_convexity":                   ("theorems", "theorem_1_convexity"),
    "theorem_2_fixed_point":                 ("theorems", "theorem_2_fixed_point"),
    "theorem_3_gradient_optimality":         ("theorems", "theorem_3_gradient_optimality"),
    "theorem_4_corner_optimality":           ("theorems", "theorem_4_corner_optimality"),
    "theorem_5_convergence_monotonicity":    ("theorems", "theorem_5_convergence_monotonicity"),
    "theorem_6_threshold_optimality":        ("theorems", "theorem_6_threshold_optimality"),
    "theorem_7_contraction_and_local_traps": ("theorems", "theorem_7_contraction_and_local_traps"),
    "theorem_8_sag_bound":                   ("theorems", "theorem_8_sag_bound"),
    "TheoremResult":                         ("theorems", "TheoremResult"),
    # session_consistency.py -- not imported by any other module.
    "SessionConsistencyProfiler": ("session_consistency", "SessionConsistencyProfiler"),
    "SessionConsistencyReport":   ("session_consistency", "SessionConsistencyReport"),
    "SessionVarianceResult":      ("session_consistency", "SessionVarianceResult"),
    "SessionSample":              ("session_consistency", "SessionSample"),
    # ground_truth.py -- not imported by any other module.
    "GroundTruth":             ("ground_truth", "GroundTruth"),
    "GroundTruthPack":         ("ground_truth", "GroundTruthPack"),
    "GroundTruthAuditor":      ("ground_truth", "GroundTruthAuditor"),
    "AccuracyResult":          ("ground_truth", "AccuracyResult"),
    "DriftQuality":            ("ground_truth", "DriftQuality"),
    "ReliabilityResult":       ("ground_truth", "ReliabilityResult"),
    "ReliabilityReport":       ("ground_truth", "ReliabilityReport"),
    "load_ground_truth_pack":  ("ground_truth", "load_ground_truth_pack"),
}


def __getattr__(name):
    """PEP 562 module-level lazy attribute access for the experimental tier.

    Keeps every `from contradish import X` call site working unchanged --
    nothing was deleted -- while deferring the import cost and surfacing a
    one-time warning about test/documentation status. See
    _EXPERIMENTAL_SYMBOLS above for what's covered and why.
    """
    entry = _EXPERIMENTAL_SYMBOLS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, source_name = entry
    _warnings.warn(
        f"contradish.{name} is part of an experimental research cluster "
        f"({module_name}.py): no test coverage, not documented in the "
        f"README, and not wired into any CLI command or the core "
        f"Suite/Judge/benchmark path. The API may change or be removed "
        f"without notice.",
        category=FutureWarning,
        stacklevel=2,
    )
    module = _importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, source_name)
    globals()[name] = value  # cache: only the first access pays the cost
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPERIMENTAL_SYMBOLS))
