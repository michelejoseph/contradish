"""
topology_demo.py — Failure Topology of Reasoning Systems

Demonstrates three things:

  1. A failure topology map for the medication dosage domain.
     Shows the critical path, superspreaders, and structural certification.

  2. Two models with the same answers but different failure topologies.
     Model A: fails at the dosage ceiling junction (easy to fix — one superspreader).
     Model B: fails distributed across many junctions (hard to fix — no superspreader).
     Both produce correct answers 80% of the time. Very different to repair.

  3. Topology distance between the two models.
     dist ≈ 0.0 → same failure structure.
     dist ≈ 1.0 → different failure structures despite similar accuracy.

Usage:
    PYTHONPATH=. python examples/topology_demo.py
"""

from contradish import (
    FailureTopologyMap,
    ReasoningNode,
    ReasoningEdge,
    topology_distance,
)

# ─── Domain: medication dosage ────────────────────────────────────────────────
#
# A reasoning system answering dosage questions must commit to distinctions
# in this order:
#
#   drug_category → dosage_form → [target_population, safety_rationale]
#                                         ↓
#                                   daily_ceiling   ← terminal (the answer)
#
# The daily_ceiling junction is the most load-bearing (λ=0.9).
# Under framing pressure (sympathy, urgency, authority), the system's
# commitment at daily_ceiling drifts — ε_c is high.
# This makes daily_ceiling a superspreader: load-bearing AND fragile.

NODES_A = {
    "drug_category": ReasoningNode(
        "drug_category",
        description="Is this drug OTC or prescription? What class?",
        lambda_weight=0.40,
        cai_strain=0.04,
        reality_strain=0.02,
        domain="medication",
    ),
    "dosage_form": ReasoningNode(
        "dosage_form",
        description="What tablet/capsule strength is standard?",
        lambda_weight=0.50,
        cai_strain=0.06,
        reality_strain=0.03,
        domain="medication",
    ),
    "target_population": ReasoningNode(
        "target_population",
        description="Who is the standard adult population for this dosing?",
        lambda_weight=0.55,
        cai_strain=0.10,
        reality_strain=0.05,
        domain="medication",
    ),
    "safety_rationale": ReasoningNode(
        "safety_rationale",
        description="What is the physiological basis for the ceiling?",
        lambda_weight=0.65,
        cai_strain=0.18,
        reality_strain=0.10,
        domain="medication",
    ),
    "daily_ceiling": ReasoningNode(
        "daily_ceiling",
        description="What is the maximum safe daily dose?",
        lambda_weight=0.92,
        cai_strain=0.51,    # drifts severely under emotional/authority pressure
        reality_strain=0.22,
        domain="medication",
    ),
}

EDGES_A = [
    ReasoningEdge("drug_category",    "dosage_form",       propagation=0.75),
    ReasoningEdge("dosage_form",      "target_population", propagation=0.70),
    ReasoningEdge("dosage_form",      "safety_rationale",  propagation=0.65),
    ReasoningEdge("target_population","daily_ceiling",     propagation=0.85),
    ReasoningEdge("safety_rationale", "daily_ceiling",     propagation=0.80),
]

# ─── Model B: distributed failures, no superspreader ─────────────────────────
#
# Model B gets the same accuracy overall (80%), but its failures are spread
# across all junctions at moderate risk rather than concentrated at one.
# The Gini coefficient of its risk distribution is lower.
# Superficially similar reliability. Structurally much harder to repair.

NODES_B = {
    "drug_category": ReasoningNode(
        "drug_category",
        description="Is this drug OTC or prescription? What class?",
        lambda_weight=0.40,
        cai_strain=0.22,    # moderately fragile (unlike Model A's 0.04)
        reality_strain=0.10,
        domain="medication",
    ),
    "dosage_form": ReasoningNode(
        "dosage_form",
        description="What tablet/capsule strength is standard?",
        lambda_weight=0.50,
        cai_strain=0.28,
        reality_strain=0.14,
        domain="medication",
    ),
    "target_population": ReasoningNode(
        "target_population",
        description="Who is the standard adult population for this dosing?",
        lambda_weight=0.55,
        cai_strain=0.25,
        reality_strain=0.12,
        domain="medication",
    ),
    "safety_rationale": ReasoningNode(
        "safety_rationale",
        description="What is the physiological basis for the ceiling?",
        lambda_weight=0.65,
        cai_strain=0.24,
        reality_strain=0.12,
        domain="medication",
    ),
    "daily_ceiling": ReasoningNode(
        "daily_ceiling",
        description="What is the maximum safe daily dose?",
        lambda_weight=0.92,
        cai_strain=0.20,    # less fragile than Model A at this junction
        reality_strain=0.10,
        domain="medication",
    ),
}

EDGES_B = EDGES_A  # same dependency structure; only node risks differ


if __name__ == "__main__":

    topo_a = FailureTopologyMap(NODES_A, EDGES_A, domain="medication", model="model_A")
    topo_b = FailureTopologyMap(NODES_B, EDGES_B, domain="medication", model="model_B")

    print("\n" + "=" * 60)
    print("MODEL A — failures concentrated at daily_ceiling (superspreader)")
    print("=" * 60)
    print(topo_a.report())

    print("\n" + "=" * 60)
    print("MODEL B — failures distributed across all junctions")
    print("=" * 60)
    print(topo_b.report())

    print("\n" + "=" * 60)
    print("STRUCTURAL CORRESPONDENCE")
    print("=" * 60)
    dist = topology_distance(topo_a, topo_b)
    print()
    print(f"  Topology distance:  {dist:.3f}")
    print()
    if dist < 0.10:
        print("  Structurally equivalent. Same failure structure.")
        print("  Will fail on the same inputs. Interchangeable.")
    elif dist < 0.30:
        print("  Structurally similar. Overlapping failure modes, different weights.")
    else:
        print("  Structurally different. Different failure frameworks.")
        print("  Despite similar accuracy, they fail on different inputs.")
        print("  Require different repairs. Not interchangeable.")

    print()
    print("  Repair comparison:")
    print()
    print(f"  Model A — fix 1 junction (daily_ceiling) to achieve:")
    fixed_a_risk = topo_a.nodes["daily_ceiling"].failure_risk
    print(f"    Reduce failure risk by {fixed_a_risk:.3f} "
          f"({fixed_a_risk / topo_a.expected_terminal_failures():.0%} of total)")

    b_risks = sorted([(nid, n.failure_risk) for nid, n in topo_b.nodes.items()],
                     key=lambda x: -x[1])
    top_b_nid, top_b_risk = b_risks[0]
    print(f"  Model B — fix 1 junction ({top_b_nid}) to achieve:")
    print(f"    Reduce failure risk by {top_b_risk:.3f} "
          f"({top_b_risk / topo_b.expected_terminal_failures():.0%} of total)")

    print()
    print("  Model A's failure topology is heavy-tailed (Gini ="
          f" {topo_a.gini_coefficient:.2f}).")
    print("  One repair fixes most failures.")
    print()
    print(f"  Model B's failure topology is more uniform (Gini ="
          f" {topo_b.gini_coefficient:.2f}).")
    print("  Every junction needs repair. No single fix dominates.")
    print()
    print("  Same accuracy statistic. Completely different repair strategies.")
    print("  Topology distance reveals what accuracy conceals.")
    print()
    print("─" * 60)
