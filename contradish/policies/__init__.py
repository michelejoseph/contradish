"""
contradish policy packs: prebuilt domain-specific test suites.

Lets a developer get meaningful CAI results in under 2 minutes
without writing a single test case.

Usage:
    from contradish.policies import load_policy, list_policies

    pack = load_policy("ecommerce")
    print(pack.display_name)    # "E-Commerce Support"
    print(len(pack))            # 12

    # Or via Suite directly:
    suite = Suite.from_policy("ecommerce", app=my_app)
    report = suite.run()

    # Or via CLI:
    # contradish --policy ecommerce --app mymodule:my_app

Available packs (20 domains):

  v1 domains:
    ecommerce           -- refunds, pricing, shipping, returns, warranties
    hr                  -- PTO, benefits, termination, leave
    healthcare          -- coverage, referrals, deductibles, eligibility
    legal               -- disclaimers, liability, professional advice boundaries
    finance             -- banking, lending, credit, account rules
    saas                -- subscriptions, billing, data, cancellation
    insurance           -- claims, premiums, coverage, exclusions
    education           -- enrollment, financial aid, grading, academic policy
    ai_safety           -- refusal stability, disclaimer consistency, identity pressure, escalation resistance

  v2 new domains:
    travel              -- airlines, hotels, rental cars, booking and cancellation policies
    mental_health       -- crisis response, therapeutic boundaries, referral thresholds
    government          -- benefits eligibility, document requirements, appeals
    automotive          -- warranties, recalls, service disputes, financing
    real_estate         -- purchase contracts, deposits, disclosure obligations
    medication          -- prescription rules, dosage guidance, refill policies
    telecommunications  -- mobile and internet contracts, billing, cancellation
    employment_disputes -- workplace rights, wrongful termination, wage claims
    immigration         -- visa applications, work authorization, green card
    food_delivery       -- missing items, refunds, driver disputes, subscriptions
    financial_planning  -- retirement, investment guidance, debt management
"""

from ._base import PolicyPack
from .ecommerce import ECOMMERCE
from .hr import HR
from .healthcare import HEALTHCARE
from .legal import LEGAL
from .finance import FINANCE
from .saas import SAAS
from .insurance import INSURANCE
from .education import EDUCATION
from .ai_safety import AI_SAFETY
from .travel import TRAVEL
from .mental_health import MENTAL_HEALTH
from .government import GOVERNMENT
from .automotive import AUTOMOTIVE
from .real_estate import REAL_ESTATE
from .medication import MEDICATION
from .telecommunications import TELECOMMUNICATIONS
from .employment_disputes import EMPLOYMENT_DISPUTES
from .immigration import IMMIGRATION
from .food_delivery import FOOD_DELIVERY
from .financial_planning import FINANCIAL_PLANNING


_REGISTRY: dict[str, PolicyPack] = {
    # v1 domains
    "ecommerce":            ECOMMERCE,
    "hr":                   HR,
    "healthcare":           HEALTHCARE,
    "legal":                LEGAL,
    "finance":              FINANCE,
    "saas":                 SAAS,
    "insurance":            INSURANCE,
    "education":            EDUCATION,
    "ai_safety":            AI_SAFETY,
    # v2 new domains
    "travel":               TRAVEL,
    "mental_health":        MENTAL_HEALTH,
    "government":           GOVERNMENT,
    "automotive":           AUTOMOTIVE,
    "real_estate":          REAL_ESTATE,
    "medication":           MEDICATION,
    "telecommunications":   TELECOMMUNICATIONS,
    "employment_disputes":  EMPLOYMENT_DISPUTES,
    "immigration":          IMMIGRATION,
    "food_delivery":        FOOD_DELIVERY,
    "financial_planning":   FINANCIAL_PLANNING,
}


def list_policies() -> list[str]:
    """
    Return the names of all available policy packs.

    Example:
        >>> list_policies()
        ['ecommerce', 'hr', 'healthcare', ..., 'financial_planning']
    """
    return list(_REGISTRY.keys())


def load_policy(name: str) -> PolicyPack:
    """
    Load a policy pack by name.

    Args:
        name: Any of the 20 available domain names.
              Call list_policies() to see all options.

    Returns:
        PolicyPack with .cases list of TestCase objects.

    Raises:
        ValueError: If the policy name is not recognized.

    Example:
        pack = load_policy("ecommerce")
        suite = Suite(app=my_app)
        for tc in pack.cases:
            suite.add(tc)
        suite.run()
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(f'"{k}"' for k in _REGISTRY)
        raise ValueError(
            f"Unknown policy pack: {name!r}. "
            f"Available: {available}. "
            f"Or call list_policies() to see all options."
        )
    return _REGISTRY[key]


__all__ = [
    "PolicyPack",
    "load_policy",
    "list_policies",
    "ECOMMERCE",
    "HR",
    "HEALTHCARE",
    "LEGAL",
    "FINANCE",
    "SAAS",
    "INSURANCE",
    "EDUCATION",
    "AI_SAFETY",
    "TRAVEL",
    "MENTAL_HEALTH",
    "GOVERNMENT",
    "AUTOMOTIVE",
    "REAL_ESTATE",
    "MEDICATION",
    "TELECOMMUNICATIONS",
    "EMPLOYMENT_DISPUTES",
    "IMMIGRATION",
    "FOOD_DELIVERY",
    "FINANCIAL_PLANNING",
]
