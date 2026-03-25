"""
contradish policy packs — prebuilt domain-specific test suites.

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

Available packs:
    ecommerce   — refunds, pricing, shipping, returns, warranties
    hr          — PTO, benefits, termination, leave
    healthcare  — coverage, referrals, deductibles, eligibility
    legal       — disclaimers, liability, professional advice boundaries
"""

from ._base import PolicyPack
from .ecommerce import ECOMMERCE
from .hr import HR
from .healthcare import HEALTHCARE
from .legal import LEGAL


_REGISTRY: dict[str, PolicyPack] = {
    "ecommerce":  ECOMMERCE,
    "hr":         HR,
    "healthcare": HEALTHCARE,
    "legal":      LEGAL,
}


def list_policies() -> list[str]:
    """
    Return the names of all available policy packs.

    Example:
        >>> list_policies()
        ['ecommerce', 'hr', 'healthcare', 'legal']
    """
    return list(_REGISTRY.keys())


def load_policy(name: str) -> PolicyPack:
    """
    Load a policy pack by name.

    Args:
        name: One of 'ecommerce', 'hr', 'healthcare', 'legal'.

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
]
