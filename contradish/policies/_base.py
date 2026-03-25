"""
PolicyPack — a named, domain-specific bundle of TestCases.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contradish.models import TestCase


@dataclass
class PolicyPack:
    """
    A prebuilt set of test cases for a specific domain.

    Attributes:
        name:         Machine-readable identifier (e.g. "ecommerce").
        display_name: Human-readable label (e.g. "E-Commerce Support").
        description:  One-line summary of what the pack tests.
        cases:        List of TestCase objects ready to load into a Suite.

    Example:
        from contradish.policies import load_policy

        pack = load_policy("ecommerce")
        suite = Suite.from_policy("ecommerce", app=my_app)
    """
    name:         str
    display_name: str
    description:  str
    cases:        list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def __repr__(self) -> str:
        return (
            f"PolicyPack(name={self.name!r}, "
            f"display_name={self.display_name!r}, "
            f"cases={len(self.cases)})"
        )
