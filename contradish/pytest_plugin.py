"""
contradish pytest plugin.

Registers fixtures so devs can write CAI assertions directly in pytest:

    def test_consistency(cai_suite):
        report = cai_suite.run()
        assert report.cai_score >= 0.80, report.summary()

    def test_no_failures(cai_suite):
        report = cai_suite.run()
        assert report.failure_count == 0, report.failures_summary()

Config is read from .contradish.yaml in the project root, or set per-test
via the `contradish_config` fixture.

Example .contradish.yaml:

    policy: ecommerce
    app: mymodule:my_app
    threshold: 0.80
    paraphrases: 5

Or configure inline:

    @pytest.fixture
    def contradish_config():
        return {"policy": "ecommerce", "threshold": 0.80}
"""

from __future__ import annotations

import os
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml_config(path: str = ".contradish.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


def _load_app(app_path: str):
    import sys
    import importlib
    if ":" not in app_path:
        raise ValueError(
            f"contradish: app must be 'module:function', got {app_path!r}"
        )
    module_str, func_str = app_path.rsplit(":", 1)
    sys.path.insert(0, os.getcwd())
    module = importlib.import_module(module_str)
    return getattr(module, func_str)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def contradish_config() -> dict[str, Any]:
    """
    Session-scoped config fixture. Override in your conftest.py to set
    policy, app, threshold, and paraphrases without a config file.

    Example conftest.py:

        @pytest.fixture(scope="session")
        def contradish_config():
            return {
                "policy": "ecommerce",
                "app": "mymodule:my_app",
                "threshold": 0.80,
                "paraphrases": 3,
            }
    """
    return _load_yaml_config()


@pytest.fixture
def cai_suite(contradish_config):
    """
    Returns a configured Suite ready to run.

    Usage:

        def test_cai_score(cai_suite):
            report = cai_suite.run()
            assert report.cai_score >= 0.80

        def test_no_failures(cai_suite):
            report = cai_suite.run()
            assert report.failure_count == 0, report.failures_summary()
    """
    from contradish import Suite

    cfg        = contradish_config
    policy     = cfg.get("policy")
    app_path   = cfg.get("app")
    paraphrases = int(cfg.get("paraphrases", 5))

    app = _load_app(app_path) if app_path else None

    if policy:
        suite = Suite.from_policy(policy=policy, app=app, verbose=False)
    elif app:
        suite = Suite(app=app)
    else:
        raise pytest.UsageError(
            "contradish: set 'policy' or 'app' in .contradish.yaml "
            "or override the `contradish_config` fixture."
        )

    suite._pytest_paraphrases = paraphrases
    return suite


@pytest.fixture
def cai_report(cai_suite):
    """
    Convenience fixture that runs the suite and returns the report directly.

    Usage:

        def test_cai(cai_report):
            assert cai_report.cai_score >= 0.80
    """
    paraphrases = getattr(cai_suite, "_pytest_paraphrases", 5)
    return cai_suite.run(paraphrases=paraphrases, verbose=False)


@pytest.fixture
def cai_threshold(contradish_config) -> float:
    """Returns the configured CAI threshold (default: 0.80)."""
    return float(contradish_config.get("threshold", 0.80))
