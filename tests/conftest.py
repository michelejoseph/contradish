"""
Tests that construct a real provider client need its SDK. Without it they would fail with contradish's own
"Install the ... SDK" ImportError, which says nothing about the code under test. Report those as skipped, with the
reason, so a checkout without the optional SDKs still gets a clean run. Any other failure is reported unchanged.
CI installs `.[dev]`, so there these tests run.
"""
import pytest

_SDK_MESSAGES = ("Install the Anthropic SDK", "Install the OpenAI SDK")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and rep.failed and call.excinfo is not None:
        exc = call.excinfo.value
        sdk_missing = (isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None) in ("anthropic", "openai"))
        if sdk_missing or (isinstance(exc, ImportError) and any(m in str(exc) for m in _SDK_MESSAGES)):
            rep.outcome = "skipped"
            rep.longrepr = (str(item.path), item.location[1] or 0,
                            f"Skipped: needs an optional provider SDK ({exc}); pip install -e '.[dev]'")
