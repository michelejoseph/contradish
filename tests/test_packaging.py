"""
Release guards. These catch the packaging mistakes that ship a broken wheel to
PyPI even when every functional test passes:

  - version drift between pyproject.toml and contradish.__version__ (the version
    lives in two files; bumping one and forgetting the other ships a mislabeled
    package);
  - an export in __all__ that does not actually resolve (a rename or a dropped
    import surfaces as an ImportError only after users `pip install`);
  - missing benchmark data (the package ships 50+ JSON fixtures; if the build
    ever drops them, `contradish benchmark` breaks for installed users).

Run with: pytest tests/test_packaging.py   (no API key, pure filesystem + import)
"""
import os
import re

import contradish

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_version() -> str:
    with open(os.path.join(_ROOT, "pyproject.toml")) as f:
        text = f.read()
    # Parse the [project] version without a TOML lib (stdlib-only, runs on 3.9).
    project = text.split("[project]", 1)[1]
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    assert m, "could not find version in [project] section of pyproject.toml"
    return m.group(1)


def test_version_in_sync():
    assert contradish.__version__ == _pyproject_version(), (
        f"version drift: contradish.__version__={contradish.__version__!r} "
        f"but pyproject.toml={_pyproject_version()!r}"
    )


def test_all_exports_resolve():
    missing = [name for name in contradish.__all__ if not hasattr(contradish, name)]
    assert not missing, f"__all__ names that do not resolve: {missing}"


def test_new_loop_symbols_exported():
    # The closed-loop surface added in 1.15/1.16 must be importable from the top level.
    for name in ("improve_from_production", "cases_from_reconciliation",
                 "reconcile", "replay"):
        assert name in contradish.__all__ and hasattr(contradish, name), name


def test_benchmark_data_present():
    bench_dir = os.path.join(_ROOT, "contradish", "benchmarks")
    jsons = []
    for root, _dirs, files in os.walk(bench_dir):
        jsons += [f for f in files if f.endswith(".json")]
    assert len(jsons) >= 20, f"expected the benchmark JSON fixtures to ship, found {len(jsons)}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
