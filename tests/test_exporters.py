"""
Full test coverage for contradish/exporters.py.

exporters.py pushes a contradish Report into Langfuse (to_langfuse) or Arize
Phoenix (to_phoenix). Neither SDK is installed in this test environment, so:

  - to_langfuse takes an already-constructed `client` argument (no import of
    the langfuse package happens inside the function at all -- it only does
    an attribute probe on whatever object is handed in), so it is fully
    testable with a small fake client, exactly like FakeClient in
    test_adapters.py.

  - to_phoenix does `import phoenix as px` lazily inside the function body.
    To exercise both the ImportError-when-missing path and the success path
    without installing arize-phoenix, we follow the pattern already used in
    tests/test_caches.py for the optional `redis` dependency:
    `monkeypatch.setitem(sys.modules, "phoenix", <None-or-fake-module>)`.

These tests build TestCase/ContradictionPair/TestResult/Report fixtures
directly (real dataclasses, no mocks of contradish's own types) and fake only
the third-party client/module boundary.
"""
import sys

import pytest

from contradish.models import ContradictionPair, Report, TestCase, TestResult
from contradish.exporters import to_langfuse, to_phoenix


# ── Fixture builders (matches the house style in tests/test_findings.py) ───

def _tc(input="a test question", name=None, **kw):
    return TestCase(input=input, name=name, **kw)


def _cp(input_a="in a", input_b="in b", output_a="out a", output_b="out b",
        severity="policy", explanation="they disagree"):
    return ContradictionPair(
        input_a=input_a, input_b=input_b,
        output_a=output_a, output_b=output_b,
        explanation=explanation, severity=severity,
    )


def _tr(tc=None, consistency_score=0.9, contradictions=None,
        unstable_patterns=None, suggestion=None, **kw):
    return TestResult(
        test_case=tc or _tc(),
        paraphrases=["p1", "p2"],
        outputs=["o0", "o1", "o2"],
        consistency_score=consistency_score,
        contradiction_score=0.0,
        contradictions=contradictions or [],
        unstable_patterns=unstable_patterns or [],
        suggestion=suggestion,
        **kw,
    )


def _failing(tc=None, contradictions=None, unstable_patterns=None, suggestion=None, **kw):
    """A TestResult that lands in report.failed (consistency below threshold)."""
    return _tr(tc=tc, consistency_score=0.3, contradictions=contradictions,
               unstable_patterns=unstable_patterns, suggestion=suggestion, **kw)


def _passing(tc=None, **kw):
    """A TestResult that lands in report.passed."""
    return _tr(tc=tc, consistency_score=0.95, **kw)


def _report(results, thresholds=None):
    return Report(results=results, thresholds=thresholds or {})


# ── Fake langfuse client ─────────────────────────────────────────────────────

class _FakeLangfuseClient:
    """Stands in for an authenticated langfuse.Langfuse() instance."""

    def __init__(self, dataset_exists=True):
        self.dataset_exists = dataset_exists
        self.get_dataset_calls = []
        self.created_datasets = []
        self.items = []

    def get_dataset(self, name):
        self.get_dataset_calls.append(name)
        if not self.dataset_exists:
            raise Exception("dataset not found")
        return {"name": name}

    def create_dataset(self, name):
        self.created_datasets.append(name)

    def create_dataset_item(self, **kwargs):
        self.items.append(kwargs)


# ── to_langfuse: client validation ──────────────────────────────────────────

def test_to_langfuse_rejects_client_without_get_dataset():
    class NotALangfuseClient:
        pass

    with pytest.raises(TypeError, match="langfuse.Langfuse"):
        to_langfuse(_report([]), NotALangfuseClient())


def test_to_langfuse_error_message_mentions_pip_install():
    with pytest.raises(TypeError, match="pip install langfuse"):
        to_langfuse(_report([]), object())


# ── to_langfuse: dataset creation ───────────────────────────────────────────

def test_to_langfuse_does_not_create_dataset_when_it_already_exists():
    client = _FakeLangfuseClient(dataset_exists=True)
    to_langfuse(_report([]), client, dataset_name="my-set")
    assert client.get_dataset_calls == ["my-set"]
    assert client.created_datasets == []


def test_to_langfuse_creates_dataset_when_missing():
    client = _FakeLangfuseClient(dataset_exists=False)
    to_langfuse(_report([]), client, dataset_name="new-set")
    assert client.created_datasets == ["new-set"]


# ── to_langfuse: empty report ────────────────────────────────────────────────

def test_to_langfuse_empty_report_creates_nothing_but_returns_shape():
    client = _FakeLangfuseClient()
    result = to_langfuse(_report([]), client)
    assert client.items == []
    assert result == {
        "dataset_name": "contradish-cai",
        "items_created": 0,
        "failures_exported": 0,
        "passing_exported": 0,
    }


# ── to_langfuse: failures with contradiction pairs ──────────────────────────

def test_to_langfuse_exports_one_item_per_contradiction_pair():
    tc = _tc(input="q1", name="rule-1")
    pair1 = _cp(input_a="a1", input_b="b1", severity="factual")
    pair2 = _cp(input_a="a2", input_b="b2", severity="logical")
    result = _failing(tc=tc, contradictions=[pair1, pair2],
                       unstable_patterns=["flip-flops"], suggestion="be firmer")

    client = _FakeLangfuseClient()
    out = to_langfuse(_report([result]), client, include_passing=False)

    assert len(client.items) == 2
    assert out["failures_exported"] == 2
    assert out["passing_exported"] == 0
    assert out["items_created"] == 2

    item0 = client.items[0]
    assert item0["input"] == {"input_a": "a1", "input_b": "b1"}
    assert item0["expected_output"] == {"should_be_consistent": True}
    meta0 = item0["metadata"]
    assert meta0["rule"] == "rule-1"
    assert meta0["cai_score"] == 0.3
    assert meta0["severity"] == "factual"
    assert meta0["unstable_patterns"] == ["flip-flops"]
    assert meta0["suggested_fix"] == "be firmer"
    assert meta0["passing"] is False

    meta1 = client.items[1]["metadata"]
    assert meta1["severity"] == "logical"


def test_to_langfuse_failing_result_without_contradictions_still_exported():
    tc = _tc(input="the raw input text", name="rule-2")
    result = _failing(tc=tc, contradictions=[], unstable_patterns=["x"], suggestion="fix it")

    client = _FakeLangfuseClient()
    out = to_langfuse(_report([result]), client, include_passing=False)

    assert len(client.items) == 1
    assert out["failures_exported"] == 1
    item = client.items[0]
    assert item["input"] == {"input": "the raw input text"}
    # No contradiction pair on this branch, so no "severity" key and no
    # expected_output key at all.
    assert "severity" not in item["metadata"]
    assert "expected_output" not in item
    assert item["metadata"]["rule"] == "rule-2"
    assert item["metadata"]["passing"] is False


# ── to_langfuse: passing results / include_passing ──────────────────────────

def test_to_langfuse_include_passing_true_by_default():
    tc = _tc(input="ok question", name="rule-3")
    result = _passing(tc=tc)

    client = _FakeLangfuseClient()
    out = to_langfuse(_report([result]), client)

    assert len(client.items) == 1
    assert out["passing_exported"] == 1
    assert out["failures_exported"] == 0
    item = client.items[0]
    assert item["input"] == {"input": "ok question"}
    assert item["metadata"] == {
        "rule": "rule-3", "cai_score": 0.95, "passing": True,
    }
    assert "expected_output" not in item


def test_to_langfuse_include_passing_false_skips_passing_results():
    result = _passing()
    client = _FakeLangfuseClient()
    out = to_langfuse(_report([result]), client, include_passing=False)

    assert client.items == []
    assert out["passing_exported"] == 0
    assert out["items_created"] == 0


# ── to_langfuse: metadata merge ─────────────────────────────────────────────

def test_to_langfuse_metadata_param_merges_and_can_override_builtin_keys():
    result = _passing(tc=_tc(name="rule-4"))
    client = _FakeLangfuseClient()
    to_langfuse(_report([result]), client, metadata={"env": "prod", "passing": "OVERRIDDEN"})

    meta = client.items[0]["metadata"]
    assert meta["env"] == "prod"
    # extra_meta is splatted last, so it wins over the built-in "passing" key --
    # documenting actual (perhaps surprising) behavior rather than assuming it.
    assert meta["passing"] == "OVERRIDDEN"


# ── to_langfuse: mixed report end-to-end ────────────────────────────────────

def test_to_langfuse_mixed_report_counts_and_totals():
    failing = _failing(tc=_tc(name="f1"), contradictions=[_cp()])
    passing = _passing(tc=_tc(name="p1"))
    client = _FakeLangfuseClient()
    out = to_langfuse(_report([failing, passing]), client)

    assert out["failures_exported"] == 1
    assert out["passing_exported"] == 1
    assert out["items_created"] == 2
    assert out["dataset_name"] == "contradish-cai"


# ── to_phoenix: ImportError when arize-phoenix is not installed ────────────

def test_to_phoenix_raises_import_error_when_phoenix_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "phoenix", None)
    with pytest.raises(ImportError, match="arize-phoenix is not installed"):
        to_phoenix(_report([]))


def test_to_phoenix_import_error_mentions_pip_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "phoenix", None)
    with pytest.raises(ImportError, match="pip install arize-phoenix"):
        to_phoenix(_report([]))


# ── Fake phoenix module/client ───────────────────────────────────────────────

class _FakePhoenixClient:
    def __init__(self):
        self.upload_calls = []

    def upload_dataset(self, dataset_name, inputs, outputs, metadata):
        self.upload_calls.append({
            "dataset_name": dataset_name,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata,
        })
        return {"id": "ds-1", "name": dataset_name, "n": len(inputs)}


def _install_fake_phoenix(monkeypatch, default_client=None):
    """Fakes `import phoenix as px` so px.Client() returns a known instance."""
    default_client = default_client or _FakePhoenixClient()
    fake_module = type(sys)("phoenix")
    fake_module.Client = lambda: default_client
    monkeypatch.setitem(sys.modules, "phoenix", fake_module)
    return default_client


# ── to_phoenix: success path, empty report ──────────────────────────────────

def test_to_phoenix_empty_report_uploads_empty_lists(monkeypatch):
    fake_client = _install_fake_phoenix(monkeypatch)
    out = to_phoenix(_report([]))

    assert len(fake_client.upload_calls) == 1
    call = fake_client.upload_calls[0]
    assert call["inputs"] == []
    assert call["outputs"] == []
    assert call["metadata"] == []
    assert out["items_created"] == 0
    assert out["failures_exported"] == 0
    assert out["passing_exported"] == 0
    assert out["dataset"] == {"id": "ds-1", "name": "contradish-cai", "n": 0}


# ── to_phoenix: default client vs explicit client ───────────────────────────

def test_to_phoenix_uses_px_client_when_none_given(monkeypatch):
    fake_client = _install_fake_phoenix(monkeypatch)
    to_phoenix(_report([]))
    assert len(fake_client.upload_calls) == 1


def test_to_phoenix_uses_explicit_client_when_given(monkeypatch):
    default_client = _install_fake_phoenix(monkeypatch)
    explicit_client = _FakePhoenixClient()

    to_phoenix(_report([]), client=explicit_client)

    assert len(explicit_client.upload_calls) == 1
    assert default_client.upload_calls == []


# ── to_phoenix: failures with contradiction pairs ───────────────────────────

def test_to_phoenix_exports_one_example_per_contradiction_pair(monkeypatch):
    fake_client = _install_fake_phoenix(monkeypatch)
    tc = _tc(input="q1", name="rule-1")
    pair = _cp(input_a="a1", input_b="b1", output_a="oa", output_b="ob", severity="factual",
               explanation="mismatch")
    result = _failing(tc=tc, contradictions=[pair], unstable_patterns=["flip"], suggestion="fix")

    out = to_phoenix(_report([result]), include_passing=False)

    call = fake_client.upload_calls[0]
    assert call["inputs"] == [{"input_a": "a1", "input_b": "b1"}]
    assert call["outputs"] == [{"output_a": "oa", "output_b": "ob"}]
    meta = call["metadata"][0]
    assert meta["rule"] == "rule-1"
    assert meta["severity"] == "factual"
    assert meta["explanation"] == "mismatch"
    assert meta["unstable_patterns"] == ["flip"]
    assert meta["suggested_fix"] == "fix"
    assert meta["passing"] is False
    assert out["failures_exported"] == 1
    assert out["items_created"] == 1


def test_to_phoenix_failing_result_without_contradictions(monkeypatch):
    fake_client = _install_fake_phoenix(monkeypatch)
    tc = _tc(input="raw text", name="rule-2")
    result = _failing(tc=tc, contradictions=[], unstable_patterns=["x"], suggestion="s")

    to_phoenix(_report([result]), include_passing=False)

    call = fake_client.upload_calls[0]
    assert call["inputs"] == [{"input": "raw text"}]
    assert call["outputs"] == [{}]
    meta = call["metadata"][0]
    assert "severity" not in meta
    assert "explanation" not in meta
    assert meta["rule"] == "rule-2"
    assert meta["passing"] is False


# ── to_phoenix: passing results / include_passing ───────────────────────────

def test_to_phoenix_include_passing_true_by_default(monkeypatch):
    fake_client = _install_fake_phoenix(monkeypatch)
    result = _passing(tc=_tc(input="ok", name="rule-3"))

    out = to_phoenix(_report([result]))

    call = fake_client.upload_calls[0]
    assert call["inputs"] == [{"input": "ok"}]
    assert call["outputs"] == [{}]
    assert call["metadata"][0] == {"rule": "rule-3", "cai_score": 0.95, "passing": True}
    assert out["passing_exported"] == 1
    assert out["failures_exported"] == 0


def test_to_phoenix_include_passing_false_skips_passing_results(monkeypatch):
    fake_client = _install_fake_phoenix(monkeypatch)
    result = _passing()

    out = to_phoenix(_report([result]), include_passing=False)

    call = fake_client.upload_calls[0]
    assert call["inputs"] == []
    assert out["passing_exported"] == 0
    assert out["items_created"] == 0


# ── to_phoenix: metadata merge ───────────────────────────────────────────────

def test_to_phoenix_metadata_param_merges_and_can_override_builtin_keys(monkeypatch):
    fake_client = _install_fake_phoenix(monkeypatch)
    result = _passing(tc=_tc(name="rule-4"))

    to_phoenix(_report([result]), metadata={"env": "prod", "passing": "OVERRIDDEN"})

    meta = fake_client.upload_calls[0]["metadata"][0]
    assert meta["env"] == "prod"
    # extra_meta is splatted last, so it wins over the built-in "passing" key --
    # same override behavior as to_langfuse, documented rather than assumed.
    assert meta["passing"] == "OVERRIDDEN"


# ── to_phoenix: mixed report, failures_exported/passing_exported counting ──

def test_to_phoenix_mixed_report_counts_derived_from_examples(monkeypatch):
    fake_client = _install_fake_phoenix(monkeypatch)
    failing = _failing(tc=_tc(name="f1"), contradictions=[_cp(), _cp()])
    passing = _passing(tc=_tc(name="p1"))

    out = to_phoenix(_report([failing, passing]))

    assert out["items_created"] == 3
    assert out["failures_exported"] == 2
    assert out["passing_exported"] == 1
    assert out["dataset_name"] == "contradish-cai"
    assert out["dataset"]["n"] == 3


def test_to_phoenix_dataset_name_is_passed_through(monkeypatch):
    fake_client = _install_fake_phoenix(monkeypatch)
    to_phoenix(_report([]), dataset_name="custom-set")
    assert fake_client.upload_calls[0]["dataset_name"] == "custom-set"
    out = to_phoenix(_report([]), dataset_name="custom-set")
    assert out["dataset_name"] == "custom-set"
