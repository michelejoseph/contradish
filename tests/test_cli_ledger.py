"""
Tests for the CLI surface of the commitment ledger: `contradish ledger
init|show|verify|anchor`, and the wiring that makes `contradish monitor`
append to it automatically. Run with: pytest tests/test_cli_ledger.py

No API key or model call required. cmd_ledger and _record_monitor_to_ledger
only read and write the local ledger JSON file; they never construct a
Judge or LLMClient.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

from contradish.cli import cmd_ledger, _record_monitor_to_ledger
from contradish.ledger import CommitmentLedger


def _run_ledger(action, path, **over):
    """Invoke cmd_ledger with a Namespace shaped like argparse's, capturing
    stdout. Returns (printed_text, system_exit_code_or_None)."""
    ns = argparse.Namespace(action=action, path=path, json=False, label=None,
                             force=False)
    for k, v in over.items():
        setattr(ns, k, v)
    buf = io.StringIO()
    code = None
    try:
        with contextlib.redirect_stdout(buf):
            cmd_ledger(ns)
    except SystemExit as e:
        code = e.code
    return buf.getvalue(), code


_ANALYSIS = {
    "log_path": "prod_logs.jsonl",
    "total_conversations": 42,
    "clusters_scored": 3,
    "drifted_clusters": 1,
    "drift_rate": 0.33,
    "avg_cts": 0.20,
    "hotspots": [
        {"topic": "bereavement leave days", "cts": 0.71,
         "summary": "answers vary on whether a day count is given",
         "example_consistent": "Bereavement leave is 3 paid days.",
         "example_drifted": "It varies by location and relationship."},
    ],
    "clean_clusters": [
        {"topic": "password reset steps", "cts": 0.05,
         "summary": "consistent across variants", "example_output": "Click forgot password."},
        {"topic": "shipping cost threshold", "cts": 0.10,
         "summary": "consistent", "example_output": "Free shipping over $50."},
    ],
}


def test_record_monitor_to_ledger_writes_commitments_and_contradictions():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        args = argparse.Namespace(ledger=path, no_ledger=False)
        summary = _record_monitor_to_ledger(args, _ANALYSIS)

        assert summary is not None
        assert summary["commitments"] == 2      # the two clean clusters
        assert summary["contradictions"] == 1    # the one hotspot
        assert summary["other_event_types"] == {"monitor_run": 1}
        assert summary["contradiction_rate"] == round(1 / 2, 3)
        assert summary["verified"] is True

        # and it actually persisted: 2 commitments + 1 contradiction + 1 run summary
        reloaded = CommitmentLedger.load(path)
        assert len(reloaded) == 4
        assert reloaded.verify() is True


def test_record_monitor_to_ledger_no_ledger_flag_skips_recording():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        args = argparse.Namespace(ledger=path, no_ledger=True)
        summary = _record_monitor_to_ledger(args, _ANALYSIS)
        assert summary is None
        assert not os.path.exists(path)


def test_record_monitor_to_ledger_accumulates_across_calls():
    """Two separate monitor runs against the same path build one growing
    chain, not two independent ones -- this is the whole point."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        args = argparse.Namespace(ledger=path, no_ledger=False)
        s1 = _record_monitor_to_ledger(args, _ANALYSIS)
        s2 = _record_monitor_to_ledger(args, _ANALYSIS)
        assert s2["entries"] == s1["entries"] * 2
        assert s2["verified"] is True
        assert s2["head"] != s1["head"]


def test_cmd_ledger_init_then_show_empty():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        out, code = _run_ledger("init", path)
        assert code is None
        assert os.path.exists(path)

        out, code = _run_ledger("show", path)
        assert "empty" in out


def test_cmd_ledger_show_after_recording():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        _record_monitor_to_ledger(argparse.Namespace(ledger=path, no_ledger=False), _ANALYSIS)
        out, code = _run_ledger("show", path)
        assert "entries:" in out and "4" in out  # 2 commitments + 1 contradiction + 1 run summary
        assert "verified" in out.lower()


def test_cmd_ledger_verify_ok_and_tampered():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        _record_monitor_to_ledger(argparse.Namespace(ledger=path, no_ledger=False), _ANALYSIS)

        out, code = _run_ledger("verify", path)
        assert code is None  # exits cleanly (no sys.exit call) when verified

        raw = json.loads(open(path).read())
        raw["entries"][0]["payload"]["confidence"] = 0.01
        open(path, "w").write(json.dumps(raw))

        out, code = _run_ledger("verify", path)
        assert code == 1  # must fail the process, this is a CI-gateable check


def test_cmd_ledger_verify_json_output_is_valid_json():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        _record_monitor_to_ledger(argparse.Namespace(ledger=path, no_ledger=False), _ANALYSIS)
        out, code = _run_ledger("verify", path, json=True)
        parsed = json.loads(out)
        assert parsed["verified"] is True


def test_cmd_ledger_anchor_includes_head_hash():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        _record_monitor_to_ledger(argparse.Namespace(ledger=path, no_ledger=False), _ANALYSIS)
        ledger = CommitmentLedger.load(path)
        out, code = _run_ledger("anchor", path, label="ci-run-42")
        assert "ci-run-42" in out
        assert ledger.head() in out


def test_cmd_ledger_init_refuses_overwrite_without_force():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.json")
        _record_monitor_to_ledger(argparse.Namespace(ledger=path, no_ledger=False), _ANALYSIS)
        before = CommitmentLedger.load(path)
        assert len(before) > 0

        out, code = _run_ledger("init", path)
        assert "already has" in out

        after = CommitmentLedger.load(path)
        assert len(after) == len(before)  # untouched


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
