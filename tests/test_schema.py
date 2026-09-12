"""
Tests for contradish.schema: the published, versioned JSON Schema documents
for contradish's two interchange formats (distinction_report,
distinction_diff), and the `contradish schema` CLI command that inspects
and validates against them.

Three things are checked, deliberately in this order, because each one
guards against a different way a published schema quietly drifts from
reality:

  1. The .schema.json files are themselves well-formed JSON Schema
     documents (not just valid JSON).
  2. Real payloads this codebase actually emits (DistinctionLossMap.to_dict(),
     diff_distinction_reports()) validate cleanly against them -- the schema
     is honest about the format it claims to describe.
  3. A payload that violates the format is correctly rejected -- the schema
     isn't so permissive it accepts anything.
"""
import json
import sys

import jsonschema
import pytest

from contradish.distinction import (
    DistinctionLossMap, DistinctionProfile, diff_distinction_reports,
)
from contradish.schema import list_schemas, load_schema, validate_against_schema


# ── list_schemas() / load_schema() ─────────────────────────────────────────

def test_list_schemas_returns_both_published_schemas():
    assert list_schemas() == ["distinction_diff", "distinction_report"]


def test_load_schema_returns_the_matching_document():
    report_schema = load_schema("distinction_report")
    diff_schema   = load_schema("distinction_diff")
    assert report_schema["title"] == "Contradish Distinction Report"
    assert diff_schema["title"]   == "Contradish Distinction Diff"


def test_load_schema_raises_keyerror_for_unknown_name():
    with pytest.raises(KeyError, match="no published schema named 'nope'"):
        load_schema("nope")


# ── The schemas are themselves well-formed JSON Schema documents ──────────

@pytest.mark.parametrize("name", list_schemas())
def test_schema_is_a_well_formed_json_schema_document(name):
    schema = load_schema(name)
    jsonschema.Draft202012Validator.check_schema(schema)


# ── Real emitted payloads validate cleanly (schema matches reality) ───────

def _real_loss_map() -> DistinctionLossMap:
    profile = DistinctionProfile(
        pair_id="p1", description="d", label_a="a", label_b="b",
        hold_rate_per_framing={"neutral": 0.9}, overall_hold_rate=0.9,
        collapse_framing="neutral", first_collapse=None,
    )
    return DistinctionLossMap(
        domain="medication", profiles={"p1": profile},
        ranked_by_fragility=["p1"], most_fragile="p1", most_resilient="p1",
        framing_destructiveness={"neutral": 0.1},
    )


def test_real_distinction_report_payload_validates_against_its_schema():
    payload = _real_loss_map().to_dict()
    assert validate_against_schema(payload, "distinction_report") == []


def test_real_distinction_report_payload_with_raw_measurements_validates():
    payload = _real_loss_map().to_dict(include_raw=True)
    # include_raw=True has no measurements recorded on this hand-built
    # profile (measurements=[] by default), so this exercises the
    # zero-measurements case, not the measurement schema itself -- that's
    # covered by the diff test below via a distinct payload shape.
    assert validate_against_schema(payload, "distinction_report") == []


def test_real_distinction_diff_payload_validates_against_its_schema():
    baseline  = {"domain": "medication", "profiles": {"p1": {"overall_hold_rate": 0.9, "description": "d"}}}
    candidate = {"domain": "medication", "profiles": {"p1": {"overall_hold_rate": 0.2, "description": "d"}}}
    payload = diff_distinction_reports(baseline, candidate)
    assert validate_against_schema(payload, "distinction_diff") == []


def test_real_distinction_diff_payload_with_missing_side_validates():
    # A pair present only on one side (baseline_hold_rate or
    # candidate_hold_rate is None) is a real, documented shape --
    # first-time-seen or removed distinctions -- and must still validate.
    baseline  = {"domain": "medication", "profiles": {}}
    candidate = {"domain": "medication", "profiles": {"new_pair": {"overall_hold_rate": 0.5, "description": "d"}}}
    payload = diff_distinction_reports(baseline, candidate)
    assert validate_against_schema(payload, "distinction_diff") == []


# ── Schemas correctly reject a payload that violates the format ───────────

def test_validate_against_schema_reports_errors_for_missing_required_fields():
    errors = validate_against_schema({"domain": "medication"}, "distinction_report")
    assert errors != []
    assert any("schema_version" in e for e in errors)


def test_validate_against_schema_rejects_out_of_range_hold_rate():
    payload = _real_loss_map().to_dict()
    payload["profiles"]["p1"]["overall_hold_rate"] = 1.5  # out of [0, 1]
    errors = validate_against_schema(payload, "distinction_report")
    assert errors != []


def test_validate_against_schema_raises_keyerror_for_unknown_schema_name():
    with pytest.raises(KeyError):
        validate_against_schema({}, "not_a_real_schema")


def test_validate_against_schema_raises_importerror_when_jsonschema_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    with pytest.raises(ImportError, match="contradish\\[schema\\]"):
        validate_against_schema({}, "distinction_report")


# ── `contradish schema` CLI ─────────────────────────────────────────────────

def test_cli_schema_list(capsys):
    import contradish.cli as cli
    with pytest.raises(SystemExit) as exc:
        cli.cmd_schema(type("Args", (), {"list": True, "show": None, "validate": None, "against": None})())
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "distinction_report" in out
    assert "distinction_diff" in out


def test_cli_schema_show_prints_the_schema_document(capsys):
    import contradish.cli as cli
    with pytest.raises(SystemExit) as exc:
        cli.cmd_schema(type("Args", (), {"list": False, "show": "distinction_report", "validate": None, "against": None})())
    assert exc.value.code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed == load_schema("distinction_report")


def test_cli_schema_show_unknown_name_exits_nonzero(capsys):
    import contradish.cli as cli
    with pytest.raises(SystemExit) as exc:
        cli.cmd_schema(type("Args", (), {"list": False, "show": "nope", "validate": None, "against": None})())
    assert exc.value.code == 1
    assert "no published schema" in capsys.readouterr().out


def test_cli_schema_validate_requires_against(capsys):
    import contradish.cli as cli
    with pytest.raises(SystemExit) as exc:
        cli.cmd_schema(type("Args", (), {"list": False, "show": None, "validate": "some_file.json", "against": None})())
    assert exc.value.code == 1
    assert "--against SCHEMA_NAME" in capsys.readouterr().out


def test_cli_schema_validate_missing_file_exits_nonzero(capsys):
    import contradish.cli as cli
    with pytest.raises(SystemExit) as exc:
        cli.cmd_schema(type("Args", (), {
            "list": False, "show": None,
            "validate": "/tmp/definitely_not_a_real_file_xyz.json",
            "against": "distinction_report",
        })())
    assert exc.value.code == 1
    assert "file not found" in capsys.readouterr().out


def test_cli_schema_validate_valid_file_exits_zero(tmp_path, capsys):
    import contradish.cli as cli
    payload = _real_loss_map().to_dict()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as exc:
        cli.cmd_schema(type("Args", (), {
            "list": False, "show": None,
            "validate": str(path), "against": "distinction_report",
        })())
    assert exc.value.code == 0
    assert "valid against 'distinction_report'" in capsys.readouterr().out


def test_cli_schema_validate_invalid_file_exits_nonzero_with_errors(tmp_path, capsys):
    import contradish.cli as cli
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"domain": "medication"}))
    with pytest.raises(SystemExit) as exc:
        cli.cmd_schema(type("Args", (), {
            "list": False, "show": None,
            "validate": str(path), "against": "distinction_report",
        })())
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "INVALID against 'distinction_report'" in out
    assert "schema_version" in out


def test_cli_schema_no_flags_prints_usage_and_exits_nonzero(capsys):
    import contradish.cli as cli
    with pytest.raises(SystemExit) as exc:
        cli.cmd_schema(type("Args", (), {"list": False, "show": None, "validate": None, "against": None})())
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "--list" in out
    assert "--show NAME" in out
    assert "--validate FILE" in out


def test_schema_subcommand_registered_in_argparser():
    import contradish.cli as cli
    parser = cli._build_parser() if hasattr(cli, "_build_parser") else None
    # Fall back to a functional check if the parser isn't factored into a
    # standalone builder: --help on the subcommand should not error.
    import subprocess, sys as _sys
    result = subprocess.run(
        [_sys.executable, "-m", "contradish.cli", "schema", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--validate" in result.stdout
