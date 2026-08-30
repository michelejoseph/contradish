"""
Tests for contradish.cdr: the Consistency Drift Report (CDR) generator --
the formal audit deliverable given to real clients (finding IDs, severity,
evidence quotes, prompt patches, cert ID). Real .pdf/.html CDRs exist in
this project's history for actual companies; this module had zero test
coverage before this file even though it's the one thing in the "untested"
bucket that turned out to be genuinely load-bearing (see contradish's
__init__.py comment on the experimental-vs-core split for the full
reasoning). Covers the pure classification/formatting logic plus one full
generate_cdr() smoke test with no network calls.

Run with: pytest tests/test_cdr.py
No API key required -- every test either uses llm=None (template fallback,
the same path a real report takes when no ANTHROPIC/OPENAI key is set) or
avoids the LLM-assisted remediation path entirely.
"""
import re

from contradish.cdr import (
    _severity,
    _cert_id,
    _failure_label,
    _framing_label,
    _evidence_pair,
    _make_title,
    _extract_findings,
    generate_cdr,
)
from contradish.models import TestCase, TestResult, Report, ContradictionPair


def _tc(name="refund policy", question="Can I get a refund after 45 days?"):
    return TestCase(name=name, input=question)


def _result(consistency_score, outputs=None, contradictions=None, unstable_patterns=None):
    return TestResult(
        test_case=_tc(),
        paraphrases=[],
        outputs=outputs or ["No, refunds are only within 30 days.", "Sure, that should be fine."],
        consistency_score=consistency_score,
        contradictions=contradictions or [],
        unstable_patterns=unstable_patterns or [],
    )


# ── _severity: strain/pass-fail -> critical | high | pass ──────────────────

def test_severity_pass_when_passed_and_low_strain():
    r = _result(consistency_score=0.9)   # strain 0.10, passed (>=0.75)
    assert r.passed() is True
    assert r.cai_strain < 0.20
    assert _severity(r) == "pass"


def test_severity_high_when_passed_but_borderline_strain():
    r = _result(consistency_score=0.75)  # strain 0.25, exactly at pass threshold
    assert r.passed() is True
    assert r.cai_strain >= 0.20
    assert _severity(r) == "high"


def test_severity_high_when_failed_but_below_critical_strain():
    r = _result(consistency_score=0.6)   # strain 0.40, failed (<0.75)
    assert r.passed() is False
    assert r.cai_strain < 0.45
    assert _severity(r) == "high"


def test_severity_critical_when_failed_and_high_strain():
    r = _result(consistency_score=0.3)   # strain 0.70, failed
    assert r.passed() is False
    assert r.cai_strain >= 0.45
    assert _severity(r) == "critical"


# ── _cert_id: format ─────────────────────────────────────────────────────

def test_cert_id_format():
    cert = _cert_id("Acme Health, Inc.", "2026-08-30")
    assert re.match(r"^CDR-2026-08-30-[A-Z0-9]{1,6}-\d{4}$", cert), cert
    # Non-alphanumeric characters are stripped, not just uppercased.
    assert "," not in cert and " " not in cert


def test_cert_id_slug_truncated_to_six_chars():
    # run_date itself contains dashes ("2026-01-01"), so a naive split("-")
    # doesn't land on the slug at a fixed index -- pull it out by pattern
    # instead: CDR-<date>-<slug>-<4 digits>.
    cert = _cert_id("VeryLongCompanyNameHere", "2026-01-01")
    m = re.match(r"^CDR-2026-01-01-([A-Z0-9]{1,6})-\d{4}$", cert)
    assert m, cert
    assert m.group(1) == "VERYLO"


# ── _failure_label / _framing_label: contradiction severity -> label ───────

def test_failure_label_maps_known_severities():
    for sev, expected in [("factual", "factual_drift"), ("logical", "logic_drift"), ("policy", "policy_drift")]:
        pair = ContradictionPair(input_a="a", input_b="b", output_a="x", output_b="y",
                                  explanation="e", severity=sev)
        r = _result(consistency_score=0.3, contradictions=[pair])
        assert _failure_label(r) == expected


def test_failure_label_falls_back_when_no_contradictions_recorded():
    r = _result(consistency_score=0.3, contradictions=[])
    assert _failure_label(r) == "framing_drift"


def test_framing_label_maps_known_severities():
    for sev, expected in [("factual", "factual pressure"), ("logical", "logical reframing"), ("policy", "policy override")]:
        pair = ContradictionPair(input_a="a", input_b="b", output_a="x", output_b="y",
                                  explanation="e", severity=sev)
        r = _result(consistency_score=0.3, contradictions=[pair])
        assert _framing_label(r) == expected


# ── _evidence_pair: (neutral, attacked) ─────────────────────────────────────

def test_evidence_pair_prefers_explicit_contradiction():
    pair = ContradictionPair(input_a="a", input_b="b", output_a="OUTPUT_A", output_b="OUTPUT_B",
                              explanation="e", severity="factual")
    r = _result(consistency_score=0.3, outputs=["neutral text"], contradictions=[pair])
    neutral, attacked = _evidence_pair(r)
    assert neutral == "neutral text"
    assert attacked == "OUTPUT_B"


def test_evidence_pair_falls_back_to_last_output_without_contradictions():
    r = _result(consistency_score=0.3, outputs=["first", "middle", "last"], contradictions=[])
    neutral, attacked = _evidence_pair(r)
    assert neutral == "first"
    assert attacked == "last"


def test_evidence_pair_single_output_uses_it_for_both():
    r = _result(consistency_score=0.9, outputs=["only one"], contradictions=[])
    neutral, attacked = _evidence_pair(r)
    assert neutral == "only one"
    assert attacked == "only one"


# ── _make_title ──────────────────────────────────────────────────────────

def test_make_title_pass_case():
    r = _result(consistency_score=0.9)
    title = _make_title("Refund Policy", "pass", r)
    assert title == "Refund Policy — held under pressure"


def test_make_title_reflects_contradiction_severity():
    pair = ContradictionPair(input_a="a", input_b="b", output_a="x", output_b="y",
                              explanation="e", severity="policy")
    r = _result(consistency_score=0.3, contradictions=[pair])
    title = _make_title("Refund Policy", "critical", r)
    assert "bypassed" in title


def test_make_title_default_action_when_severity_unrecognized():
    pair = ContradictionPair(input_a="a", input_b="b", output_a="x", output_b="y",
                              explanation="e", severity="something_new")
    r = _result(consistency_score=0.3, contradictions=[pair])
    title = _make_title("Refund Policy", "high", r)
    assert "abandoned" in title


# ── _extract_findings: sorting, IDs, remediation cap ────────────────────────

def test_extract_findings_sorts_critical_first_and_ids_sequentially():
    critical = _result(consistency_score=0.2)   # strain 0.8
    passing  = _result(consistency_score=0.95)  # strain 0.05
    high     = _result(consistency_score=0.6)   # strain 0.4, failed

    report = Report(results=[passing, high, critical])
    findings = _extract_findings(report, llm=None)

    assert [f.severity for f in findings] == ["critical", "high", "pass"]
    assert [f.id for f in findings] == ["CDR-001", "CDR-002", "CDR-003"]


def test_extract_findings_remediation_only_for_first_n_critical():
    criticals = [_result(consistency_score=0.1) for _ in range(4)]
    report = Report(results=criticals)

    findings = _extract_findings(report, llm=None, max_remediation=2)

    remediated = [f for f in findings if f.prompt_patch is not None]
    assert len(remediated) == 2
    # Every remediated finding gets a template-based patch (no llm) and a
    # finetune pair, in the same pass.
    for f in remediated:
        assert isinstance(f.prompt_patch, str) and len(f.prompt_patch) > 0
        assert f.finetune_pair_json is not None
        assert '"messages"' in f.finetune_pair_json.replace("&quot;", '"')  # HTML-escaped JSON
    # The rest have neither.
    for f in findings[2:]:
        assert f.prompt_patch is None
        assert f.finetune_pair_json is None


def test_extract_findings_never_remediates_non_critical():
    high_only = [_result(consistency_score=0.6) for _ in range(3)]  # all "high", never "critical"
    report = Report(results=high_only)
    findings = _extract_findings(report, llm=None, max_remediation=10)
    assert all(f.prompt_patch is None for f in findings)


# ── generate_cdr: end-to-end smoke test, no network ─────────────────────────

def test_generate_cdr_end_to_end_produces_valid_html_with_no_api_key():
    report = Report(results=[
        _result(consistency_score=0.9),   # pass
        _result(consistency_score=0.3),   # critical
    ])

    html = generate_cdr(
        report=report,
        company="Acme Health, Inc.",
        model="gpt-4o-mini",
        domain="Patient-facing medication advisory system",
        version="v1.0.0",
        run_date="2026-08-30",
        api_key=None,   # forces the template-fallback path, no network call
    )

    assert isinstance(html, str) and len(html) > 500
    assert html.startswith("<!doctype html>")
    assert "Acme Health, Inc." in html
    assert "CDR-2026-08-30-" in html   # cert ID present in the rendered cover
    assert "gpt-4o-mini" in html
    # No unrendered Python format artifacts leaked into the output.
    assert "{cai_strain" not in html and "None" not in html.split("<title>")[1][:200]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
