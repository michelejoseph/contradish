"""
Full test coverage for contradish/audit.py.

audit.py has one public entry point, `to_audit_html(report, **kwargs)`, which
renders a self-contained compliance/audit HTML document from a Report. It is
a pure string-builder: every branch is either a threshold comparison
(_score_color / _score_label / risk level), an "is this optional kwarg
present" conditional (app_version / policy_name / evaluator_id / notes /
system_prompt), a truncation ("..." appended past 300/80 chars), or a loop
over report.failed / report.results.

These tests build TestCase/TestResult/ContradictionPair/Report fixtures
directly (real dataclasses, no mocks) and assert on substrings of the
rendered HTML, matching this session's established style (see
tests/test_findings.py).
"""
import contradish
from contradish.audit import to_audit_html, _esc, _score_color, _score_label
from contradish.models import ContradictionPair, Report, TestCase, TestResult, RiskLevel


# ── Fixture builders ────────────────────────────────────────────────────────

def _tc(input="a test question", name=None, **kw):
    return TestCase(input=input, name=name, **kw)


def _cp(input_a="in a", input_b="in b", output_a="out a", output_b="out b",
        explanation="they disagree", severity="policy"):
    return ContradictionPair(
        input_a=input_a, input_b=input_b,
        output_a=output_a, output_b=output_b,
        explanation=explanation, severity=severity,
    )


def _tr(tc=None, consistency_score=0.9, contradictions=None, risk=RiskLevel.LOW,
        suggestion=None, **kw):
    return TestResult(
        test_case=tc or _tc(),
        paraphrases=["p1"],
        outputs=["o0", "o1"],
        consistency_score=consistency_score,
        contradiction_score=0.0,
        risk=risk,
        contradictions=contradictions or [],
        suggestion=suggestion,
        **kw,
    )


def _report(results, thresholds=None):
    return Report(results=results, thresholds=thresholds or {})


# ── _esc ─────────────────────────────────────────────────────────────────────

def test_esc_empty_string_returns_empty():
    assert _esc("") == ""


def test_esc_none_returns_empty():
    assert _esc(None) == ""


def test_esc_escapes_html_special_characters():
    out = _esc("<script>alert('x')&\"y\"</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&amp;" in out
    assert "&#x27;" in out or "'" not in out
    assert "&quot;" in out


# ── _score_color ─────────────────────────────────────────────────────────────

def test_score_color_high_is_green():
    assert _score_color(0.80) == "#16a34a"
    assert _score_color(1.0) == "#16a34a"


def test_score_color_mid_is_amber():
    assert _score_color(0.60) == "#d97706"
    assert _score_color(0.79) == "#d97706"


def test_score_color_low_is_red():
    assert _score_color(0.59) == "#dc2626"
    assert _score_color(0.0) == "#dc2626"


# ── _score_label ─────────────────────────────────────────────────────────────

def test_score_label_high_is_stable():
    assert _score_label(0.80) == "Stable"


def test_score_label_mid_is_marginal():
    assert _score_label(0.60) == "Marginal"
    assert _score_label(0.79) == "Marginal"


def test_score_label_low_is_unstable():
    assert _score_label(0.59) == "Unstable"


# ── to_audit_html: minimal / all-defaults path ──────────────────────────────

def test_minimal_call_all_passing_no_optional_kwargs():
    report = _report([_tr(consistency_score=0.95)])
    out = to_audit_html(report)

    # doc_id falls back to "MANUAL" when no evaluator_id given
    assert "CONTRADISH-" in out
    assert "-MANUAL" in out

    # version string is present in config table and footer
    assert contradish.__version__ in out

    # optional config rows are omitted entirely
    assert "App version" not in out
    assert "Policy pack" not in out
    assert "Evaluator / Run ID" not in out

    # no failures -> pass-note branch, not the failure table
    assert "No CAI failures detected" in out
    assert "CAI Failures" in out  # heading still present in the pass branch

    # optional sections omitted
    assert "Appendix A" not in out
    assert "Evaluator Notes" not in out

    # risk assessment: high score -> LOW risk
    assert "Risk Level: LOW" in out
    assert "Consistency is within acceptable bounds" in out

    # summary stats
    assert "<title>CAI Audit Report" in out
    assert "</html>" in out


def test_empty_report_defaults_agg_to_zero_and_high_risk():
    report = _report([])
    out = to_audit_html(report)

    # cai_score is None on an empty report -> agg falls back to 0.0
    assert "Risk Level: HIGH" in out
    assert "Significant inconsistency detected" in out
    assert "No CAI failures detected" in out  # report.failed is [] too

    # stat boxes render zeros
    assert ">0.00<" in out or "0.00" in out


# ── Evaluation config table: optional fields ────────────────────────────────

def test_config_rows_include_all_optional_fields_when_given():
    report = _report([_tr()])
    out = to_audit_html(
        report,
        app_version="prod-v12",
        policy_name="ecommerce",
        evaluator_id="ci-run-456",
    )
    assert "App version" in out
    assert "prod-v12" in out
    assert "Policy pack" in out
    assert "ecommerce" in out
    assert "Evaluator / Run ID" in out
    assert "ci-run-456" in out
    # doc_id uses the upper-cased evaluator_id
    assert "CI-RUN-456" in out


def test_doc_id_truncates_long_evaluator_id_to_twelve_chars():
    report = _report([_tr()])
    out = to_audit_html(report, evaluator_id="this-is-a-very-long-evaluator-id")
    # first 12 chars of the upper-cased id
    assert "THIS-IS-A-VE" in out
    assert "THIS-IS-A-VERY" not in out


# ── Risk levels ──────────────────────────────────────────────────────────────

def test_risk_medium_between_60_and_80():
    report = _report([_tr(consistency_score=0.70)])
    out = to_audit_html(report)
    assert "Risk Level: MEDIUM" in out
    assert "Marginal consistency" in out


def test_risk_high_below_60():
    report = _report([_tr(consistency_score=0.30)])
    out = to_audit_html(report)
    assert "Risk Level: HIGH" in out
    assert "Significant inconsistency detected" in out


# ── Failures section: content, contradiction pairs, truncation ─────────────

def test_failure_row_and_pair_rendering_with_escaping():
    long_a = "A" * 305
    long_b = "B" * 300  # exactly 300 -> no ellipsis
    pair = _cp(
        input_a="<q1>", input_b="<q2>",
        output_a=long_a, output_b=long_b,
        explanation="<explains & clarifies>",
        severity="critical",
    )
    failing = _tr(
        tc=_tc(input="fail case", name="<Rule Name>"),
        consistency_score=0.10,  # fails default threshold 0.75
        risk=RiskLevel.HIGH,
        contradictions=[pair],
        suggestion="<try this fix>",
    )
    report = _report([failing])
    out = to_audit_html(report)

    assert "CAI Failures" in out
    assert "&lt;Rule Name&gt;" in out
    assert "HIGH" in out  # risk.value.upper()
    assert "&lt;try this fix&gt;" in out

    # contradiction pair block
    assert "&lt;q1&gt;" in out
    assert "&lt;q2&gt;" in out
    assert "&lt;explains &amp; clarifies&gt;" in out

    # truncation: 305-char output gets cut to 300 + "..."
    assert ("A" * 300 + "...") in out
    assert ("A" * 301) not in out
    # exactly-300-char output is NOT truncated (no ellipsis appended)
    assert ("B" * 300) in out
    assert ("B" * 300 + "...") not in out


def test_no_failures_shows_pass_note_not_failure_table():
    report = _report([_tr(consistency_score=0.95)])
    out = to_audit_html(report)
    assert "All rules produced consistent responses" in out
    assert 'class="failure-row"' not in out


def test_multiple_contradiction_pairs_on_one_failure_all_rendered():
    pairs = [_cp(input_a=f"q{i}") for i in range(3)]
    failing = _tr(consistency_score=0.1, contradictions=pairs)
    out = to_audit_html(_report([failing]))
    for i in range(3):
        assert f"q{i}" in out
    assert out.count('class="pair-row"') == 3


# ── All test cases table: sorting, truncation, pass/fail status ────────────

def test_all_cases_table_sorted_by_score_and_status_labels():
    low = _tr(tc=_tc(input="low score case", name="low"), consistency_score=0.20)
    high = _tr(tc=_tc(input="high score case", name="high"), consistency_score=0.99)
    mid = _tr(tc=_tc(input="mid score case", name="mid"), consistency_score=0.60)
    report = _report([high, low, mid])
    out = to_audit_html(report)

    # sorted ascending by cai_score: low (0.20) < mid (0.60) < high (0.99)
    pos_low = out.index(">low<")
    pos_mid = out.index(">mid<")
    pos_high = out.index(">high<")
    assert pos_low < pos_mid < pos_high

    # low and mid fail the default 0.75 consistency threshold; high passes
    assert out.count("FAIL") >= 2
    assert "PASS" in out


def test_all_cases_table_handles_none_cai_score_as_zero_in_sort():
    # skipped result: consistency_score is None -> cai_score None -> sort key 0.0
    # both must PASS (score None never fails a threshold check) so neither
    # lands in the CAI Failures table, which would confound the ordering
    # check on the "All Test Cases" table below.
    skipped = _tr(tc=_tc(input="skipped case", name="skipped"),
                   consistency_score=None, skipped=True)
    scored = _tr(tc=_tc(input="scored case", name="scored"), consistency_score=0.85)
    out = to_audit_html(_report([scored, skipped]))
    # skipped (key 0.0) sorts before scored (key 0.50)
    assert out.index(">skipped<") < out.index(">scored<")
    assert "0.00" in out  # skipped result rendered with score 0.00


def test_input_column_truncates_past_eighty_chars():
    long_input = "x" * 85
    short_input = "y" * 80  # exactly 80 -> no ellipsis
    report = _report([
        _tr(tc=_tc(input=long_input, name="longcase"), consistency_score=0.9),
        _tr(tc=_tc(input=short_input, name="shortcase"), consistency_score=0.9),
    ])
    out = to_audit_html(report)
    assert ("x" * 80 + "...") in out
    assert ("x" * 81) not in out
    assert ("y" * 80) in out
    assert ("y" * 80 + "...") not in out


def test_custom_thresholds_affect_pass_fail_status():
    # consistency 0.50 would FAIL the default 0.75 threshold but PASS a
    # relaxed 0.40 threshold configured on the report.
    result = _tr(tc=_tc(name="relaxed"), consistency_score=0.50)
    report = _report([result], thresholds={"consistency": 0.40})
    out = to_audit_html(report)
    assert "PASS" in out
    assert "FAIL" not in out


# ── Appendix / notes sections ───────────────────────────────────────────────

def test_system_prompt_appendix_rendered_and_escaped():
    report = _report([_tr()])
    out = to_audit_html(report, system_prompt="You are <helpful> & careful.")
    assert "Appendix A: System Prompt Under Test" in out
    assert "You are &lt;helpful&gt; &amp; careful." in out


def test_notes_section_rendered_and_escaped():
    report = _report([_tr()])
    out = to_audit_html(report, notes="Escalate <this> & review.")
    assert "Evaluator Notes" in out
    assert "Escalate &lt;this&gt; &amp; review." in out


def test_prompt_and_notes_absent_by_default():
    report = _report([_tr()])
    out = to_audit_html(report)
    assert "Appendix A" not in out
    assert "Evaluator Notes" not in out


# ── Full-kwargs integration smoke test ──────────────────────────────────────

def test_all_kwargs_together_produce_well_formed_document():
    pair = _cp(explanation="drifted under pressure")
    failing = _tr(
        tc=_tc(name="policy rule", input="what's the refund window?"),
        consistency_score=0.40,
        risk=RiskLevel.MEDIUM,
        contradictions=[pair],
        suggestion="pin the policy fact",
    )
    passing = _tr(tc=_tc(name="stable rule"), consistency_score=0.95)
    report = _report([failing, passing])

    out = to_audit_html(
        report,
        app_version="prod-v12",
        system_prompt="You are a support agent.",
        evaluator_id="ci-run-456",
        policy_name="ecommerce",
        notes="Reviewed by compliance.",
    )

    assert out.startswith("<!DOCTYPE html>")
    assert out.rstrip().endswith("</html>")
    assert "CI-RUN-456" in out
    assert "ecommerce" in out
    assert "prod-v12" in out
    assert "Reviewed by compliance." in out
    assert "You are a support agent." in out
    assert "policy rule" in out
    assert "stable rule" in out
    assert "drifted under pressure" in out
    # NIST / EU AI Act / ISO regulatory alignment section always present
    assert "NIST AI RMF" in out
    assert "EU AI Act" in out
    assert "ISO/IEC 42001" in out
