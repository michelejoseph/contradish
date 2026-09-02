"""
Full test coverage for contradish/reporter.py -- the self-contained HTML
report generator (`to_html(report)`).

These tests build TestCase/TestResult/Report fixtures directly (real
dataclasses, no mocks) and drive every private helper plus `to_html` through
its branches: score thresholds/colors/labels, escaping, singular/plural
summary wording, contradiction/why/fix block presence, truncation of long
response text, and pluralization of the "+N more contradictions" line.

Not covered (documented, not papered over):
  - `to_html`'s `eq_subline` has an `else "audited equivalence: not yet
    reported"` branch guarded by `eq_cov is not None`. `Report.eq_coverage`
    is a property that always returns a float (0.0 for an empty report,
    never None), so through any real Report this branch is unreachable --
    it is dead code, not a real behavior to test. Left uncovered rather than
    faking a non-Report object to force it.

Two real bugs were found while writing these tests and have since been
FIXED in contradish/reporter.py (not just pinned):
  1. `to_html(report, version="x.y.z")` used to silently ignore the
     `version` parameter -- the function immediately did
     `from . import __version__; version = __version__`, clobbering
     whatever the caller passed. Fixed: the default changed from a
     hardcoded stale string to None, and the installed version is only
     used as a fallback when no explicit version is given. See
     test_to_html_version_parameter_is_honored /
     test_to_html_defaults_version_to_installed_contradish_version.
  2. `to_html` used to crash with TypeError on a Report with no results
     (or one whose results all have `consistency_score=None`), because
     `report.cai_score` is None in that case and `_score_color(None)` did
     `None >= 0.80`. Fixed: falls back to 0.0 instead of crashing. See
     test_to_html_handles_report_with_no_results /
     test_to_html_handles_report_where_all_results_have_no_consistency_score.
"""
import re

import pytest

from contradish.models import ContradictionPair, Report, TestCase, TestResult
from contradish.reporter import (
    _esc,
    _score_bg,
    _score_border,
    _score_color,
    _score_label,
    _passing_card,
    _failing_card,
    to_html,
)


# ── Fixture builders ────────────────────────────────────────────────────────

def _tc(input="a test question", name=None, contradiction_type="adversarial",
        equivalence_confidence=1.0, **kw):
    return TestCase(
        input=input,
        name=name,
        contradiction_type=contradiction_type,
        equivalence_confidence=equivalence_confidence,
        **kw,
    )


def _cp(output_a="out a", output_b="out b", severity="policy",
        explanation="they disagree", input_a="in a", input_b="in b"):
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


def _report(results, **kw):
    return Report(results=results, **kw)


# ── _esc ─────────────────────────────────────────────────────────────────

def test_esc_empty_string_returns_empty():
    assert _esc("") == ""


def test_esc_none_returns_empty():
    assert _esc(None) == ""


def test_esc_escapes_html_special_chars():
    out = _esc("<script>alert('x')&\"y\"</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&amp;" in out


# ── _score_color / _score_label / _score_bg / _score_border ────────────────

def test_score_helpers_at_high_boundary_are_green_stable():
    assert _score_color(0.80) == "#16a34a"
    assert _score_label(0.80) == "stable"
    assert _score_bg(0.80) == "#f0fdf4"
    assert _score_border(0.80) == "#bbf7d0"
    assert _score_color(1.0) == "#16a34a"


def test_score_helpers_just_below_high_boundary_are_amber_marginal():
    assert _score_color(0.79) == "#d97706"
    assert _score_label(0.79) == "marginal"
    assert _score_bg(0.79) == "#fffbeb"
    assert _score_border(0.79) == "#fde68a"


def test_score_helpers_at_mid_boundary_are_amber_marginal():
    assert _score_color(0.60) == "#d97706"
    assert _score_label(0.60) == "marginal"


def test_score_helpers_just_below_mid_boundary_are_red_unstable():
    assert _score_color(0.59) == "#dc2626"
    assert _score_label(0.59) == "unstable"
    assert _score_bg(0.59) == "#fef2f2"
    assert _score_border(0.59) == "#fecaca"


def test_score_helpers_at_zero_are_red_unstable():
    assert _score_color(0.0) == "#dc2626"
    assert _score_label(0.0) == "unstable"


# ── _passing_card ────────────────────────────────────────────────────────

def test_passing_card_renders_name_and_score():
    r = _tr(tc=_tc(name="my <rule>"), consistency_score=0.95)
    out = _passing_card(r)
    assert "rule-pass" in out
    assert "my &lt;rule&gt;" in out
    assert "0.95" in out
    assert "stable" in out


def test_passing_card_falls_back_to_zero_when_score_none():
    r = _tr(tc=_tc(name="no score"), consistency_score=None)
    out = _passing_card(r)
    assert "0.00" in out
    assert _score_color(0.0) in out


# ── _failing_card ────────────────────────────────────────────────────────

def test_failing_card_minimal_has_no_optional_blocks():
    r = _tr(consistency_score=0.2, contradictions=[], unstable_patterns=[], suggestion=None)
    out = _failing_card(r)
    assert "rule-fail" in out
    assert "contradiction-block" not in out
    assert "why-block" not in out
    assert "fix-block" not in out


def test_failing_card_shows_contradiction_block_with_escaped_text():
    pair = _cp(input_a="q<a>", output_a="answer <a>", input_b="q<b>", output_b="answer <b>")
    r = _tr(consistency_score=0.2, contradictions=[pair])
    out = _failing_card(r)
    assert "contradiction-block" in out
    assert "q&lt;a&gt;" in out
    assert "answer &lt;a&gt;" in out
    assert "q&lt;b&gt;" in out
    assert "answer &lt;b&gt;" in out
    assert "&ldquo;" in out and "&rdquo;" in out
    assert "same intent, different phrasing" in out
    assert "Both answers reached real users" in out
    assert "extra-count" not in out


def test_failing_card_single_extra_contradiction_is_singular():
    pairs = [_cp(), _cp(input_a="second")]
    r = _tr(consistency_score=0.2, contradictions=pairs)
    out = _failing_card(r)
    assert "+ 1 more contradiction on this rule" in out
    assert "contradictions" not in out.split("+ 1 more contradiction on this rule")[1].split("</p>")[0]


def test_failing_card_multiple_extra_contradictions_are_plural():
    pairs = [_cp(), _cp(input_a="second"), _cp(input_a="third")]
    r = _tr(consistency_score=0.2, contradictions=pairs)
    out = _failing_card(r)
    assert "+ 2 more contradictions on this rule" in out


def test_failing_card_truncates_long_response_text_past_200_chars():
    long_text = "x" * 250
    pair = _cp(output_a=long_text, output_b="short")
    r = _tr(consistency_score=0.2, contradictions=[pair])
    out = _failing_card(r)
    assert ("x" * 200 + "...") in out
    assert ("x" * 201) not in out


def test_failing_card_does_not_truncate_short_response_text():
    short_text = "y" * 50
    pair = _cp(output_a=short_text, output_b="short")
    r = _tr(consistency_score=0.2, contradictions=[pair])
    out = _failing_card(r)
    assert short_text in out
    assert (short_text + "...") not in out


def test_failing_card_shows_why_block_with_first_pattern_only():
    r = _tr(consistency_score=0.2, unstable_patterns=["first <pattern>", "second pattern"])
    out = _failing_card(r)
    assert "why-block" in out
    assert "first &lt;pattern&gt;" in out
    assert "second pattern" not in out


def test_failing_card_shows_fix_block_with_escaped_suggestion():
    r = _tr(consistency_score=0.2, suggestion="add <this> line")
    out = _failing_card(r)
    assert "fix-block" in out
    assert "add &lt;this&gt; line" in out


def test_failing_card_uses_marginal_styling_between_60_and_80():
    r = _tr(consistency_score=0.65)
    out = _failing_card(r)
    assert "marginal" in out
    assert _score_bg(0.65) in out
    assert _score_border(0.65) in out


def test_failing_card_uses_unstable_styling_below_60():
    r = _tr(consistency_score=0.3)
    out = _failing_card(r)
    assert "unstable" in out


def test_failing_card_score_falls_back_to_zero_when_none():
    r = _tr(consistency_score=None)
    out = _failing_card(r)
    assert "0.00" in out
    assert "unstable" in out


# ── to_html: titles / headers ───────────────────────────────────────────

def test_to_html_default_title():
    r = _report([_tr()])
    out = to_html(r)
    assert "<title>contradish CAI Report</title>" in out


def test_to_html_custom_title_override():
    r = _report([_tr()])
    out = to_html(r, title="My <Custom> Report")
    assert "<title>My &lt;Custom&gt; Report</title>" in out
    assert "contradish CAI Report</title>" not in out


def test_to_html_policy_name_shown_in_source_tag():
    r = _report([_tr()])
    out = to_html(r, policy_name="ecommerce")
    assert "policy pack: ecommerce" in out


def test_to_html_default_source_tag_without_policy_name():
    r = _report([_tr()])
    out = to_html(r)
    assert "CAI report" in out
    assert "policy pack:" not in out


def test_to_html_version_parameter_is_honored():
    # Previously a bug: the function body unconditionally did
    # `from . import __version__; version = __version__`, clobbering
    # whatever the caller passed. Fixed: an explicit version now wins, and
    # the parameter default changed from a hardcoded stale string to None
    # (meaning "use the installed contradish version").
    r = _report([_tr()])
    out = to_html(r, version="9.9.9-explicit")
    assert "v9.9.9-explicit" in out


def test_to_html_defaults_version_to_installed_contradish_version():
    from contradish import __version__ as real_version

    r = _report([_tr()])
    out = to_html(r)
    assert f"v{real_version}" in out


# ── to_html: empty / all-unscored report no longer crashes ─────────────

def test_to_html_handles_report_with_no_results():
    # Previously a bug: report.cai_score is None when nothing is scored,
    # and _score_color(agg)/_score_label(agg) did `agg >= 0.80`
    # unconditionally, so None vs. float crashed with TypeError. Fixed:
    # to_html now falls back to 0.0 for that internal color/label
    # computation rather than crashing. The visible headline number still
    # reads "n/a" (that's report.headline_strain, a separate None-safe
    # property already handled elsewhere in to_html) -- what this test
    # guards is that construction no longer raises.
    r = _report([])
    out = to_html(r)
    assert _score_color(0.0) in out  # the color the None->0.0 fallback resolves to
    assert "No CAI failures. All 0 rules stable." in out


def test_to_html_handles_report_where_all_results_have_no_consistency_score():
    # Same fix as above, reached via all-None consistency_score results
    # instead of an empty results list (both make report.cai_score None).
    r = _report([_tr(consistency_score=None), _tr(consistency_score=None)])
    out = to_html(r)  # no longer raises
    assert out.startswith("<!DOCTYPE html>")
    assert _score_color(0.0) in out


# ── to_html: summary line wording ───────────────────────────────────────

def test_to_html_summary_all_stable_singular_rule():
    r = _report([_tr(consistency_score=0.9)])
    out = to_html(r)
    assert "No CAI failures. All 1 rule stable." in out


def test_to_html_summary_all_stable_plural_rules():
    r = _report([_tr(consistency_score=0.9), _tr(tc=_tc(name="two"), consistency_score=0.95)])
    out = to_html(r)
    assert "No CAI failures. All 2 rules stable." in out


def test_to_html_summary_single_failure_single_pass_singular_wording():
    r = _report([
        _tr(tc=_tc(name="fail-one"), consistency_score=0.2),
        _tr(tc=_tc(name="pass-one"), consistency_score=0.9),
    ])
    out = to_html(r)
    assert "1 CAI failure found" in out
    assert "1 rule clean" in out


def test_to_html_summary_multiple_failures_multiple_passes_plural_wording():
    r = _report([
        _tr(tc=_tc(name="fail-a"), consistency_score=0.2),
        _tr(tc=_tc(name="fail-b"), consistency_score=0.3),
        _tr(tc=_tc(name="pass-a"), consistency_score=0.9),
        _tr(tc=_tc(name="pass-b"), consistency_score=0.95),
    ])
    out = to_html(r)
    assert "2 CAI failures found" in out
    assert "2 rules clean" in out


# ── to_html: section presence ───────────────────────────────────────────

def test_to_html_omits_rules_section_when_nothing_failed():
    r = _report([_tr(consistency_score=0.9)])
    out = to_html(r)
    assert 'class="rules-section"' not in out
    assert 'class="passing-section"' in out
    assert "passing rules" in out


def test_to_html_omits_passing_section_when_everything_failed():
    r = _report([_tr(consistency_score=0.2)])
    out = to_html(r)
    assert 'class="rules-section"' in out
    assert 'class="passing-section"' not in out


def test_to_html_includes_both_sections_when_mixed():
    r = _report([
        _tr(tc=_tc(name="fail-one"), consistency_score=0.2),
        _tr(tc=_tc(name="pass-one"), consistency_score=0.9),
    ])
    out = to_html(r)
    assert 'class="rules-section"' in out
    assert 'class="passing-section"' in out
    # failing cards render before the passing section in document order
    assert out.index('class="rules-section"') < out.index('class="passing-section"')


def test_to_html_passing_row_shows_name_and_score():
    r = _report([_tr(tc=_tc(name="my passing <rule>"), consistency_score=0.88)])
    out = to_html(r)
    assert "my passing &lt;rule&gt;" in out
    assert "0.88 stable" in out


# ── to_html: aggregate score / headline strain / eq coverage ───────────

def test_to_html_shows_headline_strain_and_eq_coverage_when_all_audited():
    r = _report([
        _tr(tc=_tc(name="a", equivalence_confidence=1.0), consistency_score=0.9),
        _tr(tc=_tc(name="b", equivalence_confidence=1.0), consistency_score=0.8),
    ])
    out = to_html(r)
    assert f"{r.headline_strain:.2f}" in out
    assert "audited equivalence: 100% of cases" in out


def test_to_html_shows_na_headline_strain_when_no_case_clears_eq_threshold():
    r = _report([
        _tr(tc=_tc(name="a", equivalence_confidence=0.5), consistency_score=0.9),
    ])
    assert r.headline_strain is None
    out = to_html(r)
    assert 'class="score-number"' in out
    assert ">n/a</div>" in out
    assert "audited equivalence: 0% of cases" in out


def test_to_html_agg_color_reflects_cai_score_tier():
    r = _report([_tr(consistency_score=0.9)])
    out = to_html(r)
    assert _score_color(r.cai_score) in out


# ── to_html: overall document shape ─────────────────────────────────────

def test_to_html_returns_full_self_contained_document():
    r = _report([
        _tr(tc=_tc(name="fail"), consistency_score=0.2, contradictions=[_cp()],
            unstable_patterns=["pattern"], suggestion="do this"),
        _tr(tc=_tc(name="pass"), consistency_score=0.9),
    ])
    out = to_html(r)
    assert out.startswith("<!DOCTYPE html>")
    assert out.rstrip().endswith("</html>")
    assert "<html lang=\"en\">" in out
    assert "generated by" in out
    assert "pip install contradish" in out
    # timestamp of the form "Month DD, YYYY at HH:MM" is present in the meta line
    assert re.search(r"[A-Z][a-z]+ \d{1,2}, \d{4} at \d{2}:\d{2}", out)
