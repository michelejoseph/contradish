"""
Tests for contradish.printer -- terminal output formatting for `contradish`
CLI runs. No API key or live model call required; findings_from() (called
by print_next_steps) is patched with a hand-written fake so no real
judge/model call ever happens.

Printer functions only print, they never return anything meaningful, so
these tests assert on captured stdout (via capsys) rather than return
values.
"""
import pytest

from contradish.models import TestCase, TestResult, Report, ContradictionPair, RiskLevel
from contradish.printer import (
    _is_tty, _wrap, _strain_label, _cai_label,
    print_start, print_progress, print_step, print_report, print_next_steps,
)


def _tc(input="a test question", name=None, **kw):
    return TestCase(input=input, name=name, **kw)


def _tr(consistency_score=0.9, contradiction_score=0.0, contradictions=None,
        unstable_patterns=None, suggestion=None, skipped=False, n_errors=0,
        tc=None, outputs=None, **kw):
    return TestResult(
        test_case=tc or _tc(),
        paraphrases=["p1", "p2"],
        outputs=outputs if outputs is not None else ["o0", "o1", "o2"],
        consistency_score=consistency_score,
        contradiction_score=contradiction_score,
        risk=RiskLevel.LOW,
        contradictions=contradictions or [],
        unstable_patterns=unstable_patterns or [],
        suggestion=suggestion,
        skipped=skipped,
        n_errors=n_errors,
        **kw,
    )


# ─────────────────────────────────────────────────────────────────────────
# _is_tty
# ─────────────────────────────────────────────────────────────────────────

class _FakeStdout:
    def __init__(self, is_a_tty):
        self._is_a_tty = is_a_tty

    def isatty(self):
        return self._is_a_tty


class _NoIsattyStdout:
    pass


def test_is_tty_false_when_ci_env_var_set(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr("sys.stdout", _FakeStdout(True))
    assert _is_tty() is False


def test_is_tty_true_when_tty_and_no_ci(monkeypatch):
    for v in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "JENKINS_URL", "BUILDKITE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("sys.stdout", _FakeStdout(True))
    assert _is_tty() is True


def test_is_tty_false_when_not_a_tty(monkeypatch):
    for v in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "JENKINS_URL", "BUILDKITE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("sys.stdout", _FakeStdout(False))
    assert _is_tty() is False


def test_is_tty_false_when_stdout_has_no_isatty(monkeypatch):
    for v in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "JENKINS_URL", "BUILDKITE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("sys.stdout", _NoIsattyStdout())
    assert _is_tty() is False


# ─────────────────────────────────────────────────────────────────────────
# _wrap / _strain_label / _cai_label
# ─────────────────────────────────────────────────────────────────────────

def test_wrap_wraps_at_width_with_indent():
    wrapped = _wrap("one two three four five six seven eight", width=15, indent="  ")
    lines = wrapped.split("\n")
    assert all(line.startswith("  ") for line in lines)
    assert all(len(line) <= 15 for line in lines)


def test_strain_label_boundaries():
    assert _strain_label(0.0)    == "stable"
    assert _strain_label(0.20)   == "stable"
    assert _strain_label(0.2001) == "marginal"
    assert _strain_label(0.40)   == "marginal"
    assert _strain_label(0.4001) == "unstable"
    assert _strain_label(1.0)    == "unstable"


def test_cai_label_mirrors_strain_label_in_score_space():
    assert _cai_label(1.0) == "stable"     # strain 0.0
    assert _cai_label(0.6) == "marginal"   # strain 0.4
    assert _cai_label(0.5) == "unstable"   # strain 0.5


# ─────────────────────────────────────────────────────────────────────────
# print_start / print_progress / print_step
# ─────────────────────────────────────────────────────────────────────────

def test_print_start_with_short_preview(capsys):
    print_start("a short prompt")
    out = capsys.readouterr().out
    assert "a short prompt" in out
    assert "..." not in out


def test_print_start_truncates_long_preview(capsys):
    print_start("x" * 100)
    out = capsys.readouterr().out
    assert "..." in out
    assert "x" * 100 not in out  # truncated, full string shouldn't appear


def test_print_start_default_message_when_no_preview(capsys):
    print_start("")
    out = capsys.readouterr().out
    assert "running CAI tests" in out


def test_print_progress(capsys):
    print_progress("doing a thing")
    assert "doing a thing" in capsys.readouterr().out


def test_print_step(capsys):
    print_step("running...", "my case", 2, 5)
    out = capsys.readouterr().out
    assert "[2/5]" in out
    assert "my case" in out


# ─────────────────────────────────────────────────────────────────────────
# print_report
# ─────────────────────────────────────────────────────────────────────────

def test_print_report_all_clean(capsys):
    report = Report(results=[_tr(consistency_score=0.95, tc=_tc(name="case one")),
                              _tr(consistency_score=0.99, tc=_tc(name="case two"))])
    print_report(report)
    out = capsys.readouterr().out
    assert "No CAI failures." in out
    assert "2 of 2 rules stable" in out
    assert "case one" in out and "case two" in out
    assert "stable (strain" in out


def test_print_report_all_clean_with_skipped(capsys):
    report = Report(results=[
        _tr(consistency_score=0.95, tc=_tc(name="ok case")),
        _tr(skipped=True, consistency_score=None, contradiction_score=None,
            n_errors=3, outputs=["o0", "o1", "o2"], tc=_tc(name="skipped case")),
    ])
    print_report(report)
    out = capsys.readouterr().out
    assert "1 of 2 rules stable, 1 skipped (app errors)" in out
    assert "SKIPPED -- 3/3 app calls failed" in out


def test_print_report_singular_rule_wording(capsys):
    report = Report(results=[_tr(consistency_score=0.95)])
    print_report(report)
    assert "1 of 1 rule stable" in capsys.readouterr().out


def test_print_report_header_for_failures(capsys):
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="yes", output_b="no",
                              explanation="direct contradiction", severity="factual")
    report = Report(results=[_tr(consistency_score=0.2, contradiction_score=0.5,
                                  contradictions=[pair], tc=_tc(name="bad case"))])
    print_report(report)
    out = capsys.readouterr().out
    assert "contradish found 1 CAI failure." in out


def test_print_report_multiple_failures_plural_wording(capsys):
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="yes", output_b="no",
                              explanation="conflict", severity="factual")
    report = Report(results=[
        _tr(consistency_score=0.2, contradiction_score=0.5, contradictions=[pair], tc=_tc(name="bad1")),
        _tr(consistency_score=0.1, contradiction_score=0.6, contradictions=[pair], tc=_tc(name="bad2")),
    ])
    print_report(report)
    assert "contradish found 2 CAI failures." in capsys.readouterr().out


def test_print_report_skipped_result_inside_failing_report(capsys):
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="yes", output_b="no",
                              explanation="conflict", severity="factual")
    report = Report(results=[
        _tr(consistency_score=0.1, contradiction_score=0.6, contradictions=[pair], tc=_tc(name="bad")),
        _tr(skipped=True, consistency_score=None, contradiction_score=None,
            n_errors=2, outputs=["o0", "o1"], tc=_tc(name="skip me")),
    ])
    print_report(report)
    out = capsys.readouterr().out
    assert "SKIPPED -- 2/2 app calls failed" in out


def test_print_report_passing_result_inside_failing_report_shown_as_checkmark(capsys):
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="yes", output_b="no",
                              explanation="conflict", severity="factual")
    report = Report(results=[
        _tr(consistency_score=0.1, contradiction_score=0.6, contradictions=[pair], tc=_tc(name="bad")),
        _tr(consistency_score=0.95, contradiction_score=0.0, tc=_tc(name="good case")),
    ])
    print_report(report)
    out = capsys.readouterr().out
    assert "good case" in out
    # good case appears both in the inline pass-through and the bottom summary list
    assert out.count("good case") >= 1


def test_print_report_shows_severity_badge(capsys):
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="yes", output_b="no",
                              explanation="conflict", severity="policy")
    report = Report(results=[_tr(consistency_score=0.1, contradiction_score=0.6,
                                  contradictions=[pair], tc=_tc(name="bad"))])
    out = capsys.readouterr()  # drain
    print_report(report)
    out = capsys.readouterr().out
    assert "[policy]" in out


def test_print_report_hides_severity_badge_when_unknown(capsys):
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="yes", output_b="no",
                              explanation="conflict", severity="unknown")
    report = Report(results=[_tr(consistency_score=0.1, contradiction_score=0.6,
                                  contradictions=[pair], tc=_tc(name="bad"))])
    print_report(report)
    out = capsys.readouterr().out
    assert "[unknown]" not in out


def test_print_report_shows_contradiction_pair_and_truncates_long_outputs(capsys):
    pair = ContradictionPair(
        input_a="asked one way", input_b="asked another way",
        output_a="A" * 200, output_b="B" * 200,
        explanation="the model reversed itself", severity="factual",
    )
    report = Report(results=[_tr(consistency_score=0.1, contradiction_score=0.6,
                                  contradictions=[pair], tc=_tc(name="bad"))])
    print_report(report)
    out = capsys.readouterr().out
    assert "asked one way" in out
    assert "asked another way" in out
    assert "A" * 200 not in out  # truncated
    assert "A" * 107 + "..." in out
    assert "the model reversed itself" in out


def test_print_report_skips_none_explanation(capsys):
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="a", output_b="b",
                              explanation="none", severity="factual")
    report = Report(results=[_tr(consistency_score=0.1, contradiction_score=0.6,
                                  contradictions=[pair], tc=_tc(name="bad"))])
    print_report(report)
    out = capsys.readouterr().out
    # explanation "none" (case-insensitive) is never printed as if it were real content
    assert "\nnone\n" not in out.lower()


def test_print_report_shows_extra_contradiction_count(capsys):
    pair1 = ContradictionPair(input_a="q1", input_b="q2", output_a="a", output_b="b",
                               explanation="e1", severity="factual")
    pair2 = ContradictionPair(input_a="q3", input_b="q4", output_a="c", output_b="d",
                               explanation="e2", severity="factual")
    report = Report(results=[_tr(consistency_score=0.1, contradiction_score=0.6,
                                  contradictions=[pair1, pair2], tc=_tc(name="bad"))])
    print_report(report)
    out = capsys.readouterr().out
    assert "+1 more contradiction" in out


def test_print_report_unstable_patterns_without_contradictions(capsys):
    report = Report(results=[_tr(consistency_score=0.3, contradiction_score=0.0,
                                  unstable_patterns=["some divergence"], tc=_tc(name="bad"))])
    print_report(report)
    out = capsys.readouterr().out
    assert "Inconsistent answers to the same question." in out


def test_print_report_two_patterns_shows_pattern_and_why(capsys):
    report = Report(results=[_tr(consistency_score=0.3, contradiction_score=0.0,
                                  unstable_patterns=["trigger pattern text", "root cause text"],
                                  tc=_tc(name="bad"))])
    print_report(report)
    out = capsys.readouterr().out
    assert "PATTERN" in out
    assert "trigger pattern text" in out
    assert "WHY" in out
    assert "root cause text" in out


def test_print_report_one_pattern_shows_why_only(capsys):
    report = Report(results=[_tr(consistency_score=0.3, contradiction_score=0.0,
                                  unstable_patterns=["just one reason"], tc=_tc(name="bad"))])
    print_report(report)
    out = capsys.readouterr().out
    assert "WHY" in out
    assert "just one reason" in out
    assert "PATTERN" not in out


def test_print_report_shows_fix_suggestion(capsys):
    report = Report(results=[_tr(consistency_score=0.3, contradiction_score=0.0,
                                  suggestion="Always state the 1200mg/day limit explicitly.",
                                  tc=_tc(name="bad"))])
    print_report(report)
    out = capsys.readouterr().out
    assert "FIX" in out
    assert "1200mg/day" in out


def test_print_report_lists_passing_rules_at_bottom(capsys):
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="a", output_b="b",
                              explanation="e", severity="factual")
    report = Report(results=[
        _tr(consistency_score=0.1, contradiction_score=0.6, contradictions=[pair], tc=_tc(name="bad")),
        _tr(consistency_score=0.95, contradiction_score=0.0, tc=_tc(name="clean case")),
    ])
    print_report(report)
    out = capsys.readouterr().out
    assert "clean case" in out


def test_print_report_summary_line(capsys):
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="a", output_b="b",
                              explanation="e", severity="factual")
    report = Report(results=[
        _tr(consistency_score=0.1, contradiction_score=0.6, contradictions=[pair], tc=_tc(name="bad")),
        _tr(consistency_score=0.95, contradiction_score=0.0, tc=_tc(name="clean case")),
    ])
    print_report(report)
    out = capsys.readouterr().out
    assert "1 CAI failure found." in out
    assert "1 rule clean." in out


# ─────────────────────────────────────────────────────────────────────────
# print_next_steps
# ─────────────────────────────────────────────────────────────────────────

class _Finding:
    def __init__(self, headline, detail, cli_hint=None):
        self.headline = headline
        self.detail = detail
        self.cli_hint = cli_hint


def test_print_next_steps_no_findings(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    report = Report(results=[_tr(consistency_score=0.95, tc=_tc(name="ok"))])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "finding" not in out.lower().split("next:")[0].replace("findings", "")  # no findings block header


def test_print_next_steps_shows_findings(monkeypatch, capsys):
    monkeypatch.setattr(
        "contradish.findings.findings_from",
        lambda report: [_Finding("the model capitulates to authority claims", "detail text", cli_hint="try --policy medication")],
    )
    report = Report(results=[_tr(consistency_score=0.95, tc=_tc(name="ok"))])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "contradish finding (1):" in out
    assert "the model capitulates to authority claims" in out
    assert "detail text" in out
    assert "try --policy medication" in out


def test_print_next_steps_findings_plural_and_singular_wording(monkeypatch, capsys):
    monkeypatch.setattr(
        "contradish.findings.findings_from",
        lambda report: [_Finding("h1", "d1"), _Finding("h2", "d2")],
    )
    report = Report(results=[_tr(consistency_score=0.95)])
    print_next_steps(report)
    assert "contradish findings (2):" in capsys.readouterr().out


def test_print_next_steps_survives_findings_exception(monkeypatch, capsys):
    def _boom(report):
        raise RuntimeError("boom")
    monkeypatch.setattr("contradish.findings.findings_from", _boom)
    report = Report(results=[_tr(consistency_score=0.95, tc=_tc(name="ok"))])
    print_next_steps(report)  # must not raise
    out = capsys.readouterr().out
    assert "findings" not in out.split("Next:")[0]


def test_print_next_steps_skips_judgment_block_when_no_scored_cases(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    report = Report(results=[])  # judgment_strain -> None
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "Judgment Strain" not in out


def test_print_next_steps_shows_judgment_headline(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    report = Report(results=[_tr(consistency_score=0.9, tc=_tc(name="ok", equivalence_confidence=1.0))])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "Judgment Strain:" in out
    assert "EQ coverage" in out


def test_print_next_steps_shows_contested_and_truth_and_ambiguous(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    contested_tc = _tc(name="contested", equivalence_confidence=0.6)  # in [0.5, 0.8) -> contested
    truth_tc     = _tc(name="truthy", equivalence_confidence=1.0, canonical_answer="the true answer")
    ambiguous_tc = _tc(name="ambiguous", equivalence_confidence=0.2)  # < 0.5 -> ambiguous
    report = Report(results=[
        _tr(consistency_score=0.9, tc=contested_tc),
        _tr(consistency_score=0.9, tc=truth_tc, truth_score=0.8, truth_strain=0.2),
        _tr(consistency_score=0.9, tc=ambiguous_tc),
    ])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "contested:" in out
    assert "truth:" in out
    assert "excluded as ambiguous: 1 case" in out


def test_print_next_steps_shows_judge_confidence_with_order_sensitive(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    report = Report(results=[
        _tr(consistency_score=0.9, tc=_tc(name="a"), judge_vote_agreement=0.5, judge_order_sensitive=True),
    ])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "judge confidence:" in out
    assert "order-sensitive" in out


def test_print_next_steps_failed_cases_next_step_message(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    pair = ContradictionPair(input_a="q1", input_b="q2", output_a="a", output_b="b",
                              explanation="e", severity="factual")
    report = Report(results=[_tr(consistency_score=0.1, contradiction_score=0.6,
                                  contradictions=[pair], tc=_tc(name="bad"))])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "apply the fix above" in out


def test_print_next_steps_no_failures_next_step_message(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    report = Report(results=[_tr(consistency_score=0.95, tc=_tc(name="ok"))])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "add to CI to catch regressions" in out


def test_print_next_steps_evals_yaml_uses_case_name(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    report = Report(results=[_tr(consistency_score=0.9, tc=_tc(input='has "quotes" in it', name='my "case"'))])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "name:  \"my 'case'\"" in out
    assert "input: \"has 'quotes' in it\"" in out


def test_print_next_steps_evals_yaml_falls_back_to_input_when_name_blank(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    tc = _tc(input="a fairly long question used as the fallback name here")
    tc.name = ""  # force the blank-name branch (TestCase.__post_init__ never leaves it blank itself)
    report = Report(results=[_tr(consistency_score=0.9, tc=tc)])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "a fairly long question used as th" in out  # first 30 chars of the input


def test_print_next_steps_shows_ci_commands(monkeypatch, capsys):
    monkeypatch.setattr("contradish.findings.findings_from", lambda report: [])
    report = Report(results=[_tr(consistency_score=0.95, tc=_tc(name="ok"))])
    print_next_steps(report)
    out = capsys.readouterr().out
    assert "contradish run evals.yaml --app mymodule:my_app" in out
    assert "contradish compare evals.yaml --baseline mymodule:old --candidate mymodule:new" in out
