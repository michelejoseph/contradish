"""
HTML report generator for contradish.

Produces a self-contained, shareable HTML file from a Report.
No external dependencies at render time. Everything is inlined.

Usage:
    from contradish.reporter import to_html

    html = to_html(report)
    with open("contradish-report.html", "w") as f:
        f.write(html)

Or from the CLI:
    contradish --policy ecommerce --app mymodule:my_app --report
    contradish "My system prompt" --report contradish-report.html
"""

from __future__ import annotations
import html as _html
import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models import Report


def _esc(text: str) -> str:
    """HTML-escape a string."""
    if not text:
        return ""
    return _html.escape(str(text))


def _score_color(score: float) -> str:
    if score >= 0.80:
        return "#16a34a"   # green
    if score >= 0.60:
        return "#d97706"   # amber
    return "#dc2626"       # red


def _score_label(score: float) -> str:
    if score >= 0.80:
        return "stable"
    if score >= 0.60:
        return "marginal"
    return "unstable"


def _score_bg(score: float) -> str:
    if score >= 0.80:
        return "#f0fdf4"
    if score >= 0.60:
        return "#fffbeb"
    return "#fef2f2"


def _score_border(score: float) -> str:
    if score >= 0.80:
        return "#bbf7d0"
    if score >= 0.60:
        return "#fde68a"
    return "#fecaca"


def _passing_card(result) -> str:
    score = result.consistency_score or 0
    color = _score_color(score)
    return f"""
<div class="rule-card rule-pass">
  <div class="rule-header">
    <span class="rule-name">{_esc(result.test_case.name)}</span>
    <span class="rule-score" style="color:{color}">
      {score:.2f} <span class="rule-label">stable</span>
    </span>
  </div>
</div>"""


def _failing_card(result) -> str:
    score = result.consistency_score or 0
    color = _score_color(score)
    label = _score_label(score)
    bg    = _score_bg(score)
    border = _score_border(score)

    contradiction_html = ""
    if result.contradictions:
        pair  = result.contradictions[0]
        q_a   = _esc(pair.input_a.strip())
        a_a   = _esc(pair.output_a.strip()[:200] + ("..." if len(pair.output_a.strip()) > 200 else ""))
        q_b   = _esc(pair.input_b.strip())
        a_b   = _esc(pair.output_b.strip()[:200] + ("..." if len(pair.output_b.strip()) > 200 else ""))
        extra = len(result.contradictions) - 1
        extra_html = f'<p class="extra-count">+ {extra} more contradiction{"s" if extra > 1 else ""} on this rule</p>' if extra else ""

        contradiction_html = f"""
<div class="contradiction-block">
  <div class="convo-row">
    <span class="convo-label">asked</span>
    <span class="convo-text">&ldquo;{q_a}&rdquo;</span>
  </div>
  <div class="convo-row">
    <span class="convo-label">said</span>
    <span class="convo-text response-a">&ldquo;{a_a}&rdquo;</span>
  </div>
  <div class="convo-divider">same intent, different phrasing</div>
  <div class="convo-row">
    <span class="convo-label">asked</span>
    <span class="convo-text">&ldquo;{q_b}&rdquo;</span>
  </div>
  <div class="convo-row">
    <span class="convo-label">said</span>
    <span class="convo-text response-b">&ldquo;{a_b}&rdquo;</span>
  </div>
  {extra_html}
  <p class="contradiction-verdict">Both answers reached real users. They can&rsquo;t both be right.</p>
</div>"""

    why_html = ""
    if result.unstable_patterns:
        pat = _esc(result.unstable_patterns[0])
        why_html = f"""
<div class="why-block">
  <span class="block-label why-label">WHY</span>
  <p>{pat}</p>
</div>"""

    fix_html = ""
    if result.suggestion:
        suggestion = _esc(result.suggestion)
        fix_html = f"""
<div class="fix-block">
  <span class="block-label fix-label">FIX</span>
  <p class="fix-intro">add this line to your system prompt:</p>
  <div class="fix-quote">&ldquo;{suggestion}&rdquo;</div>
</div>"""

    return f"""
<div class="rule-card rule-fail" style="background:{bg};border-color:{border}">
  <div class="rule-header">
    <span class="rule-name fail-name">{_esc(result.test_case.name)}</span>
    <span class="rule-score" style="color:{color}">
      {score:.2f} <span class="rule-label">{label}</span>
    </span>
  </div>
  {contradiction_html}
  {why_html}
  {fix_html}
</div>"""


def to_html(
    report,
    title:       Optional[str] = None,
    policy_name: Optional[str] = None,
    version:     str           = "0.7.0",
) -> str:
    """
    Generate a self-contained HTML report from a Report object.

    Args:
        report:      A contradish Report.
        title:       Optional page title override.
        policy_name: Optional policy pack name to display in the header.
        version:     contradish version string.

    Returns:
        A complete HTML document as a string.
    """
    from . import __version__
    version = __version__

    total   = len(report.results)
    passed  = len(report.passed)
    failed  = len(report.failed)
    agg     = report.cai_score if hasattr(report, "cai_score") else 0.0
    ts      = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")

    agg_color  = _score_color(agg)
    agg_label  = _score_label(agg)

    page_title = title or "contradish CAI Report"
    source_tag = f"policy pack: {policy_name}" if policy_name else "CAI report"

    # Summary line
    if failed == 0:
        summary_html = f'<p class="summary-line pass-summary">No CAI failures. All {total} rule{"s" if total != 1 else ""} stable.</p>'
    else:
        fail_word = "failure" if failed == 1 else "failures"
        pass_word = "rule" if passed == 1 else "rules"
        summary_html = f'<p class="summary-line fail-summary">{failed} CAI {fail_word} found &middot; {passed} {pass_word} clean</p>'

    # Rule cards: failures first, then passes
    failing_cards = "".join(_failing_card(r) for r in report.results if not r.passed(report.thresholds))
    passing_cards = "".join(_passing_card(r) for r in report.results if r.passed(report.thresholds))

    # Aggregate score display
    agg_display = f"{agg:.2f}" if agg else "n/a"

    # Passing rules section
    passing_rows = "".join(
        f'<div class="passing-rule">'
        f'<span class="passing-name">{_esc(r.test_case.name)}</span>'
        f'<span class="passing-score">{(r.consistency_score or 0):.2f} stable</span>'
        f'</div>'
        for r in report.results if r.passed(report.thresholds)
    )
    passing_section_html = (
        f'<div class="passing-section">'
        f'<p class="section-label">passing rules</p>'
        f'{passing_rows}'
        f'</div>'
    ) if passing_rows else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_esc(page_title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{
      margin: 0; padding: 0;
      background: #f5f6f7;
      font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, -apple-system, sans-serif;
      color: #1f2937;
      -webkit-font-smoothing: antialiased;
    }}
    a {{ color: inherit; text-decoration: none; }}
    .container {{ max-width: 780px; margin: 0 auto; padding: 0 24px 64px; }}

    /* ── Header ── */
    .report-header {{
      padding: 40px 0 32px;
      border-bottom: 1px solid #e5e7eb;
      margin-bottom: 32px;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
    }}
    .header-left {{ flex: 1; }}
    .brand {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 13px;
      color: #6b7280;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
    }}
    .brand a {{ color: #6b7280; }}
    .brand a:hover {{ color: #1f2937; }}
    .report-title {{
      font-size: 22px;
      font-weight: 600;
      letter-spacing: -0.4px;
      margin: 0 0 6px;
      color: #1f2937;
    }}
    .report-meta {{
      font-size: 13px;
      color: #9ca3af;
      font-family: 'IBM Plex Mono', monospace;
    }}

    /* ── Score ── */
    .score-block {{
      text-align: right;
      flex-shrink: 0;
    }}
    .score-number {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 48px;
      font-weight: 500;
      line-height: 1;
      letter-spacing: -2px;
    }}
    .score-sublabel {{
      font-size: 11px;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-top: 4px;
      font-family: 'IBM Plex Mono', monospace;
    }}

    /* ── Summary ── */
    .summary-line {{
      font-size: 15px;
      margin: 0 0 28px;
      font-weight: 500;
    }}
    .fail-summary {{ color: #dc2626; }}
    .pass-summary {{ color: #16a34a; }}

    /* ── Rule cards ── */
    .rules-section {{ margin-bottom: 8px; }}
    .section-label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 12px;
    }}
    .rule-card {{
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 12px;
      background: #fff;
    }}
    .rule-pass {{
      background: #fff;
      border-color: #e5e7eb;
    }}
    .rule-fail {{
      border-width: 1px;
    }}
    .rule-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
    }}
    .rule-name {{
      font-size: 15px;
      font-weight: 600;
      color: #1f2937;
    }}
    .fail-name {{ color: #b91c1c; }}
    .rule-score {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 15px;
      font-weight: 500;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .rule-label {{
      font-size: 11px;
      font-weight: 400;
      opacity: 0.75;
      margin-left: 4px;
    }}

    /* ── Contradiction block ── */
    .contradiction-block {{
      margin-top: 18px;
      border-left: 2px solid #fca5a5;
      padding-left: 16px;
    }}
    .convo-row {{
      display: flex;
      gap: 12px;
      margin-bottom: 6px;
      align-items: flex-start;
    }}
    .convo-label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding-top: 2px;
      width: 40px;
      flex-shrink: 0;
    }}
    .convo-text {{
      font-size: 14px;
      color: #374151;
      line-height: 1.6;
    }}
    .response-a {{ color: #1f2937; }}
    .response-b {{ color: #dc2626; font-style: italic; }}
    .convo-divider {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      color: #9ca3af;
      margin: 10px 0;
      letter-spacing: 0.04em;
    }}
    .extra-count {{
      font-size: 12px;
      color: #9ca3af;
      margin: 8px 0 0;
    }}
    .contradiction-verdict {{
      font-size: 13px;
      font-weight: 600;
      color: #b91c1c;
      margin: 12px 0 0;
    }}

    /* ── Why / Fix blocks ── */
    .why-block, .fix-block {{
      margin-top: 16px;
      display: flex;
      gap: 12px;
      align-items: flex-start;
    }}
    .block-label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding-top: 2px;
      flex-shrink: 0;
      width: 36px;
    }}
    .why-label {{ color: #d97706; }}
    .fix-label {{ color: #0284c7; }}
    .why-block p, .fix-block p {{
      font-size: 14px;
      color: #374151;
      line-height: 1.65;
      margin: 0;
    }}
    .fix-block > div {{ flex: 1; }}
    .fix-intro {{
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 8px !important;
    }}
    .fix-quote {{
      background: #f0f9ff;
      border: 1px solid #bae6fd;
      border-radius: 6px;
      padding: 12px 14px;
      font-size: 13px;
      color: #0c4a6e;
      line-height: 1.65;
      font-style: italic;
    }}

    /* ── Passing rules ── */
    .passing-section {{ margin-top: 32px; }}
    .passing-rule {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      margin-bottom: 6px;
    }}
    .passing-name {{ font-size: 14px; color: #374151; }}
    .passing-score {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 13px;
      color: #16a34a;
    }}

    /* ── Footer ── */
    .report-footer {{
      margin-top: 48px;
      padding-top: 24px;
      border-top: 1px solid #e5e7eb;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}
    .footer-brand {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 12px;
      color: #9ca3af;
    }}
    .footer-brand a {{ color: #9ca3af; border-bottom: 1px solid #e5e7eb; }}
    .footer-brand a:hover {{ color: #1f2937; border-color: #9ca3af; }}
    .footer-cta {{
      font-size: 12px;
      color: #9ca3af;
      font-family: 'IBM Plex Mono', monospace;
    }}

    @media (max-width: 600px) {{
      .report-header {{ flex-direction: column-reverse; gap: 16px; }}
      .score-block {{ text-align: left; }}
      .score-number {{ font-size: 36px; }}
    }}
  </style>
</head>
<body>
<div class="container">

  <div class="report-header">
    <div class="header-left">
      <p class="brand"><a href="https://contradish.com">contradish</a></p>
      <h1 class="report-title">CAI Report</h1>
      <p class="report-meta">{_esc(source_tag)} &middot; {_esc(ts)}</p>
    </div>
    <div class="score-block">
      <div class="score-number" style="color:{agg_color}">{agg_display}</div>
      <div class="score-sublabel">CAI score</div>
    </div>
  </div>

  {summary_html}

  {"" if not failing_cards else f'<div class="rules-section">{failing_cards}</div>'}

  {passing_section_html}

  <div class="report-footer">
    <span class="footer-brand">generated by <a href="https://contradish.com">contradish</a> v{_esc(version)}</span>
    <span class="footer-cta">pip install contradish</span>
  </div>

</div>
</body>
</html>"""
