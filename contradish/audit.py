"""
Audit export for contradish.

One function call. Timestamped compliance document you can hand to legal,
attach to a PR, or drop in a NIST AI RMF review.

Covers NIST AI RMF MAP 1.6, MEASURE 2.5, MANAGE 1.3. EU AI Act Arts 9 and 72. ISO/IEC 42001.

Usage:
    from contradish.audit import to_audit_html

    html = to_audit_html(
        report,
        app_version="prod-v12",
        system_prompt="You are a support agent...",
        evaluator_id="ci-run-456",
    )
    with open("cai-audit-2026-03-25.html", "w") as f:
        f.write(html)
"""

from __future__ import annotations
import html as _html
import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models import Report


def _esc(text: str) -> str:
    if not text:
        return ""
    return _html.escape(str(text))


def _score_color(score: float) -> str:
    if score >= 0.80:
        return "#16a34a"
    if score >= 0.60:
        return "#d97706"
    return "#dc2626"


def _score_label(score: float) -> str:
    if score >= 0.80:
        return "Stable"
    if score >= 0.60:
        return "Marginal"
    return "Unstable"


def to_audit_html(
    report: "Report",
    *,
    app_version:   Optional[str] = None,
    system_prompt: Optional[str] = None,
    evaluator_id:  Optional[str] = None,
    policy_name:   Optional[str] = None,
    notes:         Optional[str] = None,
) -> str:
    """
    Generate a timestamped compliance audit document from a Report.

    Includes evaluation config, risk assessment, all CAI failures with contradiction
    pairs, full test case results, NIST AI RMF alignment, and optional system prompt appendix.

    Args:
        report:        A contradish Report (from suite.run()).
        app_version:   Version string for the app under test (e.g. "prod-v12").
        system_prompt: System prompt used during the test run (optional).
        evaluator_id:  CI run ID, evaluator name, or other identifier.
        policy_name:   Policy pack used (e.g. "ecommerce").
        notes:         Free-text notes to include in the document.

    Returns:
        A complete self-contained HTML document as a string.

    Example:
        html = to_audit_html(report, app_version="v2.1", evaluator_id="pr-567")
        open("audit.html", "w").write(html)
    """
    from . import __version__

    ts_full   = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    ts_date   = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    doc_id    = f"CONTRADISH-{ts_date}-{(evaluator_id or 'manual').upper()[:12]}"

    total   = len(report.results)
    passed  = len(report.passed)
    failed  = len(report.failed)
    agg     = report.cai_score or 0.0
    agg_color = _score_color(agg)
    agg_label = _score_label(agg)

    # ── Evaluation config table ────────────────────────────────────────────────
    config_rows = ""
    config_rows += f"<tr><td>Document ID</td><td><code>{_esc(doc_id)}</code></td></tr>"
    config_rows += f"<tr><td>Generated</td><td>{_esc(ts_full)}</td></tr>"
    config_rows += f"<tr><td>contradish version</td><td>{_esc(__version__)}</td></tr>"
    if app_version:
        config_rows += f"<tr><td>App version</td><td>{_esc(app_version)}</td></tr>"
    if policy_name:
        config_rows += f"<tr><td>Policy pack</td><td>{_esc(policy_name)}</td></tr>"
    if evaluator_id:
        config_rows += f"<tr><td>Evaluator / Run ID</td><td>{_esc(evaluator_id)}</td></tr>"

    # ── Risk assessment ────────────────────────────────────────────────────────
    if agg >= 0.80:
        risk_level = "LOW"
        risk_color = "#16a34a"
        risk_note  = "Consistency is within acceptable bounds. No immediate remediation required."
    elif agg >= 0.60:
        risk_level = "MEDIUM"
        risk_color = "#d97706"
        risk_note  = "Marginal consistency. Review flagged rules before deploying to high-risk contexts."
    else:
        risk_level = "HIGH"
        risk_color = "#dc2626"
        risk_note  = "Significant inconsistency detected. Remediation required before production deployment."

    # ── Failure detail rows ────────────────────────────────────────────────────
    failure_rows = ""
    for result in report.failed:
        rule_score = result.cai_score or 0.0
        rule_color = _score_color(rule_score)
        failure_rows += f"""
<tr class="failure-row">
  <td>{_esc(result.test_case.name)}</td>
  <td style="color:{rule_color};font-weight:600">{rule_score:.2f}</td>
  <td>{_esc(result.risk.value.upper())}</td>
  <td>{len(result.contradictions)}</td>
  <td>{_esc(result.suggestion or "")}</td>
</tr>"""
        for pair in result.contradictions:
            failure_rows += f"""
<tr class="pair-row">
  <td colspan="5">
    <div class="pair-block">
      <div class="pair-line"><span class="pair-label">Q1</span> {_esc(pair.input_a)}</div>
      <div class="pair-line"><span class="pair-label resp-a">A1</span> {_esc(pair.output_a[:300] + ("..." if len(pair.output_a) > 300 else ""))}</div>
      <div class="pair-line"><span class="pair-label">Q2</span> {_esc(pair.input_b)}</div>
      <div class="pair-line"><span class="pair-label resp-b">A2</span> {_esc(pair.output_b[:300] + ("..." if len(pair.output_b) > 300 else ""))}</div>
      <div class="pair-explanation">{_esc(pair.explanation)}</div>
    </div>
  </td>
</tr>"""

    # ── Full test case table ───────────────────────────────────────────────────
    all_case_rows = ""
    for result in sorted(report.results, key=lambda r: r.cai_score or 0.0):
        sc     = result.cai_score or 0.0
        color  = _score_color(sc)
        status = "FAIL" if not result.passed(report.thresholds) else "PASS"
        status_color = "#dc2626" if status == "FAIL" else "#16a34a"
        all_case_rows += f"""
<tr>
  <td>{_esc(result.test_case.name)}</td>
  <td style="font-family:monospace">{_esc(result.test_case.input[:80] + ("..." if len(result.test_case.input) > 80 else ""))}</td>
  <td style="color:{color};font-weight:600">{sc:.2f}</td>
  <td style="color:{status_color};font-weight:600">{status}</td>
</tr>"""

    # ── System prompt appendix ─────────────────────────────────────────────────
    prompt_section = ""
    if system_prompt:
        prompt_section = f"""
<section>
  <h2>Appendix A: System Prompt Under Test</h2>
  <p class="meta-note">Exact system prompt used during this evaluation run.</p>
  <pre class="prompt-pre">{_esc(system_prompt)}</pre>
</section>"""

    # ── Notes section ─────────────────────────────────────────────────────────
    notes_section = ""
    if notes:
        notes_section = f"""
<section>
  <h2>Evaluator Notes</h2>
  <p>{_esc(notes)}</p>
</section>"""

    # ── Failures section heading ───────────────────────────────────────────────
    failures_section = ""
    if failed > 0:
        failures_section = f"""
<section>
  <h2>CAI Failures</h2>
  <p class="meta-note">Rules that produced contradictory responses across semantically equivalent inputs.</p>
  <table>
    <thead>
      <tr>
        <th>Rule</th>
        <th>CAI Score</th>
        <th>Risk</th>
        <th>Contradictions</th>
        <th>Suggested Fix</th>
      </tr>
    </thead>
    <tbody>
      {failure_rows}
    </tbody>
  </table>
</section>"""
    else:
        failures_section = """
<section>
  <h2>CAI Failures</h2>
  <p class="pass-note">No CAI failures detected. All rules produced consistent responses.</p>
</section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CAI Audit Report &mdash; {_esc(doc_id)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{
      margin: 0; padding: 0;
      font-family: 'Georgia', serif;
      font-size: 14px;
      line-height: 1.7;
      color: #1f2937;
      background: #fff;
    }}
    .page {{
      max-width: 860px;
      margin: 0 auto;
      padding: 48px 40px 80px;
    }}

    /* Header */
    .doc-header {{
      border-bottom: 2px solid #1f2937;
      padding-bottom: 20px;
      margin-bottom: 32px;
    }}
    .doc-brand {{
      font-family: monospace;
      font-size: 12px;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 8px;
    }}
    .doc-title {{
      font-size: 22px;
      font-weight: 700;
      margin: 0 0 4px;
      letter-spacing: -0.3px;
    }}
    .doc-id {{
      font-family: monospace;
      font-size: 12px;
      color: #6b7280;
    }}

    /* Sections */
    section {{
      margin-bottom: 40px;
    }}
    h2 {{
      font-size: 15px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 6px;
      margin: 0 0 16px;
      color: #374151;
    }}
    p {{ margin: 0 0 12px; }}
    .meta-note {{ font-size: 13px; color: #6b7280; margin-bottom: 14px; }}
    .pass-note {{ color: #16a34a; font-weight: 600; }}

    /* Summary grid */
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-box {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }}
    .stat-value {{
      font-family: monospace;
      font-size: 28px;
      font-weight: 700;
      line-height: 1;
      display: block;
      margin-bottom: 4px;
    }}
    .stat-label {{
      font-size: 11px;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .risk-box {{
      border: 1px solid;
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 16px;
    }}
    .risk-level {{
      font-weight: 700;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .risk-note {{ font-size: 13px; margin: 4px 0 0; }}

    /* Config table */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th {{
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      padding: 8px 12px;
      text-align: left;
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #6b7280;
    }}
    td {{
      border: 1px solid #e5e7eb;
      padding: 8px 12px;
      vertical-align: top;
    }}
    tr:nth-child(even) td {{ background: #fafafa; }}
    code {{
      font-family: monospace;
      font-size: 12px;
      background: #f3f4f6;
      padding: 1px 5px;
      border-radius: 3px;
    }}

    /* Failure rows */
    .failure-row td {{ background: #fef2f2; font-weight: 500; }}
    .pair-row td {{ padding: 0; border-top: none; }}
    .pair-block {{
      padding: 12px 16px 16px 32px;
      border-left: 3px solid #fca5a5;
      margin: 0 12px 8px;
      font-size: 13px;
    }}
    .pair-line {{ margin-bottom: 6px; display: flex; gap: 10px; }}
    .pair-label {{
      font-family: monospace;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      color: #9ca3af;
      flex-shrink: 0;
      width: 24px;
      padding-top: 2px;
    }}
    .resp-a {{ color: #1f2937; }}
    .resp-b {{ color: #dc2626; }}
    .pair-explanation {{
      margin-top: 8px;
      font-style: italic;
      color: #6b7280;
      font-size: 12px;
      padding-left: 34px;
    }}

    /* NIST section */
    .nist-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}
    .nist-cell {{
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 14px;
    }}
    .nist-tag {{
      font-family: monospace;
      font-size: 11px;
      font-weight: 700;
      color: #6b7280;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .nist-cell p {{
      font-size: 12px;
      color: #374151;
      margin: 0;
      line-height: 1.55;
    }}

    /* Prompt appendix */
    .prompt-pre {{
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 16px;
      font-family: monospace;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      color: #374151;
      line-height: 1.6;
    }}

    /* Footer */
    .doc-footer {{
      margin-top: 56px;
      padding-top: 20px;
      border-top: 1px solid #e5e7eb;
      font-size: 12px;
      color: #9ca3af;
      display: flex;
      justify-content: space-between;
    }}

    @media print {{
      .page {{ padding: 24px 20px; }}
      .summary-grid {{ grid-template-columns: repeat(4, 1fr); }}
    }}
  </style>
</head>
<body>
<div class="page">

  <div class="doc-header">
    <div class="doc-brand">contradish &mdash; CAI Audit Report</div>
    <h1 class="doc-title">Consistency Evaluation Record</h1>
    <div class="doc-id">{_esc(doc_id)}</div>
  </div>

  <section>
    <h2>Evaluation Summary</h2>
    <div class="summary-grid">
      <div class="stat-box">
        <span class="stat-value" style="color:{agg_color}">{agg:.2f}</span>
        <span class="stat-label">CAI Score</span>
      </div>
      <div class="stat-box">
        <span class="stat-value">{total}</span>
        <span class="stat-label">Rules Tested</span>
      </div>
      <div class="stat-box">
        <span class="stat-value" style="color:#16a34a">{passed}</span>
        <span class="stat-label">Passed</span>
      </div>
      <div class="stat-box">
        <span class="stat-value" style="color:#dc2626">{failed}</span>
        <span class="stat-label">Failed</span>
      </div>
    </div>

    <div class="risk-box" style="border-color:{risk_color};background:{'#fef2f2' if risk_level == 'HIGH' else '#fffbeb' if risk_level == 'MEDIUM' else '#f0fdf4'}">
      <div class="risk-level" style="color:{risk_color}">Risk Level: {risk_level}</div>
      <p class="risk-note">{_esc(risk_note)}</p>
    </div>

    <p style="font-size:13px;color:#374151">
      CAI (Consistency-Across-Inputs) score measures semantic invariance:
      how consistently the application responds to paraphrased versions of the same question.
      Score range 0&ndash;1. Stable: 0.80+. Marginal: 0.60&ndash;0.79. Unstable: &lt; 0.60.
    </p>
  </section>

  <section>
    <h2>Evaluation Configuration</h2>
    <table>
      <tbody>
        {config_rows}
      </tbody>
    </table>
  </section>

  {failures_section}

  <section>
    <h2>All Test Cases</h2>
    <p class="meta-note">Complete result set for this evaluation run.</p>
    <table>
      <thead>
        <tr>
          <th>Rule</th>
          <th>Input</th>
          <th>CAI Score</th>
          <th>Result</th>
        </tr>
      </thead>
      <tbody>
        {all_case_rows}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Regulatory Alignment</h2>
    <p class="meta-note">
      This evaluation record supports compliance with the following frameworks.
      Consult your legal and compliance teams for jurisdiction-specific requirements.
    </p>
    <div class="nist-grid">
      <div class="nist-cell">
        <div class="nist-tag">NIST AI RMF &mdash; MAP 1.6</div>
        <p>AI risk identification: contradictions in policy-facing outputs constitute a category of AI risk that may result in user harm, regulatory exposure, or brand damage.</p>
      </div>
      <div class="nist-cell">
        <div class="nist-tag">NIST AI RMF &mdash; MEASURE 2.5</div>
        <p>AI risk measurement: CAI score quantifies semantic inconsistency across equivalent inputs. This report provides a reproducible, versioned measurement artifact.</p>
      </div>
      <div class="nist-cell">
        <div class="nist-tag">NIST AI RMF &mdash; MANAGE 1.3</div>
        <p>AI risk response: suggested fixes and prompt repair outputs provide documented remediation actions for each identified failure.</p>
      </div>
      <div class="nist-cell">
        <div class="nist-tag">EU AI Act &mdash; Article 9</div>
        <p>Risk management system: this evaluation supports ongoing monitoring of high-risk AI system behavior in customer-facing and policy-bound contexts.</p>
      </div>
      <div class="nist-cell">
        <div class="nist-tag">EU AI Act &mdash; Article 72</div>
        <p>Technical documentation: this record documents evaluation methodology, test inputs, outputs, and scoring criteria as required for technical review.</p>
      </div>
      <div class="nist-cell">
        <div class="nist-tag">ISO/IEC 42001</div>
        <p>AI management system: contradish evaluations produce audit-ready records of AI system testing that support continual improvement requirements.</p>
      </div>
    </div>
  </section>

  {notes_section}
  {prompt_section}

  <div class="doc-footer">
    <span>Generated by contradish v{_esc(__version__)} &mdash; contradish.com</span>
    <span>{_esc(ts_full)}</span>
  </div>

</div>
</body>
</html>"""
