"""
contradish.report: render a CommitmentLedger into a shareable, verifiable audit.

This is the seal. A team runs contradish against their model over time, then
publishes the report to customers or regulators as evidence of how the model
behaved. The report carries the ledger fingerprint (the head hash), so anyone
holding the underlying ledger can re-verify that the record was not altered.

The audit attests to what the model said over time, and to the integrity of the
record. It does not assert the truth of any single statement.

    from contradish import CommitmentLedger, audit_report_html
    html = audit_report_html(ledger, model="gpt-4o")
    open("audit.html", "w").write(html)
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Optional


def _fmt_ts(ts: Optional[float]) -> str:
    if not ts:
        return "n/a"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "n/a"


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;color:#111;background:#fff;line-height:1.6;font-size:15px;-webkit-font-smoothing:antialiased}
.wrap{max-width:760px;margin:0 auto;padding:48px 24px 64px}
.eyebrow{font-family:Menlo,Monaco,Consolas,monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#999}
h1{font-size:28px;font-weight:600;letter-spacing:-.4px;margin:10px 0 4px}
.meta{font-size:13px;color:#666;font-family:Menlo,Monaco,Consolas,monospace;margin-bottom:24px}
.verdict{display:inline-block;font-family:Menlo,Monaco,Consolas,monospace;font-size:13px;font-weight:600;letter-spacing:.06em;padding:8px 14px;border-radius:6px;margin-bottom:28px}
.verdict.ok{color:#16a34a;background:#f0fdf4;border:1px solid #bbf7d0}
.verdict.bad{color:#dc2626;background:#fef2f2;border:1px solid #fecaca}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#e5e5e5;border:1px solid #e5e5e5;border-radius:8px;overflow:hidden;margin-bottom:28px}
.stat{background:#fff;padding:16px 18px}
.stat .k{font-size:12px;color:#666}
.stat .v{font-size:20px;font-weight:600;font-family:Menlo,Monaco,Consolas,monospace;margin-top:4px}
.seal{border:1px solid #e5e5e5;border-radius:8px;padding:18px 20px;margin-bottom:28px;background:#fafafa}
.seal .label{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#999;margin-bottom:8px}
.seal .hash{font-family:Menlo,Monaco,Consolas,monospace;font-size:13px;color:#111;word-break:break-all}
.seal .note{font-size:12.5px;color:#666;margin-top:8px}
.tl-label{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#999;margin-bottom:12px}
.tl{border:1px solid #e5e5e5;border-radius:8px;overflow:hidden}
.row{display:grid;grid-template-columns:128px 1fr;border-bottom:1px solid #e5e5e5}
.row:last-child{border-bottom:none}
.row .when{padding:12px 14px;font-family:Menlo,Monaco,Consolas,monospace;font-size:12px;color:#999;border-right:1px solid #e5e5e5}
.row .what{padding:12px 16px;font-size:14px;color:#111}
.row.contra .what{color:#dc2626}
.tag{font-family:Menlo,Monaco,Consolas,monospace;font-size:11px;color:#666}
.foot{font-size:12.5px;color:#999;line-height:1.6;margin-top:28px;border-top:1px solid #e5e5e5;padding-top:18px}
"""


def audit_report_html(ledger, model: Optional[str] = None, title: Optional[str] = None) -> str:
    """Render a CommitmentLedger as a self-contained, verifiable HTML audit."""
    s = ledger.audit_summary()
    entries = ledger.timeline()
    verified = bool(s.get("verified"))
    head = _esc(s.get("head"))
    model_line = _esc(model) if model else "model audit"
    page_title = _esc(title) if title else f"contradish audit: {model_line}"

    verdict = (
        '<div class="verdict ok">RECORD INTACT, INDEPENDENTLY VERIFIABLE</div>'
        if verified else
        '<div class="verdict bad">RECORD ALTERED, VERIFICATION FAILED</div>'
    )

    rate = s.get("contradiction_rate", 0.0)
    stats = "".join([
        f'<div class="stat"><div class="k">commitments observed</div><div class="v">{s.get("commitments", 0)}</div></div>',
        f'<div class="stat"><div class="k">contradictions</div><div class="v">{s.get("contradictions", 0)}</div></div>',
        f'<div class="stat"><div class="k">contradiction rate</div><div class="v">{rate}</div></div>',
        f'<div class="stat"><div class="k">window</div><div class="v" style="font-size:13px">{_fmt_ts(s.get("first_at"))} to {_fmt_ts(s.get("last_at"))}</div></div>',
    ])

    rows = []
    for e in entries[:200]:
        p = e.payload or {}
        if e.type == "contradiction":
            new_c, prior_c = _esc(p.get("new_claim")), _esc(p.get("prior_claim"))
            conf = p.get("confidence")
            conf_s = f' <span class="tag">conf {conf}</span>' if conf is not None else ""
            what = f'<strong>contradiction</strong>{conf_s}<br>now: {new_c}<br>before: {prior_c}'
            cls = "row contra"
        else:
            claim, topic, kind = _esc(p.get("claim")), _esc(p.get("topic")), _esc(p.get("kind") or "durable")
            what = f'committed: {claim} <span class="tag">[{topic} / {kind}]</span>'
            cls = "row"
        rows.append(f'<div class="{cls}"><div class="when">{_fmt_ts(e.at)}</div><div class="what">{what}</div></div>')
    timeline = "".join(rows) if rows else '<div class="row"><div class="when">n/a</div><div class="what">No entries recorded yet.</div></div>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{page_title}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<div class="eyebrow">contradish audit</div>
<h1>{model_line}</h1>
<div class="meta">generated {_fmt_ts(datetime.now().timestamp())} &middot; {s.get("entries", 0)} entries on the record</div>
{verdict}
<div class="stats">{stats}</div>
<div class="seal">
  <div class="label">ledger fingerprint</div>
  <div class="hash">{head}</div>
  <div class="note">Publish this hash to anchor the record. Anyone holding the ledger can confirm it still hashes to this value, which proves the log was not rewritten after the fact.</div>
</div>
<div class="tl-label">timeline</div>
<div class="tl">{timeline}</div>
<div class="foot">This audit attests to what the model said over time and to the integrity of the record. It does not assert the truth of any single statement. Produced by contradish.com</div>
</div></body></html>"""


__all__ = ["audit_report_html"]
