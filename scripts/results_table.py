"""
Print the CAI-Bench results table from the committed result files.

Every row is one file in results/. A score is shown only with what it was
computed on (domains, judge), because scores over different domain sets or
judges are not comparable. Run from the repository root:

    python scripts/results_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rows():
    out = []
    for p in sorted((ROOT / "results").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if "avg_cai_strain" not in d:
            continue
        domains = d.get("results") or {}
        judge = d.get("judge_model") or "not recorded"
        if d.get("judge_provider") and d.get("judge_model"):
            judge = f"{d['judge_provider']}:{d['judge_model']}"
        out.append({
            "file": p.name, "model": d.get("model"), "date": d.get("date"),
            "domains": len(domains), "judge": judge,
            "strain": d.get("avg_cai_strain"),
        })
    return out


def markdown() -> str:
    lines = ["| model | run date | domains | judge | CAI Strain | file |", "|---|---|---|---|---|---|"]
    for r in rows():
        s = "incomplete" if r["strain"] is None else f"{r['strain']:.3f}"
        lines.append(f"| {r['model']} | {r['date']} | {r['domains']} of 20 | {r['judge']} | {s} | "
                     f"[{r['file']}](results/{r['file']}) |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown())
