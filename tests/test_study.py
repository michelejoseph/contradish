"""Tests for contradish.study: the real-model study runner. No network."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from contradish import study as S

ROOT = Path(__file__).resolve().parent.parent


def test_parse_model_keeps_colons_in_model_names():
    s = S.parse_model("ollama:llama3.1:8b")
    assert (s.provider, s.model, s.id) == ("ollama", "llama3.1:8b", "ollama:llama3.1:8b")
    with pytest.raises(ValueError):
        S.parse_model("nonsense")
    with pytest.raises(ValueError):
        S.parse_model("madeup:model")


def test_anthropic_payload_joins_system_and_alternates_roles():
    msgs = [{"role": "system", "content": "policy"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "update"},
            {"role": "user", "content": "q2"},
            {"role": "user", "content": "q3"}]
    body = S._anthropic_payload(msgs, "m", 100, None)
    assert body["system"] == "policy\n\nupdate"
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert body["messages"][-1]["content"] == "q2\n\nq3"
    assert "temperature" not in body


def test_authority_suites_are_built_from_the_derivation_engine():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_authority_suites.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    suite = S.load_authority_suites()
    assert len(suite) == 28
    for t in suite:
        assert not t.lint_errors(), (t.id, t.lint_errors())
        for c in t.cases:
            if c.authority is False:
                assert c.after == c.before


def test_demo_scripted_agents_score_as_designed(tmp_path):
    study = S.run_demo(str(tmp_path))
    by = {r["model"]: r["authority"] for r in study["results"]}
    for delivery in ("fresh", "in_conversation"):
        f, r, c = (by[f"scripted:{k}"][delivery] for k in ("faithful", "rigid", "credulous"))
        assert f["legitimate_change"]["value"] == 1 and f["illegitimate_resisted"]["value"] == 1
        assert f["faithful_transitions"]["value"] == 1
        assert r["legitimate_change"]["value"] == 0 and r["illegitimate_resisted"]["value"] == 1
        assert c["legitimate_change"]["value"] == 1 and c["illegitimate_resisted"]["value"] == 0
        assert f["legitimate_change"]["n"] == 12 and f["illegitimate_resisted"]["n"] == 30
    assert (tmp_path / "REPORT.md").read_text().startswith("# contradish study report")
    assert json.loads((tmp_path / "results.json").read_text())["schema_version"] == S.STUDY_SCHEMA_VERSION


def test_call_log_counts_occurrences_resumes_and_refuses_offline_misses(tmp_path):
    spec = S.ModelSpec("ollama", "m")
    log = S.CallLog(tmp_path / "calls.jsonl")
    replies = iter(["one", "two", "three"])
    msgs = [{"role": "user", "content": "same"}]
    assert log.call("model", spec, msgs, {}, lambda: next(replies)) == "one"
    assert log.call("model", spec, msgs, {}, lambda: next(replies)) == "two"
    again = S.CallLog(tmp_path / "calls.jsonl", offline=True)
    assert again.call("model", spec, msgs, {}, lambda: "never") == "one"
    assert again.call("model", spec, msgs, {}, lambda: "never") == "two"
    with pytest.raises(S.ProviderError):
        again.call("model", spec, msgs, {}, lambda: "never")


def test_failed_calls_are_not_cached_and_are_counted(tmp_path):
    spec = S.ModelSpec("ollama", "m")
    log = S.CallLog(tmp_path / "calls.jsonl")
    tally = S.Tally()

    def boom():
        raise RuntimeError("down")
    chat_log_call = log.call
    with pytest.raises(S.ProviderError):
        chat_log_call("model", spec, [{"role": "user", "content": "x"}], {}, boom)
    assert S.CallLog(tmp_path / "calls.jsonl").records == {}
    tally.add("err")
    assert tally.errors == 1


def test_statistics():
    lo, hi = S.wilson(0, 10)
    assert lo == 0 and 0.25 < hi < 0.35
    assert S.wilson(0, 0) == (None, None)
    assert S.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(S.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9


def test_human_sample_and_agreement(tmp_path):
    rows = [{"model": m, "component": "authority", "question": f"q{i}", "answer": f"a{i}",
             "outcomes": {"x": "", "y": ""}, "label": "x" if i % 2 else "y"}
            for m in ("p:a", "p:b") for i in range(20)]
    path = tmp_path / "sample.csv"
    S.write_human_sample(path, rows, 10, seed=1)
    lines = path.read_text().splitlines()
    assert len(lines) == 11
    filled = tmp_path / "filled.csv"
    import csv
    with path.open() as f, filled.open("w", newline="") as g:
        r = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            row["human_label"] = row["judge_label"]
            w.writerow(row)
    ha = S.human_agreement(str(filled))
    assert ha["n"] == 10 and ha["agreement"]["value"] == 1.0
