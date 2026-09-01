"""
Tests for contradish.monitor -- production drift detection: load a
conversation log, cluster semantically-equivalent inputs via the judge,
score consistency within each cluster, and aggregate into a report.

No API key required. find_clusters/score_clusters take a `judge`-shaped
object; tests use a hand-written FakeJudge exposing only the two methods
these functions actually call (cluster_inputs, evaluate_consistency), so
nothing here ever calls a real model. See tests/test_judge.py for Judge's
own coverage.
"""
import csv
import json

import pytest

from contradish.monitor import (
    load_log, find_clusters, _topics_match, score_clusters,
    analyze_monitor, print_monitor_summary,
)


# ─────────────────────────────────────────────────────────────────────────
# load_log
# ─────────────────────────────────────────────────────────────────────────

def test_load_log_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_log("/no/such/file.jsonl")


def test_load_log_jsonl_happy_path(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text(
        '{"input": "hi", "output": "hello"}\n'
        '{"input": "bye", "output": "goodbye"}\n'
    )
    convs = load_log(str(p))
    assert len(convs) == 2
    assert convs[0]["input"] == "hi"
    assert convs[0]["output"] == "hello"


def test_load_log_jsonl_skips_blank_and_malformed_lines(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text(
        '{"input": "hi", "output": "hello"}\n'
        '\n'
        'not valid json\n'
        '{"input": "bye", "output": "goodbye"}\n'
    )
    convs = load_log(str(p))
    assert len(convs) == 2


def test_load_log_json_array_format(tmp_path):
    p = tmp_path / "log.json"
    p.write_text(json.dumps([{"input": "hi", "output": "hello"}]))
    convs = load_log(str(p))
    assert len(convs) == 1


def test_load_log_json_wrapped_dict_format(tmp_path):
    p = tmp_path / "log.json"
    p.write_text(json.dumps({"conversations": [{"input": "hi", "output": "hello"}]}))
    convs = load_log(str(p))
    assert len(convs) == 1


def test_load_log_json_wrapped_dict_tries_other_keys(tmp_path):
    p = tmp_path / "log.json"
    p.write_text(json.dumps({"results": [{"input": "hi", "output": "hello"}]}))
    convs = load_log(str(p))
    assert len(convs) == 1


def test_load_log_csv_format(tmp_path):
    p = tmp_path / "log.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input", "output"])
        w.writeheader()
        w.writerow({"input": "hi", "output": "hello"})
    convs = load_log(str(p))
    assert len(convs) == 1
    assert convs[0]["input"] == "hi"


def test_load_log_tsv_format(tmp_path):
    p = tmp_path / "log.tsv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["input", "output"], dialect="excel-tab")
        w.writeheader()
        w.writerow({"input": "hi", "output": "hello"})
    convs = load_log(str(p))
    assert len(convs) == 1


def test_load_log_unsupported_explicit_format_raises(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"input": "hi", "output": "hello"}\n')
    with pytest.raises(ValueError, match="Unsupported format"):
        load_log(str(p), format="xml")


def test_load_log_unknown_extension_falls_back_to_jsonl(tmp_path):
    p = tmp_path / "log.weird"
    p.write_text('{"input": "hi", "output": "hello"}\n')
    convs = load_log(str(p))
    assert len(convs) == 1


def test_load_log_accepts_field_aliases(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"question": "hi", "answer": "hello"}) + "\n")
    convs = load_log(str(p))
    assert convs[0]["input"] == "hi"
    assert convs[0]["output"] == "hello"


def test_load_log_drops_entries_missing_input_or_output(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text(
        json.dumps({"input": "hi", "output": "hello"}) + "\n"
        + json.dumps({"input": "no output here"}) + "\n"
    )
    convs = load_log(str(p))
    assert len(convs) == 1


def test_load_log_all_invalid_raises_value_error(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"input": "no output"}) + "\n")
    with pytest.raises(ValueError, match="No valid conversations"):
        load_log(str(p))


def test_load_log_caps_at_max_conversations(tmp_path):
    p = tmp_path / "log.jsonl"
    lines = [json.dumps({"input": f"q{i}", "output": f"a{i}"}) for i in range(10)]
    p.write_text("\n".join(lines) + "\n")
    convs = load_log(str(p), max_conversations=3)
    assert len(convs) == 3


def test_load_log_preserves_extra_fields(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"input": "hi", "output": "hello", "domain": "support"}) + "\n")
    convs = load_log(str(p))
    assert convs[0]["domain"] == "support"


# ─────────────────────────────────────────────────────────────────────────
# _topics_match
# ─────────────────────────────────────────────────────────────────────────

def test_topics_match_identical_strings():
    assert _topics_match("refund window", "refund window")


def test_topics_match_high_overlap():
    assert _topics_match("refund window policy", "refund window details")


def test_topics_match_low_overlap_is_false():
    assert not _topics_match("refund policy", "shipping cost")


def test_topics_match_empty_string_is_false():
    assert not _topics_match("", "refund policy")
    assert not _topics_match("refund policy", "")


def test_topics_match_stopword_only_overlap_is_false():
    assert not _topics_match("the cost of a widget", "the price of a gadget")


# ─────────────────────────────────────────────────────────────────────────
# find_clusters
# ─────────────────────────────────────────────────────────────────────────

class FakeJudge:
    """Exposes only cluster_inputs/evaluate_consistency, matching what
    find_clusters/score_clusters actually call on a real Judge."""

    def __init__(self, cluster_responses=None, consistency_responses=None):
        # cluster_responses: single dict, or list consumed per-call
        self._cluster_responses = cluster_responses
        self._consistency_responses = consistency_responses
        self.cluster_calls = []
        self.consistency_calls = []

    def cluster_inputs(self, inputs):
        self.cluster_calls.append(inputs)
        if isinstance(self._cluster_responses, list):
            idx = min(len(self.cluster_calls) - 1, len(self._cluster_responses) - 1)
            return self._cluster_responses[idx]
        return self._cluster_responses

    def evaluate_consistency(self, question, inputs, outputs):
        self.consistency_calls.append((question, inputs, outputs))
        if isinstance(self._consistency_responses, list):
            idx = min(len(self.consistency_calls) - 1, len(self._consistency_responses) - 1)
            return self._consistency_responses[idx]
        return self._consistency_responses


def _convs(n, prefix="q"):
    return [{"input": f"{prefix}{i}", "output": f"a{i}"} for i in range(n)]


def test_find_clusters_empty_conversations_returns_empty():
    assert find_clusters([], FakeJudge()) == []


def test_find_clusters_builds_and_filters_by_min_size():
    convs = _convs(5)
    judge = FakeJudge(cluster_responses={
        "clusters": [{"input_indices": [0, 1, 2], "topic": "refund window question"},
                     {"input_indices": [3, 4], "topic": "shipping cost delay"}],
        "singletons": [],
    })
    clusters = find_clusters(convs, judge, min_cluster_size=3, quiet=True)
    assert len(clusters) == 1
    assert clusters[0]["topic"] == "refund window question"
    assert clusters[0]["size"] == 3


def test_find_clusters_drops_single_conversation_clusters():
    convs = _convs(3)
    judge = FakeJudge(cluster_responses={
        "clusters": [{"input_indices": [0], "topic": "lonely"}],
        "singletons": [1, 2],
    })
    clusters = find_clusters(convs, judge, min_cluster_size=1, quiet=True)
    assert clusters == []  # single-conversation "clusters" are dropped before sizing


def test_find_clusters_merges_matching_topics_across_batches():
    convs = _convs(6)
    judge = FakeJudge(cluster_responses=[
        {"clusters": [{"input_indices": [0, 1], "topic": "refund window question"}]},
        {"clusters": [{"input_indices": [0, 1], "topic": "refund window question"}]},
        {"clusters": [{"input_indices": [0, 1], "topic": "refund window question"}]},
    ])
    clusters = find_clusters(convs, judge, min_cluster_size=3, batch_size=2, quiet=True)
    assert len(clusters) == 1
    assert clusters[0]["size"] == 6  # 2 per batch x 3 batches, all merged


def test_find_clusters_keeps_non_matching_topics_separate():
    convs = _convs(6)
    judge = FakeJudge(cluster_responses=[
        {"clusters": [{"input_indices": [0, 1], "topic": "refund window question"}]},
        {"clusters": [{"input_indices": [0, 1], "topic": "totally unrelated shipping cost"}]},
        {"clusters": [{"input_indices": [0, 1], "topic": "refund window question"}]},
    ])
    clusters = find_clusters(convs, judge, min_cluster_size=1, batch_size=2, quiet=True)
    topics = {c["topic"] for c in clusters}
    assert "refund window question" in topics
    assert "totally unrelated shipping cost" in topics


def test_find_clusters_sorted_largest_first():
    convs = _convs(6)
    judge = FakeJudge(cluster_responses={
        "clusters": [
            {"input_indices": [0, 1], "topic": "small"},
            {"input_indices": [2, 3, 4, 5], "topic": "big"},
        ],
    })
    clusters = find_clusters(convs, judge, min_cluster_size=2, quiet=True)
    assert [c["topic"] for c in clusters] == ["big", "small"]


def test_find_clusters_prints_progress_when_not_quiet(capsys):
    convs = _convs(2)
    judge = FakeJudge(cluster_responses={"clusters": []})
    find_clusters(convs, judge, quiet=False)
    assert "clustering" in capsys.readouterr().out


def test_find_clusters_ignores_out_of_range_indices():
    convs = _convs(3)
    judge = FakeJudge(cluster_responses={
        "clusters": [{"input_indices": [0, 1, 99], "topic": "t"}],
    })
    clusters = find_clusters(convs, judge, min_cluster_size=1, quiet=True)
    assert clusters[0]["size"] == 2


# ─────────────────────────────────────────────────────────────────────────
# score_clusters
# ─────────────────────────────────────────────────────────────────────────

def _cluster(n, topic="t"):
    return {"topic": topic, "conversations": _convs(n), "size": n}


def test_score_clusters_happy_path():
    judge = FakeJudge(consistency_responses={
        "consistency_score": 0.9, "per_variant_scores": [0.9, 0.9],
        "disagreements": [], "summary": "consistent",
    })
    scored = score_clusters([_cluster(3)], judge, quiet=True)
    assert len(scored) == 1
    c = scored[0]
    assert c["consistency_score"] == 0.9
    assert c["cts"] == round(1.0 - 0.9, 4)
    assert c["drifted"] is False


def test_score_clusters_drifted_boundary():
    # cts = 1 - 0.69 = 0.31 > 0.3 -> drifted
    judge = FakeJudge(consistency_responses={"consistency_score": 0.69, "per_variant_scores": []})
    scored = score_clusters([_cluster(2)], judge, quiet=True)
    assert scored[0]["drifted"] is True

    # cts = 1 - 0.70 = 0.30, not > 0.3 -> not drifted
    judge2 = FakeJudge(consistency_responses={"consistency_score": 0.70, "per_variant_scores": []})
    scored2 = score_clusters([_cluster(2)], judge2, quiet=True)
    assert scored2[0]["drifted"] is False


def test_score_clusters_missing_consistency_score_defaults_to_half():
    judge = FakeJudge(consistency_responses={})
    scored = score_clusters([_cluster(2)], judge, quiet=True)
    assert scored[0]["consistency_score"] == 0.5


def test_score_clusters_no_per_variant_scores_leaves_examples_none():
    judge = FakeJudge(consistency_responses={"consistency_score": 0.5, "per_variant_scores": []})
    scored = score_clusters([_cluster(3)], judge, quiet=True)
    assert scored[0]["example_consistent"] is None
    assert scored[0]["example_drifted"] is None


def test_score_clusters_picks_best_and_worst_examples():
    judge = FakeJudge(consistency_responses={
        "consistency_score": 0.6,
        "per_variant_scores": [0.9, 0.2],  # variant 1 -> outputs[1], variant 2 -> outputs[2]
    })
    scored = score_clusters([_cluster(3)], judge, quiet=True)
    assert scored[0]["example_consistent"] == "a1"  # best score 0.9 >= 0.7 -> outputs[1]
    assert scored[0]["example_drifted"] == "a2"      # worst score 0.2 < 0.5 -> outputs[2]


def test_score_clusters_no_example_when_scores_dont_cross_thresholds():
    judge = FakeJudge(consistency_responses={
        "consistency_score": 0.6,
        "per_variant_scores": [0.6, 0.55],  # neither >= 0.7 nor < 0.5
    })
    scored = score_clusters([_cluster(3)], judge, quiet=True)
    assert scored[0]["example_consistent"] is None
    assert scored[0]["example_drifted"] is None


def test_score_clusters_prints_progress_when_not_quiet(capsys):
    judge = FakeJudge(consistency_responses={"consistency_score": 1.0, "per_variant_scores": []})
    score_clusters([_cluster(2)], judge, quiet=False)
    assert "scoring" in capsys.readouterr().out


def test_score_clusters_passes_canonical_question_as_first_input():
    judge = FakeJudge(consistency_responses={"consistency_score": 1.0, "per_variant_scores": []})
    score_clusters([_cluster(2)], judge, quiet=True)
    question, inputs, outputs = judge.consistency_calls[0]
    assert question == inputs[0]


# ─────────────────────────────────────────────────────────────────────────
# analyze_monitor
# ─────────────────────────────────────────────────────────────────────────

def test_analyze_monitor_empty_scored_clusters():
    result = analyze_monitor([], total_conversations=10)
    assert result["clusters_scored"] == 0
    assert result["drift_rate"] == 0.0
    assert result["hotspots"] == []
    assert result["clean_clusters"] == []


def _scored(topic, cts, drifted, domain=None, size=3):
    convs = _convs(size)
    if domain is not None:
        for c in convs:
            c["domain"] = domain
    return {
        "topic": topic, "conversations": convs, "size": size,
        "consistency_score": round(1.0 - cts, 4), "cts": cts, "drifted": drifted,
        "per_conversation_scores": [], "disagreements": [], "summary": "",
        "example_input_canonical": "q0", "example_output_canonical": "a0",
        "example_consistent": None, "example_drifted": None,
    }


def test_analyze_monitor_computes_rate_and_averages():
    scored = [_scored("a", 0.5, True), _scored("b", 0.1, False)]
    result = analyze_monitor(scored, total_conversations=20)
    assert result["clusters_scored"] == 2
    assert result["drifted_clusters"] == 1
    assert result["drift_rate"] == 0.5
    assert result["avg_cts"] == round((0.5 + 0.1) / 2, 4)


def test_analyze_monitor_hotspots_sorted_worst_first_clean_sorted_best_first():
    scored = [
        _scored("mild", 0.4, True),
        _scored("severe", 0.9, True),
        _scored("ok", 0.2, False),
        _scored("great", 0.05, False),
    ]
    result = analyze_monitor(scored, total_conversations=10)
    assert [h["topic"] for h in result["hotspots"]] == ["severe", "mild"]
    assert [c["topic"] for c in result["clean_clusters"]] == ["great", "ok"]


def test_analyze_monitor_domain_breakdown():
    scored = [
        _scored("a", 0.5, True, domain="support"),
        _scored("b", 0.1, False, domain="sales"),
    ]
    result = analyze_monitor(scored, total_conversations=10)
    breakdown = result["domain_breakdown"]
    assert breakdown["support"]["drifted"] == 3  # all 3 convs in the drifted cluster
    assert breakdown["support"]["total"] == 3
    assert breakdown["support"]["drift_rate"] == 1.0
    assert breakdown["sales"]["drift_rate"] == 0.0


def test_analyze_monitor_domain_defaults_to_unknown():
    scored = [_scored("a", 0.1, False, domain=None)]
    result = analyze_monitor(scored, total_conversations=5)
    assert "unknown" in result["domain_breakdown"]


def test_analyze_monitor_slim_caps_disagreements_at_three():
    scored = [_scored("a", 0.5, True)]
    scored[0]["disagreements"] = ["d1", "d2", "d3", "d4", "d5"]
    result = analyze_monitor(scored, total_conversations=5)
    assert len(result["hotspots"][0]["disagreements"]) == 3


# ─────────────────────────────────────────────────────────────────────────
# print_monitor_summary (smoke tests: must not raise, key content present)
# ─────────────────────────────────────────────────────────────────────────

def test_print_monitor_summary_with_hotspots(capsys):
    analysis = analyze_monitor(
        [_scored("bad topic", 0.6, True), _scored("good topic", 0.1, False)],
        total_conversations=50,
    )
    print_monitor_summary(analysis, "log.jsonl")
    out = capsys.readouterr().out
    assert "DRIFT HOTSPOTS" in out
    assert "bad topic" in out
    assert "CONSISTENT" in out


def test_print_monitor_summary_no_drift(capsys):
    analysis = analyze_monitor([_scored("good topic", 0.05, False)], total_conversations=10)
    print_monitor_summary(analysis, "log.jsonl")
    out = capsys.readouterr().out
    assert "No significant drift detected" in out


def test_print_monitor_summary_hotspot_details_shown(capsys):
    scored = [_scored("bad topic", 0.6, True)]
    scored[0]["summary"] = "diverges under emotional framing"
    scored[0]["example_drifted"] = "the drifted answer text"
    scored[0]["disagreements"] = ["variant 2 contradicts variant 0"]
    analysis = analyze_monitor(scored, total_conversations=10)
    print_monitor_summary(analysis, "log.jsonl")
    out = capsys.readouterr().out
    assert "diverges under emotional framing" in out
    assert "the drifted answer text" in out
    assert "variant 2 contradicts variant 0" in out


def test_print_monitor_summary_multi_domain_breakdown_shown(capsys):
    scored = [
        _scored("a", 0.5, True, domain="support"),
        _scored("b", 0.1, False, domain="sales"),
    ]
    analysis = analyze_monitor(scored, total_conversations=10)
    print_monitor_summary(analysis, "log.jsonl")
    out = capsys.readouterr().out
    assert "BY DOMAIN" in out
