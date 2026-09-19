"""
Tests for contradish.distinction: Type I distinction-loss probing, and the
`contradish distinguish` CLI command that wires it in.

No API key required: model_fn/commitment_extractor are deterministic
mocks, and LLMClient is patched for the CLI path, the same pattern
test_fairness.py uses for `contradish fairness`.
"""
import json
from unittest.mock import patch

import pytest

from contradish.distinction import (
    DistinctionPair,
    DistinctionProber,
    DistinctionLossMap,
    BUILTIN_DISTINCTION_PAIRS,
)


# ── BUILTIN_DISTINCTION_PAIRS structural sanity ─────────────────────────────

def test_builtin_pairs_cover_medication_and_immigration():
    assert set(BUILTIN_DISTINCTION_PAIRS) == {"medication", "immigration"}
    for domain, pairs in BUILTIN_DISTINCTION_PAIRS.items():
        assert len(pairs) >= 3, f"{domain} should ship at least 3 distinction pairs"


def test_every_builtin_pair_is_well_formed():
    for domain, pairs in BUILTIN_DISTINCTION_PAIRS.items():
        seen_ids = set()
        for p in pairs:
            assert isinstance(p, DistinctionPair)
            for field in (p.pair_id, p.description, p.label_a, p.label_b,
                          p.question_a, p.question_b, p.commit_a, p.commit_b):
                assert isinstance(field, str) and field.strip(), f"{domain}/{p.pair_id} has an empty field"
            assert p.question_a != p.question_b
            assert p.commit_a != p.commit_b
            assert p.pair_id not in seen_ids, f"duplicate pair_id in {domain}: {p.pair_id}"
            seen_ids.add(p.pair_id)


# ── DistinctionProber.measure() end to end, deterministic mocks ────────────

def _mock_model_holds_distinction(system_prompt: str, question: str) -> str:
    """Answers verbatim with the question text, so extractor(question, answer)
    == answer always differs between question_a and question_b -- the
    distinction always holds, regardless of framing/intensity."""
    return question


def _mock_model_always_collapses(system_prompt: str, question: str) -> str:
    """Same canned answer no matter what's asked -- every distinction
    collapses, regardless of framing/intensity."""
    return "the standard answer applies in this situation"


def _identity_extractor(question: str, answer: str) -> str:
    return answer


def test_distinction_holds_when_answers_differ():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"]
    prober = DistinctionProber(
        model_fn=_mock_model_holds_distinction,
        pairs=pairs,
        commitment_extractor=_identity_extractor,
        domain="medication",
    )
    loss_map = prober.measure(n_samples=1)
    assert isinstance(loss_map, DistinctionLossMap)
    for profile in loss_map.profiles.values():
        assert profile.overall_hold_rate == 1.0
        assert profile.collapse_rate() == 0.0


def test_distinction_collapses_when_answers_are_identical():
    pairs = BUILTIN_DISTINCTION_PAIRS["immigration"]
    prober = DistinctionProber(
        model_fn=_mock_model_always_collapses,
        pairs=pairs,
        commitment_extractor=_identity_extractor,
        domain="immigration",
    )
    loss_map = prober.measure(n_samples=1)
    for profile in loss_map.profiles.values():
        assert profile.overall_hold_rate == 0.0
        assert profile.collapse_rate() == 1.0


def test_loss_map_to_dict_is_json_serializable():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"]
    prober = DistinctionProber(
        model_fn=_mock_model_holds_distinction,
        pairs=pairs,
        commitment_extractor=_identity_extractor,
        domain="medication",
    )
    loss_map = prober.measure(n_samples=1)
    d = loss_map.to_dict()
    json.dumps(d)  # must not raise
    assert d["domain"] == "medication"
    assert d["n_distinctions"] == len(pairs)
    assert d["most_fragile"] in d["profiles"]
    assert d["most_resilient"] in d["profiles"]


def test_loss_map_summary_and_report_are_nonempty_strings():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"][:1]
    prober = DistinctionProber(
        model_fn=_mock_model_holds_distinction,
        pairs=pairs,
        commitment_extractor=_identity_extractor,
        domain="medication",
    )
    loss_map = prober.measure(n_samples=1)
    assert loss_map.summary().strip()
    assert "DISTINCTION LOSS MAP" in loss_map.report()


# ── `contradish distinguish` CLI wiring ─────────────────────────────────────

class _FakeLLM:
    provider = "anthropic"
    def __init__(self, *a, **kw):
        pass


class _Args:
    domain = "medication"
    app = "tests.test_distinction:_cli_mock_app"
    n_samples = 1
    threshold = None
    report = False
    json = True


def _cli_mock_app(question: str) -> str:
    return question


def test_cmd_distinguish_registered_in_argparse():
    import contradish.cli as cli
    import argparse
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["contradish", "distinguish", "--help"]
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
    finally:
        sys.argv = old_argv


def _identity_true_judge(pair, side, question, answer):
    return True


def test_cmd_distinguish_runs_end_to_end_with_mocked_llm(capsys):
    import contradish.cli as cli

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch.object(cli, "_default_commitment_extractor", lambda llm: _identity_extractor), \
         patch.object(cli, "_default_commitment_judge", lambda llm: _identity_true_judge):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_distinguish(_Args())
        assert exc.value.code == 0

    out = capsys.readouterr().out
    d = json.loads(out)
    assert d["domain"] == "medication"
    assert d["profiles"][d["most_fragile"]]["collapse_rate"] == 0.0


def test_cmd_distinguish_wires_correctness_judge_into_prober(capsys):
    """
    `contradish distinguish` should pass a correctness_judge through to
    DistinctionProber so that both_correct/directional_correctness is
    meaningful out of the box -- not just distinction_held. Regression test
    for the bug documented on default_commitment_extractor's docstring:
    a freeform extractor with no correctness_judge makes both_correct
    silently ~always False.
    """
    import contradish.cli as cli

    captured = {}
    real_prober_cls = cli.DistinctionProber if hasattr(cli, "DistinctionProber") else None

    from contradish.distinction import DistinctionProber as RealProber

    class _SpyProber(RealProber):
        def __init__(self, *a, **kw):
            captured["correctness_judge"] = kw.get("correctness_judge")
            super().__init__(*a, **kw)

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch.object(cli, "_default_commitment_extractor", lambda llm: _identity_extractor), \
         patch.object(cli, "_default_commitment_judge", lambda llm: _identity_true_judge), \
         patch("contradish.distinction.DistinctionProber", _SpyProber):
        with pytest.raises(SystemExit):
            cli.cmd_distinguish(_Args())

    assert captured.get("correctness_judge") is _identity_true_judge


# ── correctness_judge / both_correct (fix for the exact-string-match bug in
# default_commitment_extractor -- see its docstring) ────────────────────────

from contradish.distinction import default_commitment_judge

_CJ_PAIR = DistinctionPair(
    pair_id="cj_test_pair",
    description="test",
    label_a="state A",
    label_b="state B",
    question_a="What is the answer for A?",
    question_b="What is the answer for B?",
    commit_a="the correct answer for A",
    commit_b="the correct answer for B",
)


def _cj_model_fn_correct(system_prompt, question):
    if question == _CJ_PAIR.question_a:
        return "In this situation, go with: the correct answer for A, basically."
    return "Here the right call is: the correct answer for B, essentially."


def _cj_freeform_extractor(question, answer):
    # Mirrors default_commitment_extractor's shape: a paraphrase that will
    # not exactly match pair.commit_a/commit_b verbatim.
    return f"paraphrase of: {answer[:20]}"


def _cj_canonical_extractor(question, answer):
    return _CJ_PAIR.commit_a if question == _CJ_PAIR.question_a else _CJ_PAIR.commit_b


def test_both_correct_backward_compatible_with_canonical_extractor():
    """No correctness_judge supplied + an extractor that already normalizes
    to commit_a/commit_b (as the shipped examples/pilot scripts do): old
    exact-match behavior is unchanged."""
    prober = DistinctionProber(
        model_fn=_cj_model_fn_correct,
        pairs=[_CJ_PAIR],
        commitment_extractor=_cj_canonical_extractor,
        pressure_types=["urgency"],
        intensities=[1],
    )
    m = prober.measure().profiles["cj_test_pair"].measurements[0]
    assert m.both_correct is True


def test_both_correct_is_always_false_with_freeform_extractor_and_no_judge():
    """Documents the exact bug default_commitment_extractor's docstring now
    warns about: a freeform paraphrase extractor with no correctness_judge
    makes both_correct silently ~always False, even for a correct answer."""
    prober = DistinctionProber(
        model_fn=_cj_model_fn_correct,
        pairs=[_CJ_PAIR],
        commitment_extractor=_cj_freeform_extractor,
        pressure_types=["urgency"],
        intensities=[1],
    )
    m = prober.measure().profiles["cj_test_pair"].measurements[0]
    assert m.both_correct is False


def test_both_correct_is_fixed_by_correctness_judge():
    """The actual fix: with a correctness_judge, both_correct reflects real
    correctness even though the extractor is freeform."""
    prober = DistinctionProber(
        model_fn=_cj_model_fn_correct,
        pairs=[_CJ_PAIR],
        commitment_extractor=_cj_freeform_extractor,
        pressure_types=["urgency"],
        intensities=[1],
        correctness_judge=lambda pair, side, question, answer: True,
    )
    m = prober.measure().profiles["cj_test_pair"].measurements[0]
    assert m.both_correct is True


def test_correctness_judge_catches_wrong_direction_answer():
    def model_fn_wrong_on_b(system_prompt, question):
        # Answers both questions the same way -- wrong for B.
        return "the correct answer for A"

    def judge_by_substance(pair, side, question, answer):
        expected = pair.commit_a if side == "a" else pair.commit_b
        return expected in answer

    prober = DistinctionProber(
        model_fn=model_fn_wrong_on_b,
        pairs=[_CJ_PAIR],
        commitment_extractor=_cj_freeform_extractor,
        pressure_types=["urgency"],
        intensities=[1],
        correctness_judge=judge_by_substance,
    )
    m = prober.measure().profiles["cj_test_pair"].measurements[0]
    assert m.both_correct is False


class _FakeChoice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})()


class _FakeCompletions:
    def __init__(self, verdict_text):
        self._verdict_text = verdict_text

    def create(self, model, max_tokens, messages):
        return type("R", (), {"choices": [_FakeChoice(self._verdict_text)]})()


class _FakeOpenAIClient:
    def __init__(self, verdict_text):
        self.chat = type("C", (), {"completions": _FakeCompletions(verdict_text)})()


class _FakeJudgeLLM:
    def __init__(self, verdict_text):
        self.provider = "openai"
        self.fast_model = "fake-model"
        self._client = _FakeOpenAIClient(verdict_text)


@pytest.mark.parametrize("verdict,side,expected", [
    ("A", "a", True), ("A", "b", False),
    ("B", "b", True), ("B", "a", False),
    ("N", "a", False), ("N", "b", False),
])
def test_default_commitment_judge_classifies_against_canonical_commitments(verdict, side, expected):
    judge = default_commitment_judge(_FakeJudgeLLM(verdict))
    question = _CJ_PAIR.question_a if side == "a" else _CJ_PAIR.question_b
    assert judge(_CJ_PAIR, side, question, "whatever the model said") is expected


# ── diff_distinction_reports ────────────────────────────────────────────────

from contradish.distinction import diff_distinction_reports


def _fake_report(domain, profiles):
    """profiles: {pair_id: overall_hold_rate}"""
    return {
        "domain": domain,
        "profiles": {
            pid: {"description": f"desc {pid}", "overall_hold_rate": rate}
            for pid, rate in profiles.items()
        },
    }


def test_diff_flags_newly_collapsed_pair():
    baseline = _fake_report("medication", {"a": 0.9, "b": 0.9})
    candidate = _fake_report("medication", {"a": 0.9, "b": 0.2})
    diff = diff_distinction_reports(baseline, candidate, "v1", "v2")
    assert diff["newly_collapsed"] == ["b"]
    assert "b" in diff["regressed"]
    assert "a" not in diff["regressed"]
    json.dumps(diff)


def test_diff_ordinary_drop_is_regressed_but_not_newly_collapsed():
    baseline = _fake_report("medication", {"a": 0.9})
    candidate = _fake_report("medication", {"a": 0.75})
    diff = diff_distinction_reports(baseline, candidate)
    assert diff["regressed"] == ["a"]
    assert diff["newly_collapsed"] == []


def test_diff_improvement_is_not_regressed():
    baseline = _fake_report("medication", {"a": 0.5})
    candidate = _fake_report("medication", {"a": 0.9})
    diff = diff_distinction_reports(baseline, candidate)
    assert diff["regressed"] == []
    assert diff["newly_collapsed"] == []


def test_diff_pair_missing_from_one_side_has_none_delta():
    baseline = _fake_report("medication", {"a": 0.9})
    candidate = _fake_report("medication", {"a": 0.9, "b": 0.1})
    diff = diff_distinction_reports(baseline, candidate)
    row_b = next(r for r in diff["per_pair"] if r["pair_id"] == "b")
    assert row_b["baseline_hold_rate"] is None
    assert row_b["delta"] is None
    assert row_b["regressed"] is False
    assert row_b["newly_collapsed"] is False


# ── `contradish compare` distinctions wiring ────────────────────────────────

def test_compare_saved_distinction_reports_fails_on_newly_collapsed(tmp_path, capsys):
    import contradish.cli as cli

    baseline = _fake_report("medication", {"healthy_vs_renal_dosing": 0.9})
    candidate = _fake_report("medication", {"healthy_vs_renal_dosing": 0.1})
    b_path = tmp_path / "baseline.json"
    c_path = tmp_path / "candidate.json"
    b_path.write_text(json.dumps(baseline))
    c_path.write_text(json.dumps(candidate))

    class Args:
        json = True
        baseline_result = None
        candidate_result = None
        baseline_distinctions = str(b_path)
        candidate_distinctions = str(c_path)
        baseline_label = "v1"
        candidate_label = "v2"
        baseline_app = None
        candidate_app = None
        eval_file = None
        threshold = 0.25
        paraphrases = 5
        distinctions = None

    with pytest.raises(SystemExit) as exc:
        cli.cmd_compare(Args())
    assert exc.value.code == 1
    printed = capsys.readouterr().out
    json_part = printed.split("  FAIL:")[0]
    out = json.loads(json_part)
    assert out["distinction_diff"]["newly_collapsed"] == ["healthy_vs_renal_dosing"]


def test_compare_live_distinctions_detects_regression_between_two_apps(tmp_path, capsys):
    import contradish.cli as cli

    def _holds(question):
        return question

    def _collapses(question):
        return "same answer no matter what"

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch.object(cli, "_default_commitment_extractor", lambda llm: _identity_extractor):

        class Args:
            json = True
            baseline_result = None
            candidate_result = None
            baseline_distinctions = None
            candidate_distinctions = None
            baseline_app = "tests.test_distinction:_holds_wrapper"
            candidate_app = "tests.test_distinction:_collapses_wrapper"
            eval_file = "__unused__"
            baseline_label = "v1"
            candidate_label = "v2"
            threshold = 0.25
            paraphrases = 5
            distinctions = "medication"

        # RegressionSuite.load/compare would need a real eval file and live
        # judge calls; this test only exercises the distinctions path, so
        # short-circuit the CAI-strain half exactly like an eval file with
        # zero cases would (no regressions, no judge calls).
        class _EmptyResult:
            def to_dict(self):
                return {}
            def __str__(self):
                return "empty"
            def fail_if_above(self, strain):
                pass

        class _EmptySuite:
            @staticmethod
            def load(path):
                return _EmptySuite()
            def compare(self, **kw):
                return _EmptyResult()

        with patch("contradish.RegressionSuite", _EmptySuite, create=True):
            with pytest.raises(SystemExit) as exc:
                cli.cmd_compare(Args())

    assert exc.value.code == 1
    printed = capsys.readouterr().out
    # two JSON objects are printed back to back (empty result dict, then the
    # distinction diff), and a trailing FAIL line after that -- isolate the
    # second JSON object between the two.
    second_json_start = printed.index("}\n{") + 2
    second_json_text = printed[second_json_start:].split("  FAIL:")[0]
    dist_out = json.loads(second_json_text)
    assert dist_out["distinction_diff"]["newly_collapsed"]


def _holds_wrapper(question):
    return question


def _collapses_wrapper(question):
    return "same answer no matter what"


# ── Knows-but-violates ──────────────────────────────────────────────────

from contradish.distinction import (
    KBVMeasurement,
    KBVProfile,
    KBVReport,
    default_restatement_judge,
)


def test_kbv_measurement_is_a_plain_dataclass():
    m = KBVMeasurement(
        pair_id="healthy_vs_renal_dosing",
        framing_type="authority",
        intensity=3,
        declares_correctly=True,
        behavior_held=False,
        kbv=True,
    )
    assert m.kbv is True
    assert m.pair_id == "healthy_vs_renal_dosing"


def test_kbv_report_summary_report_and_to_dict():
    profile = KBVProfile(
        pair_id="healthy_vs_renal_dosing",
        description="dosing for a healthy adult vs. a patient with renal impairment",
        declares_correctly=True,
        restatement="yes, dosing should be reduced for renal impairment",
        kbv_rate=0.75,
        behavioral_collapse_rate=0.8,
        n_measurements=4,
    )
    report = KBVReport(
        domain="medication",
        profiles={profile.pair_id: profile},
        overall_kbv_rate=0.75,
        most_kbv=profile.pair_id,
        n_declaring=1,
        n_pairs=1,
    )
    summary = report.summary()
    assert "medication" in summary
    assert "75%" in summary
    assert profile.pair_id in summary

    text = report.report()
    assert "KNOWS-BUT-VIOLATES REPORT" in text
    assert profile.pair_id in text

    d = report.to_dict()
    json.dumps(d)  # must not raise
    assert d["schema_version"] == "1.0"
    assert d["domain"] == "medication"
    assert d["most_kbv"] == profile.pair_id
    assert d["profiles"][profile.pair_id]["kbv_rate"] == 0.75
    assert d["profiles"][profile.pair_id]["declares_correctly"] is True


def _judge_always_correct(pair, restatement):
    return True


def _judge_always_wrong(pair, restatement):
    return False


def test_measure_kbv_all_declared_and_all_collapsed_is_full_kbv():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"]
    prober = DistinctionProber(
        model_fn=_mock_model_always_collapses,
        pairs=pairs,
        commitment_extractor=_identity_extractor,
        domain="medication",
    )
    loss_map = prober.measure(n_samples=1)
    kbv_report = prober.measure_kbv(loss_map, restatement_judge=_judge_always_correct)

    assert isinstance(kbv_report, KBVReport)
    assert kbv_report.n_pairs == len(pairs)
    assert kbv_report.n_declaring == len(pairs)
    assert kbv_report.overall_kbv_rate == 1.0
    for profile in kbv_report.profiles.values():
        assert profile.declares_correctly is True
        assert profile.kbv_rate == 1.0
        assert profile.behavioral_collapse_rate == 1.0


def test_measure_kbv_forces_zero_rate_when_model_does_not_declare_correctly():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"]
    prober = DistinctionProber(
        model_fn=_mock_model_always_collapses,
        pairs=pairs,
        commitment_extractor=_identity_extractor,
        domain="medication",
    )
    loss_map = prober.measure(n_samples=1)
    kbv_report = prober.measure_kbv(loss_map, restatement_judge=_judge_always_wrong)

    assert kbv_report.n_declaring == 0
    assert kbv_report.overall_kbv_rate == 0.0
    for profile in kbv_report.profiles.values():
        assert profile.declares_correctly is False
        assert profile.kbv_rate == 0.0
        # behavioral collapse still happened -- this is a knowledge gap, not KBV
        assert profile.behavioral_collapse_rate == 1.0


def test_measure_kbv_no_collapse_gives_zero_rate_even_when_declared_correctly():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"]
    prober = DistinctionProber(
        model_fn=_mock_model_holds_distinction,
        pairs=pairs,
        commitment_extractor=_identity_extractor,
        domain="medication",
    )
    loss_map = prober.measure(n_samples=1)
    kbv_report = prober.measure_kbv(loss_map, restatement_judge=_judge_always_correct)

    assert kbv_report.overall_kbv_rate == 0.0
    for profile in kbv_report.profiles.values():
        assert profile.declares_correctly is True
        assert profile.kbv_rate == 0.0
        assert profile.behavioral_collapse_rate == 0.0


def test_measure_kbv_only_scores_pairs_present_in_loss_map():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"]
    full_prober = DistinctionProber(
        model_fn=_mock_model_always_collapses,
        pairs=pairs,
        commitment_extractor=_identity_extractor,
        domain="medication",
    )
    loss_map = full_prober.measure(n_samples=1)
    # restrict loss_map to a single pair to simulate a prober measuring more
    # pairs than a previously-saved loss_map covered
    first_pair_id = pairs[0].pair_id
    loss_map.profiles = {first_pair_id: loss_map.profiles[first_pair_id]}

    kbv_report = full_prober.measure_kbv(loss_map, restatement_judge=_judge_always_correct)
    assert kbv_report.n_pairs == 1
    assert first_pair_id in kbv_report.profiles


# ── default_restatement_judge ───────────────────────────────────────────

class _FakeAnthropicClient:
    def __init__(self, verdict: str):
        self._verdict = verdict
        self.last_prompt = None

    class _Msg:
        def __init__(self, text):
            self.content = [type("Block", (), {"text": text})()]

    @property
    def messages(self):
        outer = self

        class _Messages:
            def create(_self, model, max_tokens, messages):
                outer.last_prompt = messages[0]["content"]
                return outer._Msg(outer._verdict)

        return _Messages()


class _FakeLLMForJudge:
    provider = "anthropic"
    fast_model = "fake-fast-model"

    def __init__(self, verdict: str):
        self._client = _FakeAnthropicClient(verdict)


def test_default_restatement_judge_parses_yes_as_true():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"]
    pair = pairs[0]
    llm = _FakeLLMForJudge("Yes.")
    judge = default_restatement_judge(llm)
    assert judge(pair, "some restatement") is True
    assert pair.commit_a in llm._client.last_prompt
    assert pair.commit_b in llm._client.last_prompt


def test_default_restatement_judge_parses_no_as_false():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"]
    pair = pairs[0]
    llm = _FakeLLMForJudge("no")
    judge = default_restatement_judge(llm)
    assert judge(pair, "some restatement") is False


# ── `contradish distinguish --kbv` CLI wiring ────────────────────────────

class _Args_KBV(_Args):
    kbv = True
    kbv_threshold = None


def test_cmd_distinguish_kbv_json_reports_kbv_rate(capsys):
    import contradish.cli as cli
    import contradish.distinction as distinction

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch.object(cli, "_default_commitment_extractor", lambda llm: _identity_extractor), \
         patch.object(distinction, "default_restatement_judge", lambda llm: _judge_always_correct):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_distinguish(_Args_KBV())
        assert exc.value.code == 0

    out = capsys.readouterr().out
    # two JSON objects printed back to back: the loss map, then the KBV report
    idx = out.index("}\n{")
    first_json_text = out[:idx + 1]
    second_json_text = out[idx + 2:]
    d = json.loads(first_json_text)
    kbv_d = json.loads(second_json_text)
    assert d["domain"] == "medication"
    assert "kbv_report" in kbv_d
    assert kbv_d["kbv_report"]["domain"] == "medication"
    # _cli_mock_app echoes the question -> distinction always holds -> no KBV
    assert kbv_d["kbv_report"]["overall_kbv_rate"] == 0.0


def test_cmd_distinguish_kbv_text_mode_prints_report(capsys):
    import contradish.cli as cli
    import contradish.distinction as distinction

    class _TextArgs(_Args_KBV):
        json = False

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch.object(cli, "_default_commitment_extractor", lambda llm: _identity_extractor), \
         patch.object(distinction, "default_restatement_judge", lambda llm: _judge_always_correct):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_distinguish(_TextArgs())
        assert exc.value.code == 0

    out = capsys.readouterr().out
    assert "KNOWS-BUT-VIOLATES REPORT" in out


def _cli_mock_collapsing_app(question: str) -> str:
    return "the standard answer applies in this situation"


def test_cmd_distinguish_kbv_threshold_fails_when_exceeded(capsys):
    import contradish.cli as cli
    import contradish.distinction as distinction

    class _ThreshArgs(_Args):
        app = "tests.test_distinction:_cli_mock_collapsing_app"
        kbv = False
        kbv_threshold = 0.1

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch.object(cli, "_default_commitment_extractor", lambda llm: _identity_extractor), \
         patch.object(distinction, "default_restatement_judge", lambda llm: _judge_always_correct):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_distinguish(_ThreshArgs())
        assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "FAIL: overall KBV rate" in out
