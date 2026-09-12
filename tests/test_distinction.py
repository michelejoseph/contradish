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


def test_cmd_distinguish_runs_end_to_end_with_mocked_llm(capsys):
    import contradish.cli as cli

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch.object(cli, "_default_commitment_extractor", lambda llm: _identity_extractor):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_distinguish(_Args())
        assert exc.value.code == 0

    out = capsys.readouterr().out
    d = json.loads(out)
    assert d["domain"] == "medication"
    assert d["profiles"][d["most_fragile"]]["collapse_rate"] == 0.0
