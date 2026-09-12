"""
Tests for contradish.resolution: the resolution operator.

No API key required: model_fn and candidate_proposer are deterministic
mocks, the same pattern tests/test_distinction.py and tests/test_cli.py
use for `contradish distinguish`.
"""
import json
from unittest.mock import patch

import pytest

from contradish.distinction import DistinctionPair, DistinctionProber, BUILTIN_DISTINCTION_PAIRS
from contradish.resolution import (
    ResolutionCandidate,
    ResolutionResult,
    discover_resolution,
    discover_resolutions_for_loss_map,
)


def _identity_extractor(question: str, answer: str) -> str:
    return answer


_PAIR = DistinctionPair(
    pair_id="test_pair",
    description="test distinction for the resolution operator",
    label_a="state A",
    label_b="state B",
    question_a="Question about A?",
    question_b="Question about B?",
    commit_a="answer for a",
    commit_b="answer for b",
)


def _one_candidate_proposer(pair: DistinctionPair, n: int) -> list[dict]:
    return [{
        "condition": "the hidden flag",
        "rationale": "the hidden flag distinguishes the two situations",
        "pole_a_statement": "The hidden flag is TRUE_A.",
        "pole_b_statement": "The hidden flag is TRUE_B.",
    }]


# -- structural sanity -------------------------------------------------------

def test_resolution_result_and_candidate_are_dataclasses():
    candidate = ResolutionCandidate(
        condition="c", rationale="r", pole_a_statement="a", pole_b_statement="b",
        direct_match_rate=1.0, flip_rate=1.0, causal_effect_size=1.0, n_probes=4,
    )
    result = ResolutionResult(
        pair_id="p", description="d", label_a="A", label_b="B",
        baseline_hold_rate=0.0, candidates=[candidate], best=candidate,
        resolved=True, validated_hold_rate=1.0, system_prompt_patch="patch text",
    )
    assert "RESOLVED" in result.report()
    assert "RESOLUTION OPERATOR" in result.report()
    assert "RESOLVED" in result.summary()

    d = result.to_dict()
    json.dumps(d)  # must not raise
    assert d["schema_version"] == "1.0"
    assert d["resolved"] is True
    assert d["best_condition"] == "c"
    assert d["candidates"][0]["causal_effect_size"] == 1.0


def test_discover_resolution_requires_llm_or_proposer():
    def model_fn(system_prompt: str, question: str) -> str:
        return "same answer"

    with pytest.raises(ValueError):
        discover_resolution(
            pair=_PAIR,
            model_fn=model_fn,
            commitment_extractor=_identity_extractor,
        )


# -- the honest-positive path: a candidate that both explains the collapse
#    and, once surfaced in the system prompt, measurably fixes it ---------

def test_discover_resolution_resolves_when_patch_measurably_helps():
    def model_fn(system_prompt: str, question: str) -> str:
        if "TRUE_A" in question:
            return "answer for a"
        if "TRUE_B" in question:
            return "answer for b"
        # No pole asserted in the question itself. Only responds correctly
        # if the resolution guidance has been surfaced in the system prompt.
        if "the hidden flag" in system_prompt:
            if question == _PAIR.question_a:
                return "answer for a"
            if question == _PAIR.question_b:
                return "answer for b"
        return "the standard answer applies in this situation"

    result = discover_resolution(
        pair=_PAIR,
        model_fn=model_fn,
        commitment_extractor=_identity_extractor,
        candidate_proposer=_one_candidate_proposer,
        validation_samples=1,
    )

    assert result.baseline_hold_rate == 0.0  # collapsed before any fix
    assert len(result.candidates) == 1
    best = result.candidates[0]
    assert best.direct_match_rate == 1.0
    assert best.flip_rate == 1.0
    assert best.causal_effect_size == 1.0

    assert result.resolved is True
    assert result.best is best
    assert result.validated_hold_rate == 1.0
    assert result.system_prompt_patch is not None
    assert "the hidden flag" in result.system_prompt_patch
    assert "RESOLVED" in result.report()


# -- the honest-negative path: model ignores the candidate entirely -------

def test_discover_resolution_reports_not_resolved_when_model_ignores_candidate():
    def model_fn(system_prompt: str, question: str) -> str:
        return "the standard answer applies in this situation"

    result = discover_resolution(
        pair=_PAIR,
        model_fn=model_fn,
        commitment_extractor=_identity_extractor,
        candidate_proposer=_one_candidate_proposer,
        validation_samples=1,
    )

    assert result.candidates[0].causal_effect_size == 0.0
    assert result.resolved is False
    assert result.system_prompt_patch is None
    assert result.validated_hold_rate is None
    assert "NOT RESOLVED" in result.report()


# -- the trap this module exists to avoid: a plausible-looking candidate
#    that passes the causal probe but whose patch doesn't actually help ---

def test_discover_resolution_refuses_to_ship_a_patch_that_does_not_help():
    def model_fn(system_prompt: str, question: str) -> str:
        # Responds correctly when the pole is asserted directly IN THE
        # QUESTION (so the causal flip test passes), but never looks at the
        # system prompt at all -- surfacing the candidate as guidance does
        # nothing, the way a plausible-sounding but inert fix would behave.
        if "TRUE_A" in question:
            return "answer for a"
        if "TRUE_B" in question:
            return "answer for b"
        return "the standard answer applies in this situation"

    result = discover_resolution(
        pair=_PAIR,
        model_fn=model_fn,
        commitment_extractor=_identity_extractor,
        candidate_proposer=_one_candidate_proposer,
        validation_samples=1,
    )

    # The causal probe alone looks perfect...
    assert result.candidates[0].causal_effect_size == 1.0
    # ...but the module still refuses to call it resolved, because the
    # patch was actually tested and it didn't move the needle.
    assert result.resolved is False
    assert result.system_prompt_patch is None
    assert result.validated_hold_rate == 0.0


def test_discover_resolution_reuses_supplied_baseline_hold_rate():
    calls = []

    def model_fn(system_prompt: str, question: str) -> str:
        calls.append((system_prompt, question))
        return "the standard answer applies in this situation"

    discover_resolution(
        pair=_PAIR,
        model_fn=model_fn,
        commitment_extractor=_identity_extractor,
        candidate_proposer=_one_candidate_proposer,
        baseline_hold_rate=0.42,
        validation_samples=1,
    )
    # No calls with the pair's own unmodified questions and no pole/patch
    # context should have happened -- baseline was supplied, not remeasured.
    bare_calls = [
        (sp, q) for sp, q in calls
        if q in (_PAIR.question_a, _PAIR.question_b) and sp == ""
    ]
    assert bare_calls == []


# -- discover_resolutions_for_loss_map: the DistinctionProber follow-up ----

def _mock_model_always_collapses(system_prompt: str, question: str) -> str:
    return "the standard answer applies in this situation"


def test_discover_resolutions_for_loss_map_only_runs_on_collapsing_pairs():
    pairs = BUILTIN_DISTINCTION_PAIRS["medication"]
    prober = DistinctionProber(
        model_fn=_mock_model_always_collapses,
        pairs=pairs,
        commitment_extractor=_identity_extractor,
        domain="medication",
    )
    loss_map = prober.measure(n_samples=1)
    for profile in loss_map.profiles.values():
        assert profile.collapse_rate() == 1.0  # every pair collapsed fully

    results = discover_resolutions_for_loss_map(
        loss_map=loss_map,
        pairs=pairs,
        model_fn=_mock_model_always_collapses,
        commitment_extractor=_identity_extractor,
        candidate_proposer=_one_candidate_proposer,
        collapse_threshold=0.3,
        validation_samples=1,
    )
    assert len(results) == len(pairs)
    for r in results:
        assert r.baseline_hold_rate == 0.0

    # Raising the threshold above every pair's actual collapse rate should
    # skip all of them.
    none_results = discover_resolutions_for_loss_map(
        loss_map=loss_map,
        pairs=pairs,
        model_fn=_mock_model_always_collapses,
        commitment_extractor=_identity_extractor,
        candidate_proposer=_one_candidate_proposer,
        collapse_threshold=1.1,
        validation_samples=1,
    )
    assert none_results == []


# -- CLI wiring: `contradish distinguish --resolve` ------------------------

class _FakeLLMWithCandidates:
    provider = "anthropic"

    def __init__(self, *a, **kw):
        pass

    def complete_json(self, prompt: str) -> dict:
        return {"candidates": [{
            "condition": "the hidden flag",
            "rationale": "test rationale",
            "pole_a_statement": "The hidden flag is TRUE_A.",
            "pole_b_statement": "The hidden flag is TRUE_B.",
        }]}


class _ResolveArgs:
    domain = "medication"
    app = "tests.test_resolution:_cli_mock_app"
    n_samples = 1
    threshold = None
    report = False
    json = True
    kbv = False
    kbv_threshold = None
    resolve = True
    resolve_collapse_threshold = 0.3
    resolve_candidates = 1
    resolve_samples = 1


def _cli_mock_app(question: str) -> str:
    return "the standard answer applies in this situation"


def test_cmd_distinguish_resolve_registered_in_argparse():
    import contradish.cli as cli
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["contradish", "distinguish", "--help"]
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
    finally:
        sys.argv = old_argv


def test_cmd_distinguish_resolve_runs_end_to_end(capsys):
    import contradish.cli as cli

    with patch("contradish.llm.LLMClient", _FakeLLMWithCandidates), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch.object(cli, "_default_commitment_extractor", lambda llm: _identity_extractor):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_distinguish(_ResolveArgs())
        assert exc.value.code == 0

    out = capsys.readouterr().out
    # Two JSON objects printed back to back: the loss map, then the
    # resolution results (same convention test_cmd_distinguish_kbv_json_*
    # uses for its own second JSON block).
    idx = out.index("}\n{")
    resolve_json_text = out[idx + 2:]
    d = json.loads(resolve_json_text)
    assert "resolution_results" in d
    assert len(d["resolution_results"]) >= 1
    for r in d["resolution_results"]:
        assert r["schema_version"] == "1.0"
        assert "resolved" in r
