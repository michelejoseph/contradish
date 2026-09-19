"""
Tests for the `contradish update` CLI command: warranted behavioral
updating (Behavioral Update Fidelity) wired end to end through
contradish.intervention_probe.

No API key or network required: LLMClient is patched with a fake (same
pattern test_distinction.py uses for `contradish distinguish`), and
default_change_judge/default_effect_judge are patched with deterministic
stubs rather than exercised for real.
"""
import json
import sys
from unittest.mock import patch

import pytest


class _FakeLLM:
    provider = "anthropic"
    def __init__(self, *a, **kw):
        pass


class _Args:
    case_file = None
    app = "tests.test_cli_update:_cli_mock_app"
    threshold = None
    json = True


def _cli_mock_app(system_prompt: str, question: str) -> str:
    window = "30" if "30 days" in system_prompt else "45"
    if question == "I bought this 32 days ago. Can I get a refund?":
        return "yes, eligible for a refund" if window == "45" else "no, past the window"
    if question == "I bought this 10 days ago. Can I get a refund?":
        return "yes, eligible for a refund"
    if question == "I bought this 50 days ago. Can I get a refund?":
        return "no, past the window"
    return "the customer pays for return shipping"


def _change_judge_stub(llm):
    def judge(question, before_answer, after_answer):
        return before_answer != after_answer
    return judge


def _effect_judge_stub(llm):
    def judge(question, after_answer, expected_effect):
        return "eligible" in after_answer
    return judge


def test_cmd_update_registered_in_argparse():
    import contradish.cli as cli

    old_argv = sys.argv
    try:
        sys.argv = ["contradish", "update", "--help"]
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
    finally:
        sys.argv = old_argv


def test_cmd_update_runs_builtin_demo_end_to_end_with_app(capsys):
    """No --case-file: runs the built-in ecommerce refund-window demo case
    (the NIST letter's own worked example) against a fake --app that models
    a perfectly-warranted update, and should report an exact match."""
    import contradish.cli as cli

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch("contradish.intervention_probe.default_change_judge", _change_judge_stub), \
         patch("contradish.intervention_probe.default_effect_judge", _effect_judge_stub):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_update(_Args())
        assert exc.value.code == 0

    out = capsys.readouterr().out
    d = json.loads(out)
    assert d["domain"] == "ecommerce"
    assert d["n_interventions"] == 1
    assert d["exact_match_rate"] == 1.0
    assert d["exact_match_rate_with_direction"] == 1.0
    assert d["interventions_with_excess"] == []
    assert d["interventions_with_deficit"] == []


def test_cmd_update_threshold_fails_when_exact_match_rate_too_low(capsys):
    import contradish.cli as cli

    class _RigidArgs(_Args):
        app = "tests.test_cli_update:_rigid_app"
        threshold = 0.5
        json = True

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch("contradish.intervention_probe.default_change_judge", _change_judge_stub), \
         patch("contradish.intervention_probe.default_effect_judge", _effect_judge_stub):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_update(_RigidArgs())
        assert exc.value.code == 1

    err_or_out = capsys.readouterr()
    assert "FAIL" in (err_or_out.out + err_or_out.err)


def _rigid_app(system_prompt: str, question: str) -> str:
    """Never updates -- rigid model, ignores the window change entirely."""
    return "no, past the window"


def test_cmd_update_case_file_yaml(tmp_path, capsys):
    import contradish.cli as cli

    # NOTE: the on-disk case-file path is named `_case_file_path`, not
    # `case_file` -- assigning `case_file = str(case_file)` inside the
    # `_FileArgs` class body below would shadow this name within the class's
    # own namespace and raise NameError, since a name a class body assigns
    # to is resolved via LOAD_NAME against the class namespace (then
    # globals), never against the enclosing function's locals, even for a
    # simple read-before-write on the same line.
    _case_file_path = tmp_path / "cases.yaml"
    _case_file_path.write_text(
        "interventions:\n"
        "  - intervention_id: refund-window-30-to-45\n"
        "    domain: ecommerce\n"
        "    before: \"Refunds accepted within 30 days, no exceptions.\"\n"
        "    after: \"Refunds accepted within 45 days, no exceptions.\"\n"
        "    justified:\n"
        "      refund_32_days:\n"
        "        question: \"Bought 32 days ago, refund?\"\n"
        "        expected_effect: \"yes, eligible\"\n"
        "    invariant:\n"
        "      refund_50_days: \"Bought 50 days ago, refund?\"\n"
    )

    class _FileArgs(_Args):
        case_file = str(_case_file_path)
        app = "tests.test_cli_update:_yaml_case_app"

    with patch("contradish.llm.LLMClient", _FakeLLM), \
         patch.object(cli, "_check_api_key", lambda: None), \
         patch("contradish.intervention_probe.default_change_judge", _change_judge_stub), \
         patch("contradish.intervention_probe.default_effect_judge", _effect_judge_stub):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_update(_FileArgs())
        assert exc.value.code == 0

    out = capsys.readouterr().out
    d = json.loads(out)
    assert d["n_interventions"] == 1
    assert "refund-window-30-to-45" in d["by_intervention"]


def _yaml_case_app(system_prompt: str, question: str) -> str:
    window = "30" if "30 days" in system_prompt else "45"
    if question == "Bought 32 days ago, refund?":
        return "yes, eligible" if window == "45" else "no"
    if question == "Bought 50 days ago, refund?":
        return "no"
    return "n/a"
