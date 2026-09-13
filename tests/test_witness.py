"""
Tests for contradish.witness: multi-witness convergence.

No API key required -- every witness here is a plain deterministic function,
not a judge-model call.
"""
from contradish.witness import WitnessPanel, build_witnessed


def test_requires_at_least_two_witnesses():
    try:
        WitnessPanel(witnesses={"only": lambda x: True})
        assert False, "expected ValueError for a single-witness panel"
    except ValueError:
        pass


def test_unanimous_agreement_returns_shared_verdict():
    panel = WitnessPanel(witnesses={"a": lambda x: True, "b": lambda x: True})
    combined = panel.combine()
    assert combined("q") is True
    report = panel.convergence_report()
    assert report.n_calls == 1
    assert report.n_agreed == 1
    assert report.n_disagreed == 0
    assert report.disagreement_rate == 0.0


def test_disagreement_returns_default_and_is_logged():
    panel = WitnessPanel(witnesses={"a": lambda x: True, "b": lambda x: False})
    combined = panel.combine(default_on_disagreement=False)
    assert combined("q") is False
    report = panel.convergence_report()
    assert report.n_disagreed == 1
    assert report.disagreement_rate == 1.0
    assert report.disagreements[0].votes == {"a": True, "b": False}


def test_default_on_disagreement_is_returned_verbatim():
    panel = WitnessPanel(witnesses={"a": lambda x: True, "b": lambda x: False})
    combined = panel.combine(default_on_disagreement="UNCERTAIN")
    assert combined("q") == "UNCERTAIN"


def test_majority_mode_returns_winner_but_still_flags_disagreement():
    panel = WitnessPanel(witnesses={
        "a": lambda x: True, "b": lambda x: True, "c": lambda x: False,
    })
    combined = panel.combine(require_unanimous=False)
    assert combined("q") is True   # majority wins
    report = panel.convergence_report()
    # not unanimous, even though a winner was returned -- this module logs
    # that distinction rather than hiding it inside a silent majority vote
    assert report.n_disagreed == 1


def test_reset_clears_recorded_calls():
    panel = WitnessPanel(witnesses={"a": lambda x: True, "b": lambda x: True})
    combined = panel.combine()
    combined("q")
    panel.reset()
    assert panel.convergence_report().n_calls == 0


def test_every_witness_is_called_every_time_no_short_circuit():
    calls = {"a": 0, "b": 0}

    def wa(x):
        calls["a"] += 1
        return True

    def wb(x):
        calls["b"] += 1
        return True

    panel = WitnessPanel(witnesses={"a": wa, "b": wb})
    combined = panel.combine()
    combined("q")
    assert calls == {"a": 1, "b": 1}


def test_combined_function_forwards_args_and_kwargs():
    panel = WitnessPanel(witnesses={
        "a": lambda pair, restatement: restatement == "yes",
        "b": lambda pair, restatement: restatement == "yes",
    })
    combined = panel.combine()
    assert combined("some_pair", restatement="yes") is True
    assert combined("some_pair", restatement="no") is False


def test_calls_property_returns_a_copy_not_the_live_list():
    panel = WitnessPanel(witnesses={"a": lambda x: True, "b": lambda x: True})
    combined = panel.combine()
    combined("q")
    calls = panel.calls
    assert len(calls) == 1
    calls.append("not a real call")   # mutating the returned list...
    assert len(panel.calls) == 1      # ...must not affect the panel's own record


# -- build_witnessed ----------------------------------------------------------

class _FakeLLM:
    def __init__(self, provider, verdict):
        self.provider = provider
        self.fast_model = "fake-fast"
        self._verdict = verdict


def _fake_judge_factory(llm):
    """Mimics the shape of default_hedge_judge(llm) etc.: llm -> judge_fn."""
    def judge(answer_text: str) -> bool:
        return llm._verdict
    return judge


def test_build_witnessed_requires_at_least_two_llms():
    try:
        build_witnessed(_fake_judge_factory, [_FakeLLM("anthropic", True)])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_witnessed_agrees_when_underlying_judges_agree():
    llms = [_FakeLLM("anthropic", True), _FakeLLM("openai", True)]
    combined, panel = build_witnessed(_fake_judge_factory, llms)
    assert combined("some answer") is True
    report = panel.convergence_report()
    assert report.n_disagreed == 0
    assert set(report.witness_names) == {"anthropic:fake-fast", "openai:fake-fast"}


def test_build_witnessed_disagrees_falls_back_to_default():
    llms = [_FakeLLM("anthropic", True), _FakeLLM("openai", False)]
    combined, panel = build_witnessed(_fake_judge_factory, llms, default_on_disagreement=False)
    assert combined("some answer") is False
    assert panel.convergence_report().n_disagreed == 1


def test_build_witnessed_custom_names_are_used():
    llms = [_FakeLLM("anthropic", True), _FakeLLM("openai", True)]
    combined, panel = build_witnessed(_fake_judge_factory, llms, names=["reviewer_1", "reviewer_2"])
    assert set(panel.witnesses) == {"reviewer_1", "reviewer_2"}
