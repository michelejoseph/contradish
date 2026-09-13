"""
Tests for contradish.provenance: Provenance Collapse.

No API key required -- model_fn, usage_judge, and hedge_judge are all
deterministic mocks, same pattern as tests/test_sacrifice.py.
"""
from contradish.provenance import ProvenanceClaim, measure_provenance

_CLAIM = ProvenanceClaim(
    claim_id="test_claim",
    domain="test",
    topic="test topic",
    source_strength="single anonymous unverified comment",
    context_block="Context: a single anonymous, unverified comment claims that X is true.",
    claim_content="X is true",
    probe_question="Is X true?",
)


def _confident_using_model(system_prompt: str, question: str) -> str:
    return "Yes, X is definitely true, no question about it."


def _hedged_using_model(system_prompt: str, question: str) -> str:
    return "One unverified source suggests X might be true, but this isn't confirmed."


def _ignoring_model(system_prompt: str, question: str) -> str:
    return "Generally speaking, unrelated established fact Y holds."


def _always_used_judge(claim, probe_question, answer) -> bool:
    return True


def _never_used_judge(claim, probe_question, answer) -> bool:
    return False


def _always_confident_hedge_judge(answer: str) -> bool:
    return True


def _always_hedged_hedge_judge(answer: str) -> bool:
    return False


# -- the core case: used + confident = collapse -------------------------------

def test_collapse_when_claim_used_and_stated_with_confidence():
    report = measure_provenance(
        _confident_using_model, [_CLAIM], _always_used_judge,
        _always_confident_hedge_judge, n_samples=1,
    )
    profile = report.profiles["test_claim"]
    assert profile.n_used == 1
    assert profile.n_collapsed == 1
    assert profile.collapse_rate == 1.0
    assert report.overall_collapse_rate == 1.0
    assert report.most_collapsing == "test_claim"


# -- used but hedged: NOT a collapse -------------------------------------------

def test_no_collapse_when_claim_used_but_hedged():
    report = measure_provenance(
        _hedged_using_model, [_CLAIM], _always_used_judge,
        _always_hedged_hedge_judge, n_samples=1,
    )
    profile = report.profiles["test_claim"]
    assert profile.n_used == 1
    assert profile.n_collapsed == 0
    assert profile.collapse_rate == 0.0


# -- claim never used: hedge_judge's answer is irrelevant ---------------------

def test_no_collapse_when_claim_never_used_even_with_confident_hedge_judge():
    report = measure_provenance(
        _ignoring_model, [_CLAIM], _never_used_judge,
        _always_confident_hedge_judge, n_samples=1,
    )
    profile = report.profiles["test_claim"]
    assert profile.n_used == 0
    assert profile.n_collapsed == 0
    assert profile.collapse_rate == 0.0


# -- collapse_rate is of USED samples, not all samples -------------------------

def test_collapse_rate_denominator_is_used_samples_not_total_samples():
    report = measure_provenance(
        _confident_using_model, [_CLAIM], _always_used_judge,
        _always_confident_hedge_judge, n_samples=3,
    )
    profile = report.profiles["test_claim"]
    assert profile.n_samples == 3
    assert profile.n_used == 3
    assert profile.n_collapsed == 3
    assert profile.collapse_rate == 1.0


def test_partial_usage_gives_partial_collapse_rate():
    calls = {"n": 0}

    def alternating_usage_judge(claim, probe_question, answer):
        calls["n"] += 1
        return calls["n"] % 2 == 1   # used on samples 1 and 3, not sample 2

    report = measure_provenance(
        _confident_using_model, [_CLAIM], alternating_usage_judge,
        _always_confident_hedge_judge, n_samples=3,
    )
    profile = report.profiles["test_claim"]
    assert profile.n_used == 2
    assert profile.n_collapsed == 2
    assert profile.collapse_rate == 1.0   # of the 2 used, both collapsed


def test_report_shape_and_domain_inference():
    report = measure_provenance(
        _confident_using_model, [_CLAIM], _always_used_judge,
        _always_confident_hedge_judge,
    )
    assert report.domain == "test"
    assert report.n_claims == 1
    assert isinstance(report.summary(), str)
    assert isinstance(report.report(), str)
