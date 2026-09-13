"""
Tests for contradish.decision_boundary.

No API key required: recover_boundary_via_binary_search takes a plain
Python callable as its oracle (same swappable-judge pattern as
default_hedge_judge elsewhere), so these tests exercise it directly with
deterministic synthetic oracles.
"""
from contradish.decision_boundary import (
    BoundaryDiscrepancyReport,
    BoundaryLadder,
    BoundaryRecoveryResult,
    illustrative_ladder,
    quantify_boundary_discrepancy,
    recover_boundary_from_observations,
    recover_boundary_via_binary_search,
)


# ── BoundaryLadder ─────────────────────────────────────────────────────────

def test_empty_rungs_is_rejected():
    try:
        BoundaryLadder(commitment_id="c", domain="d", dimension="x", rungs=[],
                        decision_a="a", decision_b="b", legitimate_boundary_index=0)
        assert False, "expected ValueError for empty rungs"
    except ValueError:
        pass


def test_out_of_range_boundary_index_is_rejected():
    try:
        BoundaryLadder(commitment_id="c", domain="d", dimension="x", rungs=[0, 1, 2],
                        decision_a="a", decision_b="b", legitimate_boundary_index=99)
        assert False, "expected ValueError for out-of-range legitimate_boundary_index"
    except ValueError:
        pass


def test_boundary_index_at_extremes_is_valid_but_degenerate():
    # 0 and len(rungs) are the two degenerate-but-legal specifications
    BoundaryLadder(commitment_id="c", domain="d", dimension="x", rungs=[0, 1, 2],
                    decision_a="a", decision_b="b", legitimate_boundary_index=0)
    BoundaryLadder(commitment_id="c", domain="d", dimension="x", rungs=[0, 1, 2],
                    decision_a="a", decision_b="b", legitimate_boundary_index=3)


def test_illustrative_ladder_has_expected_shape():
    ladder = illustrative_ladder()
    assert ladder.domain == "illustrative"
    assert len(ladder.rungs) == 11
    assert 0 < ladder.legitimate_boundary_index < len(ladder.rungs)
    assert isinstance(ladder.summary(), str)


# ── recover_boundary_from_observations ───────────────────────────────────────

def test_clean_single_transition_is_recovered():
    decisions = ["a", "a", "a", "b", "b"]
    result = recover_boundary_from_observations(decisions, "a", "b")
    assert result.recovered_boundary_index == 3
    assert result.regime == "single_transition"
    assert result.monotonic is True


def test_no_transition_reports_always_a():
    result = recover_boundary_from_observations(["a", "a", "a"], "a", "b")
    assert result.recovered_boundary_index is None
    assert result.regime == "always_a"


def test_no_transition_reports_always_b():
    result = recover_boundary_from_observations(["b", "b", "b"], "a", "b")
    assert result.regime == "always_b"


def test_oscillation_is_unstable_not_forced_to_a_number():
    decisions = ["a", "b", "a", "b", "a"]
    result = recover_boundary_from_observations(decisions, "a", "b")
    assert result.recovered_boundary_index is None
    assert result.regime == "unstable"
    assert result.monotonic is False
    assert len(result.transition_indices) > 1


def test_fewer_than_two_measured_is_insufficient_data():
    result = recover_boundary_from_observations([None, "a", None], "a", "b")
    assert result.regime == "insufficient_data"
    assert result.recovered_boundary_index is None


def test_none_entries_are_skipped_not_treated_as_a_decision():
    # rung 1 unmeasured; transition should still be found at rung 3 using
    # only the measured rungs (0, 2, 3, 4)
    decisions = ["a", None, "a", "b", "b"]
    result = recover_boundary_from_observations(decisions, "a", "b")
    assert result.recovered_boundary_index == 3
    assert result.decision_at_rungs[1] is None


# ── recover_boundary_via_binary_search ────────────────────────────────────────

def _ladder_oracle(boundary, decision_a="a", decision_b="b"):
    def oracle(i):
        return decision_a if i < boundary else decision_b
    return oracle


def test_binary_search_finds_exact_boundary_matching_full_sweep():
    n = 20
    true_boundary = 7
    oracle = _ladder_oracle(true_boundary)
    result = recover_boundary_via_binary_search(oracle, n, "a", "b")
    assert result.recovered_boundary_index == true_boundary
    assert result.regime == "single_transition"

    # cross-check against the honest full-sweep baseline
    full_sweep = [oracle(i) for i in range(n)]
    baseline = recover_boundary_from_observations(full_sweep, "a", "b")
    assert result.recovered_boundary_index == baseline.recovered_boundary_index


def test_binary_search_uses_far_fewer_queries_than_a_full_sweep():
    n = 64
    oracle = _ladder_oracle(41)
    result = recover_boundary_via_binary_search(oracle, n, "a", "b")
    assert result.n_queries < n
    assert result.n_queries <= 10   # log2(64) = 6, plus endpoints and verification


def test_binary_search_boundary_at_start_is_always_b():
    oracle = _ladder_oracle(0)   # decision_b at every rung
    result = recover_boundary_via_binary_search(oracle, 10, "a", "b")
    assert result.regime == "always_b"
    assert result.recovered_boundary_index is None
    assert result.n_queries == 2   # just the two endpoint checks


def test_binary_search_boundary_past_end_is_always_a():
    oracle = _ladder_oracle(999)   # decision_a at every rung in range
    result = recover_boundary_via_binary_search(oracle, 10, "a", "b")
    assert result.regime == "always_a"


def test_binary_search_rejects_too_few_rungs():
    try:
        recover_boundary_via_binary_search(_ladder_oracle(1), 1, "a", "b")
        assert False, "expected ValueError for n_rungs < 2"
    except ValueError:
        pass


def test_binary_search_catches_oscillation_right_at_the_boundary():
    """The search itself converges to rung 7 (it's a genuine leftmost 'b'
    from the search's point of view), but rung 6 -- one step past the
    boundary the search path never had to visit -- flips back to 'a'. That
    is real non-monotonicity, and verify's boundary+1 check is exactly the
    probe capable of catching it (boundary-1/boundary themselves are
    guaranteed consistent by the search's own termination, so they can't
    reveal this -- see the module note)."""
    def flaky_oracle(i):
        arr = ["a", "a", "a", "a", "a", "b", "a", "b", "b", "b"]
        return arr[i]

    result = recover_boundary_via_binary_search(flaky_oracle, 10, "a", "b", verify=True)
    assert result.monotonic is False
    assert result.regime == "unstable_near_boundary"
    assert result.recovered_boundary_index is None


def test_binary_search_without_verify_skips_the_local_check():
    def flaky_oracle(i):
        arr = ["a", "a", "a", "a", "a", "b", "a", "b", "b", "b"]
        return arr[i]

    result = recover_boundary_via_binary_search(flaky_oracle, 10, "a", "b", verify=False)
    assert result.monotonic is True
    assert result.recovered_boundary_index == 7


# ── quantify_boundary_discrepancy ────────────────────────────────────────────

def _ladder(legit_boundary, n=11):
    return BoundaryLadder(commitment_id="c1", domain="test", dimension="x",
                           rungs=list(range(n)), decision_a="a", decision_b="b",
                           legitimate_boundary_index=legit_boundary)


def test_exact_match_has_zero_displacement():
    ladder = _ladder(5)
    recovery = recover_boundary_from_observations(["a"] * 5 + ["b"] * 6, "a", "b")
    report = quantify_boundary_discrepancy(ladder, recovery)
    assert report.displacement == 0
    assert report.direction == "exact"
    assert report.normalized_displacement == 0.0


def test_recovered_later_than_legitimate_shifts_toward_a():
    ladder = _ladder(3)
    recovery = recover_boundary_from_observations(["a"] * 6 + ["b"] * 5, "a", "b")   # boundary at 6
    report = quantify_boundary_discrepancy(ladder, recovery)
    assert report.displacement == 3
    assert report.direction == "shifted_toward_a"
    assert report.normalized_displacement == round(3 / 11, 4)


def test_recovered_earlier_than_legitimate_shifts_toward_b():
    ladder = _ladder(8)
    recovery = recover_boundary_from_observations(["a"] * 2 + ["b"] * 9, "a", "b")   # boundary at 2
    report = quantify_boundary_discrepancy(ladder, recovery)
    assert report.displacement == -6
    assert report.direction == "shifted_toward_b"


def test_undetermined_recovery_gives_undetermined_discrepancy():
    ladder = _ladder(5)
    recovery = recover_boundary_from_observations(["a", "b", "a", "b"], "a", "b")   # unstable
    report = quantify_boundary_discrepancy(ladder, recovery)
    assert report.displacement is None
    assert report.direction == "undetermined"
    assert report.regime == "unstable"


def test_report_and_summary_render_as_strings():
    ladder = _ladder(5)
    recovery = recover_boundary_from_observations(["a"] * 5 + ["b"] * 6, "a", "b")
    report = quantify_boundary_discrepancy(ladder, recovery)
    assert isinstance(report.summary(), str)
    text = report.report()
    assert isinstance(text, str)
    assert "c1" in text
