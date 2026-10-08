"""
The invariants of Governance Fidelity are independent failure modes:
each witness agent has exactly one defect, each probe detects exactly one
invariant, and the separability matrix is diagonal.
"""
from contradish import invariants as inv


def test_ideal_agent_passes_every_probe():
    row = inv.separability_matrix()["ideal"]
    assert all(row.values()), row


def test_each_witness_fails_exactly_its_intended_invariants():
    matrix = inv.separability_matrix()
    for name, (_, intended) in inv.WITNESSES.items():
        failed = {k for k, passed in matrix[name].items() if not passed}
        assert failed == intended, (name, failed, intended)


def test_every_built_invariant_is_violated_by_some_witness():
    matrix = inv.separability_matrix()
    for i in inv.INVARIANTS:
        if i.probe is None:
            continue
        assert {n for n, row in matrix.items() if not row[i.key]}, f"no witness violates {i.key}"


def test_no_two_invariants_have_the_same_failure_signature():
    matrix = inv.separability_matrix()
    keys = [i.key for i in inv.INVARIANTS if i.probe is not None]
    sigs = {k: frozenset(n for n, row in matrix.items() if not row[k]) for k in keys}
    assert len(set(sigs.values())) == len(keys), sigs


def test_magnitude_is_declared_unbuilt_not_silently_missing():
    mag = [i for i in inv.INVARIANTS if i.key == "magnitude"]
    assert len(mag) == 1 and mag[0].probe is None and "NOT BUILT" in mag[0].statement


def test_matrix_renders():
    assert "obeys_any_source" in inv.format_matrix()
