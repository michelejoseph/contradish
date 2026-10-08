"""
Independent-implementation validation of Behavioral Update Fidelity.

contradish.reference is a second implementation of the scorer, written from
docs/BUF-SPEC.md over plain dicts and exact rationals, importing nothing from
the package. These tests hold the production scorer to it on every contract
in a finite universe, hold both to the language-neutral vectors in
conformance/vectors.json, and hold the VECTORS to a bug library: every
mutant scorer must fail at least one vector.
"""
import ast
from pathlib import Path

from contradish import conformance as cf
from contradish import reference as ref
from contradish.transition import GoverningState, Source, TransitionCase, Update
from contradish.transition_sequence import SequenceContract

VECTORS = cf.load_vectors()


def test_reference_imports_nothing_from_the_package():
    src = Path(ref.__file__).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or "").startswith("contradish"), node.module
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("contradish") for a in node.names)


def _differential(universe):
    n = 0
    for cases, prev, cur in universe:
        want = cf.encode_result(ref.score(cases, prev, cur))
        bad = cf.diff(want, cf.production_scorer(cases, prev, cur))
        assert not bad, (bad, cases, prev, cur)
        n += 1
    return n


def _grid(n, labels, grounds=(None,), stride=1):
    for cases in ref.contracts(n, labels, grounds):
        for p, q in ref.observations(cases, labels, stride):
            yield cases, p, q


def test_production_matches_reference_two_cases_two_outcomes_with_grounds():
    assert _differential(_grid(2, ("a", "b"), (None, ("g1",), ("g2",)))) > 100_000


def test_production_matches_reference_two_cases_three_outcomes():
    assert _differential(_grid(2, ("a", "b", "c"), stride=2)) > 50_000


def test_production_matches_reference_three_cases_sampled():
    assert _differential(_grid(3, ("a", "b"), (None, ("g1",)), stride=97)) > 10_000


def test_production_passes_every_vector():
    assert cf.check_scorer(cf.production_scorer, VECTORS) == []


def test_reference_passes_every_vector():
    assert cf.check_scorer(ref.score, VECTORS) == []


def test_vectors_are_deterministic_and_current(tmp_path):
    fresh = cf.build(tmp_path / "v.json")
    assert fresh == VECTORS, "conformance/vectors.json is stale: run python -m contradish.conformance --generate"


def test_every_mutant_is_killed_by_the_vectors():
    survivors = [name for name, m in cf.MUTANTS.items() if not cf.check_scorer(m, VECTORS)]
    assert survivors == []


def test_every_mutant_has_a_recorded_killer():
    for name in cf.MUTANTS:
        assert name in VECTORS["mutant_killers"], name


def test_vectors_cover_every_status_and_fate():
    statuses, fates = set(), set()
    for v in VECTORS["score_vectors"]:
        statuses |= set(v["expect"]["status"].values())
        for a, b, f in v["expect"]["broken"]:
            fates.add(f)
    assert statuses == set(ref.STATUSES)
    assert fates == {ref.SURVIVE, ref.COLLAPSE, ref.EMERGE, ref.STAY_MERGED}


def test_vectors_use_both_spellings_of_unclear():
    any_missing = any(set(c["id"] for c in v["cases"]) - set(v["current"]) for v in VECTORS["score_vectors"])
    any_explicit = any(ref.BOT in v["current"].values() for v in VECTORS["score_vectors"])
    assert any_missing and any_explicit


def _production_trajectory(initial, updates, sources, grounds):
    cases = [TransitionCase(cid, cid, lab, lab, grounds=list(grounds.get(cid, []))) for cid, lab in initial.items()]
    srcs = {k: Source(k, "", list(g)) for k, g in sources.items()}
    ups = [Update(f"u{i}", u["source"], "system", "", dict(u["asserts"])) for i, u in enumerate(updates)]
    return SequenceContract("conf", GoverningState("g", ""), cases, srcs, ups).trajectory()


def test_production_sequence_semantics_match_sequence_vectors():
    assert cf.check_sequences(VECTORS, trajectory_fn=_production_trajectory) == []


def test_reference_theorems_hold_on_the_finite_universe():
    results = ref.check_theorems("fast")
    broken = [(r["id"], r["counterexample"]) for r in results if not r["ok"]]
    assert broken == []
    assert all(r["checked"] > 0 for r in results)
