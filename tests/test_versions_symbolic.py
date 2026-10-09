"""Permissions, the symbolic complete diff, pinned versions, whole-run certificates, perspectives."""

import copy
import json
import random

import pytest

from contradish.evidence import seal
from contradish.evidence_check import check_run
from contradish.policy_program import PolicyProgram, ProgramError, load_builtin_program
from contradish.symbolic import OutsideFragment, decompose, diff_versions, regions
from contradish.versions import (
    atlas, load_perspectives, load_pinned, perspective_matrix, pin, run_certificate, transition_authority,
    version_witness,
)


@pytest.fixture(scope="module")
def returns():
    return load_builtin_program("returns")


def _v2(p, u):
    ok, _ = p.split_edits(u)
    return p.apply(ok)


def _pins(p, u):
    return (pin(p, "v1", "tester", "fixture"), pin(_v2(p, u), "v2", "tester", "fixture", issued_by=u.source))


# ── symbolic decomposition agrees with concrete evaluation ───────────────────

@pytest.mark.parametrize("uid", ["restocking_fee_20", "window_45", "exchanges_electronics",
                                 "exchange_required_gold", "no_label_final_window", "exchange_closed_late"])
def test_symbolic_norms_equal_concrete_at_every_cell(returns, uid):
    p, U = returns
    q = _v2(p, U[uid])
    d = diff_versions(p, q)
    for cell, norms, _ in d.cells:
        rep = cell.rep()
        assert cell.contains(rep)
        for k in d.steps:
            assert norms[0][k].norm_at(rep).same(p.run_norm(k, p.complete_state(rep))[0])
            assert norms[1][k].norm_at(rep).same(q.run_norm(k, q.complete_state(rep))[0])


def _random_situation(p, rng):
    s = {}
    for f, spec in p.facts.items():
        t = spec["type"]
        if t == "bool":
            s[f] = rng.random() < 0.5
        elif t == "enum":
            s[f] = rng.choice(spec["values"])
        elif t == "int":
            s[f] = rng.randint(spec["min"], spec["max"])
        else:
            s[f] = round(rng.uniform(spec["min"], 60), 2)
    return s


@pytest.mark.parametrize("uid", ["restocking_fee_20", "window_45", "exchange_closed_late"])
def test_cells_partition_the_space_and_classification_is_constant(returns, uid):
    """Every random situation lies in exactly one cell, and its concrete norms match that cell's."""
    p, U = returns
    q = _v2(p, U[uid])
    d = diff_versions(p, q)
    rng = random.Random(3)
    for _ in range(400):
        s = _random_situation(p, rng)
        hits = [(c, n, cls) for c, n, cls in d.cells if c.contains(s)]
        assert len(hits) == 1, s
        cell, norms, cls = hits[0]
        for k in d.steps:
            a = p.run_norm(k, p.complete_state(s))[0]
            b = q.run_norm(k, q.complete_state(s))[0]
            assert norms[0][k].norm_at(s).same(a) and norms[1][k].norm_at(s).same(b)
            if cls[k]["kind"].startswith("unchanged"):
                assert a.same(b)
            if cls[k]["obligation"] or cls[k]["permission"]:
                assert a.modality != b.modality


def test_null_transition_has_no_changes(returns):
    p, U = returns
    d = diff_versions(p, _v2(p, U["control_reworded"]))
    assert d.summary()["steps_that_change"] == []


def test_outside_fragment_is_refused():
    spec = {"id": "x", "facts": {"a": {"type": "int", "min": 0, "max": 9}, "b": {"type": "int", "min": 0, "max": 9}},
            "clauses": {"C": {"steps": {"s": {"tool": "t", "when": ["<", ["*", "a", "b"], 5]}}}}}
    with pytest.raises(OutsideFragment):
        decompose([PolicyProgram(spec)])
    spec2 = {"id": "x", "facts": {"a": {"type": "int", "min": 0, "max": 9}},
             "clauses": {"C": {"steps": {"s": {"tool": "t", "when": ["<", ["round", "a", 0], 5]}}}}}
    with pytest.raises(OutsideFragment):
        decompose([PolicyProgram(spec2)])


# ── permissions are tracked separately from obligations ──────────────────────

def _labels(p, U, uid):
    d = diff_versions(p, _v2(p, U[uid]))
    return {(k, r["change"]) for k in d.steps for r in regions(d, k)}


def test_permission_and_obligation_channels(returns):
    p, U = returns
    assert _labels(p, U, "exchange_closed_late") == {("exchange", "permission revoked")}
    assert _labels(p, U, "exchanges_electronics") == {("exchange", "permission granted")}
    assert _labels(p, U, "exchange_required_gold") == {("exchange", "obligation gained")}
    assert ("label", "obligation lost; permission revoked") in _labels(p, U, "no_label_final_window")


def test_regions_are_exact_conditions(returns):
    p, U = returns
    d = diff_versions(p, _v2(p, U["exchange_closed_late"]))
    (r,) = regions(d, "exchange")
    assert r["conditions"] == ["category = 'apparel' ∧ 20 ≤ days_since_delivery ≤ 30"]


# ── pinned versions ──────────────────────────────────────────────────────────

def test_pin_round_trip_and_tamper(returns):
    p, _ = returns
    pv = pin(p, "v1", "Legal", "clause-by-clause review", evidence="memo-17")
    back = load_pinned(json.loads(json.dumps(pv.to_dict())))
    assert back.digest == pv.digest
    bad = pv.to_dict()
    bad["program"]["clauses"]["R1"]["params"]["return_window_days"] = 31
    with pytest.raises(ProgramError):
        load_pinned(bad)
    bad = pv.to_dict()
    bad["verification"]["verified_by"] = ""
    with pytest.raises(ProgramError):
        load_pinned(bad)
    with pytest.raises(ValueError):
        pin(p, "v1", "", "")


def test_transition_authority(returns):
    p, U = returns
    v1 = pin(p, "v1", "t", "f")
    v2 = pin(_v2(p, U["window_45"]), "v2", "t", "f", issued_by="customer")
    assert transition_authority(v1, v2)["status"] == "unauthorized"
    v2 = pin(_v2(p, U["window_45"]), "v2", "t", "f", issued_by="policy_owner")
    assert transition_authority(v1, v2)["status"] == "authorized"


# ── whole-run certificates and the independent checker ───────────────────────

@pytest.mark.parametrize("uid", ["restocking_fee_20", "window_45", "exchange_closed_late", "customer_claims"])
def test_faithful_run_is_proved_and_verified(returns, uid):
    p, U = returns
    v1, v2 = _pins(p, U[uid])
    c = run_certificate(v1, v2, version_witness("faithful"), "witness:faithful")
    assert c["claim"]["proved"]
    v = check_run(json.loads(json.dumps(c)))
    assert v["verdict"] == "VERIFIED", v["problems"]


def test_failures_are_certified_as_not_proved(returns):
    p, U = returns
    v1, v2 = _pins(p, U["exchange_closed_late"])
    for mode in ("rigid", "stale_permissions"):
        c = run_certificate(v1, v2, version_witness(mode), "witness:" + mode)
        assert not c["claim"]["proved"] and not c["claim"]["every_required_change_happened"]
        assert check_run(c)["verdict"] == "VERIFIED"   # an honest "not proved" verifies


def test_run_certificate_tampering_is_rejected(returns):
    p, U = returns
    v1, v2 = _pins(p, U["restocking_fee_20"])
    c = run_certificate(v1, v2, version_witness("faithful"), "witness:faithful")

    bad = copy.deepcopy(c)
    bad["observations"] = bad["observations"][1:]               # drop a cell's evidence
    assert check_run(seal(bad))["verdict"] == "REJECTED"

    bad = copy.deepcopy(c)
    bad["derivation"]["changes"] = bad["derivation"]["changes"][1:]   # hide a required change
    assert check_run(seal(bad))["verdict"] == "REJECTED"

    bad = copy.deepcopy(c)
    o = next(o for o in bad["observations"] if any(x["tool"] == "issue_refund" for x in o["after"]))
    for x in o["after"]:
        if x["tool"] == "issue_refund":
            x["args"]["amount"] = 1.23                          # behavior that is wrong, claim still "proved"
    assert check_run(seal(bad))["verdict"] == "REJECTED"

    bad = copy.deepcopy(c)
    bad["after"]["program"]["clauses"]["R3"]["params"]["restocking_fee_pct"] = 25
    assert check_run(seal(bad))["verdict"] == "REJECTED"        # pin digest no longer matches

    bad = copy.deepcopy(c)
    bad["claim"]["proved"] = False
    assert check_run(seal(bad))["verdict"] == "REJECTED"

    assert check_run(c)["verdict"] == "VERIFIED"


# ── perspectives: alternative frames, consensus, contest, switching ──────────

def test_atlas_distance_is_a_pseudometric():
    _, frames = load_perspectives("dietary")
    a = atlas(frames)
    ids = [f.id for f in frames]
    D = a["distance"]
    for i in ids:
        assert D[i][i] == 0
        for j in ids:
            assert D[i][j] == D[j][i]
            for k in ids:
                assert D[i][k] <= D[i][j] + D[j][k] + 1e-12
    assert a["consensus_cases"] + a["contested_cases"] == a["cases"]


def test_atlas_blocs_match_the_cited_readings():
    _, frames = load_perspectives("dietary")
    a = atlas(frames)
    pork = [r for r in a["contested"] if r["step"] == "flag_conflict" and "meat = 'pork'" in r["condition"]
            and "not killed_for_you" in r["condition"]]
    assert pork
    flagged = next(b for b in pork[0]["blocs"] if pork[0]["norms"][b[0]]["modality"] == "O")
    assert set(flagged) == {"leviticus_11", "quran_2_173", "manusmriti_5_48"}


def test_frame_switching():
    _, frames = load_perspectives("dietary")
    m = perspective_matrix(frames, lambda: version_witness("faithful"))
    assert all(v["proved"] for v in m.values())
    m = perspective_matrix(frames, lambda: version_witness("leaky"))
    assert not m[("leviticus_11", "mark_7_acts_10")]["proved"]
    assert all(m[(f.id, f.id)]["proved"] for f in frames)        # re-declaring the same frame changes nothing
