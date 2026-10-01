"""
Tests for contradish.counterfactual: the record format and metrics, the two
pre-registered tests, and (when a STATE-Bench checkout is available) the
Contradish x STATE-Bench derivation.

No model calls anywhere. The STATE-Bench tests need a checkout; point
STATE_BENCH_ROOT at one, otherwise they skip.
"""

import json
import os
import sys

import pytest

from contradish.counterfactual.core import (
    Record, all_model_metrics, criterion_split, dump_records, is_criterion_key,
    load_records, model_metrics,
)

np = pytest.importorskip("numpy")

from contradish.counterfactual.analysis import (  # noqa: E402
    analyze, format_report, h1_update_vs_control, h2_incremental_validity,
)
from contradish.counterfactual.simulate import power, simulate_records  # noqa: E402


def R(model, task, cond, role, run, pb, pa=None):
    return Record("b", model, "d", task, cond, role, run, pb, pa)


# ── records and metrics ─────────────────────────────────────────────────────

def test_record_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        R("m", "t", "x", "weird", 0, True)
    with pytest.raises(ValueError):
        R("m", "t", "a", "changed", 0, False)            # changed needs pass_amended
    recs = [R("m", "t", "base", "base", 0, True), R("m", "t", "a", "changed", 0, False, True)]
    path = tmp_path / "r.jsonl"
    dump_records(recs, path)
    assert load_records(path) == recs
    (tmp_path / "r.json").write_text(json.dumps([r.to_dict() for r in recs]))
    assert load_records(tmp_path / "r.json") == recs


def test_metrics_condition_on_mastery():
    recs = []
    # t1 mastered (2/2), t2 not mastered (1/2), t3 never passed.
    for k, (a, b, c) in enumerate([(True, True, False), (True, False, False)]):
        recs += [R("m", "t1", "base", "base", k, a), R("m", "t2", "base", "base", k, b),
                 R("m", "t3", "base", "base", k, c)]
    # Under amendment: t1 updates once in two runs, reproducing the old outcome the other time.
    recs += [R("m", "t1", "a", "changed", 0, False, True), R("m", "t1", "a", "changed", 1, True, False)]
    # t2 fails to update, but is not mastered so must not count.
    recs += [R("m", "t2", "a", "changed", 0, True, False)]
    recs += [R("m", "t1", "c", "control", 0, True), R("m", "t1", "c", "control", 1, True)]
    m = model_metrics(recs, "b", "m", criterion=set())
    assert m.acc == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert m.n_changed == 1 and m.cuf == pytest.approx(0.5) and m.rigid == pytest.approx(0.5)
    assert m.ctrl == 1.0 and m.upd == pytest.approx(0.5) and m.buf == pytest.approx(0.5)
    assert model_metrics(recs, "b", "m", set(), mastery="majority").n_changed == 1


def test_buf_is_net_of_the_control_floor():
    # A model that fails 20% of ALL amended-or-reworded runs has no update problem.
    recs = []
    for j in range(10):
        t = f"t{j}"
        recs += [R("m", t, "base", "base", k, True) for k in range(2)]
        recs += [R("m", t, "a", "changed", k, False, k != 0 or j >= 2) for k in range(1)]
        recs += [R("m", t, "c", "control", 0, j >= 2)]
    m = model_metrics(recs, "b", "m", criterion=set())
    assert m.cuf == pytest.approx(0.8) and m.ctrl == pytest.approx(0.8)
    assert m.upd == pytest.approx(1.0)


def test_criterion_tasks_are_never_amended_tasks():
    recs = simulate_records(n_models=3, n_tasks=60, n_changed=10, n_invariant=15, runs=2, seed=3)
    crit = criterion_split(recs)
    amended = {r.task_key for r in recs if r.role != "base"}
    assert crit and not (crit & amended)
    assert all(is_criterion_key(k) for k in crit)
    m = all_model_metrics(recs)[0]
    assert m.n_criterion_tasks == len(crit) and m.n_acc_tasks == 60 - len(crit)


def test_reliability_failure_is_conditional_on_capability():
    recs = [R("m", "a", "base", "base", k, v) for k, v in enumerate([True, True, True])]
    recs += [R("m", "b", "base", "base", k, v) for k, v in enumerate([True, False, True])]
    recs += [R("m", "c", "base", "base", k, v) for k, v in enumerate([False, False, False])]
    crit = {("b", "d", "a"), ("b", "d", "b"), ("b", "d", "c")}
    m = model_metrics(recs, "b", "m", criterion=crit)
    assert m.n_capable == 2 and m.rel_fail == pytest.approx(0.5)     # c never passed: not counted


# ── H1 ──────────────────────────────────────────────────────────────────────

def test_h1_detects_an_update_hurdle_and_not_its_absence():
    with_hurdle = simulate_records(n_models=6, n_tasks=80, n_changed=19, n_invariant=20, seed=1)
    h = h1_update_vs_control(with_hurdle)
    assert h.excess > 0.05 and h.p_value < 0.01 and h.ci_low > 0
    assert 0 < h.rigid_share < 1
    none = simulate_records(n_models=6, n_tasks=80, n_changed=19, n_invariant=20, seed=1,
                            update_hurdle=False)
    h0 = h1_update_vs_control(none)
    assert abs(h0.excess) < 0.05 and h0.p_value > 0.05


def test_h1_reports_when_untestable():
    h = h1_update_vs_control([R("m", "t", "base", "base", 0, True)])
    assert h.p_value is None and h.note


def test_h1_exact_sign_flip_small_sample():
    recs = []
    for j in range(5):
        t = f"t{j}"
        recs += [R("m", t, "base", "base", 0, True), R("m", t, "a", "changed", 0, False, False),
                 R("m", t, "c", "control", 0, True)]
    h = h1_update_vs_control(recs)
    assert h.excess == 1.0 and h.p_value == pytest.approx(1 / 32)


# ── H2 ──────────────────────────────────────────────────────────────────────

def test_h2_needs_enough_models():
    recs = simulate_records(n_models=3, n_tasks=60, n_changed=10, n_invariant=15, seed=0)
    h = h2_incremental_validity(all_model_metrics(recs))
    assert h.p_value is None and "models" in h.note


def test_h2_recovers_a_planted_effect():
    recs = simulate_records(n_models=24, gamma=1.5, rho=0.5, seed=4)
    h = h2_incremental_validity(all_model_metrics(recs), permutations=4000)
    assert h.beta_buf < 0 and h.partial_r < -0.3 and h.p_value < 0.05
    assert h.delta_r2 > 0 and h.n_models + h.n_dropped == 24


def test_h2_false_positive_rate_is_controlled_when_buf_is_irrelevant():
    # Reliability failures generated from accuracy alone, with BUF correlated with accuracy.
    out = power(12, gamma=0.0, rho=0.8, sims=120, permutations=499, seed=11,
                n_tasks=118, n_changed=19, n_invariant=27)
    assert out["alpha_h2"] == 0.025
    assert out["h2_reject_rate"] <= 0.10        # generous bound for 120 simulations


def test_h2_permutes_within_benchmark():
    a = simulate_records(n_models=8, gamma=1.5, seed=5, benchmark="A")
    b = simulate_records(n_models=8, gamma=1.5, seed=6, benchmark="B")
    h = h2_incremental_validity(all_model_metrics(a + b), permutations=2000)
    assert h.n_benchmarks == 2 and h.n_models + h.n_dropped == 16 and h.p_value is not None


def test_analyze_and_report():
    recs = simulate_records(n_models=10, gamma=1.0, seed=2)
    result = analyze(recs, permutations=1000)
    assert set(result["verdict"]) == {"h1_supported", "h2_supported", "h1_testable", "h2_testable"}
    assert result["verdict"]["h1_testable"] and result["verdict"]["h2_testable"]
    assert "synthetic" in result["by_benchmark"]
    json.dumps(result)
    text = format_report(result)
    assert "H1" in text and "H2" in text and "REL_FAIL" in text


def test_cli_analyze_and_power(monkeypatch, capsys, tmp_path):
    from contradish import cli
    path = tmp_path / "records.jsonl"
    dump_records(simulate_records(n_models=8, gamma=1.0, seed=9), path)

    def run(argv):
        monkeypatch.setattr(sys, "argv", ["contradish"] + argv)
        with pytest.raises(SystemExit) as exc:
            cli.main()
        return exc.value.code

    assert run(["counterfactual", "analyze", str(path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_records"] > 0 and "h2" in out
    assert run(["counterfactual", "power", "--models", "8", "--gamma", "0", "--sims", "5"]) == 0
    assert "false-positive rate" in capsys.readouterr().out
    assert run(["counterfactual", "derive"]) == 2


# ── Contradish x STATE-Bench (needs a checkout) ─────────────────────────────

SB_ROOT = os.environ.get("STATE_BENCH_ROOT", "")
needs_sb = pytest.mark.skipif(
    not os.path.isdir(os.path.join(SB_ROOT, "state_bench")),
    reason="set STATE_BENCH_ROOT to a STATE-Bench checkout",
)


@pytest.fixture(scope="module")
def suite():
    from contradish.counterfactual import state_bench as S
    sb = S.load_state_bench(SB_ROOT)
    return S, sb, S.derive_suite(sb)


@needs_sb
def test_state_bench_every_amendment_is_valid(suite):
    S, sb, manifest = suite
    assert manifest["amendments"]
    for a in manifest["amendments"]:
        assert a["valid"], a
        assert a["unmatched_text_edits"] == []
        if a["control"]:
            assert a["counts"]["changed"] == 0 and a["counts"]["excluded"] == 0
        else:
            assert a["counts"]["changed"] >= 1


@needs_sb
def test_state_bench_constants_are_restored(suite):
    S, sb, _ = suite
    pol = sb.policies("customer_support")
    before = (pol.RESTOCKING_FEE_PCT, dict(pol.RESTOCKING_DISCOUNT_BY_TIER))
    with S.apply_amendment(sb, S.AMENDMENTS["cs_restocking_fee_20"]):
        assert pol.RESTOCKING_FEE_PCT == 20
    with S.apply_amendment(sb, S.AMENDMENTS["cs_gold_restocking_discount_75"]):
        assert pol.RESTOCKING_DISCOUNT_BY_TIER["gold"] == 0.75
    assert (pol.RESTOCKING_FEE_PCT, dict(pol.RESTOCKING_DISCOUNT_BY_TIER)) == before


@needs_sb
def test_state_bench_amended_prose_matches_amended_rule(suite):
    S, sb, manifest = suite
    a = S.AMENDMENTS["cs_restocking_fee_20"]
    tid = next(c["task_id"] for c in manifest["cases"] if c["amendment_id"] == a.id)
    cfg = S.amended_domain(sb, a)
    task = sb.task("customer_support", tid)
    env_data, _ = sb.env_loader.load_task_environment(cfg, task)
    text = json.dumps(cfg.environment_class(env_data.deep_copy(), now=task.now).get_policies({"topic": "return"}))
    assert "20% restocking fee" in text and "15% restocking fee" not in text


@needs_sb
def test_state_bench_changed_cases_are_self_consistent(suite):
    S, sb, manifest = suite
    changed = [S.case_from_dict(c) for c in manifest["cases"] if c["class"] == "changed"]
    assert len(changed) >= 20
    for case in changed:
        assert case.changed_fields and case.rigid_fields and case.tier in ("clean", "rewritten")
        a = S.AMENDMENTS[case.amendment_id]
        task = sb.task(case.domain, case.task_id)
        traj = sb.gold_trajectory(case.domain, case.task_id)
        base = S.replay(sb, case.domain, task, traj)
        oracle = S.replay(sb, case.domain, task, traj, amendment=a, base=base)
        rigid = S.replay(sb, case.domain, task, traj, amendment=a)
        assert S.score_run(sb, case, oracle.diff) == (False, True)
        assert S.score_run(sb, case, rigid.diff) == (True, False)
        assert S.score_run(sb, case, base.diff)[1] is False      # the old final state is not the new one


@needs_sb
def test_state_bench_scripted_agents(suite):
    S, sb, manifest = suite
    oracle = all_model_metrics(S.scripted_records(sb, manifest, "oracle"), criterion=set())[0]
    rigid = all_model_metrics(S.scripted_records(sb, manifest, "rigid"), criterion=set())[0]
    assert (oracle.acc, oracle.cuf, oracle.hold, oracle.ctrl, oracle.buf) == (1.0, 1.0, 1.0, 1.0, 1.0)
    assert (rigid.acc, rigid.cuf, rigid.rigid, rigid.hold, rigid.ctrl) == (1.0, 0.0, 1.0, 1.0, 1.0)
    assert rigid.buf == pytest.approx(0.5)


@needs_sb
def test_state_bench_plan_keeps_criterion_tasks_out_of_amendments(suite):
    S, sb, manifest = suite
    plan = S._plan(manifest, 12, 0)
    roles = {r for _, _, r in plan}
    assert roles == {"changed", "invariant", "control"}
    for _, c, role in plan:
        if role == "invariant":
            assert not is_criterion_key((S.BENCHMARK, c["domain"], c["task_id"]))
    changed_tasks = {c["task_id"] for _, c, r in plan if r == "changed"}
    assert {c["task_id"] for _, c, r in plan if r == "control"} == changed_tasks


@needs_sb
def test_shipped_manifest_matches_a_fresh_derivation(suite):
    S, sb, manifest = suite
    shipped_path = os.path.join(os.path.dirname(__file__), "..", "counterfactual",
                                "state_bench_manifest_v0.1.json")
    shipped = json.load(open(shipped_path))
    if shipped["state_bench_commit"] != manifest["state_bench_commit"]:
        pytest.skip("checkout is at a different STATE-Bench commit than the shipped manifest")
    assert json.loads(json.dumps(manifest)) == shipped
