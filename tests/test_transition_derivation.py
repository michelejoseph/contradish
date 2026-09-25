"""
Tests for contradish.transition_derivation.

No API key required: TransitionDerivationJudge is exercised against a fake
`.complete_json`-returning object, and ManualTransitionJudge against plain
dicts -- no model calls anywhere in this module or its tests, same
discipline test_justification_faithfulness.py already uses.
"""
from contradish.transition_derivation import (
    ScenarioPair,
    scenario_pair_from_distinction_pair,
    DerivedCommitment,
    TransitionContract,
    TransitionDerivationJudge,
    ManualTransitionJudge,
    derive_transition_contract,
    derive_transition_contracts,
    warrant_agreement,
    kappa_label,
    _cohens_kappa,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeDistinctionPair:
    def __init__(self, pair_id, domain, description, label_a, label_b,
                 question_a, question_b, commit_a, commit_b):
        self.pair_id = pair_id
        self.domain = domain
        self.description = description
        self.label_a = label_a
        self.label_b = label_b
        self.question_a = question_a
        self.question_b = question_b
        self.commit_a = commit_a
        self.commit_b = commit_b


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def complete_json(self, prompt):
        return self.response


class RaisingLLM:
    def complete_json(self, prompt):
        raise RuntimeError("simulated API failure")


def _make_scenario(pair_id="p1"):
    return ScenarioPair(
        pair_id=pair_id, domain="test", baseline_label="A", baseline_description="qa",
        changed_label="B", changed_description="qb", governing_change="X changed",
    )


# ── ScenarioPair / scenario_pair_from_distinction_pair ───────────────────────

def test_scenario_pair_to_dict_round_trips():
    sp = _make_scenario()
    d = sp.to_dict()
    assert d["pair_id"] == "p1"
    assert d["baseline_description"] == "qa"


def test_scenario_pair_from_distinction_pair_discards_commits():
    pair = FakeDistinctionPair(
        pair_id="dp1", domain="medication", description="healthy vs renal",
        label_a="healthy adult", label_b="renal patient",
        question_a="what dose for a healthy adult?", question_b="what dose for a renal patient?",
        commit_a="1200mg otc limit", commit_b="avoid or use only under supervision",
    )
    sp = scenario_pair_from_distinction_pair(pair)
    assert sp.pair_id == "dp1"
    assert sp.baseline_description == pair.question_a
    assert sp.changed_description == pair.question_b
    assert not hasattr(sp, "commit_a") and not hasattr(sp, "commit_b")
    # nothing in the produced object leaks the withheld ground truth
    assert "1200mg" not in str(sp.to_dict())
    assert "supervision" not in str(sp.to_dict())


# ── TransitionContract / bridge methods ───────────────────────────────────────

def test_transition_contract_to_json_round_trips():
    import json
    c = TransitionContract(
        pair_id="p1", domain="test",
        commitments=[DerivedCommitment("p1", warranted=True, expected_effect="x", confidence=0.8, rationale="y")],
        derived_by="manual:reviewer",
    )
    data = json.loads(c.to_json())
    assert data["schema_version"] == "1.0"
    assert data["commitments"][0]["warranted"] is True


def test_to_decision_relevance_spec_maps_warranted_to_relevance():
    c = TransitionContract(pair_id="p1", domain="d", commitments=[
        DerivedCommitment("keep", warranted=False, expected_effect="", confidence=0.9, rationale=""),
        DerivedCommitment("change", warranted=True, expected_effect="new content", confidence=0.9, rationale=""),
        DerivedCommitment("unsure", warranted=None, expected_effect="", confidence=0.0, rationale=""),
    ])
    spec = c.to_decision_relevance_spec()
    assert spec.factors["keep"].relevance == "irrelevant"
    assert spec.factors["change"].relevance == "relevant"
    assert spec.factors["change"].expected_effect == "new content"
    # undetermined is conservatively treated as irrelevant, never guessed relevant
    assert spec.factors["unsure"].relevance == "irrelevant"


def test_justified_and_invariant_commitments_excludes_undetermined():
    c = TransitionContract(pair_id="p1", domain="d", commitments=[
        DerivedCommitment("keep", warranted=False),
        DerivedCommitment("change", warranted=True, expected_effect="new content"),
        DerivedCommitment("unsure", warranted=None),
    ])
    justified, invariant = c.justified_and_invariant_commitments()
    assert justified == {"change": "new content"}
    assert invariant == ["keep"]
    assert "unsure" not in justified and "unsure" not in invariant


# ── ManualTransitionJudge ─────────────────────────────────────────────────────

def test_manual_judge_uses_explicit_entry():
    judge = ManualTransitionJudge({"p1": {"warranted": True, "expected_effect": "e", "confidence": 0.7, "rationale": "r"}})
    contract = judge.derive(_make_scenario("p1"))
    assert contract.commitments[0].warranted is True
    assert contract.commitments[0].expected_effect == "e"
    assert contract.derived_by == "manual:reviewer"


def test_manual_judge_uses_default_when_missing():
    judge = ManualTransitionJudge({})
    contract = judge.derive(_make_scenario("missing"))
    assert contract.commitments[0].warranted is None
    assert "defaulted" in contract.commitments[0].rationale.lower()


# ── TransitionDerivationJudge (live-judge-shaped, fake LLM) ──────────────────

def test_derivation_judge_parses_valid_response():
    llm = FakeLLM({"warranted": True, "expected_effect": "e", "confidence": 0.6, "rationale": "r"})
    judge = TransitionDerivationJudge(llm)
    contract = judge.derive(_make_scenario())
    assert contract.commitments[0].warranted is True
    assert contract.commitments[0].confidence == 0.6
    assert contract.derived_by == "llm:FakeLLM"


def test_derivation_judge_falls_back_to_none_on_invalid_warranted_value():
    llm = FakeLLM({"warranted": "yes probably", "confidence": 0.5, "rationale": ""})
    judge = TransitionDerivationJudge(llm)
    contract = judge.derive(_make_scenario())
    assert contract.commitments[0].warranted is None


def test_derivation_judge_handles_llm_exception_gracefully():
    judge = TransitionDerivationJudge(RaisingLLM())
    contract = judge.derive(_make_scenario())
    assert contract.commitments[0].warranted is None
    assert contract.commitments[0].confidence == 0.0
    assert "simulated API failure" in contract.commitments[0].rationale


def test_derivation_judge_clamps_confidence():
    llm = FakeLLM({"warranted": True, "confidence": 5.0, "rationale": ""})
    judge = TransitionDerivationJudge(llm)
    contract = judge.derive(_make_scenario())
    assert contract.commitments[0].confidence == 1.0


# ── batch driver ──────────────────────────────────────────────────────────────

def test_derive_transition_contracts_batch():
    judge = ManualTransitionJudge({
        "p1": {"warranted": True, "expected_effect": "e1", "confidence": 0.8, "rationale": ""},
        "p2": {"warranted": False, "expected_effect": "", "confidence": 0.8, "rationale": ""},
    })
    scenarios = [_make_scenario("p1"), _make_scenario("p2")]
    contracts = derive_transition_contracts(scenarios, judge)
    assert set(contracts) == {"p1", "p2"}
    assert contracts["p1"].commitments[0].warranted is True
    assert contracts["p2"].commitments[0].warranted is False


def test_derive_transition_contract_single():
    judge = ManualTransitionJudge({"p1": {"warranted": True, "confidence": 0.5, "rationale": ""}})
    contract = derive_transition_contract(_make_scenario("p1"), judge)
    assert contract.pair_id == "p1"


# ── kappa / agreement ──────────────────────────────────────────────────────────

def test_cohens_kappa_perfect_agreement():
    pairs = [(True, True), (False, False), (True, True), (False, False)]
    po, pe, kappa = _cohens_kappa(pairs)
    assert po == 1.0
    assert kappa == 1.0


def test_cohens_kappa_empty_returns_nan():
    po, pe, kappa = _cohens_kappa([])
    assert po == 0.0
    assert kappa != kappa  # nan


def test_kappa_label_bands():
    assert kappa_label(-0.1) == "worse than chance"
    assert kappa_label(0.1) == "slight"
    assert kappa_label(0.3) == "fair"
    assert kappa_label(0.5) == "moderate"
    assert kappa_label(0.7) == "substantial"
    assert kappa_label(0.9) == "almost perfect"
    assert kappa_label(float("nan")) == "undefined"


def test_warrant_agreement_basic():
    judge = ManualTransitionJudge({
        "p1": {"warranted": True, "confidence": 0.9, "rationale": "matches"},
        "p2": {"warranted": False, "confidence": 0.9, "rationale": "matches"},
        "p3": {"warranted": True, "confidence": 0.5, "rationale": "wrong call"},
    })
    scenarios = [_make_scenario("p1"), _make_scenario("p2"), _make_scenario("p3")]
    derived = derive_transition_contracts(scenarios, judge)
    ground_truth = {"p1": True, "p2": False, "p3": False}  # p3 disagrees
    result = warrant_agreement(derived, ground_truth)
    assert result["n_common"] == 3
    assert result["n_scored"] == 3
    assert result["percent_agreement"] == round(2 / 3, 3)
    assert len(result["disagreements"]) == 1
    assert result["disagreements"][0]["pair_id"] == "p3"


def test_warrant_agreement_excludes_undetermined_from_kappa():
    judge = ManualTransitionJudge({
        "p1": {"warranted": True, "confidence": 0.9, "rationale": ""},
        "p2": {"warranted": None, "confidence": 0.0, "rationale": "unsure"},
    })
    scenarios = [_make_scenario("p1"), _make_scenario("p2")]
    derived = derive_transition_contracts(scenarios, judge)
    ground_truth = {"p1": True, "p2": True}
    result = warrant_agreement(derived, ground_truth)
    assert result["n_scored"] == 1
    assert result["n_undetermined"] == 1
    assert "p2" in result["undetermined_pair_ids"]
