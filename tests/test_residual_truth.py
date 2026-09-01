"""
Tests for contradish.residual_truth — the Residual Truth Engine
(contradiction-forced claim extraction, incompatibility detection, and
adaptive repair-based stability scoring).
Run with: pytest tests/test_residual_truth.py
No API key required: the engine tests use deterministic hand-written mock
extractors/detectors/model functions, and the extractor/detector unit tests
are pure Python over plain strings.
"""
import pytest

from contradish.residual_truth import (
    Claim, IncompatibilityEdge, RepairStep, RepairTrace, ResidualTruthResult,
    KeywordClaimExtractor, PatternIncompatibilityDetector, ResidualTruthEngine,
)


# ── Claim dataclass ──────────────────────────────────────────────────────────

def _claim(stability, claim_id="c0", text="x"):
    return Claim(claim_id=claim_id, text=text, raw_text=text, framing="f",
                 source_answer="a", stability=stability)


def test_claim_stable_boundary():
    assert _claim(0.75).stable is True
    assert _claim(0.74).stable is False


def test_claim_fragile_boundaries():
    assert _claim(0.20).fragile is False   # exclusive lower bound
    assert _claim(0.21).fragile is True
    assert _claim(0.74).fragile is True
    assert _claim(0.75).fragile is False   # promoted to stable


def test_claim_collapsed_boundary():
    assert _claim(0.20).collapsed is True
    assert _claim(0.21).collapsed is False


def test_claim_equality_and_hash_by_id_only():
    a = _claim(0.9, claim_id="c1", text="one thing")
    b = _claim(0.1, claim_id="c1", text="a different thing")
    c = _claim(0.9, claim_id="c2", text="one thing")
    assert a == b                      # same id -> equal, regardless of content
    assert hash(a) == hash(b)
    assert a != c                      # different id -> not equal
    assert a != "not a claim"


def test_claim_repr_contains_stability_and_text():
    c = _claim(0.5, text="the recommended dose is safe for most adults")
    assert "0.50" in repr(c)
    assert "recommended dose" in repr(c)


# ── KeywordClaimExtractor ───────────────────────────────────────────────────

@pytest.fixture
def extractor():
    return KeywordClaimExtractor()


def test_extract_keeps_declarative_sentences(extractor):
    answer = "The maximum daily dose is 1200 milligrams for most adults."
    claims = extractor.extract("q", answer, "neutral")
    assert claims == ["The maximum daily dose is 1200 milligrams for most adults"]


def test_extract_drops_short_sentences(extractor):
    answer = "Yes. The maximum daily dose is 1200 milligrams for most adults."
    claims = extractor.extract("q", answer, "neutral")
    assert claims == ["The maximum daily dose is 1200 milligrams for most adults"]


def test_extract_drops_questions(extractor):
    answer = "Have you asked your doctor about the correct dosage amount?"
    assert extractor.extract("q", answer, "neutral") == []


def test_extract_drops_imperatives(extractor):
    answer = "Take ibuprofen with food to avoid stomach irritation issues."
    assert extractor.extract("q", answer, "neutral") == []


def test_extract_drops_empathetic_openers(extractor):
    answer = "I understand this can be a stressful and confusing situation."
    assert extractor.extract("q", answer, "neutral") == []


def test_extract_splits_on_semicolons(extractor):
    answer = "Ibuprofen reduces inflammation quickly; it also relieves pain effectively."
    claims = extractor.extract("q", answer, "neutral")
    assert "Ibuprofen reduces inflammation quickly" in claims
    assert "it also relieves pain effectively" in claims


def test_extract_dedupes_within_one_answer(extractor):
    answer = ("The maximum daily dose is 1200 milligrams for adults. "
              "The maximum daily dose is 1200 milligrams for adults.")
    claims = extractor.extract("q", answer, "neutral")
    assert claims == ["The maximum daily dose is 1200 milligrams for adults"]


def test_extract_normalizes_whitespace_and_trailing_punctuation(extractor):
    answer = "The   maximum   dose  is 1200 milligrams for most adults!!"
    claims = extractor.extract("q", answer, "neutral")
    assert claims == ["The maximum dose is 1200 milligrams for most adults"]


def test_extract_empty_answer_returns_empty_list(extractor):
    assert extractor.extract("q", "", "neutral") == []


# ── PatternIncompatibilityDetector ─────────────────────────────────────────────

@pytest.fixture
def detector():
    return PatternIncompatibilityDetector()


def test_numerical_conflict_detected(detector):
    a = "the maximum daily dose for adults is 1200 milligrams"
    b = "the maximum daily dose for adults is 3200 milligrams"
    ok, reason = detector.are_incompatible(a, b)
    assert ok and "Numerical conflict" in reason


def test_numerical_conflict_needs_shared_context(detector):
    a = "the maximum daily dose for adults is 1200 milligrams"
    b = "the boiling point of water is 3200 feet above sea level"
    ok, _ = detector.are_incompatible(a, b)
    assert not ok


def test_numerical_conflict_ratio_too_small_is_not_a_conflict(detector):
    a = "the maximum daily dose for adults is 1200 milligrams"
    b = "the maximum daily dose for adults is 1300 milligrams"
    ok, _ = detector.are_incompatible(a, b)
    assert not ok   # ratio ~1.08, below the 1.5x threshold


def test_numerical_conflict_ratio_too_large_is_not_a_conflict(detector):
    a = "the maximum daily dose for adults is 10 milligrams"
    b = "the maximum daily dose for adults is 500 milligrams"
    ok, _ = detector.are_incompatible(a, b)
    assert not ok   # ratio 50x, above the 30x ceiling (different scales)


def test_negation_conflict_detected(detector):
    a = "ibuprofen is safe to take with food and water"
    b = "ibuprofen is not safe to take with food and water"
    ok, reason = detector.are_incompatible(a, b)
    assert ok and "Polarity conflict" in reason


def test_negation_same_polarity_is_not_a_conflict(detector):
    a = "ibuprofen is safe to take with food and water"
    b = "ibuprofen is safe to take with food and juice"
    ok, _ = detector.are_incompatible(a, b)
    assert not ok


def test_negation_needs_enough_shared_keywords(detector):
    a = "ibuprofen is safe"
    b = "the sky is not blue today"
    ok, _ = detector.are_incompatible(a, b)
    assert not ok


def test_categorical_pair_conflict_detected(detector):
    a = "this medication is available over the counter otc for adults"
    b = "this medication is available only by prescription for adults"
    ok, reason = detector.are_incompatible(a, b)
    assert ok and "Categorical conflict" in reason


def test_comparative_safety_conflict_detected(detector):
    # Note: "depends"/"context" (not negation words) so this exercises the
    # comparative-safety check specifically, ahead of the negation check
    # which would otherwise match first (e.g. on "neither").
    a = "ibuprofen is safer than acetaminophen for most adult patients"
    b = "it depends on the context for most adult patients"
    ok, reason = detector.are_incompatible(a, b)
    assert ok and "Comparative safety conflict" in reason


def test_are_incompatible_no_conflict_returns_false_and_empty_reason(detector):
    ok, reason = detector.are_incompatible(
        "the weather today is sunny and warm",
        "the coffee shop opens at nine in the morning",
    )
    assert ok is False and reason == ""


def test_keywords_stems_plurals_and_adverbs(detector):
    kw = detector._keywords("adults typically respond pharmacologically well")
    assert "adult" in kw and "adults" in kw
    assert "pharmacological" in kw


def test_numbers_strips_thousands_commas(detector):
    assert detector._numbers("the dose is 1,200 milligrams") == [1200.0]


def test_numerical_skips_equal_or_zero_pairs_before_finding_real_conflict(detector):
    # (100, 100) is equal -> skipped; (100, 900) is a real 9x conflict.
    a = "the maximum dose for adults is 100 and 300 milligrams"
    b = "the maximum dose for adults is 100 and 900 milligrams"
    ok, reason = detector.are_incompatible(a, b)
    assert ok and "9" in reason


def test_categorical_pair_conflict_detected_mirrored_order(detector):
    # pos pattern in b, neg pattern in a -- the mirrored branch of the check.
    a = "this medication is available only by prescription for adults"
    b = "this medication is available over the counter otc for adults"
    ok, reason = detector.are_incompatible(a, b)
    assert ok and "Categorical conflict" in reason


# ── ResidualTruthResult: report() / to_html() ──────────────────────────────────

def _sample_result():
    stable = Claim(claim_id="c0", text="stable claim text", raw_text="x",
                   framing="f1", source_answer="a", stability=1.0, frequency=5)
    fragile = Claim(claim_id="c1", text="fragile claim text", raw_text="x",
                    framing="f1", source_answer="a", stability=0.5, frequency=1)
    collapsed = Claim(claim_id="c2", text="collapsed claim text", raw_text="x",
                      framing="f6", source_answer="a", stability=0.0, frequency=1)
    edge = IncompatibilityEdge(claim_a_id="c0", claim_b_id="c2", reason="test conflict")
    step = RepairStep(dropped_id="c2", kept_id="c0", dropped_text="collapsed claim text",
                      kept_text="stable claim text", reason="lower stability signal",
                      edge_reason="test conflict")
    return ResidualTruthResult(
        question="Is ibuprofen safe?", framings_used=["f1", "f6"],
        all_claims=[stable, fragile, collapsed], incompatibilities=[edge],
        traces=[], n_repairs=10,
        stable_residue=[stable], fragile_claims=[fragile],
        collapsed_assumptions=[collapsed], canonical_repair_path=[step],
    )


def test_report_contains_all_sections_and_claim_text():
    text = _sample_result().report()
    assert "RESIDUAL TRUTH ANALYSIS" in text
    assert "STABLE RESIDUE" in text and "stable claim text" in text
    assert "FRAGILE CLAIMS" in text and "fragile claim text" in text
    assert "COLLAPSED ASSUMPTIONS" in text and "collapsed claim text" in text
    assert "REPAIR PATH" in text
    assert "lower stability signal" in text


def test_report_empty_sections_show_none_placeholders():
    empty = ResidualTruthResult(question="q", framings_used=[], all_claims=[],
                                incompatibilities=[], traces=[], n_repairs=0)
    text = empty.report()
    assert "all claims are framing-dependent" in text
    assert text.count("(none)") == 2   # collapsed + fragile
    assert "REPAIR PATH" not in text   # no canonical path -> section omitted


def test_to_html_contains_stats_and_escapes_question():
    result = _sample_result()
    result.question = "Is <script>alert(1)</script> safe?"
    html = result.to_html()
    assert "<script>alert(1)</script>" not in html   # escaped, not injected raw
    assert "&lt;script&gt;" in html
    assert "stable claim text" in html
    assert "fragile claim text" in html
    assert "collapsed claim text" in html
    assert 'class="stat-val green">1' in html   # 1 stable residue claim


def test_to_html_no_incompatibilities_shows_empty_message():
    empty = ResidualTruthResult(question="q", framings_used=[], all_claims=[],
                                incompatibilities=[], traces=[], n_repairs=0)
    html = empty.to_html()
    assert "no incompatibilities found" in html
    assert "none detected" in html


def test_to_html_caps_incompatibility_graph_display_at_twenty():
    claims = [Claim(claim_id=f"c{i}", text=f"claim number {i}", raw_text="x",
                    framing="f", source_answer="a") for i in range(25)]
    edges = [IncompatibilityEdge(claim_a_id=f"c{i}", claim_b_id=f"c{i+1}", reason="r")
             for i in range(24)]
    result = ResidualTruthResult(question="q", framings_used=["f"], all_claims=claims,
                                 incompatibilities=edges, traces=[], n_repairs=0)
    html = result.to_html()
    assert html.count('class="edge-row"') == 20


# ── ResidualTruthEngine: end-to-end with deterministic mocks ───────────────────

STABLE_TEXT = "the recommended maximum dose is one thousand milligrams daily"
LOSER_TEXT  = "conflicting statement about a totally different subject entirely"
SOLO_TEXT   = "a claim that only ever appears under a single framing here"


class _StubExtractor:
    """Returns a canned claim list per framing, ignoring the actual answer text."""
    def __init__(self, by_framing):
        self.by_framing = by_framing

    def extract(self, question, answer, framing):
        return list(self.by_framing.get(framing, []))


class _StubDetector:
    """Only the STABLE_TEXT/LOSER_TEXT pair is incompatible."""
    def are_incompatible(self, a, b):
        if {a, b} == {STABLE_TEXT, LOSER_TEXT}:
            return True, "test conflict"
        return False, ""


def _make_engine(n_repairs=20, adaptive=True, seed=7):
    by_framing = {
        "f1": [STABLE_TEXT, SOLO_TEXT],
        "f2": [STABLE_TEXT],
        "f3": [STABLE_TEXT],
        "f4": [STABLE_TEXT],
        "f5": [STABLE_TEXT],
        "f6": [LOSER_TEXT],
    }
    return ResidualTruthEngine(
        claim_extractor=_StubExtractor(by_framing),
        incompatibility_detector=_StubDetector(),
        n_repairs=n_repairs, adaptive_scoring=adaptive, seed=seed,
    )


def _framings():
    return {f"f{i}": "" for i in range(1, 7)}


def test_analyze_classifies_stable_collapsed_and_demoted_fragile_claim():
    engine = _make_engine()
    result = engine.analyze("q", lambda sp, q: "irrelevant", framings=_framings())

    assert {c.text for c in result.stable_residue} == {STABLE_TEXT}
    assert {c.text for c in result.collapsed_assumptions} == {LOSER_TEXT}
    # SOLO_TEXT survives every repair (never conflicts) but only appears in
    # one framing, so it's demoted from "stable" to "fragile" by frequency.
    assert {c.text for c in result.fragile_claims} == {SOLO_TEXT}

    stable_claim = result.stable_residue[0]
    assert stable_claim.stability == 1.0 and stable_claim.frequency == 5
    collapsed_claim = result.collapsed_assumptions[0]
    assert collapsed_claim.stability == 0.0


def test_analyze_builds_expected_incompatibility_and_repair_path():
    engine = _make_engine()
    result = engine.analyze("q", lambda sp, q: "irrelevant", framings=_framings())

    assert len(result.incompatibilities) == 1
    assert result.incompatibilities[0].reason == "test conflict"

    assert len(result.canonical_repair_path) == 1
    step = result.canonical_repair_path[0]
    assert step.dropped_text == LOSER_TEXT
    assert step.kept_text == STABLE_TEXT


def test_analyze_n_repairs_controls_trace_count():
    engine = _make_engine(n_repairs=5)
    result = engine.analyze("q", lambda sp, q: "irrelevant", framings=_framings())
    assert result.n_repairs == 5
    assert len(result.traces) == 5


def test_analyze_non_adaptive_mode_still_converges_correctly():
    engine = _make_engine(adaptive=False, n_repairs=10)
    result = engine.analyze("q", lambda sp, q: "irrelevant", framings=_framings())
    assert {c.text for c in result.stable_residue} == {STABLE_TEXT}
    assert {c.text for c in result.collapsed_assumptions} == {LOSER_TEXT}


def test_analyze_same_seed_is_reproducible():
    r1 = _make_engine(seed=42).analyze("q", lambda sp, q: "x", framings=_framings())
    r2 = _make_engine(seed=42).analyze("q", lambda sp, q: "x", framings=_framings())
    s1 = {c.claim_id: c.stability for c in r1.all_claims}
    s2 = {c.claim_id: c.stability for c in r2.all_claims}
    assert s1 == s2


def test_analyze_skips_framings_where_model_fn_raises():
    def flaky_model(system_prompt, question):
        if "f6" in question:
            raise RuntimeError("simulated model failure")
        return "ok"

    by_framing = {"f1": [STABLE_TEXT], "f6": [LOSER_TEXT]}
    engine = ResidualTruthEngine(
        claim_extractor=_StubExtractor(by_framing),
        incompatibility_detector=_StubDetector(),
        n_repairs=3, seed=1,
    )
    framings = {"f1": "", "f6": "f6 marker "}
    result = engine.analyze("q", flaky_model, framings=framings)
    # f6 raised, so only f1's answer was generated -> only STABLE_TEXT extracted
    assert result.framings_used == ["f1"]
    assert {c.text for c in result.all_claims} == {STABLE_TEXT}


def test_analyze_returns_empty_result_when_no_claims_extracted():
    engine = ResidualTruthEngine(
        claim_extractor=_StubExtractor({}),  # extracts nothing for any framing
        n_repairs=5, seed=1,
    )
    result = engine.analyze("q", lambda sp, q: "some answer", framings=_framings())
    assert result.all_claims == []
    assert result.stable_residue == [] and result.n_repairs == 0
    assert result.report()  # smoke: no-crash on the empty path too


def test_analyze_empty_when_model_fn_returns_blank_for_every_framing():
    engine = _make_engine()
    result = engine.analyze("q", lambda sp, q: "   ", framings=_framings())
    assert result.all_claims == []


def test_analyze_drops_lower_scored_first_indexed_claim_in_an_edge():
    # Same STABLE/LOSER conflict as the main scenario, but with the framing
    # order reversed so the lower-scored claim is inserted (and thus
    # edge-indexed) first -- exercises the "ca has the lower score" branch
    # of the repair decision, the mirror of the main scenario above.
    by_framing = {
        "f6": [LOSER_TEXT],
        "f1": [STABLE_TEXT, SOLO_TEXT], "f2": [STABLE_TEXT], "f3": [STABLE_TEXT],
        "f4": [STABLE_TEXT], "f5": [STABLE_TEXT],
    }
    engine = ResidualTruthEngine(
        claim_extractor=_StubExtractor(by_framing),
        incompatibility_detector=_StubDetector(),
        n_repairs=10, seed=7,
    )
    reversed_framings = {"f6": "", "f1": "", "f2": "", "f3": "", "f4": "", "f5": ""}
    result = engine.analyze("q", lambda sp, q: "x", framings=reversed_framings)
    assert {c.text for c in result.stable_residue} == {STABLE_TEXT}
    assert {c.text for c in result.collapsed_assumptions} == {LOSER_TEXT}


def test_canonical_trace_returns_none_for_no_traces():
    engine = _make_engine()
    assert engine._canonical_trace([]) is None


def test_similar_returns_false_when_either_side_has_no_words():
    engine = _make_engine()
    assert engine._similar("1234 5678", "some real words here") is False
    assert engine._similar("", "") is False


def test_analyze_uses_default_framings_and_real_extractor_detector():
    # Integration smoke test: no framings passed -> falls back to
    # phi_star.FRAMING_PREFIXES, using the real (non-stub) extractor/detector.
    engine = ResidualTruthEngine(n_repairs=2, seed=3)

    def model_fn(system_prompt, question):
        return "The maximum recommended dose is 1200 milligrams for most adults."

    result = engine.analyze("Is ibuprofen safe?", model_fn)
    from contradish.phi_star import FRAMING_PREFIXES
    assert len(result.framings_used) == len(FRAMING_PREFIXES)
    assert isinstance(result.report(), str)
    assert isinstance(result.to_html(), str)
