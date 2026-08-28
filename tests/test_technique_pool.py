"""
Tests for contradish.technique_pool and its use in Runner.generate_paraphrases.

Run with: pytest tests/test_technique_pool.py   (no API key, pure stdlib)
"""
import json
import random

from contradish.technique_pool import AdversarialTechnique, DEFAULT_TECHNIQUES, TechniquePool
from contradish.runner import Runner


# ---------------------------------------------------------------------------
# TechniquePool
# ---------------------------------------------------------------------------

def test_default_pool_has_eight_bundled_techniques():
    pool = TechniquePool()
    assert len(pool) == 8
    assert len(DEFAULT_TECHNIQUES) == 8


def test_sample_none_returns_full_pool_shuffled_but_complete():
    pool = TechniquePool()
    sampled = pool.sample(k=None, rng=random.Random(1))
    assert len(sampled) == 8
    assert {t.name for t in sampled} == {t.name for t in DEFAULT_TECHNIQUES}


def test_sample_k_holds_part_of_the_pool_out():
    pool = TechniquePool()
    sampled = pool.sample(k=3, rng=random.Random(1))
    assert len(sampled) == 3
    # a different seed should be free to pick a different subset/order
    sampled_again = pool.sample(k=3, rng=random.Random(2))
    assert len(sampled_again) == 3


def test_sample_k_larger_than_pool_returns_everything():
    pool = TechniquePool()
    sampled = pool.sample(k=999, rng=random.Random(1))
    assert len(sampled) == 8


def test_add_registers_a_technique_not_in_the_bundled_set():
    pool = TechniquePool()
    pool.add(AdversarialTechnique("PRIVATE ONE", "a technique nobody published", "example input"))
    assert len(pool) == 9
    names = {t.name for t in pool.sample(k=None)}
    assert "PRIVATE ONE" in names


def test_extend_from_file_loads_json_techniques(tmp_path):
    private = tmp_path / "private_techniques.json"
    private.write_text(json.dumps([
        {"name": "HELD OUT A", "description": "d1", "example": "e1"},
        {"name": "HELD OUT B", "description": "d2", "example": "e2"},
    ]))
    pool = TechniquePool()
    added = pool.extend_from_file(private)
    assert added == 2
    assert len(pool) == 10
    names = {t.name for t in pool.sample(k=None)}
    assert {"HELD OUT A", "HELD OUT B"} <= names


def test_from_env_merges_private_file_when_set(tmp_path, monkeypatch):
    private = tmp_path / "private_techniques.json"
    private.write_text(json.dumps([
        {"name": "ENV TECHNIQUE", "description": "d", "example": "e"},
    ]))
    monkeypatch.setenv("CONTRADISH_PRIVATE_TECHNIQUES", str(private))
    pool = TechniquePool.from_env()
    assert len(pool) == 9
    assert any(t.name == "ENV TECHNIQUE" for t in pool.sample(k=None))


def test_from_env_is_default_pool_when_var_unset(monkeypatch):
    monkeypatch.delenv("CONTRADISH_PRIVATE_TECHNIQUES", raising=False)
    pool = TechniquePool.from_env()
    assert len(pool) == 8


def test_as_prompt_block_includes_name_description_and_example():
    t = AdversarialTechnique("NAME", "desc", "example text")
    block = t.as_prompt_block(3)
    assert block.startswith("3. NAME: desc")
    assert "example text" in block


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Minimal stand-in for LLMClient: records the prompt, returns canned variants."""

    fast_model = "fake-fast-model"

    def __init__(self, n_to_return=3):
        self.n_to_return = n_to_return
        self.last_prompt = None

    def complete_json(self, prompt, model=None):
        self.last_prompt = prompt
        return [f"variant {i}" for i in range(self.n_to_return)]


def test_runner_defaults_to_full_bundled_pool():
    llm = _FakeLLM(n_to_return=3)
    runner = Runner(llm, technique_pool=TechniquePool())
    variants = runner.generate_paraphrases("Can I get a refund?", n=3, rule="refund window")
    assert variants == ["variant 0", "variant 1", "variant 2"]
    assert len(runner.last_techniques_used) == 8
    for t in runner.last_techniques_used:
        assert t.name in llm.last_prompt


def test_runner_held_out_k_uses_a_subset_of_techniques():
    llm = _FakeLLM(n_to_return=2)
    runner = Runner(llm, technique_pool=TechniquePool())
    runner.generate_paraphrases(
        "Can I get a refund?", n=2, rule="refund window",
        held_out_k=3, rng=random.Random(7),
    )
    assert len(runner.last_techniques_used) == 3
    used_names = {t.name for t in runner.last_techniques_used}
    for t in DEFAULT_TECHNIQUES:
        if t.name not in used_names:
            assert t.name not in llm.last_prompt


def test_runner_semantic_mode_records_no_techniques():
    llm = _FakeLLM(n_to_return=2)
    runner = Runner(llm, technique_pool=TechniquePool())
    runner.generate_paraphrases("Can I get a refund?", n=2, adversarial=False)
    assert runner.last_techniques_used == []


def test_runner_private_technique_can_appear_in_the_prompt():
    llm = _FakeLLM(n_to_return=1)
    pool = TechniquePool()
    pool.add(AdversarialTechnique("SECRET FRAMING", "not in the public repo", "example"))
    runner = Runner(llm, technique_pool=pool)
    # k = pool size forces every technique, including the private one, into play
    runner.generate_paraphrases(
        "Can I get a refund?", n=1, held_out_k=len(pool), rng=random.Random(3)
    )
    assert "SECRET FRAMING" in llm.last_prompt
