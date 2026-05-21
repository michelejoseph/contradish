"""
Tests for the embedding-based relevance scorer in contradish.memory:
EmbeddingRelevance, openai_embedder, and ConversationMemory.with_embeddings.
Run with: pytest tests/test_memory_embeddings.py
No API key or network required: a deterministic concept embedder stands in.
"""
from contradish.memory import (
    Commitment,
    EmbeddingRelevance,
    ConversationMemory,
    InMemoryCommitmentStore,
    openai_embedder,
    _overlap_score,
    _cosine,
)


# A toy embedder where synonyms map to the same axis, so paraphrases land close
# in vector space and unrelated topics stay orthogonal. This lets us assert that
# embeddings retrieve a paraphrase that lexical overlap provably misses.
_AXES = {
    "refund": 0, "refunds": 0, "return": 0, "returns": 0,
    "window": 1, "timeframe": 1, "period": 1, "deadline": 1, "days": 1,
    "store": 2, "shop": 2,
    "hours": 3, "schedule": 3, "open": 3,
}
_DIM = 4


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * _DIM
        for raw in str(t).lower().replace(".", " ").split():
            w = "".join(ch for ch in raw if ch.isalnum())
            if w in _AXES:
                v[_AXES[w]] += 1.0
        out.append(v)
    return out


_A = Commitment(claim="Refund window is 30 days", topic="refund window")
_PARAPHRASE = Commitment(claim="Returns accepted within the return timeframe", topic="return timeframe")
_UNRELATED = Commitment(claim="Store hours are 9 to 5", topic="store hours")


def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert abs(_cosine([1, 1], [1, 0]) - (1 / (2 ** 0.5))) < 1e-9
    assert _cosine([0, 0], [1, 1]) == 0.0   # zero vector -> 0, no div by zero


def test_embeddings_catch_paraphrase_lexical_misses():
    emb = EmbeddingRelevance(fake_embed)
    # Lexical: no shared tokens -> below the 0.3 lexical threshold.
    assert _overlap_score(_A, _PARAPHRASE) < 0.3
    # Embeddings: clearly relevant.
    assert emb(_A, _PARAPHRASE) >= 0.55
    # Unrelated stays low under embeddings too.
    assert emb(_A, _UNRELATED) < 0.55


def test_embedding_score_in_unit_range():
    emb = EmbeddingRelevance(fake_embed)
    for x in (_A, _PARAPHRASE, _UNRELATED):
        for y in (_A, _PARAPHRASE, _UNRELATED):
            s = emb(x, y)
            assert 0.0 <= s <= 1.0


def test_embedding_cache_embeds_each_text_once():
    calls = {"batches": 0, "texts": 0}

    def counting_embed(texts):
        calls["batches"] += 1
        calls["texts"] += len(texts)
        return fake_embed(texts)

    emb = EmbeddingRelevance(counting_embed, cache=True)
    for _ in range(5):
        emb(_A, _PARAPHRASE)            # same two texts, five times
    # Two distinct texts ever embedded, regardless of how many comparisons.
    assert calls["texts"] == 2


def test_embedding_cache_disabled_reembeds():
    calls = {"texts": 0}

    def counting_embed(texts):
        calls["texts"] += len(texts)
        return fake_embed(texts)

    emb = EmbeddingRelevance(counting_embed, cache=False)
    emb(_A, _PARAPHRASE)
    emb(_A, _PARAPHRASE)
    assert calls["texts"] > 2           # no memoization


def test_embed_many_preserves_order_and_dedups_misses():
    calls = {"texts": 0}

    def counting_embed(texts):
        calls["texts"] += len(texts)
        return fake_embed(texts)

    emb = EmbeddingRelevance(counting_embed)
    texts = ["refund window", "store hours", "refund window"]
    vecs = emb.embed_many(texts)
    assert len(vecs) == 3
    assert vecs[0] == vecs[2]           # duplicate text -> same vector
    assert calls["texts"] == 2          # duplicate embedded once


def test_with_embeddings_retrieves_paraphrase_end_to_end():
    mem = ConversationMemory.with_embeddings(fake_embed, store=InMemoryCommitmentStore())
    mem.store.add(Commitment(claim="Refund window is 30 days", topic="refund window", session="u1"))
    mem.store.add(Commitment(claim="Store hours are 9 to 5", topic="store hours", session="u1"))
    new = [Commitment(claim="Returns accepted within the return timeframe",
                      topic="return timeframe", session="u1")]
    rel = mem.relevant("u1", new)
    assert len(rel) == 1
    assert "Refund window" in rel[0].claim   # the paraphrase was retrieved
    # Threshold was set for the embedding scale, not the lexical default.
    assert mem.relevance_threshold == 0.55


def test_with_embeddings_still_session_scoped():
    mem = ConversationMemory.with_embeddings(fake_embed, store=InMemoryCommitmentStore())
    mem.store.add(Commitment(claim="Refund window is 30 days", topic="refund window", session="u1"))
    new = [Commitment(claim="Returns accepted within the return timeframe",
                      topic="return timeframe", session="u2")]
    assert mem.relevant("u2", new) == []


# ── openai_embedder shape (fake client, no network) ──────────────────────

class _FakeEmbItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbResponse:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddings:
    def __init__(self, recorder):
        self._rec = recorder

    def create(self, model=None, input=None, **kw):
        self._rec["model"] = model
        self._rec["input"] = list(input)
        self._rec["dimensions"] = kw.get("dimensions")
        # Echo a deterministic vector per input, in order.
        return _FakeEmbResponse([_FakeEmbItem(fake_embed([t])[0]) for t in input])


class _FakeOpenAIClient:
    def __init__(self):
        self.rec = {}
        self.embeddings = _FakeEmbeddings(self.rec)


def test_openai_embedder_calls_client_and_preserves_order():
    client = _FakeOpenAIClient()
    embed = openai_embedder(model="text-embedding-3-small", client=client)
    texts = ["refund window", "store hours"]
    vecs = embed(texts)
    assert len(vecs) == 2
    assert vecs[0] == fake_embed(["refund window"])[0]
    assert vecs[1] == fake_embed(["store hours"])[0]
    assert client.rec["model"] == "text-embedding-3-small"
    assert client.rec["input"] == texts


def test_openai_embedder_passes_dimensions():
    client = _FakeOpenAIClient()
    embed = openai_embedder(client=client, dimensions=256)
    embed(["refund window"])
    assert client.rec["dimensions"] == 256


def test_openai_embedder_drives_embedding_relevance():
    client = _FakeOpenAIClient()
    emb = EmbeddingRelevance(openai_embedder(client=client))
    assert emb(_A, _PARAPHRASE) >= 0.55
    assert emb(_A, _UNRELATED) < 0.55


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
