"""Unit tests that run without downloading any models (pure logic only)."""
from eih.retrieval import HybridRetriever, _tokenize


def test_tokenize_splits_identifiers():
    assert _tokenize("auth-svc returns 401!") == ["auth", "svc", "returns", "401"]


def test_rrf_rewards_agreement():
    # a doc ranked highly by BOTH retrievers should win over one ranked high by one
    dense =  ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = HybridRetriever._rrf([dense, sparse], k=60)
    assert fused["a"] > fused["c"]      # 'a' appears in both lists
    assert fused["b"] > fused["d"]
    # 'b' is rank0+rank1, 'a' is rank1+rank0 -> tie; both beat singletons
    assert fused["b"] == fused["a"]


def test_rrf_empty():
    assert HybridRetriever._rrf([[], []], k=60) == {}


class _FakeStore:
    """Minimal stand-in so we can unit-test sparse filtering without Qdrant."""
    def __init__(self, payloads): self._p = payloads
    def all_payloads(self): return self._p


def test_sparse_path_honors_source_type_filter():
    from eih.config import RetrievalConfig
    from eih.schema import SourceType
    from eih.retrieval import HybridRetriever
    payloads = [
        {"chunk_id": "c1", "doc_id": "auth", "source_type": "code",
         "text": "validate jwt token rs256"},
        {"chunk_id": "c2", "doc_id": "inc", "source_type": "incident",
         "text": "auth outage jwt token expired"},
    ]
    r = HybridRetriever(RetrievalConfig(), embedder=None, store=_FakeStore(payloads))
    ids, _ = r._sparse("jwt token", source_types=[SourceType.INCIDENT])
    assert ids == ["c2"]                       # code chunk must be filtered out
    ids_all, _ = r._sparse("jwt token", source_types=None)
    assert set(ids_all) == {"c1", "c2"}
