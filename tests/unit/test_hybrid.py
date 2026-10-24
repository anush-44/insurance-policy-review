from app.core.schemas import ChunkType, PolicyChunk
from app.retrieval.hybrid import HybridRetriever


def chunk(chunk_id: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=chunk_id, policy_id="p1", insurer="Insurer", section="Coverage",
        clause_id=chunk_id, clause_title=chunk_id, text=f"Text for {chunk_id}",
        page_start=1, page_end=1, chunk_type=ChunkType.COVERAGE,
    )


class FakeVectorStore:
    def search(self, **kwargs):
        return [(chunk("first"), 0.8), (chunk("second"), 0.7)]


class FakeBM25:
    def search(self, **kwargs):
        return []


class FakeEmbeddings:
    def embed_query(self, query):
        return [0.0]


class FakeCrossEncoder:
    def predict(self, pairs):
        assert len(pairs) == 2
        return [0.1, 0.9]


def test_cross_encoder_reranks_candidate_pool() -> None:
    retriever = HybridRetriever(
        FakeVectorStore(), FakeBM25(), FakeEmbeddings(), reranker_enabled=True,
        reranker=FakeCrossEncoder(), retrieval_mode="dense",
    )
    results = retriever.retrieve("query", limit=2)
    assert [item[0].chunk_id for item in results] == ["second", "first"]
    assert retriever.last_reranker_status == "applied"


def test_reranker_failure_falls_back_observably() -> None:
    class BrokenReranker:
        def predict(self, pairs):
            raise RuntimeError("model offline")

    retriever = HybridRetriever(
        FakeVectorStore(), FakeBM25(), FakeEmbeddings(), reranker_enabled=True,
        reranker=BrokenReranker(), retrieval_mode="dense",
    )
    results = retriever.retrieve("query", limit=2)
    assert [item[0].chunk_id for item in results] == ["first", "second"]
    assert retriever.last_reranker_status.startswith("fallback_rrf:")
