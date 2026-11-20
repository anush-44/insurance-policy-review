import pytest

from app.core.schemas import ChunkType, PolicyChunk, PolicyDocument
from app.indexing.chunker import ClauseAwareChunker
from app.indexing.embeddings import DeterministicMockEmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore, get_qdrant_client


def test_deterministic_mock_embedding() -> None:
    embedder = DeterministicMockEmbeddingModel(dimension=32)
    assert embedder.dimension == 32

    vec1 = embedder.embed_query("What is the waiting period for pre-existing disease?")
    vec2 = embedder.embed_query("What is the waiting period for pre-existing disease?")
    vec3 = embedder.embed_query("Different query entirely about room rent cap")

    assert len(vec1) == 32
    # Determinism check
    assert vec1 == vec2
    assert vec1 != vec3

    # Unit norm check
    norm_sq = sum(x * x for x in vec1)
    assert pytest.approx(norm_sq, rel=1e-3) == 1.0


def test_qdrant_memory_indexing_and_search(dummy_policy_document: PolicyDocument) -> None:
    # 1. Chunk document
    chunker = ClauseAwareChunker(min_chunk_chars=50, max_chunk_chars=1000)
    chunks = chunker.chunk(dummy_policy_document)
    assert len(chunks) >= 4

    # 2. Embed chunks
    embedder = DeterministicMockEmbeddingModel(dimension=32)
    texts = [c.text for c in chunks]
    vectors = embedder.embed_documents(texts)

    # 3. Store in ephemeral in-memory Qdrant
    client = get_qdrant_client(mode="memory")
    store = QdrantVectorStore(client=client, collection_name="test_policy_chunks")
    store.ensure_collection(vector_dim=32, recreate=True)
    store.upsert_chunks(chunks=chunks, vectors=vectors)

    # 4. Search without filters
    q_vec = embedder.embed_query("room rent limit")
    results = store.search(query_vector=q_vec, limit=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert isinstance(top_chunk, PolicyChunk)
    assert isinstance(score, float)

    # 5. Metadata filtered search: policy_id
    filtered_results = store.search(
        query_vector=q_vec,
        limit=5,
        policy_id=dummy_policy_document.policy_id,
    )
    assert len(filtered_results) > 0
    for c, _ in filtered_results:
        assert c.policy_id == dummy_policy_document.policy_id

    # 6. Metadata filtered search: non-matching policy_id
    empty_results = store.search(
        query_vector=q_vec,
        limit=5,
        policy_id="non-existent-policy",
    )
    assert len(empty_results) == 0

    # 7. Metadata filtered search: chunk_type
    waiting_period_results = store.search(
        query_vector=q_vec,
        limit=5,
        chunk_types=[ChunkType.WAITING_PERIOD],
    )
    assert len(waiting_period_results) > 0
    for c, _ in waiting_period_results:
        assert c.chunk_type == ChunkType.WAITING_PERIOD

    removed = store.prune_policy_chunks(dummy_policy_document.policy_id, {chunks[0].chunk_id})
    assert removed == len(chunks) - 1
    assert store.delete_policy_chunks(dummy_policy_document.policy_id) == 1
    assert store.delete_policy_chunks(dummy_policy_document.policy_id) == 0
