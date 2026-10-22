from pathlib import Path

from app.core.schemas import ChunkType
from app.indexing.chunker import ClauseAwareChunker
from app.indexing.embeddings import DeterministicMockEmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore, get_qdrant_client
from app.ingestion.pdf_parser import PyMuPDFParser


def test_dense_search_acceptance(sample_pdf_path: Path) -> None:
    """Acceptance Test for Phase 2:

    Dense search returns relevant chunks with correct metadata from sample policy.
    """
    # 1. Parse sample PDF
    parser = PyMuPDFParser()
    doc = parser.parse(
        file_source=sample_pdf_path,
        policy_id="care-adv-sample",
        policy_name="Care Advantage Plan",
        insurer="Care Health Insurance Limited",
        uin="CHIHLIP21183V032021",
    )

    # 2. Chunk using ClauseAwareChunker
    chunker = ClauseAwareChunker(min_chunk_chars=80, max_chunk_chars=1000)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 5

    # 3. Embed chunks
    embedder = DeterministicMockEmbeddingModel(dimension=64)
    texts = [c.text for c in chunks]
    vectors = embedder.embed_documents(texts)

    # 4. Index into Qdrant
    client = get_qdrant_client(mode="memory")
    vector_store = QdrantVectorStore(client=client, collection_name="test_dense_search")
    vector_store.ensure_collection(vector_dim=64, recreate=True)
    vector_store.upsert_chunks(chunks=chunks, vectors=vectors)

    # 5. Execute queries and assert metadata correctness
    # Test Query 1: Pre-existing disease waiting period chunk
    target_ped_chunk = next(c for c in chunks if "3.2" in c.clause_id)
    query_vec = embedder.embed_query(target_ped_chunk.text)
    results = vector_store.search(
        query_vector=query_vec,
        limit=3,
        policy_id="care-adv-sample",
    )
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.policy_id == "care-adv-sample"
    assert top_chunk.insurer == "Care Health Insurance Limited"
    assert top_chunk.page_start == 3
    assert "3.2" in top_chunk.clause_id
    assert top_chunk.chunk_type == ChunkType.WAITING_PERIOD
    assert score >= 0.99  # Exact cosine match

    # Test Query 2: Room rent capping chunk
    target_rent_chunk = next(c for c in chunks if "2.1" in c.clause_id)
    query_vec2 = embedder.embed_query(target_rent_chunk.text)
    results2 = vector_store.search(
        query_vector=query_vec2,
        limit=3,
        policy_id="care-adv-sample",
        chunk_types=[ChunkType.SUB_LIMIT],
    )
    assert len(results2) > 0
    top_chunk2, score2 = results2[0]
    assert top_chunk2.policy_id == "care-adv-sample"
    assert top_chunk2.page_start == 2
    assert "2.1" in top_chunk2.clause_id
    assert top_chunk2.chunk_type == ChunkType.SUB_LIMIT
    assert score2 >= 0.99
