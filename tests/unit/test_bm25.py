from app.core.schemas import ChunkType, PolicyDocument
from app.indexing.bm25_index import BM25Index, tokenize_insurance_text
from app.indexing.chunker import ClauseAwareChunker


def test_tokenize_insurance_text() -> None:
    text = "Clause 4.2: Waiting period for pre-existing disease (Code-Excl04) Rs. 40,000."
    tokens = tokenize_insurance_text(text)
    assert "4.2" in tokens
    assert "excl04" in tokens
    assert "waiting" in tokens
    assert "disease" in tokens


def test_bm25_search_exact_clause(dummy_policy_document: PolicyDocument) -> None:
    chunker = ClauseAwareChunker(min_chunk_chars=50, max_chunk_chars=1000)
    chunks = chunker.chunk(dummy_policy_document)

    index = BM25Index()
    index.index_chunks(chunks)

    # Search for exact clause identifier "3.2"
    results = index.search(query="Clause 3.2 Pre-Existing Diseases", limit=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "3.2" in top_chunk.clause_id
    assert top_chunk.chunk_type == ChunkType.WAITING_PERIOD
    assert score > 0.0

    # Search for term with specific acronym
    results_sublimit = index.search(query="ICU capping room rent", limit=3)
    assert len(results_sublimit) > 0
    assert any("2.1" in c.clause_id for c, _ in results_sublimit)
