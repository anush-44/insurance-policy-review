from app.core.schemas import ChunkType, Clause, PolicyDocument, Section
from app.indexing.chunker import ClauseAwareChunker, FixedSizeChunker, extract_cross_references


def test_extract_cross_references() -> None:
    text = (
        "As specified in Clause 6.4 and subject to Section 2.1, see also Annexure I and Clause 6.4."
    )
    refs = extract_cross_references(text)
    assert refs == ["6.4", "2.1", "I"]


def test_clause_aware_chunker_basic(dummy_policy_document: PolicyDocument) -> None:
    chunker = ClauseAwareChunker(min_chunk_chars=50, max_chunk_chars=1000)
    chunks = chunker.chunk(dummy_policy_document)

    assert len(chunks) >= 4
    for chunk in chunks:
        assert chunk.policy_id == dummy_policy_document.policy_id
        assert chunk.insurer == dummy_policy_document.insurer
        assert chunk.section != ""
        assert chunk.clause_id != ""
        assert chunk.text != ""
        assert 1 <= chunk.page_start <= chunk.page_end <= dummy_policy_document.total_pages


def test_clause_aware_chunker_small_clause_merging() -> None:
    # Construct a document with two consecutive very small clauses
    doc = PolicyDocument(
        policy_id="test-merge",
        policy_name="Test Policy",
        insurer="Test Insurer",
        total_pages=1,
        sections=[
            Section(
                section_id="SEC-1",
                title="Definitions",
                page_start=1,
                page_end=1,
                clauses=[
                    Clause(
                        clause_id="1.1",
                        title="Short Term",
                        text="Term A.",  # very short (< 50 chars)
                        page_start=1,
                        page_end=1,
                        chunk_type=ChunkType.DEFINITION,
                    ),
                    Clause(
                        clause_id="1.2",
                        title="Another Short Term",
                        text="Term B is also tiny.",  # also tiny
                        page_start=1,
                        page_end=1,
                        chunk_type=ChunkType.DEFINITION,
                    ),
                    Clause(
                        clause_id="1.3",
                        title="Sufficiently Long Term Definition",
                        text="This is a full definition of hospital that exceeds the minimum character threshold easily.",
                        page_start=1,
                        page_end=1,
                        chunk_type=ChunkType.DEFINITION,
                    ),
                ],
            )
        ],
    )

    chunker = ClauseAwareChunker(min_chunk_chars=60, max_chunk_chars=1000)
    chunks = chunker.chunk(doc)

    # 1.1 and 1.2 should be merged or buffered together
    assert len(chunks) < 3
    combined_texts = " ".join(c.text for c in chunks)
    assert "Term A." in combined_texts
    assert "Term B is also tiny." in combined_texts


def test_clause_aware_chunker_large_clause_splitting() -> None:
    # Long text over 500 chars
    long_clause_text = "Standard insurance clause terms and exclusions. " * 30
    doc = PolicyDocument(
        policy_id="test-split",
        policy_name="Test Policy",
        insurer="Test Insurer",
        total_pages=1,
        sections=[
            Section(
                section_id="SEC-1",
                title="Coverage",
                page_start=1,
                page_end=1,
                clauses=[
                    Clause(
                        clause_id="1.1",
                        title="Large Inpatient Clause",
                        text=long_clause_text,
                        page_start=1,
                        page_end=1,
                        chunk_type=ChunkType.COVERAGE,
                    )
                ],
            )
        ],
    )

    chunker = ClauseAwareChunker(min_chunk_chars=50, max_chunk_chars=400, overlap_chars=50)
    chunks = chunker.chunk(doc)

    # Must be split into multiple parts
    assert len(chunks) > 1
    assert "Part 1" in chunks[0].clause_title
    assert "Part 2" in chunks[1].clause_title
    for c in chunks:
        assert len(c.text) <= 550  # 400 + header


def test_fixed_size_chunker(dummy_policy_document: PolicyDocument) -> None:
    chunker = FixedSizeChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk(dummy_policy_document)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.section == "FixedWindow"
        assert chunk.clause_id.startswith("FIXED-")
        assert len(chunk.text) <= 150
