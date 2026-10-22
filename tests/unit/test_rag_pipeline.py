"""Unit tests for the RAG pipeline using MockLLMClient and DeterministicMockEmbeddingModel."""

import pytest

from app.core.llm import MockLLMClient
from app.core.schemas import ChunkType, GroundedAnswer, PolicyChunk, PolicyDocument, Section, Clause
from app.indexing.bm25_index import BM25Index
from app.indexing.chunker import ClauseAwareChunker
from app.indexing.embeddings import DeterministicMockEmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore, get_qdrant_client
from app.rag.pipeline import RAGPipeline
from app.rag.prompts import build_qa_prompt, build_summary_prompt, build_red_flags_prompt


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_doc() -> PolicyDocument:
    """Minimal synthetic PolicyDocument for testing."""
    return PolicyDocument(
        policy_id="test_policy",
        policy_name="Test Health Policy",
        insurer="Test Insurer",
        total_pages=5,
        sections=[
            Section(
                section_id="S1",
                title="Waiting Periods",
                page_start=1,
                page_end=2,
                clauses=[
                    Clause(
                        clause_id="1.1",
                        title="PED Waiting Period",
                        text=(
                            "Expenses related to the treatment of a Pre-Existing Disease (PED) "
                            "shall be excluded until 36 months of continuous coverage have elapsed "
                            "since inception of the first policy with Us."
                        ),
                        page_start=1,
                        page_end=1,
                        chunk_type=ChunkType.WAITING_PERIOD,
                    ),
                    Clause(
                        clause_id="1.2",
                        title="Specific Disease Waiting Period",
                        text=(
                            "A waiting period of 24 months applies to specific diseases including "
                            "cataract, hernia, knee replacement, and joint replacement surgeries."
                        ),
                        page_start=2,
                        page_end=2,
                        chunk_type=ChunkType.WAITING_PERIOD,
                    ),
                ],
            ),
            Section(
                section_id="S2",
                title="Exclusions",
                page_start=3,
                page_end=4,
                clauses=[
                    Clause(
                        clause_id="2.1",
                        title="Permanent Exclusions",
                        text=(
                            "The following are permanently excluded: treatment for dental, "
                            "cosmetic surgery, infertility, self-inflicted injuries, war injuries, "
                            "and HIV/AIDS-related conditions."
                        ),
                        page_start=3,
                        page_end=3,
                        chunk_type=ChunkType.EXCLUSION,
                    ),
                    Clause(
                        clause_id="2.2",
                        title="Co-pay",
                        text=(
                            "A co-payment of 10% applies to all eligible claims where treatment "
                            "is taken in a non-network hospital."
                        ),
                        page_start=4,
                        page_end=4,
                        chunk_type=ChunkType.SUB_LIMIT,
                    ),
                ],
            ),
        ],
    )


@pytest.fixture()
def indexed_pipeline(sample_doc: PolicyDocument) -> RAGPipeline:
    """RAGPipeline with in-memory Qdrant, DeterministicMock embeddings, and MockLLM."""
    embedding_model = DeterministicMockEmbeddingModel(dimension=64)
    client = get_qdrant_client(mode="memory")
    vector_store = QdrantVectorStore(client=client)

    # Index the sample doc
    chunker = ClauseAwareChunker()
    chunks = chunker.chunk(sample_doc)
    vectors = embedding_model.embed_documents([c.text for c in chunks])
    vector_store.upsert_chunks(chunks=chunks, vectors=vectors)

    bm25 = BM25Index()
    bm25.index_chunks(chunks)

    pipeline = RAGPipeline(
        vector_store=vector_store,
        bm25_index=bm25,
        embedding_model=embedding_model,
        llm=MockLLMClient(),
        abstention_threshold=0.35,
    )
    return pipeline


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRAGPipeline:
    def test_answer_returns_grounded_answer(self, indexed_pipeline: RAGPipeline) -> None:
        """Normal query returns a GroundedAnswer with citations."""
        result = indexed_pipeline.answer(
            query="What is the waiting period for pre-existing diseases?",
            policy_id="test_policy",
        )
        assert isinstance(result, GroundedAnswer)
        assert len(result.answer) > 0
        assert result.retrieved_chunk_ids  # at least one chunk retrieved

    def test_answer_abstains_on_out_of_scope(self, indexed_pipeline: RAGPipeline) -> None:
        """Query about topics completely outside the policy triggers abstention."""
        result = indexed_pipeline.answer(
            query="Is my pet covered by this policy?",
            policy_id="test_policy",
        )
        assert result.is_abstention is True
        assert result.confidence == 0.0

    def test_answer_with_no_policy_filter(self, indexed_pipeline: RAGPipeline) -> None:
        """Q&A works without specifying a policy_id (searches all indexed chunks)."""
        result = indexed_pipeline.answer(query="What are the exclusions?")
        assert isinstance(result, GroundedAnswer)

    def test_retrieved_chunk_ids_populated(self, indexed_pipeline: RAGPipeline) -> None:
        """retrieved_chunk_ids is always populated when chunks exist."""
        result = indexed_pipeline.answer(
            query="What is the co-pay?", policy_id="test_policy"
        )
        assert result.retrieved_chunk_ids is not None

    def test_detect_red_flags_returns_report(self, indexed_pipeline: RAGPipeline) -> None:
        """Red flags detection returns a RedFlagsReport (may be empty with MockLLM)."""
        from app.rag.schemas import RedFlagsReport

        result = indexed_pipeline.detect_red_flags(policy_id="test_policy")
        assert isinstance(result, RedFlagsReport)
        assert isinstance(result.red_flags, list)

    def test_summarise_returns_card(self, indexed_pipeline: RAGPipeline) -> None:
        """Summary extraction returns a PolicySummaryCard."""
        from app.rag.schemas import PolicySummaryCard

        result = indexed_pipeline.summarise(
            policy_id="test_policy",
            policy_name="Test Health Policy",
            insurer="Test Insurer",
        )
        assert isinstance(result, PolicySummaryCard)
        # policy_name and insurer should be populated from fallback
        assert result.policy_name == "Test Health Policy"
        assert result.insurer == "Test Insurer"


class TestPromptBuilders:
    def test_build_qa_prompt_contains_query(self) -> None:
        chunks = [
            PolicyChunk(
                chunk_id="c1",
                policy_id="p1",
                insurer="Insurer",
                section="S1",
                clause_id="1.1",
                clause_title="PED Waiting",
                text="Pre-existing diseases excluded for 36 months.",
                page_start=1,
                page_end=1,
            )
        ]
        prompt = build_qa_prompt("What is the PED waiting period?", chunks)
        assert "PED waiting period" in prompt
        assert "1.1" in prompt
        assert "36 months" in prompt

    def test_build_summary_prompt_caps_chunks(self) -> None:
        # Should not crash even with many chunks
        chunks = [
            PolicyChunk(
                chunk_id=f"c{i}",
                policy_id="p1",
                insurer="I",
                section="S",
                clause_id=str(i),
                clause_title=f"Clause {i}",
                text=f"Text for clause {i}",
                page_start=i,
                page_end=i,
            )
            for i in range(60)
        ]
        prompt = build_summary_prompt(chunks)
        assert "CONTEXT" in prompt
        assert "waiting_period_ped" in prompt

    def test_build_red_flags_prompt_prioritises_exclusions(self) -> None:
        exclusion_chunk = PolicyChunk(
            chunk_id="ex1",
            policy_id="p1",
            insurer="I",
            section="Exclusions",
            clause_id="2.1",
            clause_title="Exclusion",
            text="Dental treatment is permanently excluded.",
            page_start=5,
            page_end=5,
            chunk_type=ChunkType.EXCLUSION,
        )
        coverage_chunk = PolicyChunk(
            chunk_id="cov1",
            policy_id="p1",
            insurer="I",
            section="Coverage",
            clause_id="1.1",
            clause_title="Coverage",
            text="Hospitalisation expenses are covered.",
            page_start=1,
            page_end=1,
            chunk_type=ChunkType.COVERAGE,
        )
        prompt = build_red_flags_prompt([exclusion_chunk, coverage_chunk])
        # Exclusion chunk text should appear in prompt
        assert "Dental treatment" in prompt
