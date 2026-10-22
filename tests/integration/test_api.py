"""FastAPI integration tests using TestClient (no live server required).

Tests all three API routers: /api/v1/policies, /api/v1/qa, /api/v1/analyse.
"""

import json
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.llm import MockLLMClient
from app.core.schemas import ChunkType, PolicyChunk, PolicyDocument, Section, Clause
from app.indexing.bm25_index import BM25Index
from app.indexing.chunker import ClauseAwareChunker
from app.indexing.embeddings import DeterministicMockEmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore, get_qdrant_client
from app.rag.pipeline import RAGPipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mock_pipeline() -> RAGPipeline:
    """Shared in-memory RAGPipeline with a small synthetic policy pre-indexed."""
    doc = PolicyDocument(
        policy_id="hdfc_test",
        policy_name="HDFC Test Policy",
        insurer="HDFC ERGO",
        total_pages=4,
        sections=[
            Section(
                section_id="S1",
                title="Coverage",
                page_start=1,
                page_end=2,
                clauses=[
                    Clause(
                        clause_id="1.1",
                        title="In-Patient Hospitalisation",
                        text=(
                            "The policy covers in-patient hospitalisation expenses for a minimum "
                            "of 24 hours including room rent, doctor fees, and surgery costs."
                        ),
                        page_start=1,
                        page_end=1,
                        chunk_type=ChunkType.COVERAGE,
                    ),
                    Clause(
                        clause_id="1.2",
                        title="PED Waiting Period",
                        text=(
                            "Pre-existing diseases are excluded for 36 months of continuous coverage."
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
                        title="General Exclusions",
                        text="Dental treatment, cosmetic surgery, and self-inflicted injuries are excluded.",
                        page_start=3,
                        page_end=3,
                        chunk_type=ChunkType.EXCLUSION,
                    ),
                ],
            ),
        ],
    )

    embedding_model = DeterministicMockEmbeddingModel(dimension=64)
    client = get_qdrant_client(mode="memory")
    vector_store = QdrantVectorStore(client=client)
    chunker = ClauseAwareChunker()
    chunks = chunker.chunk(doc)
    vectors = embedding_model.embed_documents([c.text for c in chunks])
    vector_store.upsert_chunks(chunks=chunks, vectors=vectors)

    bm25 = BM25Index()
    bm25.index_chunks(chunks)

    return RAGPipeline(
        vector_store=vector_store,
        bm25_index=bm25,
        embedding_model=embedding_model,
        llm=MockLLMClient(),
    )


@pytest.fixture(scope="module")
def client(mock_pipeline: RAGPipeline, tmp_path_factory) -> TestClient:
    """TestClient with the RAG pipeline patched to use our in-memory mock."""
    from app.main import app
    from app import api as api_module

    # Patch the pipeline singleton
    api_module._rag_pipeline = mock_pipeline

    # Patch the policy registry for tests
    registry = {
        "hdfc_test": {
            "policy_id": "hdfc_test",
            "policy_name": "HDFC Test Policy",
            "insurer": "HDFC ERGO",
            "total_pages": 4,
            "total_chunks": 3,
        }
    }

    with (
        patch("app.cli.list_policies", return_value=list(registry.values())),
        patch("app.api.policies.list_policies", return_value=list(registry.values())),
        patch("app.api.analyse.list_policies", return_value=list(registry.values())),
    ):
        with TestClient(app, raise_server_exceptions=True) as tc:
            yield tc


# ── Health endpoint ────────────────────────────────────────────────────────────


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_returns_service_info(client: TestClient) -> None:
    resp = client.get("/")
    data = resp.json()
    assert data["service"] == "PolicyLens"
    assert "disclaimer" in data


# ── Q&A endpoint ──────────────────────────────────────────────────────────────


def test_qa_endpoint_grounded_answer(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/qa",
        json={
            "question": "What is the waiting period for pre-existing diseases?",
            "policy_id": "hdfc_test",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "confidence" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)


def test_qa_endpoint_abstains_on_out_of_scope(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/qa",
        json={"question": "Is my pet dog covered?", "policy_id": "hdfc_test", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_abstention"] is True


def test_qa_endpoint_validation_error_on_short_question(client: TestClient) -> None:
    resp = client.post("/api/v1/qa", json={"question": "hi", "top_k": 5})
    assert resp.status_code == 422  # Pydantic min_length=5


# ── Analysis endpoints ─────────────────────────────────────────────────────────


def test_analyse_summary_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/analyse/summary/hdfc_test")
    assert resp.status_code == 200
    data = resp.json()
    assert "policy_name" in data
    assert "key_exclusions" in data


def test_analyse_red_flags_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/analyse/red-flags/hdfc_test")
    assert resp.status_code == 200
    data = resp.json()
    assert "red_flags" in data
    assert isinstance(data["red_flags"], list)


def test_analyse_missing_policy_returns_404(client: TestClient) -> None:
    resp = client.get("/api/v1/analyse/summary/nonexistent_policy_xyz")
    assert resp.status_code == 404


# ── Policies list endpoint ────────────────────────────────────────────────────


def test_list_policies_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/policies")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
