"""Dependency injection: shared RAG pipeline and BM25 index loaded at startup."""

import json
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import PolicyChunk
from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import BGEEmbeddingModel, DeterministicMockEmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore, get_qdrant_client
from app.rag.pipeline import RAGPipeline

# ── Singletons ────────────────────────────────────────────────────────────────
# These are initialised once at application startup via lifespan context.

_rag_pipeline: RAGPipeline | None = None
_bm25_index: BM25Index | None = None
_vector_store: QdrantVectorStore | None = None


def _load_bm25_from_storage() -> BM25Index:
    """Re-hydrate BM25 index from all serialised JSON files in qdrant_local_path."""
    bm25 = BM25Index()
    storage_dir = Path(settings.qdrant_local_path)
    if not storage_dir.exists():
        return bm25

    all_chunks: list[PolicyChunk] = []
    for bm25_file in storage_dir.glob("*_bm25.json"):
        try:
            raw = json.loads(bm25_file.read_text(encoding="utf-8"))
            for item in raw:
                all_chunks.append(PolicyChunk(**item))
        except Exception as exc:
            logger.warning(f"Failed to load BM25 file {bm25_file}: {exc}")

    if all_chunks:
        bm25.index_chunks(all_chunks)
        logger.info(f"Loaded BM25 index with {len(all_chunks)} chunks from storage")
    else:
        logger.info("No BM25 index files found — index is empty")

    return bm25


def create_rag_pipeline() -> RAGPipeline:
    """Construct and return a fully wired RAGPipeline."""
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client)

    # Use real embeddings if sentence_transformers is installed, else mock
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401

        embedding_model = BGEEmbeddingModel()
        logger.info("Using BGEEmbeddingModel for embeddings")
    except ImportError:
        logger.warning(
            "sentence_transformers not installed — using DeterministicMockEmbeddingModel"
        )
        embedding_model = DeterministicMockEmbeddingModel(dimension=768)

    bm25_index = _load_bm25_from_storage()

    pipeline = RAGPipeline(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_model=embedding_model,
    )
    return pipeline


def get_pipeline() -> RAGPipeline:
    """FastAPI dependency that returns the shared RAGPipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = create_rag_pipeline()
    return _rag_pipeline


def reload_bm25(pipeline: RAGPipeline) -> None:
    """Reload BM25 index from disk (called after ingestion)."""
    new_bm25 = _load_bm25_from_storage()
    pipeline.retriever.bm25_index = new_bm25
    logger.info("BM25 index reloaded after ingestion")
