from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import ChunkType, PolicyChunk
from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import EmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore
from app.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Hybrid retrieval engine combining dense vector search and BM25 with Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index,
        embedding_model: EmbeddingModel,
        top_k_dense: int | None = None,
        top_k_bm25: int | None = None,
        rrf_k: int | None = None,
        reranker_enabled: bool | None = None,
        reranker: object | None = None,
        retrieval_mode: str | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_model = embedding_model
        self.top_k_dense = top_k_dense or settings.top_k_dense
        self.top_k_bm25 = top_k_bm25 or settings.top_k_bm25
        self.rrf_k = rrf_k or settings.rrf_k
        self.reranker_enabled = (
            settings.reranker_enabled if reranker_enabled is None else reranker_enabled
        )
        self._reranker = reranker
        self.retrieval_mode = retrieval_mode or settings.retrieval_mode
        self.last_reranker_status = "disabled"

    def _rerank(
        self, query: str, candidates: list[tuple[PolicyChunk, float]], limit: int
    ) -> list[tuple[PolicyChunk, float]]:
        if not self.reranker_enabled:
            self.last_reranker_status = "disabled"
            return candidates[:limit]
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    settings.reranker_model, device=settings.embedding_device
                )
            pairs = [(query, chunk.text) for chunk, _ in candidates]
            scores = self._reranker.predict(pairs)
            ranked = sorted(
                (
                    (chunk, float(score))
                    for (chunk, _), score in zip(candidates, scores, strict=True)
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            self.last_reranker_status = "applied"
            return ranked[:limit]
        except Exception as exc:
            self.last_reranker_status = f"fallback_rrf: {type(exc).__name__}: {exc}"
            logger.warning("Cross-encoder reranking unavailable; returning RRF order (%s)", exc)
            return candidates[:limit]

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        policy_id: str | None = None,
        chunk_types: list[ChunkType] | None = None,
    ) -> list[tuple[PolicyChunk, float]]:
        """Run hybrid retrieval: Dense + BM25 merged with RRF."""
        # 1. Dense retrieval
        dense_results: list[tuple[PolicyChunk, float]] = []
        bm25_results: list[tuple[PolicyChunk, float]] = []
        if self.retrieval_mode in {"dense", "hybrid"}:
            query_vec = self.embedding_model.embed_query(query)
            dense_results = self.vector_store.search(
                query_vector=query_vec,
                limit=self.top_k_dense,
                policy_id=policy_id,
                chunk_types=chunk_types,
            )
        if self.retrieval_mode in {"bm25", "hybrid"}:
            bm25_results = self.bm25_index.search(
                query=query,
                limit=self.top_k_bm25,
                policy_id=policy_id,
                chunk_types=chunk_types,
            )

        candidate_limit = (
            max(limit, settings.reranker_candidate_pool) if self.reranker_enabled else limit
        )
        if self.retrieval_mode == "dense":
            fused = dense_results[:candidate_limit]
        elif self.retrieval_mode == "bm25":
            fused = bm25_results[:candidate_limit]
        else:
            fused = reciprocal_rank_fusion(
                ranked_lists=[dense_results, bm25_results],
                k=self.rrf_k,
                limit=candidate_limit,
            )

        logger.info(
            f"Hybrid retrieval for '{query[:40]}...': "
            f"dense={len(dense_results)}, bm25={len(bm25_results)} -> fused={len(fused)}"
        )
        return self._rerank(query, fused, limit)
