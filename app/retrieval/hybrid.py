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
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_model = embedding_model
        self.top_k_dense = top_k_dense or settings.top_k_dense
        self.top_k_bm25 = top_k_bm25 or settings.top_k_bm25
        self.rrf_k = rrf_k or settings.rrf_k

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        policy_id: str | None = None,
        chunk_types: list[ChunkType] | None = None,
    ) -> list[tuple[PolicyChunk, float]]:
        """Run hybrid retrieval: Dense + BM25 merged with RRF."""
        # 1. Dense retrieval
        query_vec = self.embedding_model.embed_query(query)
        dense_results = self.vector_store.search(
            query_vector=query_vec,
            limit=self.top_k_dense,
            policy_id=policy_id,
            chunk_types=chunk_types,
        )

        # 2. BM25 keyword retrieval
        bm25_results = self.bm25_index.search(
            query=query,
            limit=self.top_k_bm25,
            policy_id=policy_id,
            chunk_types=chunk_types,
        )

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            k=self.rrf_k,
            limit=limit,
        )

        logger.info(
            f"Hybrid retrieval for '{query[:40]}...': "
            f"dense={len(dense_results)}, bm25={len(bm25_results)} -> fused={len(fused)}"
        )
        return fused
