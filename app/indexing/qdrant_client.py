import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import ChunkType, PolicyChunk


def get_qdrant_client(mode: str | None = None, path: str | None = None) -> QdrantClient:
    """Factory creating QdrantClient based on configuration (local disk/memory vs docker server)."""
    selected_mode = mode or settings.qdrant_mode

    if selected_mode == "server":
        logger.info(f"Connecting to Qdrant server at {settings.qdrant_host}:{settings.qdrant_port}")
        return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    elif selected_mode == "memory":
        logger.info("Initializing in-memory ephemeral Qdrant client")
        return QdrantClient(":memory:")
    else:
        # local disk mode
        target_path = path or settings.qdrant_local_path
        logger.info(f"Initializing local embedded Qdrant client at '{target_path}'")
        return QdrantClient(path=str(target_path))


class QdrantVectorStore:
    """Production vector store managing policy chunk indexing and filtered dense retrieval."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.client = client or get_qdrant_client()
        self.collection_name = collection_name or settings.qdrant_collection_name

    def ensure_collection(self, vector_dim: int, recreate: bool = False) -> None:
        """Create collection if it does not already exist."""
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            logger.info(f"Deleting existing collection '{self.collection_name}' for recreation")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' (dim={vector_dim}, metric=Cosine)"
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
            # Create payload indexes on server mode if applicable
            if settings.qdrant_mode == "server":
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="policy_id",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD,
                    )
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="chunk_type",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD,
                    )
                except Exception as e:
                    logger.debug(f"Payload index creation skipped: {e}")

    def upsert_chunks(
        self,
        chunks: list[PolicyChunk],
        vectors: list[list[float]],
    ) -> None:
        """Insert or update chunks with dense vectors into Qdrant."""
        if not chunks:
            return

        vector_dim = len(vectors[0])
        self.ensure_collection(vector_dim=vector_dim)

        points: list[PointStruct] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            # Generate deterministic UUID from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "policy_id": chunk.policy_id,
                "insurer": chunk.insurer,
                "section": chunk.section,
                "clause_id": chunk.clause_id,
                "clause_title": chunk.clause_title,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_type": str(chunk.chunk_type),
                "cross_references": chunk.cross_references,
                **chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Indexed {len(points)} chunks into '{self.collection_name}'")

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        policy_id: str | None = None,
        chunk_types: list[ChunkType] | None = None,
    ) -> list[tuple[PolicyChunk, float]]:
        """Search nearest chunks by cosine similarity with optional metadata payload filtering."""
        conditions = []
        if policy_id:
            conditions.append(FieldCondition(key="policy_id", match=MatchValue(value=policy_id)))
        if chunk_types:
            conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchAny(any=[str(ct) for ct in chunk_types]),
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        # Supports both modern query_points and legacy search API
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
            hits = response.points
        else:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
            )

        results: list[tuple[PolicyChunk, float]] = []
        for hit in hits:
            p = hit.payload or {}
            chunk = PolicyChunk(
                chunk_id=str(p.get("chunk_id", hit.id)),
                policy_id=str(p.get("policy_id", "")),
                insurer=str(p.get("insurer", "")),
                section=str(p.get("section", "")),
                clause_id=str(p.get("clause_id", "")),
                clause_title=str(p.get("clause_title", "")),
                text=str(p.get("text", "")),
                page_start=int(p.get("page_start", 1)),
                page_end=int(p.get("page_end", 1)),
                chunk_type=ChunkType(p.get("chunk_type", ChunkType.OTHER)),
                cross_references=list(p.get("cross_references", [])),
                metadata=p,
            )
            results.append((chunk, float(hit.score)))

        return results
