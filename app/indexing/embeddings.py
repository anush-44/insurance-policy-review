from typing import Protocol

from app.core.config import settings
from app.core.logging import logger


class EmbeddingModel(Protocol):
    """Protocol for dense vector embedding models."""

    @property
    def dimension(self) -> int: ...

    def embed_query(self, query: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class BGEEmbeddingModel:
    """Production wrapper around BAAI/BGE sentence-transformers embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._dim: int | None = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Infer dimension
            test_vec = self._model.encode("dim test")
            self._dim = len(test_vec)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Common defaults for BGE
            if "base" in self.model_name:
                return 768
            elif "small" in self.model_name:
                return 384
            elif "large" in self.model_name:
                return 1024
            self._load_model()
        return self._dim or 768

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query string."""
        model = self._load_model()
        # BGE recommends prefixing queries with instruction if using retrieval mode
        prefix = "Represent this sentence for searching relevant passages: "
        query_text = f"{prefix}{query}" if "bge" in self.model_name.lower() else query
        vec = model.encode(
            query_text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vec.tolist()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode a batch of document texts."""
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vectors.tolist()


class DeterministicMockEmbeddingModel:
    """Lightweight pseudo-embedding model for offline testing and fast CI runs."""

    def __init__(self, dimension: int = 64) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_to_vector(self, text: str) -> list[float]:
        import hashlib
        import math

        vec = []
        for i in range(self._dim):
            seed = f"{text}_{i}".encode()
            h = int(hashlib.md5(seed).hexdigest()[:8], 16)
            vec.append((h % 1000) / 1000.0 - 0.5)

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_to_vector(query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]
