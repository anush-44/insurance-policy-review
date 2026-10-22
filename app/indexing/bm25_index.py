import re

from rank_bm25 import BM25Okapi

from app.core.logging import logger
from app.core.schemas import ChunkType, PolicyChunk


def tokenize_insurance_text(text: str) -> list[str]:
    """Tokenize policy text preserving statutory clause numbers, codes, and acronyms.

    Preserves tokens like '4.2', 'excl04', 'co-pay', 'ayush', 'rs.40000'.
    """
    clean_text = text.lower().replace("-", " ")
    tokens = re.findall(r"\b[a-z0-9]+(?:\.[a-z0-9]+)*\b", clean_text)
    return tokens


class BM25Index:
    """Exact-match keyword search engine for insurance clauses."""

    def __init__(self) -> None:
        self.chunks: list[PolicyChunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def index_chunks(self, chunks: list[PolicyChunk]) -> None:
        """Build or update BM25 index from policy chunks."""
        self.chunks = list(chunks)
        self.tokenized_corpus = [
            tokenize_insurance_text(f"{c.clause_id} {c.clause_title} {c.text}") for c in self.chunks
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Built BM25 index over {len(self.chunks)} chunks")
        else:
            self.bm25 = None

    def search(
        self,
        query: str,
        limit: int = 10,
        policy_id: str | None = None,
        chunk_types: list[ChunkType] | None = None,
    ) -> list[tuple[PolicyChunk, float]]:
        """Search BM25 index with optional policy_id and chunk_type filtering."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize_insurance_text(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)

        # Pair chunks with scores
        scored: list[tuple[PolicyChunk, float]] = []
        for chunk, score in zip(self.chunks, doc_scores, strict=True):
            if score <= 0.0:
                continue
            if policy_id and chunk.policy_id != policy_id:
                continue
            if chunk_types and chunk.chunk_type not in chunk_types:
                continue
            scored.append((chunk, float(score)))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
