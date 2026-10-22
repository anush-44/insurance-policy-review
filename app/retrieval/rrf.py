from collections import defaultdict
from collections.abc import Sequence

from app.core.config import settings
from app.core.schemas import PolicyChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[PolicyChunk, float]]],
    k: int | None = None,
    limit: int | None = None,
) -> list[tuple[PolicyChunk, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = SUM_{m in M} (1 / (k + rank_m(d)))
    where rank_m(d) is 1-indexed.

    Args:
        ranked_lists: Sequence of ranked (PolicyChunk, score) lists from different retrievers.
        k: Smoothing constant (default from settings.rrf_k, usually 60).
        limit: Max number of fused results to return.

    Returns:
        Fused list of (PolicyChunk, rrf_score) sorted descending by rrf_score.
    """
    smoothing_k = k if k is not None else settings.rrf_k
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, PolicyChunk] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked_list, start=1):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] += 1.0 / (smoothing_k + rank)

    # Sort descending by fused RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    if limit is not None:
        sorted_items = sorted_items[:limit]

    return [(chunk_map[cid], score) for cid, score in sorted_items]
