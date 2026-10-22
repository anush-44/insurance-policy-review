from app.core.schemas import ChunkType, PolicyChunk
from app.retrieval.rrf import reciprocal_rank_fusion


def make_chunk(cid: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=cid,
        policy_id="p1",
        insurer="Insurer A",
        section="Section 1",
        clause_id=f"C-{cid}",
        clause_title=f"Title {cid}",
        text=f"Text for chunk {cid}",
        page_start=1,
        page_end=1,
        chunk_type=ChunkType.COVERAGE,
    )


def test_rrf_empty_lists() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list() -> None:
    c1, c2, c3 = make_chunk("c1"), make_chunk("c2"), make_chunk("c3")
    ranked = [(c1, 0.9), (c2, 0.8), (c3, 0.7)]
    fused = reciprocal_rank_fusion([ranked], k=60)

    assert len(fused) == 3
    assert fused[0][0].chunk_id == "c1"
    assert fused[1][0].chunk_id == "c2"
    assert fused[2][0].chunk_id == "c3"
    assert fused[0][1] == 1.0 / (60 + 1)
    assert fused[1][1] == 1.0 / (60 + 2)


def test_rrf_disjoint_and_overlap() -> None:
    c1, c2, c3, c4 = make_chunk("c1"), make_chunk("c2"), make_chunk("c3"), make_chunk("c4")

    # Dense: [c1, c2, c3]
    dense = [(c1, 0.95), (c2, 0.85), (c3, 0.75)]
    # BM25: [c2, c4, c1]
    bm25 = [(c2, 12.0), (c4, 8.5), (c1, 4.0)]

    fused = reciprocal_rank_fusion([dense, bm25], k=60, limit=2)

    # c2 is rank 2 in dense and rank 1 in bm25:
    # score(c2) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~= 0.016129 + 0.016393 = 0.03252
    # c1 is rank 1 in dense and rank 3 in bm25:
    # score(c1) = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ~= 0.016393 + 0.015873 = 0.03226
    # So c2 should be #1!
    assert len(fused) == 2
    assert fused[0][0].chunk_id == "c2"
    assert fused[1][0].chunk_id == "c1"
    assert fused[0][1] > fused[1][1]
