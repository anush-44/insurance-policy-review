import math

from eval.run_retrieval_ablation import _metrics


def test_retrieval_metrics_for_single_gold_reference() -> None:
    result = _metrics(
        [
            {
                "ranks_by_reference": {0: 1},
                "references_found_at_5": {0},
                "total_references": 1,
            }
        ]
    )
    assert result["hit_at_5"] == 1.0
    assert result["recall_at_5"] == 1.0
    assert result["mrr_at_5"] == 1.0
    assert result["ndcg_at_5"] == 1.0


def test_retrieval_metrics_normalize_multiple_gold_references() -> None:
    result = _metrics(
        [
            {
                "ranks_by_reference": {0: 2, 1: 5},
                "references_found_at_5": {0, 1},
                "total_references": 2,
            }
        ]
    )
    expected = (1 / math.log2(3) + 1 / math.log2(6)) / (1 + 1 / math.log2(3))
    assert result["recall_at_5"] == 1.0
    assert result["mrr_at_5"] == 0.5
    assert abs(result["ndcg_at_5"] - expected) < 1e-9


def test_retrieval_metrics_return_zero_for_missing_evidence() -> None:
    result = _metrics(
        [
            {
                "ranks_by_reference": {},
                "references_found_at_5": set(),
                "total_references": 2,
            }
        ]
    )
    assert result["hit_at_5"] == 0.0
    assert result["recall_at_5"] == 0.0
    assert result["mrr_at_5"] == 0.0
    assert result["ndcg_at_5"] == 0.0
