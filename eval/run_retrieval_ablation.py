"""Controlled retrieval/chunking ablations over the validated QA labels.

Example: ``python -m eval.run_retrieval_ablation --policy-id icici-complete-health``
Use ``--mock-embeddings`` only for pipeline diagnostics; those results are not
evidence of retrieval quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.schemas import PolicyChunk
from app.indexing.bm25_index import BM25Index
from app.indexing.chunker import ClauseAwareChunker, FixedSizeChunker
from app.indexing.embeddings import BGEEmbeddingModel, DeterministicMockEmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore, get_qdrant_client
from app.ingestion.pdf_parser import PyMuPDFParser
from app.retrieval.hybrid import HybridRetriever
from eval.validate_eval_data import QAPair, validate_qa_references, validate_qa_set


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relevant_references(chunk: PolicyChunk, pair: QAPair) -> set[int]:
    normalized_text = "".join(ch.lower() for ch in chunk.text if ch.isalnum())
    normalized_id = "".join(ch.lower() for ch in chunk.clause_id if ch.isalnum())
    found: set[int] = set()
    for ref_index, (clause_id, page) in enumerate(zip(pair.gold_clause_ids, pair.gold_pages)):
        normalized_gold = "".join(ch.lower() for ch in clause_id if ch.isalnum())
        page_hit = chunk.page_start <= page <= chunk.page_end
        clause_hit = normalized_gold in normalized_text or normalized_gold in normalized_id
        if page_hit and clause_hit:
            found.add(ref_index)
    return found


def _metrics(results: list[dict[str, Any]]) -> dict[str, float | int]:
    n = len(results)
    hits = [item for item in results if any(rank <= 5 for rank in item["ranks_by_reference"].values())]
    recall = [
        len(item["references_found_at_5"]) / max(1, item["total_references"])
        for item in results
    ]
    reciprocal_ranks = []
    for item in results:
        relevant_ranks = [rank for rank in item["ranks_by_reference"].values() if rank <= 5]
        reciprocal_ranks.append(1 / min(relevant_ranks) if relevant_ranks else 0.0)
    ndcgs: list[float] = []
    for item in results:
        dcg = sum(1 / math.log2(rank + 1) for rank in item["ranks_by_reference"].values() if rank <= 5)
        ideal_count = min(5, item["total_references"])
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {
        "n_answerable": n,
        "hit_at_5": len(hits) / n if n else 0.0,
        "recall_at_5": sum(recall) / n if n else 0.0,
        "mrr_at_5": sum(reciprocal_ranks) / n if n else 0.0,
        "ndcg_at_5": sum(ndcgs) / n if n else 0.0,
        "missing_evidence_rate": 1 - len(hits) / n if n else 0.0,
    }


def run_ablation(policy_id: str, pdf_path: Path, mock_embeddings: bool, reranker: bool) -> dict[str, Any]:
    qa_path = Path("eval/qa_set.jsonl")
    validation = validate_qa_references(qa_path)
    if not validation["valid"]:
        raise ValueError("QA labels/reference validation failed: " + "; ".join(validation["errors"]))
    pairs = [pair for pair in validate_qa_set(qa_path) if pair.policy_id == policy_id]
    answerable = [pair for pair in pairs if pair.answerability == "answerable"]
    unanswerable = len(pairs) - len(answerable)
    if not answerable:
        raise ValueError(f"No answerable QA pairs for {policy_id}")

    with Path("data/policies.csv").open(newline="", encoding="utf-8-sig") as stream:
        manifest_row = next((row for row in csv.DictReader(stream) if row["policy_id"] == policy_id), None)
    if manifest_row is None:
        raise ValueError(f"Unknown policy ID in data/policies.csv: {policy_id}")
    doc = PyMuPDFParser().parse(
        file_source=pdf_path,
        policy_id=policy_id,
        policy_name=manifest_row["product_name"],
        insurer=manifest_row["insurer"],
        uin=manifest_row["uin"] or None,
    )
    embedding = DeterministicMockEmbeddingModel(64) if mock_embeddings else BGEEmbeddingModel()
    configurations = [
        ("bm25_only", "bm25", False),
        ("dense_only", "dense", False),
        ("hybrid_rrf", "hybrid", False),
    ]
    if reranker:
        configurations.append(("hybrid_cross_encoder", "hybrid", True))
    chunkers = {
        "clause_aware": ClauseAwareChunker(),
        "fixed_size": FixedSizeChunker(),
    }
    outputs: list[dict[str, Any]] = []
    for chunker_name, chunker in chunkers.items():
        chunks = chunker.chunk(doc)
        vectors = embedding.embed_documents([chunk.text for chunk in chunks])
        for name, mode, use_reranker in configurations:
            store = QdrantVectorStore(
                client=get_qdrant_client(mode="memory"),
                collection_name=f"eval_{chunker_name}_{name}",
            )
            store.upsert_chunks(chunks, vectors)
            bm25 = BM25Index()
            bm25.index_chunks(chunks)
            retriever = HybridRetriever(
                vector_store=store,
                bm25_index=bm25,
                embedding_model=embedding,
                retrieval_mode=mode,
                reranker_enabled=use_reranker,
            )
            question_results: list[dict[str, Any]] = []
            latencies: list[float] = []
            reranker_statuses: Counter[str] = Counter()
            errors = 0
            for pair in answerable:
                start = time.perf_counter()
                try:
                    results = retriever.retrieve(pair.question, limit=5, policy_id=pair.policy_id)
                    reranker_statuses[retriever.last_reranker_status] += 1
                    ranks_by_reference: dict[int, int] = {}
                    references_found_at_5: set[int] = set()
                    for idx, (chunk, _) in enumerate(results, start=1):
                        for ref_index in _relevant_references(chunk, pair):
                            ranks_by_reference.setdefault(ref_index, idx)
                            if idx <= 5:
                                references_found_at_5.add(ref_index)
                    question_results.append({
                        "ranks_by_reference": ranks_by_reference,
                        "references_found_at_5": references_found_at_5,
                        "total_references": len(pair.gold_clause_ids),
                    })
                except Exception:
                    errors += 1
                    question_results.append({
                        "ranks_by_reference": {},
                        "references_found_at_5": set(),
                        "total_references": len(pair.gold_clause_ids),
                    })
                latencies.append((time.perf_counter() - start) * 1000)
            metrics = _metrics(question_results)
            outputs.append({
                "chunking": chunker_name,
                "retrieval": name,
                "retrieval_mode": mode,
                "reranker_enabled": use_reranker,
                "reranker_status_counts": dict(reranker_statuses),
                "candidate_pool": settings.reranker_candidate_pool if use_reranker else max(settings.top_k_dense, settings.top_k_bm25),
                "chunk_count": len(chunks),
                "metrics": metrics,
                "latency_ms_mean": round(statistics.mean(latencies), 2),
                "latency_ms_median": round(statistics.median(latencies), 2),
                "query_failures": errors,
                "unanswerable_questions_not_scored_for_recall": unanswerable,
            })
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": str(qa_path),
        "dataset_sha256": _sha256(qa_path),
        "policy_id": policy_id,
        "pdf": str(pdf_path),
        "pdf_sha256": _sha256(pdf_path),
        "total_questions": len(pairs),
        "answerable_questions": len(answerable),
        "unanswerable_questions": unanswerable,
        "embedding_model": "deterministic-mock-64 (diagnostic only)" if mock_embeddings else settings.embedding_model,
        "reranker_model": settings.reranker_model if reranker else None,
        "reranker_requested": reranker,
        "evaluation_validity": "diagnostic_only_mock_embeddings" if mock_embeddings else "real_embeddings; LLM not used",
        "results": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--mock-embeddings", action="store_true")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run_ablation(args.policy_id, args.pdf, args.mock_embeddings, args.reranker)
    out_path = args.out or Path("eval/results") / f"retrieval_ablation_{args.policy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(report['results'])} controlled configurations to {out_path}")
    print(f"Dataset size: {report['total_questions']} ({report['answerable_questions']} answerable)")


if __name__ == "__main__":
    main()
