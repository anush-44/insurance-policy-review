"""Evaluate the existing indexed corpus with real BM25 and BGE embeddings.

This is an exploratory benchmark over the current labeled set, not a held-out
test. No document vectors are rebuilt. Run after acquisition and ingestion:
``python -m eval.evaluate_indexed_retrieval``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import settings
from app.core.schemas import PolicyChunk
from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import BGEEmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore, get_qdrant_client
from app.retrieval.hybrid import HybridRetriever
from eval.run_retrieval_ablation import _relevant_references
from eval.validate_eval_data import validate_qa_references, validate_qa_set


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scores(question_results: list[dict]) -> dict[str, float]:
    if not question_results:
        raise ValueError("No answerable questions to score")
    n = len(question_results)
    return {
        "hit_at_1": sum(any(rank == 1 for rank in r["ranks"]) for r in question_results) / n,
        "hit_at_5": sum(any(rank <= 5 for rank in r["ranks"]) for r in question_results) / n,
        "recall_at_5": sum(
            sum(rank <= 5 for rank in r["ranks"]) / r["refs"] for r in question_results
        )
        / n,
        "recall_at_10": sum(
            sum(rank <= 10 for rank in r["ranks"]) / r["refs"] for r in question_results
        )
        / n,
        "mrr_at_10": sum(
            1 / min(r["ranks"]) if min(r["ranks"]) <= 10 else 0 for r in question_results
        )
        / n,
    }


def _bootstrap(question_results: list[dict], repeats: int = 2000) -> dict[str, list[float]]:
    rng = random.Random(20261121)
    n = len(question_results)
    draws: dict[str, list[float]] = {key: [] for key in _scores(question_results)}
    for _ in range(repeats):
        sample = [question_results[rng.randrange(n)] for _ in range(n)]
        for key, value in _scores(sample).items():
            draws[key].append(value)
    return {
        key: [round(sorted(values)[49], 4), round(sorted(values)[1949], 4)]
        for key, values in draws.items()
    }


def evaluate(qa_path: Path, manifest_path: Path, index_dir: Path) -> dict:
    validation = validate_qa_references(qa_path, manifest_path)
    if not validation["valid"]:
        raise ValueError("QA references invalid: " + "; ".join(validation["errors"]))
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    official = {row["policy_id"]: row for row in rows if row["validation_status"] == "valid"}
    chunks: list[PolicyChunk] = []
    index_hashes = {}
    for policy_id in official:
        path = index_dir / f"{policy_id}_bm25.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing index for validated policy: {path}")
        index_hashes[policy_id] = _hash(path)
        chunks.extend(
            PolicyChunk.model_validate(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        )
    pairs = [
        pair
        for pair in validate_qa_set(qa_path)
        if pair.policy_id in official and pair.answerability == "answerable"
    ]
    if not pairs:
        raise ValueError("No answerable questions for validated policies")
    bm25 = BM25Index()
    bm25.index_chunks(chunks)
    client = get_qdrant_client(mode="local", path=str(index_dir))
    try:
        store = QdrantVectorStore(client=client)
        embedding = BGEEmbeddingModel()
        configurations = [
            ("bm25", "bm25", False),
            ("dense", "dense", False),
            ("hybrid_rrf", "hybrid", False),
            ("hybrid_rrf_policy_filter", "hybrid", True),
        ]
        outputs = []
        for name, mode, filtered in configurations:
            retriever = HybridRetriever(
                vector_store=store,
                bm25_index=bm25,
                embedding_model=embedding,
                retrieval_mode=mode,
                reranker_enabled=False,
            )
            question_results = []
            latencies = []
            for pair in pairs:
                started = time.perf_counter()
                found = retriever.retrieve(
                    pair.question,
                    limit=10,
                    policy_id=pair.policy_id if filtered else None,
                )
                latencies.append((time.perf_counter() - started) * 1000)
                ranks = {}
                for rank, (chunk, _) in enumerate(found, start=1):
                    if chunk.policy_id != pair.policy_id:
                        continue
                    for ref in _relevant_references(chunk, pair):
                        ranks.setdefault(ref, rank)
                question_results.append(
                    {
                        "id": pair.id,
                        "policy_id": pair.policy_id,
                        "ranks": [ranks.get(i, 11) for i in range(len(pair.gold_clause_ids))],
                        "refs": len(pair.gold_clause_ids),
                        "retrieved_chunk_ids": [chunk.chunk_id for chunk, _ in found],
                    }
                )
            outputs.append(
                {
                    "setup": name,
                    "filter_is_given_gold_policy_id": filtered,
                    "metrics": _scores(question_results),
                    "bootstrap_95_percent_ci": _bootstrap(question_results),
                    "latency_ms_median": round(statistics.median(latencies), 2),
                    "latency_ms_p95": round(sorted(latencies)[int((len(latencies) - 1) * 0.95)], 2),
                    "question_results": question_results,
                }
            )
        coverage = {}
        for pair in pairs:
            covered = set()
            for chunk in chunks:
                if chunk.policy_id == pair.policy_id:
                    covered.update(_relevant_references(chunk, pair))
            coverage[pair.id] = len(covered)
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "evaluation_status": "exploratory_dev_questions_not_held_out",
            "dataset": str(qa_path),
            "dataset_sha256": _hash(qa_path),
            "manifest_sha256": _hash(manifest_path),
            "index_sha256": index_hashes,
            "embedding_model": settings.embedding_model,
            "rrf_k": settings.rrf_k,
            "candidate_depth": {"bm25": settings.top_k_bm25, "dense": settings.top_k_dense},
            "official_policies": len(official),
            "indexed_chunks": len(chunks),
            "answerable_questions": len(pairs),
            "unanswerable_questions_not_scored": sum(
                p.answerability == "unanswerable" and p.policy_id in official
                for p in validate_qa_set(qa_path)
            ),
            "gold_references_present_in_index": sum(coverage.values()),
            "gold_references_total": sum(len(p.gold_clause_ids) for p in pairs),
            "results": outputs,
        }
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path("eval/qa_set.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/policies.csv"))
    parser.add_argument("--index-dir", type=Path, default=Path(settings.qdrant_local_path))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = evaluate(args.qa, args.manifest, args.index_dir)
    out = (
        args.out
        or Path("eval/results")
        / f"indexed_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Saved {len(report['results'])} setups on {report['answerable_questions']} questions to {out}"
    )
    for row in report["results"]:
        print(f"{row['setup']}: {row['metrics']}")


if __name__ == "__main__":
    main()
