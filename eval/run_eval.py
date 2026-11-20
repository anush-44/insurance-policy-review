"""PolicyLens Evaluation Harness.

Runs the full qa_set.jsonl against a live RAG pipeline and produces:
  - Per-question results (answer, citations, confidence, abstention, citation recall)
  - Aggregate metrics table (by category)
  - JSON results file in eval/results/

Usage (requires a real LLM key and indexed policy):
    python -m eval.run_eval --policy-id care-adv-sample --pdf data/sample_policy.pdf

Or with the MockLLMClient (no API key, for CI verification):
    python -m eval.run_eval --mock --policy-id care-adv-sample --pdf data/sample_policy.pdf
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.llm import MockLLMClient, get_llm_client
from app.core.logging import logger
from app.indexing.bm25_index import BM25Index
from app.indexing.chunker import ClauseAwareChunker
from app.indexing.embeddings import BGEEmbeddingModel, DeterministicMockEmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore, get_qdrant_client
from app.ingestion.pdf_parser import PyMuPDFParser
from app.rag.pipeline import RAGPipeline
from eval.validate_eval_data import QAPair, validate_qa_set

# ── Metrics ───────────────────────────────────────────────────────────────────


def citation_recall(
    gold_clause_id: str | None,
    predicted_clause_ids: list[str],
) -> float:
    """Return 1.0 if gold clause appears in predicted citations, 0.0 otherwise."""
    if gold_clause_id is None:
        return 1.0  # no gold citation expected
    return 1.0 if gold_clause_id in predicted_clause_ids else 0.0


def abstention_accuracy(gold_is_abstention: bool, pred_is_abstention: bool) -> float:
    """Return 1.0 if predicted abstention matches gold label."""
    return 1.0 if gold_is_abstention == pred_is_abstention else 0.0


def answer_non_empty(answer: str) -> bool:
    """True if model produced a non-trivial answer."""
    return len(answer.strip()) > 10


# ── Ingestion helper ──────────────────────────────────────────────────────────


def ingest_for_eval(
    pdf_path: Path,
    policy_id: str,
    use_mock_embeddings: bool = False,
) -> tuple[QdrantVectorStore, BM25Index, Any]:
    """Parse and index a PDF into in-memory stores for eval (no disk writes)."""
    parser = PyMuPDFParser()
    doc = parser.parse(
        file_source=pdf_path,
        policy_id=policy_id,
        policy_name=pdf_path.stem,
        insurer="EvalInsurer",
    )
    chunker = ClauseAwareChunker()
    chunks = chunker.chunk(doc)
    logger.info(f"[eval] Parsed {doc.total_pages} pages → {len(chunks)} chunks")

    emb = DeterministicMockEmbeddingModel(64) if use_mock_embeddings else BGEEmbeddingModel()
    vectors = emb.embed_documents([c.text for c in chunks])

    client = get_qdrant_client(mode="memory")
    vs = QdrantVectorStore(client=client)
    vs.upsert_chunks(chunks=chunks, vectors=vectors)

    bm25 = BM25Index()
    bm25.index_chunks(chunks)

    return vs, bm25, emb


# ── Main evaluation loop ──────────────────────────────────────────────────────


def run_evaluation(
    qa_pairs: list[QAPair],
    pipeline: RAGPipeline,
    policy_id: str,
) -> dict[str, Any]:
    """Run the RAG pipeline over all qa_pairs and collect per-question results."""
    results: list[dict[str, Any]] = []
    category_stats: dict[str, dict[str, list]] = {}

    for pair in qa_pairs:
        if pair.policy_id != policy_id:
            continue  # skip questions for other policies

        start = time.perf_counter()
        answer_obj = pipeline.answer(
            query=pair.question,
            policy_id=policy_id,
            top_k=settings.top_k_rerank,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract predicted clause IDs from citations
        predicted_clause_ids = [c.clause_id for c in answer_obj.citations]

        cite_recall = citation_recall(pair.gold_clause_id, predicted_clause_ids)
        abst_acc = abstention_accuracy(pair.is_abstention, answer_obj.is_abstention)
        has_answer = answer_non_empty(answer_obj.answer) if not answer_obj.is_abstention else True

        result: dict[str, Any] = {
            "id": pair.id,
            "category": pair.category,
            "question": pair.question,
            "gold_clause_id": pair.gold_clause_id,
            "gold_is_abstention": pair.is_abstention,
            "predicted_answer": answer_obj.answer,
            "predicted_citations": [c.model_dump() for c in answer_obj.citations],
            "predicted_clause_ids": predicted_clause_ids,
            "confidence": answer_obj.confidence,
            "predicted_is_abstention": answer_obj.is_abstention,
            "citation_recall": cite_recall,
            "abstention_accuracy": abst_acc,
            "has_non_empty_answer": has_answer,
            "latency_ms": round(elapsed_ms, 1),
        }
        results.append(result)

        # Accumulate per-category
        cat = pair.category
        if cat not in category_stats:
            category_stats[cat] = {
                "citation_recall": [],
                "abstention_accuracy": [],
                "confidence": [],
                "latency_ms": [],
            }
        category_stats[cat]["citation_recall"].append(cite_recall)
        category_stats[cat]["abstention_accuracy"].append(abst_acc)
        category_stats[cat]["confidence"].append(answer_obj.confidence)
        category_stats[cat]["latency_ms"].append(elapsed_ms)

        status = "✅" if cite_recall == 1.0 and abst_acc == 1.0 else "⚠️"
        print(
            f"  {status} [{pair.id}] {pair.question[:55]:<55} "
            f"recall={cite_recall:.0f}  conf={answer_obj.confidence:.2f}  {elapsed_ms:.0f}ms"
        )

    # Aggregate metrics
    def _avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    aggregate: dict[str, Any] = {
        "total": len(results),
        "citation_recall": _avg([r["citation_recall"] for r in results]),
        "abstention_accuracy": _avg([r["abstention_accuracy"] for r in results]),
        "avg_confidence": _avg([r["confidence"] for r in results]),
        "avg_latency_ms": _avg([r["latency_ms"] for r in results]),
        "has_non_empty_answer_rate": _avg([float(r["has_non_empty_answer"]) for r in results]),
    }

    category_summary: dict[str, dict] = {}
    for cat, stats in category_stats.items():
        category_summary[cat] = {
            "n": len(stats["citation_recall"]),
            "citation_recall": _avg(stats["citation_recall"]),
            "abstention_accuracy": _avg(stats["abstention_accuracy"]),
            "avg_confidence": _avg(stats["confidence"]),
            "avg_latency_ms": _avg(stats["latency_ms"]),
        }

    return {
        "policy_id": policy_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "aggregate": aggregate,
        "category_summary": category_summary,
        "results": results,
    }


# ── Pretty-print summary ──────────────────────────────────────────────────────


def print_summary(report: dict[str, Any]) -> None:
    agg = report["aggregate"]
    print("\n" + "=" * 70)
    print(f"  PolicyLens Evaluation — policy_id: {report['policy_id']}")
    print("=" * 70)
    print(f"  Questions evaluated : {agg['total']}")
    print(f"  Citation Recall     : {agg['citation_recall']:.1%}")
    print(f"  Abstention Accuracy : {agg['abstention_accuracy']:.1%}")
    print(f"  Avg Confidence      : {agg['avg_confidence']:.2f}")
    print(f"  Non-empty Answer %  : {agg['has_non_empty_answer_rate']:.1%}")
    print(f"  Avg Latency         : {agg['avg_latency_ms']:.0f} ms")
    print("-" * 70)
    print(f"  {'Category':<22} {'N':>4}  {'Recall':>7}  {'Abst.Acc':>9}  {'Conf':>6}  {'ms':>6}")
    print("  " + "-" * 62)
    for cat, stats in sorted(report["category_summary"].items()):
        print(
            f"  {cat:<22} {stats['n']:>4}  {stats['citation_recall']:>7.1%}  "
            f"{stats['abstention_accuracy']:>9.1%}  {stats['avg_confidence']:>6.2f}  "
            f"{stats['avg_latency_ms']:>6.0f}"
        )
    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PolicyLens evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Run with real LLM and real embeddings (needs GEMINI_API_KEY + sentence_transformers):
  python -m eval.run_eval --policy-id care-adv-sample --pdf data/sample_policy.pdf

  # Run fully offline (CI mode):
  python -m eval.run_eval --mock --policy-id care-adv-sample --pdf data/sample_policy.pdf
""",
    )
    parser.add_argument("--policy-id", default="care-adv-sample", help="Policy ID to evaluate")
    parser.add_argument(
        "--pdf",
        default=None,
        type=Path,
        help="Policy PDF (defaults to manifest path for the selected policy)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLMClient and DeterministicMockEmbeddingModel (offline, fast)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path (default: eval/results/<policy_id>_<timestamp>.json)",
    )
    args = parser.parse_args()

    # Validate eval data
    print("Loading eval dataset...")
    qa_pairs = validate_qa_set()
    relevant = [q for q in qa_pairs if q.policy_id == args.policy_id]
    print(
        f"Found {len(relevant)} questions for policy_id='{args.policy_id}' "
        f"({sum(1 for q in relevant if q.is_abstention)} abstentions)"
    )
    if not relevant:
        print(f"No questions found for policy_id='{args.policy_id}'. Check qa_set.jsonl.")
        return

    # Ingest PDF
    pdf_path = Path(args.pdf) if args.pdf else None
    if pdf_path is None:
        import csv

        with Path("data/policies.csv").open(newline="", encoding="utf-8-sig") as stream:
            row = next(
                (r for r in csv.DictReader(stream) if r["policy_id"] == args.policy_id), None
            )
        pdf_path = (
            Path(row["local_path"]) if row and row["local_path"] else Path("data/sample_policy.pdf")
        )
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    print(f"\nIngesting {pdf_path}...")
    vs, bm25, emb = ingest_for_eval(
        pdf_path=pdf_path,
        policy_id=args.policy_id,
        use_mock_embeddings=args.mock,
    )

    # Build pipeline
    llm = MockLLMClient() if args.mock else get_llm_client()
    pipeline = RAGPipeline(
        vector_store=vs,
        bm25_index=bm25,
        embedding_model=emb,
        llm=llm,
    )

    mode_label = "MOCK" if args.mock else "LIVE"
    print(f"\nRunning evaluation [{mode_label} mode]...\n")

    report = run_evaluation(
        qa_pairs=relevant,
        pipeline=pipeline,
        policy_id=args.policy_id,
    )
    report["mode"] = mode_label

    print_summary(report)

    # Save results
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else results_dir / f"{args.policy_id}_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
