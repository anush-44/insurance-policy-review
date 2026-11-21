"""Collect real Gemini answers for the verified policy QA set.

The request sends each question and up to five retrieved policy chunks to the
configured Gemini endpoint. The output is checkpointed after every question.
It records predictions and API usage, but does not claim answer correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.api import get_pipeline
from app.core.config import settings
from app.core.llm import GeminiLLMClient
from eval.validate_eval_data import validate_qa_references, validate_qa_set


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run(out: Path, limit: int | None = None) -> dict:
    if settings.llm_provider != "gemini" or not settings.gemini_api_key:
        raise ValueError("Live evaluation requires LLM_PROVIDER=gemini and GEMINI_API_KEY")
    qa_path = Path("eval/qa_set.jsonl")
    validation = validate_qa_references(qa_path)
    if not validation["valid"]:
        raise ValueError("Invalid QA references: " + "; ".join(validation["errors"]))
    with Path("data/policies.csv").open(newline="", encoding="utf-8-sig") as stream:
        official = {
            row["policy_id"]
            for row in csv.DictReader(stream)
            if row["validation_status"] == "valid"
        }
    pairs = [pair for pair in validate_qa_set(qa_path) if pair.policy_id in official]
    if limit is not None:
        pairs = pairs[:limit]
    dataset_hash = hashlib.sha256(qa_path.read_bytes()).hexdigest()
    if out.exists():
        report = json.loads(out.read_text(encoding="utf-8"))
        if report["dataset_sha256"] != dataset_hash or report["model"] != settings.llm_model:
            raise ValueError("Existing report uses a different dataset or model")
    else:
        report = {
            "started_at": datetime.now(UTC).isoformat(),
            "dataset": str(qa_path),
            "dataset_sha256": dataset_hash,
            "model": settings.llm_model,
            "max_retrieved_chunks": settings.top_k_rerank,
            "evaluation_status": "live_predictions_unscored_for_correctness",
            "records": [],
        }
    completed = {
        record["id"] for record in report["records"] if record["status"] != "llm_error_fallback"
    }
    pipeline = get_pipeline()
    llm = GeminiLLMClient(timeout_seconds=90)
    pipeline.llm = llm
    failure_streak = 0
    try:
        for index, pair in enumerate(pairs, start=1):
            if pair.id in completed:
                continue
            for attempt in range(2):
                llm.last_usage_metadata = {}
                llm.last_model_version = None
                started = time.perf_counter()
                answer = pipeline.answer(pair.question, policy_id=pair.policy_id)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                if pipeline.last_answer_status != "llm_error_fallback" or attempt == 1:
                    break
                time.sleep(10)
            status = pipeline.last_answer_status
            failure_streak = failure_streak + 1 if status == "llm_error_fallback" else 0
            report["records"] = [record for record in report["records"] if record["id"] != pair.id]
            report["records"].append(
                {
                    "id": pair.id,
                    "policy_id": pair.policy_id,
                    "question": pair.question,
                    "gold_answer": pair.gold_answer,
                    "gold_clause_ids": pair.gold_clause_ids,
                    "gold_pages": pair.gold_pages,
                    "answerability": pair.answerability,
                    "category": pair.category,
                    "status": status,
                    "answer": answer.model_dump(mode="json"),
                    "latency_ms": elapsed_ms,
                    "api_attempts": attempt + 1,
                    "model_version": llm.last_model_version,
                    "usage_metadata": llm.last_usage_metadata,
                }
            )
            report["updated_at"] = datetime.now(UTC).isoformat()
            _write(out, report)
            print(f"{index}/{len(pairs)} {pair.id}: {status} ({elapsed_ms:.0f} ms)", flush=True)
            if failure_streak >= 3:
                print("Stopping after three consecutive Gemini failures", flush=True)
                break
    finally:
        pipeline.retriever.vector_store.client.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run(args.out, args.limit)
    print(f"Recorded {len(report['records'])} predictions in {args.out}")


if __name__ == "__main__":
    main()
