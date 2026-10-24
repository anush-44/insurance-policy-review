"""Optional RAGAS evaluation using the current indexed PolicyLens corpus.

Requires the ``eval`` extra, a real evaluator LLM key (OpenAI API currently
required by RAGAS' default evaluator), and a real PolicyLens provider for answers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.api import get_pipeline
from app.core.config import settings
from app.core.llm import MockLLMClient
from eval.validate_eval_data import validate_qa_references, validate_qa_set


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-mock-app", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise SystemExit("RAGAS evaluation is unavailable. Install the project eval extra: pip install -e '.[eval]'.") from exc

    openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise SystemExit("RAGAS' configured judge needs OPENAI_API_KEY. No score was produced.")
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_key
    validation = validate_qa_references()
    if not validation["valid"]:
        raise SystemExit("Gold-label validation failed: " + "; ".join(validation["errors"]))
    pairs = [pair for pair in validate_qa_set() if pair.policy_id == args.policy_id]
    if not pairs:
        raise SystemExit(f"No QA labels for policy {args.policy_id}")

    pipeline = get_pipeline()
    if isinstance(pipeline.llm, MockLLMClient) and not args.allow_mock_app:
        raise SystemExit("PolicyLens is using MockLLMClient (likely missing GEMINI_API_KEY); pass --allow-mock-app only for diagnostic runs.")
    samples: list[dict[str, Any]] = []
    for pair in pairs:
        answer = pipeline.answer(pair.question, policy_id=pair.policy_id)
        contexts = [chunk.text for chunk, _ in pipeline.retriever.retrieve(pair.question, limit=5, policy_id=pair.policy_id)]
        samples.append({
            "question": pair.question,
            "answer": answer.answer,
            "contexts": contexts,
            "ground_truth": pair.gold_answer,
        })

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(Dataset.from_list(samples), metrics=metrics)
    try:
        metric_values = result.to_pydict()
    except AttributeError:
        metric_values = result.to_pandas().to_dict(orient="records")
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "policy_id": args.policy_id,
        "dataset": "eval/qa_set.jsonl",
        "dataset_sha256": sha256_file(Path("eval/qa_set.jsonl")),
        "sample_size": len(samples),
        "ragas_version": importlib.metadata.version("ragas"),
        "metrics": [metric.name for metric in metrics],
        "evaluator_provider": "RAGAS default evaluator (OpenAI)",
        "application_llm_provider": type(pipeline.llm).__name__,
        "application_embedding_model": settings.embedding_model,
        "results": metric_values,
        "cost_note": "RAGAS evaluator calls may incur external API cost; this report records no cost estimate.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved RAGAS report for {len(samples)} records to {args.out}")


if __name__ == "__main__":
    main()
