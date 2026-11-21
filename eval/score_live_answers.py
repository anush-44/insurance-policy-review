"""Score recorded Gemini answers against reviewed labels and source PDFs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf


def _normalized(text: str) -> str:
    return "".join(char.lower() for char in unicodedata.normalize("NFKC", text) if char.isalnum())


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(center - half_width, 4), round(center + half_width, 4)]


def score(predictions_path: Path, review_path: Path, manifest_path: Path) -> dict:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    records = predictions["records"]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(review["labels"]):
        raise ValueError("Prediction IDs must be unique and exactly match review labels")
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        manifest = {row["policy_id"]: row for row in csv.DictReader(stream)}
    page_texts = {}
    citation_checks = []
    for record in records:
        policy_id = record["policy_id"]
        if policy_id not in page_texts:
            with pymupdf.open(manifest[policy_id]["local_path"]) as doc:
                page_texts[policy_id] = [_normalized(page.get_text("text")) for page in doc]
        pages = page_texts[policy_id]
        for citation in record["answer"]["citations"]:
            page = citation["page"]
            valid_page = 1 <= page <= len(pages)
            quote = _normalized(citation["quote"])
            citation_checks.append(
                {
                    "id": record["id"],
                    "page": page,
                    "valid_page": valid_page,
                    "exact_quote_on_cited_page": bool(
                        valid_page and quote and quote in pages[page - 1]
                    ),
                    "page_in_gold_references": page in record["gold_pages"],
                }
            )
    answerable = [r for r in records if r["answerability"] == "answerable"]
    unanswerable = [r for r in records if r["answerability"] == "unanswerable"]
    labels = review["labels"]
    correct = sum(labels[r["id"]] == "correct" for r in answerable)
    correct_refusals = sum(labels[r["id"]] == "correct_refusal" for r in unanswerable)
    false_refusals = sum(r["answer"]["is_abstention"] for r in answerable)
    cited_gold_page = sum(
        any(c["page"] in record["gold_pages"] for c in record["answer"]["citations"])
        for record in answerable
    )
    exact_quotes = sum(c["exact_quote_on_cited_page"] for c in citation_checks)
    latencies = sorted(r["latency_ms"] for r in records)
    return {
        "predictions": str(predictions_path),
        "review": str(review_path),
        "reviewer": review["reviewer"],
        "model": predictions["model"],
        "sample_size": len(records),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "generation_statuses": dict(Counter(r["status"] for r in records)),
        "review_labels": dict(Counter(labels.values())),
        "assistant_reviewed_answer_correctness": {
            "correct": correct,
            "total": len(answerable),
            "rate": correct / len(answerable),
            "wilson_95_percent_ci": _wilson(correct, len(answerable)),
        },
        "refusal_accuracy": {
            "correct": correct_refusals,
            "total": len(unanswerable),
            "rate": correct_refusals / len(unanswerable),
            "wilson_95_percent_ci": _wilson(correct_refusals, len(unanswerable)),
        },
        "false_refusal_rate": {
            "count": false_refusals,
            "total": len(answerable),
            "rate": false_refusals / len(answerable),
        },
        "citation_checks": {
            "citations": len(citation_checks),
            "valid_pdf_pages": sum(c["valid_page"] for c in citation_checks),
            "exact_quotes_on_cited_pdf_page": exact_quotes,
            "exact_quote_rate": exact_quotes / len(citation_checks) if citation_checks else None,
            "answerable_questions_with_gold_page_citation": cited_gold_page,
            "answerable_questions_total": len(answerable),
            "gold_page_match_is_not_source_accuracy": True,
            "details": citation_checks,
        },
        "latency_ms_final_attempt": {
            "p50": round(statistics.median(latencies), 2),
            "p95": round(latencies[int((len(latencies) - 1) * 0.95)], 2),
        },
        "api_attempts": sum(r["api_attempts"] for r in records),
        "usage_tokens": {
            key: sum(r["usage_metadata"].get(key, 0) for r in records)
            for key in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
        },
        "limitations": [
            "Answers were reviewed by the project assistant, not an independent human annotator.",
            "Exact quote matching is conservative; PDF extraction order and typography can differ from a near-verbatim quote.",
            "Gold pages are one reference set, so a different page may still be a valid source.",
            "No faithfulness or red-flag precision/recall judge was run.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/policies.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.predictions, args.review, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Scored {result['sample_size']} answers to {args.out}")


if __name__ == "__main__":
    main()
