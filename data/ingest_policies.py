"""Ingest manifest rows that have passed acquisition/PDF validation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.cli import ingest_pdf

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = {"valid", "valid_uin_not_detected"}


def ingest_manifest(manifest_path: Path, include_unverified_local: bool = False) -> dict[str, Any]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    results: list[dict[str, Any]] = []
    for row in rows:
        status = row.get("validation_status", "")
        if status not in ACCEPTED and not (include_unverified_local and status == "valid_pdf_source_unverified"):
            results.append({"policy_id": row.get("policy_id"), "status": "skipped_unvalidated", "validation_status": status})
            continue
        pdf_path = ROOT / row["local_path"]
        try:
            response = ingest_pdf(pdf_path=pdf_path, policy_id=row["policy_id"], insurer=row["insurer"])
            results.append({"policy_id": row["policy_id"], "status": response.status, "pages": response.total_pages, "chunks": response.total_chunks})
        except Exception as exc:
            failure = {
                "policy_id": row.get("policy_id"),
                "insurer": row.get("insurer"),
                "product_name": row.get("product_name"),
                "source_pdf": str(pdf_path),
                "pages": None,
                "selectable_text_chars": None,
                "detected_clauses": None,
                "chunks": None,
                "warnings": [],
                "errors": [f"{type(exc).__name__}: {exc}"],
                "status": "ingestion_failed",
            }
            report_dir = ROOT / "data/ingestion_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / f"{row.get('policy_id', 'unknown')}.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            results.append({"policy_id": row.get("policy_id"), "status": "ingestion_failed", "error": failure["errors"][0]})
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path),
        "include_unverified_local": include_unverified_local,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/policies.csv")
    parser.add_argument("--include-unverified-local", action="store_true")
    args = parser.parse_args()
    report = ingest_manifest(args.manifest, args.include_unverified_local)
    output = ROOT / "data/ingestion_reports" / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for item in report["results"]:
        print(f"{item['policy_id']}: {item['status']}")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
