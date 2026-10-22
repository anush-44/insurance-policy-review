import json
import time
from pathlib import Path
from typing import Any

from app.core.logging import logger
from app.core.schemas import PolicyDocument
from app.ingestion.pdf_parser import PyMuPDFParser
from app.ingestion.unstructured_parser import UnstructuredParser


def compute_table_preservation_score(doc: PolicyDocument) -> dict[str, Any]:
    """Evaluate table preservation quality for an extracted document.

    Checks:
    - total tables detected
    - markdown structure validity (pipes, separator rows)
    - cell fill rate
    """
    all_tables = []
    for section in doc.sections:
        for clause in section.clauses:
            all_tables.extend(clause.tables)

    if not all_tables:
        return {
            "tables_detected": 0,
            "markdown_validity": 0.0,
            "cell_fill_rate": 0.0,
            "preservation_score": 0.0,
        }

    valid_md_count = 0
    total_cells = 0
    filled_cells = 0

    for table in all_tables:
        # Check markdown table syntax (at least header, divider, and one row)
        lines = [line.strip() for line in table.markdown.split("\n") if line.strip()]
        if len(lines) >= 3 and lines[0].startswith("|") and "---" in lines[1]:
            valid_md_count += 1

        # Check cell contents
        for row in table.rows:
            for cell in row:
                total_cells += 1
                if str(cell).strip():
                    filled_cells += 1

    md_validity = valid_md_count / len(all_tables) if all_tables else 0.0
    fill_rate = filled_cells / total_cells if total_cells else 0.0
    # Combined score weighted: 50% validity, 50% fill rate
    preservation_score = round(0.5 * md_validity + 0.5 * fill_rate, 3)

    return {
        "tables_detected": len(all_tables),
        "markdown_validity": round(md_validity, 3),
        "cell_fill_rate": round(fill_rate, 3),
        "preservation_score": preservation_score,
    }


def compare_parsers(
    pdf_path: Path | str,
    output_dir: Path | str = "eval/results",
) -> dict[str, Any]:
    """Run parser comparison between PyMuPDF and Unstructured on a given policy PDF."""
    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "file": pdf_path.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parsers": {},
    }

    # 1. Benchmark PyMuPDF
    pymupdf_parser = PyMuPDFParser()
    t0 = time.perf_counter()
    pymupdf_doc = pymupdf_parser.parse(
        file_source=pdf_path,
        policy_id="eval-doc",
        policy_name=pdf_path.stem,
        insurer="Sample Insurer",
    )
    t_pymupdf = time.perf_counter() - t0

    pymupdf_metrics = compute_table_preservation_score(pymupdf_doc)
    results["parsers"]["PyMuPDF"] = {
        "latency_seconds": round(t_pymupdf, 3),
        "latency_per_page_sec": round(t_pymupdf / max(pymupdf_doc.total_pages, 1), 4),
        "total_pages": pymupdf_doc.total_pages,
        "total_clauses": len(pymupdf_doc.all_clauses()),
        **pymupdf_metrics,
    }

    # 2. Benchmark Unstructured
    unstructured_parser = UnstructuredParser()
    t0 = time.perf_counter()
    unstruct_doc = unstructured_parser.parse(
        file_path=pdf_path,
        policy_id="eval-doc",
        policy_name=pdf_path.stem,
        insurer="Sample Insurer",
    )
    t_unstruct = time.perf_counter() - t0

    unstruct_metrics = compute_table_preservation_score(unstruct_doc)
    results["parsers"]["Unstructured"] = {
        "available": unstructured_parser.is_available,
        "latency_seconds": round(t_unstruct, 3),
        "latency_per_page_sec": round(t_unstruct / max(unstruct_doc.total_pages, 1), 4),
        "total_pages": unstruct_doc.total_pages,
        "total_clauses": len(unstruct_doc.all_clauses()),
        **unstruct_metrics,
    }

    # Save results as JSON
    json_path = out_dir / "parser_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Parser comparison saved to {json_path}")
    return results
