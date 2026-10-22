import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from app.core.logging import logger
from app.indexing.chunker import ClauseAwareChunker, FixedSizeChunker
from app.ingestion.pdf_parser import PyMuPDFParser


def evaluate_chunking(
    pdf_path: Path | str,
    output_dir: Path | str = "eval/results",
) -> dict[str, Any]:
    """Run ablation comparison between ClauseAwareChunker and FixedSizeChunker."""
    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = PyMuPDFParser()
    doc = parser.parse(
        file_source=pdf_path,
        policy_id="eval-policy",
        policy_name=pdf_path.stem,
        insurer="Sample Insurer",
    )

    # 1. Run Clause-Aware Chunker
    clause_chunker = ClauseAwareChunker(
        min_chunk_chars=120, max_chunk_chars=1200, overlap_chars=200
    )
    clause_chunks = clause_chunker.chunk(doc)

    # 2. Run Fixed-Size Baseline Chunker
    fixed_chunker = FixedSizeChunker(chunk_size=600, chunk_overlap=100)
    fixed_chunks = fixed_chunker.chunk(doc)

    def compute_stats(chunks) -> dict[str, Any]:
        lengths = [len(c.text) for c in chunks]
        refs_count = sum(len(c.cross_references) for c in chunks)
        has_clause_id = sum(
            1 for c in chunks if c.clause_id and not c.clause_id.startswith("FIXED")
        )
        has_section = sum(1 for c in chunks if c.section and c.section != "FixedWindow")

        return {
            "total_chunks": len(chunks),
            "char_length_min": min(lengths) if lengths else 0,
            "char_length_max": max(lengths) if lengths else 0,
            "char_length_mean": round(statistics.mean(lengths), 1) if lengths else 0,
            "char_length_std": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0,
            "cross_references_detected": refs_count,
            "clause_boundary_preservation_pct": round(100.0 * has_clause_id / len(chunks), 1)
            if chunks
            else 0.0,
            "section_context_preservation_pct": round(100.0 * has_section / len(chunks), 1)
            if chunks
            else 0.0,
        }

    results = {
        "file": pdf_path.name,
        "total_policy_pages": doc.total_pages,
        "clause_aware": compute_stats(clause_chunks),
        "fixed_size_baseline": compute_stats(fixed_chunks),
    }

    out_file = out_dir / "chunking_ablation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Chunking ablation results saved to {out_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunking ablation evaluation.")
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/sample_policy.pdf",
        help="Path to the PDF file to benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results",
        help="Directory to save ablation JSON.",
    )
    args = parser.parse_args()

    results = evaluate_chunking(args.pdf, output_dir=args.output_dir)

    print("\n" + "=" * 65)
    print("CHUNKING ABLATION REPORT")
    print("=" * 65)
    print(f"Policy: {results['file']} ({results['total_policy_pages']} pages)")
    print("-" * 65)
    print(f"{'Metric':<35} | {'Clause-Aware':<12} | {'Fixed-Size':<12}")
    print("-" * 65)
    ca = results["clause_aware"]
    fs = results["fixed_size_baseline"]
    metrics_to_show = [
        ("Total Chunks", "total_chunks"),
        ("Mean Chunk Length (chars)", "char_length_mean"),
        ("Min Chunk Length (chars)", "char_length_min"),
        ("Max Chunk Length (chars)", "char_length_max"),
        ("Cross-References Extracted", "cross_references_detected"),
        ("Clause Boundary Integrity %", "clause_boundary_preservation_pct"),
        ("Section Context Integrity %", "section_context_preservation_pct"),
    ]
    for label, key in metrics_to_show:
        print(f"{label:<35} | {str(ca[key]):<12} | {str(fs[key]):<12}")
    print("=" * 65)


if __name__ == "__main__":
    main()
