import argparse
import sys
from pathlib import Path

from app.core.logging import logger
from app.ingestion.parser_comparator import compare_parsers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parser comparison on policy PDF.")
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
        help="Directory to save comparison metrics JSON.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"Target PDF '{pdf_path}' does not exist.")
        sys.exit(1)

    logger.info(f"Starting parser comparison on {pdf_path}")
    results = compare_parsers(pdf_path, output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("PARSER COMPARISON RESULTS")
    print("=" * 60)
    for parser_name, metrics in results.get("parsers", {}).items():
        print(f"\n[{parser_name}]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
