"""PolicyLens ingestion CLI.

Usage:
    python -m app.cli ingest path/to/policy.pdf --policy-id hdfc_optima --insurer "HDFC ERGO"
    python -m app.cli ingest path/to/policy.pdf  # auto-derives policy_id from filename
    python -m app.cli list-policies
    python -m app.cli delete-policy hdfc_optima
"""

import argparse
import json
import sys
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.indexing.bm25_index import BM25Index
from app.indexing.chunker import ClauseAwareChunker
from app.indexing.embeddings import BGEEmbeddingModel
from app.indexing.qdrant_client import QdrantVectorStore, get_qdrant_client
from app.ingestion.pdf_parser import PyMuPDFParser
from app.rag.schemas import IngestResponse


def ingest_pdf(
    pdf_path: Path,
    policy_id: str | None = None,
    insurer: str = "Unknown Insurer",
    recreate: bool = False,
) -> IngestResponse:
    """Full ingestion pipeline: Parse → Chunk → Embed → Index (Qdrant + BM25).

    Args:
        pdf_path: Absolute or relative path to the policy PDF.
        policy_id: Unique identifier (slug). Defaults to PDF stem.
        insurer: Insurer name for metadata.
        recreate: If True, delete and recreate the Qdrant collection.

    Returns:
        IngestResponse summarising the indexed policy.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    derived_id = policy_id or pdf_path.stem.lower().replace(" ", "_")
    logger.info(f"Starting ingestion: path={pdf_path}, policy_id={derived_id}")

    # 1. Parse PDF → PolicyDocument
    parser = PyMuPDFParser()
    doc = parser.parse(pdf_path=pdf_path, policy_id=derived_id, insurer=insurer)
    logger.info(
        f"Parsed '{doc.policy_name}' — {doc.total_pages} pages, "
        f"{len(doc.all_clauses())} clauses"
    )

    # 2. Chunk → PolicyChunks
    chunker = ClauseAwareChunker()
    chunks = chunker.chunk(doc)
    logger.info(f"Chunked into {len(chunks)} retrievable units")

    if not chunks:
        raise ValueError(f"No chunks produced from {pdf_path}. Check if PDF has extractable text.")

    # 3. Embed chunks
    embedding_model = BGEEmbeddingModel()
    texts = [c.text for c in chunks]
    vectors = embedding_model.embed_documents(texts)
    logger.info(f"Embedded {len(vectors)} chunks (dim={embedding_model.dimension})")

    # 4. Index into Qdrant
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client)
    vector_store.upsert_chunks(chunks=chunks, vectors=vectors)

    # 5. Persist BM25 index alongside Qdrant storage
    bm25 = BM25Index()
    bm25.index_chunks(chunks)
    bm25_path = Path(settings.qdrant_local_path) / f"{derived_id}_bm25.json"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize chunk texts + ids for BM25 re-hydration on startup
    bm25_data = [
        {
            "chunk_id": c.chunk_id,
            "policy_id": c.policy_id,
            "insurer": c.insurer,
            "section": c.section,
            "clause_id": c.clause_id,
            "clause_title": c.clause_title,
            "text": c.text,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "chunk_type": c.chunk_type.value,
            "cross_references": c.cross_references,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    bm25_path.write_text(json.dumps(bm25_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"BM25 index serialised → {bm25_path}")

    # 6. Persist policy metadata registry
    registry_path = Path(settings.qdrant_local_path) / "policy_registry.json"
    registry: dict = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = {}
    registry[derived_id] = {
        "policy_id": derived_id,
        "policy_name": doc.policy_name,
        "insurer": doc.insurer,
        "total_pages": doc.total_pages,
        "total_chunks": len(chunks),
        "uin": doc.uin,
    }
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Registry updated → {registry_path}")

    return IngestResponse(
        policy_id=derived_id,
        policy_name=doc.policy_name,
        insurer=doc.insurer,
        total_chunks=len(chunks),
        total_pages=doc.total_pages,
        status="indexed",
    )


def list_policies() -> list[dict]:
    """Return all indexed policies from the registry."""
    registry_path = Path(settings.qdrant_local_path) / "policy_registry.json"
    if not registry_path.exists():
        return []
    try:
        return list(json.loads(registry_path.read_text(encoding="utf-8")).values())
    except Exception:
        return []


def delete_policy(policy_id: str) -> bool:
    """Remove a policy entry from the registry (does not delete Qdrant points)."""
    registry_path = Path(settings.qdrant_local_path) / "policy_registry.json"
    if not registry_path.exists():
        return False
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if policy_id not in registry:
        return False
    del registry[policy_id]
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PolicyLens ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ingest sub-command
    ingest_p = sub.add_parser("ingest", help="Ingest a policy PDF")
    ingest_p.add_argument("pdf_path", type=Path, help="Path to the policy PDF file")
    ingest_p.add_argument("--policy-id", default=None, help="Policy slug (default: PDF filename)")
    ingest_p.add_argument("--insurer", default="Unknown Insurer", help="Insurer name")
    ingest_p.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collection from scratch",
    )

    # list-policies sub-command
    sub.add_parser("list-policies", help="List all indexed policies")

    # delete-policy sub-command
    del_p = sub.add_parser("delete-policy", help="Delete a policy from the registry")
    del_p.add_argument("policy_id", help="Policy ID to delete")

    args = parser.parse_args()

    if args.command == "ingest":
        try:
            resp = ingest_pdf(
                pdf_path=args.pdf_path,
                policy_id=args.policy_id,
                insurer=args.insurer,
                recreate=args.recreate,
            )
            print(f"\n✅ Ingested: {resp.policy_name} ({resp.policy_id})")
            print(f"   Pages: {resp.total_pages} | Chunks: {resp.total_chunks}")
        except Exception as e:
            print(f"\n❌ Ingestion failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "list-policies":
        policies = list_policies()
        if not policies:
            print("No policies indexed yet.")
        else:
            print(f"\n{'ID':<25} {'Insurer':<25} {'Name':<40} Chunks")
            print("-" * 100)
            for p in policies:
                print(
                    f"{p['policy_id']:<25} {p['insurer']:<25} "
                    f"{p['policy_name'][:38]:<40} {p['total_chunks']}"
                )

    elif args.command == "delete-policy":
        if delete_policy(args.policy_id):
            print(f"✅ Deleted '{args.policy_id}' from registry.")
        else:
            print(f"❌ Policy '{args.policy_id}' not found.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
