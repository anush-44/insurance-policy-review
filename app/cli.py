"""PolicyLens ingestion CLI.

Usage:
    python -m app.cli ingest path/to/policy.pdf --policy-id hdfc_optima --insurer "HDFC ERGO"
    python -m app.cli ingest path/to/policy.pdf  # auto-derives policy_id from filename
    python -m app.cli list-policies
    python -m app.cli delete-policy hdfc_optima
"""

import argparse
import json
import os
import re
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
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", derived_id):
        raise ValueError("policy_id may contain only letters, numbers, underscores, and hyphens")
    logger.info(f"Starting ingestion: path={pdf_path}, policy_id={derived_id}")

    # 1. Parse PDF → PolicyDocument
    parser = PyMuPDFParser()
    doc = parser.parse(
        file_source=pdf_path,
        policy_id=derived_id,
        policy_name=pdf_path.stem,
        insurer=insurer,
    )
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

    # 4. Persist all three stores as one rollback-capable ingestion operation.
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client)
    bm25 = BM25Index()
    bm25.index_chunks(chunks)
    bm25_path = Path(settings.qdrant_local_path) / f"{derived_id}_bm25.json"
    registry_path = Path(settings.qdrant_local_path) / "policy_registry.json"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    old_bm25 = bm25_path.read_bytes() if bm25_path.exists() else None
    old_registry = registry_path.read_bytes() if registry_path.exists() else None
    registry: dict = {}
    if old_registry is not None:
        try:
            registry = json.loads(old_registry)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Existing policy registry is invalid; refusing to overwrite it: {registry_path}") from exc
    old_points = vector_store.snapshot_policy_points(derived_id)

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
    stale_count = 0
    try:
        vector_store.upsert_chunks(chunks=chunks, vectors=vectors)
        stale_count = vector_store.prune_policy_chunks(derived_id, {chunk.chunk_id for chunk in chunks})
        bm25_tmp = bm25_path.with_suffix(".json.tmp")
        bm25_tmp.write_text(json.dumps(bm25_data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(bm25_tmp, bm25_path)

        registry[derived_id] = {
            "policy_id": derived_id,
            "policy_name": doc.policy_name,
            "insurer": doc.insurer,
            "total_pages": doc.total_pages,
            "total_chunks": len(chunks),
            "uin": doc.uin,
        }
        registry_tmp = registry_path.with_suffix(".json.tmp")
        registry_tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(registry_tmp, registry_path)
    except Exception as exc:
        rollback_errors: list[str] = []
        try:
            vector_store.restore_policy_points(derived_id, old_points)
        except Exception as rollback_exc:
            rollback_errors.append(f"Qdrant rollback failed: {rollback_exc}")
        for path, previous in ((bm25_path, old_bm25), (registry_path, old_registry)):
            try:
                if previous is None:
                    path.unlink(missing_ok=True)
                else:
                    rollback_tmp = path.with_suffix(path.suffix + ".rollback")
                    rollback_tmp.write_bytes(previous)
                    os.replace(rollback_tmp, path)
            except Exception as rollback_exc:
                rollback_errors.append(f"{path.name} rollback failed: {rollback_exc}")
        details = "; ".join(rollback_errors) if rollback_errors else "all previous state restored"
        raise RuntimeError(f"Ingestion persistence failed ({exc}); rollback result: {details}") from exc
    logger.info(f"BM25 index and registry committed for {derived_id}; stale vectors removed={stale_count}")

    clause_counts: dict[str, int] = {}
    for section in doc.sections:
        for clause in section.clauses:
            clause_counts[str(clause.page_start)] = clause_counts.get(str(clause.page_start), 0) + 1
    warnings = []
    if len(doc.all_clauses()) < max(5, doc.total_pages // 5):
        warnings.append("Suspiciously low detected clause count; inspect parser output before trusting retrieval.")
    report = {
        "policy_id": derived_id,
        "policy_name": doc.policy_name,
        "insurer": doc.insurer,
        "uin": doc.uin,
        "pages": doc.total_pages,
        "selectable_text_chars": doc.metadata["selectable_text_chars"],
        "detected_clauses": len(doc.all_clauses()),
        "chunks": len(chunks),
        "errors": [],
        "stale_vectors_removed": stale_count,
        "clauses_by_start_page": clause_counts,
        "warnings": warnings,
        "status": "indexed_with_warnings" if warnings else "indexed",
    }
    report_dir = Path("data/ingestion_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{derived_id}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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


def delete_policy(policy_id: str, vector_store: QdrantVectorStore | None = None) -> bool:
    """Delete a policy's Qdrant points and local indexes before removing its registry entry."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", policy_id):
        raise ValueError("policy_id may contain only letters, numbers, underscores, and hyphens")
    registry_path = Path(settings.qdrant_local_path) / "policy_registry.json"
    if not registry_path.exists():
        return False
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if policy_id not in registry:
        return False
    active_store = vector_store or QdrantVectorStore(client=get_qdrant_client())
    bm25_path = Path(settings.qdrant_local_path) / f"{policy_id}_bm25.json"
    old_registry = registry_path.read_bytes()
    old_bm25 = bm25_path.read_bytes() if bm25_path.exists() else None
    old_points = active_store.snapshot_policy_points(policy_id)
    del registry[policy_id]
    registry_tmp = registry_path.with_suffix(".json.tmp")
    registry_tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        active_store.delete_policy_chunks(policy_id)
        bm25_path.unlink(missing_ok=True)
        os.replace(registry_tmp, registry_path)
    except Exception as exc:
        rollback_errors: list[str] = []
        try:
            active_store.restore_policy_points(policy_id, old_points)
        except Exception as rollback_exc:
            rollback_errors.append(f"Qdrant rollback failed: {rollback_exc}")
        try:
            if old_bm25 is None:
                bm25_path.unlink(missing_ok=True)
            else:
                bm25_tmp = bm25_path.with_suffix(".json.rollback")
                bm25_tmp.write_bytes(old_bm25)
                os.replace(bm25_tmp, bm25_path)
            registry_path.write_bytes(old_registry)
        except Exception as rollback_exc:
            rollback_errors.append(f"Metadata rollback failed: {rollback_exc}")
        registry_tmp.unlink(missing_ok=True)
        result = "; ".join(rollback_errors) if rollback_errors else "previous state restored"
        raise RuntimeError(f"Policy deletion failed ({exc}); {result}") from exc
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
            print(f"\nIngested: {resp.policy_name} ({resp.policy_id})")
            print(f"   Pages: {resp.total_pages} | Chunks: {resp.total_chunks}")
        except Exception as e:
            print(f"\nIngestion failed: {e}", file=sys.stderr)
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
            print(f"Deleted '{args.policy_id}' from registry.")
        else:
            print(f"Policy '{args.policy_id}' not found.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
