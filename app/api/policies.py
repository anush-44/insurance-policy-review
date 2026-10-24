"""FastAPI router: /api/v1/policies — ingestion and policy management."""

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api import get_pipeline, reload_bm25
from app.cli import delete_policy, ingest_pdf, list_policies
from app.core.logging import logger
from app.rag.pipeline import RAGPipeline
from app.rag.schemas import ErrorResponse, IngestResponse

router = APIRouter(prefix="/api/v1/policies", tags=["policies"])

PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a policy PDF",
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def ingest_policy(
    pipeline: PipelineDep,
    file: UploadFile = File(..., description="Policy PDF file"),
    policy_id: str | None = Form(default=None, description="Optional policy slug"),
    insurer: str = Form(default="Unknown Insurer", description="Insurer name"),
) -> IngestResponse:
    """Upload a policy PDF, parse it, embed it, and index it in Qdrant + BM25.

    Returns the ingestion summary including total chunks indexed.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are supported.",
        )

    # Write upload to a temp file so PyMuPDF can open it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        derived_id = policy_id or Path(file.filename).stem.lower().replace(" ", "_")
        logger.info(f"API ingestion triggered: policy_id={derived_id}, insurer={insurer}")
        resp = ingest_pdf(
            pdf_path=tmp_path,
            policy_id=derived_id,
            insurer=insurer,
        )
        # Hot-reload BM25 so the running pipeline reflects the new data immediately
        reload_bm25(pipeline)
        return resp
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get(
    "",
    summary="List all indexed policies",
    response_model=list[dict],
)
async def list_indexed_policies() -> list[dict]:
    """Return all policies currently in the registry."""
    return list_policies()


@router.delete(
    "/{policy_id}",
    summary="Remove a policy from the registry",
    response_model=dict,
)
async def remove_policy(policy_id: str, pipeline: PipelineDep) -> dict:
    """Delete a policy entry from the registry.

    Removes Qdrant points, the saved BM25 index, and registry metadata.
    """
    try:
        success = delete_policy(policy_id, vector_store=pipeline.retriever.vector_store)
        if success:
            reload_bm25(pipeline)
    except Exception as exc:
        logger.error(f"Policy deletion failed for {policy_id}: {exc}")
        raise HTTPException(status_code=500, detail="Policy deletion failed; registry metadata was retained where possible.") from exc
    if not success:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found.")
    return {"deleted": policy_id}
