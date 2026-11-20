"""FastAPI router: /api/v1/qa — grounded question answering."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api import get_pipeline
from app.core.config import settings
from app.core.schemas import GroundedAnswer
from app.rag.pipeline import RAGPipeline
from app.rag.schemas import ErrorResponse

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])

PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


class QARequest(BaseModel):
    """Request body for the grounded Q&A endpoint."""

    question: str = Field(..., min_length=5, description="User's plain-language question")
    policy_id: str | None = Field(default=None, description="Restrict to a specific indexed policy")
    top_k: int = Field(default=5, ge=1, le=20, description="Chunks to retrieve")


@router.post(
    "",
    response_model=GroundedAnswer,
    summary="Answer a question grounded in policy clauses",
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def answer_question(
    request: QARequest,
    pipeline: PipelineDep,
) -> GroundedAnswer:
    """Answer a user question with clause + page citations from the indexed policy.

    Returns a GroundedAnswer with:
    - **answer**: plain-language explanation.
    - **citations**: list of {clause_id, page, verbatim quote}.
    - **confidence**: 0.0–1.0.
    - **is_abstention**: true if the question cannot be answered from the policy.
    - **disclaimer**: always appended.
    """
    try:
        result = pipeline.answer(
            query=request.question,
            policy_id=request.policy_id,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG pipeline error: {e}",
        ) from e

    if result.caveats is None:
        result.caveats = []
    result.caveats.append(settings.disclaimer)

    return result
