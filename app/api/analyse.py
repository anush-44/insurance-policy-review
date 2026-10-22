"""FastAPI router: /api/v1/analyse — summary, red flags, and comparison."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api import get_pipeline
from app.cli import list_policies
from app.rag.pipeline import RAGPipeline
from app.rag.schemas import PolicyComparisonResult, PolicySummaryCard, RedFlagsReport

router = APIRouter(prefix="/api/v1/analyse", tags=["analyse"])

PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


def _require_policy(policy_id: str) -> None:
    """Raise 404 if the policy_id is not in the registry."""
    registry = {p["policy_id"]: p for p in list_policies()}
    if policy_id not in registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy '{policy_id}' is not indexed. Please ingest it first.",
        )


@router.get(
    "/summary/{policy_id}",
    response_model=PolicySummaryCard,
    summary="Auto-extract policy summary card",
)
async def get_summary(policy_id: str, pipeline: PipelineDep) -> PolicySummaryCard:
    """Extract a structured summary card for an indexed policy.

    Covers waiting periods, room rent cap, co-pay, restoration, NCB,
    sub-limits, key exclusions, and claim process.
    """
    _require_policy(policy_id)
    registry = {p["policy_id"]: p for p in list_policies()}
    meta = registry[policy_id]
    try:
        return pipeline.summarise(
            policy_id=policy_id,
            policy_name=meta.get("policy_name", ""),
            insurer=meta.get("insurer", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary extraction failed: {e}") from e


@router.get(
    "/red-flags/{policy_id}",
    response_model=RedFlagsReport,
    summary="Identify buyer-unfriendly clauses",
)
async def get_red_flags(policy_id: str, pipeline: PipelineDep) -> RedFlagsReport:
    """Scan the policy for restrictive, hidden, or buyer-unfriendly clauses.

    Each red flag includes the clause ID, page, a plain-language description,
    and a severity rating (high / medium / low).
    """
    _require_policy(policy_id)
    try:
        return pipeline.detect_red_flags(policy_id=policy_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Red flag detection failed: {e}") from e


class CompareRequest(BaseModel):
    """Request body for policy comparison."""

    scenario: str = Field(
        ...,
        min_length=5,
        description="Clinical or financial scenario, e.g. 'knee replacement surgery'",
    )
    policy_id_a: str = Field(..., description="First policy's ID")
    policy_id_b: str = Field(..., description="Second policy's ID")


@router.post(
    "/compare",
    response_model=PolicyComparisonResult,
    summary="Side-by-side policy comparison for a scenario",
)
async def compare_policies(
    request: CompareRequest,
    pipeline: PipelineDep,
) -> PolicyComparisonResult:
    """Compare two indexed policies for a specific scenario.

    Returns comparison points (coverage, waiting period, sublimits, exclusions)
    with clause citations from both policies and an overall verdict.
    """
    _require_policy(request.policy_id_a)
    _require_policy(request.policy_id_b)

    registry = {p["policy_id"]: p for p in list_policies()}
    meta_a = registry[request.policy_id_a]
    meta_b = registry[request.policy_id_b]

    try:
        return pipeline.compare(
            scenario=request.scenario,
            policy_id_a=request.policy_id_a,
            policy_name_a=meta_a.get("policy_name", request.policy_id_a),
            policy_id_b=request.policy_id_b,
            policy_name_b=meta_b.get("policy_name", request.policy_id_b),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}") from e
