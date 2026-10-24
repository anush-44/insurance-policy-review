"""Extended response schemas for PolicyLens API endpoints."""

from pydantic import BaseModel, Field

# ── Policy Summary Card ──────────────────────────────────────────────────────


class PolicySummaryCard(BaseModel):
    """Structured auto-extracted summary of an insurance policy."""

    policy_name: str = ""
    insurer: str = ""
    waiting_period_ped: str = "Not specified"
    waiting_period_specific: str = "Not specified"
    room_rent_cap: str = "Not specified"
    copay_percent: str = "Not established from retrieved clauses"
    restoration_benefit: str = "Not specified"
    no_claim_bonus: str = "Not specified"
    key_sublimits: list[str] = Field(default_factory=list)
    key_exclusions: list[str] = Field(default_factory=list)
    claim_process_summary: str = ""


# ── Red Flags ────────────────────────────────────────────────────────────────


class RedFlag(BaseModel):
    """A single restrictive or buyer-unfriendly policy clause."""

    flag_type: str
    clause_id: str
    page: int = 0
    description: str
    evidence_quote: str = ""
    severity: str = Field(
        default="medium",
        description="'high' | 'medium' | 'low'",
    )


class RedFlagsReport(BaseModel):
    """Full red flags analysis for a policy."""

    red_flags: list[RedFlag] = Field(default_factory=list)


# ── Policy Comparison ────────────────────────────────────────────────────────


class ComparisonPoint(BaseModel):
    """Single aspect comparison between two policies."""

    aspect: str
    policy_a_summary: str
    policy_a_clause_id: str = ""
    policy_a_page: int = 0
    policy_b_summary: str
    policy_b_clause_id: str = ""
    policy_b_page: int = 0
    policy_a_evidence_quote: str = ""
    policy_b_evidence_quote: str = ""
    verdict: str = "Insufficient data"


class PolicyComparisonResult(BaseModel):
    """Side-by-side comparison of two policies for a scenario."""

    scenario: str
    policy_a_name: str
    policy_b_name: str
    comparison_points: list[ComparisonPoint] = Field(default_factory=list)
    overall_verdict: str = ""


# ── Ingestion Status ─────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    """Response for PDF ingestion endpoint."""

    policy_id: str
    policy_name: str
    insurer: str
    total_chunks: int
    total_pages: int
    status: str = "indexed"


# ── API Error ────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard API error payload."""

    detail: str
    code: str = "INTERNAL_ERROR"
