from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChunkType(StrEnum):
    """Categorization of policy text chunks for filtering and weighting."""

    DEFINITION = "definition"
    COVERAGE = "coverage"
    EXCLUSION = "exclusion"
    WAITING_PERIOD = "waiting_period"
    SUB_LIMIT = "sub_limit"
    TABLE = "table"
    CLAIMS = "claims"
    OTHER = "other"


class TableBlock(BaseModel):
    """Structured table extracted from a policy PDF."""

    page: int
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    markdown: str = ""
    title: str | None = None


class Clause(BaseModel):
    """A distinct numbered or titled clause within a policy section."""

    clause_id: str = Field(description="e.g. '4.1', 'Section 3.A', or 'Clause-12'")
    title: str = Field(default="", description="Header or clause title")
    text: str = Field(description="Body content of the clause")
    page_start: int
    page_end: int
    chunk_type: ChunkType = ChunkType.OTHER
    parent_section: str | None = None
    tables: list[TableBlock] = Field(default_factory=list)


class Section(BaseModel):
    """High-level policy section (e.g., 'Section 4: Exclusions')."""

    section_id: str
    title: str
    page_start: int
    page_end: int
    clauses: list[Clause] = Field(default_factory=list)


class PolicyDocument(BaseModel):
    """Fully parsed structured representation of an insurance policy PDF."""

    policy_id: str
    policy_name: str
    insurer: str
    uin: str | None = Field(default=None, description="IRDAI Unique Identification Number")
    total_pages: int
    sections: list[Section] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def all_clauses(self) -> list[Clause]:
        """Convenience method returning flattened list of all clauses."""
        return [clause for section in self.sections for clause in section.clauses]


class PolicyChunk(BaseModel):
    """Retrievable unit indexed into Qdrant & BM25 with rich metadata."""

    chunk_id: str
    policy_id: str
    insurer: str
    section: str
    clause_id: str
    clause_title: str
    text: str
    page_start: int
    page_end: int
    chunk_type: ChunkType = ChunkType.OTHER
    cross_references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Exact source attribution for grounded statements."""

    clause_id: str
    page: int
    quote: str = Field(description="Verbatim or near-verbatim quote extracted from the context")


class GroundedAnswer(BaseModel):
    """Standardized output schema for question answering."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    is_abstention: bool = False
    caveats: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
