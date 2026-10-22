from pathlib import Path

import pytest

from app.core.schemas import ChunkType, Clause, PolicyDocument, Section, TableBlock
from data.create_sample_pdf import generate_sample_policy_pdf


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory) -> Path:
    """Generate a sample policy PDF fixture for the test session."""
    tmp_dir = tmp_path_factory.mktemp("policy_data")
    pdf_path = tmp_dir / "test_sample_policy.pdf"
    return generate_sample_policy_pdf(pdf_path)


@pytest.fixture
def dummy_policy_document() -> PolicyDocument:
    """A pre-constructed PolicyDocument instance for fast unit testing."""
    table = TableBlock(
        page=3,
        headers=["Condition", "Waiting Period", "Terms"],
        rows=[
            ["Cataract", "24 Months", "Rs. 40,000 sub-limit"],
            ["Hernia", "24 Months", "Covered after 24m"],
        ],
        markdown="| Condition | Waiting Period | Terms |\n| --- | --- | --- |\n| Cataract | 24 Months | Rs. 40,000 sub-limit |\n| Hernia | 24 Months | Covered after 24m |",
    )

    clauses = [
        Clause(
            clause_id="1.1",
            title="In-Patient Hospitalization",
            text="Covers medically necessary hospital expenses exceeding 24 hours.",
            page_start=1,
            page_end=1,
            chunk_type=ChunkType.COVERAGE,
            parent_section="Scope of Coverage",
        ),
        Clause(
            clause_id="2.1",
            title="Room Rent and ICU Capping",
            text="Room Rent capped at 1% of Sum Insured per day. ICU capped at 2% of Sum Insured.",
            page_start=2,
            page_end=2,
            chunk_type=ChunkType.SUB_LIMIT,
            parent_section="Limits and Co-Payment",
        ),
        Clause(
            clause_id="3.2",
            title="Pre-Existing Diseases Waiting Period",
            text="Pre-existing diseases are excluded until 36 months of continuous coverage have elapsed.",
            page_start=3,
            page_end=3,
            chunk_type=ChunkType.WAITING_PERIOD,
            parent_section="Waiting Periods",
            tables=[table],
        ),
        Clause(
            clause_id="4.1",
            title="Permanent Exclusions",
            text="Investigation, rest cure, cosmetic surgery, and maternity expenses are permanently excluded.",
            page_start=4,
            page_end=4,
            chunk_type=ChunkType.EXCLUSION,
            parent_section="Exclusions",
        ),
    ]

    return PolicyDocument(
        policy_id="test-care-adv",
        policy_name="Care Advantage Plan",
        insurer="Care Health Insurance Limited",
        uin="CHIHLIP21183V032021",
        total_pages=4,
        sections=[
            Section(
                section_id="SEC-1",
                title="Scope of Coverage",
                page_start=1,
                page_end=1,
                clauses=[clauses[0]],
            ),
            Section(
                section_id="SEC-2",
                title="Limits and Co-Payment",
                page_start=2,
                page_end=2,
                clauses=[clauses[1]],
            ),
            Section(
                section_id="SEC-3",
                title="Waiting Periods",
                page_start=3,
                page_end=3,
                clauses=[clauses[2]],
            ),
            Section(
                section_id="SEC-4",
                title="Exclusions",
                page_start=4,
                page_end=4,
                clauses=[clauses[3]],
            ),
        ],
    )
