from pathlib import Path

from app.core.schemas import ChunkType, PolicyDocument
from app.ingestion.parser_comparator import compute_table_preservation_score
from app.ingestion.pdf_parser import PyMuPDFParser, classify_clause_type, table_to_markdown


def test_table_to_markdown() -> None:
    headers = ["Disease", "Waiting Period", "Conditions"]
    rows = [
        ["Cataract", "24 Months", "Max Rs 40,000"],
        ["Joint Replacement", "24 Months", "Except accidental"],
    ]
    md = table_to_markdown(headers, rows)
    assert "| Disease | Waiting Period | Conditions |" in md
    assert "| --- | --- | --- |" in md
    assert "| Cataract | 24 Months | Max Rs 40,000 |" in md
    assert "| Joint Replacement | 24 Months | Except accidental |" in md


def test_table_to_markdown_empty() -> None:
    assert table_to_markdown([], []) == ""


def test_classify_clause_type() -> None:
    assert (
        classify_clause_type("Section 3", "Waiting Periods", "30 days initial wait")
        == ChunkType.WAITING_PERIOD
    )


def test_clause_header_recognizes_single_level_number_and_rejects_bare_number() -> None:
    assert PyMuPDFParser.CLAUSE_HEADER_PATTERN.match("1. Inpatient Treatment")
    assert PyMuPDFParser.CLAUSE_HEADER_PATTERN.match("2.1 Pre-existing disease")
    assert PyMuPDFParser.CLAUSE_HEADER_PATTERN.match("b. DEFINITIONS")
    assert not PyMuPDFParser.CLAUSE_HEADER_PATTERN.match("250 per challenge, maximum")


def test_parser_keeps_clause_page_spans_and_detects_uin() -> None:
    parser = PyMuPDFParser()
    sections = parser._build_sections([
        {"page_num": 1, "text": "UIN: ICIHLIP25035V082425\n1. Waiting Period\nThis clause continues", "tables": []},
        {"page_num": 2, "text": "on the next page.\n2. Exclusions\nCosmetic surgery is excluded.", "tables": []},
    ])
    clauses = [clause for section in sections for clause in section.clauses]
    waiting = next(clause for clause in clauses if clause.clause_id == "1")
    assert waiting.page_start == 1
    assert waiting.page_end == 2
    assert "continues" in waiting.text and "next page" in waiting.text
    assert PyMuPDFParser._detect_uin([{"text": "UIN: ICIHLIP25035V082425"}]) == "ICIHLIP25035V082425"
    assert (
        classify_clause_type("Section 4", "Exclusions", "Plastic surgery is not covered")
        == ChunkType.EXCLUSION
    )
    assert (
        classify_clause_type("Limits", "Room Rent Capping", "1% per day sub-limit")
        == ChunkType.SUB_LIMIT
    )
    assert (
        classify_clause_type("Claims", "Cashless Notice", "Submit within 24 hours")
        == ChunkType.CLAIMS
    )
    assert (
        classify_clause_type("General", "Definitions", "Definition of Hospital")
        == ChunkType.DEFINITION
    )
    assert (
        classify_clause_type("Section 1", "Scope of Coverage", "Hospitalization benefits")
        == ChunkType.COVERAGE
    )


def test_pymupdf_parser_on_sample_pdf(sample_pdf_path: Path) -> None:
    parser = PyMuPDFParser()
    doc = parser.parse(
        file_source=sample_pdf_path,
        policy_id="test-care-adv",
        policy_name="Care Advantage Plan",
        insurer="Care Health Insurance Limited",
        uin="CHIHLIP21183V032021",
    )

    assert isinstance(doc, PolicyDocument)
    assert doc.total_pages == 4
    assert doc.insurer == "Care Health Insurance Limited"
    assert doc.uin == "CHIHLIP21183V032021"
    assert len(doc.sections) > 0

    all_clauses = doc.all_clauses()
    assert len(all_clauses) >= 4

    # Verify clause IDs and page numbers
    clause_ids = [c.clause_id for c in all_clauses]
    assert any("1.1" in cid or "1" in cid for cid in clause_ids)
    assert any("2.1" in cid or "2" in cid for cid in clause_ids)
    assert any("3.1" in cid or "3" in cid for cid in clause_ids)

    # Verify page range tracking
    for c in all_clauses:
        assert 1 <= c.page_start <= 4
        assert 1 <= c.page_end <= 4
        assert c.page_start <= c.page_end

    # Verify table preservation
    score_report = compute_table_preservation_score(doc)
    assert "tables_detected" in score_report
    assert "preservation_score" in score_report
