import re
from pathlib import Path
from typing import BinaryIO

import pymupdf

from app.core.logging import logger
from app.core.schemas import ChunkType, Clause, PolicyDocument, Section, TableBlock


def table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Format headers and rows into a standard Markdown table."""
    if not headers and not rows:
        return ""

    # Clean strings
    clean_headers = [str(h or "").strip().replace("\n", " ") for h in headers]
    clean_rows = [[str(c or "").strip().replace("\n", " ") for c in row] for row in rows]

    # If headers are missing, synthesize col1, col2...
    col_count = max(len(clean_headers), max((len(r) for r in clean_rows), default=0))
    if col_count == 0:
        return ""

    if not clean_headers or len(clean_headers) < col_count:
        clean_headers = [f"Col {i + 1}" for i in range(col_count)]

    # Normalize row lengths
    normalized_rows = []
    for row in clean_rows:
        if len(row) < col_count:
            row = row + [""] * (col_count - len(row))
        normalized_rows.append(row[:col_count])

    md_lines = [
        "| " + " | ".join(clean_headers) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in normalized_rows:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def classify_clause_type(section_title: str, clause_title: str, text: str) -> ChunkType:
    """Infer chunk type using domain heuristics common in Indian health insurance."""
    combined = f"{section_title} {clause_title} {text[:400]}".lower()

    if any(
        k in combined
        for k in [
            "waiting period",
            "initial waiting",
            "pre-existing disease",
            "ped waiting",
            "specific disease",
        ]
    ):
        return ChunkType.WAITING_PERIOD
    if any(
        k in combined
        for k in [
            "exclusion",
            "not covered",
            "exclusions - permanent",
            "what is not covered",
            "permanent exclusion",
        ]
    ):
        return ChunkType.EXCLUSION
    if any(
        k in combined
        for k in [
            "sub-limit",
            "room rent",
            "icu capping",
            "capping on",
            "copayment",
            "co-pay",
            "proportionate",
        ]
    ):
        return ChunkType.SUB_LIMIT
    if any(k in combined for k in ["definition", "definitions"]):
        return ChunkType.DEFINITION
    if any(
        k in combined for k in ["claim", "claims", "cashless", "reimbursement", "notice of claim"]
    ):
        return ChunkType.CLAIMS
    if any(
        k in combined
        for k in [
            "coverage",
            "scope of cover",
            "in-patient",
            "hospitalization",
            "day care",
            "benefits",
        ]
    ):
        return ChunkType.COVERAGE
    if "table" in combined or "|" in text:
        return ChunkType.TABLE

    return ChunkType.OTHER


class PyMuPDFParser:
    """Production PDF parser for insurance policies using PyMuPDF.

    Extracts text, identifies section & clause hierarchy, and extracts tables as Markdown.
    """

    # Matches patterns like '4.1 Waiting Periods', 'Section 3. Scope', 'Clause 5.2', 'IV. Exclusions'
    CLAUSE_HEADER_PATTERN = re.compile(
        r"^(?P<prefix>(?:Section\s+[0-9A-Za-z\.]+|Clause\s+[0-9A-Za-z\.]+|Annexure\s+[0-9A-Za-z\.]+|[A-Z]\.\s+|\b[0-9]+\.[0-9]+(?:\.[0-9]+)*\b\.?))\s*(?P<title>[A-Za-z][^\n]{2,80})$",
        re.MULTILINE,
    )

    SECTION_HEADER_PATTERN = re.compile(
        r"^(?:SECTION|PART)\s+(?:[0-9]+|[A-Z]+)\s*[:\-–]?\s*(?P<title>[^\n]+)",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self) -> None:
        pass

    def parse(
        self,
        file_source: str | Path | BinaryIO | bytes,
        policy_id: str,
        policy_name: str,
        insurer: str,
        uin: str | None = None,
    ) -> PolicyDocument:
        """Parse a PDF file into a structured PolicyDocument."""
        if isinstance(file_source, (str, Path)):
            doc = pymupdf.open(str(file_source))
        elif isinstance(file_source, bytes):
            doc = pymupdf.open(stream=file_source, filetype="pdf")
        else:
            doc = pymupdf.open(stream=file_source.read(), filetype="pdf")

        total_pages = doc.page_count
        logger.info(f"Parsing policy '{policy_name}' ({total_pages} pages) with PyMuPDF")

        page_data: list[dict] = []

        # Step 1: Extract text and tables per page
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # Extract tables using PyMuPDF table finder
            tables: list[TableBlock] = []
            table_rects = []
            try:
                table_finder = page.find_tables()
                if table_finder and table_finder.tables:
                    for t in table_finder.tables:
                        headers = [str(col) if col is not None else "" for col in t.header.names]
                        rows = []
                        for row in t.extract():
                            rows.append([str(c) if c is not None else "" for c in row])
                        md = table_to_markdown(headers, rows)
                        tables.append(
                            TableBlock(
                                page=page_num,
                                headers=headers,
                                rows=rows,
                                markdown=md,
                            )
                        )
                        table_rects.append(t.bbox)
            except Exception as e:
                logger.warning(f"Failed extracting tables on page {page_num}: {e}")

            # Extract plain text
            raw_text = page.get_text("text")

            page_data.append(
                {
                    "page_num": page_num,
                    "text": raw_text,
                    "tables": tables,
                }
            )

        doc.close()

        # Step 2: Assemble structured sections and clauses
        sections = self._build_sections(page_data)

        # If no sections could be identified by headers, wrap everything in a default section
        if not sections or (len(sections) == 1 and not sections[0].clauses):
            sections = self._fallback_sections(page_data)

        return PolicyDocument(
            policy_id=policy_id,
            policy_name=policy_name,
            insurer=insurer,
            uin=uin,
            total_pages=total_pages,
            sections=sections,
            metadata={"parser": "PyMuPDF", "version": pymupdf.__version__},
        )

    def _build_sections(self, page_data: list[dict]) -> list[Section]:
        """Group parsed pages into sections and clauses based on header matching."""
        sections: list[Section] = []
        current_section = Section(
            section_id="SEC-0",
            title="General Policy Terms",
            page_start=1,
            page_end=1,
            clauses=[],
        )
        current_clause: Clause | None = None
        clause_counter = 1

        for page in page_data:
            page_num = page["page_num"]
            lines = page["text"].split("\n")
            current_clause_lines: list[str] = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                # Check if this line is a top-level section header
                sec_match = self.SECTION_HEADER_PATTERN.match(stripped)
                if sec_match:
                    # Flush current clause
                    if current_clause and current_clause_lines:
                        current_clause.text += "\n" + "\n".join(current_clause_lines)
                        current_clause.page_end = page_num
                        current_section.clauses.append(current_clause)
                        current_clause = None
                        current_clause_lines = []

                    # If previous section has clauses or content, push it
                    if current_section.clauses:
                        current_section.page_end = page_num
                        sections.append(current_section)

                    sec_title = sec_match.group("title").strip()
                    current_section = Section(
                        section_id=f"SEC-{len(sections) + 1}",
                        title=sec_title,
                        page_start=page_num,
                        page_end=page_num,
                        clauses=[],
                    )
                    continue

                # Check if this line is a clause header (e.g., '4.1 Waiting Periods')
                clause_match = self.CLAUSE_HEADER_PATTERN.match(stripped)
                if clause_match:
                    prefix = clause_match.group("prefix").strip(". ")
                    title = clause_match.group("title").strip()

                    # Save existing clause
                    if current_clause:
                        if current_clause_lines:
                            current_clause.text += "\n" + "\n".join(current_clause_lines)
                        current_clause.page_end = page_num
                        current_clause.chunk_type = classify_clause_type(
                            current_section.title, current_clause.title, current_clause.text
                        )
                        current_section.clauses.append(current_clause)
                        current_clause_lines = []

                    clause_id = prefix if prefix else f"C-{clause_counter}"
                    clause_counter += 1

                    current_clause = Clause(
                        clause_id=clause_id,
                        title=title,
                        text=stripped,
                        page_start=page_num,
                        page_end=page_num,
                        parent_section=current_section.title,
                        tables=[],
                    )
                else:
                    if current_clause:
                        current_clause_lines.append(stripped)
                    else:
                        # Before first clause, accumulate into a preliminary clause
                        if not current_section.clauses:
                            current_clause = Clause(
                                clause_id="PREAMBLE",
                                title="Preamble / General",
                                text=stripped,
                                page_start=page_num,
                                page_end=page_num,
                                parent_section=current_section.title,
                            )
                        else:
                            current_clause_lines.append(stripped)

            # Page transition: flush lines to active clause
            if current_clause and current_clause_lines:
                current_clause.text += "\n" + "\n".join(current_clause_lines)
                current_clause.page_end = page_num
                current_clause_lines = []

            # Attach page tables to the active clause or section
            if page["tables"]:
                if current_clause:
                    current_clause.tables.extend(page["tables"])
                    # Append markdown representation to clause text for searchability
                    for t in page["tables"]:
                        current_clause.text += f"\n\n[Table: Page {page_num}]\n{t.markdown}\n"
                else:
                    # Create a standalone table clause
                    for t in page["tables"]:
                        t_clause = Clause(
                            clause_id=f"TABLE-P{page_num}-{clause_counter}",
                            title=f"Schedule / Table (Page {page_num})",
                            text=t.markdown,
                            page_start=page_num,
                            page_end=page_num,
                            chunk_type=ChunkType.TABLE,
                            parent_section=current_section.title,
                            tables=[t],
                        )
                        clause_counter += 1
                        current_section.clauses.append(t_clause)

        # End of document: flush final clause and section
        if current_clause:
            current_clause.chunk_type = classify_clause_type(
                current_section.title, current_clause.title, current_clause.text
            )
            current_section.clauses.append(current_clause)

        if current_section.clauses:
            current_section.page_end = page_data[-1]["page_num"] if page_data else 1
            sections.append(current_section)

        return sections

    def _fallback_sections(self, page_data: list[dict]) -> list[Section]:
        """Fallback chunker if PDF structure lacks standard numbering patterns."""
        clauses: list[Clause] = []
        for page in page_data:
            page_num = page["page_num"]
            text = page["text"].strip()
            table_md = "\n\n".join(t.markdown for t in page["tables"])
            full_text = f"{text}\n\n{table_md}".strip() if table_md else text

            if full_text:
                c_type = ChunkType.TABLE if page["tables"] and not text else ChunkType.OTHER
                clauses.append(
                    Clause(
                        clause_id=f"PAGE-{page_num}",
                        title=f"Page {page_num} Content",
                        text=full_text,
                        page_start=page_num,
                        page_end=page_num,
                        chunk_type=c_type,
                        parent_section="General Terms",
                        tables=page["tables"],
                    )
                )

        return [
            Section(
                section_id="SEC-GENERAL",
                title="General Policy Terms",
                page_start=1,
                page_end=len(page_data),
                clauses=clauses,
            )
        ]
