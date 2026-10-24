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
        r"^(?P<prefix>(?:(?:Section|Clause|Annexure|Schedule|Part)\s+(?:\d+(?:\.\d+)*|[A-Z]+)|\d+\.\d+(?:\.\d+)*\.?|\d+\.|[A-Za-z]\.|[IVXLC]{1,8}\.))\s+(?P<title>[A-Za-z][^\n]{2,110})$",
        re.MULTILINE,
    )

    SECTION_HEADER_PATTERN = re.compile(
        r"^(?:SECTION|PART)\s+(?:[0-9]+|[A-Z]+)\s*[:\-–]?\s*(?P<title>[^\n]+)",
        re.MULTILINE,
    )
    LETTER_SECTION_PATTERN = re.compile(
        r"^(?P<id>[A-Z])\.\s+(?P<title>(?:(?:Basic|Optional|Additional|Waiting|Permanent|General|Policy|Covered|Coverage|Benefits|Exclusions|Definitions|Claims)\b[^\n]{0,90}))$",
        re.IGNORECASE,
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
            uin=uin or self._detect_uin(page_data),
            total_pages=total_pages,
            sections=sections,
            metadata={
                "parser": "PyMuPDF",
                "version": pymupdf.__version__,
                "selectable_text_chars": sum(len(str(page["text"]).strip()) for page in page_data),
            },
        )

    @staticmethod
    def _detect_uin(page_data: list[dict]) -> str | None:
        """Detect a policy UIN from early page text when the caller did not supply one."""
        first_pages = "\n".join(str(page.get("text", "")) for page in page_data[:5])
        match = re.search(r"\b[A-Z]{2,8}HLIP\d{5}V\d{6}\b", first_pages.upper())
        return match.group(0) if match else None

    def _build_sections(self, page_data: list[dict]) -> list[Section]:
        """Group clause headings and body text without losing page boundaries."""
        sections: list[Section] = []
        current_section = Section(section_id="SEC-1", title="General Policy Terms", page_start=1, page_end=1, clauses=[])
        current_clause: Clause | None = None

        def finish_clause() -> None:
            nonlocal current_clause
            if current_clause is not None:
                current_clause.text = current_clause.text.strip()
                current_clause.chunk_type = classify_clause_type(current_section.title, current_clause.title, current_clause.text)
                if current_clause.text:
                    current_section.clauses.append(current_clause)
                current_clause = None

        def finish_section() -> None:
            if current_section.clauses:
                current_section.page_end = max(clause.page_end for clause in current_section.clauses)
                sections.append(current_section)

        for page in page_data:
            page_num = page["page_num"]
            lines = page["text"].splitlines()
            normalized_lines: list[str] = []
            line_index = 0
            while line_index < len(lines):
                line = lines[line_index].strip()
                # Some policy PDFs put a numbered heading marker (for example,
                # "4.") on its own line and the heading text on the next line.
                if (
                    re.fullmatch(r"(?:\d+(?:\.\d+)*|[IVXLC]{1,8})\.", line, re.IGNORECASE)
                    and line_index + 1 < len(lines)
                ):
                    next_line = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", lines[line_index + 1]).strip()
                    if next_line and re.match(r"^[A-Za-z]", next_line):
                        normalized_lines.append(f"{line} {next_line}")
                        line_index += 2
                        continue
                normalized_lines.append(lines[line_index])
                line_index += 1

            for line in normalized_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                section_match = self.SECTION_HEADER_PATTERN.match(stripped)
                letter_match = self.LETTER_SECTION_PATTERN.match(stripped)
                if section_match or letter_match:
                    finish_clause()
                    finish_section()
                    title = (section_match or letter_match).group("title").strip()
                    current_section = Section(
                        section_id=f"SEC-{len(sections) + 1}", title=title,
                        page_start=page_num, page_end=page_num, clauses=[]
                    )
                    continue

                header = self.CLAUSE_HEADER_PATTERN.match(stripped)
                if header:
                    clause_id = header.group("prefix").strip(". ")
                    title = header.group("title").strip()
                    # Avoid treating running addresses or prose as structural headings.
                    if len(clause_id) > 18 or title.lower().startswith(("the company shall", "registered office")):
                        if current_clause:
                            current_clause.text += "\n" + stripped
                            current_clause.page_end = page_num
                        continue
                    finish_clause()
                    current_clause = Clause(
                        clause_id=clause_id, title=title, text=stripped,
                        page_start=page_num, page_end=page_num,
                        parent_section=current_section.title, tables=[]
                    )
                elif current_clause is None:
                    current_clause = Clause(
                        clause_id="PREAMBLE", title="Preamble / General", text=stripped,
                        page_start=page_num, page_end=page_num,
                        parent_section=current_section.title, tables=[]
                    )
                else:
                    current_clause.text += "\n" + stripped
                    current_clause.page_end = page_num

            for table in page["tables"]:
                if current_clause:
                    current_clause.tables.append(table)
                    current_clause.text += f"\n\n[Table: Page {page_num}]\n{table.markdown}\n"
                    current_clause.page_end = page_num
                else:
                    current_section.clauses.append(Clause(
                        clause_id=f"TABLE-P{page_num}-{len(current_section.clauses) + 1}",
                        title=f"Schedule / Table (Page {page_num})", text=table.markdown,
                        page_start=page_num, page_end=page_num, chunk_type=ChunkType.TABLE,
                        parent_section=current_section.title, tables=[table]
                    ))

        finish_clause()
        finish_section()
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
