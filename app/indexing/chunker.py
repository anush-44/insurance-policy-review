import re
from typing import Protocol

from app.core.schemas import ChunkType, PolicyChunk, PolicyDocument

CROSS_REF_PATTERN = re.compile(
    r"\b(?:Clause|Section|Annexure|Schedule)\s+([0-9A-Za-z\.\-]+)",
    re.IGNORECASE,
)


def extract_cross_references(text: str) -> list[str]:
    """Find referenced clauses (e.g., 'Clause 6.4', 'Annexure I') in text."""
    matches = CROSS_REF_PATTERN.findall(text)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for m in matches:
        clean = m.strip(". ")
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


class BaseChunker(Protocol):
    """Protocol for policy document chunkers."""

    def chunk(self, doc: PolicyDocument) -> list[PolicyChunk]: ...


class ClauseAwareChunker:
    """Production clause-aware chunker.

    Splits policy by legal section and numbered clause boundaries rather than arbitrary
    character counts. Merges tiny clauses with parent headings and splits oversized clauses
    with sliding overlap to prevent context truncation.
    """

    def __init__(
        self,
        min_chunk_chars: int = 120,
        max_chunk_chars: int = 1200,
        overlap_chars: int = 200,
    ) -> None:
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, doc: PolicyDocument) -> list[PolicyChunk]:
        """Convert a structured PolicyDocument into retrievable PolicyChunks."""
        chunks: list[PolicyChunk] = []

        for section in doc.sections:
            buffer_clause_text = ""
            buffer_clause_id = ""
            buffer_clause_title = ""
            buffer_page_start = 1
            buffer_page_end = 1
            buffer_chunk_type = ChunkType.OTHER

            for clause in section.clauses:
                clause_text = clause.text.strip()
                clause_len = len(clause_text)

                # Case 1: Very small clause -> accumulate or merge
                if clause_len < self.min_chunk_chars:
                    if buffer_clause_text:
                        buffer_clause_text += f"\n\n{clause.title}\n{clause_text}".strip()
                        buffer_page_end = max(buffer_page_end, clause.page_end)
                    else:
                        buffer_clause_text = (
                            f"[{section.title}] {clause.title}\n{clause_text}".strip()
                        )
                        buffer_clause_id = clause.clause_id
                        buffer_clause_title = clause.title or section.title
                        buffer_page_start = clause.page_start
                        buffer_page_end = clause.page_end
                        buffer_chunk_type = clause.chunk_type
                    continue

                # Flush any buffered small clause first
                if buffer_clause_text:
                    chunks.append(
                        self._create_chunk(
                            doc=doc,
                            section_title=section.title,
                            clause_id=buffer_clause_id or clause.clause_id,
                            clause_title=buffer_clause_title or section.title,
                            text=buffer_clause_text,
                            page_start=buffer_page_start,
                            page_end=buffer_page_end,
                            chunk_type=buffer_chunk_type,
                            chunk_index=len(chunks) + 1,
                        )
                    )
                    buffer_clause_text = ""

                # Case 2: Clause is within standard bounds
                if clause_len <= self.max_chunk_chars:
                    # Enrich with section context if not already present
                    text = clause_text
                    if section.title.lower() not in text[:100].lower():
                        text = f"[{section.title}] {clause.title}\n{text}".strip()

                    chunks.append(
                        self._create_chunk(
                            doc=doc,
                            section_title=section.title,
                            clause_id=clause.clause_id,
                            clause_title=clause.title,
                            text=text,
                            page_start=clause.page_start,
                            page_end=clause.page_end,
                            chunk_type=clause.chunk_type,
                            chunk_index=len(chunks) + 1,
                        )
                    )
                else:
                    # Case 3: Oversized clause -> split with overlap
                    sub_chunks = self._split_long_clause(
                        doc=doc,
                        section_title=section.title,
                        clause_id=clause.clause_id,
                        clause_title=clause.title,
                        text=clause_text,
                        page_start=clause.page_start,
                        page_end=clause.page_end,
                        chunk_type=clause.chunk_type,
                        start_index=len(chunks) + 1,
                    )
                    chunks.extend(sub_chunks)

            # Flush any trailing buffered small clause
            if buffer_clause_text:
                chunks.append(
                    self._create_chunk(
                        doc=doc,
                        section_title=section.title,
                        clause_id=buffer_clause_id,
                        clause_title=buffer_clause_title,
                        text=buffer_clause_text,
                        page_start=buffer_page_start,
                        page_end=buffer_page_end,
                        chunk_type=buffer_chunk_type,
                        chunk_index=len(chunks) + 1,
                    )
                )

        return chunks

    def _create_chunk(
        self,
        doc: PolicyDocument,
        section_title: str,
        clause_id: str,
        clause_title: str,
        text: str,
        page_start: int,
        page_end: int,
        chunk_type: ChunkType,
        chunk_index: int,
    ) -> PolicyChunk:
        return PolicyChunk(
            chunk_id=f"{doc.policy_id}_{clause_id}_{chunk_index}",
            policy_id=doc.policy_id,
            insurer=doc.insurer,
            section=section_title,
            clause_id=clause_id,
            clause_title=clause_title,
            text=text,
            page_start=page_start,
            page_end=page_end,
            chunk_type=chunk_type,
            cross_references=extract_cross_references(text),
            metadata={
                "uin": doc.uin,
                "policy_name": doc.policy_name,
                "chunker": "clause_aware",
            },
        )

    def _split_long_clause(
        self,
        doc: PolicyDocument,
        section_title: str,
        clause_id: str,
        clause_title: str,
        text: str,
        page_start: int,
        page_end: int,
        chunk_type: ChunkType,
        start_index: int,
    ) -> list[PolicyChunk]:
        """Split oversized text into overlapping windows while preserving clause context."""
        sub_chunks: list[PolicyChunk] = []
        step = self.max_chunk_chars - self.overlap_chars
        sub_idx = 1

        for i in range(0, len(text), step):
            sub_text = text[i : i + self.max_chunk_chars]
            if len(sub_text.strip()) < 50:
                continue

            header_prefix = f"[{section_title} | {clause_id} {clause_title} (Part {sub_idx})]\n"
            full_text = (header_prefix + sub_text).strip()

            sub_chunks.append(
                PolicyChunk(
                    chunk_id=f"{doc.policy_id}_{clause_id}_{start_index}_{sub_idx}",
                    policy_id=doc.policy_id,
                    insurer=doc.insurer,
                    section=section_title,
                    clause_id=clause_id,
                    clause_title=f"{clause_title} (Part {sub_idx})",
                    text=full_text,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_type=chunk_type,
                    cross_references=extract_cross_references(full_text),
                    metadata={
                        "uin": doc.uin,
                        "policy_name": doc.policy_name,
                        "chunker": "clause_aware",
                        "part": sub_idx,
                    },
                )
            )
            sub_idx += 1

        return sub_chunks


class FixedSizeChunker:
    """Fixed-size character window chunking baseline for ablation study."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: PolicyDocument) -> list[PolicyChunk]:
        """Chunk entire document into fixed character windows regardless of clause boundaries."""
        # Concatenate text page-by-page
        page_texts: list[tuple[int, str, int, int]] = []
        for section in doc.sections:
            for clause in section.clauses:
                page_texts.append((clause.page_start, clause.text, clause.page_start, clause.page_end))

        parts: list[str] = []
        spans: list[tuple[int, int, int, int]] = []
        char_offset = 0
        for _, text, page_start, page_end in page_texts:
            if parts:
                parts.append("\n\n")
                char_offset += 2
            start = char_offset
            parts.append(text)
            char_offset += len(text)
            spans.append((start, char_offset, page_start, page_end))
        full_doc_text = "".join(parts)
        chunks: list[PolicyChunk] = []
        step = self.chunk_size - self.chunk_overlap
        idx = 1

        for i in range(0, len(full_doc_text), step):
            window = full_doc_text[i : i + self.chunk_size]
            if not window.strip():
                continue

            window_end = i + len(window)
            overlapping_pages = [
                (page_start, page_end)
                for start, end, page_start, page_end in spans
                if start < window_end and end > i
            ]
            page_start = min((span[0] for span in overlapping_pages), default=1)
            page_end = max((span[1] for span in overlapping_pages), default=doc.total_pages)
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{doc.policy_id}_fixed_{idx}",
                    policy_id=doc.policy_id,
                    insurer=doc.insurer,
                    section="FixedWindow",
                    clause_id=f"FIXED-{idx}",
                    clause_title=f"Fixed Chunk {idx}",
                    text=window.strip(),
                    page_start=page_start,
                    page_end=page_end,
                    chunk_type=ChunkType.OTHER,
                    cross_references=extract_cross_references(window),
                    metadata={"chunker": "fixed_size", "window_size": self.chunk_size},
                )
            )
            idx += 1

        return chunks
