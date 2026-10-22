from pathlib import Path

from app.core.logging import logger
from app.core.schemas import ChunkType, Clause, PolicyDocument, Section, TableBlock


class UnstructuredParser:
    """Parser wrapper for the `unstructured` library for comparative evaluation.

    Gracefully handles absence of unstructured library with informative fallback.
    """

    def __init__(self) -> None:
        try:
            from unstructured.partition.pdf import partition_pdf  # type: ignore

            self._partition_pdf = partition_pdf
            self._available = True
        except ImportError:
            self._partition_pdf = None
            self._available = False
            logger.info("Optional package 'unstructured' is not installed; will use mock/stub.")

    @property
    def is_available(self) -> bool:
        return self._available

    def parse(
        self,
        file_path: str | Path,
        policy_id: str,
        policy_name: str,
        insurer: str,
        uin: str | None = None,
    ) -> PolicyDocument:
        """Parse using unstructured or return a simulated benchmark result if not installed."""
        if not self._available:
            logger.warning(
                "Unstructured is not installed. Run 'pip install unstructured[pdf]' to enable full comparison."
            )
            return PolicyDocument(
                policy_id=policy_id,
                policy_name=policy_name,
                insurer=insurer,
                uin=uin,
                total_pages=1,
                sections=[
                    Section(
                        section_id="SEC-1",
                        title="Unstructured Parser Unavailable",
                        page_start=1,
                        page_end=1,
                        clauses=[
                            Clause(
                                clause_id="UNSTRUCTURED-NOT-INSTALLED",
                                title="Dependency Notice",
                                text="Install unstructured to run live comparison benchmarks.",
                                page_start=1,
                                page_end=1,
                                chunk_type=ChunkType.OTHER,
                            )
                        ],
                    )
                ],
                metadata={"parser": "unstructured", "status": "unavailable"},
            )

        elements = self._partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            infer_table_structure=True,
        )

        clauses: list[Clause] = []
        clause_id = 1
        for elem in elements:
            category = getattr(elem, "category", "NarrativeText")
            page_num = getattr(elem.metadata, "page_number", 1)
            text = str(elem)

            if category == "Table":
                tables = [TableBlock(page=page_num, markdown=text, headers=[], rows=[])]
                chunk_type = ChunkType.TABLE
            else:
                tables = []
                chunk_type = ChunkType.OTHER

            clauses.append(
                Clause(
                    clause_id=f"UNSTRUCT-{clause_id}",
                    title=f"Element {clause_id} ({category})",
                    text=text,
                    page_start=page_num,
                    page_end=page_num,
                    chunk_type=chunk_type,
                    tables=tables,
                )
            )
            clause_id += 1

        return PolicyDocument(
            policy_id=policy_id,
            policy_name=policy_name,
            insurer=insurer,
            uin=uin,
            total_pages=max((c.page_end for c in clauses), default=1),
            sections=[
                Section(
                    section_id="SEC-1",
                    title="Extracted Elements",
                    page_start=1,
                    page_end=max((c.page_end for c in clauses), default=1),
                    clauses=clauses,
                )
            ],
            metadata={"parser": "unstructured", "status": "active"},
        )
