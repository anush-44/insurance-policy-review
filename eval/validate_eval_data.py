import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class QAPair(BaseModel):
    """Schema for ground-truth question-answer evaluation pair."""

    id: str
    policy_id: str
    question: str
    gold_answer: str
    gold_clause_ids: list[str] = Field(default_factory=list)
    gold_pages: list[int] = Field(default_factory=list)
    answerability: Literal["answerable", "unanswerable"]
    category: str = Field(
        default="general",
        description="Category: waiting_period | room_rent | copay | exclusion | sublimit | claims | unanswerable",
    )
    notes: str

    @property
    def gold_clause_id(self) -> str | None:
        """First gold reference, retained for the existing single-citation harness."""
        return self.gold_clause_ids[0] if self.gold_clause_ids else None

    @property
    def gold_page(self) -> int | None:
        return self.gold_pages[0] if self.gold_pages else None

    @property
    def is_abstention(self) -> bool:
        return self.answerability == "unanswerable"


class CompareCase(BaseModel):
    """Scenario for comparative policy evaluation."""

    id: str
    scenario_name: str
    procedure_or_condition: str
    patient_age: int | None = None
    policy_a_id: str
    policy_b_id: str
    expected_winner: str | None = None
    notes: str = ""


def validate_qa_set(file_path: Path | str = "eval/qa_set.jsonl") -> list[QAPair]:
    """Validate format and contents of qa_set.jsonl."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"QA set file not found: {path}")

    pairs: list[QAPair] = []
    errors: list[str] = []

    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pair = QAPair.model_validate(data)
                pairs.append(pair)
            except (json.JSONDecodeError, ValidationError) as e:
                errors.append(f"Line {idx}: {e}")

    ids = [pair.id for pair in pairs]
    if len(ids) != len(set(ids)):
        errors.append("Question IDs must be unique")
    if errors:
        raise ValueError(f"Found {len(errors)} errors in {path}:\n" + "\n".join(errors[:10]))

    return pairs


def validate_qa_references(
    qa_path: Path | str = "eval/qa_set.jsonl",
    manifest_path: Path | str = "data/policies.csv",
) -> dict[str, object]:
    """Check policy IDs, page bounds, and quoted clause identifiers against the local PDFs."""
    import csv
    import re

    import pymupdf

    from data.acquire_policies import ROOT

    qa_path, manifest_path = Path(qa_path), Path(manifest_path)
    pairs = validate_qa_set(qa_path)
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        manifest = {row["policy_id"]: row for row in csv.DictReader(stream)}
    errors: list[str] = []
    opened: dict[str, list[str]] = {}
    for pair in pairs:
        row = manifest.get(pair.policy_id)
        if row is None:
            errors.append(f"{pair.id}: unknown policy_id {pair.policy_id}")
            continue
        pdf_path = ROOT / row["local_path"]
        if not pdf_path.is_file():
            errors.append(f"{pair.id}: PDF missing for {pair.policy_id}: {pdf_path}")
            continue
        if pair.answerability == "unanswerable" and (pair.gold_clause_ids or pair.gold_pages):
            # Unanswerable cases may cite an explicit limitation, but must keep paired references.
            if len(pair.gold_clause_ids) != len(pair.gold_pages):
                errors.append(f"{pair.id}: evidence clause/page arrays must have the same length")
        if len(pair.gold_clause_ids) != len(pair.gold_pages):
            errors.append(f"{pair.id}: gold clause/page arrays must have the same length")
        try:
            if pair.policy_id not in opened:
                with pymupdf.open(pdf_path) as doc:
                    opened[pair.policy_id] = [page.get_text("text") for page in doc]
            pages = opened[pair.policy_id]
        except Exception as exc:
            errors.append(f"{pair.id}: PDF could not be opened: {exc}")
            continue
        for clause_id, page_no in zip(pair.gold_clause_ids, pair.gold_pages):
            if not 1 <= page_no <= len(pages):
                errors.append(f"{pair.id}: gold page {page_no} outside 1..{len(pages)}")
                continue
            normalized_page = re.sub(r"\s+", " ", pages[page_no - 1])
            normalized_clause = re.sub(r"\s+", " ", clause_id)
            if not re.search(re.escape(normalized_clause), normalized_page, re.IGNORECASE):
                errors.append(f"{pair.id}: gold clause/reference {clause_id!r} not found on page {page_no}")
    return {
        "questions": len(pairs),
        "policies": sorted({pair.policy_id for pair in pairs}),
        "answerable": sum(pair.answerability == "answerable" for pair in pairs),
        "unanswerable": sum(pair.answerability == "unanswerable" for pair in pairs),
        "errors": errors,
        "valid": not errors,
    }


def validate_extraction_references(
    labels_path: Path | str = "eval/extraction_labels.json",
    manifest_path: Path | str = "data/policies.csv",
) -> dict[str, object]:
    """Verify manually labeled extraction references against local PDF page text."""
    import csv
    import re

    import pymupdf

    from data.acquire_policies import ROOT

    labels_path, manifest_path = Path(labels_path), Path(manifest_path)
    records = validate_extraction_labels(labels_path)
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        manifest = {row["policy_id"]: row for row in csv.DictReader(stream)}
    errors: list[str] = []
    verified_fields = 0
    for record in records:
        row = manifest.get(str(record["policy_id"]))
        if row is None:
            errors.append(f"Unknown extraction policy_id: {record['policy_id']}")
            continue
        pdf_path = ROOT / row["local_path"]
        if not pdf_path.is_file():
            errors.append(f"Missing extraction source PDF: {pdf_path}")
            continue
        try:
            with pymupdf.open(pdf_path) as doc:
                page_texts = [re.sub(r"\s+", " ", page.get_text("text")) for page in doc]
        except Exception as exc:
            errors.append(f"Could not open extraction PDF {pdf_path}: {exc}")
            continue
        for field_name, field in record["fields"].items():
            if field["status"] != "verified":
                continue
            verified_fields += 1
            pages = field.get("gold_pages", [])
            if any(not isinstance(page, int) or page < 1 or page > len(page_texts) for page in pages):
                errors.append(f"{record['policy_id']}.{field_name}: page outside PDF bounds")
                continue
            context = " ".join(page_texts[page - 1] for page in pages)
            for clause_id in field.get("gold_clause_ids", []):
                reference = re.sub(r"\s+", " ", str(clause_id))
                if not re.search(re.escape(reference), context, re.IGNORECASE):
                    errors.append(f"{record['policy_id']}.{field_name}: reference {clause_id!r} not found on labeled pages")
    return {"policies": len(records), "verified_fields": verified_fields, "errors": errors, "valid": not errors}


def validate_extraction_labels(
    file_path: Path | str = "eval/extraction_labels.json",
) -> list[dict[str, object]]:
    """Validate format of extraction_labels.json."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Extraction labels file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("policies"), list):
        raise ValueError("extraction_labels.json must use schema_version 1 with a policies list")
    policies = data["policies"]
    seen: set[str] = set()
    for item in policies:
        if not isinstance(item, dict) or not item.get("policy_id") or not isinstance(item.get("fields"), dict):
            raise ValueError("Each extraction policy needs policy_id and a fields object")
        if item["policy_id"] in seen:
            raise ValueError(f"Duplicate extraction policy_id: {item['policy_id']}")
        seen.add(str(item["policy_id"]))
        for field_name, field in item["fields"].items():
            if not isinstance(field, dict) or "gold_value" not in field or "status" not in field:
                raise ValueError(f"{item['policy_id']}.{field_name} needs gold_value and status")
            if field["status"] == "verified" and (not field.get("gold_clause_ids") or not field.get("gold_pages")):
                raise ValueError(f"{item['policy_id']}.{field_name} verified label needs clause and page references")
    return policies


def validate_compare_cases(
    file_path: Path | str = "eval/compare_cases.jsonl",
) -> list[CompareCase]:
    """Validate format of compare_cases.jsonl."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Compare cases file not found: {path}")

    cases: list[CompareCase] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            cases.append(CompareCase.model_validate(json.loads(line)))

    return cases


if __name__ == "__main__":
    print("Validating evaluation datasets...")
    try:
        qa = validate_qa_set()
        print(
            f"[PASS] qa_set.jsonl valid: {len(qa)} questions ({sum(1 for q in qa if q.is_abstention)} abstentions)"
        )
    except Exception as e:
        print(f"[FAIL] qa_set.jsonl error: {e}")

    try:
        ext = validate_extraction_labels()
        print(f"[PASS] extraction_labels.json valid: {len(ext)} policies")
    except Exception as e:
        print(f"[FAIL] extraction_labels.json error: {e}")

    try:
        cmp = validate_compare_cases()
        print(f"[PASS] compare_cases.jsonl valid: {len(cmp)} scenarios")
    except Exception as e:
        print(f"[FAIL] compare_cases.jsonl error: {e}")

    try:
        report = validate_qa_references()
        print(f"[{'PASS' if report['valid'] else 'FAIL'}] QA references: {report['questions']} questions, {len(report['policies'])} policies")
        for error in report["errors"]:
            print(f"  - {error}")
    except Exception as e:
        print(f"[FAIL] QA reference validation error: {e}")

    try:
        report = validate_extraction_references()
        print(f"[{'PASS' if report['valid'] else 'FAIL'}] Extraction references: {report['verified_fields']} verified fields")
        for error in report["errors"]:
            print(f"  - {error}")
    except Exception as e:
        print(f"[FAIL] Extraction reference validation error: {e}")
