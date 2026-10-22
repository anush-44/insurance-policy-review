import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class QAPair(BaseModel):
    """Schema for ground-truth question-answer evaluation pair."""

    id: str
    policy_id: str
    question: str
    gold_answer: str
    gold_clause_id: str | None = None
    gold_page: int | None = None
    is_abstention: bool = Field(
        default=False,
        description="True if the question cannot be answered from the policy.",
    )
    category: str = Field(
        default="general",
        description="Category: waiting_period | room_rent | copay | exclusion | sublimit | claims | unanswerable",
    )


class FieldExtractionLabel(BaseModel):
    """Ground-truth extracted fields for an individual policy."""

    policy_id: str
    insurer: str
    uin: str | None = None
    initial_waiting_period_days: int | None = None
    ped_waiting_period_months: int | None = None
    specific_disease_waiting_months: int | None = None
    room_rent_cap_pct: float | None = None
    room_rent_category: str | None = None
    icu_cap_pct: float | None = None
    copay_pct: float | None = None
    copay_conditions: str | None = None
    cashless_notice_planned_hours: int | None = None
    cashless_notice_emergency_hours: int | None = None
    major_exclusions: list[str] = Field(default_factory=list)


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

    if errors:
        raise ValueError(f"Found {len(errors)} errors in {path}:\n" + "\n".join(errors[:5]))

    return pairs


def validate_extraction_labels(
    file_path: Path | str = "eval/extraction_labels.json",
) -> list[FieldExtractionLabel]:
    """Validate format of extraction_labels.json."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Extraction labels file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("extraction_labels.json must contain a list of policy objects")

    return [FieldExtractionLabel.model_validate(item) for item in data]


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
