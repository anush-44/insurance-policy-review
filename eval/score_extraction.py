"""Score supplied structured extraction predictions against manual labels.

Prediction JSON schema::
  {"predictions": [{"policy_id": "...", "fields": {
    "field_name": {"value": ..., "status": "known|unknown|conditional",
                   "evidence": [{"clause_id": "...", "page": 1}]}}]}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return sorted(
            (normalize(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True)
        )
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip().lower())
    return value


def score(labels_path: Path, predictions_path: Path) -> dict[str, Any]:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    predictions_data = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions = {item["policy_id"]: item for item in predictions_data.get("predictions", [])}
    per_field: dict[str, dict[str, Any]] = {}
    overall: dict[str, list[bool]] = defaultdict(list)
    missing_predictions: list[str] = []

    for policy in labels["policies"]:
        pred_policy = predictions.get(policy["policy_id"], {})
        pred_fields = pred_policy.get("fields", {})
        for name, gold in policy["fields"].items():
            status = gold["status"]
            if status == "not_labeled":
                continue
            prediction = pred_fields.get(name)
            if prediction is None:
                missing_predictions.append(f"{policy['policy_id']}.{name}")
                predicted_value = None
                predicted_status = "missing"
                evidence = []
            else:
                predicted_value = prediction.get("value")
                predicted_status = prediction.get(
                    "status", "known" if predicted_value is not None else "unknown"
                )
                evidence = prediction.get("evidence", [])
            if status == "schedule_required":
                matched = predicted_status == "unknown" or predicted_value is None
                error = None if matched else "guessed_schedule_value"
            else:
                matched = normalize(predicted_value) == normalize(gold["gold_value"])
                error = (
                    None
                    if matched
                    else ("missing_value" if predicted_value is None else "incorrect_value")
                )
            overall[name].append(bool(matched))
            entry = per_field.setdefault(
                name, {"status": status, "n": 0, "correct": 0, "predictions": []}
            )
            entry["n"] += 1
            entry["correct"] += int(matched)
            entry["predictions"].append(
                {
                    "policy_id": policy["policy_id"],
                    "gold_value": gold["gold_value"],
                    "predicted_value": predicted_value,
                    "predicted_status": predicted_status,
                    "supporting_evidence": evidence,
                    "evidence_present": bool(evidence),
                    "matched": bool(matched),
                    "error_category": error,
                }
            )

    for field in per_field.values():
        field["accuracy"] = field["correct"] / field["n"] if field["n"] else 0.0
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "labels": str(labels_path),
        "predictions": str(predictions_path),
        "policies_scored": len(labels["policies"]),
        "fields_scored": sum(len(values) for values in overall.values()),
        "accuracy": sum(sum(values) for values in overall.values())
        / max(1, sum(map(len, overall.values()))),
        "missing_predictions": missing_predictions,
        "per_field": per_field,
        "note": "Exact normalized value matching. No unsupported precision/recall calculation is made for single-valued fields.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("eval/extraction_labels.json"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.labels, args.predictions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Scored {report['fields_scored']} field predictions; accuracy={report['accuracy']:.3f}; report={args.out}"
    )


if __name__ == "__main__":
    main()
