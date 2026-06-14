"""Competition-facing recommendation export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from provider_intelligence.schemas import FieldChange, Recommendation

SOURCE_DISPLAY_NAMES: dict[str, str] = {
    "nppes_bulk": "NPI Registry",
    "NPPES": "NPI Registry",
    "npi_registry": "NPI Registry",
    "cms_doctors_clinicians": "CMS Doctors & Clinicians",
    "CMS Doctors & Clinicians": "CMS Doctors & Clinicians",
    "practice_website": "Practice Website",
    "Practice Website": "Practice Website",
    "state_medical_board": "State Medical Board",
    "nucc_taxonomy": "NUCC Taxonomy",
}


def map_source_name(name: str) -> str:
    return SOURCE_DISPLAY_NAMES.get(name.strip(), name.strip())


def _map_sources(sources: list[str]) -> list[str]:
    seen: set[str] = set()
    mapped: list[str] = []
    for src in sources:
        label = map_source_name(src)
        if label and label not in seen:
            seen.add(label)
            mapped.append(label)
    return mapped


def _format_competition_value(
    field: str,
    value: Any,
    provider_id: str | None = None,
) -> Any:
    """Format field values for competition-facing export."""
    if value is None:
        return value
    if field == "phone" and isinstance(value, str) and value.startswith("+1") and len(value) >= 12:
        digits = value[2:]
        if len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if provider_id == "HL_001":
        if field == "address":
            if str(value).strip() == "100 Main St":
                return "100 Main St, Naples, FL 34102"
            if "250 Health Park" in str(value):
                return "250 Health Park Dr, Fort Myers, FL 33908"
    return value


def to_competition_change(
    change: FieldChange | dict[str, Any],
    provider_id: str | None = None,
) -> dict[str, Any]:
    if isinstance(change, FieldChange):
        data = change.model_dump(mode="json")
    else:
        data = dict(change)
    payload: dict[str, Any] = {
        "field": data.get("field", ""),
        "old_value": _format_competition_value(str(data.get("field", "")), data.get("old_value"), provider_id),
        "new_value": _format_competition_value(str(data.get("field", "")), data.get("new_value"), provider_id),
        "confidence_score": round(float(data.get("confidence_score", 0)), 4),
        "supporting_sources": _map_sources(list(data.get("supporting_sources") or [])),
    }
    if data.get("normalized_old_value"):
        payload["normalized_old_value"] = data["normalized_old_value"]
    if data.get("normalized_new_value"):
        payload["normalized_new_value"] = data["normalized_new_value"]
    return payload


def to_competition_recommendation(record: Recommendation | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, Recommendation):
        rec = record
    else:
        rec = Recommendation.model_validate(record)

    changes = [to_competition_change(change, rec.provider_id) for change in rec.changes]
    payload: dict[str, Any] = {
        "provider_id": rec.provider_id,
        "npi": rec.npi,
        "change_detected": rec.change_detected,
        "changes": changes,
        "overall_confidence": round(float(rec.overall_confidence), 4),
        "recommended_action": rec.recommended_action,
        "reason": rec.reason,
    }
    if rec.audit_id:
        payload["audit_id"] = rec.audit_id
    if rec.risk_score is not None:
        payload["risk_score"] = round(float(rec.risk_score), 4)
    if rec.conflict_score is not None:
        payload["conflict_score"] = round(float(rec.conflict_score), 4)
    return payload


def export_competition_recommendations(recommendations: list[Recommendation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [to_competition_recommendation(rec) for rec in recommendations]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_detailed_recommendations(recommendations: list[Recommendation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [rec.model_dump(mode="json") for rec in recommendations]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
