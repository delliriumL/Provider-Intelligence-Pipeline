"""Audit trail generation and export."""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from provider_intelligence.export import export_competition_recommendations, export_detailed_recommendations
from provider_intelligence.schemas import AuditEvent, Recommendation


def new_audit_id() -> str:
    """Generate a unique audit identifier."""
    return f"AUD_{uuid.uuid4().hex[:12].upper()}"


def create_audit_event(
    provider_id: str,
    step: str,
    rule_name: str,
    decision: str,
    input_summary: str = "",
    output_summary: str = "",
    score_components: dict[str, Any] | None = None,
    source_names: list[str] | None = None,
    audit_id: str | None = None,
) -> AuditEvent:
    """Create a single audit event."""
    return AuditEvent(
        audit_id=audit_id or new_audit_id(),
        timestamp=datetime.now(timezone.utc),
        provider_id=provider_id,
        step=step,
        rule_name=rule_name,
        input_summary=input_summary,
        output_summary=output_summary,
        score_components=score_components or {},
        decision=decision,
        source_names=source_names or [],
    )


def recommendation_to_audit_events(recommendation: Recommendation) -> list[AuditEvent]:
    """Convert a recommendation into readable audit events."""
    events: list[AuditEvent] = []
    events.append(
        create_audit_event(
            provider_id=recommendation.provider_id,
            step="decision",
            rule_name="decision_engine",
            decision=recommendation.recommended_action,
            input_summary=f"npi={recommendation.npi}",
            output_summary=recommendation.reason,
            score_components={
                "risk_score": recommendation.risk_score,
                "overall_confidence": recommendation.overall_confidence,
                "conflict_score": recommendation.conflict_score,
            },
            source_names=recommendation.supporting_sources,
            audit_id=recommendation.audit_id,
        )
    )
    for change in recommendation.changes:
        events.append(
            create_audit_event(
                provider_id=recommendation.provider_id,
                step="field_change",
                rule_name=f"change_{change.field}",
                decision=recommendation.recommended_action,
                input_summary=f"old={change.old_value}",
                output_summary=f"new={change.new_value}",
                score_components={
                    "confidence_score": change.confidence_score,
                    "similarity": change.similarity,
                    "normalized_old_value": change.normalized_old_value,
                    "normalized_new_value": change.normalized_new_value,
                },
                source_names=change.supporting_sources,
                audit_id=recommendation.audit_id,
            )
        )
    return events


def write_audit_log(events: list[AuditEvent], path: Path) -> None:
    """Write audit events to a reviewer-friendly CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audit_id",
        "timestamp",
        "provider_id",
        "step",
        "rule_name",
        "input_summary",
        "output_summary",
        "score_components",
        "decision",
        "source_names",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "audit_id": event.audit_id,
                    "timestamp": event.timestamp.isoformat(),
                    "provider_id": event.provider_id,
                    "step": event.step,
                    "rule_name": event.rule_name,
                    "input_summary": event.input_summary,
                    "output_summary": event.output_summary,
                    "score_components": json.dumps(event.score_components),
                    "decision": event.decision,
                    "source_names": "|".join(event.source_names),
                }
            )


def write_recommendations(recommendations: list[Recommendation], path: Path) -> None:
    """Write competition-facing and detailed recommendation outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    export_competition_recommendations(recommendations, path)
    export_detailed_recommendations(recommendations, path.parent / "recommendations_detailed.json")


def write_action_queue(
    recommendations: list[Recommendation],
    action: str,
    path: Path,
) -> None:
    """Write action-specific CSV queue."""
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = [r for r in recommendations if r.recommended_action == action]
    fieldnames = [
        "provider_id",
        "npi",
        "change_detected",
        "risk_score",
        "overall_confidence",
        "conflict_score",
        "recommended_action",
        "reason",
        "supporting_sources",
        "audit_id",
        "changes_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in filtered:
            writer.writerow(
                {
                    "provider_id": rec.provider_id,
                    "npi": rec.npi,
                    "change_detected": rec.change_detected,
                    "risk_score": rec.risk_score,
                    "overall_confidence": rec.overall_confidence,
                    "conflict_score": rec.conflict_score,
                    "recommended_action": rec.recommended_action,
                    "reason": rec.reason,
                    "supporting_sources": "|".join(rec.supporting_sources),
                    "audit_id": rec.audit_id,
                    "changes_json": json.dumps([c.model_dump() for c in rec.changes]),
                }
            )
