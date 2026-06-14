"""Pydantic data models for the provider intelligence pipeline."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ProviderRecord(BaseModel):
    """Internal HealthLynked provider/practice directory record."""

    provider_id: str
    provider_name: str
    npi: str | None = None
    specialty: str | None = None
    taxonomy_code: str | None = None
    practice_name: str | None = None
    address_line_1: str | None = None
    address_line_2: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    phone: str | None = None
    website: str | None = None
    active_status: str | None = None
    last_verified_date: date | None = None
    source_system: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ExternalSourceRecord(BaseModel):
    """Normalized external source record used for matching and scoring."""

    source_name: str
    source_record_id: str
    provider_name: str | None = None
    npi: str | None = None
    specialty: str | None = None
    taxonomy_code: str | None = None
    practice_name: str | None = None
    address_line_1: str | None = None
    address_line_2: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    phone: str | None = None
    website: str | None = None
    active_status: str | None = None
    last_update_date: date | None = None
    source_reliability_by_field: dict[str, float] = Field(default_factory=dict)


class FieldChange(BaseModel):
    """Proposed field-level change with audit metadata."""

    field: str
    old_value: str | None = None
    new_value: str | None = None
    normalized_old_value: str | None = None
    normalized_new_value: str | None = None
    similarity: float | None = None
    confidence_score: float = 0.0
    supporting_sources: list[str] = Field(default_factory=list)
    conflict_detected: bool = False
    reason: str | None = None


class Recommendation(BaseModel):
    """Pipeline recommendation for a provider record."""

    provider_id: str
    npi: str | None = None
    change_detected: bool = False
    changes: list[FieldChange] = Field(default_factory=list)
    risk_score: float = 0.0
    overall_confidence: float = 0.0
    conflict_score: float = 0.0
    recommended_action: Literal["no_change", "auto_update", "human_review", "do_not_update"] = "no_change"
    reason: str = ""
    supporting_sources: list[str] = Field(default_factory=list)
    audit_id: str = ""
    llm_enrichment: "LLMEnrichment | None" = None


class LLMEnrichment(BaseModel):
    """Bounded LLM enrichment — does not override deterministic decisions."""

    enriched: bool = False
    use_case: str = ""
    reviewer_summary: str | None = None
    conflict_explanation: str | None = None
    specialty_hint: str | None = None
    extracted_website_fields: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: str | None = None
    llm_did_not_change_decision: bool = True


class AuditEvent(BaseModel):
    """Single audit trail event."""

    audit_id: str
    timestamp: datetime
    provider_id: str
    step: str
    rule_name: str
    input_summary: str = ""
    output_summary: str = ""
    score_components: dict[str, Any] = Field(default_factory=dict)
    decision: str = ""
    source_names: list[str] = Field(default_factory=list)


class ScoreResult(BaseModel):
    """Score with human-readable component breakdown."""

    score: float
    components: dict[str, float] = Field(default_factory=dict)


class MatchEvidence(BaseModel):
    """Evidence from provider/source matching."""

    match_type: str
    match_score: float
    matched_source: str | None = None
    exact_npi_match: bool = False
    name_similarity: float = 0.0
    address_similarity: float = 0.0
    phone_match: bool = False
    specialty_similarity: float = 0.0
    identity_conflict: bool = False
    duplicate_risk: float = 0.0


class LLMExtractionResult(BaseModel):
    """Structured LLM extraction output for reviewer enrichment."""

    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: str = ""
    confidence_hint: float = 0.0
    evidence_snippets: list[str] = Field(default_factory=list)
    recommended_review_note: str | None = None


class GroundTruthLabel(BaseModel):
    """Synthetic benchmark ground truth label."""

    provider_id: str
    mutation_type: str | None = None
    expected_action: str | None = None
    changed_fields: list[str] = Field(default_factory=list)
    notes: str = ""
