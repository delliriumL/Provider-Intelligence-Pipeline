"""Risk, confidence, and conflict scoring with component breakdowns."""

from __future__ import annotations

from datetime import date

from provider_intelligence.config import load_config
from provider_intelligence.normalize import (
    address_similarity,
    normalize_address_parts,
    normalize_phone,
    normalize_specialty,
)
from provider_intelligence.npi import is_valid_npi_format, is_valid_npi_luhn
from provider_intelligence.schemas import (
    ExternalSourceRecord,
    FieldChange,
    MatchEvidence,
    ProviderRecord,
    ScoreResult,
)


def _days_since(value: date | None) -> int | None:
    if value is None:
        return None
    return (date.today() - value).days


def compute_verification_age_risk(last_verified_date: date | None) -> float:
    """Compute risk from verification recency."""
    if last_verified_date is None:
        return 1.0
    days = _days_since(last_verified_date)
    if days is None:
        return 1.0
    if days <= 180:
        return 0.1
    if days <= 365:
        return 0.35
    if days <= 730:
        return 0.70
    return 1.0


def compute_npi_status_risk(provider: ProviderRecord, matched_source: ExternalSourceRecord | None) -> float:
    """Compute NPI-related risk."""
    if not provider.npi:
        return 0.8
    if not is_valid_npi_format(provider.npi) or not is_valid_npi_luhn(provider.npi):
        return 1.0
    if matched_source and (matched_source.active_status or "").lower() in {"deactivated", "inactive"}:
        return 1.0
    return 0.1


def compute_contact_quality_risk(
    provider: ProviderRecord,
    sources: list[ExternalSourceRecord],
) -> float:
    """Compute phone/contact quality risk."""
    risk = 0.0
    if not provider.phone or normalize_phone(provider.phone) is None:
        risk = max(risk, 0.8)
    provider_phone = normalize_phone(provider.phone)
    phones = {normalize_phone(s.phone) for s in sources if s.phone}
    phones.discard(None)
    if provider_phone and phones and provider_phone not in phones:
        risk = max(risk, 0.6)
    return risk


def compute_address_quality_risk(provider: ProviderRecord) -> float:
    """Compute address completeness and validity risk."""
    risk = 0.0
    if not provider.address_line_1 or not provider.city or not provider.state:
        risk = max(risk, 0.8)
    if provider.zip_code and len(str(provider.zip_code).replace("-", "")) < 5:
        risk = max(risk, 0.7)
    return risk


def compute_source_disagreement_risk(sources: list[ExternalSourceRecord]) -> float:
    """Compute disagreement risk across external sources."""
    if len(sources) < 2:
        return 0.0
    addresses = {
        normalize_address_parts(s.address_line_1, s.address_line_2, s.city, s.state, s.zip_code).normalized_address_line
        for s in sources
    }
    phones = {normalize_phone(s.phone) for s in sources if s.phone}
    statuses = {(s.active_status or "").lower() for s in sources}
    risk = 0.0
    if len(addresses) > 1:
        risk = max(risk, 0.7)
    if len({p for p in phones if p}) > 1:
        risk = max(risk, 0.5)
    if len(statuses) > 1:
        risk = max(risk, 0.6)
    return risk


def compute_risk_score(
    provider: ProviderRecord,
    matched_sources: list[ExternalSourceRecord],
    duplicate_risk: float,
    primary_match: ExternalSourceRecord | None = None,
) -> ScoreResult:
    """Compute overall risk score with component breakdown."""
    config = load_config()
    weights = config["field_weights"]["risk_weights"]
    components = {
        "verification_age_risk": compute_verification_age_risk(provider.last_verified_date),
        "npi_status_risk": compute_npi_status_risk(provider, primary_match),
        "contact_quality_risk": compute_contact_quality_risk(provider, matched_sources),
        "address_quality_risk": compute_address_quality_risk(provider),
        "duplicate_risk": duplicate_risk,
        "source_disagreement_risk": compute_source_disagreement_risk(matched_sources),
    }
    score = sum(weights[key] * components[key] for key in components)
    return ScoreResult(score=round(min(score, 1.0), 4), components=components)


def _source_reliability_for_field(source: ExternalSourceRecord, field: str) -> float:
    mapping = {
        "address": "address",
        "phone": "phone",
        "specialty": "specialty",
        "active_status": "status",
        "practice_name": "address",
        "npi": "identity",
    }
    key = mapping.get(field, "identity")
    return float(source.source_reliability_by_field.get(key, 0.5))


def _recency_score(source: ExternalSourceRecord) -> float:
    days = _days_since(source.last_update_date)
    if days is None:
        return 0.4
    if days <= 180:
        return 1.0
    if days <= 365:
        return 0.7
    if days <= 730:
        return 0.4
    return 0.2


def compute_field_confidence(
    field: str,
    provider: ProviderRecord,
    proposed_value: str | None,
    supporting_sources: list[ExternalSourceRecord],
    match_evidence: MatchEvidence,
    normalized_similarity: float,
) -> ScoreResult:
    """Compute field-level confidence with component breakdown."""
    config = load_config()
    weights = config["field_weights"]["confidence_weights"]

    if supporting_sources:
        source_reliability = max(_source_reliability_for_field(s, field) for s in supporting_sources)
    else:
        source_reliability = 0.3

    if match_evidence.exact_npi_match:
        exact_identifier_match = 1.0
    elif match_evidence.name_similarity >= 0.85:
        exact_identifier_match = 0.6
    else:
        exact_identifier_match = 0.0

    agreeing = sum(
        1
        for source in supporting_sources
        if _values_equal(field, proposed_value, source)
    )
    if agreeing >= 2:
        cross_source_agreement = 1.0
    elif agreeing == 1:
        cross_source_agreement = 0.6
    elif supporting_sources:
        cross_source_agreement = 0.3
    else:
        cross_source_agreement = 0.0

    field_similarity = normalized_similarity
    if cross_source_agreement >= 1.0 and len(supporting_sources) >= 2:
        field_similarity = max(normalized_similarity, 0.85)
    recency_score = max((_recency_score(s) for s in supporting_sources), default=0.3)

    components = {
        "source_reliability": source_reliability,
        "exact_identifier_match": exact_identifier_match,
        "cross_source_agreement": cross_source_agreement,
        "field_similarity": field_similarity,
        "recency_score": recency_score,
    }
    score = sum(weights[key] * components[key] for key in components)
    return ScoreResult(score=round(min(score, 1.0), 4), components=components)


def _values_equal(field: str, proposed: str | None, source: ExternalSourceRecord) -> bool:
    mapping = {
        "address": source.address_line_1,
        "phone": source.phone,
        "specialty": source.specialty,
        "active_status": source.active_status,
        "practice_name": source.practice_name,
        "npi": source.npi,
    }
    current = mapping.get(field)
    if field == "phone":
        return normalize_phone(proposed) == normalize_phone(current) and bool(proposed)
    if field == "specialty":
        return normalize_specialty(proposed) == normalize_specialty(current)
    return (proposed or "").strip().lower() == (current or "").strip().lower()


def compute_overall_confidence(changes: list[FieldChange]) -> float:
    """Compute record-level confidence as criticality-weighted average."""
    config = load_config()
    criticality = config["field_weights"]["field_criticality"]
    if not changes:
        return 1.0
    total_weight = 0.0
    weighted = 0.0
    for change in changes:
        weight = float(criticality.get(change.field, 0.5))
        total_weight += weight
        weighted += weight * change.confidence_score
    if total_weight == 0:
        return 0.0
    return round(weighted / total_weight, 4)


def compute_conflict_score(
    provider: ProviderRecord,
    sources: list[ExternalSourceRecord],
    match_evidence: MatchEvidence,
) -> ScoreResult:
    """Compute conflict score from inter-source disagreement and identity risk."""
    components: dict[str, float] = {
        "identity_conflict": 0.0,
        "address_conflict": 0.0,
        "phone_conflict": 0.0,
        "status_conflict": 0.0,
        "specialty_conflict": 0.0,
    }

    if match_evidence.identity_conflict:
        components["identity_conflict"] = 0.85

    if len(sources) >= 2:
        normalized_addrs = [
            normalize_address_parts(
                source.address_line_1, source.address_line_2, source.city, source.state, source.zip_code
            )
            for source in sources
        ]
        addr_sims: list[float] = []
        for i in range(len(normalized_addrs)):
            for j in range(i + 1, len(normalized_addrs)):
                addr_sims.append(address_similarity(normalized_addrs[i], normalized_addrs[j]))
        if addr_sims and min(addr_sims) < 0.8:
            components["address_conflict"] = 0.35

        phones = {normalize_phone(source.phone) for source in sources if source.phone}
        if len({p for p in phones if p}) > 1:
            components["phone_conflict"] = 0.25

        statuses = {(source.active_status or "").lower() for source in sources}
        if len({s for s in statuses if s}) > 1:
            if "deactivated" in statuses or "inactive" in statuses:
                components["status_conflict"] = 0.50
            else:
                components["status_conflict"] = 0.30

        specialties = {normalize_specialty(source.specialty, source.taxonomy_code) for source in sources}
        if len({s for s in specialties if s}) > 1:
            components["specialty_conflict"] = 0.20

    score = min(1.0, sum(components.values()))
    return ScoreResult(score=round(score, 4), components=components)


def detect_field_changes(
    provider: ProviderRecord,
    sources: list[ExternalSourceRecord],
    match_evidence: MatchEvidence,
) -> list[FieldChange]:
    """Detect material field changes between directory and external sources."""
    if not sources:
        return []
    primary = sources[0]
    changes: list[FieldChange] = []
    fields = ["address", "phone", "practice_name", "specialty", "active_status"]

    for field in fields:
        old_value, new_value, similarity = _compare_field(provider, primary, field)
        if old_value is None and new_value is None:
            continue
        if _is_material_change(field, old_value, new_value, similarity):
            confidence = compute_field_confidence(
                field=field,
                provider=provider,
                proposed_value=new_value,
                supporting_sources=[s for s in sources if _values_equal(field, new_value, s)],
                match_evidence=match_evidence,
                normalized_similarity=similarity,
            )
            changes.append(
                FieldChange(
                    field=field,
                    old_value=old_value,
                    new_value=new_value,
                    normalized_old_value=str(old_value).lower() if old_value else None,
                    normalized_new_value=str(new_value).lower() if new_value else None,
                    similarity=similarity,
                    confidence_score=confidence.score,
                    supporting_sources=[s.source_name for s in sources if _values_equal(field, new_value, s)],
                    conflict_detected=similarity < 0.8,
                    reason=f"{field} differs between directory and external sources",
                )
            )
    return changes


def _compare_field(
    provider: ProviderRecord,
    source: ExternalSourceRecord,
    field: str,
) -> tuple[str | None, str | None, float]:
    if field == "address":
        old_value = provider.address_line_1
        new_value = source.address_line_1
        addr_a = normalize_address_parts(
            provider.address_line_1, provider.address_line_2, provider.city, provider.state, provider.zip_code
        )
        addr_b = normalize_address_parts(
            source.address_line_1, source.address_line_2, source.city, source.state, source.zip_code
        )
        similarity = address_similarity(addr_a, addr_b)
        return old_value, new_value, similarity
    if field == "phone":
        old_value = provider.phone
        new_value = source.phone
        similarity = 1.0 if normalize_phone(old_value) == normalize_phone(new_value) else 0.0
        return old_value, new_value, similarity
    if field == "practice_name":
        old_value = provider.practice_name
        new_value = source.practice_name
        similarity = 1.0 if (old_value or "").lower() == (new_value or "").lower() else 0.5
        return old_value, new_value, similarity
    if field == "specialty":
        old_value = provider.specialty
        new_value = source.specialty
        similarity = 1.0 if normalize_specialty(old_value, provider.taxonomy_code) == normalize_specialty(
            new_value, source.taxonomy_code
        ) else 0.4
        return old_value, new_value, similarity
    if field == "active_status":
        old_value = provider.active_status
        new_value = source.active_status
        similarity = 1.0 if (old_value or "").lower() == (new_value or "").lower() else 0.0
        return old_value, new_value, similarity
    return None, None, 1.0


def _is_material_change(field: str, old_value: str | None, new_value: str | None, similarity: float) -> bool:
    if old_value is None and new_value is None:
        return False
    if field == "specialty" and similarity >= 0.9:
        return False
    return similarity < 0.95
