"""Conservative decision engine for provider update recommendations."""

from __future__ import annotations

from provider_intelligence.config import load_config
from provider_intelligence.npi import is_valid_npi_luhn
from provider_intelligence.schemas import FieldChange, MatchEvidence, Recommendation, ScoreResult

AUTO_UPDATE_ALLOWED_FIELDS = frozenset({"address", "phone"})


def decide_action(
    recommendation: Recommendation,
    match_evidence: MatchEvidence,
    risk: ScoreResult,
    conflict: ScoreResult,
    duplicate_risk: float,
) -> Recommendation:
    """Apply conservative routing rules and attach a human-readable reason."""
    config = load_config()
    thresholds = config["thresholds"]
    auto_threshold = thresholds["confidence"]["auto_update_threshold"]
    no_change_threshold = thresholds["confidence"]["no_change_threshold"]
    review_min = thresholds["confidence"]["human_review_min_threshold"]
    max_conflict_auto = thresholds["conflict"]["max_conflict_for_auto_update"]
    duplicate_threshold = config["app"]["matching"]["duplicate_risk_threshold"]

    recommendation.risk_score = risk.score
    recommendation.conflict_score = conflict.score
    recommendation.overall_confidence = _overall_confidence(recommendation.changes)

    identity_conflict = match_evidence.identity_conflict or conflict.components.get("identity_conflict", 0) >= 0.8
    invalid_npi = bool(recommendation.npi) and not is_valid_npi_luhn(recommendation.npi)
    no_exact_npi = not match_evidence.exact_npi_match

    if identity_conflict:
        recommendation.recommended_action = "human_review" if match_evidence.match_score >= 0.6 else "do_not_update"
        recommendation.reason = "Identity conflict detected; auto-update blocked for safety."
        return recommendation

    if invalid_npi and not match_evidence.exact_npi_match:
        recommendation.recommended_action = "do_not_update"
        recommendation.reason = "Invalid NPI with no reliable identity match."
        return recommendation

    if duplicate_risk > duplicate_threshold:
        recommendation.recommended_action = "human_review"
        recommendation.reason = "Duplicate risk exceeds safe threshold; requires human review."
        return recommendation

    if not recommendation.change_detected:
        recommendation.recommended_action = "no_change"
        recommendation.reason = "Directory fields materially match external sources after normalization."
        return recommendation

    change_fields = {change.field for change in recommendation.changes}
    disallowed_auto_fields = change_fields - AUTO_UPDATE_ALLOWED_FIELDS
    if disallowed_auto_fields:
        recommendation.recommended_action = "human_review"
        recommendation.reason = (
            f"Changes include {', '.join(sorted(disallowed_auto_fields))}; "
            "only address and phone may auto-update."
        )
        return recommendation

    if (
        recommendation.overall_confidence >= auto_threshold
        and recommendation.conflict_score <= max_conflict_auto
        and match_evidence.exact_npi_match
        and _has_reliable_support(recommendation.changes)
        and not identity_conflict
        and change_fields.issubset(AUTO_UPDATE_ALLOWED_FIELDS)
    ):
        recommendation.recommended_action = "auto_update"
        sources = ", ".join(recommendation.supporting_sources) or "external sources"
        recommendation.reason = (
            f"Update confirmed by {sources} with exact NPI match and no material source conflict."
        )
        return recommendation

    if recommendation.conflict_score > max_conflict_auto:
        recommendation.recommended_action = "human_review"
        recommendation.reason = (
            "External sources disagree on material fields; conflict exceeds safe auto-update threshold."
        )
        return recommendation

    if no_exact_npi and recommendation.change_detected:
        recommendation.recommended_action = "human_review"
        recommendation.reason = "Changes detected without exact NPI match; manual verification required."
        return recommendation

    if recommendation.overall_confidence >= no_change_threshold and recommendation.conflict_score <= max_conflict_auto:
        recommendation.recommended_action = "no_change"
        recommendation.reason = "Fields are sufficiently aligned; no safe update recommended."
        return recommendation

    if recommendation.overall_confidence >= review_min:
        recommendation.recommended_action = "human_review"
        recommendation.reason = "Moderate confidence with unresolved uncertainty; routed to human review."
        return recommendation

    recommendation.recommended_action = "do_not_update"
    recommendation.reason = "Low confidence and insufficient reliable evidence for update."
    return recommendation


def _overall_confidence(changes: list[FieldChange]) -> float:
    if not changes:
        return 1.0
    return round(sum(c.confidence_score for c in changes) / len(changes), 4)


def _has_reliable_support(changes: list[FieldChange], min_reliability: float = 0.80) -> bool:
    """Return True when all material changes have sufficiently reliable supporting sources."""
    if not changes:
        return False
    for change in changes:
        if not change.supporting_sources:
            return False
        if change.confidence_score < min_reliability:
            return False
    return True
