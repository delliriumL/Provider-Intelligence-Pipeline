"""Tests for conservative decision engine."""

from provider_intelligence.decision_engine import decide_action
from provider_intelligence.npi import generate_valid_npi
from provider_intelligence.schemas import FieldChange, MatchEvidence, Recommendation, ScoreResult

VALID_NPI = generate_valid_npi("146756000")


def _recommendation(**kwargs) -> Recommendation:
    defaults = {
        "provider_id": "HL_001",
        "npi": VALID_NPI,
        "change_detected": True,
        "changes": [
            FieldChange(
                field="address",
                old_value="123 Main St",
                new_value="456 Oak Ave",
                confidence_score=0.92,
                supporting_sources=["NPPES", "CMS Doctors & Clinicians"],
            )
        ],
        "supporting_sources": ["NPPES", "CMS Doctors & Clinicians"],
        "audit_id": "AUD_001",
    }
    defaults.update(kwargs)
    return Recommendation(**defaults)


def test_high_confidence_exact_npi_auto_update():
    rec = _recommendation()
    evidence = MatchEvidence(match_type="exact_npi", match_score=1.0, exact_npi_match=True)
    risk = ScoreResult(score=0.3, components={})
    conflict = ScoreResult(score=0.05, components={})
    result = decide_action(rec, evidence, risk, conflict, duplicate_risk=0.0)
    assert result.recommended_action == "auto_update"


def test_practice_name_change_routes_to_human_review():
    rec = _recommendation(
        changes=[
            FieldChange(
                field="practice_name",
                old_value="Metro Family Clinic PLLC",
                new_value="Metro Family Clinic",
                confidence_score=0.93,
                supporting_sources=["NPPES", "CMS Doctors & Clinicians"],
            )
        ]
    )
    evidence = MatchEvidence(match_type="exact_npi", match_score=1.0, exact_npi_match=True)
    risk = ScoreResult(score=0.2, components={})
    conflict = ScoreResult(score=0.0, components={})
    result = decide_action(rec, evidence, risk, conflict, duplicate_risk=0.0)
    assert result.recommended_action == "human_review"
    assert "practice_name" in result.reason


def test_conflicting_address_human_review():
    rec = _recommendation(
        changes=[
            FieldChange(
                field="address",
                old_value="123 Main St",
                new_value="999 Conflict Rd",
                confidence_score=0.7,
                supporting_sources=["NPPES"],
                conflict_detected=True,
            )
        ]
    )
    evidence = MatchEvidence(match_type="exact_npi", match_score=1.0, exact_npi_match=True)
    risk = ScoreResult(score=0.5, components={})
    conflict = ScoreResult(score=0.40, components={"address_conflict": 0.35})
    result = decide_action(rec, evidence, risk, conflict, duplicate_risk=0.0)
    assert result.recommended_action == "human_review"


def test_invalid_identity_do_not_update():
    rec = _recommendation(npi="1234567890", change_detected=True)
    evidence = MatchEvidence(match_type="weak", match_score=0.2, exact_npi_match=False, identity_conflict=False)
    risk = ScoreResult(score=0.9, components={})
    conflict = ScoreResult(score=0.0, components={})
    result = decide_action(rec, evidence, risk, conflict, duplicate_risk=0.0)
    assert result.recommended_action == "do_not_update"


def test_unchanged_fields_no_change():
    rec = _recommendation(change_detected=False, changes=[])
    evidence = MatchEvidence(match_type="exact_npi", match_score=1.0, exact_npi_match=True)
    risk = ScoreResult(score=0.1, components={})
    conflict = ScoreResult(score=0.0, components={})
    result = decide_action(rec, evidence, risk, conflict, duplicate_risk=0.0)
    assert result.recommended_action == "no_change"
