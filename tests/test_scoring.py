"""Tests for scoring functions."""

from datetime import date, timedelta

from provider_intelligence.npi import generate_valid_npi
from provider_intelligence.schemas import ExternalSourceRecord, MatchEvidence, ProviderRecord
from provider_intelligence.scoring import (
    compute_conflict_score,
    compute_field_confidence,
    compute_risk_score,
    compute_verification_age_risk,
)

VALID_NPI = generate_valid_npi("146756000")


def _provider(**kwargs) -> ProviderRecord:
    defaults = {
        "provider_id": "HL_001",
        "provider_name": "Jane Doe",
        "npi": VALID_NPI,
        "specialty": "Family Medicine",
        "taxonomy_code": "207Q00000X",
        "practice_name": "Sunrise Medical Group",
        "address_line_1": "123 Main St",
        "city": "Miami",
        "state": "FL",
        "zip_code": "33101",
        "phone": "+13055551234",
        "active_status": "active",
        "last_verified_date": date.today() - timedelta(days=30),
    }
    defaults.update(kwargs)
    return ProviderRecord(**defaults)


def _source(**kwargs) -> ExternalSourceRecord:
    defaults = {
        "source_name": "NPPES",
        "source_record_id": "NPPES_001",
        "provider_name": "Jane Doe",
        "npi": VALID_NPI,
        "specialty": "Family Medicine",
        "taxonomy_code": "207Q00000X",
        "practice_name": "Sunrise Medical Group",
        "address_line_1": "123 Main St",
        "city": "Miami",
        "state": "FL",
        "zip_code": "33101",
        "phone": "+13055551234",
        "active_status": "active",
        "last_update_date": date.today(),
        "source_reliability_by_field": {
            "identity": 0.95,
            "address": 0.80,
            "phone": 0.80,
            "specialty": 0.85,
            "status": 0.95,
        },
    }
    defaults.update(kwargs)
    return ExternalSourceRecord(**defaults)


def test_old_verification_date_increases_risk():
    stale = compute_verification_age_risk(date.today() - timedelta(days=800))
    fresh = compute_verification_age_risk(date.today() - timedelta(days=30))
    assert stale > fresh
    assert stale == 1.0


def test_exact_npi_match_increases_confidence():
    provider = _provider()
    source = _source()
    evidence = MatchEvidence(match_type="exact_npi", match_score=1.0, exact_npi_match=True, name_similarity=1.0)
    result = compute_field_confidence(
        field="address",
        provider=provider,
        proposed_value=source.address_line_1,
        supporting_sources=[source],
        match_evidence=evidence,
        normalized_similarity=1.0,
    )
    assert result.components["exact_identifier_match"] == 1.0
    assert result.score >= 0.8


def test_conflict_score_increases_with_address_mismatch():
    provider = _provider(address_line_1="123 Main St")
    source_a = _source(source_name="NPPES", address_line_1="999 Other Rd")
    source_b = _source(source_name="CMS Doctors & Clinicians", address_line_1="888 Conflict Ave")
    evidence = MatchEvidence(match_type="exact_npi", match_score=1.0, exact_npi_match=True)
    conflict = compute_conflict_score(provider, [source_a, source_b], evidence)
    assert conflict.score > 0.15
    assert conflict.components["address_conflict"] > 0


def test_risk_score_returns_components():
    provider = _provider(last_verified_date=date.today() - timedelta(days=900))
    source = _source()
    result = compute_risk_score(provider, [source], duplicate_risk=0.0, primary_match=source)
    assert "verification_age_risk" in result.components
    assert result.score > 0.2
