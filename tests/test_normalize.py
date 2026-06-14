"""Tests for field normalization."""

from provider_intelligence.normalize import (
    normalize_address_parts,
    normalize_name,
    normalize_phone,
    normalize_practice_name,
    normalize_specialty,
    normalize_zip,
)
from provider_intelligence.taxonomy import TaxonomyLookup


def test_normalize_name_removes_credentials():
    assert normalize_name("Jane Doe, MD") == "jane doe"
    assert normalize_name("John Smith DO") == "john smith"


def test_normalize_phone_us_format():
    assert normalize_phone("(305) 555-1234") == "+13055551234"
    assert normalize_phone("invalid") is None


def test_normalize_zip():
    assert normalize_zip("33101-1234") == "33101"
    assert normalize_zip("abc") is None


def test_normalize_address_abbreviations():
    addr = normalize_address_parts("123 Main St", "Ste 200", "Miami", "FL", "33101")
    assert "street" in addr.normalized_address_line
    assert addr.normalized_suite == "suite 200"
    assert addr.normalized_state == "FL"
    assert addr.normalized_zip5 == "33101"


def test_normalize_practice_name():
    assert normalize_practice_name("Sunrise Medical Group LLC") == "sunrise medical group"


def test_normalize_specialty_with_taxonomy():
    lookup = TaxonomyLookup()
    result = normalize_specialty("Family Doc", taxonomy_code="207Q00000X", taxonomy_lookup=lookup)
    assert "family medicine" in result
