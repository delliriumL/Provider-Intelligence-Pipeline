"""Tests for NPI validation."""

from provider_intelligence.npi import (
    clean_npi,
    generate_valid_npi,
    is_valid_npi_format,
    is_valid_npi_luhn,
    npi_quality_score,
)


def test_clean_npi_strips_non_digits():
    assert clean_npi("123-456-7890") == "1234567890"


def test_valid_npi_format():
    assert is_valid_npi_format("1234567890") is True
    assert is_valid_npi_format("12345") is False
    assert is_valid_npi_format(None) is False


def test_valid_npi_luhn_known_valid():
    valid_npi = generate_valid_npi("146756000")
    assert is_valid_npi_luhn(valid_npi) is True


def test_invalid_npi_check_digit():
    valid_npi = generate_valid_npi("146756000")
    invalid = valid_npi[:-1] + ("0" if valid_npi[-1] != "0" else "1")
    assert is_valid_npi_luhn(invalid) is False


def test_npi_quality_score():
    valid_npi = generate_valid_npi("123456789")
    assert npi_quality_score(valid_npi) == 1.0
    assert npi_quality_score("123") == 0.2
    assert npi_quality_score(None) == 0.0
