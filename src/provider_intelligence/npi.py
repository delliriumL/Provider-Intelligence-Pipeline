"""NPI validation using format checks and Luhn algorithm with 80840 prefix."""

from __future__ import annotations

import re


def clean_npi(npi: str | None) -> str | None:
    """Strip non-digits and return a 10-digit NPI string or None."""
    if not npi:
        return None
    digits = re.sub(r"\D", "", str(npi))
    if len(digits) != 10:
        return None
    return digits


def is_valid_npi_format(npi: str | None) -> bool:
    """Return True when the NPI is exactly 10 digits."""
    cleaned = clean_npi(npi)
    return cleaned is not None


def _luhn_check_digit(digits: str) -> int:
    """Compute Luhn check digit for a digit string without its check digit."""
    total = 0
    reverse = digits[::-1]
    for index, char in enumerate(reverse):
        value = int(char)
        if index % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return (10 - (total % 10)) % 10


def is_valid_npi_luhn(npi: str | None) -> bool:
    """Validate NPI check digit using CMS Luhn algorithm with 80840 prefix."""
    cleaned = clean_npi(npi)
    if cleaned is None:
        return False
    prefix = "80840"
    base = prefix + cleaned[:9]
    expected = _luhn_check_digit(base)
    return expected == int(cleaned[9])


def npi_quality_score(npi: str | None) -> float:
    """Return a quality score between 0.0 and 1.0 for an NPI value."""
    if not npi:
        return 0.0
    if not is_valid_npi_format(npi):
        return 0.2
    if not is_valid_npi_luhn(npi):
        return 0.4
    return 1.0


def generate_valid_npi(seed_digits: str) -> str:
    """Generate a valid 10-digit NPI from 9 seed digits (for synthetic data)."""
    base = seed_digits[:9].zfill(9)
    check = _luhn_check_digit("80840" + base)
    return base + str(check)
