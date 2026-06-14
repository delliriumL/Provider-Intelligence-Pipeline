"""Field normalization utilities — raw values are preserved separately."""

from __future__ import annotations

import re
from dataclasses import dataclass

import phonenumbers
from rapidfuzz import fuzz

from provider_intelligence.taxonomy import TaxonomyLookup

STREET_SUFFIXES = {
    "st": "street",
    "st.": "street",
    "street": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "avenue": "avenue",
    "rd": "road",
    "rd.": "road",
    "road": "road",
    "dr": "drive",
    "dr.": "drive",
    "drive": "drive",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "boulevard": "boulevard",
}

UNIT_SUFFIXES = {
    "ste": "suite",
    "ste.": "suite",
    "suite": "suite",
    "apt": "apartment",
    "apt.": "apartment",
    "apartment": "apartment",
    "fl": "floor",
    "fl.": "floor",
    "floor": "floor",
}

LEGAL_SUFFIXES = {"llc", "pllc", "pa", "inc", "ltd", "corp", "corporation"}

CREDENTIAL_PATTERN = re.compile(
    r"\b(md|do|np|pa-c|rn|phd|dds|dmd|dpm|od|pharmd|mph|ms|mba)\b",
    re.IGNORECASE,
)


@dataclass
class NormalizedAddress:
    """Normalized address components."""

    normalized_address_line: str
    normalized_city: str
    normalized_state: str
    normalized_zip5: str | None
    normalized_suite: str | None = None


def normalize_name(value: str | None) -> str:
    """Normalize provider name for comparison (credentials and punctuation removed)."""
    if not value:
        return ""
    text = value.lower()
    text = CREDENTIAL_PATTERN.sub("", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_phone(value: str | None) -> str | None:
    """Normalize US phone numbers to +1XXXXXXXXXX or return None if invalid."""
    if not value:
        return None
    try:
        parsed = phonenumbers.parse(value, "US")
        if not phonenumbers.is_valid_number(parsed):
            return None
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        digits = re.sub(r"\D", "", value)
        if len(digits) == 10:
            candidate = f"+1{digits}"
            return normalize_phone(candidate)
        if len(digits) == 11 and digits.startswith("1"):
            return normalize_phone(f"+{digits}")
        return None


def normalize_zip(value: str | None) -> str | None:
    """Return 5-digit ZIP code for matching."""
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) < 5:
        return None
    return digits[:5]


def _normalize_token(token: str, mapping: dict[str, str]) -> str:
    lower = token.lower().rstrip(".")
    return mapping.get(lower, lower)


def normalize_address_parts(
    address_line_1: str | None,
    address_line_2: str | None = None,
    city: str | None = None,
    state: str | None = None,
    zip_code: str | None = None,
) -> NormalizedAddress:
    """Normalize address components for matching."""
    line1 = (address_line_1 or "").strip()
    line2 = (address_line_2 or "").strip()
    suite: str | None = None

    if line2:
        tokens = line2.split()
        normalized_tokens = [_normalize_token(tok, UNIT_SUFFIXES) for tok in tokens]
        suite = " ".join(normalized_tokens)

    tokens = line1.split()
    normalized_tokens: list[str] = []
    for token in tokens:
        mapped = _normalize_token(token, STREET_SUFFIXES)
        normalized_tokens.append(mapped)
    normalized_line = " ".join(normalized_tokens).lower()

    normalized_city = (city or "").strip().lower()
    normalized_state = (state or "").strip().upper()
    normalized_zip5 = normalize_zip(zip_code)

    return NormalizedAddress(
        normalized_address_line=normalized_line,
        normalized_city=normalized_city,
        normalized_state=normalized_state,
        normalized_zip5=normalized_zip5,
        normalized_suite=suite,
    )


def normalize_practice_name(value: str | None) -> str:
    """Normalize practice name by removing legal suffixes and extra punctuation."""
    if not value:
        return ""
    text = value.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [tok for tok in text.split() if tok not in LEGAL_SUFFIXES]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def normalize_specialty(
    value: str | None,
    taxonomy_code: str | None = None,
    taxonomy_lookup: TaxonomyLookup | None = None,
) -> str:
    """Normalize specialty using taxonomy code or fuzzy taxonomy description match."""
    lookup = taxonomy_lookup or TaxonomyLookup()
    if taxonomy_code:
        description = lookup.get_description(taxonomy_code)
        if description:
            return description.lower()
    if not value:
        return ""
    if taxonomy_code and lookup.is_valid_code(taxonomy_code):
        return lookup.get_description(taxonomy_code).lower()
    best = lookup.fuzzy_match_description(value)
    return best.lower() if best else value.strip().lower()


def address_similarity(addr_a: NormalizedAddress, addr_b: NormalizedAddress) -> float:
    """Compute fuzzy similarity between two normalized addresses."""
    line_score = fuzz.token_sort_ratio(addr_a.normalized_address_line, addr_b.normalized_address_line) / 100.0
    city_match = 1.0 if addr_a.normalized_city == addr_b.normalized_city else 0.0
    state_match = 1.0 if addr_a.normalized_state == addr_b.normalized_state else 0.0
    zip_match = (
        1.0
        if addr_a.normalized_zip5 and addr_a.normalized_zip5 == addr_b.normalized_zip5
        else 0.0
    )
    return 0.5 * line_score + 0.2 * city_match + 0.15 * state_match + 0.15 * zip_match
