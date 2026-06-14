"""Provider and practice matching with duplicate detection."""

from __future__ import annotations

from collections import defaultdict

from rapidfuzz import fuzz

from provider_intelligence.config import load_config
from provider_intelligence.normalize import (
    address_similarity,
    normalize_address_parts,
    normalize_name,
    normalize_phone,
    normalize_practice_name,
    normalize_specialty,
)
from provider_intelligence.npi import clean_npi
from provider_intelligence.schemas import ExternalSourceRecord, MatchEvidence, ProviderRecord


def _field_value(record: ProviderRecord | ExternalSourceRecord, field: str) -> str | None:
    mapping = {
        "address": record.address_line_1,
        "phone": record.phone,
        "practice_name": record.practice_name,
        "specialty": record.specialty,
        "active_status": record.active_status,
    }
    return mapping.get(field)


def match_provider_to_sources(
    provider: ProviderRecord,
    sources: list[ExternalSourceRecord],
) -> tuple[ExternalSourceRecord | None, MatchEvidence]:
    """Match a provider to the best external source using NPI-first logic."""
    config = load_config()
    name_threshold = config["app"]["matching"]["name_similarity_threshold"]

    provider_npi = clean_npi(provider.npi)
    exact_matches = [s for s in sources if clean_npi(s.npi) == provider_npi and provider_npi]
    if exact_matches:
        best = exact_matches[0]
        evidence = _build_evidence(provider, best, exact_npi_match=True)
        evidence.match_type = "exact_npi"
        evidence.match_score = 1.0
        return best, evidence

    best_source: ExternalSourceRecord | None = None
    best_score = 0.0
    best_evidence = MatchEvidence(match_type="none", match_score=0.0)

    for source in sources:
        evidence = _build_evidence(provider, source, exact_npi_match=False)
        identity_score = evidence.name_similarity
        practice_score = fuzz.token_sort_ratio(
            normalize_practice_name(provider.practice_name),
            normalize_practice_name(source.practice_name),
        ) / 100.0
        city_match = 1.0 if (provider.city or "").lower() == (source.city or "").lower() else 0.0
        state_match = 1.0 if (provider.state or "").upper() == (source.state or "").upper() else 0.0
        composite = 0.45 * identity_score + 0.25 * practice_score + 0.15 * city_match + 0.15 * state_match
        evidence.match_score = composite
        evidence.match_type = "fuzzy_identity" if composite * 100 >= name_threshold else "weak"

        if provider_npi and source.npi and clean_npi(source.npi) != provider_npi:
            evidence.identity_conflict = True

        if composite > best_score:
            best_score = composite
            best_source = source
            best_evidence = evidence

    return best_source, best_evidence


def _build_evidence(
    provider: ProviderRecord,
    source: ExternalSourceRecord,
    exact_npi_match: bool,
) -> MatchEvidence:
    """Build match evidence between provider and source."""
    name_sim = fuzz.token_sort_ratio(normalize_name(provider.provider_name), normalize_name(source.provider_name)) / 100.0
    addr_a = normalize_address_parts(
        provider.address_line_1, provider.address_line_2, provider.city, provider.state, provider.zip_code
    )
    addr_b = normalize_address_parts(
        source.address_line_1, source.address_line_2, source.city, source.state, source.zip_code
    )
    address_sim = address_similarity(addr_a, addr_b)
    phone_match = normalize_phone(provider.phone) == normalize_phone(source.phone) and bool(provider.phone)
    specialty_sim = fuzz.token_sort_ratio(
        normalize_specialty(provider.specialty, provider.taxonomy_code),
        normalize_specialty(source.specialty, source.taxonomy_code),
    ) / 100.0

    return MatchEvidence(
        match_type="exact_npi" if exact_npi_match else "fuzzy",
        match_score=1.0 if exact_npi_match else name_sim,
        matched_source=source.source_name,
        exact_npi_match=exact_npi_match,
        name_similarity=name_sim,
        address_similarity=address_sim,
        phone_match=phone_match,
        specialty_similarity=specialty_sim,
        identity_conflict=False,
    )


def detect_duplicates(providers: list[ProviderRecord]) -> dict[str, float]:
    """Detect duplicate risk per provider_id."""
    config = load_config()
    duplicate_threshold = config["app"]["matching"]["duplicate_risk_threshold"]
    risks: dict[str, float] = {p.provider_id: 0.0 for p in providers}

    npi_map: dict[str, list[str]] = defaultdict(list)
    for provider in providers:
        npi = clean_npi(provider.npi)
        if npi:
            npi_map[npi].append(provider.provider_id)

    for provider in providers:
        risk = 0.0
        npi = clean_npi(provider.npi)
        if npi and len(npi_map[npi]) > 1:
            risk = max(risk, 0.95)

        for other in providers:
            if other.provider_id == provider.provider_id:
                continue
            name_sim = fuzz.token_sort_ratio(
                normalize_name(provider.provider_name), normalize_name(other.provider_name)
            )
            phone_match = normalize_phone(provider.phone) == normalize_phone(other.phone) and bool(provider.phone)
            addr_a = normalize_address_parts(
                provider.address_line_1, provider.address_line_2, provider.city, provider.state, provider.zip_code
            )
            addr_b = normalize_address_parts(
                other.address_line_1, other.address_line_2, other.city, other.state, other.zip_code
            )
            address_sim = address_similarity(addr_a, addr_b)

            if name_sim >= 90 and phone_match:
                risk = max(risk, 0.85)
            if name_sim >= 90 and address_sim >= 0.9:
                risk = max(risk, 0.80)
            if (
                name_sim >= 85
                and address_sim >= 0.85
                and phone_match
                and npi
                and clean_npi(other.npi) not in {None, npi}
            ):
                risk = max(risk, 0.90)

        risks[provider.provider_id] = risk if risk >= duplicate_threshold else risk

    return risks


def find_all_source_matches(
    provider: ProviderRecord,
    nppes_sources: list[ExternalSourceRecord],
    cms_sources: list[ExternalSourceRecord],
) -> list[tuple[ExternalSourceRecord, MatchEvidence]]:
    """Return all source matches for a provider."""
    matches: list[tuple[ExternalSourceRecord, MatchEvidence]] = []
    for source in nppes_sources + cms_sources:
        provider_npi = clean_npi(provider.npi)
        source_npi = clean_npi(source.npi)
        if provider_npi and source_npi == provider_npi:
            evidence = _build_evidence(provider, source, exact_npi_match=True)
            evidence.match_type = "exact_npi"
            evidence.match_score = 1.0
            matches.append((source, evidence))
        elif not provider_npi:
            evidence = _build_evidence(provider, source, exact_npi_match=False)
            if evidence.name_similarity >= 0.75:
                evidence.match_type = "fuzzy_identity"
                evidence.match_score = evidence.name_similarity
                matches.append((source, evidence))
    return matches
