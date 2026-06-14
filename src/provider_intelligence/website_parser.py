"""Practice website HTML parsing with deterministic-first, optional LLM assist."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from provider_intelligence.llm import LLMClient, LLMGatingContext


@dataclass
class WebsiteParseResult:
    """Parsed practice website fields with confidence and method."""

    fields: dict[str, str]
    confidence: float
    method: str
    evidence_snippets: list[str]


_PHONE_RE = re.compile(r"(\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})")
_ZIP_RE = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")
_STATE_RE = re.compile(r"\b([A-Z]{2})\b")


def _text_lines(soup: BeautifulSoup) -> list[str]:
    return [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]


def _extract_phone(text: str) -> str | None:
    match = _PHONE_RE.search(text)
    return match.group(1) if match else None


def _extract_address_block(lines: list[str]) -> dict[str, str]:
    """Heuristically extract address components from visible text lines."""
    address: dict[str, str] = {}
    for line in lines:
        lower = line.lower()
        if "phone" in lower or "fax" in lower or "email" in lower:
            continue
        if re.search(r"\d+\s+\w+", line) and any(
            token in lower for token in ("st", "street", "ave", "road", "rd", "blvd", "drive", "dr")
        ):
            address.setdefault("address_line_1", line)
        zip_match = _ZIP_RE.search(line)
        if zip_match and "city" not in address:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                address["city"] = parts[0]
                state_zip = parts[-1].strip()
                state_match = _STATE_RE.search(state_zip)
                if state_match:
                    address["state"] = state_match.group(1)
                address["zip_code"] = zip_match.group(1)
    return address


def parse_html_deterministic(html: str) -> WebsiteParseResult:
    """Parse practice website HTML using BeautifulSoup heuristics."""
    soup = BeautifulSoup(html, "html.parser")
    fields: dict[str, str] = {}
    evidence: list[str] = []

    title = soup.find("title")
    if title and title.get_text(strip=True):
        fields["practice_name"] = title.get_text(strip=True)
        evidence.append(fields["practice_name"])

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        fields.setdefault("practice_name", h1.get_text(strip=True))
        evidence.append(h1.get_text(strip=True))

    lines = _text_lines(soup)
    full_text = " ".join(lines)

    phone = _extract_phone(full_text)
    if phone:
        fields["phone"] = phone
        evidence.append(phone)

    address_fields = _extract_address_block(lines)
    fields.update({k: v for k, v in address_fields.items() if v})

    specialty_el = soup.find(class_=re.compile(r"specialty|service", re.I))
    if specialty_el:
        fields["specialty"] = specialty_el.get_text(strip=True)
        evidence.append(fields["specialty"])

    found = sum(1 for key in ("practice_name", "phone", "address_line_1", "city") if fields.get(key))
    confidence = round(min(0.95, 0.35 + found * 0.15), 2)

    return WebsiteParseResult(
        fields=fields,
        confidence=confidence,
        method="deterministic",
        evidence_snippets=evidence[:5],
    )


def parse_practice_page(
    html: str,
    *,
    provider_id: str = "WEB_SAMPLE",
    risk_score: float = 0.0,
    conflict_score: float = 0.0,
    llm_client: LLMClient | None = None,
    audit_path: Path | None = None,
) -> WebsiteParseResult:
    """Parse practice page; optionally enrich with gated LLM when confidence is low."""
    deterministic = parse_html_deterministic(html)
    if deterministic.confidence >= 0.60:
        return deterministic

    client = llm_client or LLMClient()
    context = LLMGatingContext(
        provider_id=provider_id,
        risk_score=max(risk_score, 0.70),
        conflict_score=conflict_score,
        use_case="website_extraction",
        parser_confidence=deterministic.confidence,
    )

    llm_result = client.extract(
        "website_extraction",
        context,
        prompt=json.dumps(
            {
                "html_excerpt": html[:4000],
                "deterministic_fields": deterministic.fields,
            }
        ),
        deterministic_fields=deterministic.fields,
        audit_path=audit_path,
    )

    merged = {**deterministic.fields, **{k: str(v) for k, v in llm_result.extracted_fields.items() if v}}
    method = "llm_assisted" if llm_result.confidence_hint > deterministic.confidence else "deterministic"
    confidence = max(deterministic.confidence, min(llm_result.confidence_hint, 0.85))

    return WebsiteParseResult(
        fields=merged,
        confidence=round(confidence, 2),
        method=method,
        evidence_snippets=(llm_result.evidence_snippets or deterministic.evidence_snippets)[:5],
    )


def parse_practice_file(
    path: Path,
    **kwargs: Any,
) -> WebsiteParseResult:
    """Load and parse a local practice website HTML file."""
    html = path.read_text(encoding="utf-8")
    return parse_practice_page(html, **kwargs)

