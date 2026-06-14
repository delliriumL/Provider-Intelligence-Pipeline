"""Tests for dashboard data adapters (no Streamlit runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from adapters import (  # noqa: E402
    format_percent,
    normalize_confidence_fraction,
    normalize_sources,
    parse_changes,
    safe_get_action,
    safe_get_confidence,
)


def test_normalize_sources_list_of_strings() -> None:
    result = normalize_sources(["NPPES", "CMS Doctors & Clinicians"])
    assert len(result) == 2
    assert result[0]["source"] in {"NPI Registry", "NPPES"}


def test_normalize_sources_pipe_string() -> None:
    result = normalize_sources("CMS Doctors & Clinicians|NPPES")
    assert len(result) == 2


def test_normalize_sources_dict() -> None:
    result = normalize_sources([{"source": "NPPES", "reliability": 0.85}])
    assert result[0]["source"] == "NPI Registry"
    assert result[0]["reliability"] == 0.85


def test_parse_changes_json_string() -> None:
    raw = '[{"field": "phone", "old_value": "1", "new_value": "2", "confidence_score": 0.9}]'
    changes = parse_changes(raw)
    assert len(changes) == 1
    assert changes[0]["field"] == "phone"


def test_parse_changes_list() -> None:
    changes = parse_changes([{"field": "address", "old_value": "a", "new_value": "b"}])
    assert changes[0]["field"] == "address"


def test_format_percent_fraction_and_whole() -> None:
    assert format_percent(0.88) == "88.0%"
    assert format_percent(88) == "88.0%"
    assert normalize_confidence_fraction(92.9) == 0.929
    assert normalize_confidence_fraction(0.929) == 0.929


def test_safe_get_action_and_confidence() -> None:
    assert safe_get_action({"recommended_action": "human_review"}) == "human_review"
    assert safe_get_action({"decision": "auto_update"}) == "auto_update"
    assert safe_get_confidence({"overall_confidence": 0.9}) == 0.9
    assert safe_get_confidence({"confidence_score": 0.85}) == 0.85


def test_describe_llm_run_status_gated_not_error() -> None:
    from adapters import describe_llm_run_status

    status = describe_llm_run_status(
        {
            "calls_attempted": 0,
            "calls_failed": 0,
            "calls_succeeded": 0,
            "credentials_status": "present",
            "eligible_records": 0,
            "last_error_type": "deterministic_confidence_sufficient",
        }
    )
    assert "not required" in status["status"]
    assert not status["show_last_error"]


def test_describe_llm_run_status_enrichment_used() -> None:
    from adapters import describe_llm_run_status

    status = describe_llm_run_status(
        {
            "calls_attempted": 3,
            "calls_failed": 0,
            "calls_succeeded": 3,
            "credentials_status": "present",
            "eligible_records": 4,
        }
    )
    assert "bounded enrichment" in status["status"]


def test_build_source_evidence_text_fallback() -> None:
    from adapters import build_source_evidence_text

    record = {
        "npi": "1234567893",
        "reason": "source conflict on address",
        "conflict_score": 0.8,
        "changes": [{"field": "address", "old_value": "A", "new_value": "B", "conflict_detected": True}],
    }
    text = build_source_evidence_text(record, {"source": "NPI Registry", "field": "address"})
    assert "conflict" in text.lower()


def test_build_full_audit_event_text_no_truncation() -> None:
    from adapters import build_full_audit_event_text

    row = {
        "timestamp": "2026-06-14T12:00:00+00:00",
        "output_summary": "risk_score=0.82 with detailed explanation that should not be truncated",
        "input_summary": "provider HL_002 matched against NPPES and CMS",
    }
    text = build_full_audit_event_text(row)
    assert "detailed explanation" in text
    assert "NPPES" in text
