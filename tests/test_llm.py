"""Tests for adaptive LLM module — no network, LLM_MODE=off."""

from __future__ import annotations

from pathlib import Path

import pytest

from provider_intelligence.config import load_config
from provider_intelligence.llm import (
    LLMClient,
    LLMGatingContext,
    LLMGatingBudget,
    _parse_json_payload,
    validate_llm_result,
)
from provider_intelligence.website_parser import parse_html_deterministic, parse_practice_file


@pytest.fixture(autouse=True)
def llm_off_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force offline deterministic mode for all tests."""
    monkeypatch.setenv("LLM_MODE", "off")


def test_llm_mode_off_never_calls_network(tmp_path: Path) -> None:
    client = LLMClient()
    assert client.mode == "off"
    assert not client._can_call_network()

    context = LLMGatingContext(
        provider_id="HL_001",
        risk_score=0.95,
        conflict_score=0.80,
        use_case="conflict_explanation",
    )
    should_call, reason = client.should_use_llm(context)
    assert should_call is False
    assert reason == "llm_mode_off"

    audit_path = tmp_path / "audit_llm_calls.csv"
    result = client.extract(
        "conflict_explanation",
        context,
        prompt="{}",
        sources=["nppes"],
        audit_path=audit_path,
    )
    assert result.reasoning_summary
    assert audit_path.exists()
    rows = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert "attempted" in rows[0]
    assert "success" in rows[0]


def test_gating_budget_caps_calls() -> None:
    budget = LLMGatingBudget(max_share=0.08, total_records=60)
    assert budget.max_calls == 4
    for _ in range(budget.max_calls):
        budget.record_call()
    assert budget.remaining_calls == 0


def _client_with_mode(mode: str) -> LLMClient:
    """Build an LLMClient with an explicit mode (avoids env cache issues in tests)."""
    config = load_config()
    config = dict(config)
    config["llm"] = dict(config["llm"])
    config["llm"]["mode"] = mode
    config["llm"]["api"] = {"base_url": "", "api_key": "", "model": ""}
    return LLMClient(config=config)


def test_gating_risk_threshold_in_auto_mode() -> None:
    client = _client_with_mode("auto")
    client.reset_budget(total_records=100)

    low = LLMGatingContext(
        provider_id="HL_LOW",
        risk_score=0.20,
        conflict_score=0.10,
        use_case="reviewer_summary",
    )
    should_call, reason = client.should_use_llm(low)
    assert should_call is False
    assert reason in {"below_gating_thresholds", "credentials_unavailable"}

    high = LLMGatingContext(
        provider_id="HL_HIGH",
        risk_score=0.85,
        conflict_score=0.10,
        use_case="reviewer_summary",
    )
    passes, reason = client.passes_gating_rules(high)
    assert passes is True
    assert "risk_threshold" in reason


def test_validate_llm_result_accepts_valid_payload() -> None:
    payload = {
        "extracted_fields": {"phone": "3055550142"},
        "reasoning_summary": "Found phone in contact section.",
        "confidence_hint": 0.72,
        "evidence_snippets": ["Phone: (305) 555-0142"],
        "recommended_review_note": None,
    }
    result = validate_llm_result(payload)
    assert result is not None
    assert result.extracted_fields["phone"] == "3055550142"


def test_validate_llm_result_rejects_invalid_payload() -> None:
    assert validate_llm_result(None) is None
    assert validate_llm_result({"confidence_hint": "not-a-float"}) is None


def test_parse_json_payload_extracts_embedded_json() -> None:
    text = 'Here is the result:\n{"reasoning_summary": "ok", "confidence_hint": 0.5}'
    parsed = _parse_json_payload(text)
    assert parsed is not None
    assert parsed["reasoning_summary"] == "ok"


def test_deterministic_fallback_for_specialty() -> None:
    client = LLMClient()
    context = LLMGatingContext(
        provider_id="HL_002",
        risk_score=0.5,
        conflict_score=0.2,
        use_case="specialty_fallback",
        specialty_confidence=0.4,
    )
    result = client.deterministic_fallback(
        "specialty_fallback",
        context,
        specialty="Family Medicine",
    )
    assert "specialty" in result.extracted_fields or result.reasoning_summary


def test_llm_never_changes_decision_engine_contract() -> None:
    """LLM enrichment must not expose auto-update approval helpers."""
    client = LLMClient()
    context = LLMGatingContext(
        provider_id="HL_003",
        risk_score=0.99,
        conflict_score=0.99,
        use_case="reviewer_summary",
    )
    result = client.extract("reviewer_summary", context, prompt="{}")
    assert "recommended_action" not in result.extracted_fields
    assert result.extracted_fields.get("recommended_action") is None


def test_website_parser_deterministic_no_network() -> None:
    sample = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "practice_websites"
        / "sample_practice_page.html"
    )
    result = parse_practice_file(sample)
    assert result.method == "deterministic"
    assert result.fields.get("practice_name")
    assert result.fields.get("phone")
    assert result.confidence >= 0.60


def test_website_parser_html_string() -> None:
    html = "<html><head><title>Test Clinic</title></head><body><p>100 Main St</p></body></html>"
    result = parse_html_deterministic(html)
    assert result.fields.get("practice_name") == "Test Clinic"


def test_force_mode_without_credentials_falls_back(tmp_path: Path) -> None:
    client = _client_with_mode("force")
    context = LLMGatingContext(
        provider_id="HL_004",
        risk_score=0.9,
        conflict_score=0.5,
        use_case="source_evidence_summary",
    )
    audit_path = tmp_path / "audit.csv"
    result = client.extract(
        "source_evidence_summary",
        context,
        prompt="{}",
        sources=["nppes", "cms"],
        audit_path=audit_path,
    )
    assert result.reasoning_summary
    assert client.mode == "force"
