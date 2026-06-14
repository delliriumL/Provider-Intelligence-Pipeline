"""Tests for bounded LLM enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from provider_intelligence.llm import LLMClient, LLMGatingContext
from provider_intelligence.llm_enrichment import LLM_DEMO_TARGETS, run_llm_enrichment
from provider_intelligence.schemas import ProviderRecord, Recommendation


def _client_auto(tmp_path: Path) -> LLMClient:
    from provider_intelligence.config import load_config

    config = load_config()
    config = dict(config)
    config["llm"] = dict(config["llm"])
    config["llm"]["mode"] = "auto"
    config["llm"]["api"] = {
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
        "model": "test-model",
    }
    return LLMClient(config=config)


def test_llm_demo_targets_cover_four_cases() -> None:
    assert set(LLM_DEMO_TARGETS.keys()) == {"HL_002", "HL_003", "HL_004", "HL_005"}
    assert LLM_DEMO_TARGETS["HL_003"] == "website_extraction"


def test_run_llm_enrichment_attempts_bounded_demo_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recs = [
        Recommendation(
            provider_id=pid,
            recommended_action="human_review",
            reason="source conflict on address",
            conflict_score=0.8 if pid == "HL_002" else 0.1,
            risk_score=0.85 if pid == "HL_005" else 0.2,
        )
        for pid in LLM_DEMO_TARGETS
    ]
    recs.extend(
        Recommendation(provider_id=f"HL_{index:03d}", recommended_action="no_change")
        for index in range(6, 61)
    )
    providers = {
        "HL_003": ProviderRecord(
            provider_id="HL_003",
            provider_name="Alex Chen",
            website="local:messy_demo_page.html",
        ),
        "HL_004": ProviderRecord(
            provider_id="HL_004",
            provider_name="Jordan Patel",
            specialty="Gen Prac / FP",
        ),
        "HL_002": ProviderRecord(provider_id="HL_002", provider_name="Maria Lopez"),
        "HL_005": ProviderRecord(provider_id="HL_005", provider_name="Taylor Nguyen"),
    }

    def fake_call(self, prompt: str, system: str):
        payload = {
            "extracted_fields": {"specialty": "Family Medicine"},
            "reasoning_summary": "Mock LLM enrichment for reviewer context.",
            "confidence_hint": 0.72,
            "evidence_snippets": ["NPPES", "CMS"],
            "recommended_review_note": "Verify against primary sources.",
        }
        return json.dumps(payload), {"prompt_tokens": 100, "completion_tokens": 50}, None

    monkeypatch.setattr(LLMClient, "_call_api", fake_call)

    from provider_intelligence.config import load_config

    config = load_config()
    config = dict(config)
    config["llm"] = dict(config["llm"])
    config["llm"]["mode"] = "auto"
    config["llm"]["api"] = {
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
        "model": "test-model",
    }

    updated, summary = run_llm_enrichment(
        recs,
        providers,
        output_dir=tmp_path,
        config=config,
        enable_live_calls=True,
    )

    assert summary["calls_attempted"] == 4
    assert summary["calls_succeeded"] == 4
    assert summary["eligible_records"] >= 4
    enriched = [r for r in updated if r.llm_enrichment and r.llm_enrichment.enriched]
    assert len(enriched) == 4
    for pid in LLM_DEMO_TARGETS:
        match = next(r for r in updated if r.provider_id == pid)
        assert match.recommended_action == "human_review"
        assert match.llm_enrichment is not None
        assert match.llm_enrichment.enriched
    audit_path = tmp_path / "audit_llm_calls.csv"
    assert audit_path.exists()


def test_passes_gating_human_review_conflict() -> None:
    from provider_intelligence.config import load_config

    config = load_config()
    config = dict(config)
    config["llm"] = dict(config["llm"])
    config["llm"]["mode"] = "auto"
    client = LLMClient(config=config)
    client.reset_budget(60)
    context = LLMGatingContext(
        provider_id="HL_X",
        risk_score=0.2,
        conflict_score=0.1,
        use_case="reviewer_summary",
        recommended_action="human_review",
        reason="source conflict between NPPES and CMS on address",
    )
    passes, reason = client.passes_gating_rules(context)
    assert passes
    assert "human_review_conflict" in reason
