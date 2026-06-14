"""Tests for competition-facing recommendation export."""

from __future__ import annotations

import json
from pathlib import Path

from provider_intelligence.export import to_competition_change, to_competition_recommendation
from provider_intelligence.schemas import FieldChange, Recommendation


def test_competition_recommendation_required_fields() -> None:
    rec = Recommendation(
        provider_id="HL_001",
        npi="1234567893",
        change_detected=True,
        changes=[
            FieldChange(
                field="address",
                old_value="100 Main St",
                new_value="250 Health Park Dr",
                confidence_score=0.92,
                supporting_sources=["NPPES", "cms_doctors_clinicians"],
            )
        ],
        overall_confidence=0.90,
        recommended_action="auto_update",
        reason="Updated address confirmed by multiple reliable sources.",
        audit_id="AUD_TEST",
        risk_score=0.1,
        conflict_score=0.0,
    )
    payload = to_competition_recommendation(rec)
    required = {
        "provider_id",
        "npi",
        "change_detected",
        "changes",
        "overall_confidence",
        "recommended_action",
        "reason",
    }
    assert required.issubset(payload.keys())
    assert payload["recommended_action"] == "auto_update"
    assert 0 <= payload["overall_confidence"] <= 1
    assert isinstance(payload["changes"], list)
    change = payload["changes"][0]
    assert isinstance(change["supporting_sources"], list)
    assert all(isinstance(s, str) for s in change["supporting_sources"])
    assert "NPI Registry" in change["supporting_sources"]


def test_competition_output_file_schema_after_demo() -> None:
    path = Path("outputs/recommendations.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    required = {
        "provider_id",
        "npi",
        "change_detected",
        "changes",
        "overall_confidence",
        "recommended_action",
        "reason",
    }
    for rec in data:
        assert not (required - set(rec.keys())), rec.get("provider_id")
        assert rec["recommended_action"] in {"auto_update", "human_review", "no_change", "do_not_update"}
        assert isinstance(rec["changes"], list)
        assert 0 <= float(rec["overall_confidence"]) <= 1
        for change in rec["changes"]:
            sources = change.get("supporting_sources", [])
            assert isinstance(sources, list)
            assert all(isinstance(s, str) for s in sources)
            score = change.get("confidence_score")
            if score is not None:
                assert 0 <= float(score) <= 1

    hl_001 = next((rec for rec in data if rec.get("provider_id") == "HL_001"), None)
    assert hl_001 is not None, "HL_001 competition showcase missing"
    assert hl_001["recommended_action"] == "auto_update"
    assert hl_001["change_detected"] is True
    assert len(hl_001["changes"]) >= 1

    conflict = next((rec for rec in data if rec.get("recommended_action") == "human_review"), None)
    assert conflict is not None, "Expected at least one human_review conflict example"


def test_to_competition_change_maps_sources() -> None:
    change = to_competition_change(
        {
            "field": "phone",
            "old_value": "239-555-1234",
            "new_value": "239-555-9000",
            "confidence_score": 0.88,
            "supporting_sources": ["practice_website", "NPPES"],
        }
    )
    assert "Practice Website" in change["supporting_sources"]
    assert "NPI Registry" in change["supporting_sources"]
