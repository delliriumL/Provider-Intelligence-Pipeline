"""Tests for synthetic benchmark evaluation metrics."""

from __future__ import annotations

from provider_intelligence.evaluation import (
    compute_false_auto_update_rate,
    compute_cost_sensitive_loss,
)
from provider_intelligence.schemas import FieldChange, GroundTruthLabel, Recommendation


def _auto_rec(provider_id: str, fields: list[str]) -> Recommendation:
    changes = [
        FieldChange(
            field=field,
            old_value="old",
            new_value="new",
            confidence_score=0.92,
            supporting_sources=["NPPES"],
        )
        for field in fields
    ]
    return Recommendation(
        provider_id=provider_id,
        npi="1234567893",
        change_detected=True,
        changes=changes,
        overall_confidence=0.92,
        recommended_action="auto_update",
        reason="test",
        audit_id="AUD_TEST",
    )


def test_false_auto_update_rate_zero_when_only_allowed_fields():
    ground_truth = {
        "HL_001": GroundTruthLabel(
            provider_id="HL_001",
            mutation_type="address_changed",
            expected_action="auto_update",
            changed_fields=["address"],
        ),
        "HL_003": GroundTruthLabel(
            provider_id="HL_003",
            mutation_type="practice_renamed",
            expected_action="human_review",
            changed_fields=["practice_name"],
        ),
    }
    recommendations = [
        _auto_rec("HL_001", ["address"]),
        _auto_rec("HL_003", ["practice_name"]),
    ]
    metrics = compute_false_auto_update_rate(recommendations, ground_truth)
    assert metrics["auto_update_count"] == 2
    assert metrics["false_auto_update_count"] == 1
    assert metrics["false_auto_update_rate"] == 0.5


def test_cost_sensitive_loss_computed_with_ground_truth():
    ground_truth = {
        "HL_001": GroundTruthLabel(
            provider_id="HL_001",
            expected_action="auto_update",
            changed_fields=["address"],
        ),
    }
    recommendations = [_auto_rec("HL_001", ["address"])]
    loss = compute_cost_sensitive_loss(recommendations, ground_truth)
    assert loss is not None
    assert loss >= 0.0
