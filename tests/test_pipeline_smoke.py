"""End-to-end smoke test: demo data generation and pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from provider_intelligence.cli import _do_generate_demo_data, _do_run_pipeline

VALID_ACTIONS = {"auto_update", "human_review", "no_change", "do_not_update"}

EXPECTED_OUTPUTS = [
    "recommendations.json",
    "audit_log.csv",
    "human_review_queue.csv",
    "auto_updates.csv",
    "no_change.csv",
    "do_not_update.csv",
]


@pytest.fixture
def smoke_output_dir(tmp_path: Path) -> Path:
    """Isolated outputs directory for smoke test."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out


def test_pipeline_smoke_generates_outputs(smoke_output_dir: Path) -> None:
    """Generate demo data, run pipeline, assert expected artifacts exist."""
    _do_generate_demo_data(smoke_output_dir)
    result = _do_run_pipeline(generate_data=False, output_dir=smoke_output_dir)

    assert result["records_processed"] > 0

    for filename in EXPECTED_OUTPUTS:
        path = smoke_output_dir / filename
        assert path.exists(), f"Missing output: {filename}"
        assert path.stat().st_size > 0, f"Empty output: {filename}"

    rec_path = smoke_output_dir / "recommendations.json"
    recommendations = json.loads(rec_path.read_text(encoding="utf-8"))
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

    for rec in recommendations:
        action = rec.get("recommended_action")
        assert action in VALID_ACTIONS, f"Invalid action: {action}"


def test_action_counts_match_recommendations(smoke_output_dir: Path) -> None:
    """Action queue row counts should sum to processed records."""
    _do_generate_demo_data(smoke_output_dir)
    result = _do_run_pipeline(generate_data=False, output_dir=smoke_output_dir)

    recs = json.loads((smoke_output_dir / "recommendations.json").read_text(encoding="utf-8"))
    queue_total = sum(
        max(0, sum(1 for _ in open(smoke_output_dir / f, encoding="utf-8")) - 1)
        for f in ["human_review_queue.csv", "auto_updates.csv", "no_change.csv", "do_not_update.csv"]
        if (smoke_output_dir / f).exists()
    )
    assert queue_total == len(recs)
    assert sum(result["action_counts"].values()) == len(recs)
