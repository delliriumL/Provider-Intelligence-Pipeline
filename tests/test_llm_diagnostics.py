"""Tests for LLM audit CSV loading with mixed legacy schemas."""

from __future__ import annotations

from pathlib import Path

from provider_intelligence.llm_diagnostics import (
    AUDIT_FIELDS,
    read_llm_audit_csv,
    summarize_llm_audit,
    summarize_llm_audit_df,
)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_llm_audit_csv_skips_legacy_rows(tmp_path: Path) -> None:
    header = ",".join(AUDIT_FIELDS)
    new_row = (
        "2026-06-14T12:00:00+00:00,HL_001,auto,reviewer_summary,True,"
        "deterministic_confidence_sufficient,False,False,deterministic_confidence_sufficient,,"
        "openai/gpt-oss-120b,0,0.0,False,False"
    )
    legacy_row = (
        "2026-06-14T12:20:54+00:00,HL_003,reviewer_summary,off,llm_mode_off,"
        "False,True,openai/gpt-oss-120b,0,0,0.0,True,"
    )
    _write_lines(tmp_path / "audit_llm_calls.csv", [header, new_row, legacy_row])

    df = read_llm_audit_csv(tmp_path / "audit_llm_calls.csv")

    assert len(df) == 1
    assert df.iloc[0]["provider_id"] == "HL_001"
    assert df.iloc[0]["error_type"] == "deterministic_confidence_sufficient"


def test_summarize_llm_audit_tolerates_mixed_file(tmp_path: Path) -> None:
    header = ",".join(AUDIT_FIELDS)
    new_row = (
        "2026-06-14T12:00:00+00:00,HL_001,auto,reviewer_summary,True,"
        "deterministic_confidence_sufficient,False,False,deterministic_confidence_sufficient,,"
        "openai/gpt-oss-120b,0,0.0,False,False"
    )
    legacy_row = (
        "2026-06-14T12:20:54+00:00,HL_003,reviewer_summary,off,llm_mode_off,"
        "False,True,openai/gpt-oss-120b,0,0,0.0,True,"
    )
    audit_path = tmp_path / "audit_llm_calls.csv"
    _write_lines(audit_path, [header, new_row, legacy_row])

    summary = summarize_llm_audit(audit_path, records_processed=60)

    assert summary["calls_attempted"] == 0
    assert summary["calls_succeeded"] == 0


def test_summarize_llm_audit_df_from_legacy_columns() -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "provider_id": "HL_003",
                "llm_mode": "off",
                "gating_reason": "llm_mode_off",
                "called": False,
                "success": False,
                "error_type": "mode_off",
            }
        ]
    )
    summary = summarize_llm_audit_df(df, records_processed=10)

    assert summary["llm_mode"] == "off"
    assert summary["calls_attempted"] == 0
