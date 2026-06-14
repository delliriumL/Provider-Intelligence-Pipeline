"""LLM call diagnostics and audit logging for dashboard transparency."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from provider_intelligence.config import load_config
from provider_intelligence.llm import LLMClient

LLMMode = Literal["off", "auto", "force"]

AUDIT_FIELDS = [
    "timestamp",
    "provider_id",
    "mode",
    "use_case",
    "credentials_present",
    "candidate_reason",
    "attempted",
    "success",
    "error_type",
    "http_status",
    "model",
    "estimated_tokens",
    "estimated_cost",
    "output_valid",
    "fallback_used",
]


def _classify_error(error: str | None, gating_reason: str, attempted: bool) -> str:
    if not attempted:
        if gating_reason == "llm_mode_off":
            return "mode_off"
        if gating_reason in {"credentials_unavailable", "missing_credentials"}:
            return "missing_credentials"
        if gating_reason == "budget_exhausted":
            return "budget_cap_reached"
        if gating_reason == "below_gating_thresholds":
            return "deterministic_confidence_sufficient"
        if gating_reason == "use_case_disabled":
            return "gated_out"
        return "gated_out"

    if not error:
        return "none"

    lowered = error.lower()
    if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        return "unauthorized"
    if "429" in lowered or "rate limit" in lowered:
        return "rate_limited"
    if "quota" in lowered or "insufficient" in lowered:
        return "quota_exceeded"
    if "timeout" in lowered or "connection" in lowered or "network" in lowered:
        return "network_error"
    if "validation" in lowered:
        return "invalid_response"
    return "network_error"


def init_llm_audit_file(path: Path, mode: LLMMode) -> None:
    """Create audit_llm_calls.csv with headers when LLM mode is auto or force."""
    if mode == "off":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()


def append_llm_audit_row(path: Path, row: dict[str, Any]) -> None:
    """Append one diagnostics row without exposing secrets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in AUDIT_FIELDS})


def read_llm_audit_csv(path: Path) -> pd.DataFrame:
    """Load audit_llm_calls.csv, skipping legacy rows with incompatible schemas."""

    if not path.exists():
        return pd.DataFrame(columns=AUDIT_FIELDS)

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return pd.DataFrame(columns=AUDIT_FIELDS)

        header = [cell.strip() for cell in header]
        for line in reader:
            if not line or not any(cell.strip() for cell in line):
                continue
            if len(line) == len(AUDIT_FIELDS):
                rows.append(dict(zip(AUDIT_FIELDS, line, strict=True)))
            elif len(line) == len(header):
                rows.append(dict(zip(header, line, strict=True)))

    if not rows:
        return pd.DataFrame(columns=AUDIT_FIELDS)

    df = pd.DataFrame(rows)
    if "gating_reason" in df.columns and "candidate_reason" not in df.columns:
        df = df.rename(columns={"gating_reason": "candidate_reason"})
    if "mode" in df.columns:
        pass
    elif "llm_mode" in df.columns:
        df = df.rename(columns={"llm_mode": "mode", "called": "attempted"})
        if "success" not in df.columns and "called" in df.columns:
            df["success"] = df["called"]
    for field in AUDIT_FIELDS:
        if field not in df.columns:
            df[field] = ""
    return df[AUDIT_FIELDS] if all(col in df.columns for col in AUDIT_FIELDS) else df


def summarize_llm_audit_df(df: pd.DataFrame, records_processed: int) -> dict[str, Any]:
    """Summarize in-memory LLM audit rows for dashboard display."""

    cfg = load_config()
    budget_cap = cfg["llm"]["gating"].get("max_record_share", 0.08)
    if df is None or df.empty:
        client = LLMClient(cfg)
        return {
            "llm_mode": cfg["llm"].get("mode", "auto"),
            "credentials_status": "present" if client.credentials_available else "missing",
            "eligible_records": 0,
            "calls_attempted": 0,
            "calls_succeeded": 0,
            "calls_failed": 0,
            "actual_llm_share": 0.0,
            "attempted_llm_share": 0.0,
            "budget_cap": budget_cap,
            "last_error_type": None,
            "scenario_llm_calls": None,
        }

    work = df.copy()
    if "provider_id" in work.columns:
        work = work[work["provider_id"].astype(str) != "__run_summary__"]

    def _bool_sum(column: str) -> int:
        if column not in work.columns:
            return 0
        return int(work[column].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"}).sum())

    attempted = _bool_sum("attempted")
    if attempted == 0 and "called" in work.columns:
        attempted = _bool_sum("called")
    succeeded = _bool_sum("success")
    failed = max(attempted - succeeded, 0)
    if "candidate_reason" in work.columns:
        eligible = int(
            work["candidate_reason"].astype(str).apply(
                lambda r: r not in {"deterministic_confidence_sufficient", "mode_off", "no_supported_use_case", ""}
            ).sum()
        )
    elif "gating_reason" in work.columns:
        eligible = int((work["gating_reason"].astype(str) != "below_gating_thresholds").sum())
    else:
        eligible = len(work)

    mode = str(work["mode"].iloc[0]) if "mode" in work.columns and len(work) else cfg["llm"].get("mode", "auto")
    last_error = None
    if attempted > 0 and failed > 0 and "error_type" in work.columns:
        attempt_col = "attempted" if "attempted" in work.columns else "called"
        attempted_mask = work[attempt_col].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        errors = work[
            attempted_mask
            & work["error_type"].astype(str).notna()
            & ~work["error_type"].astype(str).isin({"none", "gated_out", "mode_off", "deterministic_confidence_sufficient"})
        ]
        if not errors.empty:
            last_error = str(errors.iloc[-1]["error_type"])

    denom = records_processed or 1
    client = LLMClient(cfg)
    return {
        "llm_mode": mode,
        "credentials_status": "present" if client.credentials_available else "missing",
        "eligible_records": eligible,
        "calls_attempted": attempted,
        "calls_succeeded": succeeded,
        "calls_failed": failed,
        "actual_llm_share": succeeded / denom,
        "attempted_llm_share": attempted / denom,
        "budget_cap": budget_cap,
        "last_error_type": last_error,
        "scenario_llm_calls": None,
    }


def summarize_llm_audit(audit_path: Path, records_processed: int) -> dict[str, Any]:
    """Summarize audit_llm_calls.csv for dashboard display."""
    import pandas as pd

    if not audit_path.exists():
        return summarize_llm_audit_df(pd.DataFrame(), records_processed)

    df = read_llm_audit_csv(audit_path)
    if df.empty:
        return summarize_llm_audit_df(pd.DataFrame(), records_processed)

    return summarize_llm_audit_df(df, records_processed)
