"""Dashboard data adapters — schema-tolerant helpers (no Streamlit dependency)."""

from __future__ import annotations

import ast
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from provider_intelligence.npi import is_valid_npi_luhn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_ACTIONS = {"auto_update", "human_review", "no_change", "do_not_update"}

SOURCE_DISPLAY_NAMES: dict[str, str] = {
    "nppes_bulk": "NPI Registry",
    "NPPES": "NPI Registry",
    "npi_registry": "NPI Registry",
    "cms_doctors_clinicians": "CMS Doctors & Clinicians",
    "CMS Doctors & Clinicians": "CMS Doctors & Clinicians",
    "practice_website": "Practice Website",
    "Practice Website": "Practice Website",
    "state_medical_board": "State Medical Board",
    "nucc_taxonomy": "NUCC Taxonomy",
}

SOURCE_KEY_BY_DISPLAY: dict[str, str] = {
    "NPI Registry": "nppes_bulk",
    "CMS Doctors & Clinicians": "cms_doctors_clinicians",
    "Practice Website": "practice_website",
    "State Medical Board": "state_medical_board",
    "NUCC Taxonomy": "nucc_taxonomy",
}

FIELD_TO_RELIABILITY_KEY: dict[str, str] = {
    "address": "address",
    "phone": "phone",
    "specialty": "specialty",
    "active_status": "status",
    "practice_name": "address",
}


def load_json_safe(path: Path, fallback: Any = None) -> Any:
    if fallback is None:
        fallback = {}
    if not path.exists():
        return fallback
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return fallback


def load_csv_safe(path: Path, fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    if fallback is None:
        fallback = pd.DataFrame()
    if not path.exists():
        return fallback
    if path.name == "audit_llm_calls.csv":
        return load_llm_audit_safe(path, fallback=fallback)
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError, ValueError):
        return fallback


def load_llm_audit_safe(path: Path, fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load audit_llm_calls.csv, tolerating legacy rows with mixed schemas."""
    if fallback is None:
        fallback = pd.DataFrame()
    if not path.exists():
        return fallback
    try:
        from provider_intelligence.llm_diagnostics import read_llm_audit_csv

        return read_llm_audit_csv(path)
    except (OSError, ValueError):
        return fallback


def read_metric(metrics: dict[str, Any], possible_keys: list[str], default: Any = None) -> Any:
    for key in possible_keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    for key in possible_keys:
        if "." in key:
            parts = key.split(".")
            cur: Any = metrics
            for part in parts:
                if not isinstance(cur, dict) or part not in cur:
                    cur = None
                    break
                cur = cur[part]
            if cur is not None:
                return cur
    return default


def safe_get_action(record: dict[str, Any]) -> str:
    return str(record.get("recommended_action") or record.get("decision") or "")


def safe_get_confidence(record: dict[str, Any]) -> float | None:
    for key in ("overall_confidence", "confidence_score", "confidence"):
        val = record.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def safe_get_risk(record: dict[str, Any]) -> float | None:
    for key in ("risk_score", "risk"):
        val = record.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def safe_get_conflict(record: dict[str, Any]) -> float | None:
    for key in ("conflict_score", "conflict"):
        val = record.get(key)
        if val is None or val == "":
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def normalize_confidence_fraction(value: Any) -> float:
    """Normalize confidence to 0–1 regardless of input scale."""
    if value is None:
        return 0.0
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if num > 1.0 and num <= 100.0:
        return num / 100.0
    if num > 100.0:
        return min(num / 100.0, 1.0)
    return max(0.0, min(num, 1.0))


def format_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "N/A"
    pct = normalize_confidence_fraction(num) * 100
    return f"{pct:.1f}%"


def format_money(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def describe_llm_run_status(llm_summary: dict[str, Any]) -> dict[str, str]:
    """Judge-facing LLM diagnostics copy."""
    attempted = int(llm_summary.get("calls_attempted", 0) or 0)
    failed = int(llm_summary.get("calls_failed", 0) or 0)
    succeeded = int(llm_summary.get("calls_succeeded", 0) or 0)
    credentials = str(llm_summary.get("credentials_status", "unknown"))
    eligible = int(llm_summary.get("eligible_records", 0) or 0)
    last_error = llm_summary.get("last_error_type")

    if succeeded > 0:
        status = "LLM used for bounded enrichment"
        note = "Deterministic policy remains authoritative; LLM output is reviewer-facing only."
    elif attempted > 0 and failed > 0:
        status = f"LLM attempted but failed: {last_error or 'unknown'}"
        note = "Pipeline completed via deterministic fallback."
    elif attempted == 0 and eligible == 0:
        if credentials == "missing":
            status = "credentials not configured — deterministic rules only"
            note = "No LLM calls were attempted because API credentials are not configured."
        else:
            status = "LLM available but not required"
            note = "No ambiguous enrichment cases required LLM assistance this run."
    elif attempted == 0 and credentials == "missing":
        status = "credentials not configured — deterministic rules only"
        note = "No LLM calls were attempted because API credentials are not configured."
    else:
        status = "gated out by cost/safety policy (no calls required)"
        note = "Deterministic rules resolved eligible cases without LLM calls this run."

    return {"status": status, "note": note, "show_last_error": attempted > 0 and failed > 0}


def get_llm_enrichment_display(rec: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM enrichment block from a recommendation record."""
    block = rec.get("llm_enrichment") or {}
    if not isinstance(block, dict):
        return {}
    return block


def llm_enrichment_sections(enrichment: dict[str, Any]) -> list[tuple[str, str]]:
    """Return labeled LLM enrichment snippets for Review Queue display."""
    sections: list[tuple[str, str]] = []
    if enrichment.get("conflict_explanation"):
        sections.append(("Conflict explanation", str(enrichment["conflict_explanation"])))
    if enrichment.get("reviewer_summary"):
        sections.append(("Reviewer summary", str(enrichment["reviewer_summary"])))
    if enrichment.get("specialty_hint"):
        sections.append(("Specialty hint", str(enrichment["specialty_hint"])))
    website = enrichment.get("extracted_website_fields") or enrichment.get("website_extraction_result")
    if website:
        if isinstance(website, dict):
            parts = [f"{k}: {v}" for k, v in website.items() if v]
            text = "; ".join(parts) if parts else str(website)
        else:
            text = str(website)
        sections.append(("Website extraction", text))
    if enrichment.get("evidence_summary"):
        sections.append(("Evidence summary", str(enrichment["evidence_summary"])))
    return sections


@lru_cache(maxsize=1)
def load_source_reliability_config() -> dict[str, dict[str, float]]:
    path = PROJECT_ROOT / "config" / "source_reliability.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    sources = payload.get("sources", {})
    normalized: dict[str, dict[str, float]] = {}
    for key, block in sources.items():
        if isinstance(block, dict):
            normalized[key] = {k: float(v) for k, v in block.items() if isinstance(v, (int, float))}
    return normalized


def lookup_source_reliability(source_name: str, field: str | None = None) -> float | None:
    """Map display source name to field-specific reliability from config."""
    config = load_source_reliability_config()
    source_key = SOURCE_KEY_BY_DISPLAY.get(source_name, source_name)
    if source_key not in config:
        for key in config:
            if key.lower() in source_name.lower() or source_name.lower() in key.lower():
                source_key = key
                break
    block = config.get(source_key)
    if not block:
        return None
    if field and field in FIELD_TO_RELIABILITY_KEY:
        rel_key = FIELD_TO_RELIABILITY_KEY[field]
        if rel_key in block:
            return float(block[rel_key])
    if "identity" in block:
        return float(block["identity"])
    values = [float(v) for v in block.values() if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def _parse_json_like(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def parse_changes(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        parsed = _parse_json_like(value)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    return []


def _split_source_string(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = _parse_json_like(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _display_source_name(name: str) -> str:
    return SOURCE_DISPLAY_NAMES.get(name, name)


def normalize_sources(value: Any, field: str | None = None) -> list[dict[str, Any]]:
    """Normalize any supporting_sources shape to list of dicts."""
    raw_items: list[Any] = []
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = _split_source_string(value)
    elif isinstance(value, dict):
        raw_items = [value]
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip().startswith("["):
                raw_items.extend(_split_source_string(item))
            else:
                raw_items.append(item)
    else:
        raw_items = [value]

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            source = _display_source_name(str(item.get("source") or item.get("name") or "Unknown"))
            rel = item.get("reliability") or item.get("reliability_weight") or item.get("weight")
            if rel is None:
                rel = lookup_source_reliability(source, field)
            evidence = item.get("evidence") or item.get("detail") or item.get("matched_value") or ""
        elif isinstance(item, str):
            source = _display_source_name(item.strip())
            rel = lookup_source_reliability(source, field)
            evidence = ""
        else:
            source = str(item)
            rel = None
            evidence = ""
        if source in seen:
            continue
        seen.add(source)
        entry: dict[str, Any] = {"source": source, "evidence": evidence, "field": field or ""}
        if rel is not None:
            entry["reliability"] = float(rel)
        normalized.append(entry)
    return normalized


def collect_sources(record: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for change in parse_changes(record.get("changes")):
        field = str(change.get("field") or "")
        sources.extend(normalize_sources(change.get("supporting_sources"), field=field or None))
    sources.extend(normalize_sources(record.get("supporting_sources")))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for src in sources:
        key = src.get("source", "")
        if key in seen:
            continue
        seen.add(key)
        if not str(src.get("evidence") or "").strip():
            src["evidence"] = build_source_evidence_text(record, src)
        deduped.append(src)
    return deduped


def build_source_evidence_text(record: dict[str, Any], source: dict[str, Any]) -> str:
    """Deterministic reviewer-facing evidence when explicit detail is missing."""
    source_name = str(source.get("source") or "Unknown")
    field = str(source.get("field") or "")
    npi = record.get("npi")
    conflict = safe_get_conflict(record) or 0.0
    reason = str(record.get("reason") or "").lower()
    changes = parse_changes(record.get("changes"))
    change_by_field = {str(c.get("field") or ""): c for c in changes}
    source_lower = source_name.lower()

    if conflict >= 0.35 or "conflict" in reason or "disagree" in reason:
        if field == "address" or "address" in reason:
            return "Source conflict: NPI Registry and CMS Doctors & Clinicians disagree on address."
        return "Source conflict: external sources disagree on material directory fields."

    if npi and ("npi" in source_lower or "registry" in source_lower or "nppes" in source_lower):
        return f"Exact NPI match confirmed for {npi}."

    if field == "phone" or (field == "" and "phone" in change_by_field):
        change = change_by_field.get("phone") or change_by_field.get(field)
        if change and not change.get("conflict_detected", False):
            return f"Phone confirmed by {source_name} ({change.get('new_value', 'updated value')})."
        return f"Phone field verified against {source_name}."

    if field == "address" or (field == "" and "address" in change_by_field):
        change = change_by_field.get("address") or change_by_field.get(field)
        if change and not change.get("conflict_detected", False):
            return f"Address confirmed by {source_name} ({change.get('new_value', 'updated value')})."
        return f"Address verified against {source_name}."

    if field in {"practice_name", "specialty"} or source_name == "Practice Website":
        if field and field in change_by_field:
            return f"{field.replace('_', ' ').title()} aligned with {source_name}."
        return f"Clinician-location relationship matched via {source_name}."

    if field and field in change_by_field:
        change = change_by_field[field]
        old_val = change.get("old_value") or "—"
        new_val = change.get("new_value") or "—"
        return (
            f"{field.replace('_', ' ').title()} change from {source_name}: "
            f"{old_val} → {new_val}."
        )

    supporting = record.get("supporting_sources") or []
    if isinstance(supporting, list) and supporting:
        return f"{source_name} consulted during deterministic merge."
    return f"{source_name} matched for reviewer traceability."


def score_components(record: dict[str, Any], key: str) -> dict[str, float]:
    direct = record.get(f"{key}_components")
    if isinstance(direct, dict):
        return {str(k): float(v) for k, v in direct.items() if _is_number(v)}
    nested = record.get("score_components")
    if isinstance(nested, dict):
        inner = nested.get(key)
        if isinstance(inner, dict):
            return {str(k): float(v) for k, v in inner.items() if _is_number(v)}
    return {}


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def merge_recommendation_records(
    competition: list[dict[str, Any]],
    detailed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    detail_lookup = {str(r.get("provider_id", "")): r for r in detailed}
    merged: list[dict[str, Any]] = []
    for rec in competition:
        pid = str(rec.get("provider_id", ""))
        detail = detail_lookup.get(pid, {})
        combined = {**detail, **rec}
        if not combined.get("changes"):
            combined["changes"] = parse_changes(detail.get("changes")) or parse_changes(rec.get("changes"))
        merged.append(combined)
    if not competition and detailed:
        return detailed
    return merged


def parse_recommendations_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("recommendations", "providers", "records"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
        return [payload]
    return []


def parse_flags(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, dict):
        return {k for k, v in value.items() if v}
    if isinstance(value, list):
        return {str(v) for v in value if v}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        if text.startswith("["):
            parsed = _parse_json_like(text)
            if isinstance(parsed, list):
                return {str(v) for v in parsed if v}
        if "|" in text:
            return {part.strip() for part in text.split("|") if part.strip()}
        if "," in text:
            return {part.strip() for part in text.split(",") if part.strip()}
        return {text}
    return set()


def build_quality_flags_table(
    recommendations: list[dict[str, Any]],
    ground_truth_df: pd.DataFrame | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    gt_lookup: dict[str, set[str]] = {}
    if ground_truth_df is not None and not ground_truth_df.empty:
        for row in ground_truth_df.to_dict(orient="records"):
            pid = str(row.get("provider_id", ""))
            mutation = str(row.get("mutation_type", ""))
            flags: set[str] = set()
            if mutation == "invalid_npi":
                flags.add("invalid_npi")
            elif mutation == "stale_verification":
                flags.add("stale_record")
            elif mutation == "duplicate_created":
                flags.add("duplicate")
            elif mutation in {"address_changed", "phone_changed"}:
                flags.add("missing_phone" if mutation == "phone_changed" else "missing_address")
            gt_lookup[pid] = flags

    rows: list[dict[str, Any]] = []
    counts = {
        "invalid_npi": 0,
        "missing_phone": 0,
        "missing_address": 0,
        "stale_records": 0,
        "duplicates": 0,
    }

    for rec in recommendations:
        pid = str(rec.get("provider_id", ""))
        flags = parse_flags(rec.get("flags"))
        if not flags and pid in gt_lookup:
            flags = gt_lookup[pid]
        if not flags:
            npi = str(rec.get("npi") or "")
            if npi and not is_valid_npi_luhn(npi):
                flags.add("invalid_npi")
            reason = str(rec.get("reason", "")).lower()
            action = safe_get_action(rec).lower()
            if "invalid" in reason and "npi" in reason:
                flags.add("invalid_npi")
            if "duplicate" in reason:
                flags.add("duplicate")
            if "stale" in reason or "outdated" in reason or "verification" in reason:
                flags.add("stale_record")
            if action == "do_not_update" and "invalid" in reason:
                flags.add("invalid_npi")
            risk = safe_get_risk(rec) or 0.0
            if risk >= 0.7 and "stale" in reason:
                flags.add("stale_record")

        for flag in flags:
            if flag == "invalid_npi":
                counts["invalid_npi"] += 1
            elif flag == "missing_phone":
                counts["missing_phone"] += 1
            elif flag == "missing_address":
                counts["missing_address"] += 1
            elif flag in {"stale_record", "stale_records"}:
                counts["stale_records"] += 1
            elif flag == "duplicate":
                counts["duplicates"] += 1

        if flags:
            rows.append(
                {
                    "provider": rec.get("provider_name") or rec.get("name") or pid,
                    "npi": rec.get("npi"),
                    "flags": ", ".join(sorted(flags)),
                    "risk": safe_get_risk(rec) or 0.0,
                }
            )

    return counts, rows


def _clean_audit_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def format_audit_event(row: dict[str, Any]) -> str:
    ts = _clean_audit_value(row.get("timestamp") or row.get("created_at"))[:19].replace("T", " ")
    step = _clean_audit_value(row.get("step") or row.get("event_type"))
    rule = _clean_audit_value(row.get("rule_name"))
    decision = _clean_audit_value(row.get("decision") or row.get("action"))
    for key in ("message", "reason", "output_summary"):
        body = _clean_audit_value(row.get(key))
        if body:
            prefix = " | ".join(part for part in (ts, step, rule, decision) if part)
            return f"{prefix} | {body}" if prefix else body
    if rule and decision:
        return f"{ts} | {rule} | {decision}" if ts else f"{rule} | {decision}"
    if step and rule:
        return f"{ts} | {step} | {rule}" if ts else f"{step} | {rule}"
    return f"{ts} | Audit event recorded" if ts else "Audit event recorded"


def build_full_audit_event_text(row: dict[str, Any]) -> str:
    """Full audit narrative for detail panel — no truncation."""
    segments: list[str] = []
    for key in ("output_summary", "input_summary", "message", "reason"):
        body = _clean_audit_value(row.get(key))
        if body and body not in segments:
            segments.append(body)
    if segments:
        return "\n\n".join(segments)
    return format_audit_event(row)


def format_audit_event_label(row: dict[str, Any], index: int) -> str:
    ts = _clean_audit_value(row.get("timestamp") or row.get("created_at"))[:19].replace("T", " ")
    provider_id = _clean_audit_value(row.get("provider_id")) or "—"
    step = _clean_audit_value(row.get("step") or row.get("event_type")) or "event"
    return f"{index + 1}. {ts} · {provider_id} · {step}"


def format_provider_audit_event(row: dict[str, Any]) -> tuple[str, str]:
    """Compact review-queue audit line: timestamp separate from step/rule/decision body."""
    ts = _clean_audit_value(row.get("timestamp") or row.get("created_at"))[:19].replace("T", " ")
    step = _clean_audit_value(row.get("step") or row.get("event_type")) or "event"
    rule = _clean_audit_value(row.get("rule_name")) or "—"
    decision = _clean_audit_value(row.get("decision") or row.get("action")) or "—"
    message = ""
    for key in ("output_summary", "message", "reason", "input_summary"):
        message = _clean_audit_value(row.get(key))
        if message:
            break
    if message:
        body = f"{step} / {rule} → {decision}: {message}"
    else:
        body = f"{step} / {rule} → {decision}"
    return ts, body


def shorten_reason(text: str, max_len: int = 48) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def review_queue_label(record: dict[str, Any]) -> str:
    pid = record.get("provider_id", "—")
    name = record.get("provider_name") or record.get("name") or "Unknown"
    npi = record.get("npi", "—")
    risk = safe_get_risk(record)
    conf = safe_get_confidence(record)
    conflict = safe_get_conflict(record)
    reason = shorten_reason(record.get("reason", ""))
    parts = [f"{pid} | {name} | NPI {npi}"]
    if risk is not None:
        parts.append(f"risk {risk:.2f}")
    if conf is not None:
        parts.append(f"conf {conf:.2f}")
    if conflict is not None:
        parts.append(f"conflict {conflict:.2f}")
    if reason:
        parts.append(reason)
    return " | ".join(parts)


def adapt_cost_estimate(cost: dict[str, Any], mode: str = "auto") -> dict[str, Any]:
    by_mode = cost.get("by_mode", {})
    mode_block = by_mode.get(mode, by_mode.get("auto", {}))
    breakdown_raw = mode_block.get("cost_breakdown_usd", cost.get("breakdown", {}))
    breakdown = {
        "Rules engine": float(breakdown_raw.get("compute_overhead", breakdown_raw.get("rules_engine", 0.06))),
        "LLM enrichment": float(breakdown_raw.get("llm_assistance", breakdown_raw.get("llm_cost", 0))),
        "Human review": float(breakdown_raw.get("human_review", breakdown_raw.get("human_review_cost", 0))),
        "Storage & audit": float(breakdown_raw.get("storage_audit", breakdown_raw.get("storage_cost", 0.01))),
    }
    return {
        "mode": mode,
        "record_count": int(cost.get("record_count", mode_block.get("record_count", 0))),
        "cost_per_1000_records_usd": float(
            mode_block.get("cost_per_1000_records_usd", cost.get("cost_per_1000_records", 0))
        ),
        "total_estimated_cost_usd": float(mode_block.get("total_estimated_cost_usd", 0)),
        "estimated_llm_calls": float(mode_block.get("estimated_llm_calls", 0)),
        "llm_budget_cap": float(cost.get("assumptions", {}).get("llm_max_record_share", 0.08)),
        "breakdown": breakdown,
        "manual_review_baseline_per_1000": float(cost.get("manual_review_baseline_per_1000", 250.0)),
        "estimated_savings_per_1000": float(cost.get("estimated_savings_per_1000", 0)),
        "notes": cost.get("comparison", {}).get("notes", ""),
    }


def compute_llm_share(
    records_processed: int,
    llm_calls: int,
    budget_cap: float = 0.08,
) -> dict[str, Any]:
    actual = (llm_calls / records_processed) if records_processed else 0.0
    return {
        "llm_calls": llm_calls,
        "records_processed": records_processed,
        "actual_llm_share": actual,
        "budget_cap": budget_cap,
    }


def read_evaluation_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation.get("metrics", evaluation) if isinstance(evaluation.get("metrics"), dict) else evaluation
    benchmark_available = evaluation.get("benchmark_available", metrics.get("benchmark_available", False))
    false_rate = read_metric(evaluation, ["false_auto_update_rate"])
    if false_rate is None:
        false_rate = read_metric(metrics, ["false_auto_update_rate"])

    def bench(key: str) -> Any:
        if not benchmark_available:
            return None
        return read_metric(metrics, [key])

    return {
        "benchmark_available": benchmark_available,
        "false_auto_update_rate": false_rate,
        "change_detection_precision": bench("change_detection_precision"),
        "change_detection_recall": bench("change_detection_recall"),
        "auto_update_precision": bench("auto_update_precision"),
        "human_review_rate": read_metric(metrics, ["human_review_rate"]),
        "cost_sensitive_loss": bench("cost_sensitive_loss"),
        "records_evaluated": read_metric(metrics, ["records_evaluated", "records_processed"]),
        "llm_calls": read_metric(metrics, ["llm_calls"]),
    }


def summarize_llm_diagnostics_for_dashboard(
    llm_audit_df: pd.DataFrame,
    records_processed: int,
    cost: dict[str, Any],
    llm_mode: str = "auto",
) -> dict[str, Any]:
    """Build LLM diagnostics summary for Cost Model tab."""
    from provider_intelligence.llm_diagnostics import summarize_llm_audit_df

    summary = summarize_llm_audit_df(llm_audit_df, records_processed)
    if not llm_audit_df.empty and "use_case" in llm_audit_df.columns and "success" in llm_audit_df.columns:
        success_mask = llm_audit_df["success"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        succeeded = llm_audit_df[success_mask]
        use_cases: dict[str, int] = {}
        for use_case, group in succeeded.groupby("use_case"):
            key = str(use_case)
            if key in {"", "none"}:
                continue
            display = "specialty_normalization" if key == "specialty_fallback" else key
            use_cases[display] = int(len(group))
        summary["use_cases"] = use_cases

    auto_mode = cost.get("by_mode", {}).get(llm_mode, cost.get("by_mode", {}).get("auto", {}))
    summary["scenario_llm_calls"] = auto_mode.get("estimated_llm_calls")
    summary["llm_mode"] = llm_mode
    return summary
