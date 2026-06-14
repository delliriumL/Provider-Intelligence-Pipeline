"""Data ingestion for demo and optional real-file modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from provider_intelligence.config import load_config
from provider_intelligence.schemas import ExternalSourceRecord, ProviderRecord


def _parse_date(value: Any) -> Any:
    if pd.isna(value) or value == "":
        return None
    return pd.to_datetime(value).date()


def _parse_datetime(value: Any) -> Any:
    if pd.isna(value) or value == "":
        return None
    return pd.to_datetime(value).to_pydatetime()


def load_provider_records(path: Path | None = None) -> list[ProviderRecord]:
    """Load internal directory records from CSV."""
    config = load_config()
    file_path = path or Path(config["paths"]["sample_dir"]) / config["app"]["demo"]["sample_files"][
        "current_directory"
    ]
    if not file_path.exists():
        raise FileNotFoundError(f"Provider directory not found: {file_path}")
    df = pd.read_csv(file_path, dtype=str).fillna("")
    records: list[ProviderRecord] = []
    for row in df.to_dict(orient="records"):
        records.append(
            ProviderRecord(
                provider_id=row["provider_id"],
                provider_name=row["provider_name"],
                npi=row.get("npi") or None,
                specialty=row.get("specialty") or None,
                taxonomy_code=row.get("taxonomy_code") or None,
                practice_name=row.get("practice_name") or None,
                address_line_1=row.get("address_line_1") or None,
                address_line_2=row.get("address_line_2") or None,
                city=row.get("city") or None,
                state=row.get("state") or None,
                zip_code=row.get("zip_code") or None,
                phone=row.get("phone") or None,
                website=row.get("website") or None,
                active_status=row.get("active_status") or None,
                last_verified_date=_parse_date(row.get("last_verified_date")),
                source_system=row.get("source_system") or None,
                created_at=_parse_datetime(row.get("created_at")),
                updated_at=_parse_datetime(row.get("updated_at")),
            )
        )
    return records


def _source_reliability(source_key: str) -> dict[str, float]:
    config = load_config()
    sources = config["source_reliability"]["sources"]
    entry = sources.get(source_key, {})
    return {
        "identity": float(entry.get("identity", 0.5)),
        "address": float(entry.get("address", 0.5)),
        "phone": float(entry.get("phone", 0.5)),
        "specialty": float(entry.get("specialty", 0.5)),
        "status": float(entry.get("status", 0.5)),
    }


def load_nppes_records(path: Path | None = None) -> list[ExternalSourceRecord]:
    """Load NPPES-like external records."""
    config = load_config()
    file_path = path or Path(config["paths"]["sample_dir"]) / config["app"]["demo"]["sample_files"][
        "external_nppes"
    ]
    if not file_path.exists():
        raise FileNotFoundError(f"NPPES snapshot not found: {file_path}")
    df = pd.read_csv(file_path, dtype=str).fillna("")
    reliability = _source_reliability("nppes_bulk")
    records: list[ExternalSourceRecord] = []
    for row in df.to_dict(orient="records"):
        first = row.get("provider_first_name", "")
        last = row.get("provider_last_name", "")
        provider_name = f"{first} {last}".strip()
        status = row.get("npi_status") or "active"
        records.append(
            ExternalSourceRecord(
                source_name="NPPES",
                source_record_id=row.get("source_record_id") or row["npi"],
                provider_name=provider_name,
                npi=row.get("npi") or None,
                specialty=row.get("taxonomy_description") or None,
                taxonomy_code=row.get("primary_taxonomy_code") or None,
                practice_name=row.get("organization_name") or None,
                address_line_1=row.get("practice_address_line_1") or None,
                address_line_2=row.get("practice_address_line_2") or None,
                city=row.get("practice_city") or None,
                state=row.get("practice_state") or None,
                zip_code=row.get("practice_zip") or None,
                phone=row.get("practice_phone") or None,
                active_status=status,
                last_update_date=_parse_date(row.get("last_update_date")),
                source_reliability_by_field=reliability,
            )
        )
    return records


def load_cms_records(path: Path | None = None) -> list[ExternalSourceRecord]:
    """Load CMS Doctors & Clinicians-like external records."""
    config = load_config()
    file_path = path or Path(config["paths"]["sample_dir"]) / config["app"]["demo"]["sample_files"][
        "external_cms"
    ]
    if not file_path.exists():
        raise FileNotFoundError(f"CMS snapshot not found: {file_path}")
    df = pd.read_csv(file_path, dtype=str).fillna("")
    reliability = _source_reliability("cms_doctors_clinicians")
    records: list[ExternalSourceRecord] = []
    for row in df.to_dict(orient="records"):
        records.append(
            ExternalSourceRecord(
                source_name="CMS Doctors & Clinicians",
                source_record_id=row.get("source_record_id") or row["npi"],
                provider_name=row.get("provider_name") or None,
                npi=row.get("npi") or None,
                specialty=row.get("specialty") or None,
                taxonomy_code=row.get("taxonomy_code") or None,
                practice_name=row.get("practice_name") or None,
                address_line_1=row.get("address_line_1") or None,
                address_line_2=row.get("address_line_2") or None,
                city=row.get("city") or None,
                state=row.get("state") or None,
                zip_code=row.get("zip_code") or None,
                phone=row.get("phone") or None,
                active_status=row.get("active_status") or None,
                last_update_date=_parse_date(row.get("last_update_date")),
                source_reliability_by_field=reliability,
            )
        )
    return records


def ingest_all(
    mode: str = "demo",
    current_path: Path | None = None,
    nppes_path: Path | None = None,
    cms_path: Path | None = None,
) -> dict[str, list[Any]]:
    """Ingest all data sources for demo or real-file mode."""
    if mode not in {"demo", "real"}:
        raise ValueError(f"Unsupported ingest mode: {mode}")
    return {
        "providers": load_provider_records(current_path),
        "nppes": load_nppes_records(nppes_path),
        "cms": load_cms_records(cms_path),
    }
