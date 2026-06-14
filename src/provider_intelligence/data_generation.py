"""Synthetic demo data generation with ground truth labels."""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from provider_intelligence.config import ensure_outputs_dir, load_config
from provider_intelligence.npi import generate_valid_npi
from provider_intelligence.schemas import GroundTruthLabel

MUTATION_TYPES = [
    "address_changed",
    "phone_changed",
    "practice_renamed",
    "provider_deactivated",
    "provider_moved_group",
    "duplicate_created",
    "invalid_npi",
    "specialty_alias_changed",
    "stale_verification",
    "conflicting_source_created",
]

FIRST_NAMES = ["James", "Maria", "Robert", "Linda", "Michael", "Sarah", "David", "Emily"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
PRACTICES = [
    "Sunrise Medical Group",
    "Coastal Health Partners",
    "Valley Primary Care",
    "Metro Family Clinic",
    "Harborview Medical Center",
]
STREETS = ["Main St", "Oak Ave", "Pine Rd", "Maple Dr", "Cedar Blvd"]
CITIES = [
    ("Miami", "FL", "33101"),
    ("Orlando", "FL", "32801"),
    ("Tampa", "FL", "33602"),
    ("Jacksonville", "FL", "32202"),
    ("Atlanta", "GA", "30303"),
]
SPECIALTIES = [
    ("Family Medicine", "207Q00000X"),
    ("Internal Medicine", "207R00000X"),
    ("Pediatrics", "208000000X"),
    ("Dermatology", "207N00000X"),
    ("Emergency Medicine", "207P00000X"),
]


def _base_record(index: int, seed: int) -> dict[str, Any]:
    """Create a clean baseline provider record."""
    rng = random.Random(seed + index)
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    practice = rng.choice(PRACTICES)
    street_num = rng.randint(100, 9999)
    street = rng.choice(STREETS)
    city, state, zip_code = rng.choice(CITIES)
    specialty, taxonomy = rng.choice(SPECIALTIES)
    npi = generate_valid_npi(str(100000000 + index))
    phone = f"+1{rng.randint(200,999)}{rng.randint(200,999)}{rng.randint(1000,9999)}"
    verified = date.today() - timedelta(days=rng.randint(30, 120))
    now = datetime.now(timezone.utc)
    return {
        "provider_id": f"HL_{index:03d}",
        "provider_name": f"{first} {last}",
        "npi": npi,
        "specialty": specialty,
        "taxonomy_code": taxonomy,
        "practice_name": practice,
        "address_line_1": f"{street_num} {street}",
        "address_line_2": f"Ste {rng.randint(100, 500)}",
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "phone": phone,
        "website": f"https://www.{practice.lower().replace(' ', '')}.example",
        "active_status": "active",
        "last_verified_date": verified.isoformat(),
        "source_system": "healthlynked",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


def _nppes_from_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Map internal record to NPPES-like external snapshot."""
    parts = record["provider_name"].split()
    return {
        "npi": record["npi"],
        "entity_type_code": "1",
        "provider_first_name": parts[0],
        "provider_last_name": parts[-1],
        "provider_middle_name": "",
        "provider_credential_text": "MD",
        "organization_name": record["practice_name"],
        "practice_address_line_1": record["address_line_1"],
        "practice_address_line_2": record["address_line_2"],
        "practice_city": record["city"],
        "practice_state": record["state"],
        "practice_zip": record["zip_code"],
        "practice_phone": record["phone"],
        "mailing_address_line_1": record["address_line_1"],
        "mailing_address_line_2": record["address_line_2"],
        "mailing_city": record["city"],
        "mailing_state": record["state"],
        "mailing_zip": record["zip_code"],
        "mailing_phone": record["phone"],
        "primary_taxonomy_code": record["taxonomy_code"],
        "taxonomy_description": record["specialty"],
        "enumeration_date": "2015-01-01",
        "last_update_date": date.today().isoformat(),
        "deactivation_date": "",
        "npi_status": "active" if record["active_status"] == "active" else "deactivated",
        "source_record_id": f"NPPES_{index:03d}",
    }


def _cms_from_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Map internal record to CMS Doctors & Clinicians-like snapshot."""
    return {
        "npi": record["npi"],
        "provider_name": record["provider_name"],
        "specialty": record["specialty"],
        "taxonomy_code": record["taxonomy_code"],
        "practice_name": record["practice_name"],
        "address_line_1": record["address_line_1"],
        "address_line_2": record["address_line_2"],
        "city": record["city"],
        "state": record["state"],
        "zip_code": record["zip_code"],
        "phone": record["phone"],
        "active_status": record["active_status"],
        "last_update_date": date.today().isoformat(),
        "source_record_id": f"CMS_{index:03d}",
    }


def _apply_mutation(
    current: dict[str, Any],
    external_nppes: dict[str, Any],
    external_cms: dict[str, Any],
    mutation: str,
    rng: random.Random,
) -> GroundTruthLabel:
    """Apply a mutation to current directory and optionally external sources."""
    changed_fields: list[str] = []
    expected_action = "human_review"
    notes = ""

    if mutation == "address_changed":
        current["address_line_1"] = f"{rng.randint(100,999)} New Address Rd"
        changed_fields = ["address"]
        expected_action = "auto_update"
        notes = "Directory address stale; external sources agree on new address."
    elif mutation == "phone_changed":
        current["phone"] = f"+1{rng.randint(200,999)}{rng.randint(200,999)}{rng.randint(1000,9999)}"
        changed_fields = ["phone"]
        expected_action = "auto_update"
    elif mutation == "practice_renamed":
        current["practice_name"] = f"{current['practice_name']} PLLC"
        changed_fields = ["practice_name"]
        expected_action = "human_review"
    elif mutation == "provider_deactivated":
        current["active_status"] = "active"
        external_nppes["npi_status"] = "deactivated"
        external_nppes["deactivation_date"] = date.today().isoformat()
        external_cms["active_status"] = "inactive"
        changed_fields = ["active_status"]
        expected_action = "human_review"
        notes = "Deactivation signal from NPPES."
    elif mutation == "provider_moved_group":
        current["practice_name"] = "New Group Medical Associates"
        current["address_line_1"] = f"{rng.randint(100,999)} Transfer Blvd"
        changed_fields = ["practice_name", "address"]
        expected_action = "human_review"
    elif mutation == "duplicate_created":
        current["provider_id"] = f"{current['provider_id']}_DUP"
        changed_fields = ["duplicate"]
        expected_action = "human_review"
        notes = "Duplicate provider_id pattern with same NPI."
    elif mutation == "invalid_npi":
        current["npi"] = "1234567890"
        changed_fields = ["npi"]
        expected_action = "do_not_update"
    elif mutation == "specialty_alias_changed":
        current["specialty"] = "Family Doc"
        changed_fields = ["specialty"]
        expected_action = "no_change"
        notes = "Alias maps to same taxonomy family."
    elif mutation == "stale_verification":
        current["last_verified_date"] = (date.today() - timedelta(days=800)).isoformat()
        changed_fields = ["last_verified_date"]
        expected_action = "human_review"
    elif mutation == "conflicting_source_created":
        external_cms["address_line_1"] = f"{rng.randint(100,999)} Conflicting Ave"
        changed_fields = ["address"]
        expected_action = "human_review"
        notes = "NPPES and CMS disagree on address."

    return GroundTruthLabel(
        provider_id=current["provider_id"],
        mutation_type=mutation,
        expected_action=expected_action,
        changed_fields=changed_fields,
        notes=notes,
    )


def _competition_showcase() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[GroundTruthLabel],
]:
    """Deterministic competition-facing demo records (HL_001 auto_update, HL_002 conflict)."""
    competition_npi = generate_valid_npi("123456789")
    competition_phone = "239-555-1234"
    competition_phone_e164 = "+12395551234"
    competition_phone_new = "239-555-9000"
    competition_phone_new_e164 = "+12395559000"
    now = datetime.now(timezone.utc)

    showcase_clean = {
        "provider_id": "HL_001",
        "provider_name": "John Smith, MD",
        "npi": competition_npi,
        "specialty": "Cardiology",
        "taxonomy_code": "207RC0000X",
        "practice_name": "ABC Heart Group",
        "address_line_1": "100 Main St",
        "address_line_2": "",
        "city": "Naples",
        "state": "FL",
        "zip_code": "34102",
        "phone": competition_phone_e164,
        "website": "https://www.abcheartgroup.example",
        "active_status": "active",
        "last_verified_date": "2023-09-01",
        "source_system": "healthlynked",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    current_001 = showcase_clean.copy()
    nppes_001 = _nppes_from_record(showcase_clean, 1)
    cms_001 = _cms_from_record(showcase_clean, 1)
    nppes_001.update(
        {
            "practice_address_line_1": "250 Health Park Dr",
            "practice_city": "Fort Myers",
            "practice_state": "FL",
            "practice_zip": "33908",
            "practice_phone": competition_phone_new_e164,
            "mailing_address_line_1": "250 Health Park Dr",
            "mailing_city": "Fort Myers",
            "mailing_state": "FL",
            "mailing_zip": "33908",
            "mailing_phone": competition_phone_new_e164,
        }
    )
    cms_001.update(
        {
            "address_line_1": "250 Health Park Dr",
            "city": "Fort Myers",
            "state": "FL",
            "zip_code": "33908",
            "phone": competition_phone_new_e164,
        }
    )
    label_001 = GroundTruthLabel(
        provider_id="HL_001",
        mutation_type="competition_auto_update",
        expected_action="auto_update",
        changed_fields=["address", "phone"],
        notes=(
            "Competition showcase (John Smith, MD): address "
            f"100 Main St, Naples, FL 34102 -> 250 Health Park Dr, Fort Myers, FL 33908; "
            f"phone {competition_phone} -> {competition_phone_new}."
        ),
    )

    conflict_npi = generate_valid_npi("223456789")
    conflict_clean = {
        "provider_id": "HL_002",
        "provider_name": "Maria Lopez, DO",
        "npi": conflict_npi,
        "specialty": "Family Medicine",
        "taxonomy_code": "207Q00000X",
        "practice_name": "Gulf Coast Family Clinic",
        "address_line_1": "410 Palm Ave",
        "address_line_2": "Ste 200",
        "city": "Sarasota",
        "state": "FL",
        "zip_code": "34236",
        "phone": "+19415551234",
        "website": "https://www.gulfcoastfamily.example",
        "active_status": "active",
        "last_verified_date": "2024-01-15",
        "source_system": "healthlynked",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    current_002 = conflict_clean.copy()
    nppes_002 = _nppes_from_record(conflict_clean, 2)
    cms_002 = _cms_from_record(conflict_clean, 2)
    nppes_002.update(
        {
            "practice_address_line_1": "250 Health Park Dr",
            "practice_city": "Fort Myers",
            "practice_state": "FL",
            "practice_zip": "33908",
        }
    )
    cms_002.update(
        {
            "address_line_1": "900 Conflicting Blvd",
            "city": "Tampa",
            "state": "FL",
            "zip_code": "33602",
        }
    )
    label_002 = GroundTruthLabel(
        provider_id="HL_002",
        mutation_type="competition_conflict",
        expected_action="human_review",
        changed_fields=["address"],
        notes="Competition showcase: NPI Registry and CMS Doctors & Clinicians disagree on address.",
    )

    return (
        [current_001, current_002],
        [nppes_001, nppes_002],
        [cms_001, cms_002],
        [showcase_clean, conflict_clean],
        [label_001, label_002],
    )


def _llm_demo_showcase() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[GroundTruthLabel],
]:
    """Deterministic LLM enrichment demo records (HL_003–HL_005)."""
    now = datetime.now(timezone.utc)

    website_npi = generate_valid_npi("323456789")
    website_clean = {
        "provider_id": "HL_003",
        "provider_name": "Alex Chen, MD",
        "npi": website_npi,
        "specialty": "Internal Medicine",
        "taxonomy_code": "207R00000X",
        "practice_name": "Bayview Wellness",
        "address_line_1": "800 Old Harbor Rd",
        "address_line_2": "",
        "city": "Tampa",
        "state": "FL",
        "zip_code": "33602",
        "phone": "+18135550199",
        "website": "local:messy_demo_page.html",
        "active_status": "active",
        "last_verified_date": "2023-06-01",
        "source_system": "healthlynked",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    current_003 = website_clean.copy()
    nppes_003 = _nppes_from_record(website_clean, 3)
    cms_003 = _cms_from_record(website_clean, 3)
    cms_003.update(
        {
            "address_line_1": "1420 Oak Ave Suite 210",
            "city": "Miami",
            "state": "FL",
            "zip_code": "33101",
            "phone": "+13055550142",
        }
    )
    label_003 = GroundTruthLabel(
        provider_id="HL_003",
        mutation_type="llm_website_extraction",
        expected_action="human_review",
        changed_fields=["address", "phone"],
        notes="LLM demo: messy practice website HTML; deterministic parser confidence < 0.60.",
    )

    specialty_npi = generate_valid_npi("423456789")
    specialty_clean = {
        "provider_id": "HL_004",
        "provider_name": "Jordan Patel, DO",
        "npi": specialty_npi,
        "specialty": "Gen Prac / FP",
        "taxonomy_code": "",
        "practice_name": "Coastal Primary Partners",
        "address_line_1": "220 Palm Trace",
        "address_line_2": "Ste 110",
        "city": "Fort Lauderdale",
        "state": "FL",
        "zip_code": "33301",
        "phone": "+19545550144",
        "website": "https://www.coastalprimary.example",
        "active_status": "active",
        "last_verified_date": "2024-02-10",
        "source_system": "healthlynked",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    current_004 = specialty_clean.copy()
    nppes_004 = _nppes_from_record(specialty_clean, 4)
    cms_004 = _cms_from_record(specialty_clean, 4)
    nppes_004.update({"taxonomy_description": "Family Medicine", "primary_taxonomy_code": "207Q00000X"})
    cms_004.update({"specialty": "Family Medicine", "taxonomy_code": "207Q00000X"})
    label_004 = GroundTruthLabel(
        provider_id="HL_004",
        mutation_type="llm_specialty_normalization",
        expected_action="human_review",
        changed_fields=["specialty"],
        notes="LLM demo: messy specialty alias; taxonomy mapping confidence < 0.70.",
    )

    review_npi = generate_valid_npi("523456789")
    review_clean = {
        "provider_id": "HL_005",
        "provider_name": "Taylor Nguyen, MD",
        "npi": review_npi,
        "specialty": "Emergency Medicine",
        "taxonomy_code": "207P00000X",
        "practice_name": "Metro Emergency Associates",
        "address_line_1": "500 Transfer Blvd",
        "address_line_2": "",
        "city": "Orlando",
        "state": "FL",
        "zip_code": "32801",
        "phone": "+14075550155",
        "website": "https://www.metroemergency.example",
        "active_status": "active",
        "last_verified_date": (date.today() - timedelta(days=900)).isoformat(),
        "source_system": "healthlynked",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    current_005 = review_clean.copy()
    nppes_005 = _nppes_from_record(review_clean, 5)
    cms_005 = _cms_from_record(review_clean, 5)
    nppes_005.update({"npi_status": "deactivated", "deactivation_date": date.today().isoformat()})
    cms_005.update({"active_status": "inactive"})
    current_005["practice_name"] = "New Group Medical Associates"
    label_005 = GroundTruthLabel(
        provider_id="HL_005",
        mutation_type="llm_reviewer_summary",
        expected_action="human_review",
        changed_fields=["active_status", "practice_name", "last_verified_date"],
        notes="LLM demo: complex human_review case (deactivation + group move + stale verification).",
    )

    return (
        [current_003, current_004, current_005],
        [nppes_003, nppes_004, nppes_005],
        [cms_003, cms_004, cms_005],
        [website_clean, specialty_clean, review_clean],
        [label_003, label_004, label_005],
    )


def generate_demo_data(output_dir: Path | None = None) -> dict[str, Path]:
    """Generate synthetic directory, external sources, and ground truth."""
    config = load_config()
    seed = config["app"]["demo"]["seed"]
    record_count = config["app"]["demo"]["record_count"]
    sample_dir = Path(config["paths"]["sample_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = output_dir or ensure_outputs_dir()

    rng = random.Random(seed)
    clean_records: list[dict[str, Any]] = []
    current_records: list[dict[str, Any]] = []
    nppes_records: list[dict[str, Any]] = []
    cms_records: list[dict[str, Any]] = []
    labels: list[GroundTruthLabel] = []

    showcase_current, showcase_nppes, showcase_cms, showcase_clean, showcase_labels = _competition_showcase()
    current_records.extend(showcase_current)
    nppes_records.extend(showcase_nppes)
    cms_records.extend(showcase_cms)
    clean_records.extend(showcase_clean)
    labels.extend(showcase_labels)

    llm_current, llm_nppes, llm_cms, llm_clean, llm_labels = _llm_demo_showcase()
    current_records.extend(llm_current)
    nppes_records.extend(llm_nppes)
    cms_records.extend(llm_cms)
    clean_records.extend(llm_clean)
    labels.extend(llm_labels)

    for index in range(6, record_count + 1):
        clean = _base_record(index, seed)
        clean_records.append(clean.copy())
        current = clean.copy()
        nppes = _nppes_from_record(clean, index)
        cms = _cms_from_record(clean, index)

        mutation = MUTATION_TYPES[(index - 1) % len(MUTATION_TYPES)]
        label = _apply_mutation(current, nppes, cms, mutation, rng)
        labels.append(label)

        current_records.append(current)
        nppes_records.append(nppes)
        cms_records.append(cms)

    current_path = sample_dir / config["app"]["demo"]["sample_files"]["current_directory"]
    nppes_path = sample_dir / config["app"]["demo"]["sample_files"]["external_nppes"]
    cms_path = sample_dir / config["app"]["demo"]["sample_files"]["external_cms"]
    ground_truth_path = outputs_dir / "synthetic_ground_truth.csv"

    pd.DataFrame(current_records).to_csv(current_path, index=False)
    pd.DataFrame(nppes_records).to_csv(nppes_path, index=False)
    pd.DataFrame(cms_records).to_csv(cms_path, index=False)
    pd.DataFrame([label.model_dump() for label in labels]).to_csv(ground_truth_path, index=False)

    return {
        "current_directory": current_path,
        "external_nppes": nppes_path,
        "external_cms": cms_path,
        "ground_truth": ground_truth_path,
    }
