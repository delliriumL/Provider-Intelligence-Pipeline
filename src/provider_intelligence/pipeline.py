"""End-to-end provider intelligence pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from provider_intelligence.audit import (
    create_audit_event,
    new_audit_id,
    recommendation_to_audit_events,
    write_action_queue,
    write_audit_log,
    write_recommendations,
)
from provider_intelligence.config import ensure_outputs_dir
from provider_intelligence.data_generation import generate_demo_data
from provider_intelligence.decision_engine import decide_action
from provider_intelligence.ingest import ingest_all
from provider_intelligence.llm_enrichment import run_llm_enrichment
from provider_intelligence.match import detect_duplicates, find_all_source_matches
from provider_intelligence.schemas import Recommendation
from provider_intelligence.scoring import (
    compute_conflict_score,
    compute_overall_confidence,
    compute_risk_score,
    detect_field_changes,
)


def process_provider(
    provider,
    nppes_sources,
    cms_sources,
    duplicate_risks: dict[str, float],
) -> tuple[Recommendation, list]:
    """Process a single provider through match, score, and decision stages."""
    matched = find_all_source_matches(provider, nppes_sources, cms_sources)
    sources = [source for source, _ in matched]
    match_evidence = matched[0][1] if matched else None

    if match_evidence is None:
        from provider_intelligence.schemas import MatchEvidence

        match_evidence = MatchEvidence(match_type="none", match_score=0.0)

    duplicate_risk = duplicate_risks.get(provider.provider_id, 0.0)
    match_evidence.duplicate_risk = duplicate_risk

    primary_match = sources[0] if sources else None
    risk = compute_risk_score(provider, sources, duplicate_risk, primary_match)
    conflict = compute_conflict_score(provider, sources, match_evidence)
    changes = detect_field_changes(provider, sources, match_evidence)
    overall_confidence = compute_overall_confidence(changes) if changes else 1.0

    audit_id = new_audit_id()
    recommendation = Recommendation(
        provider_id=provider.provider_id,
        npi=provider.npi,
        change_detected=bool(changes),
        changes=changes,
        risk_score=risk.score,
        overall_confidence=overall_confidence,
        conflict_score=conflict.score,
        supporting_sources=sorted({s.source_name for s in sources}),
        audit_id=audit_id,
    )

    recommendation = decide_action(
        recommendation,
        match_evidence,
        risk,
        conflict,
        duplicate_risk,
    )

    events = [
        create_audit_event(
            provider_id=provider.provider_id,
            step="risk_scoring",
            rule_name="compute_risk_score",
            decision="scored",
            output_summary=f"risk_score={risk.score}",
            score_components=risk.components,
            source_names=recommendation.supporting_sources,
            audit_id=audit_id,
        ),
        create_audit_event(
            provider_id=provider.provider_id,
            step="conflict_scoring",
            rule_name="compute_conflict_score",
            decision="scored",
            output_summary=f"conflict_score={conflict.score}",
            score_components=conflict.components,
            source_names=recommendation.supporting_sources,
            audit_id=audit_id,
        ),
    ]
    events.extend(recommendation_to_audit_events(recommendation))
    return recommendation, events


def run_pipeline(
    mode: str = "demo",
    generate_data: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full provider intelligence pipeline."""
    outputs_dir = output_dir or ensure_outputs_dir()

    if generate_data or mode == "demo":
        sample_paths = {p: Path(p) for p in generate_demo_data(outputs_dir).values()}
    else:
        sample_paths = {}

    data = ingest_all(mode=mode)
    providers = data["providers"]
    nppes_sources = data["nppes"]
    cms_sources = data["cms"]

    duplicate_risks = detect_duplicates(providers)
    recommendations: list[Recommendation] = []
    audit_events = []

    for provider in providers:
        recommendation, events = process_provider(provider, nppes_sources, cms_sources, duplicate_risks)
        recommendations.append(recommendation)
        audit_events.extend(events)

    provider_map = {p.provider_id: p for p in providers}
    recommendations, llm_diagnostics = run_llm_enrichment(
        recommendations,
        provider_map,
        output_dir=outputs_dir,
    )

    rec_path = outputs_dir / "recommendations.json"
    audit_path = outputs_dir / "audit_log.csv"
    write_recommendations(recommendations, rec_path)
    write_audit_log(audit_events, audit_path)
    write_action_queue(recommendations, "human_review", outputs_dir / "human_review_queue.csv")
    write_action_queue(recommendations, "auto_update", outputs_dir / "auto_updates.csv")
    write_action_queue(recommendations, "no_change", outputs_dir / "no_change.csv")
    write_action_queue(recommendations, "do_not_update", outputs_dir / "do_not_update.csv")

    return {
        "recommendations_path": rec_path,
        "audit_log_path": audit_path,
        "records_processed": len(providers),
        "action_counts": _action_counts(recommendations),
        "sample_paths": sample_paths,
        "llm_diagnostics": llm_diagnostics,
    }


def _action_counts(recommendations: list[Recommendation]) -> dict[str, int]:
    counts = {"auto_update": 0, "human_review": 0, "no_change": 0, "do_not_update": 0}
    for rec in recommendations:
        counts[rec.recommended_action] += 1
    return counts
