"""Bounded LLM enrichment for ambiguous demo cases — decisions stay deterministic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from provider_intelligence.config import PROJECT_ROOT, ensure_outputs_dir, load_config
from provider_intelligence.llm import LLMClient, LLMGatingContext, UseCase
from provider_intelligence.llm_diagnostics import append_llm_audit_row, init_llm_audit_file, summarize_llm_audit
from provider_intelligence.schemas import LLMEnrichment, LLMExtractionResult, ProviderRecord, Recommendation
from provider_intelligence.taxonomy import get_taxonomy_lookup
from provider_intelligence.website_parser import parse_html_deterministic

LLMMode = Literal["off", "auto", "force"]

LLM_DEMO_TARGETS: dict[str, UseCase] = {
    "HL_002": "conflict_explanation",
    "HL_003": "website_extraction",
    "HL_004": "specialty_fallback",
    "HL_005": "reviewer_summary",
}

_DEMO_PRIORITY = {pid: idx for idx, pid in enumerate(LLM_DEMO_TARGETS)}


@dataclass
class _LLMCandidate:
    recommendation: Recommendation
    provider: ProviderRecord | None
    use_case: UseCase
    context: LLMGatingContext
    candidate_reason: str
    is_demo: bool
    priority: float


def _website_html_for_provider(provider: ProviderRecord | None) -> tuple[str, float] | None:
    if provider is None or not provider.website:
        return None
    website = provider.website.strip()
    if website.startswith("local:"):
        filename = website.split(":", 1)[1]
        path = PROJECT_ROOT / "data" / "raw" / "practice_websites" / filename
        if path.exists():
            html = path.read_text(encoding="utf-8")
            parsed = parse_html_deterministic(html)
            return html, parsed.confidence
    return None


def _select_use_case(rec: Recommendation, forced: UseCase | None = None) -> UseCase:
    if forced:
        return forced
    if rec.conflict_score >= 0.35:
        return "conflict_explanation"
    if rec.risk_score >= 0.70:
        return "reviewer_summary"
    if any(change.field == "specialty" for change in rec.changes):
        return "specialty_fallback"
    return "source_evidence_summary"


def _not_attempted_reason(gating_reason: str) -> str:
    mapping = {
        "llm_mode_off": "mode_off",
        "credentials_unavailable": "missing_credentials",
        "budget_exhausted": "budget_cap_reached",
        "below_gating_thresholds": "deterministic_confidence_sufficient",
        "use_case_disabled": "no_supported_use_case",
    }
    return mapping.get(gating_reason, gating_reason)


def _build_candidate(
    rec: Recommendation,
    provider: ProviderRecord | None,
    client: LLMClient,
) -> _LLMCandidate:
    demo_use_case = LLM_DEMO_TARGETS.get(rec.provider_id)
    use_case = _select_use_case(rec, demo_use_case)
    is_demo = rec.provider_id in LLM_DEMO_TARGETS

    parser_confidence: float | None = None
    if use_case == "website_extraction" or demo_use_case == "website_extraction":
        website_info = _website_html_for_provider(provider)
        if website_info:
            _, parser_confidence = website_info
        use_case = "website_extraction"

    specialty_confidence: float | None = None
    if provider and (use_case == "specialty_fallback" or demo_use_case == "specialty_fallback"):
        specialty_confidence = get_taxonomy_lookup().fuzzy_match_confidence(provider.specialty or "")
        use_case = "specialty_fallback"

    context = LLMGatingContext(
        provider_id=rec.provider_id,
        risk_score=rec.risk_score,
        conflict_score=rec.conflict_score,
        use_case=use_case,
        parser_confidence=parser_confidence,
        specialty_confidence=specialty_confidence,
        recommended_action=rec.recommended_action,
        reason=rec.reason,
        force_enrichment=is_demo,
    )

    passes, pass_reason = client.passes_gating_rules(context)
    if is_demo:
        candidate_reason = "llm_demo_case|" + (pass_reason if passes else str(demo_use_case or use_case))
    elif passes:
        candidate_reason = pass_reason
    else:
        candidate_reason = _not_attempted_reason(pass_reason)

    priority = max(rec.risk_score, rec.conflict_score)
    if is_demo:
        priority += 10.0 - _DEMO_PRIORITY.get(rec.provider_id, 99) * 0.1

    return _LLMCandidate(
        recommendation=rec,
        provider=provider,
        use_case=use_case,
        context=context,
        candidate_reason=candidate_reason,
        is_demo=is_demo,
        priority=priority,
    )


def _build_prompt(
    candidate: _LLMCandidate,
    *,
    website_html: str | None = None,
    specialty: str | None = None,
) -> str:
    rec = candidate.recommendation
    payload: dict[str, Any] = {
        "provider_id": rec.provider_id,
        "risk_score": rec.risk_score,
        "conflict_score": rec.conflict_score,
        "recommended_action": rec.recommended_action,
        "reason": rec.reason,
        "sources": rec.supporting_sources,
        "changes": [change.model_dump(mode="json") for change in rec.changes],
    }
    if website_html:
        payload["html_excerpt"] = website_html[:4000]
    if specialty:
        payload["specialty"] = specialty
    return json.dumps(payload)


def _apply_enrichment(
    rec: Recommendation,
    use_case: UseCase,
    result: LLMExtractionResult,
    *,
    success: bool,
) -> Recommendation:
    enrichment = LLMEnrichment(
        enriched=success,
        use_case=use_case,
        llm_did_not_change_decision=True,
    )
    if use_case == "conflict_explanation":
        enrichment.conflict_explanation = result.reasoning_summary
    elif use_case == "reviewer_summary":
        enrichment.reviewer_summary = result.reasoning_summary
        if result.recommended_review_note:
            enrichment.reviewer_summary = (
                f"{result.reasoning_summary} {result.recommended_review_note}".strip()
            )
    elif use_case == "specialty_fallback":
        enrichment.specialty_hint = str(
            result.extracted_fields.get("specialty") or result.reasoning_summary
        )
    elif use_case == "website_extraction":
        enrichment.extracted_website_fields = {
            k: str(v) for k, v in result.extracted_fields.items() if v
        }
    else:
        enrichment.evidence_summary = result.reasoning_summary

    return rec.model_copy(update={"llm_enrichment": enrichment})


def _write_audit_row(
    audit_path: Path,
    *,
    mode: LLMMode,
    candidate: _LLMCandidate,
    client: LLMClient,
    attempted: bool,
    success: bool,
    error_type: str,
    http_status: str = "",
    estimated_tokens: int = 0,
    estimated_cost: float = 0.0,
    output_valid: bool = False,
    fallback_used: bool = False,
) -> None:
    append_llm_audit_row(
        audit_path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider_id": candidate.recommendation.provider_id,
            "mode": mode,
            "use_case": candidate.use_case,
            "credentials_present": client.credentials_available,
            "candidate_reason": candidate.candidate_reason,
            "attempted": attempted,
            "success": success,
            "error_type": error_type,
            "http_status": http_status,
            "model": client.api.get("model", ""),
            "estimated_tokens": estimated_tokens,
            "estimated_cost": estimated_cost,
            "output_valid": output_valid,
            "fallback_used": fallback_used,
        },
    )


def run_llm_enrichment(
    recommendations: list[Recommendation],
    providers_by_id: dict[str, ProviderRecord],
    output_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    *,
    enable_live_calls: bool | None = None,
) -> tuple[list[Recommendation], dict[str, Any]]:
    """Evaluate LLM candidates, attempt bounded live calls, apply enrichment only."""
    cfg = config or load_config()
    outputs_dir = output_dir or ensure_outputs_dir()
    audit_path = outputs_dir / "audit_llm_calls.csv"
    mode: LLMMode = cfg["llm"].get("mode", "auto")
    client = LLMClient(cfg)
    client.reset_budget(len(recommendations))

    if mode == "off":
        return recommendations, {
            "llm_mode": mode,
            "credentials_status": "n/a",
            "audit_path": str(audit_path),
            "eligible_records": 0,
            "calls_attempted": 0,
            "calls_succeeded": 0,
            "calls_failed": 0,
            "last_error_type": None,
        }

    init_llm_audit_file(audit_path, mode)
    credentials_status = "present" if client.credentials_available else "missing"
    live_calls = enable_live_calls
    if live_calls is None:
        live_calls = client.credentials_available and mode in {"auto", "force"}

    candidates = [
        _build_candidate(rec, providers_by_id.get(rec.provider_id), client)
        for rec in recommendations
    ]
    eligible = [
        c
        for c in candidates
        if c.is_demo
        or c.candidate_reason
        not in {"deterministic_confidence_sufficient", "mode_off", "no_supported_use_case"}
    ]
    call_queue = sorted(
        [c for c in eligible if c.is_demo or client.passes_gating_rules(c.context)[0]],
        key=lambda c: (-c.priority, c.recommendation.provider_id),
    )
    budget = client._budget.max_calls if client._budget else max(1, int(len(recommendations) * 0.08))
    allocated_ids = {c.recommendation.provider_id for c in call_queue[:budget]}

    enriched: dict[str, Recommendation] = {}
    last_error_type: str | None = None

    for candidate in candidates:
        rec = candidate.recommendation
        provider = candidate.provider
        should_call, gating_reason = client.should_use_llm(candidate.context)
        in_budget = rec.provider_id in allocated_ids
        call_attempted = False
        call_success = False
        output_valid = False
        fallback_used = False
        error_type = "deterministic_confidence_sufficient"
        http_status = ""
        tokens = 0
        cost = 0.0

        if candidate.is_demo or candidate.candidate_reason != "deterministic_confidence_sufficient":
            may_call = (should_call or candidate.is_demo) and live_calls and in_budget
            if not in_budget and (should_call or candidate.is_demo):
                error_type = "budget_cap_reached"
            elif may_call:
                website_info = _website_html_for_provider(provider)
                website_html = website_info[0] if website_info else None
                specialty = provider.specialty if provider else None
                deterministic_fields = None
                if website_html:
                    deterministic_fields = parse_html_deterministic(website_html).fields

                result = client.extract(
                    candidate.use_case,
                    candidate.context,
                    _build_prompt(candidate, website_html=website_html, specialty=specialty),
                    deterministic_fields=deterministic_fields,
                    sources=rec.supporting_sources,
                    specialty=specialty,
                    audit_path=None,
                    write_audit=False,
                )
                call_meta = client.last_call
                call_attempted = bool(call_meta.get("attempted"))
                fallback_used = bool(call_meta.get("fallback_used"))
                output_valid = bool(call_meta.get("success"))
                call_success = output_valid
                usage = call_meta.get("usage") or {}
                tokens = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
                http_status = str(call_meta.get("http_status") or "")
                if call_attempted and not call_success:
                    err = str(call_meta.get("error") or "")
                    error_type = "invalid_response"
                    if err:
                        from provider_intelligence.llm_diagnostics import _classify_error

                        error_type = _classify_error(err, candidate.candidate_reason, True)
                    last_error_type = error_type
                else:
                    error_type = "none"
                enriched[rec.provider_id] = _apply_enrichment(
                    rec, candidate.use_case, result, success=call_success
                )
                usage_cost = cfg["llm"].get("costs", {}).get("estimated_cost_per_call_usd", 0.002)
                cost = usage_cost if call_success else 0.0
            elif (should_call or candidate.is_demo) and not client.credentials_available:
                error_type = "missing_credentials"
                last_error_type = error_type
            elif not should_call and not candidate.is_demo:
                error_type = _not_attempted_reason(gating_reason)

        _write_audit_row(
            audit_path,
            mode=mode,
            candidate=candidate,
            client=client,
            attempted=call_attempted,
            success=call_success,
            error_type=error_type,
            http_status=http_status,
            estimated_tokens=tokens,
            estimated_cost=cost,
            output_valid=output_valid,
            fallback_used=fallback_used,
        )

    updated = [enriched.get(rec.provider_id, rec) for rec in recommendations]
    summary = summarize_llm_audit(audit_path, len(recommendations))
    use_cases = _summarize_use_cases(audit_path)
    return updated, {
        "llm_mode": mode,
        "credentials_status": credentials_status,
        "audit_path": str(audit_path),
        "eligible_records": len(eligible),
        "calls_attempted": summary["calls_attempted"],
        "calls_succeeded": summary["calls_succeeded"],
        "calls_failed": summary["calls_failed"],
        "last_error_type": summary.get("last_error_type") or last_error_type,
        "use_cases": use_cases,
    }


def _summarize_use_cases(audit_path: Path) -> dict[str, int]:
    from provider_intelligence.llm_diagnostics import read_llm_audit_csv

    df = read_llm_audit_csv(audit_path)
    if df.empty or "use_case" not in df.columns:
        return {}
    success_col = df.get("success")
    if success_col is None:
        return {}
    success_mask = success_col.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    succeeded = df[success_mask]
    counts: dict[str, int] = {}
    for use_case, group in succeeded.groupby("use_case"):
        key = str(use_case)
        if key in {"none", ""}:
            continue
        display = "specialty_normalization" if key == "specialty_fallback" else key
        counts[display] = int(len(group))
    return counts


def write_llm_diagnostics(
    recommendations: list[Recommendation],
    output_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    providers_by_id: dict[str, ProviderRecord] | None = None,
) -> dict[str, Any]:
    """Backward-compatible entry point — runs bounded enrichment when credentials exist."""
    providers_by_id = providers_by_id or {}
    _, summary = run_llm_enrichment(
        recommendations,
        providers_by_id,
        output_dir=output_dir,
        config=config,
    )
    return summary
