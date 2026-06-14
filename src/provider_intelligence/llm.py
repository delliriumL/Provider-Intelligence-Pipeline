"""Adaptive LLM assistance with gating, validation, and deterministic fallbacks.

The deterministic core owns all decisions. LLM output is enrichment-only and
must never approve auto-updates or override decision engine thresholds.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from provider_intelligence.config import ensure_outputs_dir, load_config
from provider_intelligence.schemas import LLMExtractionResult
from provider_intelligence.taxonomy import get_taxonomy_lookup

logger = logging.getLogger(__name__)

LLMMode = Literal["off", "auto", "force"]
UseCase = Literal[
    "website_extraction",
    "conflict_explanation",
    "specialty_fallback",
    "reviewer_summary",
    "source_evidence_summary",
]


@dataclass
class LLMGatingContext:
    """Context for deciding whether an LLM call is permitted."""

    provider_id: str
    risk_score: float
    conflict_score: float
    use_case: UseCase
    parser_confidence: float | None = None
    specialty_confidence: float | None = None
    recommended_action: str | None = None
    reason: str | None = None
    force_enrichment: bool = False
    priority: float = 0.0

    def __post_init__(self) -> None:
        if self.priority == 0.0:
            self.priority = max(self.risk_score, self.conflict_score)


@dataclass
class LLMGatingBudget:
    """Tracks LLM call budget across a pipeline run."""

    max_share: float
    total_records: int
    calls_made: int = 0
    _eligible_queue: list[LLMGatingContext] = field(default_factory=list)

    @property
    def max_calls(self) -> int:
        return max(1, int(self.total_records * self.max_share))

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.calls_made)

    def record_call(self) -> None:
        self.calls_made += 1


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    """Safely parse JSON from LLM response text."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def validate_llm_result(payload: dict[str, Any] | None) -> LLMExtractionResult | None:
    """Validate structured LLM output with Pydantic; return None if invalid."""
    if not payload:
        return None
    try:
        return LLMExtractionResult.model_validate(payload)
    except Exception:
        logger.debug("LLM output failed Pydantic validation", exc_info=True)
        return None


class LLMClient:
    """Mode-aware LLM client with gating and audit logging."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.mode: LLMMode = self.config["llm"].get("mode", "auto")
        self.gating = self.config["llm"]["gating"]
        self.api = self.config["llm"]["api"]
        self.use_cases = self.config["llm"].get("use_cases", {})
        self._budget: LLMGatingBudget | None = None
        self._openai_client: Any = None
        self.last_call: dict[str, Any] = {}

    def reset_budget(self, total_records: int) -> None:
        """Initialize per-run LLM call budget."""
        self._budget = LLMGatingBudget(
            max_share=self.gating.get("max_record_share", 0.08),
            total_records=total_records,
        )

    @property
    def credentials_available(self) -> bool:
        """Return True when API credentials are configured."""
        return bool(self.api.get("base_url") and self.api.get("api_key") and self.api.get("model"))

    def _can_call_network(self) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "force":
            return self.credentials_available
        return self.credentials_available

    def passes_gating_rules(self, context: LLMGatingContext) -> tuple[bool, str]:
        """Evaluate whether context meets LLM gating thresholds."""
        if self.mode == "off":
            return False, "llm_mode_off"

        min_risk = self.gating.get("min_risk_for_call", 0.70)
        min_conflict = self.gating.get("min_conflict_for_call", 0.35)
        min_parser = self.gating.get("min_parser_confidence", 0.60)
        min_specialty = self.gating.get("min_specialty_confidence", 0.70)

        reasons: list[str] = []
        if context.force_enrichment and self.mode != "off":
            reasons.append("llm_demo_case")
        if context.risk_score >= min_risk:
            reasons.append("risk_threshold")
        if context.conflict_score >= min_conflict:
            reasons.append("conflict_threshold")
        if context.parser_confidence is not None and context.parser_confidence < min_parser:
            reasons.append("low_parser_confidence")
        if context.specialty_confidence is not None and context.specialty_confidence < min_specialty:
            reasons.append("low_specialty_confidence")
        if context.recommended_action == "human_review" and context.reason:
            reason_lower = context.reason.lower()
            if "conflict" in reason_lower or "disagree" in reason_lower or "source" in reason_lower:
                reasons.append("human_review_conflict")

        if not reasons and self.mode != "force":
            return False, "below_gating_thresholds"

        if self._budget and self._budget.remaining_calls <= 0 and self.mode != "force":
            return False, "budget_exhausted"

        if not self.use_cases.get(context.use_case, True):
            return False, "use_case_disabled"

        return True, "|".join(reasons) if reasons else "force_mode"

    def should_use_llm(self, context: LLMGatingContext) -> tuple[bool, str]:
        """Determine if LLM should be invoked for a gated use case."""
        if self.mode == "off":
            return False, "llm_mode_off"
        passes, reason = self.passes_gating_rules(context)
        if not passes:
            return False, reason
        if not self._can_call_network():
            return False, "credentials_unavailable"
        return True, reason

    def _get_openai_client(self) -> Any:
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(
                base_url=self.api["base_url"],
                api_key=self.api["api_key"],
            )
        return self._openai_client

    def _call_api(self, prompt: str, system: str) -> tuple[str | None, dict[str, int], str | None]:
        """Call OpenAI-compatible API; return content, token usage, error."""
        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model=self.api["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            }
            return content, usage, None
        except Exception as exc:
            logger.warning("LLM API call failed: %s", exc)
            return None, {"prompt_tokens": 0, "completion_tokens": 0}, str(exc)

    def deterministic_fallback(
        self,
        use_case: UseCase,
        context: LLMGatingContext,
        *,
        deterministic_fields: dict[str, Any] | None = None,
        sources: list[str] | None = None,
        specialty: str | None = None,
    ) -> LLMExtractionResult:
        """Produce deterministic enrichment when LLM is unavailable or invalid."""
        lookup = get_taxonomy_lookup()

        if use_case == "website_extraction":
            fields = deterministic_fields or {}
            return LLMExtractionResult(
                extracted_fields=fields,
                reasoning_summary="Deterministic HTML parse (LLM not used).",
                confidence_hint=0.55,
                evidence_snippets=[str(v) for v in fields.values() if v][:3],
            )

        if use_case == "conflict_explanation":
            return LLMExtractionResult(
                extracted_fields={},
                reasoning_summary=(
                    f"Conflict score {context.conflict_score:.2f} with risk "
                    f"{context.risk_score:.2f}. Sources disagree on material fields."
                ),
                confidence_hint=0.5,
                evidence_snippets=sources or [],
                recommended_review_note="Verify conflicting fields against primary sources.",
            )

        if use_case == "specialty_fallback":
            mapped = lookup.fuzzy_match_description(specialty or "", threshold=75)
            code = lookup.fuzzy_match_code(specialty or "", threshold=75)
            return LLMExtractionResult(
                extracted_fields={"specialty": mapped, "taxonomy_code": code},
                reasoning_summary=f"Taxonomy fuzzy match for '{specialty or ''}'.",
                confidence_hint=0.65 if mapped else 0.3,
                evidence_snippets=[mapped] if mapped else [],
            )

        if use_case == "reviewer_summary":
            return LLMExtractionResult(
                extracted_fields={
                    "risk_score": context.risk_score,
                    "conflict_score": context.conflict_score,
                },
                reasoning_summary=(
                    f"Provider {context.provider_id}: risk={context.risk_score:.2f}, "
                    f"conflict={context.conflict_score:.2f}. Route to human review."
                ),
                confidence_hint=0.6,
                recommended_review_note="Review score breakdown and supporting sources.",
            )

        if use_case == "source_evidence_summary":
            source_list = sources or []
            return LLMExtractionResult(
                extracted_fields={"sources": source_list},
                reasoning_summary=(
                    f"{len(source_list)} sources consulted. "
                    "Deterministic merge applied; LLM summary unavailable."
                ),
                confidence_hint=0.5,
                evidence_snippets=source_list[:5],
            )

        return LLMExtractionResult(
            reasoning_summary="Deterministic fallback applied.",
            confidence_hint=0.0,
        )

    def extract(
        self,
        use_case: UseCase,
        context: LLMGatingContext,
        prompt: str,
        *,
        system_prompt: str | None = None,
        deterministic_fields: dict[str, Any] | None = None,
        sources: list[str] | None = None,
        specialty: str | None = None,
        audit_path: Path | None = None,
        write_audit: bool = True,
    ) -> LLMExtractionResult:
        """Run gated LLM extraction with validation and deterministic fallback."""
        should_call, gating_reason = self.should_use_llm(context)

        def fallback() -> LLMExtractionResult:
            return self.deterministic_fallback(
                use_case,
                context,
                deterministic_fields=deterministic_fields,
                sources=sources,
                specialty=specialty,
            )

        if not should_call:
            result = fallback()
            self.last_call = {
                "attempted": False,
                "success": False,
                "fallback_used": True,
                "error": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "http_status": "",
            }
            if write_audit:
                self._write_audit_row(
                    audit_path,
                    context,
                    use_case,
                    called=False,
                    fallback_used=True,
                    gating_reason=gating_reason,
                    validation_ok=True,
                )
            return result

        if self.mode == "force":
            logger.warning("LLM_MODE=force: not cost-optimized; decisions remain deterministic.")

        system = system_prompt or (
            "Return JSON matching LLMExtractionResult schema. "
            "Do not recommend auto-updates. Enrichment only."
        )
        content, usage, error = self._call_api(prompt, system)
        payload = _parse_json_payload(content or "")
        validated = validate_llm_result(payload)

        if validated is None:
            result = fallback()
            self.last_call = {
                "attempted": True,
                "success": False,
                "fallback_used": True,
                "error": error or "validation_failed",
                "usage": usage,
                "http_status": "",
            }
            if write_audit:
                self._write_audit_row(
                    audit_path,
                    context,
                    use_case,
                    called=True,
                    fallback_used=True,
                    gating_reason=gating_reason,
                    validation_ok=False,
                    usage=usage,
                    error=error or "validation_failed",
                )
            return result

        if self._budget:
            self._budget.record_call()

        self.last_call = {
            "attempted": True,
            "success": True,
            "fallback_used": False,
            "error": None,
            "usage": usage,
            "http_status": "200",
        }
        if write_audit:
            self._write_audit_row(
                audit_path,
                context,
                use_case,
                called=True,
                fallback_used=False,
                gating_reason=gating_reason,
                validation_ok=True,
                usage=usage,
            )
        return validated

    def _write_audit_row(
        self,
        audit_path: Path | None,
        context: LLMGatingContext,
        use_case: UseCase,
        *,
        called: bool,
        fallback_used: bool,
        gating_reason: str,
        validation_ok: bool,
        usage: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        """Append a row to audit_llm_calls.csv using the diagnostics schema."""
        from provider_intelligence.llm_diagnostics import _classify_error, append_llm_audit_row

        path = audit_path or ensure_outputs_dir() / "audit_llm_calls.csv"
        usage = usage or {"prompt_tokens": 0, "completion_tokens": 0}
        cost_per_call = self.config["llm"].get("costs", {}).get("estimated_cost_per_call_usd", 0.002)
        estimated_cost = cost_per_call if called and not fallback_used else 0.0
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        success = called and not fallback_used and validation_ok and not error
        error_type = _classify_error(error, gating_reason, called)
        if called and fallback_used:
            error_type = "fallback_used"
        elif called and not validation_ok:
            error_type = "invalid_response"

        append_llm_audit_row(
            path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider_id": context.provider_id,
                "mode": self.mode,
                "use_case": use_case,
                "credentials_present": self.credentials_available,
                "candidate_reason": gating_reason,
                "attempted": called,
                "success": success,
                "error_type": error_type if not success else "none",
                "http_status": "",
                "model": self.api.get("model", ""),
                "estimated_tokens": prompt_tokens + completion_tokens,
                "estimated_cost": round(estimated_cost, 6),
                "output_valid": success,
                "fallback_used": fallback_used,
            },
        )

    def enrich_recommendation_explanation(
        self,
        provider_id: str,
        risk_score: float,
        conflict_score: float,
        reason: str,
        sources: list[str],
        *,
        audit_path: Path | None = None,
    ) -> str:
        """Add reviewer-facing explanation text without changing decisions."""
        context = LLMGatingContext(
            provider_id=provider_id,
            risk_score=risk_score,
            conflict_score=conflict_score,
            use_case="conflict_explanation",
        )
        result = self.extract(
            "conflict_explanation",
            context,
            prompt=json.dumps(
                {
                    "provider_id": provider_id,
                    "risk_score": risk_score,
                    "conflict_score": conflict_score,
                    "existing_reason": reason,
                    "sources": sources,
                }
            ),
            sources=sources,
            audit_path=audit_path,
        )
        if result.reasoning_summary and result.reasoning_summary != reason:
            return f"{reason} | LLM note: {result.reasoning_summary}"
        return reason
