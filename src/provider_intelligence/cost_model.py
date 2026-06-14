"""Operational cost estimation for provider intelligence pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from provider_intelligence.config import ensure_outputs_dir, load_config


LLMMode = Literal["off", "auto", "force"]


def _llm_calls_per_1000(mode: LLMMode, config: dict[str, Any]) -> float:
    """Estimate LLM calls per 1,000 records for a given mode."""
    gating = config["llm"]["gating"]
    max_share = gating.get("max_record_share", 0.08)

    if mode == "off":
        return 0.0
    if mode == "force":
        return 1000.0 * max_share * 3
    return 1000.0 * max_share * 1.5


def estimate_mode_costs(
    record_count: int,
    mode: LLMMode,
    config: dict[str, Any] | None = None,
    action_distribution: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate costs for a single LLM mode."""
    cfg = config or load_config()
    costs = cfg["costs"]
    llm_cfg = cfg["llm"]

    dist = action_distribution or {
        "auto_update": 0.05,
        "human_review": 0.25,
        "no_change": 0.65,
        "do_not_update": 0.05,
    }

    human_review_count = record_count * dist.get("human_review", 0.25)
    auto_update_count = record_count * dist.get("auto_update", 0.05)
    llm_calls = _llm_calls_per_1000(mode, cfg) * (record_count / 1000.0)
    cost_per_call = llm_cfg.get("costs", {}).get("estimated_cost_per_call_usd", 0.002)

    human_review_cost = human_review_count * costs["human_review_per_record"]
    llm_cost = llm_calls * cost_per_call
    wrong_auto_risk_cost = auto_update_count * 0.02 * costs["wrong_auto_update"]
    missed_update_risk_cost = human_review_count * 0.10 * costs["missed_update"]
    compute_cost = record_count * 0.001

    total = human_review_cost + llm_cost + wrong_auto_risk_cost + missed_update_risk_cost + compute_cost

    return {
        "mode": mode,
        "record_count": record_count,
        "estimated_llm_calls": round(llm_calls, 2),
        "cost_breakdown_usd": {
            "human_review": round(human_review_cost, 2),
            "llm_assistance": round(llm_cost, 2),
            "wrong_auto_update_risk": round(wrong_auto_risk_cost, 2),
            "missed_update_risk": round(missed_update_risk_cost, 2),
            "compute_overhead": round(compute_cost, 2),
        },
        "total_estimated_cost_usd": round(total, 2),
        "cost_per_1000_records_usd": round(total * 1000 / record_count, 2) if record_count else 0.0,
    }


def estimate_all_modes(
    record_count: int = 1000,
    config: dict[str, Any] | None = None,
    action_distribution: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate costs across off, auto, and force LLM modes."""
    cfg = config or load_config()
    modes: list[LLMMode] = ["off", "auto", "force"]
    by_mode = {
        mode: estimate_mode_costs(record_count, mode, cfg, action_distribution) for mode in modes
    }

    off_total = by_mode["off"]["total_estimated_cost_usd"]
    auto_total = by_mode["auto"]["total_estimated_cost_usd"]
    force_total = by_mode["force"]["total_estimated_cost_usd"]
    manual_baseline_per_1000 = record_count and round(
        (record_count * cfg["costs"]["human_review_per_record"]) * 1000 / record_count, 2
    ) or 0.0
    auto_per_1000 = by_mode["auto"]["cost_per_1000_records_usd"]
    savings_per_1000 = round(max(manual_baseline_per_1000 - auto_per_1000, 0), 2)

    return {
        "record_count": record_count,
        "manual_review_baseline_per_1000": manual_baseline_per_1000,
        "estimated_savings_per_1000": savings_per_1000,
        "assumptions": {
            "human_review_cost_per_record": cfg["costs"]["human_review_per_record"],
            "wrong_auto_update_cost": cfg["costs"]["wrong_auto_update"],
            "missed_update_cost": cfg["costs"]["missed_update"],
            "llm_cost_per_call": cfg["llm"].get("costs", {}).get("estimated_cost_per_call_usd", 0.002),
            "llm_max_record_share": cfg["llm"]["gating"].get("max_record_share", 0.08),
            "action_distribution": action_distribution
            or {
                "auto_update": 0.05,
                "human_review": 0.25,
                "no_change": 0.65,
                "do_not_update": 0.05,
            },
        },
        "by_mode": by_mode,
        "comparison": {
            "auto_vs_off_delta_usd": round(auto_total - off_total, 2),
            "force_vs_auto_delta_usd": round(force_total - auto_total, 2),
            "recommended_mode": "auto",
            "notes": (
                "LLM_MODE=auto adds bounded LLM cost while preserving deterministic decisions. "
                "LLM_MODE=force is for experimentation only."
            ),
        },
    }


def write_cost_estimate(
    output_dir: Path | None = None,
    record_count: int | None = None,
) -> dict[str, Any]:
    """Write cost_estimate.json to outputs directory."""
    config = load_config()
    outputs_dir = output_dir or ensure_outputs_dir()
    count = record_count or config["app"]["demo"]["record_count"]
    estimate = estimate_all_modes(record_count=count, config=config)
    out_path = outputs_dir / "cost_estimate.json"
    out_path.write_text(json.dumps(estimate, indent=2), encoding="utf-8")
    estimate["output_path"] = str(out_path)
    return estimate
