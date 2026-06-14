"""Provider Intelligence operational risk dashboard."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from adapters import (  # noqa: E402
    adapt_cost_estimate,
    build_quality_flags_table,
    collect_sources,
    compute_llm_share,
    describe_llm_run_status,
    format_audit_event,
    format_audit_event_label,
    get_llm_enrichment_display,
    format_money,
    format_percent,
    load_csv_safe,
    load_json_safe,
    merge_recommendation_records,
    normalize_confidence_fraction,
    parse_changes,
    parse_recommendations_payload,
    read_evaluation_metrics,
    review_queue_label,
    safe_get_action,
    safe_get_confidence,
    safe_get_conflict,
    safe_get_risk,
    score_components,
    shorten_reason,
    summarize_llm_diagnostics_for_dashboard,
)
from components import (  # noqa: E402
    audit_trail,
    decision_badge,
    field_comparison_card,
    inject_styles,
    plot_action_distribution,
    plot_confidence_distribution,
    plot_cost_breakdown,
    plot_data_quality_issues,
    plot_risk_reasons,
    reason_summary_table,
    render_demo_banner,
    render_audit_event_detail,
    render_header,
    render_llm_enrichment_block,
    render_run_summary_panels,
    render_metrics_row,
    score_breakdown,
    source_evidence,
)

PROJECT_ROOT = APP_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DIR = PROJECT_ROOT / "docs" / "sample_outputs"
REVIEWER_ACTIONS_PATH = OUTPUTS_DIR / "reviewer_actions.csv"

REVIEWER_ACTIONS = [
    "Approve",
    "Reject",
    "Needs phone verification",
    "Mark duplicate",
    "Defer",
]


def _file_map(use_samples: bool) -> dict[str, Path]:
    base = SAMPLE_DIR if use_samples else OUTPUTS_DIR
    return {
        "recommendations": base / "recommendations_sample.json" if use_samples else OUTPUTS_DIR / "recommendations.json",
        "recommendations_detailed": base / "recommendations_detailed_sample.json" if use_samples else OUTPUTS_DIR / "recommendations_detailed.json",
        "audit_log": base / "audit_log_sample.csv" if use_samples else OUTPUTS_DIR / "audit_log.csv",
        "review_queue": base / "human_review_queue_sample.csv" if use_samples else OUTPUTS_DIR / "human_review_queue.csv",
        "auto_updates": base / "auto_updates_sample.csv" if use_samples else OUTPUTS_DIR / "auto_updates.csv",
        "evaluation": base / "evaluation_metrics_sample.json" if use_samples else OUTPUTS_DIR / "evaluation_metrics.json",
        "cost": base / "cost_estimate_sample.json" if use_samples else OUTPUTS_DIR / "cost_estimate.json",
        "ground_truth": OUTPUTS_DIR / "synthetic_ground_truth.csv",
        "llm_audit": OUTPUTS_DIR / "audit_llm_calls.csv",
    }


def _outputs_available() -> bool:
    return (OUTPUTS_DIR / "recommendations.json").exists()


@st.cache_data(show_spinner=False)
def load_dashboard_data(use_samples: bool) -> dict:
    paths = _file_map(use_samples)
    competition = parse_recommendations_payload(load_json_safe(paths["recommendations"], []))
    detailed = parse_recommendations_payload(load_json_safe(paths["recommendations_detailed"], []))
    recommendations = merge_recommendation_records(competition, detailed)

    llm_audit_df = load_csv_safe(paths["llm_audit"])
    llm_calls = 0
    if not llm_audit_df.empty and "success" in llm_audit_df.columns:
        llm_calls = int(llm_audit_df["success"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    elif not llm_audit_df.empty and "called" in llm_audit_df.columns:
        llm_calls = int(llm_audit_df["called"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    elif not llm_audit_df.empty:
        llm_calls = len(llm_audit_df)

    cost_raw = load_json_safe(paths["cost"], {})
    return {
        "recommendations": recommendations,
        "audit_log": load_csv_safe(paths["audit_log"]),
        "review_queue": load_csv_safe(paths["review_queue"]),
        "auto_updates": load_csv_safe(paths["auto_updates"]),
        "evaluation": load_json_safe(paths["evaluation"], {}),
        "cost": cost_raw,
        "ground_truth": load_csv_safe(paths["ground_truth"]),
        "llm_audit": llm_audit_df,
        "llm_calls": llm_calls,
        "reviewer_actions": load_csv_safe(REVIEWER_ACTIONS_PATH),
        "use_samples": use_samples,
    }


def _recommendation_lookup(recommendations: list[dict]) -> dict[str, dict]:
    return {str(rec.get("provider_id", rec.get("npi", ""))): rec for rec in recommendations}


def _provider_audit(audit_df: pd.DataFrame, provider_id: str) -> list[dict]:
    if audit_df.empty or "provider_id" not in audit_df.columns:
        return []
    subset = audit_df[audit_df["provider_id"].astype(str) == str(provider_id)].copy()
    if "timestamp" in subset.columns:
        subset = subset.sort_values("timestamp", ascending=False)
    return subset.to_dict("records")


def _append_reviewer_action(provider_id: str, npi: str, action: str, reviewer: str, notes: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider_id": provider_id,
        "npi": npi,
        "action": action,
        "reviewer": reviewer,
        "notes": notes,
    }
    df = load_csv_safe(REVIEWER_ACTIONS_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(REVIEWER_ACTIONS_PATH, index=False)
    st.cache_data.clear()


def _avg_auto_confidence(auto_recs: list[dict]) -> str:
    if not auto_recs:
        return "N/A"
    values = [safe_get_confidence(rec) for rec in auto_recs]
    values = [v for v in values if v is not None]
    if not values:
        return "N/A"
    return format_percent(sum(values) / len(values))


def _compute_overview_metrics(data: dict) -> list[dict]:
    recs = data["recommendations"]
    total = len(recs)
    review_count = sum(1 for r in recs if safe_get_action(r) == "human_review")
    auto_count = sum(1 for r in recs if safe_get_action(r) == "auto_update")
    conf_values = [safe_get_confidence(r) for r in recs if safe_get_confidence(r) is not None]
    avg_conf = format_percent(sum(conf_values) / len(conf_values)) if conf_values else "N/A"
    risk_values = [safe_get_risk(r) for r in recs if safe_get_risk(r) is not None]
    avg_risk = sum(risk_values) / len(risk_values) if risk_values else 0.0

    eval_data = read_evaluation_metrics(data.get("evaluation", {}))
    false_auto = eval_data.get("false_auto_update_rate")
    false_auto_str = format_percent(false_auto) if false_auto is not None else "N/A"

    return [
        {"label": "Providers screened", "value": total, "delta": "Pipeline run", "delta_class": "neutral"},
        {"label": "Human review queue", "value": review_count, "delta": f"{review_count / total:.0%} of total" if total else "—", "delta_class": "negative" if review_count else "positive"},
        {"label": "Auto-updates", "value": auto_count, "delta": "High-confidence only", "delta_class": "positive"},
        {"label": "Avg confidence", "value": avg_conf, "delta": f"Avg risk {avg_risk:.2f}", "delta_class": "neutral"},
        {"label": "False auto-update rate", "value": false_auto_str, "delta": "Primary safety metric", "delta_class": "positive"},
    ]


def _benchmark_cards(data: dict) -> list[dict]:
    metrics = read_evaluation_metrics(data.get("evaluation", {}))
    benchmark_available = metrics.get("benchmark_available", False)

    def bench_value(key: str, as_percent: bool = True) -> str:
        val = metrics.get(key)
        if not benchmark_available or val is None:
            return "N/A"
        return format_percent(val) if as_percent else f"{float(val):.2f}"

    return [
        {"label": "Change detection precision", "value": bench_value("change_detection_precision")},
        {"label": "Change detection recall", "value": bench_value("change_detection_recall")},
        {"label": "Auto-update precision", "value": bench_value("auto_update_precision")},
        {"label": "False auto-update rate", "value": bench_value("false_auto_update_rate")},
        {"label": "Human review rate", "value": bench_value("human_review_rate")},
        {"label": "Cost-sensitive loss", "value": bench_value("cost_sensitive_loss", as_percent=False)},
    ]


def _cost_efficiency_cards(data: dict) -> list[dict]:
    recs = data["recommendations"]
    total = len(recs) or 1
    review_count = sum(1 for r in recs if safe_get_action(r) == "human_review")
    cost_raw = data.get("cost", {})
    cost = adapt_cost_estimate(cost_raw, mode="auto")
    llm_info = compute_llm_share(total, data.get("llm_calls", 0), cost.get("llm_budget_cap", 0.08))
    return [
        {"label": "Cost per 1,000 records", "value": format_money(cost.get("cost_per_1000_records_usd")), "delta": "Estimated pipeline cost", "delta_class": "neutral"},
        {"label": "Manual review baseline", "value": format_money(cost.get("manual_review_baseline_per_1000")), "delta": "Review-all scenario", "delta_class": "neutral"},
        {"label": "Estimated savings", "value": format_money(cost.get("estimated_savings_per_1000")), "delta": "Vs manual baseline", "delta_class": "positive"},
        {"label": "Actual LLM share", "value": format_percent(llm_info["actual_llm_share"]), "delta": f"Budget cap {format_percent(llm_info['budget_cap'])}", "delta_class": "neutral"},
        {"label": "LLM calls used", "value": llm_info["llm_calls"], "delta": "This run", "delta_class": "neutral"},
        {"label": "Human review share", "value": format_percent(review_count / total), "delta": f"{review_count} records", "delta_class": "neutral"},
    ]


def render_overview_run_summary(data: dict) -> None:
    """Styled benchmark + cost panels under charts."""
    metrics = read_evaluation_metrics(data.get("evaluation", {}))
    caption = None
    if metrics.get("benchmark_available"):
        false_rate = metrics.get("false_auto_update_rate")
        false_str = format_percent(false_rate) if false_rate is not None else "0.0%"
        caption = (
            f"Conservative routing prioritizes safe precision ({false_str} false auto-updates on synthetic benchmark). "
            "LLM enrichment is bounded and does not override deterministic decisions."
        )
    else:
        caption = "Run make demo to populate benchmark metrics and cost estimates."
    render_run_summary_panels(
        _benchmark_cards(data),
        _cost_efficiency_cards(data),
        caption=caption,
    )


def render_overview_tab(data: dict) -> None:
    recs = data["recommendations"]
    render_metrics_row(_compute_overview_metrics(data))

    chart1, chart2, chart3 = st.columns(3)
    with chart1:
        st.plotly_chart(plot_action_distribution(recs), use_container_width=True)
    with chart2:
        st.plotly_chart(plot_confidence_distribution(recs), use_container_width=True)
    with chart3:
        st.plotly_chart(plot_risk_reasons(recs), use_container_width=True)

    render_overview_run_summary(data)

    reason_df = reason_summary_table(recs)
    if not reason_df.empty:
        with st.expander("Top review reasons (full text)", expanded=False):
            st.dataframe(reason_df, use_container_width=True, hide_index=True)


def render_review_queue_tab(data: dict) -> None:
    recs = data["recommendations"]
    queue_df = data["review_queue"]
    audit_df = data["audit_log"]
    lookup = _recommendation_lookup(recs)

    review_recs = [r for r in recs if safe_get_action(r) == "human_review"]
    if not review_recs and not queue_df.empty:
        for _, row in queue_df.iterrows():
            pid = str(row.get("provider_id", ""))
            if pid in lookup:
                review_recs.append(lookup[pid])

    if not review_recs:
        st.info("No providers currently in the human review queue.")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        min_risk = st.slider("Min risk score", 0.0, 1.0, 0.0, 0.05)
    with f2:
        max_conf = st.slider("Max confidence", 0.0, 1.0, 1.0, 0.05)
    with f3:
        search = st.text_input("Search NPI or name", "")

    filtered = []
    for rec in review_recs:
        risk = safe_get_risk(rec) or 0.0
        conf = safe_get_confidence(rec) or 1.0
        name = str(rec.get("provider_name", rec.get("name", ""))).lower()
        npi = str(rec.get("npi", ""))
        if risk < min_risk or conf > max_conf:
            continue
        if search and search.lower() not in name and search not in npi and search not in str(rec.get("provider_id", "")):
            continue
        filtered.append(rec)

    if not filtered:
        st.warning("No providers match the current filters.")
        return

    options = {review_queue_label(r): str(r.get("provider_id", r.get("npi", ""))) for r in filtered}
    selected_label = st.selectbox("Select provider", list(options.keys()))
    provider_id = options[selected_label]
    rec = lookup.get(provider_id, filtered[0])

    risk = safe_get_risk(rec)
    conf = safe_get_confidence(rec)
    conflict = safe_get_conflict(rec)
    action = safe_get_action(rec) or "human_review"

    risk_text = f"{risk:.2f}" if risk is not None else "—"
    conf_text = format_percent(conf) if conf is not None else "—"
    conflict_text = f"{conflict:.2f}" if conflict is not None else "—"
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">Provider Summary</div>
                <p><strong>{rec.get('provider_name', rec.get('name', 'Unknown'))}</strong></p>
                <p>ID: <code>{rec.get('provider_id', '—')}</code> &nbsp;·&nbsp; NPI: <code>{rec.get('npi', '—')}</code></p>
                <p>{decision_badge(action)}</p>
                {"<p><span class='badge badge-review'>LLM enriched</span></p>" if get_llm_enrichment_display(rec).get("enriched") else ""}
                <p>Risk: <strong>{risk_text}</strong> &nbsp;·&nbsp;
                   Confidence: <strong>{conf_text}</strong> &nbsp;·&nbsp;
                   Conflict: <strong>{conflict_text}</strong></p>
                <p style="color:#64748B;font-size:0.88rem;">{rec.get('reason', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        enrichment = get_llm_enrichment_display(rec)
        if enrichment.get("enriched"):
            render_llm_enrichment_block(enrichment)
        changes = parse_changes(rec.get("changes"))
        if changes:
            st.markdown("##### Field comparison")
            for change in changes:
                field_comparison_card(change)
        else:
            st.caption("No proposed field changes for this provider.")

    with right:
        risk_c = score_components(rec, "risk")
        conf_c = score_components(rec, "confidence")
        conflict_c = score_components(rec, "conflict")
        if risk_c:
            score_breakdown("Risk Score Breakdown", risk_c, "risk")
        elif risk is not None:
            score_breakdown("Risk Score Breakdown", {"overall_risk": risk}, "risk")
        if conf_c:
            score_breakdown("Confidence Breakdown", conf_c, "confidence")
        elif conf is not None:
            score_breakdown("Confidence Breakdown", {"overall_confidence": conf}, "confidence")
        if conflict_c:
            score_breakdown("Conflict Score Breakdown", conflict_c, "conflict")
        elif conflict is not None:
            score_breakdown("Conflict Score Breakdown", {"overall_conflict": conflict}, "conflict")
        if not (risk_c or conf_c or conflict_c) and risk is None and conf is None:
            st.caption("Component breakdown unavailable; showing top-level scores only.")
        source_evidence(collect_sources(rec))

    audit_trail(_provider_audit(audit_df, provider_id), compact=True)

    st.markdown("---")
    st.markdown("##### Reviewer actions")
    reviewer = st.text_input("Reviewer name", value="analyst@healthorg.org", key="reviewer_name")
    notes = st.text_area("Notes (optional)", key="review_notes")
    btn_cols = st.columns(len(REVIEWER_ACTIONS))
    for col, action in zip(btn_cols, REVIEWER_ACTIONS, strict=True):
        if col.button(action, key=f"action_{action}", use_container_width=True):
            _append_reviewer_action(provider_id, str(rec.get("npi", "")), action, reviewer, notes)
            st.success(f"Recorded {action} for provider {provider_id}.")
            st.rerun()


def render_auto_updates_tab(data: dict) -> None:
    recs = data["recommendations"]
    auto_df = data["auto_updates"]
    auto_recs = [r for r in recs if safe_get_action(r) == "auto_update"]

    render_metrics_row([
        {"label": "Auto-updated", "value": len(auto_recs), "delta": "High-confidence changes", "delta_class": "positive"},
        {"label": "Avg confidence", "value": _avg_auto_confidence(auto_recs), "delta": "Threshold ≥ 88%", "delta_class": "neutral"},
        {"label": "Fields changed", "value": sum(len(parse_changes(r.get("changes"))) for r in auto_recs), "delta": "Across all auto-updates", "delta_class": "neutral"},
    ])

    display_cols = [
        "provider_id", "npi", "field", "old_value", "new_value",
        "overall_confidence", "recommended_action", "reason",
    ]
    rows: list[dict] = []
    if not auto_df.empty:
        for _, row in auto_df.iterrows():
            changes = parse_changes(row.get("changes_json"))
            if changes:
                for change in changes:
                    rows.append({
                        "provider_id": row.get("provider_id"),
                        "npi": row.get("npi"),
                        "field": change.get("field"),
                        "old_value": change.get("old_value"),
                        "new_value": change.get("new_value"),
                        "overall_confidence": row.get("overall_confidence"),
                        "recommended_action": row.get("recommended_action"),
                        "reason": shorten_reason(str(row.get("reason", ""))),
                    })
            else:
                rows.append({k: row.get(k) for k in display_cols if k in row})
    elif auto_recs:
        for rec in auto_recs:
            for change in parse_changes(rec.get("changes")):
                rows.append({
                    "provider_id": rec.get("provider_id"),
                    "npi": rec.get("npi"),
                    "field": change.get("field"),
                    "old_value": change.get("old_value"),
                    "new_value": change.get("new_value"),
                    "overall_confidence": safe_get_confidence(rec),
                    "recommended_action": safe_get_action(rec),
                    "reason": shorten_reason(str(rec.get("reason", ""))),
                })

    if not rows:
        st.info("No auto-updates in the current dataset.")
        return

    table_df = pd.DataFrame(rows)
    hide_cols = [c for c in table_df.columns if c.lower().endswith("_json") or "changes_json" in c.lower()]
    table_df = table_df.drop(columns=hide_cols, errors="ignore")
    if "overall_confidence" in table_df.columns:
        table_df["confidence_display"] = table_df["overall_confidence"].apply(
            lambda v: format_percent(normalize_confidence_fraction(v))
        )
        table_df["confidence_pct"] = table_df["overall_confidence"].apply(
            lambda v: normalize_confidence_fraction(v) * 100
        )
    display_df = table_df.drop(columns=["overall_confidence", "confidence_display"], errors="ignore")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence_pct": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "confidence_display": st.column_config.TextColumn("Confidence %"),
        },
    )

    options = table_df["provider_id"].astype(str).unique().tolist()
    selected = st.selectbox("Select auto-update detail", options)
    detail = next((r for r in auto_recs if str(r.get("provider_id")) == selected), None)
    if detail:
        st.markdown("##### Selected auto-update detail")
        st.markdown(
            f"**Provider:** {detail.get('provider_id')} · **NPI:** {detail.get('npi')} · "
            f"**Action:** {safe_get_action(detail)} · **Confidence:** {_avg_auto_confidence([detail])}"
        )
        st.caption(detail.get("reason", ""))
        for change in parse_changes(detail.get("changes")):
            field_comparison_card(change)
        source_evidence(collect_sources(detail))
        with st.expander("Raw record (debug)"):
            st.json({k: v for k, v in detail.items() if not str(k).endswith("_json")})


def render_data_quality_tab(data: dict) -> None:
    recs = data["recommendations"]
    quality_counts, flag_rows = build_quality_flags_table(recs, data.get("ground_truth"))
    render_metrics_row([
        {"label": "Invalid NPI", "value": quality_counts["invalid_npi"], "delta": "Checksum / format failures", "delta_class": "negative"},
        {"label": "Missing phone", "value": quality_counts["missing_phone"], "delta": "Contact gaps", "delta_class": "negative"},
        {"label": "Missing address", "value": quality_counts["missing_address"], "delta": "Location gaps", "delta_class": "negative"},
        {"label": "Stale records", "value": quality_counts["stale_records"], "delta": "> 24 months since verify", "delta_class": "negative"},
        {"label": "Duplicate risk", "value": quality_counts["duplicates"], "delta": "Flagged duplicates", "delta_class": "negative"},
    ])
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(plot_data_quality_issues(quality_counts), use_container_width=True)
    with c2:
        st.markdown("##### Quality flags by provider")
        if flag_rows:
            st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No quality flags detected in current records.")


def render_cost_model_tab(data: dict) -> None:
    cost_raw = data.get("cost", {})
    recs = data["recommendations"]
    total = len(recs) or int(cost_raw.get("record_count", 0)) or 1
    review_count = sum(1 for r in recs if safe_get_action(r) == "human_review")

    st.sidebar.markdown("### Cost assumptions")
    default_mode = str(cost_raw.get("assumptions", {}).get("llm_mode", "auto"))
    if default_mode not in {"off", "auto", "force"}:
        default_mode = "auto"
    mode_index = ["off", "auto", "force"].index(default_mode)
    llm_mode = st.sidebar.selectbox("LLM mode (scenario)", ["off", "auto", "force"], index=mode_index)
    cost = adapt_cost_estimate(cost_raw, mode=llm_mode)
    llm_summary = summarize_llm_diagnostics_for_dashboard(
        data.get("llm_audit", pd.DataFrame()),
        total,
        cost_raw,
        llm_mode=llm_mode,
    )
    scenario_calls = cost.get("estimated_llm_calls", 0)

    render_metrics_row([
        {"label": "Normalized cost / 1k", "value": format_money(cost.get("cost_per_1000_records_usd")), "delta": f"Scenario mode: {llm_mode}", "delta_class": "neutral"},
        {"label": "Actual LLM share", "value": format_percent(llm_summary.get("actual_llm_share", 0)), "delta": f"Budget cap {format_percent(llm_summary.get('budget_cap', 0.08))}", "delta_class": "neutral"},
        {"label": "Calls succeeded", "value": llm_summary.get("calls_succeeded", 0), "delta": f"Attempted {llm_summary.get('calls_attempted', 0)}", "delta_class": "neutral"},
        {"label": "Human review share", "value": format_percent(review_count / total), "delta": f"{review_count} records", "delta_class": "neutral"},
    ])

    c1, c2 = st.columns(2)
    calls_succeeded = int(llm_summary.get("calls_succeeded", 0) or 0)
    llm_status = describe_llm_run_status(llm_summary)
    with c1:
        st.plotly_chart(
            plot_cost_breakdown(cost, calls_succeeded=calls_succeeded),
            use_container_width=True,
        )
        if calls_succeeded == 0:
            st.caption("LLM enrichment bar reflects scenario planning cost, not actual API spend this run.")
    with c2:
        st.markdown("##### Cost model detail")
        st.markdown(f"- **Current run total:** {format_money(cost.get('total_estimated_cost_usd'))} for {total} records")
        st.markdown(f"- **Normalized cost:** {format_money(cost.get('cost_per_1000_records_usd'))} per 1,000 records *(scenario model)*")
        st.markdown(f"- **Manual baseline:** {format_money(cost.get('manual_review_baseline_per_1000'))} per 1,000 records")
        st.markdown(f"- **Estimated savings:** {format_money(cost.get('estimated_savings_per_1000'))} per 1,000 records")
        st.markdown("##### LLM diagnostics (this run)")
        st.markdown(f"- **LLM_MODE:** `{llm_summary.get('llm_mode', llm_mode)}`")
        st.markdown(f"- **Credentials:** {llm_summary.get('credentials_status', 'unknown')}")
        st.markdown(f"- **LLM candidates detected:** {llm_summary.get('eligible_records', 0)}")
        st.markdown(f"- **Calls attempted:** {llm_summary.get('calls_attempted', 0)}")
        st.markdown(f"- **Calls succeeded:** {llm_summary.get('calls_succeeded', 0)}")
        st.markdown(f"- **Calls failed:** {llm_summary.get('calls_failed', 0)}")
        st.markdown(f"- **Actual LLM share:** {format_percent(llm_summary.get('actual_llm_share', 0))}")
        st.markdown(f"- **Attempted LLM share:** {format_percent(llm_summary.get('attempted_llm_share', 0))}")
        st.markdown(f"- **LLM budget cap:** {format_percent(llm_summary.get('budget_cap', 0.08))}")
        st.markdown(f"- **LLM status:** {llm_status['status']}")
        use_cases = llm_summary.get("use_cases") or {}
        if use_cases:
            st.markdown("- **Main use cases used:**")
            for name, count in sorted(use_cases.items()):
                st.markdown(f"  - `{name}`: {count}")
        st.markdown(f"- **Scenario LLM calls (mode={llm_mode}):** {scenario_calls:.1f}")
        if llm_status["note"]:
            st.caption(llm_status["note"])
        if llm_status["show_last_error"] and llm_summary.get("last_error_type"):
            st.markdown(f"- **Last error:** `{llm_summary['last_error_type']}`")
        if cost.get("notes"):
            st.caption(cost["notes"])


def render_audit_log_tab(data: dict) -> None:
    audit_df = data["audit_log"]
    actions_df = data["reviewer_actions"]
    if audit_df.empty:
        st.info("No audit log entries available.")
        return

    audit_df = audit_df.copy()
    audit_df["event_text"] = audit_df.apply(lambda row: format_audit_event(row.to_dict()), axis=1)

    f1, f2, f3 = st.columns(3)
    with f1:
        steps = ["All"] + sorted(audit_df["step"].dropna().unique().tolist()) if "step" in audit_df.columns else ["All"]
        selected_step = st.selectbox("Step", steps)
    with f2:
        providers = ["All"] + sorted(audit_df["provider_id"].astype(str).unique().tolist())
        selected_provider = st.selectbox("Provider", providers)
    with f3:
        search = st.text_input("Search audit log", key="audit_search")

    filtered = audit_df.copy()
    if selected_step != "All":
        filtered = filtered[filtered["step"] == selected_step]
    if selected_provider != "All":
        filtered = filtered[filtered["provider_id"].astype(str) == selected_provider]
    if search:
        mask = filtered.astype(str).apply(lambda row: search.lower() in row.str.lower().str.cat(sep=" "), axis=1)
        filtered = filtered[mask]

    display_df = filtered[["timestamp", "provider_id", "step", "rule_name", "decision", "event_text"]].copy() if "event_text" in filtered.columns else filtered
    render_metrics_row([
        {"label": "Audit events", "value": len(filtered), "delta": f"{len(audit_df)} total", "delta_class": "neutral"},
        {"label": "Reviewer actions", "value": len(actions_df), "delta": "Logged decisions", "delta_class": "neutral"},
        {"label": "Unique providers", "value": filtered["provider_id"].nunique() if "provider_id" in filtered.columns else "—", "delta": "Filtered view", "delta_class": "neutral"},
    ])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if not filtered.empty:
        st.markdown("##### Selected audit event detail")
        event_rows = filtered.reset_index(drop=True)
        labels = [format_audit_event_label(row.to_dict(), i) for i, row in event_rows.iterrows()]
        selected = st.selectbox("Select event", range(len(labels)), format_func=lambda i: labels[i], key="audit_event_select")
        render_audit_event_detail(event_rows.iloc[selected].to_dict())

    if not actions_df.empty:
        st.markdown("##### Reviewer action log")
        st.dataframe(actions_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Provider Intelligence", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")
    inject_styles(str(APP_DIR / "style.css"))

    runtime_available = _outputs_available()
    sample_available = (SAMPLE_DIR / "recommendations_sample.json").exists()
    use_samples = not runtime_available and sample_available
    data = load_dashboard_data(use_samples)

    render_header(
        "Operational Risk Dashboard",
        "Trust-aware monitoring for healthcare provider directory accuracy",
    )

    if not runtime_available:
        render_demo_banner(has_samples=sample_available)

    if not runtime_available and not sample_available:
        st.stop()

    tabs = st.tabs(["Overview", "Review Queue", "Auto Updates", "Data Quality", "Cost Model", "Audit Log"])
    with tabs[0]:
        render_overview_tab(data)
    with tabs[1]:
        render_review_queue_tab(data)
    with tabs[2]:
        render_auto_updates_tab(data)
    with tabs[3]:
        render_data_quality_tab(data)
    with tabs[4]:
        render_cost_model_tab(data)
    with tabs[5]:
        render_audit_log_tab(data)


if __name__ == "__main__":
    main()
