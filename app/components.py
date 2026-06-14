"""Reusable UI components for the Provider Intelligence Streamlit dashboard."""

from __future__ import annotations

import html
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def inject_styles(css_path: str) -> None:
    """Load and inject custom CSS."""
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_summary_panels(
    benchmark_metrics: list[dict[str, Any]],
    cost_metrics: list[dict[str, Any]],
    *,
    caption: str | None = None,
) -> None:
    """Benchmark + cost summary with equal panel and KPI card sizes."""
    st.markdown("#### Run summary")
    with st.container(border=True):
        head_left, head_right = st.columns(2, gap="medium")
        head_left.markdown("**Benchmark**")
        head_right.markdown("**Cost & LLM**")

        row_count = max(
            (len(benchmark_metrics) + 1) // 2,
            (len(cost_metrics) + 1) // 2,
        )
        for row in range(row_count):
            panel_left, panel_right = st.columns(2, gap="medium")
            with panel_left:
                card_l, card_r = st.columns(2, gap="small")
                for col, idx in ((card_l, row * 2), (card_r, row * 2 + 1)):
                    if idx < len(benchmark_metrics):
                        _render_metric_card(
                            col,
                            benchmark_metrics[idx],
                            card_class="metric-card compact uniform",
                            uniform=True,
                        )
            with panel_right:
                card_l, card_r = st.columns(2, gap="small")
                for col, idx in ((card_l, row * 2), (card_r, row * 2 + 1)):
                    if idx < len(cost_metrics):
                        _render_metric_card(
                            col,
                            cost_metrics[idx],
                            card_class="metric-card compact uniform",
                            uniform=True,
                        )
    if caption:
        st.caption(caption)


def render_demo_banner(has_samples: bool = False) -> None:
    if has_samples:
        st.markdown(
            """
            <div class="demo-banner">
                <strong>Showing committed sample outputs.</strong>
                Run <code>make demo</code> to generate fresh runtime outputs in <code>outputs/</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="demo-banner">
                <strong>Run the demo pipeline first:</strong> <code>make demo</code>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metrics_row(metrics: list[dict[str, Any]]) -> None:
    """Render a row of KPI metric cards."""
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics, strict=True):
        _render_metric_card(col, metric)


def render_metrics_grid(metrics: list[dict[str, Any]], *, columns: int = 3, compact: bool = False) -> None:
    """Render KPI cards in a wrapped grid (for side-by-side summary panels)."""
    card_class = "metric-card compact" if compact else "metric-card"
    for start in range(0, len(metrics), columns):
        chunk = metrics[start : start + columns]
        cols = st.columns(columns)
        for col, metric in zip(cols, chunk, strict=False):
            _render_metric_card(col, metric, card_class=card_class)


def _render_metric_card(
    col: Any,
    metric: dict[str, Any],
    *,
    card_class: str = "metric-card",
    uniform: bool = False,
) -> None:
    delta = metric.get("delta")
    delta_class = metric.get("delta_class", "neutral")
    if delta is not None:
        delta_html = f'<div class="metric-delta {delta_class}">{html.escape(str(delta))}</div>'
    elif uniform:
        delta_html = '<div class="metric-delta neutral metric-delta-empty">&nbsp;</div>'
    else:
        delta_html = ""
    col.markdown(
        f"""
        <div class="{card_class}">
            <div class="metric-label">{html.escape(metric["label"])}</div>
            <div class="metric-value">{html.escape(str(metric["value"]))}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_audit_event_detail(row: dict[str, Any]) -> None:
    """Full audit event card — no truncation."""
    from adapters import build_full_audit_event_text, format_audit_event

    timestamp = str(row.get("timestamp") or row.get("created_at") or "—")
    provider_id = str(row.get("provider_id") or "—")
    step = str(row.get("step") or row.get("event_type") or "—")
    rule_name = str(row.get("rule_name") or "—")
    decision = str(row.get("decision") or row.get("action") or "—")
    event_text = build_full_audit_event_text(row) or format_audit_event(row)

    st.markdown(
        f"""
        <div class="panel audit-detail-card">
            <div class="panel-title">Selected Audit Event</div>
            <div class="audit-detail-grid">
                <div><span class="audit-detail-label">Timestamp</span><div class="audit-detail-value">{html.escape(timestamp)}</div></div>
                <div><span class="audit-detail-label">Provider ID</span><div class="audit-detail-value">{html.escape(provider_id)}</div></div>
                <div><span class="audit-detail-label">Step</span><div class="audit-detail-value">{html.escape(step)}</div></div>
                <div><span class="audit-detail-label">Rule</span><div class="audit-detail-value">{html.escape(rule_name)}</div></div>
                <div><span class="audit-detail-label">Decision</span><div class="audit-detail-value">{html.escape(decision)}</div></div>
            </div>
            <div class="audit-detail-label" style="margin-top:0.75rem;">Event text</div>
            <div class="audit-detail-body">{html.escape(event_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_llm_enrichment_block(enrichment: dict[str, Any]) -> None:
    """Show actual LLM enrichment content in Review Queue."""
    from adapters import llm_enrichment_sections

    if not enrichment.get("enriched"):
        return
    st.markdown("##### LLM enrichment (reviewer-only)")
    for title, text in llm_enrichment_sections(enrichment):
        st.markdown(f"**{html.escape(title)}**")
        st.info(text)
    st.caption(
        "LLM did not change the final decision; deterministic policy remains authoritative."
    )


def render_summary_panel(title: str, body_fn: Callable[[], None]) -> None:
    """Bordered panel wrapper for overview summary sections."""
    st.markdown(f'<div class="summary-panel-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        body_fn()


def decision_badge(decision: str) -> str:
    css_class = f"badge badge-{decision.replace(' ', '_')}"
    label = decision.replace("_", " ").title()
    return f'<span class="{css_class}">{html.escape(label)}</span>'


def field_comparison_card(change: dict[str, Any]) -> None:
    """Side-by-side current vs proposed field comparison."""
    field = change.get("field", "unknown")
    confidence = change.get("confidence_score", 0)
    old_val = change.get("old_value", "—")
    new_val = change.get("new_value", "—")
    norm_old = change.get("normalized_old_value", "")
    norm_new = change.get("normalized_new_value", "")
    changed = old_val != new_val
    changed_class = " field-changed" if changed else ""

    norm_old_html = (
        f'<div class="field-normalized">normalized: {html.escape(str(norm_old))}</div>'
        if norm_old
        else ""
    )
    norm_new_html = (
        f'<div class="field-normalized">normalized: {html.escape(str(norm_new))}</div>'
        if norm_new
        else ""
    )

    st.markdown(
        f"""
        <div class="field-comparison{changed_class}">
            <div class="field-comparison-header">
                <span>{html.escape(field.replace("_", " ").title())}</span>
                <span class="field-confidence">{confidence:.0%} confidence</span>
            </div>
            <div class="field-comparison-body">
                <div class="field-side current">
                    <div class="field-side-label">Current</div>
                    <div class="field-value">{html.escape(str(old_val))}</div>
                    {norm_old_html}
                </div>
                <div class="field-side proposed">
                    <div class="field-side-label">Proposed</div>
                    <div class="field-value">{html.escape(str(new_val))}</div>
                    {norm_new_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_breakdown(
    title: str,
    components: dict[str, float],
    bar_class: str = "risk",
) -> None:
    """Horizontal bar breakdown for risk, confidence, or conflict scores."""
    st.markdown(f'<div class="panel"><div class="panel-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    for name, value in components.items():
        pct = min(max(float(value) * 100, 0), 100)
        label = name.replace("_", " ").title()
        st.markdown(
            f"""
            <div class="score-bar-row">
                <div class="score-bar-label">
                    <span>{html.escape(label)}</span>
                    <span>{value:.2f}</span>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill {bar_class}" style="width: {pct:.1f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def source_evidence(sources: list[Any]) -> None:
    """Display matched external sources with reliability weights."""
    from adapters import lookup_source_reliability, normalize_sources

    normalized = normalize_sources(sources)
    if not normalized and isinstance(sources, list):
        for item in sources:
            if isinstance(item, dict) and item.get("source"):
                normalized.append(item)
    st.markdown('<div class="panel"><div class="panel-title">Source Evidence</div>', unsafe_allow_html=True)
    if not normalized:
        st.markdown(
            '<p style="color:#64748B;font-size:0.88rem;">No external sources matched.</p>',
            unsafe_allow_html=True,
        )
    for src in normalized:
        name = src.get("source", "Unknown")
        field = src.get("field") or ""
        rel = src.get("reliability")
        if rel is None:
            rel = lookup_source_reliability(name, field or None)
        evidence = src.get("evidence", "")
        if rel is not None:
            field_note = f" ({field} reliability)" if field else ""
            weight_label = f"{rel:.0%} reliable{field_note}"
        else:
            weight_label = "field not specified" if field else "reliability unavailable"
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-card-header">
                    <span class="source-name">{html.escape(str(name))}</span>
                    <span class="source-weight">{html.escape(weight_label)}</span>
                </div>
                <div class="source-detail">{html.escape(str(evidence)) if evidence else "—"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def audit_trail(events: list[dict[str, Any]], limit: int = 12, *, compact: bool = False) -> None:
    """Timeline of audit events for a provider."""
    from adapters import format_audit_event, format_provider_audit_event

    st.markdown('<div class="panel"><div class="panel-title">Audit Trail</div>', unsafe_allow_html=True)
    if not events:
        st.markdown(
            '<p style="color:#64748B;font-size:0.88rem;">No audit events recorded.</p>',
            unsafe_allow_html=True,
        )
    for event in events[:limit]:
        row = dict(event)
        if compact:
            ts, body = format_provider_audit_event(row)
            ts_html = (
                f'<div class="audit-event-time">{html.escape(ts)}</div>' if ts else ""
            )
            st.markdown(
                f"""
                <div class="audit-event">
                    {ts_html}
                    <div class="audit-event-detail">{html.escape(body)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            text = format_audit_event(row)
            st.markdown(
                f"""
                <div class="audit-event">
                    <div class="audit-event-detail">{html.escape(text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def _rec_action(rec: dict[str, Any]) -> str:
    return str(rec.get("recommended_action", rec.get("decision", "unknown")))


def _rec_confidence(rec: dict[str, Any]) -> float:
    return float(rec.get("overall_confidence", rec.get("confidence_score", 0)))


def plot_action_distribution(recommendations: list[dict[str, Any]]) -> go.Figure:
    decisions = [_rec_action(r) for r in recommendations]
    df = pd.DataFrame({"decision": decisions})
    counts = df["decision"].value_counts().reset_index()
    counts.columns = ["decision", "count"]
    labels = {d: d.replace("_", " ").title() for d in counts["decision"]}
    counts["label"] = counts["decision"].map(labels)
    colors = {
        "auto_update": "#14B8A6",
        "human_review": "#F59E0B",
        "do_not_update": "#DC2626",
        "no_change": "#2563EB",
    }
    fig = px.pie(
        counts,
        values="count",
        names="label",
        color="decision",
        color_discrete_map=colors,
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#0F172A"),
        title=dict(text="Decision Distribution", font=dict(size=14), x=0.02, xanchor="left"),
        height=320,
        showlegend=False,
    )
    return fig


def plot_confidence_distribution(recommendations: list[dict[str, Any]]) -> go.Figure:
    scores = [_rec_confidence(r) for r in recommendations]
    decisions = [_rec_action(r).replace("_", " ").title() for r in recommendations]
    df = pd.DataFrame({"confidence": scores, "decision": decisions})
    fig = px.histogram(
        df,
        x="confidence",
        color="decision",
        nbins=20,
        barmode="overlay",
        opacity=0.75,
        color_discrete_sequence=["#2563EB", "#14B8A6", "#F59E0B", "#DC2626"],
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=72, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#0F172A"),
        title=dict(text="Confidence Score Distribution", font=dict(size=14), x=0.02, xanchor="left", y=0.98),
        xaxis_title="Confidence",
        yaxis_title="Providers",
        height=320,
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def plot_risk_reasons(recommendations: list[dict[str, Any]], top_n: int = 8) -> go.Figure:
    from adapters import shorten_reason

    reasons: list[str] = []
    full_reasons: list[str] = []
    for rec in recommendations:
        reason = str(rec.get("reason", "Unspecified"))
        if reason:
            full_reasons.append(reason)
            reasons.append(shorten_reason(reason, 32))
    if not reasons:
        reasons = ["No risk reasons recorded"]
        full_reasons = reasons
    df = pd.DataFrame({"reason": reasons, "full_reason": full_reasons})
    counts = df.groupby(["reason", "full_reason"], as_index=False).size()
    counts.columns = ["reason", "full_reason", "count"]
    counts = counts.sort_values("count").tail(top_n)
    fig = px.bar(
        counts,
        x="count",
        y="reason",
        orientation="h",
        color="count",
        color_continuous_scale=["#DBEAFE", "#2563EB"],
        custom_data=["full_reason"],
        labels={"reason": "Reason (short)", "count": "Count"},
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Count: %{x}<extra></extra>",
    )
    fig.update_layout(
        margin=dict(l=140, r=40, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#0F172A"),
        title=dict(text="Top Risk / Review Reasons", font=dict(size=14)),
        xaxis_title="Count",
        yaxis_title="",
        height=380,
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


def reason_summary_table(recommendations: list[dict[str, Any]], top_n: int = 6) -> pd.DataFrame:
    """Compact full-text reason counts for overview tab."""
    rows: list[dict[str, Any]] = []
    for rec in recommendations:
        reason = str(rec.get("reason", "")).strip()
        if reason:
            rows.append({"reason": reason})
    if not rows:
        return pd.DataFrame(columns=["Reason", "Count"])
    df = pd.DataFrame(rows)
    counts = df.groupby("reason", as_index=False).size()
    counts.columns = ["reason", "count"]
    counts = counts.sort_values("count", ascending=False).head(top_n)
    counts["Reason"] = counts["reason"]
    counts["Count"] = counts["count"]
    return counts[["Reason", "Count"]]


def plot_cost_breakdown(cost_data: dict[str, Any], *, calls_succeeded: int = 0) -> go.Figure:
    breakdown = dict(cost_data.get("breakdown", {}))
    if not breakdown or all(float(v or 0) == 0 for v in breakdown.values()):
        breakdown = {
            "Rules engine": 0.06,
            "LLM enrichment": 0.0,
            "Human review": 7.5,
            "Storage & audit": 0.01,
        }
    llm_key = "LLM enrichment"
    llm_cost = float(breakdown.get(llm_key, 0) or 0)
    if llm_cost > 0:
        breakdown.pop(llm_key, None)
        label = "LLM enrichment (scenario est.)" if calls_succeeded == 0 else "LLM enrichment (actual)"
        breakdown[label] = llm_cost
    df = pd.DataFrame({"component": list(breakdown.keys()), "cost": [float(v) for v in breakdown.values()]})
    chart_title = (
        "Cost Breakdown — scenario estimates (USD)"
        if calls_succeeded == 0 and llm_cost > 0
        else "Cost Breakdown (USD)"
    )
    fig = px.bar(
        df,
        x="component",
        y="cost",
        color="cost",
        color_continuous_scale=["#99F6E4", "#14B8A6", "#0F766E"],
        text="cost",
        custom_data=[df["component"]],
    )
    fig.update_traces(
        texttemplate="$%{text:.2f}",
        textposition="outside",
        hovertemplate="%{customdata[0]}: $%{y:.2f}<extra></extra>",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=100),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#0F172A"),
        title=dict(text=chart_title, font=dict(size=14)),
        yaxis_title="USD",
        xaxis_tickangle=-25,
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


def plot_data_quality_issues(quality: dict[str, Any]) -> go.Figure:
    issues = {
        "Invalid NPI": quality.get("invalid_npi", 0),
        "Missing phone": quality.get("missing_phone", 0),
        "Missing address": quality.get("missing_address", 0),
        "Stale records": quality.get("stale_records", 0),
        "Duplicates": quality.get("duplicates", 0),
    }
    df = pd.DataFrame({"issue": list(issues.keys()), "count": list(issues.values())})
    fig = px.bar(
        df,
        x="issue",
        y="count",
        color="count",
        color_continuous_scale=["#FEF3C7", "#F59E0B", "#D97706"],
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#0F172A"),
        title=dict(text="Data Quality Issues", font=dict(size=14)),
        yaxis_title="Records",
        height=340,
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig
