"""Synthetic benchmark evaluation with optional LLM comparison."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from provider_intelligence.config import ensure_outputs_dir, load_config
from provider_intelligence.schemas import GroundTruthLabel, Recommendation

AUTO_UPDATE_ALLOWED_FIELDS = frozenset({"address", "phone"})


def load_ground_truth(path: Path) -> dict[str, GroundTruthLabel]:
    """Load synthetic ground truth labels keyed by provider_id."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    labels: dict[str, GroundTruthLabel] = {}
    for row in df.to_dict(orient="records"):
        raw_fields = row.get("changed_fields") or "[]"
        if isinstance(raw_fields, list):
            changed_fields = raw_fields
        elif raw_fields.startswith("["):
            try:
                changed_fields = ast.literal_eval(raw_fields)
            except (SyntaxError, ValueError):
                changed_fields = [f for f in raw_fields.split("|") if f]
        else:
            changed_fields = [f for f in raw_fields.split("|") if f]

        label = GroundTruthLabel(
            provider_id=row["provider_id"],
            mutation_type=row.get("mutation_type") or None,
            expected_action=row.get("expected_action") or None,
            changed_fields=changed_fields,
            notes=row.get("notes") or "",
        )
        labels[label.provider_id] = label
    return labels


def load_recommendations(path: Path) -> list[Recommendation]:
    """Load pipeline recommendations from JSON (prefers detailed export)."""
    detailed_path = path.parent / "recommendations_detailed.json"
    source = detailed_path if detailed_path.exists() else path
    if not source.exists():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [Recommendation.model_validate(item) for item in payload]


def _changed_field_names(rec: Recommendation) -> list[str]:
    return [change.field for change in rec.changes]


def _fields_match(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return not actual
    return set(expected) == set(actual)


def _is_false_auto_update(rec: Recommendation, label: GroundTruthLabel | None) -> tuple[bool, str]:
    """Return whether an auto_update is incorrect and why."""
    if label is None:
        return True, "no_ground_truth_label"

    if label.expected_action != "auto_update":
        return True, f"expected_action={label.expected_action}"

    actual_fields = set(_changed_field_names(rec))
    expected_fields = set(label.changed_fields or [])
    if expected_fields and actual_fields != expected_fields:
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields
        parts = []
        if missing:
            parts.append(f"missing_fields={sorted(missing)}")
        if extra:
            parts.append(f"unexpected_fields={sorted(extra)}")
        return True, "; ".join(parts) or "field_mismatch"

    disallowed = actual_fields - AUTO_UPDATE_ALLOWED_FIELDS
    if disallowed:
        return True, f"auto_update_not_allowed_for={sorted(disallowed)}"

    return False, "correct_auto_update"


def compute_false_auto_update_rate(
    recommendations: list[Recommendation],
    ground_truth: dict[str, GroundTruthLabel],
) -> dict[str, Any]:
    """Compute primary safety metric: rate of incorrect auto-updates."""
    auto_updates = [r for r in recommendations if r.recommended_action == "auto_update"]
    if not auto_updates:
        return {
            "false_auto_update_rate": 0.0,
            "auto_update_count": 0,
            "false_auto_update_count": 0,
            "correct_auto_update_count": 0,
        }

    false_count = 0
    correct_count = 0
    for rec in auto_updates:
        label = ground_truth.get(rec.provider_id)
        is_false, _ = _is_false_auto_update(rec, label)
        if is_false:
            false_count += 1
        else:
            correct_count += 1

    total = len(auto_updates)
    return {
        "false_auto_update_rate": round(false_count / total, 4),
        "auto_update_count": total,
        "false_auto_update_count": false_count,
        "correct_auto_update_count": correct_count,
    }


def compute_change_detection_metrics(
    recommendations: list[Recommendation],
    ground_truth: dict[str, GroundTruthLabel],
) -> dict[str, Any]:
    """Precision/recall for detecting material field changes."""
    if not ground_truth:
        return {
            "change_detection_precision": None,
            "change_detection_recall": None,
        }

    predicted_positive = 0
    true_positive = 0
    false_positive = 0
    actual_positive = 0

    for rec in recommendations:
        label = ground_truth.get(rec.provider_id)
        if label is None:
            continue
        expected_change = bool(label.changed_fields)
        predicted_change = rec.change_detected or bool(rec.changes)
        if expected_change:
            actual_positive += 1
        if predicted_change:
            predicted_positive += 1
            if expected_change:
                true_positive += 1
            else:
                false_positive += 1

    precision = round(true_positive / predicted_positive, 4) if predicted_positive else None
    recall = round(true_positive / actual_positive, 4) if actual_positive else None
    return {
        "change_detection_precision": precision,
        "change_detection_recall": recall,
        "change_detection_true_positives": true_positive,
        "change_detection_false_positives": false_positive,
        "change_detection_actual_positives": actual_positive,
    }


def compute_auto_update_precision(
    recommendations: list[Recommendation],
    ground_truth: dict[str, GroundTruthLabel],
) -> float | None:
    """Share of auto_updates that match ground truth auto_update expectations."""
    auto_updates = [r for r in recommendations if r.recommended_action == "auto_update"]
    if not auto_updates:
        return None
    if not ground_truth:
        return None
    correct = sum(
        1
        for rec in auto_updates
        if not _is_false_auto_update(rec, ground_truth.get(rec.provider_id))[0]
    )
    return round(correct / len(auto_updates), 4)


def compute_cost_sensitive_loss(
    recommendations: list[Recommendation],
    ground_truth: dict[str, GroundTruthLabel],
    config: dict[str, Any] | None = None,
) -> float | None:
    """Weighted routing loss using configurable cost weights."""
    if not ground_truth:
        return None
    cfg = config or load_config()
    costs = cfg["costs"]
    c_wrong = float(costs.get("wrong_auto_update", 10.0))
    c_missed = float(costs.get("missed_update", 3.0))
    c_review = float(costs.get("human_review_per_record", 0.5))

    wrong_auto = 0
    missed_auto = 0
    human_review_count = 0
    matched = 0

    for rec in recommendations:
        label = ground_truth.get(rec.provider_id)
        if label is None or not label.expected_action:
            continue
        matched += 1
        if rec.recommended_action == "auto_update":
            is_false, _ = _is_false_auto_update(rec, label)
            if is_false:
                wrong_auto += 1
        elif label.expected_action == "auto_update":
            missed_auto += 1
        if rec.recommended_action == "human_review":
            human_review_count += 1

    if matched == 0:
        return None

    total_loss = (c_wrong * wrong_auto) + (c_missed * missed_auto) + (c_review * human_review_count)
    return round(total_loss / matched, 4)


def build_auto_update_evaluation_debug(
    recommendations: list[Recommendation],
    ground_truth: dict[str, GroundTruthLabel],
) -> list[dict[str, Any]]:
    """Build per-auto-update debug rows for judge review."""
    rows: list[dict[str, Any]] = []
    for rec in recommendations:
        if rec.recommended_action != "auto_update":
            continue
        label = ground_truth.get(rec.provider_id)
        is_false, false_reason = _is_false_auto_update(rec, label)
        change_summaries = []
        for change in rec.changes:
            change_summaries.append(
                f"{change.field}: {change.old_value!r} -> {change.new_value!r}"
            )
        rows.append(
            {
                "provider_id": rec.provider_id,
                "npi": rec.npi,
                "changed_fields": "|".join(_changed_field_names(rec)),
                "proposed_changes": " ; ".join(change_summaries),
                "overall_confidence": rec.overall_confidence,
                "conflict_score": rec.conflict_score,
                "mutation_type": label.mutation_type if label else "",
                "expected_action": label.expected_action if label else "",
                "expected_changed_fields": "|".join(label.changed_fields) if label else "",
                "is_false_auto_update": is_false,
                "false_reason": false_reason if is_false else "",
                "recommendation_reason": rec.reason,
            }
        )
    return rows


def write_auto_update_evaluation_debug(
    recommendations: list[Recommendation],
    ground_truth: dict[str, GroundTruthLabel],
    path: Path,
) -> None:
    """Write auto_update evaluation debug CSV."""
    rows = build_auto_update_evaluation_debug(recommendations, ground_truth)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "provider_id",
        "npi",
        "changed_fields",
        "proposed_changes",
        "overall_confidence",
        "conflict_score",
        "mutation_type",
        "expected_action",
        "expected_changed_fields",
        "is_false_auto_update",
        "false_reason",
        "recommendation_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_action_accuracy(
    recommendations: list[Recommendation],
    ground_truth: dict[str, GroundTruthLabel],
) -> dict[str, Any]:
    """Compute overall action routing accuracy against ground truth."""
    if not ground_truth:
        return {"action_accuracy": None, "matched_records": 0, "correct_actions": 0}

    correct = 0
    matched = 0
    for rec in recommendations:
        label = ground_truth.get(rec.provider_id)
        if label is None or not label.expected_action:
            continue
        matched += 1
        if rec.recommended_action == label.expected_action:
            correct += 1

    accuracy = round(correct / matched, 4) if matched else None
    return {
        "action_accuracy": accuracy,
        "matched_records": matched,
        "correct_actions": correct,
    }


def _heuristic_clarity_score(recommendations: list[Recommendation]) -> float:
    """Heuristic human-review clarity based on non-empty reasons and changes."""
    review_cases = [r for r in recommendations if r.recommended_action == "human_review"]
    if not review_cases:
        return 1.0
    clear = sum(
        1
        for rec in review_cases
        if rec.reason.strip() and (rec.changes or rec.conflict_score >= 0.35)
    )
    return round(clear / len(review_cases), 4)


def _specialty_mapping_coverage(recommendations: list[Recommendation]) -> float:
    """Share of specialty changes with normalized values present."""
    specialty_changes = [
        change
        for rec in recommendations
        for change in rec.changes
        if change.field == "specialty"
    ]
    if not specialty_changes:
        return 1.0
    covered = sum(1 for c in specialty_changes if c.normalized_new_value)
    return round(covered / len(specialty_changes), 4)


def _conflict_explanation_completeness(recommendations: list[Recommendation]) -> float:
    """Heuristic completeness of conflict explanations for review cases."""
    conflict_cases = [r for r in recommendations if r.conflict_score >= 0.35]
    if not conflict_cases:
        return 1.0
    complete = sum(1 for rec in conflict_cases if rec.reason.strip() and rec.changes)
    return round(complete / len(conflict_cases), 4)


def _website_extraction_recall(
    ground_truth: dict[str, GroundTruthLabel],
    website_results: dict[str, dict[str, Any]] | None = None,
) -> float | None:
    """Recall of website field extraction against synthetic labels (if available)."""
    if not website_results:
        return None
    website_mutations = {
        pid for pid, label in ground_truth.items() if "website" in label.changed_fields
    }
    if not website_mutations:
        return None
    extracted = sum(
        1
        for pid in website_mutations
        if website_results.get(pid, {}).get("extracted_fields")
    )
    return round(extracted / len(website_mutations), 4)


def _count_llm_successes(llm_audit_path: Path | None) -> int:
    if not llm_audit_path or not llm_audit_path.exists():
        return 0
    from provider_intelligence.llm_diagnostics import read_llm_audit_csv

    df = read_llm_audit_csv(llm_audit_path)
    if df.empty:
        return 0
    if "success" in df.columns:
        return int(df["success"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    if "called" in df.columns:
        return int(df["called"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    return len(df)


def _mode_metrics(
    recommendations: list[Recommendation],
    ground_truth: dict[str, GroundTruthLabel],
    llm_audit_path: Path | None = None,
    website_results: dict[str, dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build evaluation metrics for a single pipeline mode."""
    cfg = config or load_config()
    false_auto = compute_false_auto_update_rate(recommendations, ground_truth)
    action_acc = compute_action_accuracy(recommendations, ground_truth)
    change_metrics = compute_change_detection_metrics(recommendations, ground_truth)
    auto_precision = compute_auto_update_precision(recommendations, ground_truth)
    cost_loss = compute_cost_sensitive_loss(recommendations, ground_truth, cfg)
    review_count = sum(1 for r in recommendations if r.recommended_action == "human_review")
    human_review_rate = round(review_count / len(recommendations), 4) if recommendations else None

    benchmark_available = bool(ground_truth)
    metrics: dict[str, Any] = {
        "records_evaluated": len(recommendations),
        "benchmark_available": benchmark_available,
        "false_auto_update_rate": false_auto["false_auto_update_rate"],
        "auto_update_count": false_auto["auto_update_count"],
        "false_auto_update_count": false_auto["false_auto_update_count"],
        "correct_auto_update_count": false_auto["correct_auto_update_count"],
        "action_accuracy": action_acc["action_accuracy"],
        "human_review_rate": human_review_rate,
        "conflict_explanation_completeness": _conflict_explanation_completeness(recommendations),
        "specialty_mapping_coverage": _specialty_mapping_coverage(recommendations),
        "website_extraction_recall": _website_extraction_recall(ground_truth, website_results),
        "human_review_clarity_score": _heuristic_clarity_score(recommendations),
        "llm_calls": _count_llm_successes(llm_audit_path),
    }

    if benchmark_available:
        metrics.update(change_metrics)
        metrics["auto_update_precision"] = auto_precision
        metrics["cost_sensitive_loss"] = cost_loss
    else:
        metrics.update(
            {
                "change_detection_precision": None,
                "change_detection_recall": None,
                "auto_update_precision": None,
                "cost_sensitive_loss": None,
            }
        )

    return metrics


def build_llm_comparison(
    rule_based_metrics: dict[str, Any],
    adaptive_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare rule-only vs adaptive LLM enrichment metrics."""
    rule_false = rule_based_metrics.get("false_auto_update_rate", 0.0)
    adaptive_false = adaptive_metrics.get("false_auto_update_rate", 0.0)
    safety_preserved = adaptive_false <= rule_false

    improvements: dict[str, Any] = {}
    for key in (
        "conflict_explanation_completeness",
        "specialty_mapping_coverage",
        "website_extraction_recall",
        "human_review_clarity_score",
    ):
        rule_val = rule_based_metrics.get(key)
        adaptive_val = adaptive_metrics.get(key)
        if rule_val is None or adaptive_val is None:
            improvements[key] = None
        else:
            improvements[key] = round(adaptive_val - rule_val, 4)

    return {
        "rule_based_only": rule_based_metrics,
        "adaptive_llm": adaptive_metrics,
        "safety_constraint_met": safety_preserved,
        "false_auto_update_rate_delta": round(adaptive_false - rule_false, 4),
        "improvements": improvements,
        "summary": (
            "Adaptive LLM preserved safety"
            if safety_preserved
            else "Adaptive LLM increased false auto-update rate — not recommended"
        ),
    }


def run_evaluation(
    output_dir: Path | None = None,
    compare_llm: bool = False,
    rule_based_dir: Path | None = None,
    adaptive_dir: Path | None = None,
) -> dict[str, Any]:
    """Run synthetic benchmark evaluation and write evaluation_metrics.json."""
    config = load_config()
    outputs_dir = output_dir or ensure_outputs_dir()
    ground_truth_path = outputs_dir / "synthetic_ground_truth.csv"
    recommendations_path = outputs_dir / "recommendations.json"
    llm_audit_path = outputs_dir / "audit_llm_calls.csv"
    debug_path = outputs_dir / "auto_update_evaluation_debug.csv"

    ground_truth = load_ground_truth(ground_truth_path)
    recommendations = load_recommendations(recommendations_path)
    primary_metrics = _mode_metrics(recommendations, ground_truth, llm_audit_path, config=config)

    write_auto_update_evaluation_debug(recommendations, ground_truth, debug_path)

    result: dict[str, Any] = {
        "benchmark": "synthetic_ground_truth" if ground_truth else "unavailable",
        "benchmark_available": bool(ground_truth),
        "primary_safety_metric": "false_auto_update_rate",
        "false_auto_update_rate": primary_metrics["false_auto_update_rate"],
        "metrics": primary_metrics,
        "ground_truth_records": len(ground_truth),
        "recommendations_path": str(recommendations_path),
        "ground_truth_path": str(ground_truth_path),
        "auto_update_debug_path": str(debug_path),
    }

    if compare_llm:
        rule_dir = rule_based_dir or outputs_dir
        adaptive_output_dir = adaptive_dir or outputs_dir
        rule_recs = load_recommendations(rule_dir / "recommendations.json")
        adaptive_recs = load_recommendations(adaptive_output_dir / "recommendations.json")
        rule_metrics = _mode_metrics(
            rule_recs,
            ground_truth,
            rule_dir / "audit_llm_calls.csv",
            config=config,
        )
        adaptive_metrics = _mode_metrics(
            adaptive_recs,
            ground_truth,
            adaptive_output_dir / "audit_llm_calls.csv",
            config=config,
        )
        result["llm_comparison"] = build_llm_comparison(rule_metrics, adaptive_metrics)
    elif llm_audit_path.exists():
        result["llm_enrichment"] = {
            "llm_calls": primary_metrics["llm_calls"],
            "mode": config["llm"].get("mode", "auto"),
        }

    out_path = outputs_dir / "evaluation_metrics.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_path"] = str(out_path)
    return result
