#!/usr/bin/env python3
"""Build a poster-facing generalization matrix from the saved training report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    DEFAULT_TRAINING_PATH_NAME,
    MODEL_ONLY_NAMESPACE,
    POSTER_DEFAULT_TRAINING_PATH_NAME,
    TRAINING_PATH_LEGACY_FPRIME_BASELINE,
    aggregate_evaluation_runs,
    aggregate_mean,
    aggregate_min_per_class_recall,
    aggregate_stat,
    run_training,
)
from runtime import CLASS_NAMES, save_json

GENERALIZATION_MATRIX_SCHEMA_VERSION = "generalization_matrix.v1"

EVALUATION_DISPLAY_NAMES = {
    "repeated_grouped_cv": "grouped_run_splits",
    "scenario_family_holdout": "scenario_window_holdout",
    "command_family_holdout": "command_family_holdout",
    "protocol_family_holdout": "protocol_family_holdout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for reports and model outputs")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for grouped evaluation")
    parser.add_argument(
        "--training-path",
        choices=[POSTER_DEFAULT_TRAINING_PATH_NAME, TRAINING_PATH_LEGACY_FPRIME_BASELINE],
        default=DEFAULT_TRAINING_PATH_NAME,
        help="Training/evaluation path to run before building the matrix",
    )
    parser.add_argument(
        "--blue-feature-policy-name",
        default=None,
        help="Optional explicit blue feature policy name to pass through to training",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Forward plot generation to the training report path before building the matrix",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _aggregate_path(payload: dict[str, Any], *path: Any) -> float:
    return float(aggregate_mean(payload, *path))


def _aggregate_path_std(payload: dict[str, Any], *path: Any) -> float:
    return float(aggregate_stat(payload, "std", *path))


def _scalar_path(payload: dict[str, Any], *path: Any) -> float:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return 0.0


def _min_recall_from_model_metrics(model_metrics: dict[str, Any]) -> float:
    per_class = dict(model_metrics.get(MODEL_ONLY_NAMESPACE, {}).get("multiclass_metrics", {}).get("per_class", {}))
    recalls = [_scalar_path(per_class, class_name, "recall") for class_name in CLASS_NAMES]
    return min(recalls) if recalls else 0.0


def build_metric_summary(model_metrics: dict[str, Any], *, aggregated: bool) -> dict[str, Any]:
    if aggregated:
        summary = {
            "multiclass_accuracy": _aggregate_path(model_metrics, MODEL_ONLY_NAMESPACE, "multiclass_metrics", "accuracy"),
            "multiclass_accuracy_std": _aggregate_path_std(
                model_metrics,
                MODEL_ONLY_NAMESPACE,
                "multiclass_metrics",
                "accuracy",
            ),
            "multiclass_macro_f1": _aggregate_path(model_metrics, MODEL_ONLY_NAMESPACE, "multiclass_metrics", "macro_f1"),
            "multiclass_macro_f1_std": _aggregate_path_std(
                model_metrics,
                MODEL_ONLY_NAMESPACE,
                "multiclass_metrics",
                "macro_f1",
            ),
            "min_per_class_recall": float(aggregate_min_per_class_recall(model_metrics)),
            "cyber_f1": _aggregate_path(model_metrics, MODEL_ONLY_NAMESPACE, "cyber_binary_metrics", "f1"),
            "cyber_f1_std": _aggregate_path_std(model_metrics, MODEL_ONLY_NAMESPACE, "cyber_binary_metrics", "f1"),
            "anomaly_f1": _aggregate_path(model_metrics, MODEL_ONLY_NAMESPACE, "anomaly_binary_metrics", "f1"),
            "anomaly_f1_std": _aggregate_path_std(model_metrics, MODEL_ONLY_NAMESPACE, "anomaly_binary_metrics", "f1"),
        }
        if "stacked_detector" in model_metrics:
            summary["stacked_anomaly_f1"] = _aggregate_path(
                model_metrics,
                "stacked_detector",
                "anomaly_binary_metrics",
                "f1",
            )
            summary["stacked_anomaly_f1_std"] = _aggregate_path_std(
                model_metrics,
                "stacked_detector",
                "anomaly_binary_metrics",
                "f1",
            )
        return summary

    summary = {
        "multiclass_accuracy": _scalar_path(model_metrics, MODEL_ONLY_NAMESPACE, "multiclass_metrics", "accuracy"),
        "multiclass_macro_f1": _scalar_path(model_metrics, MODEL_ONLY_NAMESPACE, "multiclass_metrics", "macro_f1"),
        "min_per_class_recall": _min_recall_from_model_metrics(model_metrics),
        "cyber_f1": _scalar_path(model_metrics, MODEL_ONLY_NAMESPACE, "cyber_binary_metrics", "f1"),
        "anomaly_f1": _scalar_path(model_metrics, MODEL_ONLY_NAMESPACE, "anomaly_binary_metrics", "f1"),
    }
    if "stacked_detector" in model_metrics:
        summary["stacked_anomaly_f1"] = _scalar_path(
            model_metrics,
            "stacked_detector",
            "anomaly_binary_metrics",
            "f1",
        )
    return summary


def build_baseline_summary(baseline: dict[str, Any]) -> dict[str, Any]:
    best_metric_value = baseline.get("best_metric_value")
    near_perfect = baseline.get("near_perfect")
    not_applicable = baseline.get("not_applicable")
    near_perfect_threshold = baseline.get("near_perfect_threshold")
    if isinstance(best_metric_value, dict):
        best_metric_value = best_metric_value.get("mean", 0.0)
    if isinstance(near_perfect, dict):
        near_perfect_value = bool(float(near_perfect.get("true_fraction", 0.0)) >= 0.5)
        near_perfect_fraction = float(near_perfect.get("true_fraction", 0.0))
    else:
        near_perfect_value = bool(near_perfect)
        near_perfect_fraction = 1.0 if near_perfect_value else 0.0
    if isinstance(not_applicable, dict):
        not_applicable_value = bool(float(not_applicable.get("true_fraction", 0.0)) >= 0.5)
        not_applicable_fraction = float(not_applicable.get("true_fraction", 0.0))
    else:
        not_applicable_value = bool(not_applicable)
        not_applicable_fraction = 1.0 if not_applicable_value else 0.0
    if isinstance(near_perfect_threshold, dict):
        near_perfect_threshold = near_perfect_threshold.get("mean", 0.0)
    return {
        "feature_names": list(baseline.get("feature_names") or []),
        "best_metric_path": baseline.get("best_metric_path"),
        "best_metric_value": float(best_metric_value or 0.0),
        "near_perfect": near_perfect_value,
        "near_perfect_fraction": near_perfect_fraction,
        "near_perfect_threshold": float(near_perfect_threshold or 0.0),
        "not_applicable": not_applicable_value,
        "not_applicable_fraction": not_applicable_fraction,
        **({"reason": str(baseline.get("reason") or "")} if baseline.get("reason") else {}),
    }


def summarize_gate(report: dict[str, Any]) -> dict[str, Any]:
    gate = dict(report.get("evaluation", {}).get("deployment_gate", {}))
    models = dict(gate.get("models") or {})
    return {
        "passed": bool(gate.get("passed")),
        "eligible_models": list(gate.get("eligible_models") or []),
        "thresholds": dict(gate.get("thresholds") or {}),
        "models": {
            model_name: {
                "eligible_for_deployment": bool(payload.get("eligible_for_deployment")),
                "checks": list(payload.get("checks") or []),
            }
            for model_name, payload in models.items()
        },
    }


def build_aggregate_rows(
    evaluation_name: str,
    evaluation_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = dict(evaluation_report.get("aggregate") or {})
    model_aggregates = dict(aggregate.get("models") or {})
    aggregate_models: dict[str, Any] = {}
    for model_name, model_metrics in sorted(model_aggregates.items()):
        summary = build_metric_summary(dict(model_metrics), aggregated=True)
        aggregate_models[model_name] = summary
        rows.append(
            {
                "evaluation_name": evaluation_name,
                "display_name": EVALUATION_DISPLAY_NAMES.get(evaluation_name, evaluation_name),
                "scope": "aggregate",
                "model_name": model_name,
                "metrics": summary,
                "run_count": int(evaluation_report.get("total_runs") or 0),
            }
        )
    return rows, aggregate_models


def build_heldout_value_rows(
    evaluation_name: str,
    evaluation_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in evaluation_report.get("runs", []):
        grouped_runs[str(run.get("heldout_value") or "unknown")].append(dict(run))
    rows: list[dict[str, Any]] = []
    per_value: dict[str, Any] = {}
    for heldout_value in sorted(grouped_runs):
        aggregate = aggregate_evaluation_runs(grouped_runs[heldout_value])
        model_payload = dict(aggregate.get("models") or {})
        model_summaries: dict[str, Any] = {}
        for model_name, model_metrics in sorted(model_payload.items()):
            summary = build_metric_summary(dict(model_metrics), aggregated=True)
            model_summaries[model_name] = summary
            rows.append(
                {
                    "evaluation_name": evaluation_name,
                    "display_name": EVALUATION_DISPLAY_NAMES.get(evaluation_name, evaluation_name),
                    "scope": "heldout_value",
                    "heldout_value": heldout_value,
                    "family_field": str(evaluation_report.get("family_field") or ""),
                    "model_name": model_name,
                    "metrics": summary,
                    "run_count": len(grouped_runs[heldout_value]),
                }
            )
        per_value[heldout_value] = {
            "run_count": len(grouped_runs[heldout_value]),
            "models": model_summaries,
        }
    return rows, per_value


def build_experiment_summary(evaluation_name: str, evaluation_report: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate_rows, aggregate_models = build_aggregate_rows(evaluation_name, evaluation_report)
    heldout_rows: list[dict[str, Any]] = []
    heldout_values: dict[str, Any] = {}
    if evaluation_name != "repeated_grouped_cv":
        heldout_rows, heldout_values = build_heldout_value_rows(evaluation_name, evaluation_report)
    payload = {
        "display_name": EVALUATION_DISPLAY_NAMES.get(evaluation_name, evaluation_name),
        "feasible": True if evaluation_name == "repeated_grouped_cv" else bool(evaluation_report.get("feasible")),
        "group_key": str(evaluation_report.get("group_key") or ""),
        "family_field": str(evaluation_report.get("family_field") or ""),
        "seeds": list(evaluation_report.get("seeds") or []),
        "total_runs": int(evaluation_report.get("total_runs") or 0),
        "evaluated_values": list(evaluation_report.get("evaluated_values") or []),
        "skipped": list(evaluation_report.get("skipped") or []),
        "aggregate_models": aggregate_models,
        "heldout_values": heldout_values,
        "shortcut_baselines": {
            "protocol_only": build_baseline_summary(
                dict(evaluation_report.get("aggregate", {}).get("protocol_only_baseline") or {})
            ),
            "raw_protocol_shortcuts": build_baseline_summary(
                dict(evaluation_report.get("aggregate", {}).get("raw_protocol_shortcuts_baseline") or {})
            ),
        },
    }
    return payload, [*aggregate_rows, *heldout_rows]


def build_evaluation_matrix(
    training_report: dict[str, Any],
    *,
    metrics_report_path: Path,
    training_invocation_blocked: bool = False,
    training_invocation_message: str | None = None,
) -> dict[str, Any]:
    evaluation = dict(training_report.get("evaluation") or {})
    grouped_report = dict(evaluation.get("repeated_grouped_cv") or {})
    scenario_report = dict(evaluation.get("scenario_family_holdout") or {})
    command_report = dict(evaluation.get("command_family_holdout") or {})
    protocol_report = dict(evaluation.get("protocol_family_holdout") or {})
    matrix_rows: list[dict[str, Any]] = []
    experiments: dict[str, Any] = {}
    for evaluation_name, evaluation_report in (
        ("repeated_grouped_cv", grouped_report),
        ("scenario_family_holdout", scenario_report),
        ("command_family_holdout", command_report),
        ("protocol_family_holdout", protocol_report),
    ):
        experiment_summary, experiment_rows = build_experiment_summary(evaluation_name, evaluation_report)
        experiments[EVALUATION_DISPLAY_NAMES.get(evaluation_name, evaluation_name)] = experiment_summary
        matrix_rows.extend(experiment_rows)
    return {
        "record_kind": "generalization_matrix",
        "schema_version": GENERALIZATION_MATRIX_SCHEMA_VERSION,
        "dataset": str(training_report.get("dataset") or ""),
        "rows": int(training_report.get("rows") or 0),
        "class_counts": dict(training_report.get("class_counts") or {}),
        "training_path": dict(training_report.get("training_path") or {}),
        "comparison_only": bool(training_report.get("comparison_only")),
        "report_source": {
            "metrics_report_path": str(metrics_report_path.resolve()),
            "summary_report_path": str((metrics_report_path.parent / "summary.txt").resolve()),
        },
        "training_invocation": {
            "blocked": bool(training_invocation_blocked),
            **({"message": str(training_invocation_message)} if training_invocation_message else {}),
        },
        "deployment": {
            "ready": bool(training_report.get("deployment_ready")),
            "blocked_reason": training_report.get("deployment_blocked_reason"),
            "winner": dict(training_report.get("deployment_winner") or {}),
            "gate": summarize_gate(training_report),
        },
        "shortcut_baselines": {
            "protocol_only": build_baseline_summary(dict(training_report.get("protocol_only_baseline") or {})),
            "raw_protocol_shortcuts": build_baseline_summary(
                dict(training_report.get("raw_protocol_shortcuts_baseline") or {})
            ),
        },
        "dataset_sanity": {
            "passed": bool(training_report.get("dataset_sanity", {}).get("passed")),
            "blocking_issues": list(training_report.get("dataset_sanity", {}).get("blocking_issues") or []),
        },
        "ranking_metric_order": list(training_report.get("ranking_metric_order") or []),
        "experiments": experiments,
        "cross_protocol_generalization": {
            "evaluation_name": "protocol_family_holdout",
            "display_name": EVALUATION_DISPLAY_NAMES["protocol_family_holdout"],
            "feasible": bool(protocol_report.get("feasible")),
            "evaluated_values": list(protocol_report.get("evaluated_values") or []),
            "aggregate_models": dict(
                experiments.get(EVALUATION_DISPLAY_NAMES["protocol_family_holdout"], {}).get("aggregate_models") or {}
            ),
            "heldout_values": dict(
                experiments.get(EVALUATION_DISPLAY_NAMES["protocol_family_holdout"], {}).get("heldout_values") or {}
            ),
        },
        "matrix_rows": matrix_rows,
    }


def render_evaluation_matrix_summary(matrix_report: dict[str, Any]) -> str:
    lines = [
        "Generalization evaluation matrix",
        f"dataset: {matrix_report['dataset']}",
        f"training path: {matrix_report['training_path'].get('name', '')}",
        f"comparison only: {'yes' if matrix_report.get('comparison_only') else 'no'}",
        f"deployment ready: {'yes' if matrix_report['deployment'].get('ready') else 'no'}",
    ]
    blocked_reason = matrix_report["deployment"].get("blocked_reason")
    if blocked_reason:
        lines.append(f"deployment blocked reason: {blocked_reason}")
    if matrix_report.get("training_invocation", {}).get("blocked"):
        lines.append(f"training invocation: recovered saved report after block ({matrix_report['training_invocation'].get('message', '')})")

    shortcut_baselines = dict(matrix_report.get("shortcut_baselines") or {})
    protocol_only = dict(shortcut_baselines.get("protocol_only") or {})
    raw_shortcuts = dict(shortcut_baselines.get("raw_protocol_shortcuts") or {})
    lines.extend(
        [
            "",
            "Shortcut baseline checks",
            (
                "protocol-only baseline: "
                f"near_perfect={str(bool(protocol_only.get('near_perfect'))).lower()} "
                f"best={float(protocol_only.get('best_metric_value') or 0.0):.4f} "
                f"path={protocol_only.get('best_metric_path') or 'n/a'}"
            ),
            (
                "raw protocol shortcuts baseline: "
                f"near_perfect={str(bool(raw_shortcuts.get('near_perfect'))).lower()} "
                f"best={float(raw_shortcuts.get('best_metric_value') or 0.0):.4f} "
                f"path={raw_shortcuts.get('best_metric_path') or 'n/a'}"
            ),
        ]
    )

    experiments = dict(matrix_report.get("experiments") or {})
    for experiment_key in (
        "grouped_run_splits",
        "scenario_window_holdout",
        "command_family_holdout",
        "protocol_family_holdout",
    ):
        experiment = dict(experiments.get(experiment_key) or {})
        if not experiment:
            continue
        lines.extend(
            [
                "",
                f"{experiment.get('display_name', experiment_key)}",
                f"feasible: {str(bool(experiment.get('feasible'))).lower()}",
                f"total runs: {int(experiment.get('total_runs') or 0)}",
            ]
        )
        if experiment.get("evaluated_values"):
            lines.append(
                "evaluated values: " + ", ".join(str(value) for value in experiment.get("evaluated_values") or [])
            )
        for model_name, metrics in sorted(dict(experiment.get("aggregate_models") or {}).items()):
            lines.append(
                f"{model_name}: macro_f1={float(metrics.get('multiclass_macro_f1') or 0.0):.4f} "
                f"min_recall={float(metrics.get('min_per_class_recall') or 0.0):.4f} "
                f"cyber_f1={float(metrics.get('cyber_f1') or 0.0):.4f} "
                f"anomaly_f1={float(metrics.get('anomaly_f1') or 0.0):.4f}"
            )
        for heldout_value, payload in sorted(dict(experiment.get("heldout_values") or {}).items()):
            for model_name, metrics in sorted(dict(payload.get("models") or {}).items()):
                lines.append(
                    f"held out {heldout_value} {model_name}: "
                    f"macro_f1={float(metrics.get('multiclass_macro_f1') or 0.0):.4f} "
                    f"min_recall={float(metrics.get('min_per_class_recall') or 0.0):.4f} "
                    f"cyber_f1={float(metrics.get('cyber_f1') or 0.0):.4f} "
                    f"anomaly_f1={float(metrics.get('anomaly_f1') or 0.0):.4f}"
                )

    deployment_gate = dict(matrix_report.get("deployment", {}).get("gate") or {})
    lines.extend(
        [
            "",
            "Deployment gate",
            f"passed: {str(bool(deployment_gate.get('passed'))).lower()}",
            "eligible models: " + ", ".join(str(name) for name in deployment_gate.get("eligible_models") or []) if deployment_gate.get("eligible_models") else "eligible models: none",
        ]
    )
    return "\n".join(lines) + "\n"


def write_evaluation_matrix_report(output_dir: Path, matrix_report: dict[str, Any]) -> dict[str, Path]:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = report_dir / "evaluation_matrix.json"
    summary_path = report_dir / "evaluation_matrix_summary.txt"
    save_json(matrix_path, matrix_report)
    summary_path.write_text(render_evaluation_matrix_summary(matrix_report), encoding="utf-8")
    return {
        "matrix_path": matrix_path,
        "summary_path": summary_path,
    }


def run_training_capture_report(
    dataset_path: Path,
    output_dir: Path,
    *,
    seed: int,
    make_plots: bool,
    training_path_name: str,
    blue_feature_policy_name: str | None,
) -> tuple[dict[str, Any], bool, str | None, Path]:
    metrics_path = output_dir / "reports" / "metrics.json"
    blocked = False
    blocked_message: str | None = None
    try:
        report = run_training(
            dataset_path,
            output_dir,
            seed=seed,
            make_plots=make_plots,
            blue_feature_policy_name=blue_feature_policy_name,
            training_path_name=training_path_name,
        )
        return report, blocked, blocked_message, metrics_path
    except SystemExit as exc:
        blocked = True
        blocked_message = str(exc)
        if metrics_path.exists():
            return read_json(metrics_path), blocked, blocked_message, metrics_path
        raise


def evaluate_generalization(
    dataset_path: Path,
    output_dir: Path,
    *,
    seed: int = 7,
    make_plots: bool = False,
    training_path_name: str = DEFAULT_TRAINING_PATH_NAME,
    blue_feature_policy_name: str | None = None,
) -> dict[str, Any]:
    training_report, blocked, blocked_message, metrics_path = run_training_capture_report(
        dataset_path,
        output_dir,
        seed=seed,
        make_plots=make_plots,
        training_path_name=training_path_name,
        blue_feature_policy_name=blue_feature_policy_name,
    )
    matrix_report = build_evaluation_matrix(
        training_report,
        metrics_report_path=metrics_path,
        training_invocation_blocked=blocked,
        training_invocation_message=blocked_message,
    )
    write_evaluation_matrix_report(output_dir, matrix_report)
    return matrix_report


def main() -> None:
    args = parse_args()
    evaluate_generalization(
        args.dataset,
        args.output_dir,
        seed=int(args.seed),
        make_plots=bool(args.make_plots),
        training_path_name=str(args.training_path),
        blue_feature_policy_name=args.blue_feature_policy_name,
    )


if __name__ == "__main__":
    main()
