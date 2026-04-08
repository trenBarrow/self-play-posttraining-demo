#!/usr/bin/env python3
"""Generate initial-training and autoresearch progress plots for F´ and MAVLink."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.checkpointing import load_self_play_state

PROTOCOL_PROGRESS_MANIFEST_SCHEMA_VERSION = "protocol_progress_manifest.v1"

PALETTE = {
    "ink": "#1f2430",
    "paper": "#f5f2ea",
    "fprime": "#1b4965",
    "mavlink": "#8c3c1f",
    "train": "#2a9d8f",
    "validation": "#264653",
    "macro": "#457b9d",
    "cyber": "#e76f51",
    "anomaly": "#6a994e",
    "reward": "#c58b2b",
    "alignment": "#7d4e8c",
    "grid": "#d8c8a8",
    "muted": "#7d8799",
}


class ProtocolProgressPlotError(ValueError):
    """Raised when protocol progress plots cannot be generated safely."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolProgressPlotError(f"{path} must contain a JSON object")
    return payload


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _slug(value: str) -> str:
    chars: list[str] = []
    last_sep = False
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
            last_sep = False
            continue
        if not last_sep:
            chars.append("_")
            last_sep = True
    return "".join(chars).strip("_") or "figure"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "font.size": 11,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
        }
    )


def _save_figure_variants(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "svg_path": str(svg_path.resolve()),
        "png_path": str(png_path.resolve()),
    }


def _resolve_metrics_report(path_or_dir: Path) -> Path:
    resolved = path_or_dir.resolve()
    if resolved.is_file():
        return resolved
    candidate = resolved / "reports" / "metrics.json"
    if candidate.exists():
        return candidate
    raise ProtocolProgressPlotError(f"Missing metrics report under {resolved}")


def _resolve_self_play_dir(path_or_dir: Path) -> Path:
    resolved = path_or_dir.resolve()
    if resolved.is_file():
        resolved = resolved.parent
    state_path = resolved / "self_play_state.json"
    if not state_path.exists():
        raise ProtocolProgressPlotError(f"Missing self_play_state.json under {resolved}")
    return resolved


def _training_history_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = _as_mapping(report.get("blue_model")).get("training_history")
    if not isinstance(history, list) or not history:
        raise ProtocolProgressPlotError("Training report is missing blue_model.training_history")
    rows: list[dict[str, Any]] = []
    for row in history:
        item = _as_mapping(row)
        epoch = int(_number(item.get("epoch")) or 0)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(_number(item.get("train_loss")) or 0.0),
                "validation_loss": float(_number(item.get("validation_loss")) or 0.0),
                "train_macro_f1": float(_number(item.get("train_macro_f1")) or 0.0),
                "validation_macro_f1": float(_number(item.get("validation_macro_f1")) or 0.0),
                "epoch_rows": int(_number(item.get("epoch_rows")) or 0),
            }
        )
    return rows


def _extract_initial_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    neural = _as_mapping(_as_mapping(_as_mapping(report.get("metrics")).get("model_only")).get("neural_net"))
    multiclass = _as_mapping(neural.get("multiclass_metrics"))
    cyber = _as_mapping(neural.get("cyber_binary_metrics"))
    anomaly = _as_mapping(neural.get("anomaly_binary_metrics"))
    return {
        "macro_f1": float(_number(multiclass.get("macro_f1")) or 0.0),
        "accuracy": float(_number(multiclass.get("accuracy")) or 0.0),
        "cyber_f1": float(_number(cyber.get("f1")) or 0.0),
        "anomaly_f1": float(_number(anomaly.get("f1")) or 0.0),
        "deployment_ready": bool(report.get("deployment_ready")),
        "deployment_blocked_reason": _text(report.get("deployment_blocked_reason")),
    }


def _load_initial_training_artifact(path_or_dir: Path, *, protocol_label: str) -> dict[str, Any]:
    report_path = _resolve_metrics_report(path_or_dir)
    report = _read_json(report_path)
    history = _training_history_rows(report)
    training_summary = _as_mapping(_as_mapping(report.get("blue_model")).get("training_summary"))
    best_epoch_row = min(history, key=lambda row: row["validation_loss"])
    return {
        "protocol_label": protocol_label,
        "report_path": str(report_path.resolve()),
        "history": history,
        "training_summary": {
            "epochs": len(history),
            "best_validation_loss": float(best_epoch_row["validation_loss"]),
            "best_validation_loss_epoch": int(best_epoch_row["epoch"]),
            "best_validation_macro_f1": float(
                _number(training_summary.get("best_validation_macro_f1")) or best_epoch_row["validation_macro_f1"]
            ),
            "final_validation_macro_f1": float(
                _number(training_summary.get("final_validation_macro_f1")) or history[-1]["validation_macro_f1"]
            ),
        },
        "metrics": _extract_initial_metrics(report),
    }


def _load_autoresearch_artifact(path_or_dir: Path, *, protocol_label: str) -> dict[str, Any]:
    self_play_dir = _resolve_self_play_dir(path_or_dir)
    state = load_self_play_state(self_play_dir)
    if state is None:
        raise ProtocolProgressPlotError(f"Missing self-play state in {self_play_dir}")
    rounds_payload = state.get("rounds")
    if not isinstance(rounds_payload, list) or not rounds_payload:
        raise ProtocolProgressPlotError(f"No self-play rounds recorded in {self_play_dir}")
    rounds: list[dict[str, Any]] = []
    for round_payload in rounds_payload:
        item = _as_mapping(round_payload)
        blue_summary = _as_mapping(_as_mapping(item.get("blue_update")).get("summary"))
        blue_report_path = _text(_as_mapping(item.get("blue_update")).get("report_path"))
        reward_summary = _as_mapping(item.get("reward_summary"))
        candidate_alignment = _as_mapping(item.get("candidate_red_policy_alignment"))
        if not blue_summary:
            continue
        blue_loss_summary = _load_round_blue_loss_summary(blue_report_path)
        rounds.append(
            {
                "round_index": int(_number(item.get("round_index")) or 0),
                "macro_f1": float(_number(blue_summary.get("macro_f1")) or 0.0),
                "accuracy": float(_number(blue_summary.get("accuracy")) or 0.0),
                "cyber_f1": float(_number(blue_summary.get("cyber_f1")) or 0.0),
                "anomaly_f1": float(_number(blue_summary.get("anomaly_f1")) or 0.0),
                "deployment_ready": bool(blue_summary.get("deployment_ready")),
                "deployment_blocked_reason": _text(blue_summary.get("deployment_blocked_reason")),
                "reward_mean": float(_number(reward_summary.get("reward_mean")) or 0.0),
                "reward_min": float(_number(reward_summary.get("reward_min")) or 0.0),
                "reward_max": float(_number(reward_summary.get("reward_max")) or 0.0),
                "red_alignment": float(_number(candidate_alignment.get("joint_exact_match_accuracy")) or 0.0),
                "blue_report_path": blue_report_path,
                "best_validation_loss": float(blue_loss_summary.get("best_validation_loss") or 0.0),
                "final_validation_loss": float(blue_loss_summary.get("final_validation_loss") or 0.0),
                "final_train_loss": float(blue_loss_summary.get("final_train_loss") or 0.0),
                "best_epoch": int(blue_loss_summary.get("best_epoch") or 0),
                "epochs_completed": int(blue_loss_summary.get("epochs_completed") or 0),
            }
        )
    if not rounds:
        raise ProtocolProgressPlotError(f"No blue-update summaries recorded in {self_play_dir}")
    final_round = rounds[-1]
    best_round = max(rounds, key=lambda row: row["macro_f1"])
    return {
        "protocol_label": protocol_label,
        "self_play_dir": str(self_play_dir.resolve()),
        "rounds_completed": len(rounds),
        "rounds": rounds,
        "final_round": dict(final_round),
        "best_round": dict(best_round),
    }


def _load_round_blue_loss_summary(report_path: str | None) -> dict[str, Any]:
    if report_path is None:
        return {}
    path = Path(report_path).resolve()
    if not path.exists():
        return {}
    report = _read_json(path)
    blue_model = _as_mapping(report.get("blue_model"))
    training_summary = _as_mapping(blue_model.get("training_summary"))
    history = blue_model.get("training_history")
    final_train_loss = 0.0
    if isinstance(history, list) and history:
        final_train_loss = float(_number(_as_mapping(history[-1]).get("train_loss")) or 0.0)
    return {
        "best_validation_loss": float(_number(training_summary.get("best_validation_cross_entropy")) or 0.0),
        "final_validation_loss": float(_number(training_summary.get("final_validation_cross_entropy")) or 0.0),
        "final_train_loss": final_train_loss,
        "best_epoch": int(_number(training_summary.get("best_epoch")) or 0),
        "epochs_completed": int(_number(training_summary.get("epochs_completed")) or 0),
    }


def _annotate_metric_box(ax: plt.Axes, lines: Iterable[str], *, x: float = 0.02, y: float = 0.98) -> None:
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85, "edgecolor": PALETTE["muted"]},
    )


def build_initial_loss_figure(training: Mapping[str, Any]) -> plt.Figure:
    history = list(training["history"])
    epochs = [int(row["epoch"]) for row in history]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, [float(row["train_loss"]) for row in history], color=PALETTE["train"], linewidth=2.4, label="train_loss")
    ax.plot(
        epochs,
        [float(row["validation_loss"]) for row in history],
        color=PALETTE["validation"],
        linewidth=2.4,
        label="validation_loss",
    )
    ax.set_title(f"{training['protocol_label']} Initial Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    ax.set_xlim(min(epochs), max(epochs))
    metrics = _as_mapping(training.get("metrics"))
    summary = _as_mapping(training.get("training_summary"))
    _annotate_metric_box(
        ax,
        [
            f"epochs={int(summary.get('epochs') or 0)}",
            f"best_val_loss={float(summary.get('best_validation_loss') or 0.0):.4f}",
            f"best_val_macro_f1={float(summary.get('best_validation_macro_f1') or 0.0):.4f}",
            f"test_macro_f1={float(metrics.get('macro_f1') or 0.0):.4f}",
            f"test_anomaly_f1={float(metrics.get('anomaly_f1') or 0.0):.4f}",
        ],
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def build_unified_initial_loss_figure(fprime_training: Mapping[str, Any], mavlink_training: Mapping[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, training, accent in (
        (axes[0], fprime_training, PALETTE["fprime"]),
        (axes[1], mavlink_training, PALETTE["mavlink"]),
    ):
        history = list(training["history"])
        epochs = [int(row["epoch"]) for row in history]
        ax.plot(epochs, [float(row["train_loss"]) for row in history], color=PALETTE["train"], linewidth=2.3, label="train_loss")
        ax.plot(
            epochs,
            [float(row["validation_loss"]) for row in history],
            color=accent,
            linewidth=2.3,
            label="validation_loss",
        )
        ax.set_title(f"{training['protocol_label']} Initial Loss")
        ax.set_xlabel("Epoch")
        ax.set_xlim(min(epochs), max(epochs))
        metrics = _as_mapping(training.get("metrics"))
        summary = _as_mapping(training.get("training_summary"))
        _annotate_metric_box(
            ax,
            [
                f"best_val_loss={float(summary.get('best_validation_loss') or 0.0):.4f}",
                f"best_val_macro_f1={float(summary.get('best_validation_macro_f1') or 0.0):.4f}",
                f"test_macro_f1={float(metrics.get('macro_f1') or 0.0):.4f}",
            ],
        )
    axes[0].set_ylabel("Loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.suptitle("Initial Training Loss From Fresh Cross-Protocol Runs", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return fig


def _plot_autoresearch_axes(
    *,
    upper_ax: plt.Axes,
    middle_ax: plt.Axes,
    lower_ax: plt.Axes,
    autoresearch: Mapping[str, Any],
    accent: str,
) -> None:
    rounds = list(autoresearch["rounds"])
    round_ids = [int(row["round_index"]) for row in rounds]
    upper_ax.plot(round_ids, [float(row["macro_f1"]) for row in rounds], marker="o", linewidth=2.2, color=PALETTE["macro"], label="macro_f1")
    upper_ax.plot(round_ids, [float(row["cyber_f1"]) for row in rounds], marker="o", linewidth=2.2, color=PALETTE["cyber"], label="cyber_f1")
    upper_ax.plot(round_ids, [float(row["anomaly_f1"]) for row in rounds], marker="o", linewidth=2.2, color=PALETTE["anomaly"], label="anomaly_f1")
    upper_ax.set_ylabel("Blue F1")
    upper_ax.set_ylim(0.0, 1.05)
    upper_ax.set_xlim(min(round_ids), max(round_ids))
    best_round = _as_mapping(autoresearch.get("best_round"))
    final_round = _as_mapping(autoresearch.get("final_round"))
    _annotate_metric_box(
        upper_ax,
        [
            f"best_round={int(best_round.get('round_index') or 0)}",
            f"best_macro_f1={float(best_round.get('macro_f1') or 0.0):.4f}",
            f"final_macro_f1={float(final_round.get('macro_f1') or 0.0):.4f}",
            f"final_anomaly_f1={float(final_round.get('anomaly_f1') or 0.0):.4f}",
        ],
    )
    middle_ax.plot(
        round_ids,
        [float(row["best_validation_loss"]) for row in rounds],
        marker="o",
        linewidth=2.2,
        color=PALETTE["validation"],
        label="best_val_loss",
    )
    middle_ax.plot(
        round_ids,
        [float(row["final_validation_loss"]) for row in rounds],
        marker="o",
        linewidth=2.2,
        color=accent,
        label="final_val_loss",
    )
    middle_ax.plot(
        round_ids,
        [float(row["final_train_loss"]) for row in rounds],
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color=PALETTE["train"],
        label="final_train_loss",
    )
    middle_ax.set_ylabel("Blue Loss")
    middle_ax.set_xlim(min(round_ids), max(round_ids))
    lower_ax.bar(round_ids, [float(row["reward_mean"]) for row in rounds], color=PALETTE["reward"], alpha=0.6, label="reward_mean")
    lower_ax.plot(round_ids, [float(row["red_alignment"]) for row in rounds], marker="s", linewidth=2.0, color=accent, label="red_alignment")
    lower_ax.set_xlabel("Autoresearch Round")
    lower_ax.set_ylabel("Reward / Alignment")
    lower_ax.set_xlim(min(round_ids) - 0.5, max(round_ids) + 0.5)


def build_autoresearch_progress_figure(autoresearch: Mapping[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1]})
    accent = PALETTE["fprime"] if autoresearch["protocol_label"] == "F´" else PALETTE["mavlink"]
    _plot_autoresearch_axes(
        upper_ax=axes[0],
        middle_ax=axes[1],
        lower_ax=axes[2],
        autoresearch=autoresearch,
        accent=accent,
    )
    axes[0].set_title(f"{autoresearch['protocol_label']} Autoresearch Progress")
    upper_handles, upper_labels = axes[0].get_legend_handles_labels()
    middle_handles, middle_labels = axes[1].get_legend_handles_labels()
    lower_handles, lower_labels = axes[2].get_legend_handles_labels()
    fig.legend(
        upper_handles + middle_handles + lower_handles,
        upper_labels + middle_labels + lower_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


def build_unified_autoresearch_progress_figure(
    fprime_autoresearch: Mapping[str, Any],
    mavlink_autoresearch: Mapping[str, Any],
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(15, 10.5), sharex="col", gridspec_kw={"height_ratios": [2, 1.2, 1]})
    _plot_autoresearch_axes(
        upper_ax=axes[0, 0],
        middle_ax=axes[1, 0],
        lower_ax=axes[2, 0],
        autoresearch=fprime_autoresearch,
        accent=PALETTE["fprime"],
    )
    _plot_autoresearch_axes(
        upper_ax=axes[0, 1],
        middle_ax=axes[1, 1],
        lower_ax=axes[2, 1],
        autoresearch=mavlink_autoresearch,
        accent=PALETTE["mavlink"],
    )
    axes[0, 0].set_title("F´ Autoresearch")
    axes[0, 1].set_title("MAVLink Autoresearch")
    upper_handles, upper_labels = axes[0, 0].get_legend_handles_labels()
    middle_handles, middle_labels = axes[1, 0].get_legend_handles_labels()
    lower_handles, lower_labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(
        upper_handles + middle_handles + lower_handles,
        upper_labels + middle_labels + lower_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Autoresearch Progress Across Fresh F´ And MAVLink Runs", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return fig


def _summary_series(training: Mapping[str, Any], autoresearch: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    initial = _as_mapping(training.get("metrics"))
    final_round = _as_mapping(autoresearch.get("final_round"))
    best_round = _as_mapping(autoresearch.get("best_round"))
    return {
        "initial": {
            "macro_f1": float(initial.get("macro_f1") or 0.0),
            "cyber_f1": float(initial.get("cyber_f1") or 0.0),
            "anomaly_f1": float(initial.get("anomaly_f1") or 0.0),
        },
        "best_autoresearch": {
            "macro_f1": float(best_round.get("macro_f1") or 0.0),
            "cyber_f1": float(best_round.get("cyber_f1") or 0.0),
            "anomaly_f1": float(best_round.get("anomaly_f1") or 0.0),
        },
        "final_autoresearch": {
            "macro_f1": float(final_round.get("macro_f1") or 0.0),
            "cyber_f1": float(final_round.get("cyber_f1") or 0.0),
            "anomaly_f1": float(final_round.get("anomaly_f1") or 0.0),
        },
    }


def _plot_summary_axis(ax: plt.Axes, *, title: str, summary: Mapping[str, Mapping[str, float]], accent: str) -> None:
    metric_names = ["macro_f1", "cyber_f1", "anomaly_f1"]
    stage_names = ["initial", "best_autoresearch", "final_autoresearch"]
    colors = [accent, PALETTE["reward"], PALETTE["alignment"]]
    x_positions = list(range(len(metric_names)))
    width = 0.22
    for index, stage_name in enumerate(stage_names):
        offset = (index - 1) * width
        values = [float(_as_mapping(summary.get(stage_name)).get(metric_name) or 0.0) for metric_name in metric_names]
        bars = ax.bar(
            [position + offset for position in x_positions],
            values,
            width=width,
            label=stage_name,
            color=colors[index],
            alpha=0.82,
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=8)
    ax.set_title(title)
    ax.set_xticks(x_positions, ["macro_f1", "cyber_f1", "anomaly_f1"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")


def build_protocol_performance_summary_figure(
    training: Mapping[str, Any],
    autoresearch: Mapping[str, Any],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    accent = PALETTE["fprime"] if training["protocol_label"] == "F´" else PALETTE["mavlink"]
    _plot_summary_axis(
        ax,
        title=f"{training['protocol_label']} Initial Vs Autoresearch Performance",
        summary=_summary_series(training, autoresearch),
        accent=accent,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def build_unified_protocol_performance_summary_figure(
    fprime_training: Mapping[str, Any],
    fprime_autoresearch: Mapping[str, Any],
    mavlink_training: Mapping[str, Any],
    mavlink_autoresearch: Mapping[str, Any],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    _plot_summary_axis(
        axes[0],
        title="F´ Initial Vs Autoresearch",
        summary=_summary_series(fprime_training, fprime_autoresearch),
        accent=PALETTE["fprime"],
    )
    _plot_summary_axis(
        axes[1],
        title="MAVLink Initial Vs Autoresearch",
        summary=_summary_series(mavlink_training, mavlink_autoresearch),
        accent=PALETTE["mavlink"],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.suptitle("Fresh First-Pass Performance Vs Subsequent Autoresearch Progress", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return fig


def _caption_lines(
    *,
    fprime_training: Mapping[str, Any],
    fprime_autoresearch: Mapping[str, Any],
    mavlink_training: Mapping[str, Any],
    mavlink_autoresearch: Mapping[str, Any],
) -> list[str]:
    fprime_initial = _as_mapping(fprime_training.get("metrics"))
    fprime_final = _as_mapping(fprime_autoresearch.get("final_round"))
    mavlink_initial = _as_mapping(mavlink_training.get("metrics"))
    mavlink_final = _as_mapping(mavlink_autoresearch.get("final_round"))
    return [
        "# Protocol Progress Captions",
        "",
        "## Unified Initial Training Loss",
        f"Fresh first-pass blue training loss curves for F´ and MAVLink. F´ closed at macro_f1={float(fprime_initial.get('macro_f1') or 0.0):.4f}; MAVLink closed at macro_f1={float(mavlink_initial.get('macro_f1') or 0.0):.4f}.",
        "",
        "## Unified Autoresearch Progress",
        f"Round-level blue F1, blue loss, mean red reward, and candidate red alignment after the initial pass. F´ ended at macro_f1={float(fprime_final.get('macro_f1') or 0.0):.4f}; MAVLink ended at macro_f1={float(mavlink_final.get('macro_f1') or 0.0):.4f}.",
        "",
        "## Unified Performance Summary",
        "Grouped comparison of initial first-pass performance against best and final autoresearch rounds for each protocol.",
    ]


def generate_protocol_progress_plots(
    *,
    fprime_training_dir: Path,
    mavlink_training_dir: Path,
    fprime_self_play_dir: Path,
    mavlink_self_play_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _configure_matplotlib()
    resolved_output_dir = output_dir.resolve()
    figures_dir = resolved_output_dir / "figures"
    fprime_training = _load_initial_training_artifact(fprime_training_dir, protocol_label="F´")
    mavlink_training = _load_initial_training_artifact(mavlink_training_dir, protocol_label="MAVLink")
    fprime_autoresearch = _load_autoresearch_artifact(fprime_self_play_dir, protocol_label="F´")
    mavlink_autoresearch = _load_autoresearch_artifact(mavlink_self_play_dir, protocol_label="MAVLink")

    figure_specs = [
        ("fig01_unified_initial_training_loss", "Unified Initial Training Loss", build_unified_initial_loss_figure(fprime_training, mavlink_training)),
        ("fig02_fprime_initial_training_loss", "F´ Initial Training Loss", build_initial_loss_figure(fprime_training)),
        ("fig03_mavlink_initial_training_loss", "MAVLink Initial Training Loss", build_initial_loss_figure(mavlink_training)),
        (
            "fig04_unified_autoresearch_progress",
            "Unified Autoresearch Progress",
            build_unified_autoresearch_progress_figure(fprime_autoresearch, mavlink_autoresearch),
        ),
        ("fig05_fprime_autoresearch_progress", "F´ Autoresearch Progress", build_autoresearch_progress_figure(fprime_autoresearch)),
        ("fig06_mavlink_autoresearch_progress", "MAVLink Autoresearch Progress", build_autoresearch_progress_figure(mavlink_autoresearch)),
        (
            "fig07_unified_protocol_performance_summary",
            "Unified Performance Summary",
            build_unified_protocol_performance_summary_figure(
                fprime_training,
                fprime_autoresearch,
                mavlink_training,
                mavlink_autoresearch,
            ),
        ),
        (
            "fig08_fprime_protocol_performance_summary",
            "F´ Performance Summary",
            build_protocol_performance_summary_figure(fprime_training, fprime_autoresearch),
        ),
        (
            "fig09_mavlink_protocol_performance_summary",
            "MAVLink Performance Summary",
            build_protocol_performance_summary_figure(mavlink_training, mavlink_autoresearch),
        ),
    ]
    assets: list[dict[str, Any]] = []
    for asset_id, title, figure in figure_specs:
        asset_paths = _save_figure_variants(figure, figures_dir, _slug(asset_id))
        assets.append(
            {
                "asset_id": asset_id,
                "title": title,
                **asset_paths,
            }
        )

    captions_path = resolved_output_dir / "captions.md"
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path.write_text(
        "\n".join(
            _caption_lines(
                fprime_training=fprime_training,
                fprime_autoresearch=fprime_autoresearch,
                mavlink_training=mavlink_training,
                mavlink_autoresearch=mavlink_autoresearch,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": PROTOCOL_PROGRESS_MANIFEST_SCHEMA_VERSION,
        "record_kind": "protocol_progress_manifest",
        "sources": {
            "fprime_training_dir": str(Path(fprime_training_dir).resolve()),
            "mavlink_training_dir": str(Path(mavlink_training_dir).resolve()),
            "fprime_self_play_dir": str(Path(fprime_self_play_dir).resolve()),
            "mavlink_self_play_dir": str(Path(mavlink_self_play_dir).resolve()),
        },
        "assets": assets,
        "summaries": {
            "fprime": {
                "initial": fprime_training["metrics"],
                "training_summary": fprime_training["training_summary"],
                "autoresearch_final": fprime_autoresearch["final_round"],
                "autoresearch_best": fprime_autoresearch["best_round"],
            },
            "mavlink": {
                "initial": mavlink_training["metrics"],
                "training_summary": mavlink_training["training_summary"],
                "autoresearch_final": mavlink_autoresearch["final_round"],
                "autoresearch_best": mavlink_autoresearch["best_round"],
            },
        },
        "captions_path": str(captions_path.resolve()),
    }
    _save_json(resolved_output_dir / "asset_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fresh-run progress plots that compare first-pass blue training against "
            "subsequent autoresearch progress for F´ and MAVLink."
        )
    )
    parser.add_argument("--fprime-training-dir", required=True)
    parser.add_argument("--mavlink-training-dir", required=True)
    parser.add_argument("--fprime-self-play-dir", required=True)
    parser.add_argument("--mavlink-self-play-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_protocol_progress_plots(
        fprime_training_dir=Path(args.fprime_training_dir),
        mavlink_training_dir=Path(args.mavlink_training_dir),
        fprime_self_play_dir=Path(args.fprime_self_play_dir),
        mavlink_self_play_dir=Path(args.mavlink_self_play_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
