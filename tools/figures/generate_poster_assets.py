#!/usr/bin/env python3
"""Generate poster-ready figures and captions from repository artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared.feature_policy import BLUE_FEATURE_POLICY_POSTER_DEFAULT, load_blue_feature_policy

POSTER_ASSET_MANIFEST_SCHEMA_VERSION = "poster_asset_manifest.v1"

PALETTE = {
    "ink": "#1f2430",
    "paper": "#f5f2ea",
    "blue": "#1b4965",
    "teal": "#1f7a8c",
    "green": "#5b8e55",
    "gold": "#c58b2b",
    "orange": "#c97239",
    "red": "#a94438",
    "sand": "#d8c8a8",
    "muted": "#7d8799",
}


class PosterAssetError(ValueError):
    """Raised when poster assets cannot be generated safely."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PosterAssetError(f"{path} must contain a JSON object")
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


def _compact_category(value: Any, default: str = "unknown") -> str:
    text = (_text(value) or default).strip().lower()
    chars: list[str] = []
    last_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_sep = False
            continue
        if not last_sep:
            chars.append("_")
            last_sep = True
    compact = "".join(chars).strip("_")
    return compact or default


def _display_slug(value: str) -> str:
    normalized = _compact_category(value, "figure")
    return normalized.replace("__", "_")


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "font.size": 11,
            "axes.titleweight": "bold",
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


def _add_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str | None = None,
    fontsize: int = 11,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        facecolor=facecolor,
        edgecolor=edgecolor or PALETTE["ink"],
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2.0,
        y + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
    )


def _add_arrow(
    ax: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    text: str | None = None,
    text_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            (start[0] + end[0]) / 2.0 + text_offset[0],
            (start[1] + end[1]) / 2.0 + text_offset[1],
            text,
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            weight="bold",
        )


def _diagram_axes(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    return fig, ax


def build_architecture_figure() -> plt.Figure:
    fig, ax = _diagram_axes((14, 8))
    ax.text(0.02, 0.96, "Protocol-General Poster Architecture", fontsize=20, weight="bold", ha="left")
    ax.text(0.02, 0.92, "Real protocol stacks feed one shared evidence contract, one canonical surface, and one evaluation story.", fontsize=12, ha="left")

    ax.text(0.17, 0.84, "Real Stacks", fontsize=14, weight="bold", ha="center")
    ax.text(0.50, 0.84, "Shared Evidence And Semantics", fontsize=14, weight="bold", ha="center")
    ax.text(0.83, 0.84, "Learning, Runtime, And Evidence", fontsize=14, weight="bold", ha="center")

    _add_box(ax, x=0.06, y=0.58, width=0.22, height=0.16, text="F´ real stack\nlive schedules + packets + logs", facecolor="#dce9f2")
    _add_box(ax, x=0.06, y=0.32, width=0.22, height=0.16, text="MAVLink real stack\nArduPilot SITL + MAVProxy", facecolor="#dce9f2")

    _add_box(ax, x=0.38, y=0.65, width=0.24, height=0.12, text="raw_packets.jsonl\nraw_transactions.jsonl", facecolor="#e8eadf")
    _add_box(ax, x=0.38, y=0.47, width=0.24, height=0.12, text="canonical_command_rows.jsonl\nnormalized_state + semantics", facecolor="#e8eadf")
    _add_box(ax, x=0.38, y=0.29, width=0.24, height=0.12, text="poster_blue_default\nfeature policy", facecolor="#e8eadf")

    _add_box(ax, x=0.72, y=0.66, width=0.22, height=0.11, text="Blue neural detector\nposter_blue_mlp_v1", facecolor="#f3e7d8")
    _add_box(ax, x=0.72, y=0.48, width=0.22, height=0.11, text="Bounded red policy\ntranscript + reward", facecolor="#f3e7d8")
    _add_box(ax, x=0.72, y=0.30, width=0.22, height=0.11, text="Runtime bundle + reports\nblue_model.json + figures", facecolor="#f3e7d8")

    _add_arrow(ax, start=(0.28, 0.66), end=(0.38, 0.71), color=PALETTE["blue"])
    _add_arrow(ax, start=(0.28, 0.40), end=(0.38, 0.71), color=PALETTE["blue"])
    _add_arrow(ax, start=(0.50, 0.65), end=(0.50, 0.59), color=PALETTE["green"])
    _add_arrow(ax, start=(0.50, 0.47), end=(0.50, 0.41), color=PALETTE["green"])
    _add_arrow(ax, start=(0.62, 0.71), end=(0.72, 0.71), color=PALETTE["orange"])
    _add_arrow(ax, start=(0.62, 0.53), end=(0.72, 0.53), color=PALETTE["orange"])
    _add_arrow(ax, start=(0.62, 0.35), end=(0.72, 0.35), color=PALETTE["orange"])
    _add_arrow(ax, start=(0.83, 0.66), end=(0.83, 0.59), color=PALETTE["red"], text="offline reward", text_offset=(0.08, 0.0))
    _add_arrow(ax, start=(0.83, 0.48), end=(0.83, 0.41), color=PALETTE["red"], text="evaluation", text_offset=(0.08, 0.0))
    return fig


def build_pipeline_figure() -> plt.Figure:
    fig, ax = _diagram_axes((15, 6.5))
    ax.text(0.02, 0.94, "Raw-To-Canonical Evidence Pipeline", fontsize=20, weight="bold", ha="left")
    ax.text(0.02, 0.89, "Both real protocol families emit the same replayable artifact contract before learning touches the data.", fontsize=12, ha="left")

    _add_box(ax, x=0.04, y=0.44, width=0.14, height=0.14, text="real schedules\n+ live services", facecolor="#dce9f2")
    _add_box(ax, x=0.24, y=0.44, width=0.14, height=0.14, text="pcaps + logs\n+ sender traces", facecolor="#dce9f2")
    _add_box(ax, x=0.44, y=0.44, width=0.16, height=0.14, text="raw_packets.jsonl", facecolor="#e8eadf")
    _add_box(ax, x=0.64, y=0.44, width=0.16, height=0.14, text="raw_transactions.jsonl", facecolor="#e8eadf")
    _add_box(ax, x=0.84, y=0.44, width=0.14, height=0.14, text="canonical_command_rows.jsonl", facecolor="#f3e7d8")

    _add_box(ax, x=0.64, y=0.17, width=0.16, height=0.12, text="audit + provenance\nretained for replay", facecolor="#f0ebe0", fontsize=10)
    _add_box(ax, x=0.84, y=0.17, width=0.14, height=0.12, text="dataset.jsonl\nruntime rows", facecolor="#f0ebe0", fontsize=10)

    _add_arrow(ax, start=(0.18, 0.51), end=(0.24, 0.51), color=PALETTE["blue"])
    _add_arrow(ax, start=(0.38, 0.51), end=(0.44, 0.51), color=PALETTE["green"])
    _add_arrow(ax, start=(0.60, 0.51), end=(0.64, 0.51), color=PALETTE["green"])
    _add_arrow(ax, start=(0.80, 0.51), end=(0.84, 0.51), color=PALETTE["orange"])
    _add_arrow(ax, start=(0.72, 0.44), end=(0.72, 0.29), color=PALETTE["muted"], text="forensics", text_offset=(0.05, 0.0))
    _add_arrow(ax, start=(0.91, 0.44), end=(0.91, 0.29), color=PALETTE["red"], text="learning", text_offset=(0.05, 0.0))

    ax.text(0.30, 0.67, "protocol-specific parsing stays here", color=PALETTE["blue"], fontsize=11, ha="center", weight="bold")
    ax.text(0.71, 0.67, "shared raw contract", color=PALETTE["green"], fontsize=11, ha="center", weight="bold")
    ax.text(0.91, 0.67, "protocol-neutral learned surface", color=PALETTE["orange"], fontsize=11, ha="center", weight="bold")
    return fig


def build_blue_red_loop_figure() -> plt.Figure:
    fig, ax = _diagram_axes((14, 7))
    ax.text(0.02, 0.95, "Blue-Red Auto-Research Loop", fontsize=20, weight="bold", ha="left")
    ax.text(0.02, 0.90, "The current loop is checkpointed and offline: the opponent is frozen within a round, and learned red is evaluated through replay.", fontsize=12, ha="left")

    _add_box(ax, x=0.08, y=0.58, width=0.20, height=0.13, text="1. generate or replay\nmixed-protocol datasets", facecolor="#dce9f2")
    _add_box(ax, x=0.38, y=0.58, width=0.20, height=0.13, text="2. rebuild replayable\nred examples", facecolor="#dce9f2")
    _add_box(ax, x=0.68, y=0.58, width=0.22, height=0.13, text="3. freeze blue bundle\n+ score reward", facecolor="#f3e7d8")
    _add_box(ax, x=0.68, y=0.30, width=0.22, height=0.13, text="4. retrain red from\nwarm-start + replay", facecolor="#f0ebe0")
    _add_box(ax, x=0.38, y=0.30, width=0.20, height=0.13, text="5. retrain blue on\nround dataset", facecolor="#e8eadf")
    _add_box(ax, x=0.08, y=0.30, width=0.20, height=0.13, text="6. checkpoint state\n+ write reports", facecolor="#e8eadf")

    _add_arrow(ax, start=(0.28, 0.645), end=(0.38, 0.645), color=PALETTE["blue"])
    _add_arrow(ax, start=(0.58, 0.645), end=(0.68, 0.645), color=PALETTE["orange"])
    _add_arrow(ax, start=(0.79, 0.58), end=(0.79, 0.43), color=PALETTE["red"])
    _add_arrow(ax, start=(0.68, 0.365), end=(0.58, 0.365), color=PALETTE["green"])
    _add_arrow(ax, start=(0.38, 0.365), end=(0.28, 0.365), color=PALETTE["green"])
    _add_arrow(ax, start=(0.18, 0.43), end=(0.18, 0.58), color=PALETTE["muted"], text="next round", text_offset=(-0.07, 0.0))

    ax.text(0.80, 0.51, "reward uses true outcomes\nplus blue detections", fontsize=10, color=PALETTE["red"], ha="center")
    ax.text(0.18, 0.20, "state, checkpoints, and round artifacts make the loop resumable and reviewable", fontsize=10, color=PALETTE["muted"], ha="left")
    return fig


def _matrix_row_metric(
    rows: Iterable[Mapping[str, Any]],
    *,
    evaluation_name: str,
    scope: str,
    model_name: str,
    heldout_value: str | None = None,
    metric_path: tuple[str, ...] = ("multiclass_macro_f1",),
) -> float | None:
    for row in rows:
        if row.get("evaluation_name") != evaluation_name or row.get("scope") != scope or row.get("model_name") != model_name:
            continue
        if heldout_value is not None and row.get("heldout_value") != heldout_value:
            continue
        metrics = _as_mapping(row.get("metrics"))
        current: Any = metrics
        for key in metric_path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        return _number(current)
    return None


def build_cross_protocol_figure(report: Mapping[str, Any]) -> plt.Figure:
    matrix_rows = list(report.get("matrix_rows", []))
    shortcut_baselines = _as_mapping(report.get("shortcut_baselines"))
    cross_protocol = _as_mapping(report.get("cross_protocol_generalization"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor(PALETTE["paper"])
    ax_left, ax_right = axes

    evaluation_order = [
        ("repeated_grouped_cv", "grouped"),
        ("scenario_family_holdout", "window"),
        ("command_family_holdout", "command"),
        ("protocol_family_holdout", "protocol"),
    ]
    macro_values: list[float] = []
    anomaly_values: list[float] = []
    labels: list[str] = []
    for evaluation_name, label in evaluation_order:
        labels.append(label)
        macro_values.append(
            _matrix_row_metric(matrix_rows, evaluation_name=evaluation_name, scope="aggregate", model_name="neural_net")
            or 0.0
        )
        anomaly_values.append(
            _matrix_row_metric(
                matrix_rows,
                evaluation_name=evaluation_name,
                scope="aggregate",
                model_name="neural_net",
                metric_path=("anomaly_f1",),
            )
            or 0.0
        )

    x = list(range(len(labels)))
    width = 0.34
    ax_left.bar([value - width / 2.0 for value in x], macro_values, width=width, color=PALETTE["blue"], label="macro F1")
    ax_left.bar([value + width / 2.0 for value in x], anomaly_values, width=width, color=PALETTE["orange"], label="anomaly F1")
    ax_left.set_title("Aggregate evaluation views")
    ax_left.set_ylabel("score")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_ylim(0.0, max([0.9] + macro_values + anomaly_values))
    ax_left.legend(frameon=False, loc="upper right")
    ax_left.grid(axis="y", color="#d8d5cc", linewidth=0.8, alpha=0.7)

    heldout_values = list(cross_protocol.get("evaluated_values", []))
    heldout_macro: list[float] = []
    heldout_anomaly: list[float] = []
    for heldout_value in heldout_values:
        heldout_macro.append(
            _matrix_row_metric(
                matrix_rows,
                evaluation_name="protocol_family_holdout",
                scope="heldout_value",
                model_name="neural_net",
                heldout_value=heldout_value,
            )
            or 0.0
        )
        heldout_anomaly.append(
            _matrix_row_metric(
                matrix_rows,
                evaluation_name="protocol_family_holdout",
                scope="heldout_value",
                model_name="neural_net",
                heldout_value=heldout_value,
                metric_path=("anomaly_f1",),
            )
            or 0.0
        )
    protocol_x = list(range(len(heldout_values)))
    ax_right.set_title("Held-out protocol families")
    ax_right.set_ylabel("score")
    if heldout_values:
        ax_right.bar([value - width / 2.0 for value in protocol_x], heldout_macro, width=width, color=PALETTE["teal"], label="macro F1")
        ax_right.bar([value + width / 2.0 for value in protocol_x], heldout_anomaly, width=width, color=PALETTE["gold"], label="anomaly F1")
        ax_right.set_xticks(protocol_x)
        ax_right.set_xticklabels(heldout_values)
        ax_right.set_ylim(0.0, max([0.9] + heldout_macro + heldout_anomaly))
        ax_right.legend(frameon=False, loc="upper right")
    else:
        ax_right.set_xticks([0])
        ax_right.set_xticklabels(["not feasible"])
        ax_right.set_ylim(0.0, 1.0)
        ax_right.text(
            0.5,
            0.55,
            "Protocol holdout is infeasible\nfor this dataset mix.",
            transform=ax_right.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=PALETTE["muted"],
        )
    ax_right.grid(axis="y", color="#d8d5cc", linewidth=0.8, alpha=0.7)

    protocol_only = _as_mapping(shortcut_baselines.get("protocol_only"))
    raw_shortcuts = _as_mapping(shortcut_baselines.get("raw_protocol_shortcuts"))
    note_lines = [
        f"protocol_only best={float(protocol_only.get('best_metric_value') or 0.0):.2f}",
        f"raw_shortcuts best={float(raw_shortcuts.get('best_metric_value') or 0.0):.2f}",
    ]
    if bool(cross_protocol.get("feasible")):
        note_lines.append("protocol holdout feasible across: " + ", ".join(str(value) for value in heldout_values))
    else:
        note_lines.append("protocol holdout infeasible for this dataset")
    fig.suptitle("Cross-Protocol Generalization", fontsize=18, fontweight="bold", x=0.07, ha="left", y=0.98)
    fig.text(0.07, 0.02, " | ".join(note_lines), fontsize=10, color=PALETTE["muted"])
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    return fig


def build_adversary_vs_blue_figure(report: Mapping[str, Any]) -> plt.Figure:
    summary_rows = [dict(row) for row in report.get("summary_rows", []) if row.get("scope") == "overall"]
    comparison_rows = [dict(row) for row in report.get("comparison_rows", []) if row.get("scope") == "overall"]
    static_rows = [row for row in summary_rows if row.get("adversary_kind") == "static_schedule_replay"]
    learned_rows = [row for row in summary_rows if row.get("adversary_kind") == "learned_red_policy_retrieval"]
    if not summary_rows:
        raise PosterAssetError("red_blue_evaluation report must contain overall summary_rows")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=False)
    fig.patch.set_facecolor(PALETTE["paper"])
    ax_top, ax_bottom = axes

    top_rows = sorted(summary_rows, key=lambda row: (row.get("adversary_kind") != "static_schedule_replay", row.get("red_id") or "", row.get("transcript_length") or 0))
    top_labels: list[str] = []
    reward_values: list[float] = []
    success_values: list[float] = []
    for row in top_rows:
        if row.get("adversary_kind") == "static_schedule_replay":
            top_labels.append(f"static\nn={row['transcript_length']}")
        else:
            top_labels.append(f"{row.get('red_id')}\nn={row['transcript_length']}")
        reward_values.append(float(row.get("coverage_adjusted_reward_mean") or 0.0))
        success_values.append(float(row.get("coverage_adjusted_adversary_success_rate") or 0.0))
    x = list(range(len(top_labels)))
    ax_top.plot(x, reward_values, color=PALETTE["orange"], marker="o", linewidth=2.5, label="coverage-adjusted reward")
    ax_top.plot(x, success_values, color=PALETTE["red"], marker="s", linewidth=2.5, label="coverage-adjusted success")
    ax_top.set_title("Observed difficulty under static and learned adversaries")
    ax_top.set_ylabel("score")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(top_labels)
    ax_top.grid(axis="y", color="#d8d5cc", linewidth=0.8, alpha=0.7)
    ax_top.legend(frameon=False, loc="upper left")

    learned_labels = [f"{row.get('red_id')}\nn={row['transcript_length']}" for row in learned_rows] or ["no learned rows"]
    diag_x = list(range(len(learned_labels)))
    coverage_values = [float(row.get("retrieval_coverage_rate") or 0.0) for row in learned_rows] or [0.0]
    alignment_values = [float(row.get("alignment_joint_exact_match_accuracy") or 0.0) for row in learned_rows] or [0.0]
    precision_deltas = [float(row.get("delta_blue_precision_under_attack") or 0.0) for row in comparison_rows] or [0.0]
    ax_bottom.plot(diag_x, coverage_values, color=PALETTE["blue"], marker="o", linewidth=2.5, label="retrieval coverage")
    ax_bottom.plot(diag_x, alignment_values, color=PALETTE["green"], marker="s", linewidth=2.5, label="joint action alignment")
    ax_bottom.plot(diag_x, precision_deltas, color=PALETTE["teal"], marker="^", linewidth=2.5, label="delta blue precision")
    ax_bottom.set_title("Learned red diagnostics against the static baseline")
    ax_bottom.set_ylabel("score")
    ax_bottom.set_ylim(0.0, max([1.0] + coverage_values + alignment_values + precision_deltas))
    ax_bottom.set_xticks(diag_x)
    ax_bottom.set_xticklabels(learned_labels)
    ax_bottom.grid(axis="y", color="#d8d5cc", linewidth=0.8, alpha=0.7)
    ax_bottom.legend(frameon=False, loc="upper left")

    baseline_note = ""
    if static_rows:
        baseline = static_rows[0]
        baseline_note = (
            f"static baseline precision={float(baseline.get('blue_precision_under_attack') or 0.0):.2f} "
            f"recall={float(baseline.get('blue_recall_under_attack') or 0.0):.2f}"
        )
    fig.suptitle("Adversary-Vs-Blue Curves", fontsize=18, fontweight="bold", x=0.08, ha="left", y=0.98)
    if baseline_note:
        fig.text(0.08, 0.02, baseline_note, fontsize=10, color=PALETTE["muted"])
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    return fig


def _allowed_family(name: str) -> tuple[str, str]:
    if name == "platform_family":
        return "Platform context", name
    if name.startswith("actor_context."):
        return "Sender context", name
    if name.startswith("mission_context."):
        return "Mission / window context", name
    if name.startswith("command_semantics."):
        return "Canonical command semantics", name
    if name.startswith("argument_profile."):
        return "Argument profile", name
    if name.startswith("normalized_state."):
        return "Normalized state summaries", name
    if name.startswith("recent_behavior."):
        return "Bounded recent behavior", name
    return "Other poster-safe fields", name


def _forbidden_family(name: str) -> str:
    if name in {"protocol_family", "protocol_version", "service_id", "command_id", "target_node_id"}:
        return "Protocol identity shortcuts"
    if name in {"run_id", "episode_id"} or name.startswith("audit_context.") or name.startswith("observability."):
        return "Audit / provenance only"
    if name in {"label", "label_name", "attack_family"}:
        return "Ground-truth / evaluation metadata"
    if name in {"actor", "actor_role", "actor_trust", "actor_context.trust_score"}:
        return "Raw identity / trust leakage"
    if name in {"arg_count", "arg_norm", "arg_out_of_range"}:
        return "Legacy flat argument surface"
    if name in {
        "resp_bytes",
        "latency_ms",
        "gds_accept",
        "sat_success",
        "timeout",
        "response_code",
        "request_to_uplink_ms",
        "uplink_to_sat_response_ms",
        "sat_response_to_final_ms",
        "response_direction_seen",
        "final_observed_on_wire",
        "txn_warning_events",
        "txn_error_events",
    }:
        return "Terminal / response-only outcomes"
    if name in {"panomaly", "pcyber", "rules", "novelty"}:
        return "Legacy auxiliary detector outputs"
    if any(name.startswith(prefix) for prefix in ("target_", "peer_", "fprime_", "mavlink_", "native_", "raw_")):
        return "Raw protocol-native fields"
    return "Other forbidden leakage"


def _feature_policy_table_rows() -> tuple[list[list[str]], list[list[str]]]:
    policy = load_blue_feature_policy(BLUE_FEATURE_POLICY_POSTER_DEFAULT)
    allowed_groups: dict[str, list[str]] = defaultdict(list)
    for name in policy["allowed_features"]:
        group_name, example = _allowed_family(str(name))
        allowed_groups[group_name].append(example)

    forbidden_groups: dict[str, list[str]] = defaultdict(list)
    for name in policy["forbidden_features"]:
        forbidden_groups[_forbidden_family(str(name))].append(str(name))
    for prefix in policy["forbidden_prefixes"]:
        forbidden_groups[_forbidden_family(str(prefix))].append(str(prefix))

    allowed_rows = []
    for group_name in sorted(allowed_groups):
        examples = allowed_groups[group_name][:3]
        allowed_rows.append(
            [
                group_name,
                str(len(allowed_groups[group_name])),
                _wrap(", ".join(examples), 26),
            ]
        )
    forbidden_rows = []
    for group_name in sorted(forbidden_groups):
        examples = forbidden_groups[group_name][:3]
        forbidden_rows.append(
            [
                group_name,
                str(len(forbidden_groups[group_name])),
                _wrap(", ".join(examples), 26),
            ]
        )
    return allowed_rows, forbidden_rows


def _styled_table(ax: plt.Axes, rows: list[list[str]], title: str, header_color: str) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    table = ax.table(
        cellText=rows,
        colLabels=["Family", "Count", "Examples"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["muted"])
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color=PALETTE["ink"])
        else:
            cell.set_facecolor(PALETTE["paper"])


def build_feature_family_table_figure() -> plt.Figure:
    allowed_rows, forbidden_rows = _feature_policy_table_rows()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(PALETTE["paper"])
    _styled_table(axes[0], allowed_rows, "Allowed poster-blue families", "#dce9f2")
    _styled_table(axes[1], forbidden_rows, "Forbidden poster-blue families", "#f3d8d3")
    fig.suptitle("Blue Feature Families", fontsize=18, fontweight="bold", x=0.06, ha="left", y=0.98)
    fig.text(
        0.06,
        0.02,
        "Counts come directly from the machine-checked poster_blue_default allowlist and denylist.",
        fontsize=10,
        color=PALETTE["muted"],
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    return fig


def _asset_captions(evaluation_matrix: Mapping[str, Any], red_blue: Mapping[str, Any]) -> dict[str, str]:
    cross_protocol = _as_mapping(evaluation_matrix.get("cross_protocol_generalization"))
    heldout_values = ", ".join(str(value) for value in cross_protocol.get("evaluated_values", [])) or "none"
    learned_rows = [row for row in red_blue.get("summary_rows", []) if row.get("adversary_kind") == "learned_red_policy_retrieval"]
    best_coverage = max([float(row.get("retrieval_coverage_rate") or 0.0) for row in learned_rows], default=0.0)
    best_alignment = max([float(row.get("alignment_joint_exact_match_accuracy") or 0.0) for row in learned_rows], default=0.0)
    return {
        "fig01_architecture_overview": (
            "Real F´ and MAVLink stacks now terminate in one shared raw/canonical contract, "
            "one poster-blue neural detector, and one bounded red/evaluation layer."
        ),
        "fig02_raw_to_canonical_pipeline": (
            "Protocol-specific capture stays confined to the left side of the pipeline; "
            "the learned surface begins only after shared raw and canonical artifacts are written."
        ),
        "fig03_blue_red_loop": (
            "The current blue/red workflow is checkpointed and offline: each round freezes the opponent, "
            "scores replayed evidence, updates the learner, and writes resumable artifacts."
        ),
        "fig04_cross_protocol_generalization": (
            "Grouped, command-family, scenario-window, and protocol-family views come from one evaluation matrix. "
            f"Protocol holdout is feasible across: {heldout_values}."
        ),
        "fig05_adversary_vs_blue_curves": (
            "Static replay pressure and learned bounded red checkpoints are compared on the same blue bundle. "
            f"Current learned-red diagnostics reach retrieval coverage up to {best_coverage:.3f} and joint action alignment up to {best_alignment:.3f}."
        ),
        "fig06_blue_feature_families": (
            "The poster claim depends on what the blue model is forbidden to see. "
            "This table is built directly from the machine-checked allowlist and denylist."
        ),
    }


def _caption_markdown(assets: list[Mapping[str, Any]], manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Poster Figure Captions",
        "",
        "These captions are generated alongside the poster assets so the figure message stays synchronized with the repository outputs.",
        "",
        f"- Evaluation matrix source: `{manifest['sources']['evaluation_matrix_path']}`",
        f"- Red-vs-blue source: `{manifest['sources']['red_blue_evaluation_path']}`",
        "",
    ]
    for asset in assets:
        lines.append(f"## {asset['asset_id']} — {asset['title']}")
        lines.append("")
        lines.append(f"- SVG: `{asset['svg_path']}`")
        lines.append(f"- PNG: `{asset['png_path']}`")
        lines.append(f"- Caption: {asset['caption']}")
        lines.append("")
    return "\n".join(lines)


def generate_poster_assets(
    *,
    evaluation_matrix_path: Path,
    red_blue_evaluation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _configure_matplotlib()
    evaluation_matrix = _read_json(evaluation_matrix_path.resolve())
    red_blue = _read_json(red_blue_evaluation_path.resolve())
    resolved_output_dir = output_dir.resolve()
    figure_dir = resolved_output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    captions = _asset_captions(evaluation_matrix, red_blue)
    figure_builders = [
        ("fig01_architecture_overview", "Architecture Overview", build_architecture_figure()),
        ("fig02_raw_to_canonical_pipeline", "Raw-To-Canonical Pipeline", build_pipeline_figure()),
        ("fig03_blue_red_loop", "Blue-Red Loop", build_blue_red_loop_figure()),
        ("fig04_cross_protocol_generalization", "Cross-Protocol Generalization", build_cross_protocol_figure(evaluation_matrix)),
        ("fig05_adversary_vs_blue_curves", "Adversary-Vs-Blue Curves", build_adversary_vs_blue_figure(red_blue)),
        ("fig06_blue_feature_families", "Blue Feature Families", build_feature_family_table_figure()),
    ]

    assets: list[dict[str, Any]] = []
    for asset_id, title, figure in figure_builders:
        saved = _save_figure_variants(figure, figure_dir, _display_slug(asset_id))
        assets.append(
            {
                "asset_id": asset_id,
                "title": title,
                "caption": captions[asset_id],
                **saved,
            }
        )

    manifest = {
        "schema_version": POSTER_ASSET_MANIFEST_SCHEMA_VERSION,
        "record_kind": "poster_asset_manifest",
        "sources": {
            "evaluation_matrix_path": str(evaluation_matrix_path.resolve()),
            "red_blue_evaluation_path": str(red_blue_evaluation_path.resolve()),
            "feature_policy_name": BLUE_FEATURE_POLICY_POSTER_DEFAULT,
        },
        "output_dir": str(resolved_output_dir),
        "assets": assets,
    }
    _save_json(resolved_output_dir / "asset_manifest.json", manifest)
    (resolved_output_dir / "captions.md").write_text(_caption_markdown(assets, manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate poster-ready diagrams, plots, and caption notes from the saved evaluation matrix, "
            "red-vs-blue report, and machine-checked blue feature policy."
        )
    )
    parser.add_argument("--evaluation-matrix", type=Path, required=True, help="Path to evaluation_matrix.json")
    parser.add_argument("--red-blue-evaluation", type=Path, required=True, help="Path to red_blue_evaluation.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated poster assets")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_poster_assets(
        evaluation_matrix_path=args.evaluation_matrix.resolve(),
        red_blue_evaluation_path=args.red_blue_evaluation.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "record_kind": manifest["record_kind"],
                "asset_count": len(manifest["assets"]),
                "manifest_path": str((args.output_dir.resolve() / "asset_manifest.json").resolve()),
                "captions_path": str((args.output_dir.resolve() / "captions.md").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
