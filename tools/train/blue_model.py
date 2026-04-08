#!/usr/bin/env python3
from __future__ import annotations

import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from runtime import CLASS_NAMES, export_sklearn_mlp
from tools.shared.feature_policy import (
    BLUE_FEATURE_POLICY_POSTER_DEFAULT,
    validate_blue_feature_names,
)

POSTER_BLUE_MODEL_FAMILY = "poster_blue_mlp_v1"


@dataclass(frozen=True)
class PosterBlueArchitectureConfig:
    hidden_layer_sizes: tuple[int, ...] = (96, 48)
    activation: str = "relu"
    feature_encoding: str = "canonical_allowlisted_numeric_plus_stable_token_id"
    scaler: str = "standard_scaler"
    output_head: str = "three_class_softmax"
    family: str = POSTER_BLUE_MODEL_FAMILY


@dataclass(frozen=True)
class PosterBlueTrainingConfig:
    max_epochs: int = 48
    min_epochs: int = 8
    patience_epochs: int = 8
    min_improvement: float = 1e-4
    batch_size: int = 128
    learning_rate_init: float = 1e-3
    alpha: float = 5e-4
    class_balance_strategy: str = "deterministic_oversample_to_max_class"
    validation_source: str = "calibration_rows"
    early_stopping_metric: str = "multiclass_cross_entropy"


DEFAULT_POSTER_BLUE_ARCHITECTURE = PosterBlueArchitectureConfig()
DEFAULT_POSTER_BLUE_TRAINING_CONFIG = PosterBlueTrainingConfig()


def validate_poster_blue_feature_names(feature_names: list[str]) -> None:
    report = validate_blue_feature_names(feature_names, BLUE_FEATURE_POLICY_POSTER_DEFAULT)
    if not bool(report.get("passed", report.get("valid"))):
        raise ValueError(
            "Poster blue model received disallowed feature names. "
            f"details={report}"
        )


def poster_blue_output_formulation() -> dict[str, Any]:
    return {
        "class_names": list(CLASS_NAMES),
        "head": "softmax_over_benign_cyber_fault",
        "cyber_score": "p(cyber)",
        "fault_score": "p(fault)",
        "anomaly_score": "p(cyber)+p(fault)",
        "unsafe_risk_score": "1 - p(benign)",
    }


def poster_blue_architecture_report(
    feature_names: list[str],
    *,
    architecture: PosterBlueArchitectureConfig = DEFAULT_POSTER_BLUE_ARCHITECTURE,
) -> dict[str, Any]:
    return {
        "family": architecture.family,
        "feature_count": int(len(feature_names)),
        "feature_encoding": architecture.feature_encoding,
        "scaler": architecture.scaler,
        "backbone": {
            "type": "mlp",
            "hidden_layer_sizes": list(architecture.hidden_layer_sizes),
            "activation": architecture.activation,
        },
        "output_head": {
            "type": architecture.output_head,
            "class_names": list(CLASS_NAMES),
        },
    }


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for label in labels.astype(int).tolist():
        if 0 <= int(label) < len(CLASS_NAMES):
            counts[CLASS_NAMES[int(label)]] += 1
    return counts


def _align_probabilities(probabilities: np.ndarray, observed_labels: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], len(CLASS_NAMES)), dtype=float)
    for index, label in enumerate(observed_labels.astype(int).tolist()):
        if 0 <= label < len(CLASS_NAMES):
            aligned[:, label] = probabilities[:, index]
    return aligned


def _multiclass_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    chosen = clipped[np.arange(len(labels)), labels.astype(int)]
    return float(-np.mean(np.log(chosen)))


def _macro_f1_from_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> float:
    predictions = np.argmax(probabilities, axis=1)
    _, _, f1_scores, _ = precision_recall_fscore_support(
        labels.astype(int),
        predictions.astype(int),
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    return float(np.mean(f1_scores)) if len(f1_scores) else 0.0


def _balanced_epoch_indices(labels: np.ndarray, *, seed: int, epoch: int) -> np.ndarray:
    labels = labels.astype(int)
    rng = np.random.default_rng(seed * 1009 + epoch * 37 + 17)
    label_values = sorted({int(value) for value in labels.tolist()})
    if not label_values:
        return np.array([], dtype=int)
    by_label = {
        label_value: np.flatnonzero(labels == label_value)
        for label_value in label_values
    }
    max_count = max(len(indices) for indices in by_label.values())
    epoch_indices: list[np.ndarray] = []
    for label_value in label_values:
        indices = np.array(by_label[label_value], dtype=int)
        if len(indices) >= max_count:
            sampled = np.array(indices, copy=True)
        else:
            extra = rng.choice(indices, size=max_count - len(indices), replace=True)
            sampled = np.concatenate([indices, extra])
        rng.shuffle(sampled)
        epoch_indices.append(sampled)
    merged = np.concatenate(epoch_indices)
    rng.shuffle(merged)
    return merged.astype(int)


def fit_poster_blue_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    feature_names: list[str],
    seed: int,
    architecture: PosterBlueArchitectureConfig = DEFAULT_POSTER_BLUE_ARCHITECTURE,
    training_config: PosterBlueTrainingConfig = DEFAULT_POSTER_BLUE_TRAINING_CONFIG,
) -> dict[str, Any]:
    validate_poster_blue_feature_names(feature_names)
    if X_train.ndim != 2:
        raise ValueError("Poster blue training requires a 2D training matrix")
    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            "Poster blue feature name count must match training matrix width. "
            f"features={len(feature_names)} width={X_train.shape[1]}"
        )
    if len(X_train) <= 0 or len(y_train) <= 0:
        raise ValueError("Poster blue training requires non-empty training rows")
    observed_train_labels = sorted({int(value) for value in y_train.astype(int).tolist()})
    if observed_train_labels != [0, 1, 2]:
        raise ValueError(
            "Poster blue training requires all three classes in the training split. "
            f"observed={observed_train_labels}"
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(np.asarray(X_train, dtype=float))
    X_validation_scaled = scaler.transform(np.asarray(X_validation, dtype=float))
    mlp = MLPClassifier(
        hidden_layer_sizes=architecture.hidden_layer_sizes,
        activation=architecture.activation,
        alpha=training_config.alpha,
        batch_size=max(1, min(int(training_config.batch_size), int(len(y_train)))),
        learning_rate_init=training_config.learning_rate_init,
        max_iter=1,
        random_state=seed,
        shuffle=False,
        warm_start=True,
    )

    history: list[dict[str, Any]] = []
    best_state: MLPClassifier | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_macro_f1 = 0.0
    patience = 0
    epochs_completed = 0
    validation_enabled = len(X_validation_scaled) > 0 and len(y_validation) > 0
    if not validation_enabled:
        best_state = copy.deepcopy(mlp)

    for epoch in range(1, training_config.max_epochs + 1):
        if training_config.class_balance_strategy == "deterministic_oversample_to_max_class":
            epoch_indices = _balanced_epoch_indices(y_train, seed=seed, epoch=epoch)
        else:
            epoch_indices = np.arange(len(y_train), dtype=int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            mlp.fit(X_train_scaled[epoch_indices], y_train[epoch_indices])

        train_probabilities = _align_probabilities(mlp.predict_proba(X_train_scaled), mlp.classes_)
        train_loss = _multiclass_cross_entropy(y_train, train_probabilities)
        train_macro_f1 = _macro_f1_from_probabilities(y_train, train_probabilities)

        validation_loss = train_loss
        validation_macro_f1 = train_macro_f1
        if validation_enabled:
            validation_probabilities = _align_probabilities(
                mlp.predict_proba(X_validation_scaled),
                mlp.classes_,
            )
            validation_loss = _multiclass_cross_entropy(y_validation, validation_probabilities)
            validation_macro_f1 = _macro_f1_from_probabilities(y_validation, validation_probabilities)

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_macro_f1": float(train_macro_f1),
                "validation_loss": float(validation_loss),
                "validation_macro_f1": float(validation_macro_f1),
                "epoch_rows": int(len(epoch_indices)),
            }
        )
        epochs_completed = epoch

        improved = False
        if best_state is None:
            improved = True
        else:
            if validation_loss < (best_validation_loss - training_config.min_improvement):
                improved = True
            elif abs(validation_loss - best_validation_loss) <= training_config.min_improvement:
                improved = validation_macro_f1 > (best_validation_macro_f1 + 1e-6)

        if improved:
            best_state = copy.deepcopy(mlp)
            best_epoch = epoch
            best_validation_loss = float(validation_loss)
            best_validation_macro_f1 = float(validation_macro_f1)
            patience = 0
        else:
            patience += 1

        if validation_enabled and epoch >= training_config.min_epochs and patience >= training_config.patience_epochs:
            break

    if best_state is None:
        raise RuntimeError("Poster blue training did not produce a fitted model")

    pipeline = Pipeline([("scale", scaler), ("mlp", best_state)])
    final_validation_loss = best_validation_loss if validation_enabled else history[-1]["validation_loss"]
    final_validation_macro_f1 = best_validation_macro_f1 if validation_enabled else history[-1]["validation_macro_f1"]
    training_summary = {
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "train_class_counts": _class_counts(y_train),
        "validation_class_counts": _class_counts(y_validation),
        "epochs_completed": int(epochs_completed),
        "best_epoch": int(best_epoch if best_epoch > 0 else epochs_completed),
        "stopped_early": bool(validation_enabled and epochs_completed < training_config.max_epochs),
        "best_validation_cross_entropy": float(best_validation_loss),
        "best_validation_macro_f1": float(best_validation_macro_f1),
        "final_validation_cross_entropy": float(final_validation_loss),
        "final_validation_macro_f1": float(final_validation_macro_f1),
        "validation_source": training_config.validation_source,
        "early_stopping_metric": training_config.early_stopping_metric,
        "class_balance_strategy": training_config.class_balance_strategy,
    }
    return {
        "pipeline": pipeline,
        "architecture": poster_blue_architecture_report(feature_names, architecture=architecture),
        "output_formulation": poster_blue_output_formulation(),
        "training_config": asdict(training_config),
        "training_summary": training_summary,
        "training_history": history,
    }


def export_poster_blue_model_payload(
    pipeline: Any,
    feature_names: list[str],
    feature_tier: str,
    *,
    architecture: dict[str, Any],
    output_formulation: dict[str, Any],
    training_config: dict[str, Any],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    return export_sklearn_mlp(
        pipeline,
        feature_names,
        feature_tier,
        extra_fields={
            "architecture": dict(architecture),
            "output_formulation": dict(output_formulation),
            "training_config": dict(training_config),
            "training_summary": dict(training_summary),
        },
    )
