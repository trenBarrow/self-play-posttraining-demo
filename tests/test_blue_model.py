from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import SimpleMLP
from tools.train.blue_model import (
    DEFAULT_POSTER_BLUE_TRAINING_CONFIG,
    POSTER_BLUE_MODEL_FAMILY,
    export_poster_blue_model_payload,
    fit_poster_blue_model,
)
from tools.train.poster_default import POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER, poster_default_model_feature_names


def build_small_blue_training_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_names = poster_default_model_feature_names()[:4]
    X_train = np.array(
        [
            [0.0, 0.0, 0.1, 0.2],
            [0.1, 0.1, 0.0, 0.1],
            [0.2, 0.0, 0.2, 0.0],
            [2.0, 2.1, 1.8, 2.2],
            [2.2, 1.9, 2.1, 1.8],
            [1.8, 2.0, 2.2, 2.1],
            [4.0, 4.1, 4.2, 4.0],
            [4.1, 3.9, 4.0, 4.2],
            [3.9, 4.2, 3.8, 4.1],
        ],
        dtype=float,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
    X_validation = np.array(
        [
            [0.15, 0.05, 0.1, 0.15],
            [2.05, 2.0, 1.95, 2.05],
            [4.05, 4.0, 4.1, 4.05],
            [0.05, 0.1, 0.05, 0.05],
            [2.15, 2.05, 2.1, 2.0],
            [4.1, 4.15, 4.0, 4.1],
        ],
        dtype=float,
    )
    y_validation = np.array([0, 1, 2, 0, 1, 2], dtype=int)
    return X_train, y_train, X_validation, y_validation, feature_names


class BlueModelTests(unittest.TestCase):
    def test_fit_poster_blue_model_records_architecture_and_training_metadata(self) -> None:
        X_train, y_train, X_validation, y_validation, feature_names = build_small_blue_training_matrices()

        result = fit_poster_blue_model(
            X_train,
            y_train,
            X_validation,
            y_validation,
            feature_names=feature_names,
            seed=7,
        )

        self.assertEqual(result["architecture"]["family"], POSTER_BLUE_MODEL_FAMILY)
        self.assertEqual(result["architecture"]["feature_count"], len(feature_names))
        self.assertEqual(
            result["training_config"]["class_balance_strategy"],
            DEFAULT_POSTER_BLUE_TRAINING_CONFIG.class_balance_strategy,
        )
        self.assertGreaterEqual(int(result["training_summary"]["best_epoch"]), 1)
        self.assertEqual(result["output_formulation"]["unsafe_risk_score"], "1 - p(benign)")

        payload = export_poster_blue_model_payload(
            result["pipeline"],
            feature_names,
            POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER,
            architecture=result["architecture"],
            output_formulation=result["output_formulation"],
            training_config=result["training_config"],
            training_summary=result["training_summary"],
        )

        self.assertEqual(payload["architecture"]["family"], POSTER_BLUE_MODEL_FAMILY)
        self.assertEqual(payload["feature_names"], feature_names)
        self.assertEqual(payload["feature_tier"], POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER)
        self.assertEqual(payload["training_summary"]["best_epoch"], result["training_summary"]["best_epoch"])

        runtime_model = SimpleMLP.from_payload(payload)
        probabilities = runtime_model.predict_proba_one(
            {
                feature_name: float(X_validation[0][index])
                for index, feature_name in enumerate(feature_names)
            }
        )
        self.assertEqual(len(probabilities), 3)

    def test_fit_poster_blue_model_rejects_forbidden_feature_names(self) -> None:
        X_train, y_train, X_validation, y_validation, feature_names = build_small_blue_training_matrices()
        invalid_feature_names = ["service_id", *feature_names[1:]]

        with self.assertRaises(ValueError):
            fit_poster_blue_model(
                X_train,
                y_train,
                X_validation,
                y_validation,
                feature_names=invalid_feature_names,
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
