from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bg_pcyber
from runtime import (
    POSTER_BLUE_MODEL_ARTIFACT_NAME,
    POSTER_BLUE_RUNTIME_KIND,
    RUNTIME_BUNDLE_MANIFEST_NAME,
    RuntimeBundle,
    build_poster_blue_runtime_manifest,
    load_runtime_bundle,
    save_json,
)
from tools.train.blue_model import export_poster_blue_model_payload, fit_poster_blue_model
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


def build_saved_poster_runtime_bundle(model_dir: Path) -> tuple[dict[str, float], dict[str, object]]:
    X_train, y_train, X_validation, y_validation, feature_names = build_small_blue_training_matrices()
    result = fit_poster_blue_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_names=feature_names,
        seed=7,
    )
    payload = export_poster_blue_model_payload(
        result["pipeline"],
        feature_names,
        POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER,
        architecture=result["architecture"],
        output_formulation=result["output_formulation"],
        training_config=result["training_config"],
        training_summary=result["training_summary"],
    )
    manifest = build_poster_blue_runtime_manifest(
        training_path="poster_default_canonical",
        feature_tier=POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME, payload)
    save_json(model_dir / RUNTIME_BUNDLE_MANIFEST_NAME, manifest)
    scoring_row = {
        feature_name: float(X_validation[0][index])
        for index, feature_name in enumerate(feature_names)
    }
    return scoring_row, manifest


class BlueRuntimeTests(unittest.TestCase):
    def test_load_runtime_bundle_scores_saved_poster_model_without_legacy_sidecars(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            scoring_row, manifest = build_saved_poster_runtime_bundle(model_dir)

            bundle = load_runtime_bundle(model_dir)
            factory_bundle = RuntimeBundle(model_dir)

            self.assertEqual(getattr(bundle, "runtime_kind", ""), POSTER_BLUE_RUNTIME_KIND)
            self.assertEqual(getattr(factory_bundle, "runtime_kind", ""), POSTER_BLUE_RUNTIME_KIND)
            self.assertEqual(bundle.manifest["runtime_kind"], manifest["runtime_kind"])

            scored = bundle.score_row(scoring_row)
            self.assertEqual(scored["runtime_kind"], POSTER_BLUE_RUNTIME_KIND)
            self.assertIn("risk", scored)
            self.assertIn("unsafe_risk", scored)
            self.assertIn("panomaly", scored)
            self.assertNotIn("rules", scored)
            self.assertNotIn("novelty", scored)
            self.assertAlmostEqual(float(scored["risk"]), float(scored["unsafe_risk"]), places=9)
            self.assertAlmostEqual(float(scored["risk"]), float(scored["panomaly"]), places=9)
            self.assertAlmostEqual(
                float(scored["risk"]),
                float(scored["pcyber"]) + float(scored["pfault"]),
                places=9,
            )

    def test_bg_pcyber_uses_manifest_driven_poster_bundle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            scoring_row, _ = build_saved_poster_runtime_bundle(model_dir)
            expected_pcyber = float(load_runtime_bundle(model_dir).score_row(scoring_row)["pcyber"])

            with mock.patch.object(bg_pcyber, "DEFAULT_MODEL_DIR", model_dir):
                bg_pcyber._BUNDLE = None
                try:
                    observed_pcyber = float(bg_pcyber.pcyber_score(scoring_row))
                finally:
                    bg_pcyber._BUNDLE = None

            self.assertAlmostEqual(observed_pcyber, expected_pcyber, places=9)


if __name__ == "__main__":
    unittest.main()
