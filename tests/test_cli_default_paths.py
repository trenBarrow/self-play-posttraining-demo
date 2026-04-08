from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as poster_main
from tools.train.poster_default import POSTER_DEFAULT_TRAINING_PATH_NAME


class CliDefaultPathTests(unittest.TestCase):
    def test_parser_help_marks_default_and_legacy_modes_explicitly(self) -> None:
        help_text = poster_main.build_parser().format_help()
        self.assertIn("poster-default canonical neural path", help_text)
        self.assertIn("comparison-only", help_text)
        self.assertIn("train-legacy", help_text)
        self.assertIn("run-legacy", help_text)

    def test_train_command_dispatches_poster_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "dataset.jsonl"
            output_dir = tmp_path / "out"
            fake_report = {"metrics": {}}
            argv = [
                "main.py",
                "train",
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("main.run_training", return_value=fake_report) as run_training_mock:
                    with mock.patch("builtins.print"):
                        poster_main.main()
            self.assertEqual(
                run_training_mock.call_args.kwargs["training_path_name"],
                POSTER_DEFAULT_TRAINING_PATH_NAME,
            )

    def test_train_legacy_command_dispatches_comparison_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "dataset.jsonl"
            output_dir = tmp_path / "out"
            fake_report = {"metrics": {}}
            argv = [
                "main.py",
                "train-legacy",
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("main.assert_legacy_training_dataset_supported") as guard_mock:
                    with mock.patch("main.run_training", return_value=fake_report) as run_training_mock:
                        with mock.patch("builtins.print"):
                            poster_main.main()
            guard_mock.assert_called_once_with(dataset_path)
            self.assertEqual(
                run_training_mock.call_args.kwargs["training_path_name"],
                poster_main.TRAINING_PATH_LEGACY_FPRIME_BASELINE,
            )

    def test_run_command_dispatches_poster_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = tmp_path / "out"
            dataset_path = output_dir / "data" / "dataset.jsonl"
            fake_report = {"metrics": {}}
            argv = [
                "main.py",
                "run",
                "--rows",
                "24",
                "--output-dir",
                str(output_dir),
                "--no-plots",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("main.run_generate", return_value=dataset_path) as generate_mock:
                    with mock.patch("main.run_training", return_value=fake_report) as run_training_mock:
                        with mock.patch("builtins.print"):
                            poster_main.main()
            generate_mock.assert_called_once()
            self.assertEqual(
                run_training_mock.call_args.kwargs["training_path_name"],
                POSTER_DEFAULT_TRAINING_PATH_NAME,
            )

    def test_run_legacy_command_dispatches_comparison_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = tmp_path / "out"
            dataset_path = output_dir / "data" / "dataset.jsonl"
            fake_report = {"metrics": {}}
            argv = [
                "main.py",
                "run-legacy",
                "--rows",
                "24",
                "--output-dir",
                str(output_dir),
                "--no-plots",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("main.assert_legacy_generation_protocol_mode") as generation_guard_mock:
                    with mock.patch("main.run_generate", return_value=dataset_path) as generate_mock:
                        with mock.patch("main.assert_legacy_training_dataset_supported") as training_guard_mock:
                            with mock.patch("main.run_training", return_value=fake_report) as run_training_mock:
                                with mock.patch("builtins.print"):
                                    poster_main.main()
            generation_guard_mock.assert_called_once_with("fprime")
            generate_mock.assert_called_once()
            training_guard_mock.assert_called_once_with(dataset_path)
            self.assertEqual(
                run_training_mock.call_args.kwargs["training_path_name"],
                poster_main.TRAINING_PATH_LEGACY_FPRIME_BASELINE,
            )

    def test_train_pipeline_script_dry_run_separates_default_and_legacy_modes(self) -> None:
        script_path = REPO_ROOT / "scripts" / "fprime_real" / "train_pipeline.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            default_proc = subprocess.run(
                ["bash", str(script_path), "--dry-run", "--output-dir", str(Path(tmpdir) / "poster")],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(default_proc.returncode, 0, msg=default_proc.stderr + default_proc.stdout)
            self.assertIn("mode: poster_default_headline", default_proc.stdout)
            self.assertIn("comparison_only: false", default_proc.stdout)
            self.assertIn("run_subcommand: run", default_proc.stdout)

            legacy_proc = subprocess.run(
                ["bash", str(script_path), "--dry-run", "--legacy", "--output-dir", str(Path(tmpdir) / "legacy")],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(legacy_proc.returncode, 0, msg=legacy_proc.stderr + legacy_proc.stdout)
            self.assertIn("mode: legacy_comparison_only", legacy_proc.stdout)
            self.assertIn("comparison_only: true", legacy_proc.stdout)
            self.assertIn("run_subcommand: run-legacy", legacy_proc.stdout)


if __name__ == "__main__":
    unittest.main()
