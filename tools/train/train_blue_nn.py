#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import DEFAULT_OUTPUT_DIR, resolve_dataset_path, run_training
from tools.train.poster_default import POSTER_DEFAULT_TRAINING_PATH_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the poster-default blue neural detector")
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--blue-feature-policy")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    dataset_path = resolve_dataset_path(output_dir, args.dataset)
    report = run_training(
        dataset_path=dataset_path,
        output_dir=output_dir,
        seed=args.seed,
        make_plots=not args.no_plots,
        blue_feature_policy_name=args.blue_feature_policy,
        training_path_name=POSTER_DEFAULT_TRAINING_PATH_NAME,
    )
    print(
        json.dumps(
            {
                "blue_model": report["blue_model"],
                "metrics": report["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
