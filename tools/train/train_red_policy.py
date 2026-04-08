#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train.red_policy_model import (
    DEFAULT_RED_POLICY_TRAINING_CONFIG,
    SUPPORTED_RED_POLICY_PROTOCOL_MODES,
    run_red_policy_warmstart_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the bounded red warm-start policy from the seeded cyber schedule generators"
    )
    parser.add_argument(
        "--protocol-mode",
        choices=SUPPORTED_RED_POLICY_PROTOCOL_MODES,
        default=DEFAULT_RED_POLICY_TRAINING_CONFIG.protocol_mode,
    )
    parser.add_argument(
        "--rows-per-protocol",
        type=int,
        default=DEFAULT_RED_POLICY_TRAINING_CONFIG.warm_start_rows_per_protocol,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-history-entries", type=int)
    parser.add_argument("--no-examples-dump", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/red_policy_latest")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    result = run_red_policy_warmstart_training(
        output_dir=output_dir,
        protocol_mode=args.protocol_mode,
        rows_per_protocol=args.rows_per_protocol,
        seed=args.seed,
        max_history_entries=args.max_history_entries,
        dump_examples=not args.no_examples_dump,
    )
    print(
        json.dumps(
            {
                "artifacts": result["report"]["artifacts"],
                "warm_start": result["report"]["warm_start"],
                "evaluation": result["report"]["evaluation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
