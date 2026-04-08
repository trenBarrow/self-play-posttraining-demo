#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import run_train_novelty


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Gaussian novelty from transaction replay on nominal generated traffic")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_train_novelty(Path(args.dataset).resolve(), Path(args.output).resolve())
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
