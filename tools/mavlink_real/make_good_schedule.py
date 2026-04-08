#!/usr/bin/env python3
"""Build a nominal MAVLink schedule for trusted operator traffic."""

from __future__ import annotations

import argparse
from pathlib import Path

from schedule_profiles import build_benign_rows, write_schedule_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--class-name", default="benign")
    parser.add_argument("--attack-family", default="none")
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--episode-span", type=int, default=18)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = build_benign_rows(
        target_rows=args.target_rows,
        seed=args.seed,
        label=args.label,
        class_name=args.class_name,
        attack_family=args.attack_family,
        episode_offset=args.episode_offset,
        episode_span=args.episode_span,
    )
    write_schedule_csv(Path(args.out), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
