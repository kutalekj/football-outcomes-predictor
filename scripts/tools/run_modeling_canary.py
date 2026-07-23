from __future__ import annotations

import argparse
import sys
from pathlib import Path

from football_outcomes.experiments.canary import (
    CanaryConfig,
    run_modeling_canary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run the Step 8 chronological neural-model canary against the " "validated frozen snapshot.")
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to the frozen full snapshot.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Parent directory for the manifest-backed run directory.",
    )
    parser.add_argument("--window-rounds", type=int, default=25)
    parser.add_argument("--fold-count", type=int, default=2)
    parser.add_argument("--start-fold-offset", type=int, default=0)
    parser.add_argument("--epochs-per-fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--minimum-support", type=int, default=20)
    parser.add_argument("--neutral-value", type=float, default=50.0)
    parser.add_argument(
        "--model-version",
        choices=("v1", "v2"),
        default="v2",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing run directory with the same deterministic ID.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()

    config = CanaryConfig(
        window_rounds=arguments.window_rounds,
        fold_count=arguments.fold_count,
        start_fold_offset=arguments.start_fold_offset,
        epochs_per_fold=arguments.epochs_per_fold,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        minimum_group_support=arguments.minimum_support,
        neutral_value=arguments.neutral_value,
        model_version=arguments.model_version,
    )

    run_directory = run_modeling_canary(
        repository_root=Path.cwd(),
        snapshot_path=arguments.snapshot,
        output_root=arguments.output_root,
        config=config,
        command=tuple(sys.argv),
        overwrite=arguments.overwrite,
    )

    print("modeling canary: PASS")
    print(f"run directory: {run_directory}")
    print(f"manifest: {run_directory / 'manifest.json'}")
    print(f"predictions: {run_directory / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
