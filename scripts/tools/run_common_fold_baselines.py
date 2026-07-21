from __future__ import annotations

import argparse
import sys
from pathlib import Path

from football_outcomes.experiments.baselines import (
    BaselineConfig,
    run_common_fold_baselines,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run prevalence, majority-class, and logistic-regression "
            "baselines on the exact validation rows of a manifest-backed "
            "neural reference run."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to the validated frozen snapshot.",
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="Manifest-backed canary or benchmark run directory.",
    )
    parser.add_argument(
        "--reference-model-name",
        default=None,
        help="Required only when the reference predictions contain multiple models.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Parent directory for the baseline run directory.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing run directory with the same deterministic ID.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    config = BaselineConfig(
        seed=arguments.seed,
        logistic_c=arguments.logistic_c,
        logistic_max_iter=arguments.logistic_max_iter,
    )
    run_directory = run_common_fold_baselines(
        repository_root=Path.cwd(),
        snapshot_path=arguments.snapshot,
        reference_run_directory=arguments.reference_run,
        output_root=arguments.output_root,
        config=config,
        command=tuple(sys.argv),
        reference_model_name=arguments.reference_model_name,
        overwrite=arguments.overwrite,
    )

    print("common-fold baselines: PASS")
    print(f"run directory: {run_directory}")
    print(f"manifest: {run_directory / 'manifest.json'}")
    print(f"predictions: {run_directory / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
