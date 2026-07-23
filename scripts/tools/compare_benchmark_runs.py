from __future__ import annotations

import argparse
import sys
from pathlib import Path

from football_outcomes.experiments.comparison import (
    ComparisonConfig,
    run_benchmark_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Compare a full neural benchmark with its manifest-backed " "common-fold baseline run.")
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        help="Path to the validated frozen snapshot.",
    )
    parser.add_argument(
        "--neural-run",
        required=True,
        type=Path,
        help="Completed full neural benchmark run directory.",
    )
    parser.add_argument(
        "--baseline-run",
        required=True,
        type=Path,
        help="Completed common-fold baseline run directory.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Parent directory for the comparison run directory.",
    )
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--neural-model-name",
        default="v2-benchmark",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing comparison directory with the same run ID.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    config = ComparisonConfig(
        calibration_bins=arguments.calibration_bins,
        neural_model_name=arguments.neural_model_name,
    )
    run_directory = run_benchmark_comparison(
        repository_root=Path.cwd(),
        snapshot_path=arguments.snapshot,
        neural_run=arguments.neural_run,
        baseline_run=arguments.baseline_run,
        output_root=arguments.output_root,
        config=config,
        command=tuple(sys.argv),
        overwrite=arguments.overwrite,
    )
    print("benchmark comparison: PASS")
    print(f"run directory: {run_directory}")
    print(f"manifest: {run_directory / 'manifest.json'}")
    print(f"comparison: {run_directory / 'comparison.json'}")
    print(f"summary: {run_directory / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
