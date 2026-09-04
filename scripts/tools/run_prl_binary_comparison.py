from __future__ import annotations

import argparse
from pathlib import Path

from football_outcomes.experiments.publication_binary import (
    BinaryEstimatorConfig,
    PublicationBinaryConfig,
    run_publication_binary_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run the PRL binary publication comparison on common chronological folds.")
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-rounds", type=int, default=25)
    parser.add_argument("--fold-count", type=int, default=None)
    parser.add_argument("--start-fold-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--minimum-support", type=int, default=20)
    parser.add_argument("--neutral-value", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--rf-estimators", type=int, default=120)
    parser.add_argument("--rf-max-depth", type=int, default=12)
    parser.add_argument("--xgb-estimators", type=int, default=160)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--mlp-max-iter", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimator_config = BinaryEstimatorConfig(
        seed=args.seed,
        random_forest_estimators=args.rf_estimators,
        random_forest_max_depth=args.rf_max_depth,
        xgboost_estimators=args.xgb_estimators,
        xgboost_max_depth=args.xgb_max_depth,
        xgboost_learning_rate=args.xgb_learning_rate,
        mlp_max_iter=args.mlp_max_iter,
    )
    config = PublicationBinaryConfig(
        window_rounds=args.window_rounds,
        fold_count=args.fold_count,
        start_fold_offset=args.start_fold_offset,
        seed=args.seed,
        minimum_group_support=args.minimum_support,
        neutral_value=args.neutral_value,
        estimators=estimator_config,
    )
    run_directory = run_publication_binary_experiment(
        snapshot_path=args.snapshot,
        output_root=args.output_root,
        config=config,
        overwrite=args.overwrite,
    )
    print(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
