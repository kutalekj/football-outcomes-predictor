from __future__ import annotations

from pathlib import Path

import football_outcomes.config.fs_settings as sett
from football_outcomes.training.fs_training_utils import build_categorical_maps
from football_outcomes.training.train_mlp_rolling import train_rolling
from scripts.tools.thesis_epoch_experiment_utils import (
    build_fig54_v1_or_v2lite_cfg,
    build_fig54_v2_full_approx_cfg,
    collect_run_summary,
    prepare_clean_matches,
    write_summary_files,
)

OUT_ROOT = Path(sett.DATA_DIR) / "comparison" / "thesis_fig54_epoch_metrics"


def main() -> None:
    matches = prepare_clean_matches()
    cat_maps = build_categorical_maps(matches)

    runs = [
        (
            "v1_full",
            build_fig54_v1_or_v2lite_cfg(
                run_name="thesis_fig54_epoch_v1_full",
                model_version="v1",
            ),
        ),
        (
            "v2_lite_full",
            build_fig54_v1_or_v2lite_cfg(
                run_name="thesis_fig54_epoch_v2_lite_full",
                model_version="v2",
            ),
        ),
        (
            "v2_full_approx",
            build_fig54_v2_full_approx_cfg(
                run_name="thesis_fig54_epoch_v2_full_approx",
            ),
        ),
    ]

    summaries = []

    for label, cfg in runs:
        print("\n" + "=" * 80)
        print(f"[FIG 5.4 EPOCH METRICS] {label} -> {cfg.run_name}")
        print("=" * 80)

        train_rolling(matches, cat_maps, cfg)
        summaries.append(collect_run_summary(cfg.run_name, label))

    write_summary_files(
        OUT_ROOT,
        summaries,
        stem="thesis_fig54_epoch_metrics_results",
    )


if __name__ == "__main__":
    main()
