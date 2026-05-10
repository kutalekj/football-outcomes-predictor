from __future__ import annotations

from pathlib import Path

import football_outcomes.config.fs_settings as sett
from football_outcomes.training.fs_training_utils import build_categorical_maps
from football_outcomes.training.train_mlp_rolling import train_rolling
from scripts.tools.thesis_epoch_experiment_utils import (
    build_base_v1_cfg,
    collect_run_summary,
    prepare_clean_matches,
    write_summary_files,
)

OUT_ROOT = Path(sett.DATA_DIR) / "comparison" / "thesis_fig52_epoch_metrics"


def main() -> None:
    matches = prepare_clean_matches()
    cat_maps = build_categorical_maps(matches)

    runs = []

    cfg_full = build_base_v1_cfg("thesis_fig52_epoch_v1_full")
    runs.append(("full", cfg_full))

    cfg_no_strength = build_base_v1_cfg("thesis_fig52_epoch_v1_no_strength")
    cfg_no_strength.use_team_strength = False
    cfg_no_strength.use_position_embedding = False
    runs.append(("no_strength", cfg_no_strength))

    cfg_no_positions = build_base_v1_cfg("thesis_fig52_epoch_v1_no_positions")
    cfg_no_positions.use_team_strength = True
    cfg_no_positions.use_position_embedding = False
    runs.append(("no_positions", cfg_no_positions))

    summaries = []

    for label, cfg in runs:
        print("\n" + "=" * 80)
        print(f"[FIG 5.2 EPOCH METRICS] {label} -> {cfg.run_name}")
        print("=" * 80)

        train_rolling(matches, cat_maps, cfg)
        summaries.append(collect_run_summary(cfg.run_name, label))

    write_summary_files(
        OUT_ROOT,
        summaries,
        stem="thesis_fig52_epoch_metrics_results",
    )


if __name__ == "__main__":
    main()
