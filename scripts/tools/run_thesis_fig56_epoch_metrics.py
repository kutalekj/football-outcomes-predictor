from __future__ import annotations

import os
from pathlib import Path

import football_outcomes.config.fs_settings as sett
from football_outcomes.training.fs_training_utils import build_categorical_maps
from football_outcomes.training.train_mlp_rolling import train_rolling
from scripts.tools.thesis_epoch_experiment_utils import (
    build_fig56_v1_cfg,
    build_v1_model_with_optional_pretraining,
    collect_run_summary,
    prepare_clean_matches,
    write_summary_files,
)

OUT_ROOT = Path(sett.DATA_DIR) / "comparison" / "thesis_fig56_epoch_metrics"

DEFAULT_PRETRAINED_PATH = (
    Path(sett.DATA_DIR) / "tensorboard_logs" / "strength_pretrain_v1_u25_smallLR" / "pretrained_model.keras"
)


def main() -> None:
    matches = prepare_clean_matches()
    cat_maps = build_categorical_maps(matches)

    pretrained_path = Path(os.environ.get("PRETRAINED_V1_BRANCH_PATH", str(DEFAULT_PRETRAINED_PATH)))

    runs = [
        {
            "label": "scratch",
            "cfg": build_fig56_v1_cfg(
                run_name="thesis_fig56_epoch_v1_full_scratch_lr8e5",
                freeze_rounds=0,
            ),
            "use_pretraining": False,
        },
        {
            "label": "pretrained_init",
            "cfg": build_fig56_v1_cfg(
                run_name="thesis_fig56_epoch_v1_full_pretrained_init_lr8e5",
                freeze_rounds=0,
            ),
            "use_pretraining": True,
        },
        {
            "label": "pretrained_freeze3",
            "cfg": build_fig56_v1_cfg(
                run_name="thesis_fig56_epoch_v1_full_pretrained_init_freeze3_lr8e5",
                freeze_rounds=3,
            ),
            "use_pretraining": True,
        },
        {
            "label": "pretrained_freeze25",
            "cfg": build_fig56_v1_cfg(
                run_name="thesis_fig56_epoch_v1_full_pretrained_init_freeze25_lr8e5",
                freeze_rounds=25,
            ),
            "use_pretraining": True,
        },
    ]

    summaries = []

    for r in runs:
        label = r["label"]
        cfg = r["cfg"]

        print("\n" + "=" * 80)
        print(f"[FIG 5.6 EPOCH METRICS] {label} -> {cfg.run_name}")
        print("=" * 80)

        if r["use_pretraining"]:
            model = build_v1_model_with_optional_pretraining(
                matches=matches,
                cat_maps=cat_maps,
                cfg=cfg,
                pretrained_path=pretrained_path,
            )
            train_rolling(
                matches,
                cat_maps,
                cfg,
                model=model,
                pretrained_branch_version="v1",
            )
        else:
            train_rolling(matches, cat_maps, cfg)

        row = collect_run_summary(cfg.run_name, label)
        row["pretrained_path"] = str(pretrained_path) if r["use_pretraining"] else ""
        row["freeze_pretrained_branch_rounds"] = cfg.freeze_pretrained_branch_rounds
        summaries.append(row)

    write_summary_files(
        OUT_ROOT,
        summaries,
        stem="thesis_fig56_epoch_metrics_results",
    )


if __name__ == "__main__":
    main()
