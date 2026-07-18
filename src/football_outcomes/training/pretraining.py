from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    TensorBoard,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import (
    FSMatch,
)
from football_outcomes.datasets.arrays import (
    build_strength_only_arrays_for_matches,
)
from football_outcomes.datasets.rounds import (
    distribute_matches_into_rounds,
    summarize_rounds,
)
from football_outcomes.evaluation.metrics import (
    binary_summary,
)
from football_outcomes.evaluation.persistence import (
    write_json,
    write_records_csv,
)
from football_outcomes.evaluation.plots import (
    save_pretrain_round_plot,
)
from football_outcomes.modeling.strength_pretraining import (
    build_strength_pretrain_model,
)
from football_outcomes.training.configs import (
    StrengthPretrainConfig,
)
from football_outcomes.training.runtime import (
    set_global_seed,
)


def train_strength_pretrain_rolling(
    matches_sorted: List[FSMatch],
    cfg: StrengthPretrainConfig,
) -> Model:
    rounds = distribute_matches_into_rounds(matches_sorted)
    round_info = summarize_rounds(rounds)
    print(f"[pretrain-rounds] {round_info}")

    if cfg.seed is not None:
        set_global_seed(cfg.seed)
        print("[pretrain-seed] " f"Using seed={cfg.seed}")

    model = build_strength_pretrain_model(cfg)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or ("strength_pretrain_" f"{cfg.branch_version}_" f"{cfg.mode}_{run_stamp}")

    log_root = Path(sett.DATA_DIR) / "tensorboard_logs"
    log_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_dir = str(log_root / run_name)

    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )
    tensorboard_writer = tf.summary.create_file_writer(log_dir)

    print("[pretrain-tensorboard] " f"logging to {log_dir}")

    round_records = []
    oos_rows = []

    for round_index in range(
        cfg.window_rounds,
        len(rounds),
    ):
        train_matches = [
            match for round_matches in rounds[round_index - cfg.window_rounds : round_index] for match in round_matches
        ]
        validation_matches = rounds[round_index]

        (
            strength_train,
            home_positions_train,
            away_positions_train,
            y_train,
        ) = build_strength_only_arrays_for_matches(
            train_matches,
            cfg.mode,
            cfg.max_goals_class,
        )

        (
            strength_validation,
            home_positions_validation,
            away_positions_validation,
            y_validation,
        ) = build_strength_only_arrays_for_matches(
            validation_matches,
            cfg.mode,
            cfg.max_goals_class,
        )

        print(
            f"[pretrain] round "
            f"{round_index + 1}/"
            f"{len(rounds)} "
            f"train={len(train_matches)} "
            "val="
            f"{len(validation_matches)} "
            "branch="
            f"{cfg.branch_version}"
        )

        early_stopping = EarlyStopping(
            patience=(cfg.early_stopping_patience),
            min_delta=(cfg.early_stopping_min_delta),
            restore_best_weights=True,
            monitor="val_loss",
            mode="min",
        )

        model.fit(
            [
                strength_train,
                home_positions_train,
                away_positions_train,
            ],
            y_train,
            validation_data=(
                [
                    strength_validation,
                    home_positions_validation,
                    away_positions_validation,
                ],
                y_validation,
            ),
            epochs=cfg.epochs_per_step,
            batch_size=cfg.batch_size,
            callbacks=[
                early_stopping,
                tensorboard,
            ],
            verbose=1,
        )

        validation_metrics = model.evaluate(
            [
                strength_validation,
                home_positions_validation,
                away_positions_validation,
            ],
            y_validation,
            verbose=0,
            return_dict=True,
        )
        validation_probability = (
            model.predict(
                [
                    strength_validation,
                    home_positions_validation,
                    away_positions_validation,
                ],
                verbose=0,
            )
            .ravel()
            .astype(np.float32)
        )

        auc_metric = AUC(curve="ROC")
        auc_metric.update_state(
            y_validation.astype(np.float32),
            validation_probability,
        )
        validation_auc = float(auc_metric.result().numpy())
        validation_brier = float(np.mean((validation_probability - y_validation.astype(np.float32)) ** 2))
        validation_accuracy = float(
            np.mean((validation_probability >= 0.5).astype(np.float32) == y_validation.astype(np.float32))
        )
        validation_loss = float(
            validation_metrics.get(
                "loss",
                np.nan,
            )
        )

        round_step = int(round_index + 1)

        round_records.append(
            {
                "round_idx": round_step,
                "train_size": len(train_matches),
                "val_size": len(validation_matches),
                "positive_rate_val": (float(np.mean(y_validation))),
                "val_loss": (validation_loss),
                "val_accuracy": (validation_accuracy),
                "val_auc": (validation_auc),
                "val_brier": (validation_brier),
                "branch_version": (cfg.branch_version),
                "representation": (cfg.representation),
                "use_strength_masks": (bool(cfg.use_strength_masks)),
                ("use_position_" "embedding"): bool(cfg.use_position_embedding),
            }
        )

        for (
            match,
            target,
            probability,
        ) in zip(
            validation_matches,
            y_validation,
            validation_probability,
        ):
            oos_rows.append(
                {
                    "round_idx": (round_step),
                    "match_id": match.id,
                    "season": (match.season),
                    "competition": (match.comp_name),
                    "y_true": float(target),
                    "y_prob_under25": (float(probability)),
                    "branch_version": (cfg.branch_version),
                    "representation": (cfg.representation),
                }
            )

        with tensorboard_writer.as_default():
            tf.summary.scalar(
                "round/val_loss",
                validation_loss,
                step=round_step,
            )
            tf.summary.scalar(
                "round/val_accuracy",
                validation_accuracy,
                step=round_step,
            )
            tf.summary.scalar(
                "round/val_auc",
                validation_auc,
                step=round_step,
            )
            tf.summary.scalar(
                "round/val_brier",
                validation_brier,
                step=round_step,
            )
            tf.summary.scalar(
                "round/val_size",
                len(validation_matches),
                step=round_step,
            )
            tf.summary.scalar(
                ("round/" "positive_rate_val"),
                float(np.mean(y_validation)),
                step=round_step,
            )
            tensorboard_writer.flush()

    round_metrics_path = Path(log_dir) / "round_metrics.csv"
    write_records_csv(
        round_metrics_path,
        round_records,
    )

    if cfg.save_oos_predictions:
        prediction_path = Path(log_dir) / "oos_predictions.csv"
        write_records_csv(
            prediction_path,
            oos_rows,
        )

    summary = {
        "run_name": run_name,
        "branch_version": (cfg.branch_version),
        "mode": cfg.mode,
        "representation": (cfg.representation),
        "use_strength_masks": bool(cfg.use_strength_masks),
        "use_position_embedding": bool(cfg.use_position_embedding),
        "round_stats": round_info,
    }

    if oos_rows:
        y_true = np.asarray(
            [row["y_true"] for row in oos_rows],
            dtype=np.float32,
        )
        y_probability = np.asarray(
            [row["y_prob_under25"] for row in oos_rows],
            dtype=np.float32,
        )

        summary.update(
            binary_summary(
                y_true,
                y_probability,
            )
        )

    summary_path = Path(log_dir) / "summary.json"
    write_json(
        summary_path,
        summary,
    )

    config_path = Path(log_dir) / "pretrain_config.json"
    write_json(
        config_path,
        asdict(cfg),
    )

    save_pretrain_round_plot(
        log_dir=log_dir,
        round_records=round_records,
        title=("Structured branch " "pretraining " f"({cfg.branch_version})"),
    )

    model_path = Path(log_dir) / "pretrained_model.keras"
    model.save(model_path)

    print(f"[pretrain-summary] {summary}")
    print("[pretrain] model saved to " f"{model_path}")

    return model
