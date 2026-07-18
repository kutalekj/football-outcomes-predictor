from __future__ import annotations

from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def main_loss_and_metrics_for_mode(
    cfg,
):
    if cfg.mode == "binary_u25":
        return (
            "binary_crossentropy",
            [
                "accuracy",
                AUC(name="auc"),
            ],
        )

    if cfg.mode == "goals_dist":
        return (
            "sparse_categorical_crossentropy",
            ["accuracy"],
        )

    if cfg.mode == "goals_reg":
        return (
            "mae",
            ["mae"],
        )

    raise ValueError(f"Unknown mode: {cfg.mode}")


def auxiliary_loss_and_metrics_for_task(
    auxiliary_task: str,
):
    if auxiliary_task == "binary_u25":
        return (
            "binary_crossentropy",
            ["accuracy"],
        )

    if auxiliary_task == "goals_dist":
        return (
            "sparse_categorical_crossentropy",
            ["accuracy"],
        )

    if auxiliary_task == "goals_reg":
        return (
            "mae",
            ["mae"],
        )

    raise ValueError("Unknown aux_task: " f"{auxiliary_task}")


def compile_model_for_config(
    model: Model,
    cfg,
) -> None:
    """Compile an existing model after trainable flags change."""

    optimizer = Adam(learning_rate=cfg.learning_rate)

    if cfg.model_version == "v1":
        if cfg.mode == "binary_u25":
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            return

        if cfg.mode == "goals_dist":
            model.compile(
                optimizer=optimizer,
                loss=("sparse_categorical_" "crossentropy"),
                metrics=["accuracy"],
            )
            return

        if cfg.mode == "goals_reg":
            model.compile(
                optimizer=optimizer,
                loss="mae",
                metrics=["mae"],
            )
            return

        raise ValueError(f"Unknown mode: {cfg.mode}")

    if cfg.model_version == "v2":
        (
            main_loss,
            main_metrics,
        ) = main_loss_and_metrics_for_mode(cfg)

        if cfg.use_team_aux_head and cfg.aux_task is not None:
            (
                auxiliary_loss,
                auxiliary_metrics,
            ) = auxiliary_loss_and_metrics_for_task(cfg.aux_task)

            model.compile(
                optimizer=optimizer,
                loss={
                    "output_main": (main_loss),
                    "output_team_aux": (auxiliary_loss),
                },
                loss_weights={
                    "output_main": 1.0,
                    "output_team_aux": (cfg.aux_weight),
                },
                metrics={
                    "output_main": (main_metrics),
                    "output_team_aux": (auxiliary_metrics),
                },
            )
            return

        model.compile(
            optimizer=optimizer,
            loss={"output_main": main_loss},
            metrics={"output_main": main_metrics},
        )
        return

    raise ValueError("Unknown model_version: " f"{cfg.model_version}")
