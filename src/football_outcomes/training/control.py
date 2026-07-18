from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def learning_rate_for_round(
    cfg,
    round_offset: int,
    total_rounds: int,
) -> float:
    base = float(cfg.learning_rate)

    if cfg.lr_schedule == "constant":
        return base

    if cfg.lr_schedule == "exponential":
        learning_rate = base * (float(cfg.lr_decay_rate) ** int(round_offset))

        return max(
            float(cfg.min_learning_rate),
            float(learning_rate),
        )

    if cfg.lr_schedule == "cosine":
        if total_rounds <= 1:
            return base

        progress = min(
            1.0,
            max(
                0.0,
                round_offset / float(total_rounds - 1),
            ),
        )

        learning_rate = cfg.min_learning_rate + 0.5 * (base - cfg.min_learning_rate) * (1.0 + np.cos(np.pi * progress))

        return float(learning_rate)

    raise ValueError("Unknown lr_schedule: " f"{cfg.lr_schedule}")


def set_optimizer_learning_rate(
    model: Model,
    learning_rate: float,
) -> None:
    try:
        model.optimizer.learning_rate.assign(float(learning_rate))
    except Exception:
        tf.keras.backend.set_value(
            model.optimizer.learning_rate,
            float(learning_rate),
        )


def get_strength_branch_layer_names(
    branch_version: str,
) -> list[str]:
    if branch_version == "v1":
        return [
            "position_embedding",
            "strength_dense_1",
            "strength_dense_2",
            "strength_projection",
        ]

    if branch_version == "v2":
        return [
            "position_embedding",
            "home_row_dense_1",
            "home_row_dense_2",
            "home_role_post_dense_1",
            "home_team_repr",
            "away_row_dense_1",
            "away_row_dense_2",
            "away_role_post_dense_1",
            "away_team_repr",
            "team_branch_proj",
        ]

    raise ValueError("Unknown branch_version: " f"{branch_version}")


def transfer_pretrained_strength_branch_weights(
    pretrained_model: Model,
    full_model: Model,
    branch_version: str,
) -> None:
    """Copy matching strength-branch layers by name."""

    layer_names = get_strength_branch_layer_names(branch_version)

    for name in layer_names:
        try:
            source_layer = pretrained_model.get_layer(name)
            destination_layer = full_model.get_layer(name)
        except ValueError:
            print("[transfer] skipping missing " f"layer: {name}")
            continue

        destination_layer.set_weights(source_layer.get_weights())
        print(f"[transfer] copied layer: {name}")

    print("[transfer] copied pretrained " f"{branch_version} branch weights " "into full model")


def set_layers_trainable(
    model: Model,
    layer_names: Sequence[str],
    trainable: bool,
) -> None:
    for name in layer_names:
        try:
            model.get_layer(name).trainable = trainable
        except ValueError:
            continue
