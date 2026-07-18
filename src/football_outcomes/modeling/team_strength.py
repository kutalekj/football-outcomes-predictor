from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Lambda,
)

from football_outcomes.config import fs_settings as sett


def abs_diff(
    first,
    second,
    name: str,
):
    return Lambda(
        lambda tensors: tf.abs(tensors[0] - tensors[1]),
        name=name,
    )([first, second])


def vec_diff(
    first,
    second,
    name: str,
):
    return Lambda(
        lambda tensors: (tensors[0] - tensors[1]),
        name=name,
    )([first, second])


def safe_zero_vec(
    input_tensor,
    width: int,
    name: str,
):
    return Lambda(
        lambda tensor, size=int(width): (
            tf.zeros(
                (
                    tf.shape(tensor)[0],
                    size,
                ),
                dtype=tf.float32,
            )
        ),
        name=name,
    )(input_tensor)


def safe_zero_vec_from_inputs(
    inputs,
    width: int,
    name: str,
):
    return Lambda(
        lambda tensors, size=int(width): (
            tf.zeros(
                (
                    tf.shape(tensors[0])[0],
                    size,
                ),
                dtype=tf.float32,
            )
        ),
        name=name,
    )(inputs)


def split_strength_tensor(
    strength_tensor,
):
    home_values = Lambda(
        lambda tensor: tensor[:, 0],
        name="home_strength_values",
    )(strength_tensor)
    home_mask = Lambda(
        lambda tensor: tensor[:, 1],
        name="home_strength_mask",
    )(strength_tensor)
    away_values = Lambda(
        lambda tensor: tensor[:, 2],
        name="away_strength_values",
    )(strength_tensor)
    away_mask = Lambda(
        lambda tensor: tensor[:, 3],
        name="away_strength_mask",
    )(strength_tensor)

    return (
        home_values,
        home_mask,
        away_values,
        away_mask,
    )


def _row_valid_mask(
    mask_tensor,
    prefix: str,
):
    return Lambda(
        lambda tensor: tf.cast(
            tf.reduce_max(
                tensor,
                axis=-1,
                keepdims=True,
            )
            > 0.0,
            tf.float32,
        ),
        name=f"{prefix}_row_valid_mask",
    )(mask_tensor)


def _role_average_pool(
    encoded_rows,
    position_ids,
    row_valid_mask,
    role_index: int,
    prefix: str,
):
    role_mask = Lambda(
        lambda positions, role=int(role_index): tf.cast(
            tf.equal(
                positions,
                role,
            ),
            tf.float32,
        )[..., None],
        name=(f"{prefix}_role" f"{role_index}_mask"),
    )(position_ids)

    combined_mask = Lambda(
        lambda tensors: (tensors[0] * tensors[1]),
        name=(f"{prefix}_role" f"{role_index}_combined_mask"),
    )([role_mask, row_valid_mask])

    masked_sum = Lambda(
        lambda tensors: tf.reduce_sum(
            tensors[0] * tensors[1],
            axis=1,
        ),
        name=(f"{prefix}_role" f"{role_index}_sum"),
    )([encoded_rows, combined_mask])

    denominator = Lambda(
        lambda tensor: tf.maximum(
            tf.reduce_sum(
                tensor,
                axis=1,
            ),
            1e-6,
        ),
        name=(f"{prefix}_role" f"{role_index}_denom"),
    )(combined_mask)

    return Lambda(
        lambda tensors: (tensors[0] / tensors[1]),
        name=(f"{prefix}_role" f"{role_index}_avg"),
    )([masked_sum, denominator])


def build_team_repr_v2(
    team_values,
    team_mask,
    team_position_ids,
    position_embedding_layer,
    cfg,
    prefix: str,
):
    """Build one role-aware team representation."""

    row_hidden = int(cfg.player_row_hidden_dim)
    role_hidden = int(cfg.role_post_hidden_dim)
    output_width = int(cfg.strength_emb_dim)

    if position_embedding_layer is not None:
        team_position_embedding = position_embedding_layer(team_position_ids)
    else:
        position_width = int(cfg.position_emb_dim)
        team_position_embedding = Lambda(
            lambda positions, width=(position_width): tf.zeros(
                (
                    tf.shape(positions)[0],
                    11,
                    width,
                ),
                dtype=tf.float32,
            ),
            name=f"{prefix}_position_zero",
        )(team_position_ids)

    team_input = Concatenate(
        axis=-1,
        name=f"{prefix}_strength_concat",
    )(
        [
            team_values,
            team_mask,
            team_position_embedding,
        ]
    )

    encoded_rows = Dense(
        row_hidden,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name=f"{prefix}_row_dense_1",
    )(team_input)

    encoded_rows = Dense(
        row_hidden,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name=f"{prefix}_row_dense_2",
    )(encoded_rows)

    row_valid = _row_valid_mask(
        team_mask,
        prefix,
    )

    if cfg.use_position_embedding:
        goalkeeper_index = int(sett.FS_PLAYER_POSITION_TO_IDX["Goalkeeper"])
        defender_index = int(sett.FS_PLAYER_POSITION_TO_IDX["Defender"])
        midfielder_index = int(sett.FS_PLAYER_POSITION_TO_IDX["Midfielder"])
        forward_index = int(sett.FS_PLAYER_POSITION_TO_IDX["Forward"])

        goalkeeper_pool = _role_average_pool(
            encoded_rows,
            team_position_ids,
            row_valid,
            goalkeeper_index,
            prefix,
        )
        defender_pool = _role_average_pool(
            encoded_rows,
            team_position_ids,
            row_valid,
            defender_index,
            prefix,
        )
        midfielder_pool = _role_average_pool(
            encoded_rows,
            team_position_ids,
            row_valid,
            midfielder_index,
            prefix,
        )
        forward_pool = _role_average_pool(
            encoded_rows,
            team_position_ids,
            row_valid,
            forward_index,
            prefix,
        )

        pooled_roles = Concatenate(
            name=f"{prefix}_role_concat",
        )(
            [
                goalkeeper_pool,
                defender_pool,
                midfielder_pool,
                forward_pool,
            ]
        )
    else:
        pooled_roles = GlobalAveragePooling1D(
            name=(f"{prefix}_global_pool_" "no_positions"),
        )(encoded_rows)

    team_representation = Dense(
        role_hidden,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name=(f"{prefix}_role_post_dense_1"),
    )(pooled_roles)

    team_representation = Dropout(
        cfg.team_dropout,
        name=(f"{prefix}_role_post_dropout"),
    )(team_representation)

    return Dense(
        output_width,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name=f"{prefix}_team_repr",
    )(team_representation)
