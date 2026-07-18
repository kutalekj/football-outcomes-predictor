from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    Lambda,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from football_outcomes.config import fs_settings as sett
from football_outcomes.modeling.common import (
    zero_mask_like,
)
from football_outcomes.modeling.team_strength import (
    abs_diff,
    build_team_repr_v2,
    split_strength_tensor,
    vec_diff,
)


def _apply_strength_representation_v1(home_vals, home_mask, away_vals, away_mask, x_hp, x_ap, cfg):
    if not cfg.use_strength_masks:
        home_mask = zero_mask_like(home_vals, "home_strength_mask_zero")
        away_mask = zero_mask_like(away_vals, "away_strength_mask_zero")

    if cfg.use_position_embedding:
        position_emb_layer = Embedding(
            input_dim=len(sett.FS_PLAYER_POSITION_TO_IDX),
            output_dim=cfg.position_emb_dim,
            name="position_embedding",
        )
        home_pos_e = position_emb_layer(x_hp)
        away_pos_e = position_emb_layer(x_ap)
    else:
        pos_dim = int(cfg.position_emb_dim)
        home_pos_e = Lambda(
            lambda p, d=pos_dim: tf.zeros((tf.shape(p)[0], 11, d), dtype=tf.float32),
            name="home_position_zero",
        )(x_hp)
        away_pos_e = Lambda(
            lambda p, d=pos_dim: tf.zeros((tf.shape(p)[0], 11, d), dtype=tf.float32),
            name="away_position_zero",
        )(x_ap)

    return home_mask, away_mask, home_pos_e, away_pos_e


def build_strength_pretrain_model_v1(
    cfg,
) -> Model:
    """
    Standalone pretraining model using the same structured branch design as v1:
      position embedding + shared dense row encoder + global average pooling + projection
    """
    x_s = Input((4, 11, 34), name="strength")
    x_hp = Input((11,), dtype="int32", name="home_positions")
    x_ap = Input((11,), dtype="int32", name="away_positions")

    home_vals = Lambda(lambda t: t[:, 0], name="home_strength_values")(x_s)
    home_mask = Lambda(lambda t: t[:, 1], name="home_strength_mask")(x_s)
    away_vals = Lambda(lambda t: t[:, 2], name="away_strength_values")(x_s)
    away_mask = Lambda(lambda t: t[:, 3], name="away_strength_mask")(x_s)

    home_mask, away_mask, home_pos_e, away_pos_e = _apply_strength_representation_v1(
        home_vals, home_mask, away_vals, away_mask, x_hp, x_ap, cfg
    )
    home_team_input = Concatenate(axis=-1, name="home_strength_concat")([home_vals, home_mask, home_pos_e])
    away_team_input = Concatenate(axis=-1, name="away_strength_concat")([away_vals, away_mask, away_pos_e])

    strength_dense_1 = Dense(64, activation="relu", name="strength_dense_1")
    strength_dense_2 = Dense(32, activation="relu", name="strength_dense_2")
    strength_pool = GlobalAveragePooling1D(name="strength_pool")
    strength_proj = Dense(cfg.strength_emb_dim, activation="relu", name="strength_projection")

    def encode_team(team_tensor):
        z = strength_dense_1(team_tensor)
        z = strength_dense_2(z)
        z = strength_pool(z)
        z = strength_proj(z)
        return z

    home_s = Lambda(lambda t: t, name="home_strength_embedding")(encode_team(home_team_input))
    away_s = Lambda(lambda t: t, name="away_strength_embedding")(encode_team(away_team_input))

    diff = Lambda(lambda xs: xs[0] - xs[1], name="strength_diff")([home_s, away_s])
    absdiff = Lambda(lambda xs: tf.abs(xs[0] - xs[1]), name="strength_absdiff")([home_s, away_s])

    z = Concatenate(name="pretrain_fusion")([home_s, away_s, diff, absdiff])
    z = Dense(cfg.compare_hidden_dim, activation="relu", name="pretrain_head_dense_1")(z)
    z = Dropout(cfg.compare_dropout, name="pretrain_head_dropout")(z)

    if cfg.mode == "binary_u25":
        y = Dense(1, activation="sigmoid", name="output_binary")(z)
        model = Model([x_s, x_hp, x_ap], y)
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
        raise ValueError("Strength pretraining is currently intended for mode='binary_u25'.")

    return model


def build_strength_pretrain_model_v2(
    cfg,
) -> Model:
    """
    Standalone pretraining model using the role-aware v2/v2-lite branch design.
    """
    x_s = Input((4, 11, 34), name="strength")
    x_hp = Input((11,), dtype="int32", name="home_positions")
    x_ap = Input((11,), dtype="int32", name="away_positions")

    home_vals, home_mask, away_vals, away_mask = split_strength_tensor(x_s)

    if not cfg.use_strength_masks:
        home_mask = Lambda(lambda t: tf.ones_like(t), name="home_strength_mask_constant")(home_vals)
        away_mask = Lambda(lambda t: tf.ones_like(t), name="away_strength_mask_constant")(away_vals)

    if cfg.use_position_embedding:
        position_emb_layer = Embedding(
            input_dim=len(sett.FS_PLAYER_POSITION_TO_IDX),
            output_dim=cfg.position_emb_dim,
            name="position_embedding",
        )
    else:
        position_emb_layer = None

    home_team_repr = build_team_repr_v2(
        home_vals,
        home_mask,
        x_hp,
        position_emb_layer,
        cfg,
        prefix="home",
    )

    away_team_repr = build_team_repr_v2(
        away_vals,
        away_mask,
        x_ap,
        position_emb_layer,
        cfg,
        prefix="away",
    )

    team_repr_diff = vec_diff(home_team_repr, away_team_repr, "team_repr_diff")
    team_repr_absdiff = abs_diff(home_team_repr, away_team_repr, "team_repr_absdiff")

    z = Concatenate(name="team_branch_concat")([home_team_repr, away_team_repr, team_repr_diff, team_repr_absdiff])
    z = Dense(
        cfg.team_branch_dim,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name="team_branch_proj",
    )(z)
    z = Dropout(cfg.team_dropout, name="team_branch_dropout")(z)

    z = Dense(cfg.compare_hidden_dim, activation="relu", name="pretrain_head_dense_1")(z)
    z = Dropout(cfg.compare_dropout, name="pretrain_head_dropout")(z)

    if cfg.mode == "binary_u25":
        y = Dense(1, activation="sigmoid", name="output_binary")(z)
        model = Model([x_s, x_hp, x_ap], y)
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", AUC(name="auc")],
        )
    else:
        raise ValueError("Strength pretraining is currently intended for mode='binary_u25'.")

    return model


def build_strength_pretrain_model(
    cfg,
) -> Model:
    if cfg.branch_version == "v1":
        return build_strength_pretrain_model_v1(cfg)
    if cfg.branch_version == "v2":
        return build_strength_pretrain_model_v2(cfg)
    raise ValueError(f"Unknown branch_version: {cfg.branch_version}")
