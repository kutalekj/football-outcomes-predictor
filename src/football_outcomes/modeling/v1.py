from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from football_outcomes.config import fs_settings as sett
from football_outcomes.modeling.common import (
    zero_mask_like,
    zero_vec_from_scalar_input,
)


def build_model_v1(
    num_num: int,
    num_teams: int,
    num_comps: int,
    cfg,
) -> Model:
    x_num = Input((num_num,), name="num")
    x_h = Input((1,), dtype="int32", name="home_id")
    x_a = Input((1,), dtype="int32", name="away_id")
    x_c = Input((1,), dtype="int32", name="comp_id")

    # Team-strength tensor: [home_values, home_mask, away_values, away_mask]
    x_s = Input((4, 11, 34), name="strength")

    # Coarse FootyStats player positions for home/away lineups
    x_hp = Input((11,), dtype="int32", name="home_positions")
    x_ap = Input((11,), dtype="int32", name="away_positions")

    # Team embeddings
    if cfg.use_team_ids:
        team_emb = Embedding(num_teams, cfg.team_emb_dim, name="team_embedding")
        home_e = Flatten(name="home_embedding_flat")(team_emb(x_h))
        away_e = Flatten(name="away_embedding_flat")(team_emb(x_a))
    else:
        home_e = zero_vec_from_scalar_input(x_h, cfg.team_emb_dim, "home_embedding_zero")
        away_e = zero_vec_from_scalar_input(x_a, cfg.team_emb_dim, "away_embedding_zero")

    # Competition embedding
    if cfg.use_comp_embedding:
        comp_emb_layer = Embedding(num_comps, cfg.comp_emb_dim, name="competition_embedding")
        comp_e = Flatten(name="competition_embedding_flat")(comp_emb_layer(x_c))
    else:
        comp_e = zero_vec_from_scalar_input(x_c, cfg.comp_emb_dim, "competition_embedding_zero")

    # Player positions
    if cfg.use_position_embedding:
        position_emb_layer = Embedding(
            input_dim=len(sett.FS_PLAYER_POSITION_TO_IDX),
            output_dim=cfg.position_emb_dim,
            name="position_embedding",
        )
        home_pos_e = position_emb_layer(x_hp)  # (batch, 11, position_emb_dim)
        away_pos_e = position_emb_layer(x_ap)  # (batch, 11, position_emb_dim)
    else:
        pos_dim = int(cfg.position_emb_dim)
        home_pos_e = Lambda(
            lambda t, d=pos_dim: tf.zeros((tf.shape(t)[0], 11, d), dtype=tf.float32),
            name="home_position_zero",
        )(x_hp)
        away_pos_e = Lambda(
            lambda t, d=pos_dim: tf.zeros((tf.shape(t)[0], 11, d), dtype=tf.float32),
            name="away_position_zero",
        )(x_ap)

    if cfg.use_team_strength:
        # Split strength tensor
        # x_s shape = (batch, 4, 11, 34)
        home_vals = Lambda(lambda t: t[:, 0], name="home_strength_values")(x_s)  # (batch, 11, 34)
        home_mask = Lambda(lambda t: t[:, 1], name="home_strength_mask")(x_s)  # (batch, 11, 34)
        away_vals = Lambda(lambda t: t[:, 2], name="away_strength_values")(x_s)  # (batch, 11, 34)
        away_mask = Lambda(lambda t: t[:, 3], name="away_strength_mask")(x_s)  # (batch, 11, 34)

        if not cfg.use_strength_masks:
            home_mask = zero_mask_like(home_vals, "home_strength_mask_zero")
            away_mask = zero_mask_like(away_vals, "away_strength_mask_zero")

        # Concatenate values + mask + position embedding per team
        home_team_input = Concatenate(axis=-1, name="home_strength_concat")([home_vals, home_mask, home_pos_e])
        away_team_input = Concatenate(axis=-1, name="away_strength_concat")([away_vals, away_mask, away_pos_e])

        # Shared team-strength encoder
        strength_dense_1 = Dense(64, activation="relu", name="strength_dense_1")
        strength_dense_2 = Dense(32, activation="relu", name="strength_dense_2")
        strength_pool = GlobalAveragePooling1D(name="strength_pool")
        strength_proj = Dense(cfg.strength_emb_dim, activation="relu", name="strength_projection")

        def encode_team(team_tensor):
            z = strength_dense_1(team_tensor)  # (batch, 11, 64)
            z = strength_dense_2(z)  # (batch, 11, 32)
            z = strength_pool(z)  # (batch, 32)
            z = strength_proj(z)  # (batch, strength_emb_dim)
            return z

        home_s = Lambda(lambda t: t, name="home_strength_embedding")(encode_team(home_team_input))
        away_s = Lambda(lambda t: t, name="away_strength_embedding")(encode_team(away_team_input))
    else:
        home_s = zero_vec_from_scalar_input(x_h, cfg.strength_emb_dim, "home_strength_embedding_zero")
        away_s = zero_vec_from_scalar_input(x_a, cfg.strength_emb_dim, "away_strength_embedding_zero")

    z = Concatenate(name="fusion")([x_num, home_e, away_e, comp_e, home_s, away_s])

    z = Dense(cfg.mlp_hidden_1, activation="relu", name="mlp_dense_1")(z)
    z = Dropout(cfg.mlp_dropout_1, name="mlp_dropout_1")(z)
    z = Dense(cfg.mlp_hidden_2, activation="relu", name="mlp_dense_2")(z)
    z = Dropout(cfg.mlp_dropout_2, name="mlp_dropout_2")(z)
    z = Dense(cfg.mlp_hidden_3, activation="relu", name="mlp_dense_3")(z)

    inputs = [x_num, x_h, x_a, x_c, x_s, x_hp, x_ap]

    if cfg.mode == "binary_u25":
        y = Dense(1, activation="sigmoid", name="output_binary")(z)
        model = Model(inputs, y)
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    elif cfg.mode == "goals_dist":
        y = Dense(cfg.max_goals_class + 1, activation="softmax", name="output_multiclass")(z)
        model = Model(inputs, y)
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    elif cfg.mode == "goals_reg":
        y = Dense(1, activation="linear", name="output_regression")(z)
        model = Model(inputs, y)
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="mae",
            metrics=["mae"],
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    return model
