from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    Lambda,
    LayerNormalization,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from football_outcomes.config import fs_settings as sett
from football_outcomes.modeling.team_strength import (
    abs_diff,
    build_team_repr_v2,
    safe_zero_vec,
    safe_zero_vec_from_inputs,
    split_strength_tensor,
    vec_diff,
)


def _main_loss_and_metrics_for_mode(cfg):
    if cfg.mode == "binary_u25":
        return "binary_crossentropy", ["accuracy", AUC(name="auc")]
    if cfg.mode == "goals_dist":
        return "sparse_categorical_crossentropy", ["accuracy"]
    if cfg.mode == "goals_reg":
        return "mae", ["mae"]
    raise ValueError(f"Unknown mode: {cfg.mode}")


def _aux_loss_and_metrics_for_task(aux_task):
    if aux_task == "binary_u25":
        return "binary_crossentropy", ["accuracy"]
    if aux_task == "goals_dist":
        return "sparse_categorical_crossentropy", ["accuracy"]
    if aux_task == "goals_reg":
        return "mae", ["mae"]
    raise ValueError(f"Unknown aux_task: {aux_task}")


def build_model_v2(
    num_num: int,
    num_teams: int,
    num_comps: int,
    cfg,
) -> Model:
    x_num = Input((num_num,), name="num")
    x_h = Input((1,), dtype="int32", name="home_id")
    x_a = Input((1,), dtype="int32", name="away_id")
    x_c = Input((1,), dtype="int32", name="comp_id")
    x_s = Input((4, 11, 34), name="strength")
    x_hp = Input((11,), dtype="int32", name="home_positions")
    x_ap = Input((11,), dtype="int32", name="away_positions")

    # Branch 1: numerical/context branch
    z_num = Dense(
        96,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.num_l2),
        name="num_branch_dense_1",
    )(x_num)
    z_num = Dropout(cfg.tabular_dropout, name="num_branch_dropout")(z_num)
    z_num = Dense(
        cfg.num_branch_dim,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.num_l2),
        name="num_branch_proj",
    )(z_num)
    z_num = LayerNormalization(name="num_branch_ln")(z_num)

    # Branch 2: categorical branch with explicit comparisons
    if cfg.use_team_ids:
        team_emb = Embedding(
            num_teams,
            cfg.team_emb_dim,
            name="team_embedding",
        )
        home_e = Flatten(
            name="home_embedding_flat",
        )(team_emb(x_h))
        away_e = Flatten(
            name="away_embedding_flat",
        )(team_emb(x_a))
    else:
        home_e = safe_zero_vec(
            x_h,
            cfg.team_emb_dim,
            "home_embedding_zero",
        )
        away_e = safe_zero_vec(
            x_a,
            cfg.team_emb_dim,
            "away_embedding_zero",
        )

    if cfg.use_comp_embedding:
        comp_emb_layer = Embedding(
            num_comps,
            cfg.comp_emb_dim,
            name="competition_embedding",
        )
        comp_e = Flatten(
            name="competition_embedding_flat",
        )(comp_emb_layer(x_c))
    else:
        comp_e = safe_zero_vec(
            x_c,
            cfg.comp_emb_dim,
            "competition_embedding_zero",
        )

    team_diff = vec_diff(home_e, away_e, "team_embedding_diff")
    team_absdiff = abs_diff(home_e, away_e, "team_embedding_absdiff")

    z_cat = Concatenate(name="cat_branch_concat")([home_e, away_e, team_diff, team_absdiff, comp_e])
    z_cat = Dense(
        cfg.cat_branch_dim,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.cat_l2),
        name="cat_branch_proj",
    )(z_cat)
    z_cat = Dropout(cfg.cat_dropout, name="cat_branch_dropout")(z_cat)
    z_cat = LayerNormalization(name="cat_branch_ln")(z_cat)

    # Branch 3: structured team-strength branch
    if cfg.use_team_strength:
        home_vals, home_mask, away_vals, away_mask = split_strength_tensor(x_s)

        if not cfg.use_strength_masks:
            home_mask = Lambda(
                lambda t: tf.ones_like(t),
                name="home_strength_mask_constant",
            )(home_vals)
            away_mask = Lambda(
                lambda t: tf.ones_like(t),
                name="away_strength_mask_constant",
            )(away_vals)

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

        team_repr_diff = vec_diff(
            home_team_repr,
            away_team_repr,
            "team_repr_diff",
        )
        team_repr_absdiff = abs_diff(
            home_team_repr,
            away_team_repr,
            "team_repr_absdiff",
        )

        z_team = Concatenate(
            name="team_branch_concat",
        )(
            [
                home_team_repr,
                away_team_repr,
                team_repr_diff,
                team_repr_absdiff,
            ]
        )
        z_team = Dense(
            cfg.team_branch_dim,
            activation="relu",
            kernel_regularizer=regularizers.l2(cfg.team_l2),
            name="team_branch_proj",
        )(z_team)
        z_team = Dropout(
            cfg.team_dropout,
            name="team_branch_dropout",
        )(z_team)
        z_team = LayerNormalization(
            name="team_branch_ln",
        )(z_team)
    else:
        z_team = safe_zero_vec_from_inputs(
            [x_s, x_hp, x_ap],
            cfg.team_branch_dim,
            "team_branch_zero",
        )

    # Fusion
    z = Concatenate(name="fusion")([z_num, z_cat, z_team])
    z = Dense(
        cfg.fusion_hidden_dim_1,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.fusion_l2),
        name="fusion_dense_1",
    )(z)
    z = Dropout(cfg.fusion_dropout_1, name="fusion_dropout_1")(z)
    z = Dense(
        cfg.fusion_hidden_dim_2,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.fusion_l2),
        name="fusion_dense_2",
    )(z)
    z = Dropout(cfg.fusion_dropout_2, name="fusion_dropout_2")(z)

    # Main output
    if cfg.mode == "binary_u25":
        output_main = Dense(1, activation="sigmoid", name="output_main")(z)
    elif cfg.mode == "goals_dist":
        output_main = Dense(cfg.max_goals_class + 1, activation="softmax", name="output_main")(z)
    elif cfg.mode == "goals_reg":
        output_main = Dense(1, activation="linear", name="output_main")(z)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    outputs = [output_main]

    # Optional auxiliary output from team branch only
    if cfg.use_team_aux_head and cfg.aux_task is not None:
        z_aux = Dense(32, activation="relu", name="team_aux_hidden")(z_team)

        if cfg.aux_task == "binary_u25":
            output_aux = Dense(1, activation="sigmoid", name="output_team_aux")(z_aux)
        elif cfg.aux_task == "goals_dist":
            output_aux = Dense(cfg.max_goals_class + 1, activation="softmax", name="output_team_aux")(z_aux)
        elif cfg.aux_task == "goals_reg":
            output_aux = Dense(1, activation="linear", name="output_team_aux")(z_aux)
        else:
            raise ValueError(f"Unknown aux_task: {cfg.aux_task}")

        outputs.append(output_aux)

    model = Model(inputs=[x_num, x_h, x_a, x_c, x_s, x_hp, x_ap], outputs=outputs)

    main_loss, main_metrics = _main_loss_and_metrics_for_mode(cfg)

    if cfg.use_team_aux_head and cfg.aux_task is not None:
        aux_loss, aux_metrics = _aux_loss_and_metrics_for_task(cfg.aux_task)

        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss={
                "output_main": main_loss,
                "output_team_aux": aux_loss,
            },
            loss_weights={
                "output_main": 1.0,
                "output_team_aux": cfg.aux_weight,
            },
            metrics={
                "output_main": main_metrics,
                "output_team_aux": aux_metrics,
            },
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss={"output_main": main_loss},
            metrics={"output_main": main_metrics},
        )

    return model
