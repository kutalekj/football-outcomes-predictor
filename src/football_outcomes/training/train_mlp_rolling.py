from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
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
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.training.fs_training_utils import (
    CatMaps,
    build_arrays_for_matches,
    distribute_matches_into_rounds,
    extract_numerical_features,
)


@dataclass
class TrainConfig:
    mode: str = "binary_u25"  # "binary_u25" | "goals_dist" | "goals_reg"
    window_rounds: int = 25
    epochs_per_step: int = 5
    learning_rate: float = 0.0001
    batch_size: int = 64

    team_emb_dim: int = 8
    comp_emb_dim: int = 5
    strength_emb_dim: int = 24

    max_goals_class: int = 10
    seed: int | None = 42


def set_global_seed(seed: int) -> None:
    """
    Best-effort reproducibility for TF/Keras.
    Note: full determinism on GPU may still vary unless you also enable deterministic ops.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Optional: more determinism (can be slower)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_model(num_num, num_teams, num_comps, cfg: TrainConfig) -> Model:
    x_num = Input((num_num,), name="num")
    x_h = Input((1,), dtype="int32", name="home_id")
    x_a = Input((1,), dtype="int32", name="away_id")
    x_c = Input((1,), dtype="int32", name="comp_id")

    # Team-Strength tensor: [home_values, home_mask, away_values, away_mask]
    x_s = Input((4, 11, 34), name="strength")

    # Categorical branches
    team_emb = Embedding(num_teams, cfg.team_emb_dim, name="team_embedding")
    home_e = Flatten(name="home_embedding_flat")(team_emb(x_h))
    away_e = Flatten(name="away_embedding_flat")(team_emb(x_a))

    comp_emb_layer = Embedding(num_comps, cfg.comp_emb_dim, name="competition_embedding")
    comp_e = Flatten(name="competition_embedding_flat")(comp_emb_layer(x_c))

    # Split strength tensor
    # x_s shape = (batch, 4, 11, 34)
    home_vals = Lambda(lambda t: t[:, 0], name="home_strength_values")(x_s)  # (batch, 11, 34)
    home_mask = Lambda(lambda t: t[:, 1], name="home_strength_mask")(x_s)  # (batch, 11, 34)
    away_vals = Lambda(lambda t: t[:, 2], name="away_strength_values")(x_s)  # (batch, 11, 34)
    away_mask = Lambda(lambda t: t[:, 3], name="away_strength_mask")(x_s)  # (batch, 11, 34)

    # Concatenate values + mask per team => (batch, 11, 68)
    home_team_input = Concatenate(axis=-1, name="home_strength_concat")([home_vals, home_mask])
    away_team_input = Concatenate(axis=-1, name="away_strength_concat")([away_vals, away_mask])

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

    home_s = encode_team(home_team_input)
    away_s = encode_team(away_team_input)

    z = Concatenate(name="fusion")([x_num, home_e, away_e, comp_e, home_s, away_s])

    z = Dense(128, activation="relu", name="mlp_dense_1")(z)
    z = Dropout(0.5, name="mlp_dropout_1")(z)
    z = Dense(64, activation="relu", name="mlp_dense_2")(z)
    z = Dropout(0.4, name="mlp_dropout_2")(z)
    z = Dense(32, activation="relu", name="mlp_dense_3")(z)

    if cfg.mode == "binary_u25":
        y = Dense(1, activation="sigmoid", name="output_binary")(z)
        model = Model([x_num, x_h, x_a, x_c, x_s], y)
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    elif cfg.mode == "goals_dist":
        y = Dense(cfg.max_goals_class + 1, activation="softmax", name="output_multiclass")(z)
        model = Model([x_num, x_h, x_a, x_c, x_s], y)
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    elif cfg.mode == "goals_reg":
        y = Dense(1, activation="linear", name="output_regression")(z)
        model = Model([x_num, x_h, x_a, x_c, x_s], y)
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="mae",
            metrics=["mae"],
        )

    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    return model


def train_rolling(
    matches_sorted: List[FSMatch],
    cat_maps: CatMaps,
    cfg: TrainConfig,
) -> Model:
    rounds = distribute_matches_into_rounds(matches_sorted)

    sample_feat = matches_sorted[0].features_before_match
    num_num = extract_numerical_features(sample_feat).shape[0]

    if cfg.seed is not None:
        set_global_seed(cfg.seed)
        print(f"[seed] Using seed={cfg.seed}")

    model = build_model(
        num_num=num_num,
        num_teams=len(cat_maps.team_id_map),
        num_comps=len(cat_maps.comp_id_map),
        cfg=cfg,
    )

    # ---- TensorBoard logging (always under sett.DATA_DIR)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = Path(sett.DATA_DIR) / "tensorboard_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    log_dir = str(log_root / f"mlp_{cfg.mode}_{run_stamp}")
    tb = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
    )
    print(f"[tensorboard] logging to {log_dir}")

    # Extra per-round metrics: show in TensorBoard one point per rolling round
    tb_writer = tf.summary.create_file_writer(log_dir)

    for i in range(cfg.window_rounds, len(rounds) - 1):
        train_ms = [m for r in rounds[i - cfg.window_rounds : i] for m in r]
        val_ms = rounds[i]

        X = build_arrays_for_matches(train_ms, cat_maps, cfg.mode, cfg.max_goals_class)
        V = build_arrays_for_matches(val_ms, cat_maps, cfg.mode, cfg.max_goals_class)

        print(f"[train] round {i+1}/{len(rounds)}  train={len(train_ms)} val={len(val_ms)}")

        early = EarlyStopping(
            patience=2,
            restore_best_weights=True,
            monitor="val_loss",
            mode="min",
        )

        model.fit(
            X[:-1],
            X[-1],
            validation_data=(V[:-1], V[-1]),
            epochs=cfg.epochs_per_step,
            batch_size=cfg.batch_size,
            callbacks=[early, tb],
            verbose=1,
        )

        val_metrics = model.evaluate(V[:-1], V[-1], verbose=0)
        round_step = int(i + 1)  # 1-based round index for readability in TensorBoard

        if cfg.mode == "binary_u25":  # binary classification
            val_loss, val_acc = val_metrics
            print(f"[round {i + 1}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        elif cfg.mode == "goals_dist":  # multi-class classification
            val_loss, val_acc = val_metrics
            print(f"[round {i + 1}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            probabilities = model.predict(V[:-1], verbose=0)
            expected = (probabilities * np.arange(cfg.max_goals_class + 1)).sum(axis=1)
            mae = np.mean(np.abs(expected - V[-1]))
            print(f"[round {i + 1}] expected_goals_MAE={mae:.3f}")

            # Derived Under 2.5 metrics from distribution (p_under25 = p0 + p1 + p2)
            p_under25 = probabilities[:, :3].sum(axis=1).astype(np.float32)
            y_under25 = (V[-1] <= 2).astype(np.float32)

            derived_under25_acc = float(np.mean((p_under25 >= 0.5) == (y_under25 >= 0.5)))
            brier = float(np.mean((p_under25 - y_under25) ** 2))

            auc_metric = AUC(curve="ROC")
            auc_metric.update_state(y_under25, p_under25)
            derived_under25_auc = float(auc_metric.result().numpy())

            print(
                f"[round {i + 1}] derived_under25_acc(p0+p1+p2)={derived_under25_acc:.3f} "
                f"auc={derived_under25_auc:.3f} brier={brier:.4f}"
            )

            # Manual TensorBoard scalars (one point per round)
            with tb_writer.as_default():
                tf.summary.scalar("round/val_loss", float(val_loss), step=round_step)
                tf.summary.scalar("round/val_accuracy", float(val_acc), step=round_step)
                tf.summary.scalar("round/expected_goals_mae", float(mae), step=round_step)
                tf.summary.scalar("round/derived_under25_accuracy", derived_under25_acc, step=round_step)
                tf.summary.scalar("round/derived_under25_auc", derived_under25_auc, step=round_step)
                tf.summary.scalar("round/derived_under25_brier", brier, step=round_step)
                tb_writer.flush()

        elif cfg.mode == "goals_reg":  # regression
            val_loss, val_mae = val_metrics
            print(f"[round {i + 1}] val_MAE={val_mae:.3f}")

            predictions = model.predict(V[:-1], verbose=0).ravel()
            under25_acc = np.mean((predictions < 2.5) == (V[-1] < 2.5))
            print(f"[round {i + 1}] derived_under25_acc={under25_acc:.3f}")

            # Manual TensorBoard scalars (one point per round)
            with tb_writer.as_default():
                tf.summary.scalar("round/val_mae", float(val_mae), step=round_step)
                tf.summary.scalar("round/derived_under25_accuracy", float(under25_acc), step=round_step)
                tb_writer.flush()

    return model
