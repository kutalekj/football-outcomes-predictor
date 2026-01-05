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
    Reshape,
)
from tensorflow.keras.models import Model

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
    mode: str = "binary_u25"  # or "goals_dist"
    window_rounds: int = 25
    epochs_per_step: int = 5
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
    x_s = Input((2, 11, 34), name="strength")

    team_emb = Embedding(num_teams, cfg.team_emb_dim)
    home_e = Flatten()(team_emb(x_h))
    away_e = Flatten()(team_emb(x_a))
    comp_e = Flatten()(Embedding(num_comps, cfg.comp_emb_dim)(x_c))

    s = Reshape((22, 34))(x_s)
    s = Dense(64, activation="relu")(s)
    s = GlobalAveragePooling1D()(s)
    s = Dense(cfg.strength_emb_dim, activation="relu")(s)

    z = Concatenate()([x_num, home_e, away_e, comp_e, s])
    z = Dense(128, activation="relu")(z)
    z = Dropout(0.5)(z)
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.4)(z)
    z = Dense(32, activation="relu")(z)

    if cfg.mode == "binary_u25":
        y = Dense(1, activation="sigmoid")(z)
        model = Model([x_num, x_h, x_a, x_c, x_s], y)
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    else:
        y = Dense(cfg.max_goals_class + 1, activation="softmax")(z)
        model = Model([x_num, x_h, x_a, x_c, x_s], y)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def train_rolling(
    matches_sorted: List[FSMatch],
    cat_maps: CatMaps,
    cfg: TrainConfig,
) -> Model:
    rounds = distribute_matches_into_rounds(matches_sorted)  # TODO: Room for improvement

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

    early = EarlyStopping(patience=2, restore_best_weights=True)

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

    for i in range(cfg.window_rounds, len(rounds) - 1):
        train_ms = [m for r in rounds[i - cfg.window_rounds : i] for m in r]
        val_ms = rounds[i]

        X = build_arrays_for_matches(train_ms, cat_maps, cfg.mode, cfg.max_goals_class)
        V = build_arrays_for_matches(val_ms, cat_maps, cfg.mode, cfg.max_goals_class)

        print(f"[train] round {i+1}/{len(rounds)}  train={len(train_ms)} val={len(val_ms)}")

        model.fit(
            X[:-1],
            X[-1],
            validation_data=(V[:-1], V[-1]),
            epochs=cfg.epochs_per_step,
            batch_size=cfg.batch_size,
            callbacks=[early, tb],
            verbose=1,
        )

        val_loss, val_acc = model.evaluate(V[:-1], V[-1], verbose=0)
        print(f"[round {i + 1}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")  # binary mode

        probs = model.predict(V[:-1], verbose=0)
        expected = (probs * np.arange(cfg.max_goals_class + 1)).sum(axis=1)
        mae = np.mean(np.abs(expected - V[-1]))
        print(f"[round {i + 1}] expected_goals_MAE={mae:.3f}")  # total goals regression

    return model
