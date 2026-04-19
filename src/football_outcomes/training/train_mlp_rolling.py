from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard
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
    summarize_rounds,
)


@dataclass
class TrainConfig:
    mode: str = "binary_u25"
    window_rounds: int = 25
    epochs_per_step: int = 5
    learning_rate: float = 0.0001
    batch_size: int = 64

    team_emb_dim: int = 8
    comp_emb_dim: int = 5
    strength_emb_dim: int = 24
    position_emb_dim: int = 3

    max_goals_class: int = 10
    seed: int | None = 42

    # Logging and evaluation
    run_name: str | None = None
    min_warning_val_size: int = 20
    save_oos_predictions: bool = True

    # Observability and diagnostics
    enable_branch_diagnostics: bool = True
    probe_matches: int = 64
    use_team_strength: bool = True
    use_team_ids: bool = True
    use_comp_embedding: bool = True
    use_position_embedding: bool = True


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _zero_vec_from_scalar_input(x, width: int, name: str):
    return Lambda(
        lambda t: tf.zeros((tf.shape(t)[0], width), dtype=tf.float32),
        name=name,
    )(x)


class LayerDriftLogger(Callback):
    """
    Logs L2 drift from initialization for selected layers.
    """

    def __init__(self, layer_names: List[str], writer, every_epoch: bool = True):
        super().__init__()
        self.layer_names = layer_names
        self.writer = writer
        self.every_epoch = every_epoch
        self._initial = {}

    def on_train_begin(self, logs=None):
        for name in self.layer_names:
            layer = self.model.get_layer(name)
            self._initial[name] = [w.numpy().copy() for w in layer.weights]

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for name in self.layer_names:
                layer = self.model.get_layer(name)
                init_ws = self._initial[name]
                curr_ws = layer.weights
                sq = 0.0
                for w0, w1 in zip(init_ws, curr_ws):
                    diff = w1.numpy() - w0
                    sq += float(np.sum(diff * diff))
                drift = float(np.sqrt(sq))
                tf.summary.scalar(f"diag_drift/{name}", drift, step=epoch + 1)
            self.writer.flush()


class BranchProbeLogger(Callback):
    """
    Logs activation variance on a fixed probe batch to detect dead branches.
    """

    def __init__(self, probe_inputs, writer, layer_names: List[str]):
        super().__init__()
        self.probe_inputs = probe_inputs
        self.writer = writer
        self.layer_names = layer_names
        self._submodels = {}

    def on_train_begin(self, logs=None):
        for name in self.layer_names:
            self._submodels[name] = Model(self.model.inputs, self.model.get_layer(name).output)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for name, sm in self._submodels.items():
                out = sm.predict(self.probe_inputs, verbose=0)
                tf.summary.scalar(f"diag_probe_meanabs/{name}", float(np.mean(np.abs(out))), step=epoch + 1)
                tf.summary.scalar(f"diag_probe_std/{name}", float(np.std(out)), step=epoch + 1)
            self.writer.flush()


def build_model(num_num, num_teams, num_comps, cfg: TrainConfig) -> Model:
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
        home_e = _zero_vec_from_scalar_input(x_h, cfg.team_emb_dim, "home_embedding_zero")
        away_e = _zero_vec_from_scalar_input(x_a, cfg.team_emb_dim, "away_embedding_zero")

    # Competition embedding
    if cfg.use_comp_embedding:
        comp_emb_layer = Embedding(num_comps, cfg.comp_emb_dim, name="competition_embedding")
        comp_e = Flatten(name="competition_embedding_flat")(comp_emb_layer(x_c))
    else:
        comp_e = _zero_vec_from_scalar_input(x_c, cfg.comp_emb_dim, "competition_embedding_zero")

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
        home_pos_e = Lambda(
            lambda t: tf.zeros((tf.shape(t)[0], 11, cfg.position_emb_dim), dtype=tf.float32),
            name="home_position_zero",
        )(x_hp)
        away_pos_e = Lambda(
            lambda t: tf.zeros((tf.shape(t)[0], 11, cfg.position_emb_dim), dtype=tf.float32),
            name="away_position_zero",
        )(x_ap)

    if cfg.use_team_strength:
        # Split strength tensor
        # x_s shape = (batch, 4, 11, 34)
        home_vals = Lambda(lambda t: t[:, 0], name="home_strength_values")(x_s)  # (batch, 11, 34)
        home_mask = Lambda(lambda t: t[:, 1], name="home_strength_mask")(x_s)  # (batch, 11, 34)
        away_vals = Lambda(lambda t: t[:, 2], name="away_strength_values")(x_s)  # (batch, 11, 34)
        away_mask = Lambda(lambda t: t[:, 3], name="away_strength_mask")(x_s)  # (batch, 11, 34)

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
        home_s = _zero_vec_from_scalar_input(x_h, cfg.strength_emb_dim, "home_strength_embedding_zero")
        away_s = _zero_vec_from_scalar_input(x_a, cfg.strength_emb_dim, "away_strength_embedding_zero")

    z = Concatenate(name="fusion")([x_num, home_e, away_e, comp_e, home_s, away_s])

    z = Dense(128, activation="relu", name="mlp_dense_1")(z)
    z = Dropout(0.5, name="mlp_dropout_1")(z)
    z = Dense(64, activation="relu", name="mlp_dense_2")(z)
    z = Dropout(0.4, name="mlp_dropout_2")(z)
    z = Dense(32, activation="relu", name="mlp_dense_3")(z)

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


def _binary_summary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_hat = (y_prob >= 0.5).astype(np.float32)
    acc = float(np.mean(y_hat == y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    auc_metric = AUC(curve="ROC")
    auc_metric.update_state(y_true.astype(np.float32), y_prob.astype(np.float32))
    auc = float(auc_metric.result().numpy())
    return {"pooled_accuracy": acc, "pooled_brier": brier, "pooled_auc": auc}


def _reg_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {"pooled_mae": mae, "pooled_rmse": rmse}


def train_rolling(
    matches_sorted: List[FSMatch],
    cat_maps: CatMaps,
    cfg: TrainConfig,
) -> Model:
    rounds = distribute_matches_into_rounds(matches_sorted)
    round_info = summarize_rounds(rounds)
    print(f"[rounds] {round_info}")

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

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"mlp_{cfg.mode}_{run_stamp}"

    # TensorBoard logging (always under sett.DATA_DIR)
    log_root = Path(sett.DATA_DIR) / "tensorboard_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = str(log_root / run_name)

    tb = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
    )
    # Extra per-round metrics: show in TensorBoard one point per rolling round
    tb_writer = tf.summary.create_file_writer(log_dir)

    print(f"[tensorboard] logging to {log_dir}")

    round_records = []
    oos_rows = []

    probe_inputs = None
    if cfg.enable_branch_diagnostics:
        probe_ms = matches_sorted[: min(len(matches_sorted), cfg.probe_matches)]
        probe_arrays = build_arrays_for_matches(probe_ms, cat_maps, cfg.mode, cfg.max_goals_class)
        probe_inputs = probe_arrays[:-1]

    callbacks_common: List[Callback] = [tb]
    if cfg.enable_branch_diagnostics and probe_inputs is not None:
        drift_names = []
        if cfg.use_team_ids:
            drift_names.append("team_embedding")
        if cfg.use_comp_embedding:
            drift_names.append("competition_embedding")
        if cfg.use_position_embedding:
            drift_names.append("position_embedding")
        if cfg.use_team_strength:
            drift_names.extend(["strength_dense_1", "strength_dense_2", "strength_projection"])

        if drift_names:
            callbacks_common.append(LayerDriftLogger(drift_names, tb_writer))

        branch_probe_layers = [
            "home_embedding_flat" if cfg.use_team_ids else "home_embedding_zero",
            "competition_embedding_flat" if cfg.use_comp_embedding else "competition_embedding_zero",
            "home_strength_embedding" if cfg.use_team_strength else "home_strength_embedding_zero",
        ]
        callbacks_common.append(BranchProbeLogger(probe_inputs, tb_writer, branch_probe_layers))

    for i in range(cfg.window_rounds, len(rounds) - 1):
        train_ms = [m for r in rounds[i - cfg.window_rounds : i] for m in r]
        val_ms = rounds[i]

        X = build_arrays_for_matches(train_ms, cat_maps, cfg.mode, cfg.max_goals_class)
        V = build_arrays_for_matches(val_ms, cat_maps, cfg.mode, cfg.max_goals_class)

        print(f"[train] round {i+1}/{len(rounds)} train={len(train_ms)} val={len(val_ms)}")

        if len(val_ms) < cfg.min_warning_val_size:
            print(f"[warn] round {i+1} has small validation size: {len(val_ms)}")

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
            callbacks=[early] + callbacks_common,
            verbose=1,
        )

        val_metrics = model.evaluate(V[:-1], V[-1], verbose=0)
        round_step = int(i + 1)  # 1-based round index for readability in TensorBoard

        if cfg.mode == "binary_u25":
            val_loss, val_acc = val_metrics
            val_prob = model.predict(V[:-1], verbose=0).ravel().astype(np.float32)
            y_true = V[-1].astype(np.float32)

            auc_metric = AUC(curve="ROC")
            auc_metric.update_state(y_true, val_prob)
            val_auc = float(auc_metric.result().numpy())
            val_brier = float(np.mean((val_prob - y_true) ** 2))

            round_records.append(
                {
                    "round_idx": round_step,
                    "train_size": len(train_ms),
                    "val_size": len(val_ms),
                    "positive_rate_val": float(np.mean(y_true)),
                    "mode": cfg.mode,
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "val_auc": val_auc,
                    "val_brier": val_brier,
                }
            )

            for m, yt, yp in zip(val_ms, y_true, val_prob):
                oos_rows.append(
                    {
                        "round_idx": round_step,
                        "match_id": m.id,
                        "season": m.season,
                        "competition": m.comp_name,
                        "y_true": float(yt),
                        "y_prob_under25": float(yp),
                    }
                )

            with tb_writer.as_default():
                tf.summary.scalar("round/val_loss", float(val_loss), step=round_step)
                tf.summary.scalar("round/val_accuracy", float(val_acc), step=round_step)
                tf.summary.scalar("round/val_auc", val_auc, step=round_step)
                tf.summary.scalar("round/val_brier", val_brier, step=round_step)
                tf.summary.scalar("round/val_size", len(val_ms), step=round_step)
                tf.summary.scalar("round/positive_rate_val", float(np.mean(y_true)), step=round_step)
                tb_writer.flush()

        elif cfg.mode == "goals_dist":
            val_loss, val_acc = val_metrics
            probabilities = model.predict(V[:-1], verbose=0)
            expected = (probabilities * np.arange(cfg.max_goals_class + 1)).sum(axis=1)
            mae = np.mean(np.abs(expected - V[-1]))

            round_records.append(
                {
                    "round_idx": round_step,
                    "train_size": len(train_ms),
                    "val_size": len(val_ms),
                    "mode": cfg.mode,
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "expected_goals_mae": float(mae),
                }
            )

            # Manual TensorBoard scalars (one point per round)
            with tb_writer.as_default():
                tf.summary.scalar("round/val_loss", float(val_loss), step=round_step)
                tf.summary.scalar("round/val_accuracy", float(val_acc), step=round_step)
                tf.summary.scalar("round/expected_goals_mae", float(mae), step=round_step)
                tf.summary.scalar("round/val_size", len(val_ms), step=round_step)
                tb_writer.flush()

        elif cfg.mode == "goals_reg":
            val_loss, val_mae = val_metrics
            predictions = model.predict(V[:-1], verbose=0).ravel()
            rmse = float(np.sqrt(np.mean((predictions - V[-1]) ** 2)))

            round_records.append(
                {
                    "round_idx": round_step,
                    "train_size": len(train_ms),
                    "val_size": len(val_ms),
                    "mode": cfg.mode,
                    "val_mae": float(val_mae),
                    "val_rmse": rmse,
                }
            )

            for m, yt, yp in zip(val_ms, V[-1], predictions):
                oos_rows.append(
                    {
                        "round_idx": round_step,
                        "match_id": m.id,
                        "season": m.season,
                        "competition": m.comp_name,
                        "y_true_goals": float(yt),
                        "y_pred_goals": float(yp),
                    }
                )

            # Manual TensorBoard scalars (one point per round)
            with tb_writer.as_default():
                tf.summary.scalar("round/val_mae", float(val_mae), step=round_step)
                tf.summary.scalar("round/val_rmse", rmse, step=round_step)
                tf.summary.scalar("round/val_size", len(val_ms), step=round_step)
                tb_writer.flush()

    csv_path = Path(log_dir) / "round_metrics.csv"
    if round_records:
        fieldnames = sorted({k for rec in round_records for k in rec.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(round_records)
        print(f"[metrics] saved round-level metrics to {csv_path}")

    if cfg.save_oos_predictions and oos_rows:
        pred_path = Path(log_dir) / "oos_predictions.csv"
        fieldnames = sorted({k for rec in oos_rows for k in rec.keys()})
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(oos_rows)
        print(f"[metrics] saved pooled OOS predictions to {pred_path}")

    summary = {"run_name": run_name, "mode": cfg.mode, "round_stats": round_info}

    if cfg.mode == "binary_u25" and oos_rows:
        y_true = np.asarray([r["y_true"] for r in oos_rows], dtype=np.float32)
        y_prob = np.asarray([r["y_prob_under25"] for r in oos_rows], dtype=np.float32)
        summary.update(_binary_summary(y_true, y_prob))

    if cfg.mode == "goals_reg" and oos_rows:
        y_true = np.asarray([r["y_true_goals"] for r in oos_rows], dtype=np.float32)
        y_pred = np.asarray([r["y_pred_goals"] for r in oos_rows], dtype=np.float32)
        summary.update(_reg_summary(y_true, y_pred))

    summary_path = Path(log_dir) / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[metrics] saved summary to {summary_path}")
    print(f"[summary] {summary}")

    return model
