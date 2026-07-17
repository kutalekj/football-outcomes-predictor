from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
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
    LayerNormalization,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.training.fs_training_utils import (
    CatMaps,
    build_arrays_for_matches,
    build_aux_targets_for_matches,
    build_strength_only_arrays_for_matches,
    distribute_matches_into_rounds,
    extract_numerical_features,
    summarize_rounds,
)

matplotlib.use("Agg")


@dataclass
class TrainConfig:
    mode: str = "binary_u25"  # "binary_u25" | "goals_dist" | "goals_reg"
    window_rounds: int = 25
    epochs_per_step: int = 5
    learning_rate: float = 0.0001
    batch_size: int = 64

    team_emb_dim: int = 8
    comp_emb_dim: int = 5
    strength_emb_dim: int = 16
    position_emb_dim: int = 3

    max_goals_class: int = 10
    seed: int | None = 42

    # New:
    model_version: str = "v2"  # "v1" | "v2"
    use_team_aux_head: bool = False
    aux_task: str | None = None
    aux_weight: float = 0.15

    # Branch widths (v2-lite)
    num_branch_dim: int = 48
    cat_branch_dim: int = 32
    team_branch_dim: int = 32
    player_row_hidden_dim: int = 32
    role_post_hidden_dim: int = 32
    fusion_hidden_dim_1: int = 64
    fusion_hidden_dim_2: int = 32

    # Regularization (v2-lite)
    tabular_dropout: float = 0.20
    cat_dropout: float = 0.15
    team_dropout: float = 0.25
    fusion_dropout_1: float = 0.45
    fusion_dropout_2: float = 0.30

    # L2 regularization (v2-lite)
    num_l2: float = 1e-5
    cat_l2: float = 1e-5
    team_l2: float = 5e-5
    fusion_l2: float = 5e-5

    early_stopping_patience: int = 1
    early_stopping_min_delta: float = 0.0
    freeze_pretrained_branch_rounds: int = 0

    # Logging and evaluation
    run_name: str | None = None
    min_warning_val_size: int = 20
    save_oos_predictions: bool = True

    # Observability and diagnostics
    enable_branch_diagnostics: bool = True
    probe_matches: int = 32
    use_team_strength: bool = True
    use_team_ids: bool = True
    use_comp_embedding: bool = True
    use_position_embedding: bool = True

    representation: str = "full"
    use_strength_masks: bool = True

    # v1 MLP architecture
    mlp_hidden_1: int = 128
    mlp_hidden_2: int = 64
    mlp_hidden_3: int = 32
    mlp_dropout_1: float = 0.50
    mlp_dropout_2: float = 0.40

    # Learning-rate schedule across rolling rounds
    lr_schedule: str = "constant"  # "constant" | "exponential" | "cosine"
    lr_decay_rate: float = 0.997
    min_learning_rate: float = 2e-5


@dataclass
class StrengthPretrainConfig:
    branch_version: str = "v1"  # "v1" | "v2"
    mode: str = "binary_u25"  # keep binary_u25 as the main use case now

    window_rounds: int = 25
    epochs_per_step: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 64

    max_goals_class: int = 10
    seed: int | None = 42
    run_name: str | None = None

    early_stopping_patience: int = 1
    early_stopping_min_delta: float = 0.0

    # shared branch dimensions
    strength_emb_dim: int = 16
    position_emb_dim: int = 3

    # v2 branch params
    player_row_hidden_dim: int = 32
    role_post_hidden_dim: int = 32
    team_branch_dim: int = 32
    team_dropout: float = 0.25
    team_l2: float = 5e-5

    # small standalone classifier head
    compare_hidden_dim: int = 32
    compare_dropout: float = 0.20

    # logging
    save_oos_predictions: bool = True

    # Structured input representation variant:
    # "full"        = skills + masks + positions
    # "no_positions"= skills + masks
    # "no_masks"    = skills + positions
    # "skills_only" = skills only
    representation: str = "full"
    use_strength_masks: bool = True
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
    Skips layers that are not present in the current model variant.
    """

    def __init__(self, layer_names: List[str], writer, every_epoch: bool = True):
        super().__init__()
        self.layer_names = layer_names
        self.writer = writer
        self.every_epoch = every_epoch
        self._initial = {}
        self._present_layer_names = []

    def on_train_begin(self, logs=None):
        self._present_layer_names = []
        for name in self.layer_names:
            try:
                layer = self.model.get_layer(name)
            except ValueError:
                continue
            self._initial[name] = [w.numpy().copy() for w in layer.weights]
            self._present_layer_names.append(name)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for name in self._present_layer_names:
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
    Skips layers that are not present in the current model variant.
    """

    def __init__(self, probe_inputs, writer, layer_names: List[str]):
        super().__init__()
        self.probe_inputs = probe_inputs
        self.writer = writer
        self.layer_names = layer_names
        self._submodels = {}

    def on_train_begin(self, logs=None):
        self._submodels = {}
        for name in self.layer_names:
            try:
                layer = self.model.get_layer(name)
            except ValueError:
                continue
            self._submodels[name] = Model(self.model.inputs, layer.output)

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for name, sm in self._submodels.items():
                out = sm.predict(self.probe_inputs, verbose=0)
                tf.summary.scalar(f"diag_probe_meanabs/{name}", float(np.mean(np.abs(out))), step=epoch + 1)
                tf.summary.scalar(f"diag_probe_std/{name}", float(np.std(out)), step=epoch + 1)
            self.writer.flush()


class BranchDiagnosticsCsvLogger(Callback):
    """
    Writes diagnostics to CSV so figures can be regenerated as vector plots.

    One row is written per layer and epoch.
    Drift rows store value=drift.
    Probe rows store value=mean(abs(activation)) and probe_std=std(activation).
    """

    def __init__(
        self,
        csv_path: Path,
        drift_layer_names: List[str],
        probe_layer_names: List[str],
        probe_inputs,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.drift_layer_names = drift_layer_names
        self.probe_layer_names = probe_layer_names
        self.probe_inputs = probe_inputs

        self._initial = {}
        self._present_drift_names = []
        self._submodels = {}

        self._global_step = 0
        self._round_idx = None
        self._train_size = None
        self._val_size = None

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "diag_step",
                    "round_idx",
                    "train_size",
                    "val_size",
                    "local_epoch",
                    "metric_family",
                    "layer",
                    "value",
                    "probe_std",
                ],
            )
            writer.writeheader()

    def set_round_context(
        self,
        round_idx: int,
        train_size: int,
        val_size: int,
        learning_rate: float | None = None,
    ) -> None:
        self._round_idx = round_idx
        self._train_size = train_size
        self._val_size = val_size

    def on_train_begin(self, logs=None):
        self._initial = {}
        self._present_drift_names = []

        for name in self.drift_layer_names:
            try:
                layer = self.model.get_layer(name)
            except ValueError:
                continue

            self._initial[name] = [w.numpy().copy() for w in layer.weights]
            self._present_drift_names.append(name)

        self._submodels = {}
        for name in self.probe_layer_names:
            try:
                layer = self.model.get_layer(name)
            except ValueError:
                continue

            self._submodels[name] = Model(self.model.inputs, layer.output)

    def on_epoch_end(self, epoch, logs=None):
        self._global_step += 1
        rows = []

        for name in self._present_drift_names:
            layer = self.model.get_layer(name)
            init_ws = self._initial[name]
            curr_ws = layer.weights

            sq = 0.0
            for w0, w1 in zip(init_ws, curr_ws):
                diff = w1.numpy() - w0
                sq += float(np.sum(diff * diff))

            rows.append(
                {
                    "diag_step": self._global_step,
                    "round_idx": self._round_idx,
                    "train_size": self._train_size,
                    "val_size": self._val_size,
                    "local_epoch": epoch + 1,
                    "metric_family": "drift",
                    "layer": name,
                    "value": float(np.sqrt(sq)),
                    "probe_std": "",
                }
            )

        for name, sm in self._submodels.items():
            out = sm.predict(self.probe_inputs, verbose=0)

            rows.append(
                {
                    "diag_step": self._global_step,
                    "round_idx": self._round_idx,
                    "train_size": self._train_size,
                    "val_size": self._val_size,
                    "local_epoch": epoch + 1,
                    "metric_family": "probe_meanabs",
                    "layer": name,
                    "value": float(np.mean(np.abs(out))),
                    "probe_std": float(np.std(out)),
                }
            )

        if rows:
            with self.csv_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writerows(rows)


class EpochMetricsCsvLogger(Callback):
    """
    Writes Keras per-epoch fit() metrics to CSV.

    This is intended for plots that need train/validation epoch-level
    accuracy and loss curves, not only round-level validation metrics.
    """

    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = Path(csv_path)
        self._global_epoch_step = 0
        self._round_idx = None
        self._train_size = None
        self._val_size = None
        self._learning_rate = None

        self.fieldnames = [
            "global_epoch_step",
            "round_idx",
            "train_size",
            "val_size",
            "local_epoch",
            "learning_rate",
            "loss",
            "accuracy",
            "auc",
            "mae",
            "val_loss",
            "val_accuracy",
            "val_auc",
            "val_mae",
            "output_main_loss",
            "output_main_accuracy",
            "output_main_auc",
            "output_main_mae",
            "val_output_main_loss",
            "val_output_main_accuracy",
            "val_output_main_auc",
            "val_output_main_mae",
        ]

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def set_round_context(
        self,
        round_idx: int,
        train_size: int,
        val_size: int,
        learning_rate: float | None = None,
    ) -> None:
        self._round_idx = round_idx
        self._train_size = train_size
        self._val_size = val_size
        self._learning_rate = learning_rate

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._global_epoch_step += 1

        row = {
            "global_epoch_step": self._global_epoch_step,
            "round_idx": self._round_idx,
            "train_size": self._train_size,
            "val_size": self._val_size,
            "local_epoch": epoch + 1,
            "learning_rate": self._learning_rate,
        }

        for key in self.fieldnames:
            if key in row:
                continue
            value = logs.get(key)
            row[key] = "" if value is None else float(value)

        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def _abs_diff(a, b, name: str):
    return Lambda(lambda xs: tf.abs(xs[0] - xs[1]), name=name)([a, b])


def _vec_diff(a, b, name: str):
    return Lambda(lambda xs: xs[0] - xs[1], name=name)([a, b])


def _safe_zero_vec(x, width: int, name: str):
    return Lambda(
        lambda t, d=int(width): tf.zeros((tf.shape(t)[0], d), dtype=tf.float32),
        name=name,
    )(x)


def _split_strength_tensor(x_s):
    home_vals = Lambda(lambda t: t[:, 0], name="home_strength_values")(x_s)  # (B,11,34)
    home_mask = Lambda(lambda t: t[:, 1], name="home_strength_mask")(x_s)  # (B,11,34)
    away_vals = Lambda(lambda t: t[:, 2], name="away_strength_values")(x_s)  # (B,11,34)
    away_mask = Lambda(lambda t: t[:, 3], name="away_strength_mask")(x_s)  # (B,11,34)
    return home_vals, home_mask, away_vals, away_mask


def _zero_mask_like(values, name: str):
    return Lambda(
        lambda t: tf.zeros_like(t),
        name=name,
    )(values)


def _position_embedding_or_zero(pos_ids, cfg, name_prefix: str):
    if cfg.use_position_embedding:
        position_emb_layer = Embedding(
            input_dim=len(sett.FS_PLAYER_POSITION_TO_IDX),
            output_dim=cfg.position_emb_dim,
            name="position_embedding",
        )
        return position_emb_layer(pos_ids), position_emb_layer

    pos_dim = int(cfg.position_emb_dim)
    zero_pos = Lambda(
        lambda p, d=pos_dim: tf.zeros((tf.shape(p)[0], 11, d), dtype=tf.float32),
        name=f"{name_prefix}_position_zero",
    )(pos_ids)
    return zero_pos, None


def _apply_strength_representation_v1(home_vals, home_mask, away_vals, away_mask, x_hp, x_ap, cfg):
    if not cfg.use_strength_masks:
        home_mask = _zero_mask_like(home_vals, "home_strength_mask_zero")
        away_mask = _zero_mask_like(away_vals, "away_strength_mask_zero")

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


def _row_valid_mask(mask_tensor, prefix: str):
    # Convert (B,11,34) -> (B,11,1): row is valid if at least one skill is observed.
    return Lambda(
        lambda m: tf.cast(tf.reduce_max(m, axis=-1, keepdims=True) > 0.0, tf.float32),
        name=f"{prefix}_row_valid_mask",
    )(mask_tensor)


def _role_average_pool(encoded_rows, pos_ids, row_valid_mask, role_idx: int, prefix: str):
    """
    masked average over players whose coarse position equals role_idx.
    pos_ids: (B,11)
    row_valid_mask: (B,11,1)
    encoded_rows: (B,11,H)
    returns: (B,H)
    """
    role_mask = Lambda(
        lambda p, r=int(role_idx): tf.cast(tf.equal(p, r), tf.float32)[..., None],
        name=f"{prefix}_role{role_idx}_mask",
    )(pos_ids)

    combined_mask = Lambda(
        lambda xs: xs[0] * xs[1],
        name=f"{prefix}_role{role_idx}_combined_mask",
    )([role_mask, row_valid_mask])

    masked_sum = Lambda(
        lambda xs: tf.reduce_sum(xs[0] * xs[1], axis=1),
        name=f"{prefix}_role{role_idx}_sum",
    )([encoded_rows, combined_mask])

    denom = Lambda(
        lambda m: tf.maximum(tf.reduce_sum(m, axis=1), 1e-6),
        name=f"{prefix}_role{role_idx}_denom",
    )(combined_mask)

    pooled = Lambda(
        lambda xs: xs[0] / xs[1],
        name=f"{prefix}_role{role_idx}_avg",
    )([masked_sum, denom])

    return pooled


def _build_team_repr_v2(
    team_vals,
    team_mask,
    team_pos_ids,
    position_emb_layer,
    cfg,
    prefix: str,
):
    """
    Build one team's structured representation using:
      values + mask + position embedding
      -> shared row encoder
      -> role-aware pooling (GK/DEF/MID/FWD)
      -> post-pooling projection
    """
    row_hidden = int(cfg.player_row_hidden_dim)
    role_hidden = int(cfg.role_post_hidden_dim)
    out_dim = int(cfg.strength_emb_dim)

    if position_emb_layer is not None:
        team_pos_e = position_emb_layer(team_pos_ids)
    else:
        pos_dim = int(cfg.position_emb_dim)
        team_pos_e = Lambda(
            lambda p, d=pos_dim: tf.zeros((tf.shape(p)[0], 11, d), dtype=tf.float32),
            name=f"{prefix}_position_zero",
        )(team_pos_ids)
    team_input = Concatenate(axis=-1, name=f"{prefix}_strength_concat")([team_vals, team_mask, team_pos_e])

    row_h1 = Dense(
        row_hidden,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name=f"{prefix}_row_dense_1",
    )(team_input)
    row_h2 = Dense(
        row_hidden,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name=f"{prefix}_row_dense_2",
    )(row_h1)

    row_valid = _row_valid_mask(team_mask, prefix)

    if cfg.use_position_embedding:
        gk_idx = int(sett.FS_PLAYER_POSITION_TO_IDX["Goalkeeper"])
        def_idx = int(sett.FS_PLAYER_POSITION_TO_IDX["Defender"])
        mid_idx = int(sett.FS_PLAYER_POSITION_TO_IDX["Midfielder"])
        fwd_idx = int(sett.FS_PLAYER_POSITION_TO_IDX["Forward"])

        gk_pool = _role_average_pool(row_h2, team_pos_ids, row_valid, gk_idx, prefix)
        def_pool = _role_average_pool(row_h2, team_pos_ids, row_valid, def_idx, prefix)
        mid_pool = _role_average_pool(row_h2, team_pos_ids, row_valid, mid_idx, prefix)
        fwd_pool = _role_average_pool(row_h2, team_pos_ids, row_valid, fwd_idx, prefix)

        role_cat = Concatenate(name=f"{prefix}_role_concat")([gk_pool, def_pool, mid_pool, fwd_pool])
    else:
        # fallback to global pooling if no positions in use
        role_cat = GlobalAveragePooling1D(name=f"{prefix}_global_pool_no_positions")(row_h2)

    z = Dense(
        role_hidden,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name=f"{prefix}_role_post_dense_1",
    )(role_cat)
    z = Dropout(cfg.team_dropout, name=f"{prefix}_role_post_dropout")(z)
    z = Dense(
        out_dim,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name=f"{prefix}_team_repr",
    )(z)

    return z


def _main_loss_and_metrics_for_mode(cfg: TrainConfig):
    if cfg.mode == "binary_u25":
        return "binary_crossentropy", ["accuracy", AUC(name="auc")]
    if cfg.mode == "goals_dist":
        return "sparse_categorical_crossentropy", ["accuracy"]
    if cfg.mode == "goals_reg":
        return "mae", ["mae"]
    raise ValueError(f"Unknown mode: {cfg.mode}")


def _aux_loss_and_metrics_for_task(aux_task: str):
    if aux_task == "binary_u25":
        return "binary_crossentropy", ["accuracy"]
    if aux_task == "goals_dist":
        return "sparse_categorical_crossentropy", ["accuracy"]
    if aux_task == "goals_reg":
        return "mae", ["mae"]
    raise ValueError(f"Unknown aux_task: {aux_task}")


def build_model_v1(num_num, num_teams, num_comps, cfg: TrainConfig) -> Model:
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
            home_mask = _zero_mask_like(home_vals, "home_strength_mask_zero")
            away_mask = _zero_mask_like(away_vals, "away_strength_mask_zero")

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


def build_model_v2(num_num, num_teams, num_comps, cfg: TrainConfig) -> Model:
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
    team_emb = Embedding(num_teams, cfg.team_emb_dim, name="team_embedding")
    home_e = Flatten(name="home_embedding_flat")(team_emb(x_h))
    away_e = Flatten(name="away_embedding_flat")(team_emb(x_a))

    comp_emb_layer = Embedding(num_comps, cfg.comp_emb_dim, name="competition_embedding")
    comp_e = Flatten(name="competition_embedding_flat")(comp_emb_layer(x_c))

    team_diff = _vec_diff(home_e, away_e, "team_embedding_diff")
    team_absdiff = _abs_diff(home_e, away_e, "team_embedding_absdiff")

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
    home_vals, home_mask, away_vals, away_mask = _split_strength_tensor(x_s)

    position_emb_layer = Embedding(
        input_dim=len(sett.FS_PLAYER_POSITION_TO_IDX),
        output_dim=cfg.position_emb_dim,
        name="position_embedding",
    )

    home_team_repr = _build_team_repr_v2(
        home_vals,
        home_mask,
        x_hp,
        position_emb_layer,
        cfg,
        prefix="home",
    )

    away_team_repr = _build_team_repr_v2(
        away_vals,
        away_mask,
        x_ap,
        position_emb_layer,
        cfg,
        prefix="away",
    )

    team_repr_diff = _vec_diff(home_team_repr, away_team_repr, "team_repr_diff")
    team_repr_absdiff = _abs_diff(home_team_repr, away_team_repr, "team_repr_absdiff")

    z_team = Concatenate(name="team_branch_concat")([home_team_repr, away_team_repr, team_repr_diff, team_repr_absdiff])
    z_team = Dense(
        cfg.team_branch_dim,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.team_l2),
        name="team_branch_proj",
    )(z_team)
    z_team = Dropout(cfg.team_dropout, name="team_branch_dropout")(z_team)
    z_team = LayerNormalization(name="team_branch_ln")(z_team)

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


def build_model(num_num, num_teams, num_comps, cfg: TrainConfig) -> Model:
    if cfg.model_version == "v1":
        return build_model_v1(num_num, num_teams, num_comps, cfg)
    if cfg.model_version == "v2":
        return build_model_v2(num_num, num_teams, num_comps, cfg)
    raise ValueError(f"Unknown model_version: {cfg.model_version}")


def build_strength_pretrain_model_v1(cfg: StrengthPretrainConfig) -> Model:
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


def build_strength_pretrain_model_v2(cfg: StrengthPretrainConfig) -> Model:
    """
    Standalone pretraining model using the role-aware v2/v2-lite branch design.
    """
    x_s = Input((4, 11, 34), name="strength")
    x_hp = Input((11,), dtype="int32", name="home_positions")
    x_ap = Input((11,), dtype="int32", name="away_positions")

    home_vals, home_mask, away_vals, away_mask = _split_strength_tensor(x_s)

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

    home_team_repr = _build_team_repr_v2(
        home_vals,
        home_mask,
        x_hp,
        position_emb_layer,
        cfg,
        prefix="home",
    )

    away_team_repr = _build_team_repr_v2(
        away_vals,
        away_mask,
        x_ap,
        position_emb_layer,
        cfg,
        prefix="away",
    )

    team_repr_diff = _vec_diff(home_team_repr, away_team_repr, "team_repr_diff")
    team_repr_absdiff = _abs_diff(home_team_repr, away_team_repr, "team_repr_absdiff")

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


def build_strength_pretrain_model(cfg: StrengthPretrainConfig) -> Model:
    if cfg.branch_version == "v1":
        return build_strength_pretrain_model_v1(cfg)
    if cfg.branch_version == "v2":
        return build_strength_pretrain_model_v2(cfg)
    raise ValueError(f"Unknown branch_version: {cfg.branch_version}")


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


def _multiclass_summary(y_true: np.ndarray, y_prob: np.ndarray, max_goals_class: int) -> Dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
    classes = np.arange(max_goals_class + 1)

    expected_goals = (y_prob * classes[None, :]).sum(axis=1)
    acc = float(np.mean(y_pred == y_true))
    mae = float(np.mean(np.abs(expected_goals - y_true.astype(np.float32))))

    eps = 1e-8
    clipped = np.clip(y_prob, eps, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    nll = -np.log(clipped[np.arange(len(y_true)), y_true])
    logloss = float(np.mean(nll))

    return {
        "pooled_accuracy": acc,
        "pooled_expected_goals_mae": mae,
        "pooled_log_loss": logloss,
    }


def _save_pretrain_round_plot(log_dir: str, round_records: List[dict], title: str) -> None:
    if not round_records:
        return

    xs = [r["round_idx"] for r in round_records]

    fig = plt.figure(figsize=(12, 8))

    if "val_accuracy" in round_records[0]:
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(xs, [r["val_accuracy"] for r in round_records])
        ax1.set_title("Round val accuracy")

    if "val_auc" in round_records[0]:
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(xs, [r["val_auc"] for r in round_records])
        ax2.set_title("Round val AUC")

    if "val_brier" in round_records[0]:
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(xs, [r["val_brier"] for r in round_records])
        ax3.set_title("Round val Brier")

    if "val_loss" in round_records[0]:
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(xs, [r["val_loss"] for r in round_records])
        ax4.set_title("Round val loss")

    fig.suptitle(title)
    plt.tight_layout()
    out_path = Path(log_dir) / "round_overview.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _make_train_targets(matches: List[FSMatch], y_main: np.ndarray, cfg: TrainConfig):
    if cfg.model_version == "v2" and cfg.use_team_aux_head and cfg.aux_task is not None:
        y_aux = build_aux_targets_for_matches(matches, cfg.aux_task, cfg.max_goals_class)
        return {
            "output_main": y_main,
            "output_team_aux": y_aux,
        }
    return y_main


def _extract_main_predictions(pred):
    """
    model.predict(...) returns:
      - ndarray for single-output
      - list for multi-output
    We always want the main output.
    """
    if isinstance(pred, list):
        return pred[0]
    return pred


def _lr_for_round(cfg: TrainConfig, round_offset: int, total_rounds: int) -> float:
    base = float(cfg.learning_rate)

    if cfg.lr_schedule == "constant":
        return base

    if cfg.lr_schedule == "exponential":
        lr = base * (float(cfg.lr_decay_rate) ** int(round_offset))
        return max(float(cfg.min_learning_rate), float(lr))

    if cfg.lr_schedule == "cosine":
        if total_rounds <= 1:
            return base
        progress = min(1.0, max(0.0, round_offset / float(total_rounds - 1)))
        lr = cfg.min_learning_rate + 0.5 * (base - cfg.min_learning_rate) * (1.0 + np.cos(np.pi * progress))
        return float(lr)

    raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule}")


def _set_optimizer_lr(model: Model, lr: float) -> None:
    try:
        model.optimizer.learning_rate.assign(float(lr))
    except Exception:
        tf.keras.backend.set_value(model.optimizer.learning_rate, float(lr))


def train_rolling(
    matches_sorted: List[FSMatch],
    cat_maps: CatMaps,
    cfg: TrainConfig,
    model: Model | None = None,
    pretrained_branch_version: str | None = None,
) -> Model:
    rounds = distribute_matches_into_rounds(matches_sorted)
    round_info = summarize_rounds(rounds)
    print(f"[rounds] {round_info}")

    sample_feat = matches_sorted[0].features_before_match
    num_num = extract_numerical_features(sample_feat).shape[0]

    if cfg.seed is not None:
        set_global_seed(cfg.seed)
        print(f"[seed] Using seed={cfg.seed}")

    if model is None:
        model = build_model(
            num_num=num_num,
            num_teams=len(cat_maps.team_id_map),
            num_comps=len(cat_maps.comp_id_map),
            cfg=cfg,
        )
        print("[model] built fresh model")
    else:
        print("[model] using externally prepared model")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"mlp_{cfg.mode}_{run_stamp}"

    # TensorBoard logging (always under sett.DATA_DIR)
    log_root = Path(sett.DATA_DIR) / "tensorboard_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = str(log_root / run_name)

    tb = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )
    # Extra per-round metrics: show in TensorBoard one point per rolling round
    tb_writer = tf.summary.create_file_writer(log_dir)

    print(f"[tensorboard] logging to {log_dir}")
    cfg_json_path = Path(log_dir) / "train_config.json"
    with cfg_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    round_records = []
    oos_rows = []

    # Build probe batch once (from first available training window)
    if cfg.enable_branch_diagnostics:
        probe_ms = matches_sorted[: cfg.probe_matches]
        probe_arr = build_arrays_for_matches(probe_ms, cat_maps, cfg.mode, cfg.max_goals_class)
        probe_inputs = probe_arr[:-1]  # exclude targets
    else:
        probe_inputs = None

    callbacks_common: List[Callback] = [tb]
    callbacks_common.append(EpochMetricsCsvLogger(Path(log_dir) / "epoch_metrics.csv"))
    if cfg.enable_branch_diagnostics and probe_inputs is not None:
        drift_names = []

        if cfg.model_version == "v1":
            if cfg.use_team_ids:
                drift_names.append("team_embedding")
            if cfg.use_comp_embedding:
                drift_names.append("competition_embedding")
            if cfg.use_team_strength and cfg.use_position_embedding:
                drift_names.append("position_embedding")
            if cfg.use_team_strength:
                drift_names.extend(["strength_dense_1", "strength_dense_2", "strength_projection"])

            branch_probe_layers = [
                "home_embedding_flat" if cfg.use_team_ids else "home_embedding_zero",
                "competition_embedding_flat" if cfg.use_comp_embedding else "competition_embedding_zero",
                "home_strength_embedding" if cfg.use_team_strength else "home_strength_embedding_zero",
            ]

        elif cfg.model_version == "v2":
            drift_names.extend(
                [
                    "team_embedding",
                    "competition_embedding",
                    "position_embedding",
                    "home_row_dense_1",
                    "home_row_dense_2",
                    "home_team_repr",
                    "team_branch_proj",
                ]
            )

            branch_probe_layers = [
                "home_embedding_flat",
                "competition_embedding_flat",
                "home_team_repr",
                "team_branch_proj",
            ]

        if drift_names:
            callbacks_common.append(LayerDriftLogger(drift_names, tb_writer))

        if branch_probe_layers:
            callbacks_common.append(BranchProbeLogger(probe_inputs, tb_writer, branch_probe_layers))
            callbacks_common.append(
                BranchDiagnosticsCsvLogger(
                    csv_path=Path(log_dir) / "diagnostics.csv",
                    drift_layer_names=drift_names,
                    probe_layer_names=branch_probe_layers,
                    probe_inputs=probe_inputs,
                )
            )

    frozen_branch_layer_names = None
    if cfg.freeze_pretrained_branch_rounds > 0:
        branch_version = pretrained_branch_version or cfg.model_version
        frozen_branch_layer_names = get_strength_branch_layer_names(branch_version)
        set_layers_trainable(model, frozen_branch_layer_names, False)
        compile_model_for_cfg(model, cfg)
        print(
            f"[freeze] froze pretrained branch ({branch_version}) "
            f"for first {cfg.freeze_pretrained_branch_rounds} rolling rounds"
        )

    for i in range(cfg.window_rounds, len(rounds)):
        train_ms = [m for r in rounds[i - cfg.window_rounds : i] for m in r]
        val_ms = rounds[i]

        X = build_arrays_for_matches(train_ms, cat_maps, cfg.mode, cfg.max_goals_class)
        V = build_arrays_for_matches(val_ms, cat_maps, cfg.mode, cfg.max_goals_class)

        y_train = _make_train_targets(train_ms, X[-1], cfg)
        y_val = _make_train_targets(val_ms, V[-1], cfg)

        print(f"[train] round {i+1}/{len(rounds)} train={len(train_ms)} val={len(val_ms)}")

        if len(val_ms) < cfg.min_warning_val_size:
            print(f"[warn] round {i+1} has small validation size: {len(val_ms)}")

        monitor_name = (
            "val_output_main_loss"
            if (cfg.model_version == "v2" and cfg.use_team_aux_head and cfg.aux_task is not None)
            else "val_loss"
        )

        early = EarlyStopping(
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
            restore_best_weights=True,
            monitor=monitor_name,
            mode="min",
        )

        if frozen_branch_layer_names is not None and i == cfg.window_rounds + cfg.freeze_pretrained_branch_rounds:
            set_layers_trainable(model, frozen_branch_layer_names, True)
            compile_model_for_cfg(model, cfg)
            print("[freeze] unfroze pretrained branch and recompiled model")
            frozen_branch_layer_names = None

        round_offset = i - cfg.window_rounds
        total_train_rounds = max(1, len(rounds) - cfg.window_rounds)
        current_lr = _lr_for_round(cfg, round_offset, total_train_rounds)
        _set_optimizer_lr(model, current_lr)

        for cb in callbacks_common:
            if hasattr(cb, "set_round_context"):
                cb.set_round_context(
                    round_idx=int(i + 1),
                    train_size=len(train_ms),
                    val_size=len(val_ms),
                    learning_rate=float(current_lr),
                )

        model.fit(
            X[:-1],
            y_train,
            validation_data=(V[:-1], y_val),
            epochs=cfg.epochs_per_step,
            batch_size=cfg.batch_size,
            callbacks=[early] + callbacks_common,
            verbose=1,
        )

        val_metrics = model.evaluate(V[:-1], y_val, verbose=0, return_dict=True)
        round_step = int(i + 1)  # 1-based round index for readability in TensorBoard

        if cfg.mode == "binary_u25":
            val_loss = float(val_metrics.get("output_main_loss", val_metrics.get("loss")))
            val_acc = float(val_metrics.get("output_main_accuracy", val_metrics.get("accuracy")))

            raw_pred = model.predict(V[:-1], verbose=0)
            val_prob = _extract_main_predictions(raw_pred).ravel().astype(np.float32)
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
                    "learning_rate": float(current_lr),
                    "lr_schedule": cfg.lr_schedule,
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
                tf.summary.scalar("round/learning_rate", float(current_lr), step=round_step)
                tb_writer.flush()

        elif cfg.mode == "goals_dist":
            val_loss = float(val_metrics.get("output_main_loss", val_metrics.get("loss")))
            val_acc = float(val_metrics.get("output_main_accuracy", val_metrics.get("accuracy")))

            raw_pred = model.predict(V[:-1], verbose=0)
            probabilities = _extract_main_predictions(raw_pred)
            pred_cls = np.argmax(probabilities, axis=1)
            expected = (probabilities * np.arange(cfg.max_goals_class + 1)).sum(axis=1)
            mae = np.mean(np.abs(expected - V[-1]))

            for m, yt, yp, eg in zip(val_ms, V[-1], pred_cls, expected):
                oos_rows.append(
                    {
                        "round_idx": round_step,
                        "match_id": m.id,
                        "season": m.season,
                        "competition": m.comp_name,
                        "y_true_class": int(yt),
                        "y_pred_class": int(yp),
                        "y_pred_expected_goals": float(eg),
                    }
                )

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
            val_loss = float(val_metrics.get("output_main_loss", val_metrics.get("loss")))
            val_mae = float(val_metrics.get("output_main_mae", val_metrics.get("mae")))

            raw_pred = model.predict(V[:-1], verbose=0)
            predictions = _extract_main_predictions(raw_pred).ravel()
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

    if cfg.mode == "goals_dist" and oos_rows:
        y_true = np.asarray([r["y_true_class"] for r in oos_rows], dtype=np.int32)
        y_pred_cls = np.asarray([r["y_pred_class"] for r in oos_rows], dtype=np.int32)
        y_exp = np.asarray([r["y_pred_expected_goals"] for r in oos_rows], dtype=np.float32)

        summary.update(
            {
                "pooled_accuracy": float(np.mean(y_pred_cls == y_true)),
                "pooled_expected_goals_mae": float(np.mean(np.abs(y_exp - y_true.astype(np.float32)))),
            }
        )

    summary_path = Path(log_dir) / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[metrics] saved summary to {summary_path}")
    print(f"[summary] {summary}")

    cfg_json_path = Path(log_dir) / "train_config.json"
    with cfg_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return model


def train_strength_pretrain_rolling(
    matches_sorted: List[FSMatch],
    cfg: StrengthPretrainConfig,
) -> Model:
    rounds = distribute_matches_into_rounds(matches_sorted)
    round_info = summarize_rounds(rounds)
    print(f"[pretrain-rounds] {round_info}")

    if cfg.seed is not None:
        set_global_seed(cfg.seed)
        print(f"[pretrain-seed] Using seed={cfg.seed}")

    model = build_strength_pretrain_model(cfg)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"strength_pretrain_{cfg.branch_version}_{cfg.mode}_{run_stamp}"

    log_root = Path(sett.DATA_DIR) / "tensorboard_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = str(log_root / run_name)

    tb = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )
    tb_writer = tf.summary.create_file_writer(log_dir)

    print(f"[pretrain-tensorboard] logging to {log_dir}")

    round_records = []
    oos_rows = []

    for i in range(cfg.window_rounds, len(rounds)):
        train_ms = [m for r in rounds[i - cfg.window_rounds : i] for m in r]
        val_ms = rounds[i]

        Xs_train, Xhp_train, Xap_train, y_train = build_strength_only_arrays_for_matches(
            train_ms, cfg.mode, cfg.max_goals_class
        )
        Xs_val, Xhp_val, Xap_val, y_val = build_strength_only_arrays_for_matches(val_ms, cfg.mode, cfg.max_goals_class)

        print(
            f"[pretrain] round {i+1}/{len(rounds)} "
            f"train={len(train_ms)} val={len(val_ms)} branch={cfg.branch_version}"
        )

        early = EarlyStopping(
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
            restore_best_weights=True,
            monitor="val_loss",
            mode="min",
        )

        model.fit(
            [Xs_train, Xhp_train, Xap_train],
            y_train,
            validation_data=([Xs_val, Xhp_val, Xap_val], y_val),
            epochs=cfg.epochs_per_step,
            batch_size=cfg.batch_size,
            callbacks=[early, tb],
            verbose=1,
        )

        val_metrics = model.evaluate([Xs_val, Xhp_val, Xap_val], y_val, verbose=0, return_dict=True)
        val_prob = model.predict([Xs_val, Xhp_val, Xap_val], verbose=0).ravel().astype(np.float32)

        auc_metric = AUC(curve="ROC")
        auc_metric.update_state(y_val.astype(np.float32), val_prob)
        val_auc = float(auc_metric.result().numpy())
        val_brier = float(np.mean((val_prob - y_val.astype(np.float32)) ** 2))
        val_acc = float(np.mean((val_prob >= 0.5).astype(np.float32) == y_val.astype(np.float32)))
        val_loss = float(val_metrics.get("loss", np.nan))

        round_step = int(i + 1)

        round_records.append(
            {
                "round_idx": round_step,
                "train_size": len(train_ms),
                "val_size": len(val_ms),
                "positive_rate_val": float(np.mean(y_val)),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_auc,
                "val_brier": val_brier,
                "branch_version": cfg.branch_version,
                "representation": cfg.representation,
                "use_strength_masks": bool(cfg.use_strength_masks),
                "use_position_embedding": bool(cfg.use_position_embedding),
            }
        )

        for m, yt, yp in zip(val_ms, y_val, val_prob):
            oos_rows.append(
                {
                    "round_idx": round_step,
                    "match_id": m.id,
                    "season": m.season,
                    "competition": m.comp_name,
                    "y_true": float(yt),
                    "y_prob_under25": float(yp),
                    "branch_version": cfg.branch_version,
                    "representation": cfg.representation,
                }
            )

        with tb_writer.as_default():
            tf.summary.scalar("round/val_loss", val_loss, step=round_step)
            tf.summary.scalar("round/val_accuracy", val_acc, step=round_step)
            tf.summary.scalar("round/val_auc", val_auc, step=round_step)
            tf.summary.scalar("round/val_brier", val_brier, step=round_step)
            tf.summary.scalar("round/val_size", len(val_ms), step=round_step)
            tf.summary.scalar("round/positive_rate_val", float(np.mean(y_val)), step=round_step)
            tb_writer.flush()

    csv_path = Path(log_dir) / "round_metrics.csv"
    if round_records:
        fieldnames = sorted({k for rec in round_records for k in rec.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(round_records)

    if cfg.save_oos_predictions and oos_rows:
        pred_path = Path(log_dir) / "oos_predictions.csv"
        fieldnames = sorted({k for rec in oos_rows for k in rec.keys()})
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(oos_rows)

    summary = {
        "run_name": run_name,
        "branch_version": cfg.branch_version,
        "mode": cfg.mode,
        "representation": cfg.representation,
        "use_strength_masks": bool(cfg.use_strength_masks),
        "use_position_embedding": bool(cfg.use_position_embedding),
        "round_stats": round_info,
    }

    if oos_rows:
        y_true = np.asarray([r["y_true"] for r in oos_rows], dtype=np.float32)
        y_prob = np.asarray([r["y_prob_under25"] for r in oos_rows], dtype=np.float32)
        summary.update(_binary_summary(y_true, y_prob))

    summary_path = Path(log_dir) / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    cfg_json_path = Path(log_dir) / "pretrain_config.json"
    with cfg_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    _save_pretrain_round_plot(
        log_dir=log_dir,
        round_records=round_records,
        title=f"Structured branch pretraining ({cfg.branch_version})",
    )

    model_path = Path(log_dir) / "pretrained_model.keras"
    model.save(model_path)

    print(f"[pretrain-summary] {summary}")
    print(f"[pretrain] model saved to {model_path}")

    return model


def get_strength_branch_layer_names(branch_version: str) -> List[str]:
    if branch_version == "v1":
        return [
            "position_embedding",
            "strength_dense_1",
            "strength_dense_2",
            "strength_projection",
        ]
    elif branch_version == "v2":
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
    else:
        raise ValueError(f"Unknown branch_version: {branch_version}")


def transfer_pretrained_strength_branch_weights(
    pretrained_model: Model,
    full_model: Model,
    branch_version: str,
) -> None:
    """
    Copy branch weights by layer name from standalone pretraining model
    into the corresponding full model.
    """
    layer_names = get_strength_branch_layer_names(branch_version)

    for name in layer_names:
        try:
            src = pretrained_model.get_layer(name)
            dst = full_model.get_layer(name)
        except ValueError:
            print(f"[transfer] skipping missing layer: {name}")
            continue

        dst.set_weights(src.get_weights())
        print(f"[transfer] copied layer: {name}")

    print(f"[transfer] copied pretrained {branch_version} branch weights into full model")


def set_layers_trainable(model: Model, layer_names: List[str], trainable: bool) -> None:
    for name in layer_names:
        try:
            layer = model.get_layer(name)
            layer.trainable = trainable
        except ValueError:
            continue


def compile_model_for_cfg(model: Model, cfg: TrainConfig) -> None:
    """
    Recompile an existing model after changing trainable flags.
    Mirrors the compilation logic used by build_model_v1/build_model_v2.
    """
    if cfg.model_version == "v1":
        if cfg.mode == "binary_u25":
            model.compile(
                optimizer=Adam(learning_rate=cfg.learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        elif cfg.mode == "goals_dist":
            model.compile(
                optimizer=Adam(learning_rate=cfg.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        elif cfg.mode == "goals_reg":
            model.compile(
                optimizer=Adam(learning_rate=cfg.learning_rate),
                loss="mae",
                metrics=["mae"],
            )
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    elif cfg.model_version == "v2":
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
    else:
        raise ValueError(f"Unknown model_version: {cfg.model_version}")
