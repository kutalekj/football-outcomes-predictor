from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, TensorBoard
from tensorflow.keras.layers import (
    Embedding,
    Lambda,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.datasets.arrays import (
    build_arrays_for_matches,
    build_strength_only_arrays_for_matches,
    extract_numerical_features,
)
from football_outcomes.datasets.mappings import CatMaps
from football_outcomes.datasets.rounds import (
    distribute_matches_into_rounds,
    summarize_rounds,
)
from football_outcomes.evaluation import metrics as _evaluation_metrics
from football_outcomes.evaluation.persistence import (
    write_json,
    write_records_csv,
)
from football_outcomes.modeling import compilation as _compilation
from football_outcomes.modeling.strength_pretraining import (
    build_strength_pretrain_model as build_strength_pretrain_model_impl,
)
from football_outcomes.modeling.strength_pretraining import (
    build_strength_pretrain_model_v1 as build_strength_pretrain_model_v1_impl,
)
from football_outcomes.modeling.strength_pretraining import (
    build_strength_pretrain_model_v2 as build_strength_pretrain_model_v2_impl,
)
from football_outcomes.modeling.v1 import build_model_v1 as build_model_v1_impl
from football_outcomes.modeling.v2 import build_model_v2 as build_model_v2_impl
from football_outcomes.training import callbacks as _training_callbacks
from football_outcomes.training import control as _control
from football_outcomes.training import runtime as _training_runtime

matplotlib.use("Agg")

# Compatibility exports during the incremental refactor.
_lr_for_round = _control.learning_rate_for_round
_set_optimizer_lr = _control.set_optimizer_learning_rate
get_strength_branch_layer_names = _control.get_strength_branch_layer_names
transfer_pretrained_strength_branch_weights = _control.transfer_pretrained_strength_branch_weights
set_layers_trainable = _control.set_layers_trainable
compile_model_for_cfg = _compilation.compile_model_for_config
_binary_summary = _evaluation_metrics.binary_summary
_reg_summary = _evaluation_metrics.regression_summary
_multiclass_summary = _evaluation_metrics.multiclass_summary
LayerDriftLogger = _training_callbacks.LayerDriftLogger
BranchProbeLogger = _training_callbacks.BranchProbeLogger
BranchDiagnosticsCsvLogger = _training_callbacks.BranchDiagnosticsCsvLogger
EpochMetricsCsvLogger = _training_callbacks.EpochMetricsCsvLogger
set_global_seed = _training_runtime.set_global_seed
_make_train_targets = _training_runtime.make_train_targets
_extract_main_predictions = _training_runtime.extract_main_predictions


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


def build_model_v1(
    num_num,
    num_teams,
    num_comps,
    cfg: TrainConfig,
) -> Model:
    """Compatibility wrapper for the extracted v1 builder."""

    return build_model_v1_impl(
        num_num=num_num,
        num_teams=num_teams,
        num_comps=num_comps,
        cfg=cfg,
    )


def build_model_v2(
    num_num,
    num_teams,
    num_comps,
    cfg: TrainConfig,
) -> Model:
    """Compatibility wrapper for the extracted v2 builder."""

    return build_model_v2_impl(
        num_num=num_num,
        num_teams=num_teams,
        num_comps=num_comps,
        cfg=cfg,
    )


def build_model(num_num, num_teams, num_comps, cfg: TrainConfig) -> Model:
    if cfg.model_version == "v1":
        return build_model_v1(num_num, num_teams, num_comps, cfg)
    if cfg.model_version == "v2":
        return build_model_v2(num_num, num_teams, num_comps, cfg)
    raise ValueError(f"Unknown model_version: {cfg.model_version}")


def build_strength_pretrain_model_v1(
    cfg: StrengthPretrainConfig,
) -> Model:
    return build_strength_pretrain_model_v1_impl(cfg)


def build_strength_pretrain_model_v2(
    cfg: StrengthPretrainConfig,
) -> Model:
    return build_strength_pretrain_model_v2_impl(cfg)


def build_strength_pretrain_model(
    cfg: StrengthPretrainConfig,
) -> Model:
    return build_strength_pretrain_model_impl(cfg)


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


def train_rolling(
    matches_sorted: List[FSMatch],
    cat_maps: CatMaps,
    cfg: TrainConfig,
    model: Model | None = None,
    pretrained_branch_version: str | None = None,
    competition_names: Sequence[str] | None = None,
) -> Model:
    if competition_names is None:
        competition_names = sett.COMPS_LEAGUE

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
    write_json(
        cfg_json_path,
        asdict(cfg),
    )

    round_records = []
    oos_rows = []

    # Build probe batch once (from first available training window)
    if cfg.enable_branch_diagnostics:
        probe_ms = matches_sorted[: cfg.probe_matches]
        probe_arr = build_arrays_for_matches(
            matches=probe_ms,
            cat_maps=cat_maps,
            competition_names=competition_names,
            mode=cfg.mode,
            max_goals_class=(cfg.max_goals_class),
        )
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

        X = build_arrays_for_matches(
            matches=train_ms,
            cat_maps=cat_maps,
            competition_names=competition_names,
            mode=cfg.mode,
            max_goals_class=(cfg.max_goals_class),
        )
        V = build_arrays_for_matches(
            matches=val_ms,
            cat_maps=cat_maps,
            competition_names=competition_names,
            mode=cfg.mode,
            max_goals_class=(cfg.max_goals_class),
        )

        y_train = _make_train_targets(train_ms, X[-1], cfg)
        y_val = _make_train_targets(val_ms, V[-1], cfg)

        print(f"[train] round {i + 1}/{len(rounds)} train={len(train_ms)} val={len(val_ms)}")

        if len(val_ms) < cfg.min_warning_val_size:
            print(f"[warn] round {i + 1} has small validation size: {len(val_ms)}")

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

    if write_records_csv(
        csv_path,
        round_records,
    ):
        print("[metrics] saved round-level " f"metrics to {csv_path}")

    if cfg.save_oos_predictions:
        prediction_path = Path(log_dir) / "oos_predictions.csv"

        if write_records_csv(
            prediction_path,
            oos_rows,
        ):
            print("[metrics] saved pooled OOS " "predictions to " f"{prediction_path}")

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
    write_json(
        summary_path,
        summary,
    )
    print(f"[metrics] saved summary to {summary_path}")
    print(f"[summary] {summary}")

    cfg_json_path = Path(log_dir) / "train_config.json"
    write_json(
        cfg_json_path,
        asdict(cfg),
    )

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
            f"[pretrain] round {i + 1}/{len(rounds)} "
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

    write_records_csv(
        csv_path,
        round_records,
    )

    if cfg.save_oos_predictions:
        prediction_path = Path(log_dir) / "oos_predictions.csv"

        if write_records_csv(
            prediction_path,
            oos_rows,
        ):
            print("[metrics] saved pooled OOS " "predictions to " f"{prediction_path}")

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
    write_json(
        summary_path,
        summary,
    )

    cfg_json_path = Path(log_dir) / "pretrain_config.json"
    write_json(
        cfg_json_path,
        asdict(cfg),
    )

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
