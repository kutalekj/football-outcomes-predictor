from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding,
    Lambda,
)
from tensorflow.keras.models import Model

from football_outcomes.config import fs_settings as sett
from football_outcomes.evaluation import metrics as _evaluation_metrics
from football_outcomes.evaluation import plots as _evaluation_plots
from football_outcomes.modeling import compilation as _compilation
from football_outcomes.modeling import factory as _model_factory
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
from football_outcomes.training import pretraining as _pretraining
from football_outcomes.training import rolling as _rolling
from football_outcomes.training import runtime as _training_runtime
from football_outcomes.training.configs import (
    StrengthPretrainConfig,
    TrainConfig,
)

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
_save_pretrain_round_plot = _evaluation_plots.save_pretrain_round_plot
train_strength_pretrain_rolling = _pretraining.train_strength_pretrain_rolling
build_model = _model_factory.build_model
train_rolling = _rolling.train_rolling


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
