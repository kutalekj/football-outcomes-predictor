from __future__ import annotations

import os
import random
from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from football_outcomes.data.fs_models import (
    FSMatch,
)
from football_outcomes.datasets.targets import (
    build_targets_for_matches,
)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        (tf.config.experimental.enable_op_determinism())
    except Exception:
        pass


def make_train_targets(
    matches: Sequence[FSMatch],
    y_main: np.ndarray,
    cfg,
):
    """
    Package targets for single-output or
    auxiliary-head training.
    """

    if cfg.model_version == "v2" and cfg.use_team_aux_head and cfg.aux_task is not None:
        y_auxiliary = build_targets_for_matches(
            matches=matches,
            mode=cfg.aux_task,
            max_goals_class=(cfg.max_goals_class),
        )

        return {
            "output_main": y_main,
            "output_team_aux": (y_auxiliary),
        }

    return y_main


def extract_main_predictions(
    predictions,
):
    """
    Return the main prediction output.

    Keras returns an ndarray for one output and
    a list for multiple outputs.
    """

    if isinstance(predictions, list):
        return predictions[0]

    return predictions
