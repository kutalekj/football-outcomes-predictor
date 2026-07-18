from __future__ import annotations

import numpy as np
from tensorflow.keras.metrics import AUC


def binary_summary(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Summarize pooled binary predictions."""

    predicted_class = (y_probability >= 0.5).astype(np.float32)

    accuracy = float(np.mean(predicted_class == y_true))
    brier = float(np.mean((y_probability - y_true) ** 2))

    auc_metric = AUC(curve="ROC")
    auc_metric.update_state(
        y_true.astype(np.float32),
        y_probability.astype(np.float32),
    )
    auc = float(auc_metric.result().numpy())

    return {
        "pooled_accuracy": accuracy,
        "pooled_brier": brier,
        "pooled_auc": auc,
    }


def regression_summary(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> dict[str, float]:
    """Summarize pooled goal-regression predictions."""

    mae = float(np.mean(np.abs(y_prediction - y_true)))
    rmse = float(np.sqrt(np.mean((y_prediction - y_true) ** 2)))

    return {
        "pooled_mae": mae,
        "pooled_rmse": rmse,
    }


def multiclass_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    max_goals_class: int,
) -> dict[str, float]:
    """Summarize pooled clipped goal-count predictions."""

    y_true = y_true.astype(np.int32)
    predicted_class = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int32)

    classes = np.arange(max_goals_class + 1)
    expected_goals = (probabilities * classes[None, :]).sum(axis=1)

    accuracy = float(np.mean(predicted_class == y_true))
    expected_goals_mae = float(np.mean(np.abs(expected_goals - y_true.astype(np.float32))))

    epsilon = 1e-8
    clipped = np.clip(
        probabilities,
        epsilon,
        1.0,
    )
    clipped = clipped / clipped.sum(
        axis=1,
        keepdims=True,
    )

    negative_log_likelihood = -np.log(
        clipped[
            np.arange(len(y_true)),
            y_true,
        ]
    )
    log_loss = float(np.mean(negative_log_likelihood))

    return {
        "pooled_accuracy": accuracy,
        "pooled_expected_goals_mae": (expected_goals_mae),
        "pooled_log_loss": log_loss,
    }
