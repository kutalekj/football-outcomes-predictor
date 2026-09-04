from __future__ import annotations

import numpy as np
import pytest

from football_outcomes.experiments.publication_binary import (
    BinaryEstimatorConfig,
    PublicationBinaryConfig,
    binary_metrics,
    choose_publication_fold_indices,
    fit_flat_mlp_probabilities,
    fit_logistic_probabilities,
    fit_random_forest_probabilities,
    fit_xgboost_probabilities,
)


def _classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    training = rng.normal(size=(80, 6)).astype(np.float32)
    target = (training[:, 0] + 0.7 * training[:, 1] > 0.0).astype(np.int32)
    validation = rng.normal(size=(20, 6)).astype(np.float32)
    return training, target, validation


def test_choose_publication_fold_indices_full_and_partial() -> None:
    full = choose_publication_fold_indices(
        round_count=320,
        window_rounds=25,
        fold_count=None,
        start_fold_offset=0,
    )
    assert len(full) == 295
    assert full[0] == 25
    assert full[-1] == 319

    partial = choose_publication_fold_indices(
        round_count=320,
        window_rounds=25,
        fold_count=3,
        start_fold_offset=4,
    )
    assert partial == (29, 30, 31)


def test_publication_binary_config_validation() -> None:
    PublicationBinaryConfig().validate()
    with pytest.raises(ValueError, match="fold_count"):
        PublicationBinaryConfig(fold_count=0).validate()


def test_binary_metrics() -> None:
    metrics = binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.4, 0.6, 0.9]),
    )
    assert metrics["prediction_count"] == 4
    assert metrics["accuracy_at_0_5"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_logistic_random_forest_and_mlp_probabilities() -> None:
    X_train, y_train, X_validation = _classification_data()
    config = BinaryEstimatorConfig(
        random_forest_estimators=20,
        mlp_hidden_layers=(12, 6),
        mlp_max_iter=10,
        mlp_batch_size=16,
    )
    for result in (
        fit_logistic_probabilities(X_train, y_train, X_validation, config),
        fit_random_forest_probabilities(X_train, y_train, X_validation, config),
        fit_flat_mlp_probabilities(X_train, y_train, X_validation, config),
    ):
        assert result.probabilities.shape == (20,)
        assert np.isfinite(result.probabilities).all()
        assert np.all(result.probabilities >= 0.0)
        assert np.all(result.probabilities <= 1.0)
        assert result.feature_count == 6


def test_xgboost_probabilities_when_available() -> None:
    pytest.importorskip("xgboost")
    X_train, y_train, X_validation = _classification_data()
    config = BinaryEstimatorConfig(
        xgboost_estimators=12,
        xgboost_max_depth=2,
        xgboost_n_jobs=1,
    )
    result = fit_xgboost_probabilities(X_train, y_train, X_validation, config)
    assert result.probabilities.shape == (20,)
    assert np.isfinite(result.probabilities).all()


def test_single_class_training_falls_back_to_constant() -> None:
    X_train = np.ones((12, 4), dtype=np.float32)
    y_train = np.ones(12, dtype=np.int32)
    X_validation = np.zeros((3, 4), dtype=np.float32)
    result = fit_logistic_probabilities(
        X_train,
        y_train,
        X_validation,
        BinaryEstimatorConfig(),
    )
    assert result.fit_status == "single-class-constant"
    assert result.probabilities.tolist() == [1.0, 1.0, 1.0]
