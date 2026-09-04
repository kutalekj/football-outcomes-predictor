from __future__ import annotations

import numpy as np
import pytest

from football_outcomes.experiments.publication_binary import (
    PublicationBinaryConfig,
    _selected_v1_train_config,
    _validate_numerical_unit_interval,
)


def test_publication_binary_reproduction_defaults() -> None:
    config = PublicationBinaryConfig()

    assert config.rebuild_match_features is True
    assert config.enable_strength_imputation is False


def test_selected_v1_config_reuses_authoritative_application_values() -> None:
    config = PublicationBinaryConfig()
    train_config = _selected_v1_train_config(config)

    assert train_config.mode == "binary_u25"
    assert train_config.model_version == "v1"
    assert train_config.strength_emb_dim == 24
    assert train_config.mlp_hidden_1 == 128
    assert train_config.mlp_hidden_2 == 64
    assert train_config.mlp_hidden_3 == 32
    assert train_config.mlp_dropout_1 == pytest.approx(0.30)
    assert train_config.mlp_dropout_2 == pytest.approx(0.20)
    assert train_config.lr_schedule == "exponential"
    assert train_config.lr_decay_rate == pytest.approx(0.997)
    assert train_config.enable_strength_imputation is False


def test_numerical_unit_interval_gate_accepts_normalized_matrix() -> None:
    summary = _validate_numerical_unit_interval(
        np.asarray([[0.0, 0.25, 0.5], [0.75, 1.0, 1e-6]], dtype=np.float32),
        name="test",
    )

    assert summary["minimum"] == pytest.approx(0.0)
    assert summary["maximum"] == pytest.approx(1.0)


def test_numerical_unit_interval_gate_rejects_legacy_season_scale() -> None:
    with pytest.raises(RuntimeError, match=r"must lie in \[0, 1\]"):
        _validate_numerical_unit_interval(
            np.asarray([[2021.0, 0.5, 0.5]], dtype=np.float32),
            name="legacy",
        )
