from types import SimpleNamespace

import numpy as np
import pytest

from football_outcomes.training import (
    control,
    train_mlp_rolling,
)


def make_config(
    schedule: str,
):
    return SimpleNamespace(
        learning_rate=0.1,
        lr_schedule=schedule,
        lr_decay_rate=0.5,
        min_learning_rate=0.02,
    )


def test_learning_rate_schedules() -> None:
    constant = make_config("constant")

    assert control.learning_rate_for_round(
        constant,
        round_offset=3,
        total_rounds=5,
    ) == pytest.approx(0.1)

    exponential = make_config("exponential")

    assert control.learning_rate_for_round(
        exponential,
        round_offset=3,
        total_rounds=5,
    ) == pytest.approx(0.02)

    cosine = make_config("cosine")

    assert control.learning_rate_for_round(
        cosine,
        round_offset=0,
        total_rounds=5,
    ) == pytest.approx(0.1)
    assert control.learning_rate_for_round(
        cosine,
        round_offset=4,
        total_rounds=5,
    ) == pytest.approx(0.02)


def test_unknown_learning_rate_schedule_is_rejected() -> None:
    config = make_config("unsupported")

    with pytest.raises(
        ValueError,
        match="Unknown lr_schedule",
    ):
        control.learning_rate_for_round(
            config,
            round_offset=0,
            total_rounds=5,
        )


def test_strength_branch_layer_names() -> None:
    assert control.get_strength_branch_layer_names("v1") == [
        "position_embedding",
        "strength_dense_1",
        "strength_dense_2",
        "strength_projection",
    ]

    v2_names = control.get_strength_branch_layer_names("v2")

    assert "home_team_repr" in v2_names
    assert "away_team_repr" in v2_names
    assert "team_branch_proj" in v2_names

    with pytest.raises(
        ValueError,
        match="Unknown branch_version",
    ):
        control.get_strength_branch_layer_names("unsupported")


def test_legacy_exports_are_direct_aliases() -> None:
    assert train_mlp_rolling._lr_for_round is control.learning_rate_for_round
    assert train_mlp_rolling._set_optimizer_lr is control.set_optimizer_learning_rate
    assert train_mlp_rolling.get_strength_branch_layer_names is control.get_strength_branch_layer_names
    assert (
        train_mlp_rolling.transfer_pretrained_strength_branch_weights
        is control.transfer_pretrained_strength_branch_weights
    )
    assert train_mlp_rolling.set_layers_trainable is control.set_layers_trainable


class FakeLayer:
    def __init__(self, weights) -> None:
        self._weights = weights
        self.trainable = True

    def get_weights(self):
        return self._weights

    def set_weights(self, weights) -> None:
        self._weights = weights


class FakeModel:
    def __init__(self, layers) -> None:
        self.layers = layers

    def get_layer(self, name):
        if name not in self.layers:
            raise ValueError(name)

        return self.layers[name]


def test_transfer_and_trainable_helpers() -> None:
    source_layer = FakeLayer([np.asarray([2.0])])
    destination_layer = FakeLayer([np.asarray([0.0])])

    source = FakeModel({"position_embedding": (source_layer)})
    destination = FakeModel({"position_embedding": (destination_layer)})

    control.transfer_pretrained_strength_branch_weights(
        pretrained_model=source,
        full_model=destination,
        branch_version="v1",
    )

    np.testing.assert_array_equal(
        destination_layer.get_weights()[0],
        np.asarray([2.0]),
    )

    control.set_layers_trainable(
        destination,
        (
            "position_embedding",
            "missing_layer",
        ),
        False,
    )

    assert destination_layer.trainable is False
