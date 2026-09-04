from __future__ import annotations

import numpy as np
import pytest

from football_outcomes.experiments.publication_representations import (
    PrelearningRepresentationLayout,
    build_final_hidden_extractor,
    build_prelearning_flat_representation,
    resolve_main_output_layer,
)


def _arrays(*, rows: int = 2, num_features: int = 3):
    x_num = np.arange(rows * num_features, dtype=np.float32).reshape(rows, num_features) / 100.0
    x_home = np.arange(rows, dtype=np.int32)[:, None]
    x_away = np.arange(rows, dtype=np.int32)[::-1, None]
    x_comp = np.zeros((rows, 1), dtype=np.int32)
    x_strength = np.zeros((rows, 4, 11, 34), dtype=np.float32)
    x_strength[:, 0, :, :] = 0.5
    x_strength[:, 1, :, :] = 1.0
    x_home_pos = np.tile(np.arange(11, dtype=np.int32) % 5, (rows, 1))
    x_away_pos = np.tile(np.arange(11, dtype=np.int32)[::-1] % 5, (rows, 1))
    y = np.arange(rows, dtype=np.float32) % 2
    return (
        x_num,
        x_home,
        x_away,
        x_comp,
        x_strength,
        x_home_pos,
        x_away_pos,
        y,
    )


def test_prelearning_representation_has_expected_layout() -> None:
    arrays = _arrays()

    matrix, layout = build_prelearning_flat_representation(
        arrays,
        team_count=2,
        competition_count=1,
        position_count=5,
    )

    assert isinstance(layout, PrelearningRepresentationLayout)
    assert layout.total_features == 3 + 2 + 2 + 1 + (4 * 11 * 34) + 55 + 55
    assert matrix.shape == (2, layout.total_features)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()

    slices = layout.group_slices
    assert np.allclose(
        matrix[:, slices["numerical"]],
        arrays[0],
    )
    assert matrix[:, slices["home_team_one_hot"]].sum(axis=1).tolist() == [
        1.0,
        1.0,
    ]
    assert matrix[:, slices["away_team_one_hot"]].sum(axis=1).tolist() == [
        1.0,
        1.0,
    ]
    assert matrix[:, slices["competition_one_hot"]].sum(axis=1).tolist() == [
        1.0,
        1.0,
    ]
    assert matrix[:, slices["home_positions_one_hot"]].sum(axis=1).tolist() == [
        11.0,
        11.0,
    ]
    assert matrix[:, slices["away_positions_one_hot"]].sum(axis=1).tolist() == [
        11.0,
        11.0,
    ]


def test_prelearning_representation_accepts_inputs_without_target() -> None:
    arrays = _arrays()

    with_target, _ = build_prelearning_flat_representation(
        arrays,
        team_count=2,
        competition_count=1,
        position_count=5,
    )
    without_target, _ = build_prelearning_flat_representation(
        arrays[:-1],
        team_count=2,
        competition_count=1,
        position_count=5,
    )

    np.testing.assert_array_equal(with_target, without_target)


def test_prelearning_representation_rejects_out_of_range_ids() -> None:
    arrays = list(_arrays())
    arrays[1] = np.asarray([[0], [2]], dtype=np.int32)

    with pytest.raises(ValueError, match="home team IDs must lie"):
        build_prelearning_flat_representation(
            arrays,
            team_count=2,
            competition_count=1,
            position_count=5,
        )


def test_prelearning_representation_rejects_wrong_strength_shape() -> None:
    arrays = list(_arrays())
    arrays[4] = np.zeros((2, 2, 11, 34), dtype=np.float32)

    with pytest.raises(ValueError, match="Strength tensor must have shape"):
        build_prelearning_flat_representation(
            arrays,
            team_count=2,
            competition_count=1,
            position_count=5,
        )


def test_resolve_main_output_layer_prefers_known_binary_name() -> None:
    known = object()
    fallback = object()

    class FakeModel:
        outputs = [object()]
        layers = [fallback]

        @staticmethod
        def get_layer(name: str):
            if name == "output_binary":
                return known
            raise ValueError(name)

    assert resolve_main_output_layer(FakeModel()) is known


def test_final_hidden_extractor_returns_output_layer_input() -> None:
    tf = pytest.importorskip("tensorflow")

    x_num = tf.keras.Input((4,), name="num")
    latent = tf.keras.layers.Dense(3, activation="relu", name="mlp_dense_3")(x_num)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output_binary")(latent)
    model = tf.keras.Model(x_num, output, name="synthetic_v1")

    extractor = build_final_hidden_extractor(model)

    assert extractor.output_shape == (None, 3)
    assert extractor.output.name.split("/")[0].startswith("mlp_dense_3")
