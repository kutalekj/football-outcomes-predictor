from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from football_outcomes.config import fs_settings as sett


@dataclass(frozen=True)
class PrelearningRepresentationLayout:
    """Column layout of the model-independent flat pre-learning representation."""

    numerical_features: int
    team_count: int
    competition_count: int
    position_count: int
    strength_shape: tuple[int, int, int] = (4, 11, 34)

    def __post_init__(self) -> None:
        for name in (
            "numerical_features",
            "team_count",
            "competition_count",
            "position_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if self.strength_shape != (4, 11, 34):
            raise ValueError("strength_shape must be exactly (4, 11, 34).")

    @property
    def strength_features(self) -> int:
        return int(np.prod(self.strength_shape))

    @property
    def position_features_per_team(self) -> int:
        return 11 * self.position_count

    @property
    def total_features(self) -> int:
        return (
            self.numerical_features
            + self.team_count
            + self.team_count
            + self.competition_count
            + self.strength_features
            + self.position_features_per_team
            + self.position_features_per_team
        )

    @property
    def group_slices(self) -> dict[str, slice]:
        start = 0
        groups: dict[str, slice] = {}

        widths = (
            ("numerical", self.numerical_features),
            ("home_team_one_hot", self.team_count),
            ("away_team_one_hot", self.team_count),
            ("competition_one_hot", self.competition_count),
            ("strength", self.strength_features),
            ("home_positions_one_hot", self.position_features_per_team),
            ("away_positions_one_hot", self.position_features_per_team),
        )

        for name, width in widths:
            groups[name] = slice(start, start + width)
            start += width

        return groups


def _as_integer_ids(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,) or (n, 1).")

    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.allclose(array, rounded):
            raise ValueError(f"{name} must contain integer IDs.")
        array = rounded

    return array.astype(np.int64, copy=False)


def _validate_id_range(values: np.ndarray, *, depth: int, name: str) -> None:
    if values.size == 0:
        return
    minimum = int(values.min())
    maximum = int(values.max())
    if minimum < 0 or maximum >= depth:
        raise ValueError(f"{name} IDs must lie in [0, {depth - 1}], " f"found range [{minimum}, {maximum}].")


def _one_hot(values: np.ndarray, *, depth: int) -> np.ndarray:
    result = np.zeros((len(values), depth), dtype=np.float32)
    if len(values):
        result[np.arange(len(values)), values] = 1.0
    return result


def _positions_one_hot(values: np.ndarray, *, depth: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 11:
        raise ValueError(f"{name} must have shape (n, 11).")

    ids = _as_integer_ids(array.reshape(-1), name=name)
    _validate_id_range(ids, depth=depth, name=name)
    encoded = _one_hot(ids, depth=depth)
    return encoded.reshape(len(array), 11 * depth)


def _model_inputs(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    if len(arrays) not in (7, 8):
        raise ValueError("Expected seven model-input arrays, optionally followed by one target array.")
    return tuple(np.asarray(array) for array in arrays[:7])


def build_prelearning_flat_representation(
    arrays: Sequence[np.ndarray],
    *,
    team_count: int,
    competition_count: int,
    position_count: int | None = None,
) -> tuple[np.ndarray, PrelearningRepresentationLayout]:
    """Flatten the same pre-match inputs before any trainable model branch.

    Categorical team/competition IDs and player-position IDs are represented by
    deterministic one-hot blocks. The strength tensor is already numerical and
    contains its value/mask channels, so it is flattened without modification.
    """

    if type(team_count) is not int or team_count <= 0:
        raise ValueError("team_count must be a positive integer.")
    if type(competition_count) is not int or competition_count <= 0:
        raise ValueError("competition_count must be a positive integer.")

    resolved_position_count = len(sett.FS_PLAYER_POSITION_TO_IDX) if position_count is None else position_count
    if type(resolved_position_count) is not int or resolved_position_count <= 0:
        raise ValueError("position_count must be a positive integer.")

    x_num, x_home, x_away, x_comp, x_strength, x_home_pos, x_away_pos = _model_inputs(arrays)

    if x_num.ndim != 2:
        raise ValueError("Numerical features must have shape (n, d).")

    row_count = int(x_num.shape[0])
    if row_count == 0:
        raise ValueError("At least one row is required.")

    for name, array in (
        ("home team IDs", x_home),
        ("away team IDs", x_away),
        ("competition IDs", x_comp),
        ("strength tensor", x_strength),
        ("home positions", x_home_pos),
        ("away positions", x_away_pos),
    ):
        if len(array) != row_count:
            raise ValueError(f"{name} row count does not match numerical-feature row count.")

    if x_strength.shape != (row_count, 4, 11, 34):
        raise ValueError("Strength tensor must have shape (n, 4, 11, 34); " f"found {x_strength.shape}.")

    if not np.isfinite(x_num).all():
        raise ValueError("Numerical features must be finite.")
    if not np.isfinite(x_strength).all():
        raise ValueError("Strength inputs must be finite.")

    home_ids = _as_integer_ids(x_home, name="home team")
    away_ids = _as_integer_ids(x_away, name="away team")
    comp_ids = _as_integer_ids(x_comp, name="competition")

    _validate_id_range(home_ids, depth=team_count, name="home team")
    _validate_id_range(away_ids, depth=team_count, name="away team")
    _validate_id_range(comp_ids, depth=competition_count, name="competition")

    layout = PrelearningRepresentationLayout(
        numerical_features=int(x_num.shape[1]),
        team_count=team_count,
        competition_count=competition_count,
        position_count=resolved_position_count,
    )

    matrix = np.concatenate(
        (
            x_num.astype(np.float32, copy=False),
            _one_hot(home_ids, depth=team_count),
            _one_hot(away_ids, depth=team_count),
            _one_hot(comp_ids, depth=competition_count),
            x_strength.astype(np.float32, copy=False).reshape(row_count, -1),
            _positions_one_hot(
                x_home_pos,
                depth=resolved_position_count,
                name="home positions",
            ),
            _positions_one_hot(
                x_away_pos,
                depth=resolved_position_count,
                name="away positions",
            ),
        ),
        axis=1,
    ).astype(np.float32, copy=False)

    if matrix.shape != (row_count, layout.total_features):
        raise RuntimeError("Pre-learning representation shape does not match its declared layout.")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Pre-learning representation contains non-finite values.")

    return matrix, layout


_DEFAULT_MAIN_OUTPUT_LAYER_NAMES = (
    "output_binary",
    "output_main",
    "output_multiclass",
    "output_regression",
)


def resolve_main_output_layer(model: Any, output_layer_name: str | None = None) -> Any:
    """Resolve the prediction layer whose input is the final hidden representation."""

    if output_layer_name is not None:
        if not output_layer_name:
            raise ValueError("output_layer_name must not be empty.")
        return model.get_layer(output_layer_name)

    for candidate in _DEFAULT_MAIN_OUTPUT_LAYER_NAMES:
        try:
            return model.get_layer(candidate)
        except (ValueError, KeyError):
            continue

    outputs = getattr(model, "outputs", None)
    layers = getattr(model, "layers", None)
    if outputs is not None and len(outputs) == 1 and layers:
        return layers[-1]

    raise ValueError("Could not resolve the model's main prediction layer. " "Pass output_layer_name explicitly.")


def build_final_hidden_extractor(
    model: Any,
    *,
    output_layer_name: str | None = None,
) -> Any:
    """Create a Keras model exposing the tensor immediately before prediction.

    For the selected v1 binary model this resolves ``output_binary.input``, i.e.
    the output of ``mlp_dense_3``. This is the final fused latent vector used by
    Experiment II before the original sigmoid prediction head.
    """

    output_layer = resolve_main_output_layer(model, output_layer_name)
    hidden_tensor = getattr(output_layer, "input", None)
    if hidden_tensor is None:
        raise ValueError("Resolved output layer does not expose an input tensor.")
    if isinstance(hidden_tensor, (list, tuple)):
        raise ValueError("Main output layer must consume exactly one hidden tensor.")

    from tensorflow.keras.models import Model

    name = f"{getattr(model, 'name', 'model')}_final_hidden"
    return Model(inputs=model.inputs, outputs=hidden_tensor, name=name)


def extract_final_hidden_representation(
    model: Any,
    arrays: Sequence[np.ndarray],
    *,
    output_layer_name: str | None = None,
    batch_size: int = 256,
    verbose: int = 0,
) -> np.ndarray:
    """Evaluate the final fused latent vector for the supplied model inputs."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    inputs = _model_inputs(arrays)
    row_count = len(inputs[0])
    if any(len(array) != row_count for array in inputs):
        raise ValueError("All model-input arrays must have the same row count.")

    extractor = build_final_hidden_extractor(
        model,
        output_layer_name=output_layer_name,
    )
    values = np.asarray(
        extractor.predict(
            list(inputs),
            batch_size=batch_size,
            verbose=verbose,
        ),
        dtype=np.float32,
    )

    if values.ndim != 2 or values.shape[0] != row_count:
        raise RuntimeError("Final hidden representation must have shape (n, latent_dimension).")
    if values.shape[1] <= 0:
        raise RuntimeError("Final hidden representation has zero width.")
    if not np.isfinite(values).all():
        raise RuntimeError("Final hidden representation contains non-finite values.")

    return values
