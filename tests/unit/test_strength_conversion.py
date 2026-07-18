from __future__ import annotations

import numpy as np

from football_outcomes.datasets.arrays import (
    extract_numerical_features,
    strength_to_value_and_mask,
)
from football_outcomes.training.fs_training_utils import (
    _strength_to_value_and_mask as legacy_strength_to_value_and_mask,
)
from football_outcomes.training.fs_training_utils import extract_numerical_features as legacy_extract_numerical_features


def test_strength_to_value_and_mask() -> None:
    raw = np.full((11, 34), 50.0, dtype=np.float32)
    raw[2, 4] = -1.0

    values, mask = strength_to_value_and_mask(raw)

    assert values.shape == (11, 34)
    assert mask.shape == (11, 34)
    assert values.dtype == np.float32
    assert mask.dtype == np.float32

    assert values[0, 0] == 0.5
    assert mask[0, 0] == 1.0

    assert values[2, 4] == 0.0
    assert mask[2, 4] == 0.0


def test_missing_strength_matrix_becomes_zero_values_and_mask() -> None:
    values, mask = strength_to_value_and_mask(None)

    np.testing.assert_array_equal(
        values,
        np.zeros((11, 34), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        mask,
        np.zeros((11, 34), dtype=np.float32),
    )


class MinimalFeatures:
    season = 0.5

    def __getattr__(self, name):
        return None


def test_numerical_features_preserve_order_and_fill_missing_values() -> None:
    values = extract_numerical_features(MinimalFeatures())

    assert values.dtype == np.float32
    assert values.ndim == 1
    assert values.size > 1
    assert values[0] == np.float32(0.5)
    assert np.count_nonzero(values) == 1


def test_legacy_conversion_exports_use_new_implementation() -> None:
    assert legacy_extract_numerical_features is extract_numerical_features
    assert legacy_strength_to_value_and_mask is strength_to_value_and_mask
