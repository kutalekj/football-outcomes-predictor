from __future__ import annotations

import numpy as np

from football_outcomes.training.fs_training_utils import (
    _strength_to_value_and_mask,
)


def test_strength_to_value_and_mask() -> None:
    raw = np.full((11, 34), 50.0, dtype=np.float32)
    raw[2, 4] = -1.0

    values, mask = _strength_to_value_and_mask(raw)

    assert values.shape == (11, 34)
    assert mask.shape == (11, 34)
    assert values.dtype == np.float32
    assert mask.dtype == np.float32

    assert values[0, 0] == 0.5
    assert mask[0, 0] == 1.0

    assert values[2, 4] == 0.0
    assert mask[2, 4] == 0.0


def test_missing_strength_matrix_becomes_zero_values_and_mask() -> None:
    values, mask = _strength_to_value_and_mask(None)

    np.testing.assert_array_equal(
        values,
        np.zeros((11, 34), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        mask,
        np.zeros((11, 34), dtype=np.float32),
    )
