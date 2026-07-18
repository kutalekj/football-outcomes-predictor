from types import SimpleNamespace

import numpy as np
import pytest

from football_outcomes.datasets.targets import (
    build_targets_for_matches,
    target_for_match,
)


def make_match(
    home_goals,
    away_goals,
) -> SimpleNamespace:
    return SimpleNamespace(
        home_goals=home_goals,
        away_goals=away_goals,
    )


def test_binary_under_25_targets() -> None:
    matches = [
        make_match(1, 1),
        make_match(2, 1),
        make_match(None, 2),
    ]

    targets = build_targets_for_matches(
        matches,
        mode="binary_u25",
    )

    np.testing.assert_array_equal(
        targets,
        np.asarray(
            [1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
    )
    assert targets.dtype == np.float32


def test_goal_regression_and_distribution_targets() -> None:
    matches = [
        make_match(1, 1),
        make_match(4, 3),
    ]

    regression_targets = build_targets_for_matches(
        matches,
        mode="goals_reg",
    )
    distribution_targets = build_targets_for_matches(
        matches,
        mode="goals_dist",
        max_goals_class=5,
    )

    np.testing.assert_array_equal(
        regression_targets,
        np.asarray(
            [2.0, 7.0],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        distribution_targets,
        np.asarray(
            [2, 5],
            dtype=np.int32,
        ),
    )


def test_unknown_target_mode_is_rejected() -> None:
    match = make_match(1, 0)

    with pytest.raises(
        ValueError,
        match="Unknown mode",
    ):
        target_for_match(
            match,
            mode="unsupported",
        )
