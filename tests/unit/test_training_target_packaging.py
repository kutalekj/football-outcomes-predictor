from types import SimpleNamespace

import numpy as np

from football_outcomes.training.train_mlp_rolling import (
    TrainConfig,
    _make_train_targets,
)


def test_main_only_targets_remain_unchanged() -> None:
    y_main = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    configs = (
        TrainConfig(
            model_version="v1",
            use_team_aux_head=False,
        ),
        TrainConfig(
            model_version="v2",
            use_team_aux_head=False,
        ),
    )

    for config in configs:
        result = _make_train_targets(
            matches=[],
            y_main=y_main,
            cfg=config,
        )

        assert result is y_main


def test_auxiliary_targets_are_packaged_by_output_name() -> None:
    matches = [
        SimpleNamespace(
            home_goals=1,
            away_goals=1,
        ),
        SimpleNamespace(
            home_goals=2,
            away_goals=1,
        ),
    ]
    y_main = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    config = TrainConfig(
        model_version="v2",
        use_team_aux_head=True,
        aux_task="binary_u25",
    )

    result = _make_train_targets(
        matches=matches,
        y_main=y_main,
        cfg=config,
    )

    assert set(result) == {
        "output_main",
        "output_team_aux",
    }
    assert result["output_main"] is y_main

    np.testing.assert_array_equal(
        result["output_team_aux"],
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )
